// kt2n native HiFT vocoder + fused streaming tail (lane L6, 10 Oct 2026).
//
// Starts from the M3 spike (spikes/m3-render/hift/hift_native.cu, 1.30 ms at
// B1 x 10 frames) and changes how the work is scheduled, not the model math:
//   * Branch parallelism: the f0/source/STFT chain and the three source
//     resblock chains run on their own stream; the three resblocks of each
//     stage run on three streams and are averaged in the next op's prologue.
//     Critical path ~110 dependent kernels -> ~30 (one CUDA graph, fork/join
//     by events, so it captures).
//   * Split-K reduction inside the conv kernel (last block of a tile sums the
//     partials in fixed split order): no separate reduce kernels, still
//     deterministic and batch-invariant.
//   * More splits at small shapes (~512 blocks per row instead of ~160), so each
//     warp walks <= 4 input channels: the conv is memory-latency bound there.
//   * Programmatic dependent launch: weights are prefetched before
//     griddepcontrol.wait, so a kernel's weight latency overlaps its
//     predecessor's tail.
//   * One fused tail kernel: (y0 + y1 + y2) / 3 -> leaky_relu(0.01) -> conv_post
//     -> exp (clip 1e2) / sin -> inverse DFT -> overlap-add / envelope -> clamp
//     0.99 -> trim-fade or Hamming seam crossfade -> held tail -> splice of the
//     previous first-commit tail -> clip +-1 -> published float samples.
//   * Native draws: torch CUDA randn (Philox4x32-10, curand_normal4, torch's
//     grid-stride layout) per row seed; mt19937 phases on the host.
//
// Exactness notes (same as the spike unless stated): Snake order as hift_fast,
// torch-order cumsum for the voiced branch (bitwise), conv / DFT arithmetic in
// fp32 FMA in a fixed order (production: cuDNN TF32, cuFFT; tolerance class).
//
// Build (box): nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler -fPIC,-ffp-contract=off -shared
//              -I include -o libkt2n_hift.so src/render/hift/hift.cu
// No fast-math.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kt2n/cuda_caps.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kt2n/render/hift.h"

namespace {

enum Pre : int { PRE_NONE = 0, PRE_LEAKY = 1, PRE_ELU = 2, PRE_SNAKE = 3 };
#ifndef KT2N_HIFT_FUSE_SNAKE_BUILD
#define KT2N_HIFT_FUSE_SNAKE_BUILD 0  // 1: build the Snake-in-epilogue experiment (then KT2N_HIFT_FUSE_SNAKE=1)
#endif

constexpr int CO_T = 16;   // output channels per block (split kernel)
constexpr int T_T = 32;    // output samples per block (one per lane)
constexpr int NW = 8;      // warps per block
constexpr int XW_MAX = 96;
constexpr int WK_MAX = 16;
constexpr int CT_WK_MAX = 16;
constexpr int CT_XW_MAX = 32;
constexpr int kSrcCache = 3840;
constexpr int kTrim = 960;

__device__ __forceinline__ void grid_dep_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void grid_dep_launch() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Input of a conv: one tensor, or the average ((a + b) + c) * scale of the three
// resblock outputs (production: xs = b0(x); xs += b1(x); xs += b2(x); x = xs / 3,
// and "divide by a CPU scalar" is a multiply by its float reciprocal).
struct InView {
  const float* a;
  const float* b;
  const float* c;
  float scale;
};
__device__ __forceinline__ float in_load(const InView& v, size_t i) {
  if (v.b == nullptr) return v.a[i];
  const float s = __fadd_rn(__fadd_rn(v.a[i], v.b[i]), v.c[i]);
  return __fmul_rn(s, v.scale);
}
InView single(const float* a) { return InView{a, nullptr, nullptr, 1.f}; }

struct ConvP {
  InView x; int cin; int lin;
  float* y; int cout; int lout;
  const float* w;                        // [cin][K][cout]
  const float* bias;
  int K, dil, pad;
  int pre; float slope; const float* alpha; const float* inv;
  const float* res;                      // v = (acc + b) + res
  const float* add_before;               // v = add_before + v
  // Fused next pre-activation: y_act = act(post, v) with this channel's (post_alpha, post_inv), so the
  // next conv stages its input without recomputing Snake per block (y may be null when only the
  // activated copy is consumed, e.g. a resblock's inner conv output).
  float* y_act; int post; const float* post_alpha; const float* post_inv;
  const uint32_t* tc;                    // tensor-core weight layout (null: SIMT paths only)
  // Polyphase ConvTranspose on the tensor-core kernel (out_stride > 0): grid.z also enumerates the
  // out_stride phases; output u of phase r lands at t = u * out_stride + r + out_shift (+1 and a copy
  // to t = 0 when reflect: ReflectionPad1d((1, 0))), kept when 0 <= t < out_len (pre-reflect length).
  int out_stride, out_shift, out_len, reflect;
  size_t phase_words;                    // tc words per phase
};

__device__ __forceinline__ float act(int pre, float v, float slope, float a, float inv) {
  if (pre == PRE_LEAKY) return v > 0.f ? v : __fmul_rn(v, slope);
  if (pre == PRE_ELU) return v > 0.f ? v : expm1f(v);
  if (pre == PRE_SNAKE) {
    const float s = sinf(__fmul_rn(v, a));
    return __fadd_rn(v, __fmul_rn(inv, __fmul_rn(s, s)));
  }
  return v;
}

__device__ __forceinline__ void conv_store(const ConvP& p, int b, int co, int t, float s) {
  float v = __fadd_rn(s, p.bias[co]);
  const size_t idx = ((size_t)b * p.cout + co) * p.lout + t;
  if (p.res) v = __fadd_rn(v, p.res[idx]);
  if (p.add_before) v = __fadd_rn(p.add_before[idx], v);
#if KT2N_HIFT_FUSE_SNAKE_BUILD
  if (p.y) p.y[idx] = v;
  if (p.y_act) p.y_act[idx] = act(p.post, v, 0.f, p.post_alpha[co], p.post_inv[co]);
#else
  p.y[idx] = v;   // the fused-Snake epilogue is compiled out (it cost registers in every conv: 0-5% at batch)
#endif
}

// Deterministic split-K finish: every split block writes its partial tile; the
// last block to arrive (atomic ticket per tile) sums the partials in split order
// 0..ks-1 and applies the epilogue, then re-arms the ticket for the next replay.
template <typename Store>
__device__ __forceinline__ void split_finish(int ks, int split, int B, int cout, int lout, int b, int co0, int t0,
                                             float* part, unsigned* ctr, const float (*red)[CO_T][T_T + 1],
                                             int* s_last, Store store) {
  const size_t plane = (size_t)B * cout * lout;
  for (int o = threadIdx.x; o < CO_T * T_T; o += blockDim.x) {
    const int c = o / T_T, tt = o % T_T;
    const int co = co0 + c, t = t0 + tt;
    if (co >= cout || t >= lout) continue;
    float s = red[0][c][tt];
#pragma unroll
    for (int w = 1; w < NW; ++w) s = __fadd_rn(s, red[w][c][tt]);
    if (ks == 1) store(b, co, t, s);
    else part[(size_t)split * plane + ((size_t)b * cout + co) * lout + t] = s;
  }
  if (ks == 1) return;
  __threadfence();
  __syncthreads();
  const int tile = (b * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) *s_last = (atomicAdd(&ctr[tile], 1u) == (unsigned)(ks - 1));
  __syncthreads();
  if (!*s_last) return;
  __threadfence();
  for (int o = threadIdx.x; o < CO_T * T_T; o += blockDim.x) {
    const int c = o / T_T, tt = o % T_T;
    const int co = co0 + c, t = t0 + tt;
    if (co >= cout || t >= lout) continue;
    const size_t i = ((size_t)b * cout + co) * lout + t;
    float s = __ldcg(&part[i]);
    for (int k = 1; k < ks; ++k) s = __fadd_rn(s, __ldcg(&part[(size_t)k * plane + i]));
    store(b, co, t, s);
  }
  if (threadIdx.x == 0) ctr[tile] = 0u;
}

// Stride-1 conv1d, small / medium rows. grid (ceil(lout/32), ceil(cout/16), B * ks), 256 threads.
__global__ void __launch_bounds__(256) conv1d_k(ConvP p, int ks, float* part, unsigned* ctr) {
  __shared__ float xs[NW][XW_MAX];
  __shared__ __align__(16) float ws[NW][WK_MAX * CO_T];
  __shared__ float red[NW][CO_T][T_T + 1];
  __shared__ int s_last;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = blockIdx.x * T_T, co0 = blockIdx.y * CO_T;
  const int B = gridDim.z / ks;
  const int b = blockIdx.z / ks, split = blockIdx.z % ks;
  const int win = T_T + (p.K - 1) * p.dil;
  const int nwk = p.K * CO_T;
  float acc[CO_T];
#pragma unroll
  for (int c = 0; c < CO_T; ++c) acc[c] = 0.f;
  const int cps = p.cin / ks;
  const int cpw = cps / NW;
  const int ci_begin = split * cps + warp * cpw;
  const size_t xb = (size_t)b * p.cin * p.lin;
  float xr[3], wr[6];
  auto loadw = [&](int ci) {
    const float* wrow = p.w + (size_t)ci * p.K * p.cout + co0;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      const int i = lane + 32 * k;
      const int j = i / CO_T, c = i % CO_T;
      wr[k] = (i < nwk && co0 + c < p.cout) ? __ldg(&wrow[(size_t)j * p.cout + c]) : 0.f;
    }
  };
  auto loadx = [&](int ci) {
    const size_t row = xb + (size_t)ci * p.lin;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const int i = lane + 32 * k;
      const int src = t0 - p.pad + i;
      xr[k] = (i < win && src >= 0 && src < p.lin) ? in_load(p.x, row + src) : 0.f;
    }
  };
  if (cpw > 0) loadw(ci_begin);
  grid_dep_wait();
  if (cpw > 0) loadx(ci_begin);
  for (int q = 0; q < cpw; ++q) {
    const int ci = ci_begin + q;
    float a = 0.f, iv = 0.f;
    if (p.pre == PRE_SNAKE) { a = p.alpha[ci]; iv = p.inv[ci]; }
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      const int i = lane + 32 * k;
      const int src = t0 - p.pad + i;
      if (i < win) xs[warp][i] = (src >= 0 && src < p.lin) ? act(p.pre, xr[k], p.slope, a, iv) : 0.f;
    }
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      const int i = lane + 32 * k;
      if (i < nwk) ws[warp][i] = wr[k];
    }
    __syncwarp();
    if (q + 1 < cpw) { loadw(ci + 1); loadx(ci + 1); }
    for (int j = 0; j < p.K; ++j) {
      const float xv = xs[warp][lane + j * p.dil];
      const float4* w4 = reinterpret_cast<const float4*>(&ws[warp][j * CO_T]);
#pragma unroll
      for (int c4 = 0; c4 < CO_T / 4; ++c4) {
        const float4 w = w4[c4];
        acc[c4 * 4 + 0] = fmaf(w.x, xv, acc[c4 * 4 + 0]);
        acc[c4 * 4 + 1] = fmaf(w.y, xv, acc[c4 * 4 + 1]);
        acc[c4 * 4 + 2] = fmaf(w.z, xv, acc[c4 * 4 + 2]);
        acc[c4 * 4 + 3] = fmaf(w.w, xv, acc[c4 * 4 + 3]);
      }
    }
    __syncwarp();
  }
#pragma unroll
  for (int c = 0; c < CO_T; ++c) red[warp][c][lane] = acc[c];
  __syncthreads();
  grid_dep_launch();
  split_finish(ks, split, B, p.cout, p.lout, b, co0, t0, part, ctr, red, &s_last,
               [&](int bb, int co, int t, float s) { conv_store(p, bb, co, t, s); });
}

// Large-row conv1d (fp32 SIMT): block tile 64 co x 128 t, 8 input channels per step.
constexpr int BG_CO = 64, BG_T = 128, BG_CI = 8;
constexpr int BG_XW = 128 + 50 + 2;
__global__ void __launch_bounds__(256) conv1d_big_k(ConvP p) {
  __shared__ float xs[BG_CI][BG_XW];
  __shared__ __align__(16) float ws[BG_CI][WK_MAX][BG_CO];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = blockIdx.x * BG_T, co0 = blockIdx.y * BG_CO, b = blockIdx.z;
  const int win = BG_T + (p.K - 1) * p.dil;
  float acc[8][4];
#pragma unroll
  for (int c = 0; c < 8; ++c)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[c][r] = 0.f;
  grid_dep_wait();
  const size_t xb = (size_t)b * p.cin * p.lin;
  for (int c0 = 0; c0 < p.cin; c0 += BG_CI) {
    for (int i = threadIdx.x; i < BG_CI * win; i += blockDim.x) {
      const int cc = i / win, k = i % win;
      const int ci = c0 + cc;
      const int src = t0 - p.pad + k;
      float v = 0.f;
      if (src >= 0 && src < p.lin) {
        float a = 0.f, iv = 0.f;
        if (p.pre == PRE_SNAKE) { a = p.alpha[ci]; iv = p.inv[ci]; }
        v = act(p.pre, in_load(p.x, xb + (size_t)ci * p.lin + src), p.slope, a, iv);
      }
      xs[cc][k] = v;
    }
    const int nw = BG_CI * p.K * BG_CO;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
      const int c = i % BG_CO, jk = i / BG_CO;
      const int j = jk % p.K, cc = jk / p.K;
      ws[cc][j][c] = (co0 + c < p.cout) ? p.w[((size_t)(c0 + cc) * p.K + j) * p.cout + co0 + c] : 0.f;
    }
    __syncthreads();
#pragma unroll 1
    for (int cc = 0; cc < BG_CI; ++cc) {
      for (int j = 0; j < p.K; ++j) {
        float xv[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) xv[r] = xs[cc][lane + 32 * r + j * p.dil];
        const float4 wa = *reinterpret_cast<const float4*>(&ws[cc][j][warp * 8]);
        const float4 wb = *reinterpret_cast<const float4*>(&ws[cc][j][warp * 8 + 4]);
        const float wv[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int c = 0; c < 8; ++c)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[c][r] = fmaf(wv[c], xv[r], acc[c][r]);
      }
    }
    __syncthreads();
  }
  grid_dep_launch();
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const int co = co0 + warp * 8 + c;
    if (co >= p.cout) continue;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int t = t0 + lane + 32 * r;
      if (t < p.lout) conv_store(p, b, co, t, acc[c][r]);
    }
  }
}

// Large-row conv1d on tensor cores (mma.sync m16n8k8 TF32, fp32 accumulate).
constexpr int MM_XW = 184;
__device__ __forceinline__ uint32_t to_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int SPLIT3>
__global__ void __launch_bounds__(256) conv1d_mma_k(ConvP p) {
  extern __shared__ __align__(16) float smem[];
  const int wstride = p.K * BG_CO + 8;
  float* xs = smem;
  float* ws = smem + BG_CI * MM_XW;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, tig = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int t0 = blockIdx.x * BG_T, co0 = blockIdx.y * BG_CO, b = blockIdx.z;
  const int win = BG_T + (p.K - 1) * p.dil;
  float acc[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.f;
  grid_dep_wait();
  const size_t xb = (size_t)b * p.cin * p.lin;
  for (int c0 = 0; c0 < p.cin; c0 += BG_CI) {
    for (int i = threadIdx.x; i < BG_CI * win; i += blockDim.x) {
      const int cc = i / win, k = i % win;
      const int ci = c0 + cc;
      const int src = t0 - p.pad + k;
      float v = 0.f;
      if (src >= 0 && src < p.lin) {
        float a = 0.f, iv = 0.f;
        if (p.pre == PRE_SNAKE) { a = p.alpha[ci]; iv = p.inv[ci]; }
        v = act(p.pre, in_load(p.x, xb + (size_t)ci * p.lin + src), p.slope, a, iv);
      }
      xs[cc * MM_XW + k] = v;
    }
    const int nw = BG_CI * p.K * BG_CO;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) {
      const int c = i % BG_CO, jk = i / BG_CO;
      const int j = jk % p.K, cc = jk / p.K;
      ws[cc * wstride + j * BG_CO + c] =
          (co0 + c < p.cout) ? p.w[((size_t)(c0 + cc) * p.K + j) * p.cout + co0 + c] : 0.f;
    }
    __syncthreads();
    for (int j = 0; j < p.K; ++j) {
      uint32_t ah[2][4], al[2][4], bh[4][2], bl[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int co = wm * 32 + mi * 16 + g;
        const float v[4] = {ws[tig * wstride + j * BG_CO + co], ws[tig * wstride + j * BG_CO + co + 8],
                            ws[(tig + 4) * wstride + j * BG_CO + co], ws[(tig + 4) * wstride + j * BG_CO + co + 8]};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          ah[mi][r] = to_tf32(v[r]);
          if (SPLIT3) al[mi][r] = to_tf32(v[r] - __uint_as_float(ah[mi][r]));
        }
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int t = wn * 32 + ni * 8 + g + j * p.dil;
        const float v0 = xs[tig * MM_XW + t], v1 = xs[(tig + 4) * MM_XW + t];
        bh[ni][0] = to_tf32(v0);
        bh[ni][1] = to_tf32(v1);
        if (SPLIT3) {
          bl[ni][0] = to_tf32(v0 - __uint_as_float(bh[ni][0]));
          bl[ni][1] = to_tf32(v1 - __uint_as_float(bh[ni][1]));
        }
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          if (SPLIT3) {
            mma_tf32(acc[mi][ni], al[mi], bh[ni]);
            mma_tf32(acc[mi][ni], ah[mi], bl[ni]);
          }
          mma_tf32(acc[mi][ni], ah[mi], bh[ni]);
        }
    }
    __syncthreads();
  }
  grid_dep_launch();
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int co = co0 + wm * 32 + mi * 16 + g + ((r >> 1) << 3);
        const int t = t0 + wn * 32 + ni * 8 + 2 * tig + (r & 1);
        if (co < p.cout && t < p.lout) conv_store(p, b, co, t, acc[mi][ni][r]);
      }
}

// Tensor-core conv1d (large rows): implicit GEMM with M = cout (64 per block), N = t (BN per
// block), K = 8 input channels x all taps per pipeline stage; TF32 mma.sync m16n8k8, fp32 accumulate
// (production's precision class: cuDNN TF32 convolutions).
//  * Weights are re-laid out once at create in A-fragment order as tf32 bits
//    ([co64][chunk][tap][m16][lane][4]): one LDS.128 per m16 tile, no per-use conversion.
//  * x is staged once per chunk with the conv's pre-activation applied, converted to tf32 and
//    channel-permuted ([t][c'] with c' = 2 (c % 4) + c / 4) so a lane's two B values are adjacent
//    (one conflict-free LDS.64), and reused by every tap.
//  * Two-stage pipeline: the next chunk's weights stream in by cp.async and its x through
//    registers while the current chunk computes.
//  * Optional split over channel chunks, finished by the deterministic last-block reduction.
constexpr int TC_BM = 64;
// x is prefetched one chunk ahead (two chunks ahead was measured slower: more registers, fewer blocks per SM). Taps per pipeline step (TG < K): the weights of a chunk are staged in tap groups of
// 4, so a K = 11 or K = 7 conv block needs ~28 KB of shared memory and 70 registers and THREE blocks fit per SM
// (all 11 taps double-buffered = 45 KB of weights, 56.5 KB per block: one block per SM; K = 7: two). Same taps in
// the same order into the same accumulators, so bitwise equal to TG = K;
// B10 x 108 1.07x -> 1.19x production.
#ifndef KT2N_TC_TG11
#define KT2N_TC_TG11 4
#endif
#ifndef KT2N_TC_TG7
#define KT2N_TC_TG7 4
#endif
#ifndef KT2N_TC_TARGET_MUL
#define KT2N_TC_TARGET_MUL 2  // channel splits until a conv launches >= this x SMs blocks
#endif
#ifndef KT2N_TC_BN64_MUL
#define KT2N_TC_BN64_MUL KT2N_TC_TARGET_MUL  // 64-wide tiles when 128-wide ones give < this x SMs blocks
#endif
#ifndef KT2N_TC_WIDE_MINK
#define KT2N_TC_WIDE_MINK 0  // >0: 256-wide tiles (warp tile 32 x 64) for convs with K >= this when they fill the GPU
#endif
#ifndef KT2N_TC_SWIZZLE
#define KT2N_TC_SWIZZLE 0   // pair-slot swizzle of the staged x; measured with the fused Snake as a pair: slower
#endif
// One barrier per pipeline step (v23): the next weight copy is issued AFTER the step's barrier, into the stage
// that every warp finished reading in the previous step, so the second (end-of-step) barrier of the v10-v21 loop
// is not needed. KT2N_TC_WSN = weight stages in the ring (2 or 3; 3 keeps the copy two steps ahead). The math,
// and so the output, is unchanged (bitwise; `same` mode).
#ifndef KT2N_TC_ONEBAR
#define KT2N_TC_ONEBAR 0
#endif
#ifndef KT2N_TC_WSN
#define KT2N_TC_WSN 2
#endif
#if !KT2N_TC_ONEBAR && KT2N_TC_WSN != 2
#error "KT2N_TC_WSN != 2 needs KT2N_TC_ONEBAR=1"
#endif
// 8-channel chunks per pipeline step for the K <= 3 convs (and the K = 2 / 3 polyphase ConvTranspose): with
// XC = 2 a step stages 16 input channels and runs chunk q's taps then chunk q + 1's, the XC = 1 order (bitwise).
#ifndef KT2N_TC_XCS
#define KT2N_TC_XCS 1
#endif
constexpr int TC_MAX_DIL = 5;   // largest dilation in HiFT (resblock dilations 1 / 3 / 5); sizes the x halo
__host__ __device__ constexpr int tc_xc(int K) { return K <= 3 ? KT2N_TC_XCS : 1; }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
[[maybe_unused]] __device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

#ifndef KT2N_TC_BPRE
#define KT2N_TC_BPRE 0     // v27 experiment: load a tap's B fragments before its MMAs (bitwise equal)
#endif
#ifndef KT2N_TC_MMA_NV
#define KT2N_TC_MMA_NV 0   // v27 experiment: non-volatile mma asm (the compiler may schedule it; bitwise equal)
#endif
#if KT2N_TC_MMA_NV
#define KT2N_MMA_ASM asm
#else
#define KT2N_MMA_ASM asm volatile
#endif
__device__ __forceinline__ void mma_tf32u(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
  KT2N_MMA_ASM(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// KT2N_TC_MINB: blocks per SM the compiler must fit by registers (__launch_bounds__ min blocks; 3 -> <= 80
// registers). v23's one-barrier loop let ptxas grow the 128-wide K = 11 kernel from 70 to 120 registers.
#ifndef KT2N_TC_MINB
#define KT2N_TC_MINB 1
#endif
// (MINB = 1 keeps plain __launch_bounds__(256): an explicit minimum of 1 block let ptxas grow these kernels from
// 70-72 to 110-120 registers, cuobjdump 16:06.)
#if KT2N_TC_MINB > 1
// (BN = 64 kernels: at least 4 blocks per SM, i.e. <= 64 registers; v21's BN = 64 kernels use 48 and fit 4-5 blocks
// by shared memory, which a plain minimum of 3 would let ptxas give away: 70-78 registers, v27d 18:04.)
#define KT2N_TC_LB __launch_bounds__(256, (BN == 128 ? KT2N_TC_MINB : BN == 64 ? 4 : 1))
#else
#define KT2N_TC_LB __launch_bounds__(256)
#endif
// v26: the same body as a device function with an explicit z index (bz) and z extent (nz), so that one launch can
// carry several convs (conv1d_tc3_k: the three resblock branches' layer-k convs).
template <int BN, int K, int TG = K>
__device__ __forceinline__ void conv_tc_body(const ConvP& p, const uint32_t* __restrict__ wtc, int ks, float* part,
                                             unsigned* ctr, const int bz, const int nz) {
#define KT2N_TC_BZ bz
#define KT2N_TC_NZ nz
#include "conv_tc_body.inc"
#undef KT2N_TC_BZ
#undef KT2N_TC_NZ
}

template <int BN, int K, int TG = K>
__global__ void KT2N_TC_LB conv1d_tc_k(ConvP p, const uint32_t* __restrict__ wtc, int ks, float* part,
                                                   unsigned* ctr) {
#define KT2N_TC_BZ blockIdx.z
#define KT2N_TC_NZ gridDim.z
#include "conv_tc_body.inc"
#undef KT2N_TC_BZ
#undef KT2N_TC_NZ
}

// v26: the three resblock branches' layer-k convs in ONE launch (same cin / cout / lin / lout / dilation; only the
// taps K = 3 / 7 / 11, the weights, Snake's alpha and the buffers differ). Replaces 3 launches on 3 streams (fork /
// join events) with 1 launch on the caller's stream: ~2x fewer kernels per call, no side streams for the resblocks.
// grid.z = B x (ks11 + ks7 + ks3): z in [0, B ks11) = the K = 11 branch (blocks dispatch in index order, so the
// longest blocks start first), then K = 7, then K = 3. Each branch runs conv_tc_body with its own params, split
// count, partial-sum buffer and tickets, so with v21's tile / split choice (mode 1) the output is bitwise v21's.
struct ConvP3 {
  ConvP p[3];          // branch j (K = 3, 7, 11 for j = 0, 1, 2)
  float* part[3];
  unsigned* ctr[3];
  int ks[3];           // channel splits per branch (mode 2 balances block work: ks ~ K)
};
template <int BN>
#ifndef KT2N_TC3_GRIDCONST
#define KT2N_TC3_GRIDCONST 0   // 1: __grid_constant__ parameter (an nsys --cuda-graph-trace=node run faulted with it)
#endif
#if KT2N_TC3_GRIDCONST
#define KT2N_TC3_PARAM const __grid_constant__ ConvP3 g
#else
#define KT2N_TC3_PARAM const ConvP3 g
#endif
__global__ void KT2N_TC_LB conv1d_tc3_k(KT2N_TC3_PARAM, int B) {
  const int n11 = B * g.ks[2], n7 = B * g.ks[1];
  const int zz = blockIdx.z;
  if (zz < n11) {
    conv_tc_body<BN, 11, KT2N_TC_TG11>(g.p[2], g.p[2].tc, g.ks[2], g.part[2], g.ctr[2], zz, n11);
  } else if (zz < n11 + n7) {
    conv_tc_body<BN, 7, KT2N_TC_TG7>(g.p[1], g.p[1].tc, g.ks[1], g.part[1], g.ctr[1], zz - n11, n7);
  } else {
    conv_tc_body<BN, 3, 3>(g.p[0], g.p[0].tc, g.ks[0], g.part[0], g.ctr[0], zz - n11 - n7, B * g.ks[0]);
  }
}

// [cin][K][cout] fp32 -> [co64][chunk][tap][m16][lane][4] tf32 (A-fragment order of m16n8k8:
// a0 = A[g][tig], a1 = A[g + 8][tig], a2 = A[g][tig + 4], a3 = A[g + 8][tig + 4]).
__global__ void tc_arrange_k(const float* w, int cin, int K, int cout, uint32_t* out) {
  const size_t total = (size_t)cin * K * cout;
  for (size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x; o < total; o += (size_t)gridDim.x * blockDim.x) {
    size_t rem = o;
    const int r = rem % 4; rem /= 4;
    const int lane = rem % 32; rem /= 32;
    const int mt = rem % 4; rem /= 4;
    const int j = rem % K; rem /= K;
    const int q = rem % (cin / 8); rem /= (cin / 8);
    const int cb = (int)rem;
    const int g = lane >> 2, tig = lane & 3;
    const int co = cb * 64 + mt * 16 + g + ((r & 1) << 3);
    const int ci = q * 8 + tig + ((r >> 1) << 2);
    out[o] = to_tf32(w[((size_t)ci * K + j) * cout + co]);
  }
}

// Polyphase weights of a ConvTranspose1d (torch [cin][cout][K], stride s) for phase r as a stride-1
// conv in the [cin][Kmax][cout] layout tc_arrange_k reads: tap j' = Kmax - 1 - m holds W[ci][co][r + m s]
// (zero where r + m s >= K), so output u of phase r = sum_m x[u - m] W[r + m s] reads x[u - (Kmax - 1) + j'].
__global__ void convT_phase_w_k(const float* w, int cin, int cout, int K, int s, int r, int kmax, float* out) {
  const size_t total = (size_t)cin * kmax * cout;
  for (size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x; o < total; o += (size_t)gridDim.x * blockDim.x) {
    const int co = o % cout;
    const int jp = (o / cout) % kmax;
    const int ci = o / ((size_t)cout * kmax);
    const int j = r + (kmax - 1 - jp) * s;
    out[o] = j < K ? w[((size_t)ci * cout + co) * K + j] : 0.f;
  }
}

struct ConvTP {
  InView x; int cin; int lin;
  float* y; int cout; int lout;   // logical output length (after optional reflect pad)
  const float* w;                 // torch layout [cin][cout][K]
  const float* bias;
  int K, stride, pad;
  float slope;                    // leaky_relu on the input
  int reflect_left;               // ReflectionPad1d((1, 0)) after the transpose conv
};

// ConvTranspose1d (gather form), leaky_relu prologue, optional left reflect pad.
__global__ void __launch_bounds__(256) convT_k(ConvTP p, int ks, float* part, unsigned* ctr) {
  __shared__ float xs[NW][CT_XW_MAX];
  __shared__ float ws[NW][CT_WK_MAX * CO_T];
  __shared__ float red[NW][CO_T][T_T + 1];
  __shared__ int s_last;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = blockIdx.x * T_T, co0 = blockIdx.y * CO_T;
  const int B = gridDim.z / ks;
  const int b = blockIdx.z / ks, split = blockIdx.z % ks;
  auto src_of = [&](int tl) { return p.reflect_left ? (tl == 0 ? 1 : tl - 1) : tl; };
  const int my_t = src_of(t0 + lane);
  const int tmin = p.reflect_left ? max(0, t0 - 1) : t0;
  const int tmax = p.reflect_left ? max(1, t0 + T_T - 2) : t0 + T_T - 1;
  const int num_lo = tmin + p.pad - (p.K - 1);
  const int i_min = num_lo >= 0 ? num_lo / p.stride : -((-num_lo + p.stride - 1) / p.stride);
  const int i_max = (tmax + p.pad) / p.stride;
  const int nx = i_max - i_min + 1;
  const int nwk = p.K * CO_T;
  float acc[CO_T];
#pragma unroll
  for (int c = 0; c < CO_T; ++c) acc[c] = 0.f;
  const int cps = p.cin / ks;
  const int cpw = cps / NW;
  const int ci_begin = split * cps + warp * cpw;
  float xr = 0.f, wr[8];
  auto loadw = [&](int ci) {
    const float* wrow = p.w + ((size_t)ci * p.cout + co0) * p.K;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = lane + 32 * k;
      const int c = i / p.K;
      wr[k] = (i < nwk && co0 + c < p.cout) ? __ldg(&wrow[i]) : 0.f;
    }
  };
  auto loadx = [&](int ci) {
    const size_t row = ((size_t)b * p.cin + ci) * p.lin;
    const int src = i_min + lane;
    xr = (lane < nx && src >= 0 && src < p.lin) ? in_load(p.x, row + src) : 0.f;
  };
  if (cpw > 0) loadw(ci_begin);
  grid_dep_wait();
  if (cpw > 0) loadx(ci_begin);
  const int num = my_t + p.pad;
  for (int q = 0; q < cpw; ++q) {
    if (lane < nx) xs[warp][lane] = xr > 0.f ? xr : __fmul_rn(xr, p.slope);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = lane + 32 * k;
      if (i < nwk) ws[warp][i] = wr[k];
    }
    __syncwarp();
    if (q + 1 < cpw) { loadw(ci_begin + q + 1); loadx(ci_begin + q + 1); }
    for (int j = num % p.stride; j < p.K; j += p.stride) {
      const int n2 = num - j;
      if (n2 < 0) break;
      const int i = n2 / p.stride;
      if (i >= p.lin) continue;
      const float xv = xs[warp][i - i_min];
#pragma unroll
      for (int c = 0; c < CO_T; ++c) acc[c] = fmaf(ws[warp][c * p.K + j], xv, acc[c]);
    }
    __syncwarp();
  }
#pragma unroll
  for (int c = 0; c < CO_T; ++c) red[warp][c][lane] = acc[c];
  __syncthreads();
  grid_dep_launch();
  split_finish(ks, split, B, p.cout, p.lout, b, co0, t0, part, ctr, red, &s_last,
               [&](int bb, int co, int t, float s) {
                 p.y[((size_t)bb * p.cout + co) * p.lout + t] = __fadd_rn(s, p.bias[co]);
               });
}

// Large-row ConvTranspose1d (gather form), register-blocked: block tile 64 co x 128 t, 8 input
// channels staged per step (leaky_relu applied once at staging), each thread 8 co x 4 t. Output t takes
// taps j = (t + pad) mod stride + m * stride from input i = (t + pad - j) / stride: lanes of a warp sit
// on different phases, so their tap loops differ only in start (2-3 taps each). Optional left reflect pad.
constexpr int CTB_CO = 64, CTB_T = 128, CTB_CI = 8, CTB_XW = 128 / 3 + 8;
__global__ void __launch_bounds__(256) convT_big_k(ConvTP p) {
  __shared__ float xs[CTB_CI][CTB_XW];
  __shared__ float ws[CTB_CI][CT_WK_MAX][CTB_CO];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t0 = blockIdx.x * CTB_T, co0 = blockIdx.y * CTB_CO, b = blockIdx.z;
  // raw (pre-reflect) output range of this block
  const int r_lo = p.reflect_left ? max(0, t0 - 1) : t0;
  const int r_hi = p.reflect_left ? t0 + CTB_T - 2 : t0 + CTB_T - 1;
  const int num_lo = r_lo + p.pad - (p.K - 1);
  const int i_min = num_lo >= 0 ? num_lo / p.stride : -((-num_lo + p.stride - 1) / p.stride);
  const int i_max = (max(r_hi, 1) + p.pad) / p.stride;
  const int nx = i_max - i_min + 1;   // <= 128 / stride + K / stride + 2 <= CTB_XW
  float acc[8][4];
#pragma unroll
  for (int c = 0; c < 8; ++c)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[c][r] = 0.f;
  int num[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int tl = t0 + lane + 32 * r;
    const int raw = p.reflect_left ? (tl == 0 ? 1 : tl - 1) : tl;
    num[r] = raw + p.pad;
  }
  grid_dep_wait();
  for (int c0 = 0; c0 < p.cin; c0 += CTB_CI) {
    for (int i = threadIdx.x; i < CTB_CI * nx; i += blockDim.x) {
      const int cc = i / nx, k = i % nx;
      const int src = i_min + k;
      float v = 0.f;
      if (src >= 0 && src < p.lin) {
        v = in_load(p.x, ((size_t)b * p.cin + c0 + cc) * p.lin + src);
        v = v > 0.f ? v : __fmul_rn(v, p.slope);
      }
      xs[cc][k] = v;
    }
    for (int i = threadIdx.x; i < CTB_CI * CTB_CO * p.K; i += blockDim.x) {
      const int j = i % p.K, c = (i / p.K) % CTB_CO, cc = i / (p.K * CTB_CO);
      ws[cc][j][c] = (co0 + c < p.cout) ? p.w[((size_t)(c0 + cc) * p.cout + co0 + c) * p.K + j] : 0.f;
    }
    __syncthreads();
#pragma unroll 1
    for (int cc = 0; cc < CTB_CI; ++cc) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        for (int j = num[r] % p.stride; j < p.K; j += p.stride) {
          const int n2 = num[r] - j;
          if (n2 < 0) break;
          const int i = n2 / p.stride;
          if (i >= p.lin) continue;
          const float xv = xs[cc][i - i_min];
          const float4 wa = *reinterpret_cast<const float4*>(&ws[cc][j][warp * 8]);
          const float4 wb = *reinterpret_cast<const float4*>(&ws[cc][j][warp * 8 + 4]);
          acc[0][r] = fmaf(wa.x, xv, acc[0][r]); acc[1][r] = fmaf(wa.y, xv, acc[1][r]);
          acc[2][r] = fmaf(wa.z, xv, acc[2][r]); acc[3][r] = fmaf(wa.w, xv, acc[3][r]);
          acc[4][r] = fmaf(wb.x, xv, acc[4][r]); acc[5][r] = fmaf(wb.y, xv, acc[5][r]);
          acc[6][r] = fmaf(wb.z, xv, acc[6][r]); acc[7][r] = fmaf(wb.w, xv, acc[7][r]);
        }
      }
    }
    __syncthreads();
  }
  grid_dep_launch();
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const int co = co0 + warp * 8 + c;
    if (co >= p.cout) continue;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int t = t0 + lane + 32 * r;
      if (t < p.lout) p.y[((size_t)b * p.cout + co) * p.lout + t] = __fadd_rn(acc[c][r], p.bias[co]);
    }
  }
}

// Strided conv1d, one output per thread (small calls: more threads, shorter serial chains).
__global__ void conv_strided1_k(const float* x, int cin, int lin, float* y, int cout, int lout, const float* w,
                                const float* bias, int K, int stride, int pad, int B) {
  grid_dep_wait();
  const size_t total = (size_t)B * cout * lout;
  for (size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x; o < total; o += (size_t)gridDim.x * blockDim.x) {
    const int t = o % lout;
    const int co = (o / lout) % cout;
    const int b = o / ((size_t)lout * cout);
    float acc = 0.f;
    for (int ci = 0; ci < cin; ++ci) {
      const float* xr = x + ((size_t)b * cin + ci) * lin;
      const float* wr = w + ((size_t)co * cin + ci) * K;
      for (int j = 0; j < K; ++j) {
        const int src = t * stride + j - pad;
        if (src >= 0 && src < lin) acc = fmaf(wr[j], xr[src], acc);
      }
    }
    y[o] = __fadd_rn(acc, bias[co]);
  }
  grid_dep_launch();
}

// Strided conv1d (source downsamplers, 18 input channels), torch weight layout [cout][18][K].
// Block = 256 outputs x 8 output channels; the 8 channels' weights sit in shared memory, so each
// x load feeds 8 FMAs (the spike's one-output-per-thread form did 2 loads per FMA: 2 TFLOPS).
constexpr int SD_CO = 8;
__global__ void __launch_bounds__(256) conv_strided_k(const float* x, int cin, int lin, float* y, int cout, int lout,
                                                      const float* w, const float* bias, int K, int stride, int pad) {
  extern __shared__ float wsd[];  // [SD_CO][cin][K]
  const int co0 = blockIdx.y * SD_CO, b = blockIdx.z;
  const int nw = SD_CO * cin * K;
  for (int i = threadIdx.x; i < nw; i += blockDim.x) wsd[i] = w[(size_t)co0 * cin * K + i];
  grid_dep_wait();
  __syncthreads();
  const int t = blockIdx.x * 256 + threadIdx.x;
  if (t < lout) {
    float acc[SD_CO];
#pragma unroll
    for (int c = 0; c < SD_CO; ++c) acc[c] = 0.f;
    const float* xb = x + (size_t)b * cin * lin;
    for (int ci = 0; ci < cin; ++ci) {
      const float* xr = xb + (size_t)ci * lin;
      for (int j = 0; j < K; ++j) {
        const int src = t * stride + j - pad;
        if (src < 0 || src >= lin) continue;
        const float xv = __ldg(&xr[src]);
#pragma unroll
        for (int c = 0; c < SD_CO; ++c) acc[c] = fmaf(wsd[(c * cin + ci) * K + j], xv, acc[c]);
      }
    }
#pragma unroll
    for (int c = 0; c < SD_CO; ++c) y[((size_t)b * cout + co0 + c) * lout + t] = __fadd_rn(acc[c], bias[co0 + c]);
  }
  grid_dep_launch();
}

// Strided conv1d of the source downsamplers, staged through shared memory (v24). conv_strided_k reads x straight
// from global memory at a stride of S floats across the warp (S = 15 / 3), so every warp load touches up to 15
// 128-byte segments: ~7% of a B32 x 58 call for ~0.6% of its FLOPs (nsys_launches_v17b.json). Here a block
// (256 outputs x 8 output channels) copies one input channel's window (255 S + K samples, coalesced) into shared
// memory at a time, the next channel prefetched into registers while the current one computes, and reads the 8
// channels' weights as two broadcast float4 ([ci][j][c] layout). Per output the loop order (ci, then j), the
// skipped out-of-range taps and the fmaf chain are those of conv_strided_k, so the output is bitwise equal.
#ifndef KT2N_SD_SM
#define KT2N_SD_SM 0
#endif
template <int S, int K>
__global__ void __launch_bounds__(256) conv_strided_sm_k(const float* x, int cin, int lin, float* y, int cout, int lout,
                                                         const float* w, const float* bias, int pad) {
  constexpr int XW = 255 * S + K;          // input samples per channel one block reads
  constexpr int XR = (XW + 255) / 256;     // per thread
  extern __shared__ __align__(16) float ssm[];
  float* wsd = ssm;                                     // [cin][K][SD_CO]
  float* xs = ssm + ((cin * K * SD_CO + 3) & ~3);       // [2][XW]
  const int co0 = blockIdx.y * SD_CO, b = blockIdx.z;
  const int t0 = blockIdx.x * 256;
  const int nw = SD_CO * cin * K;
  for (int i = threadIdx.x; i < nw; i += 256) {         // torch [cout][cin][K] -> [ci][j][c]
    const int c = i / (cin * K), rem = i - c * (cin * K);
    wsd[rem * SD_CO + c] = w[(size_t)co0 * cin * K + i];
  }
  grid_dep_wait();
  const float* xb = x + (size_t)b * cin * lin;
  const int base = t0 * S - pad;
  float xr[XR];
  auto load = [&](int ci) {
#pragma unroll
    for (int k = 0; k < XR; ++k) {
      const int e = threadIdx.x + 256 * k, src = base + e;
      xr[k] = (e < XW && src >= 0 && src < lin) ? __ldg(&xb[(size_t)ci * lin + src]) : 0.f;
    }
  };
  auto store = [&](int buf) {
#pragma unroll
    for (int k = 0; k < XR; ++k) {
      const int e = threadIdx.x + 256 * k;
      if (e < XW) xs[buf * XW + e] = xr[k];
    }
  };
  load(0);
  store(0);
  __syncthreads();
  const int t = t0 + threadIdx.x;
  float acc[SD_CO];
#pragma unroll
  for (int c = 0; c < SD_CO; ++c) acc[c] = 0.f;
  for (int ci = 0; ci < cin; ++ci) {
    if (ci + 1 < cin) load(ci + 1);
    const float* xc = xs + (ci & 1) * XW + threadIdx.x * S;
    const float* wc = wsd + ci * K * SD_CO;
#pragma unroll
    for (int j = 0; j < K; ++j) {
      const int src = t * S + j - pad;
      if (src < 0 || src >= lin) continue;
      const float xv = xc[j];
      const float4 w0 = *reinterpret_cast<const float4*>(&wc[j * SD_CO]);
      const float4 w1 = *reinterpret_cast<const float4*>(&wc[j * SD_CO + 4]);
      acc[0] = fmaf(w0.x, xv, acc[0]); acc[1] = fmaf(w0.y, xv, acc[1]);
      acc[2] = fmaf(w0.z, xv, acc[2]); acc[3] = fmaf(w0.w, xv, acc[3]);
      acc[4] = fmaf(w1.x, xv, acc[4]); acc[5] = fmaf(w1.y, xv, acc[5]);
      acc[6] = fmaf(w1.z, xv, acc[6]); acc[7] = fmaf(w1.w, xv, acc[7]);
    }
    // the other buffer was last read in iteration ci - 1, which every thread finished before the last barrier
    if (ci + 1 < cin) store((ci + 1) & 1);
    __syncthreads();
  }
  if (t < lout) {
#pragma unroll
    for (int c = 0; c < SD_CO; ++c) y[((size_t)b * cout + co0 + c) * lout + t] = __fadd_rn(acc[c], bias[co0 + c]);
  }
  grid_dep_launch();
}
template <int S, int K>
constexpr size_t sd_sm_bytes(int cin) {
  return (size_t)(((cin * K * SD_CO + 3) & ~3) + 2 * (255 * S + K)) * sizeof(float);
}

// f0 = |Linear(512 -> 1)(ELU(h))| per (b, t). One warp per output.
__global__ void f0_head_k(const float* h, int C, int T, const float* w, const float* bias, float* f0, int B) {
  grid_dep_wait();
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_global < B * T) {
    const int b = warp_global / T, t = warp_global % T;
    float acc = 0.f;
    for (int c = lane; c < C; c += 32) {
      float v = h[((size_t)b * C + c) * T + t];
      v = v > 0.f ? v : expm1f(v);
      acc = fmaf(w[c], v, acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, off));
    if (lane == 0) f0[b * T + t] = fabsf(__fadd_rn(acc, bias[0]));
  }
  grid_dep_launch();
}

// torch 2.7's innermost-dim scan geometry (ScanUtils.cuh).
int torch_scan_log_nx(long rows, long row_size) {
  int lx = 0, ly = 0;
  while ((1L << lx) < row_size) ++lx;
  while ((1L << ly) < rows) ++ly;
  int v = (9 + (lx - ly)) / 2;
  return v < 4 ? 4 : (v > 9 ? 9 : v);
}

// Voiced branch: harmonic sines 0.1 sin(2 pi (cumsum(f0_up (h+1) / sr) % 1) + phase)
// of one (b, h) row with torch's scan order (bitwise). Rows with no voiced frame
// skip it: their sine term is multiplied by uv = 0 anyway (exact).
__global__ void sine_rows_k(const float* f0, int T, const float* phase, int phase_stride, float* sines, int log_nx,
                            int harmonics) {
  extern __shared__ float row_buf[];
  grid_dep_wait();
  const int row = blockIdx.x;
  const int b = row / harmonics, h = row % harmonics;
  __shared__ int s_voiced;
  if (threadIdx.x == 0) {
    int v = 0;
    for (int t = 0; t < T; ++t) v |= f0[b * T + t] > 10.f;
    s_voiced = v;
  }
  __syncthreads();
  if (!s_voiced) return;
  const int nx = 1 << log_nx;
  const int L = T * 480;
  const float up_scale = (float)(1.0 / 480.0);
  const float inv_sr = 1.0f / 24000.0f;
  const float two_pi = (float)(2.0 * M_PI);
  const float ph = phase[b * phase_stride + h];
  float* out = sines + (size_t)row * L;
  const int tid = threadIdx.x;
  float block_total = 0.f;
  for (int block_col = 0; block_col < L; block_col += 2 * nx) {
    const int col1 = block_col + tid, col2 = block_col + nx + tid;
    float va = 0.f, vb = 0.f;
    if (col1 < L) {
      const float f0a = f0[b * T + min((int)floorf(__fmul_rn((float)col1, up_scale)), T - 1)];
      va = __fmul_rn(__fmul_rn(f0a, (float)(h + 1)), inv_sr);
    }
    if (col2 < L) {
      const float f0b = f0[b * T + min((int)floorf(__fmul_rn((float)col2, up_scale)), T - 1)];
      vb = __fmul_rn(__fmul_rn(f0b, (float)(h + 1)), inv_sr);
    }
    row_buf[tid] = va;
    row_buf[nx + tid] = vb;
    if (tid == 0) row_buf[0] = __fadd_rn(row_buf[0], block_total);
    __syncthreads();
    for (int m = 0; m <= log_nx; ++m) {
      const int s = 1 << m;
      const int a = ((tid >> m) << (m + 1)) | s;
      const int ti = a + (tid % s);
      const int si = a - 1;
      row_buf[ti] = __fadd_rn(row_buf[ti], row_buf[si]);
      __syncthreads();
    }
    const float ca = row_buf[tid], cb = row_buf[nx + tid];
    block_total = row_buf[2 * nx - 1];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int n = k ? col2 : col1;
      if (n >= L) continue;
      const float c = k ? cb : ca;
      const float theta = __fmul_rn(two_pi, fmodf(c, 1.0f));
      out[n] = __fmul_rn(0.1f, sinf(__fadd_rn(theta, ph)));
    }
  }
}

// Per row source arguments (core mode: shared external cache; row mode: Kt2nHiftRow).
struct SrcArgs {
  const Kt2nHiftRow* rows;  // null in core mode
  const float* cache;       // core mode [B][cache_len]
  int cache_len;
};

__constant__ float c_cos[9][16];
__constant__ float c_sin[9][16];
__constant__ float c_trim[kTrim];

// Source merge + STFT in one kernel: block = 64 STFT frames of one row. It computes the
// source s on its 256 samples plus a halo (16 left, 8 right; reflect padding stays inside),
// writes s (each sample by exactly one block) and the new source cache, then the 64 frames.
__device__ __forceinline__ float source_sample(const float* f0, int T, int L, const float* sines, const float* noise,
                                               const float* lin_w, const float* lin_b, const float* cache,
                                               int cache_len, int b, int n) {
  if (n < cache_len) return cache[n];
  const float f0v = f0[b * T + min((int)floorf(__fmul_rn((float)n, (float)(1.0 / 480.0))), T - 1)];
  const float uv = f0v > 10.f ? 1.f : 0.f;
  const float amp = __fadd_rn(__fmul_rn(uv, 0.003f), __fmul_rn(__fmul_rn(__fsub_rn(1.f, uv), 0.1f), 1.0f / 3.0f));
  const float* nz = noise + (size_t)b * 9 * L + n;
  const float* sw = sines + (size_t)b * 9 * L + n;
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 9; ++k) {
    const float sk = uv != 0.f ? sw[(size_t)k * L] : 0.f;
    const float e = __fadd_rn(__fmul_rn(sk, uv), __fmul_rn(amp, nz[(size_t)k * L]));
    acc = fmaf(lin_w[k], e, acc);
  }
  return tanhf(__fadd_rn(acc, lin_b[0]));
}

__global__ void __launch_bounds__(256) source_stft_k(const float* f0, int T, const float* sines, const float* noise,
                                                     const float* lin_w, const float* lin_b, SrcArgs sa, float* s,
                                                     const float* win, float* sst, int F) {
  __shared__ float sv[256 + 24];   // 16-sample left halo: a last frame's reflection reaches L - 9
  grid_dep_wait();
  const int b = blockIdx.y;
  const int L = T * 480;
  const int fb = blockIdx.x * 64;
  const int base = 4 * fb - 16;
  const float* cache = sa.rows ? sa.rows[b].src_in : (sa.cache ? sa.cache + (size_t)b * sa.cache_len : nullptr);
  const int cache_len = sa.rows ? sa.rows[b].src_len : sa.cache_len;
  const bool hold = sa.rows && (sa.rows[b].flags & KT2N_HIFT_HOLD) && sa.rows[b].src_out;
  for (int i = threadIdx.x; i < 256 + 24; i += 256) {
    const int n = base + i;
    float v = 0.f;
    if (n >= 0 && n < L) {
      v = source_sample(f0, T, L, sines, noise, lin_w, lin_b, cache, cache_len, b, n);
      if (i >= 16 && i < 16 + 256) {
        s[(size_t)b * L + n] = v;
        const int keep = min(3840, L);
        if (hold && n >= L - keep) sa.rows[b].src_out[n - (L - keep)] = v;
      }
    }
    sv[i] = v;
  }
  __syncthreads();
  grid_dep_launch();
  const int fl = threadIdx.x;
  const int f = fb + fl;
  if (fl >= 64 || f >= F) return;
  float v[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    int idx = 4 * f + k - 8;
    if (idx < 0) idx = -idx;
    if (idx >= L) idx = 2 * (L - 1) - idx;
    v[k] = __fmul_rn(sv[idx - base], win[k]);
  }
#pragma unroll
  for (int m = 0; m < 9; ++m) {
    float re = 0.f, im = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      re = fmaf(v[k], c_cos[m][k], re);
      im = fmaf(-v[k], c_sin[m][k], im);
    }
    sst[((size_t)b * 18 + m) * F + f] = re;
    sst[((size_t)b * 18 + 9 + m) * F + f] = im;
  }
}

// Torch randn((1, 9, L)) on a fresh CUDA generator seeded with rows' seeds: torch's
// distribution_elementwise_grid_stride_kernel with curand_normal4 (unroll 4), grid
// min(SMs * maxThreadsPerSM / 256, ceil(numel / 256)), offset 0. grid.y = row.
// Block 0 of each row also copies the row's new mel cache.
struct NoiseArgs {
  const Kt2nHiftRow* rows;   // seeds from rows (row mode)
  const uint64_t* seeds;     // or a device seed array (check mode)
  const float* mel; int T;   // mel cache copy (row mode)
};
__global__ void __launch_bounds__(256) noise_k(NoiseArgs na, int numel, float* out) {
  grid_dep_wait();
  const int b = blockIdx.y;
  const uint64_t seed = na.rows ? na.rows[b].seed : na.seeds[b];
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);
  const int64_t rounded = ((numel - 1) / (stride * 4) + 1) * stride * 4;
  float* o = out + (size_t)b * numel;
  for (int64_t li0 = idx; li0 < rounded; li0 += stride * 4) {
    const float4 r = curand_normal4(&state);
    const float rv[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      const int64_t li = li0 + stride * ii;
      if (li < numel) o[li] = rv[ii];
    }
  }
  if (na.rows && blockIdx.x == 0 && (na.rows[b].flags & KT2N_HIFT_HOLD) && na.rows[b].mel_cache_out) {
    // [80][8] slot layout (StateView::hift_mel_cache); the first `keep` columns are valid
    const int keep = min(8, na.T);
    for (int i = threadIdx.x; i < 80 * keep; i += blockDim.x) {
      const int c = i / keep, t = i % keep;
      na.rows[b].mel_cache_out[c * 8 + t] = na.mel[((size_t)b * 80 + c) * na.T + (na.T - keep + t)];
    }
  }
  grid_dep_launch();
}

// HiFT input assembly for the render pipeline: out[b][c][t] = mel cache (first cache_frames[b] columns of the
// row's [80][8] slot cache) followed by the row's new frames from a frame-major packed buffer
// (CallDesc::mel_out layout [total_frames][80], row b's first new frame at frame_off[b]).
__global__ void pack_mel_k(int T, const float* fm, const int32_t* frame_off, const float* const* cache,
                           const int32_t* cache_frames, float* out) {
  const int b = blockIdx.y;
  const int cf = cache_frames ? cache_frames[b] : 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 80 * T; i += gridDim.x * blockDim.x) {
    const int c = i / T, t = i % T;
    float v;
    if (t < cf) v = cache[b][c * 8 + t];
    else v = fm[((size_t)frame_off[b] + (t - cf)) * 80 + c];
    out[((size_t)b * 80 + c) * T + t] = v;
  }
}

// Fused tail. One block = 256 output samples of one row.
// TL_N samples per block (256, or 128 when a call has too few blocks to fill the GPU).
template <int TL_N> struct TailGeo {
  static constexpr int F = TL_N / 4 + 4;     // frames whose windows touch the block
  static constexpr int XF = F + 6;           // conv_post input frames (k = 7, pad 3)
  static constexpr size_t smem = (64 * XF + 64 * 7 * 18 + 18 * F + F * 16) * sizeof(float);
};
struct TailArgs {
  InView x;                 // stage-2 resblock outputs [B][64][F]
  const float* w;           // conv_post [64][7][18]
  const float* bias;        // [18]
  const float* win;         // hann [16]
  const float* ham;         // np.hamming(7680) as float [7680]
  int F, L;
  float* wave;              // optional [B][L]: the clamped waveform (production's _hift_rows output)
  const Kt2nHiftRow* rows;  // optional: streaming epilogue
};

__device__ __forceinline__ float np_hamming(int n, int M) {
  // np.hamming(M)[n] = 0.54 + 0.46 cos(pi * (2n + 1 - M) / (M - 1)) in float64, then float32.
  if (M == 1) return 1.f;
  const double v = 0.54 + 0.46 * cos(M_PI * (double)(2 * n + 1 - M) / (double)(M - 1));
  return (float)v;
}

template <int TL_N>
__global__ void __launch_bounds__(256) tail_k(TailArgs ta) {
  constexpr int TL_F = TailGeo<TL_N>::F, TL_XF = TailGeo<TL_N>::XF;
  extern __shared__ __align__(16) float sm[];
  float* xin = sm;                        // [64][TL_XF]
  float* wsm = xin + 64 * TL_XF;          // [64 * 7 * 18]
  float* spec = wsm + 64 * 7 * 18;        // [18][TL_F]
  float* frm = spec + 18 * TL_F;          // [TL_F][16]
  const int b = blockIdx.y;
  const int n0 = blockIdx.x * TL_N;
  const int F = ta.F, L = ta.L;
  // frames f with 4f <= pos <= 4f + 15 for pos in [n0 + 8, n0 + 8 + 255]
  const int f_lo = max(0, (n0 + 8 - 15 + 3) / 4);
  const int f_hi = min(F - 1, (n0 + 8 + TL_N - 1) / 4);
  const int nf = f_hi - f_lo + 1;
  for (int i = threadIdx.x; i < 64 * 7 * 18; i += blockDim.x) wsm[i] = __ldg(&ta.w[i]);
  grid_dep_wait();
  for (int i = threadIdx.x; i < 64 * (nf + 6); i += blockDim.x) {
    const int c = i / (nf + 6), k = i % (nf + 6);
    const int f = f_lo - 3 + k;
    float v = 0.f;
    if (f >= 0 && f < F) {
      v = in_load(ta.x, ((size_t)b * 64 + c) * F + f);
      v = v > 0.f ? v : __fmul_rn(v, 0.01f);
    }
    xin[c * TL_XF + k] = v;
  }
  __syncthreads();
  for (int o = threadIdx.x; o < 18 * nf; o += blockDim.x) {
    const int co = o / nf, fl = o % nf;
    float acc = 0.f;
    for (int ci = 0; ci < 64; ++ci) {
      const float* xr = xin + ci * TL_XF + fl;
      const float* wr = wsm + ci * 7 * 18 + co;
#pragma unroll
      for (int j = 0; j < 7; ++j) acc = fmaf(wr[j * 18], xr[j], acc);
    }
    spec[co * TL_F + fl] = __fadd_rn(acc, ta.bias[co]);
  }
  __syncthreads();
  for (int fl = threadIdx.x; fl < nf; fl += blockDim.x) {
    float re[9], im[9];
#pragma unroll
    for (int m = 0; m < 9; ++m) {
      float mag = expf(spec[m * TL_F + fl]);
      mag = mag > 100.f ? 100.f : mag;
      const float ph = sinf(spec[(9 + m) * TL_F + fl]);
      re[m] = __fmul_rn(mag, cosf(ph));
      im[m] = __fmul_rn(mag, sinf(ph));
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      float acc = 0.f;
#pragma unroll
      for (int m = 1; m < 8; ++m) {
        acc = fmaf(re[m], c_cos[m][k], acc);
        acc = fmaf(-im[m], c_sin[m][k], acc);
      }
      float x = re[0] + ((k & 1) ? -re[8] : re[8]) + 2.f * acc;
      x = __fmul_rn(x, 1.0f / 16.0f);
      frm[fl * 16 + k] = __fmul_rn(x, ta.win[k]);
    }
  }
  __syncthreads();
  grid_dep_launch();
  if (threadIdx.x >= TL_N) return;
  const int n = n0 + threadIdx.x;
  if (n >= L) return;
  const int pos = n + 8;
  const int fa = max(0, (pos - 15 + 3) / 4), fb = min(F - 1, pos / 4);
  float y = 0.f, env = 0.f;
  for (int f = fa; f <= fb; ++f) {
    const int k = pos - 4 * f;
    y = __fadd_rn(y, frm[(f - f_lo) * 16 + k]);
    env = __fadd_rn(env, __fmul_rn(ta.win[k], ta.win[k]));
  }
  float v = __fdiv_rn(y, env);
  v = fminf(fmaxf(v, -0.99f), 0.99f);
  if (ta.wave) ta.wave[(size_t)b * L + n] = v;
  if (!ta.rows) return;
  const Kt2nHiftRow& R = ta.rows[b];
  if (R.flags & KT2N_HIFT_FIRST) {
    if (n < kTrim) v = __fmul_rn(v, c_trim[n]);
  } else if (n < R.overlap) {
    float rise, fall;
    if (R.flags & KT2N_HIFT_SHORT_XFADE) {
      rise = np_hamming(n, 2 * R.overlap);
      fall = np_hamming(R.overlap + n, 2 * R.overlap);
    } else {
      rise = ta.ham[n];
      fall = ta.ham[kSrcCache + n];
    }
    const float h = R.held_in[R.held_len - R.overlap + n];
    v = __fadd_rn(__fmul_rn(v, rise), __fmul_rn(h, fall));
  }
  if ((R.flags & KT2N_HIFT_HOLD) && R.held_out && n >= L - R.tail_len) R.held_out[n - (L - R.tail_len)] = v;
  if (n >= R.emit_len || R.out == nullptr) return;
  if (R.flags & KT2N_HIFT_SPLICE) {
    if (n < R.skip) return;
    const int o = n - R.skip;
    float e = v;
    if (o < R.splice_count) {
      // torch.linspace(0, 1, count + 2)[1:-1] on the device (fp32, ATen's two-sided formula).
      const int steps = R.splice_count + 2;
      const float step = __fdiv_rn(1.0f, (float)(steps - 1));
      const int ind = o + 1;
      const float ramp = ind < steps / 2 ? __fmul_rn(step, (float)ind) : fmaf(-step, (float)(steps - ind - 1), 1.0f);
      const float old = R.held_in[R.splice_start + o];
      e = __fadd_rn(__fmul_rn(old, __fsub_rn(1.0f, ramp)), __fmul_rn(v, ramp));
    }
    R.out[o] = fminf(fmaxf(e, -1.f), 1.f);
  } else {
    R.out[n] = fminf(fmaxf(v, -1.f), 1.f);
  }
}

// ------------------------------------------------------------------ host side

struct Conv { const float* w; const float* b; int cin, cout, K, dil, pad; const uint32_t* tc = nullptr; };
struct SnakeP { const float* a; const float* inv; };
struct ResBlockW { Conv c1[3], c2[3]; SnakeP s1[3], s2[3]; };
struct StageW {
  const float* up_w; const float* up_b; int up_cin, up_cout, up_k, up_s, up_p;
  const uint32_t* up_tc = nullptr; int up_kmax = 0; size_t up_phase_words = 0;
  const float* sd_w; const float* sd_b; int sd_cout, sd_k, sd_s, sd_p;
  ResBlockW sres; ResBlockW res[3];
};

#ifndef KT2N_HIFT_GROUP_DEFAULT
#define KT2N_HIFT_GROUP_DEFAULT 0
#endif
constexpr int kStreams = 4;  // 0 = caller (main), 1 = source chain, 2/3 = resblocks 1 and 2
struct StreamCtx {
  cudaStream_t s = nullptr;
  float* part = nullptr;
  size_t part_floats = 0;
};

struct Hift {
  Conv f0c[5]; const float* f0_cls_w; const float* f0_cls_b;
  const float* src_w; const float* src_b;
  Conv pre; StageW st[3]; Conv post; const float* win;
  float* post_w = nullptr;     // conv_post [64][7][18] (same as post.w)
  int max_b = 0, max_t = 0, max_fr = 0;
  int sms = 0, max_threads_sm = 0;
  int dev_sms = 0;   // the device's SM count: the Philox noise grid must match torch's (sms may be overridden)
  int tc_min = -1;   // per-handle tensor-core threshold (B * cout * lout >= tc_min * 1024); -1: g_big_min_tiles
  // v26 per-handle resblock grouping (kt2n_hift_set_group / KT2N_HIFT_GROUP): 0 = the three branches on three
  // streams (v21); 1 = one launch per layer for all three branches, v21's tiles / splits (bitwise v21); 2 = one
  // launch, splits balanced by taps (K = 3 / 7 / 11 -> u / 2u / 4u); 3 = one launch, uniform tiles / splits sized
  // for the 3x larger grid (2 and 3: fewer channel splits than v21; rounding-level change).
  int group = KT2N_HIFT_GROUP_DEFAULT;
  // Tensor-core split target (blocks >= tc_target_mul x SMs before channel splits stop; KT2N_HIFT_TC_TARGET_MUL at
  // create). More = shorter blocks (a higher-priority kernel waits less for a free SM slot) but more partial sums.
  int tc_target_mul = KT2N_TC_TARGET_MUL;
  // workspace
  float *f0a, *f0b, *f0, *noise, *sines, *s, *sst, *xpre, *xup;
  float* chain[3][3];          // resblock branch j: t1, ra, rb (y_j = ra)
  float* chain_sx[3];          // resblock branch j: Snake-activated residual stream
  float* src_chain[3][4];      // per stage: si, t1, ra, rb
  float* src_sx[3];
  float* ham = nullptr;        // np.hamming(7680) float
  unsigned* ctr = nullptr;
  size_t ctr_n = 0, ctr_next = 0;
  StreamCtx sc[kStreams];
  // Side streams (source chain, resblock branches 1 / 2), one set per stream-priority level: run_impl uses the
  // set whose priority equals the caller stream's, so an urgent call's forked branches stay urgent in eager
  // use (L7 found the side streams at the lowest priority: 3 of HiFT's 4 branches of an urgent first commit
  // ran at bulk priority). Inside a captured graph the launch stream's priority applies to every node anyway.
  static constexpr int kMaxPrio = 8;
  cudaStream_t side[kMaxPrio][kStreams - 1] = {};
  int prio_least = 0, prio_greatest = 0, n_prio = 0;
  cudaEvent_t ev_fork, ev_noise, ev_src[3], ev_x[3], ev_y[3][2];
  std::vector<void*> allocs;
};

const int kUpRates[3] = {8, 5, 3};
const int kUpK[3] = {16, 11, 7};
const int kChan[3] = {256, 128, 64};
const int kSrcK[3] = {7, 7, 11};
const int kResK[3] = {3, 7, 11};
const int kDil[3] = {1, 3, 5};
constexpr int kHiftWeightPtrs = 16 + 3 * 100 + 3;
static_assert(kHiftWeightPtrs == 319, "the HiFT weight list has 319 tensors");

char g_err[512];
int g_conv_mode = 4;          // 4: tensor-core conv (TF32, production's precision class) above
int g_big_min_tiles = 128;    // mode 4: B * cout * lout >= this * 1024; modes 1-3: 64x128 tiles per row
int g_pdl = 1;
int g_fuse_snake = 0;       // KT2N_HIFT_FUSE_SNAKE=1: Snake in the producer's epilogue (measured slower at batch)
int g_convt_big = 0;        // KT2N_HIFT_CONVT_BIG=1: register-blocked convT (measured slower: 3.6 vs 1.4 ms at B10x58)
int g_parallel = 1;
// Grid sizing (v25 experiment): every split / tile heuristic sizes its grid for h.sms SMs and kTargetBlocks SIMT
// blocks. Fewer = fewer channel splits (less partial-sum work and fewer blocks per call, longer latency): a bulk
// handle sharing the GPU with the LM may prefer that. KT2N_HIFT_SMS / KT2N_HIFT_TARGET_BLOCKS override at create.
#ifndef KT2N_HIFT_TARGET_DEFAULT
#define KT2N_HIFT_TARGET_DEFAULT 512
#endif
#ifndef KT2N_HIFT_SMS_DEFAULT
#define KT2N_HIFT_SMS_DEFAULT 0   // 0: the device's SM count
#endif
int kTargetBlocks = KT2N_HIFT_TARGET_DEFAULT;  // SIMT split target per call (KT2N_HIFT_TARGET_BLOCKS, read at create)

template <typename... KArgs, typename... Args>
void launch(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = kt2n::cuda::pdl_enabled(g_pdl != 0) ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

// Split count from the call's tile count (rows x tiles per row). Not batch-invariant: a per-row
// split (the spike's rule) over-splits batched calls (B10 x 10 frames: 5,120 blocks of 4 channels
// plus 16x partial-sum traffic) and the tensor-core path already sizes its split per call.
int choose_ks(long tiles, int cin) {
  const int per_warp_max = cin / NW;
  int target = (int)(kTargetBlocks / (tiles > 0 ? tiles : 1));
  if (target < 1) target = 1;
  for (int ks = std::min(target, per_warp_max); ks > 1; --ks)
    if (per_warp_max % ks == 0) return ks;
  return 1;
}

unsigned* take_ctr(Hift& h, size_t n) {
  if (h.ctr_next + n > h.ctr_n) return nullptr;
  unsigned* p = h.ctr + h.ctr_next;
  h.ctr_next += n;
  return p;
}

int tc_tg(int K) { return K == 11 ? KT2N_TC_TG11 : K == 7 ? KT2N_TC_TG7 : K; }
size_t tc_smem_bytes(int K, int dil, int BN) {
  const int xw = BN + (K - 1) * dil;
  const int xw_pad = (xw + 3) & ~3;
  return (size_t)(KT2N_TC_WSN * tc_xc(K) * tc_tg(K) * 4 * 32 * 4 + 2 * tc_xc(K) * xw_pad * 8) * sizeof(uint32_t);
}

bool launch_conv(Hift& h, const ConvP& p, int B, StreamCtx& sc) {
  const long tc_min = h.tc_min >= 0 ? h.tc_min : g_big_min_tiles;
  if (g_conv_mode >= 4 && p.tc && (long)B * p.cout * p.lout >= tc_min * 1024 && p.cin % 8 == 0 &&
      p.cout % TC_BM == 0 && (p.K == 3 || p.K == 7 || p.K == 11) && p.dil * (p.K - 1) <= 50 &&
      (tc_xc(p.K) == 1 || (p.dil <= TC_MAX_DIL && (p.cin / 8) % tc_xc(p.K) == 0))) {
    const int target = h.tc_target_mul * h.sms;
    int BN = 128;
    long blocks = (long)B * (p.cout / TC_BM) * ((p.lout + 127) / 128);
    if (KT2N_TC_WIDE_MINK > 0 && p.K >= KT2N_TC_WIDE_MINK &&
        (long)B * (p.cout / TC_BM) * ((p.lout + 255) / 256) >= h.sms) {
      BN = 256;
      blocks = (long)B * (p.cout / TC_BM) * ((p.lout + 255) / 256);
    } else if (blocks < KT2N_TC_BN64_MUL * h.sms && g_conv_mode == 4) {  // mode 5: always 128-wide tiles
      BN = 64;
      blocks = (long)B * (p.cout / TC_BM) * ((p.lout + 63) / 64);
    }
    const int nch = p.cin / 8;
    int ks = 1;
    if (blocks < target && BN != 256) {
      const int want = (int)((target + blocks - 1) / blocks);
      for (int c = std::min(want, nch / 2); c > 1; --c)
        if (nch % c == 0 && (nch / c) % tc_xc(p.K) == 0) { ks = c; break; }
    }
    const int bt = (p.lout + BN - 1) / BN, bc = p.cout / TC_BM;
    if (ks > 1 && (size_t)ks * B * p.cout * p.lout > sc.part_floats) ks = 1;
    unsigned* ctr = nullptr;
    if (ks > 1) {
      ctr = take_ctr(h, (size_t)bt * bc * B);
      if (!ctr) ks = 1;
    }
    const size_t smem = tc_smem_bytes(p.K, p.dil, BN);
    const dim3 grid(bt, bc, B * ks);
#define KT2N_TC(BNV, KV) launch(conv1d_tc_k<BNV, KV, (KV == 11 ? KT2N_TC_TG11 : KV == 7 ? KT2N_TC_TG7 : KV)>, grid, \
                                dim3(256), smem, sc.s, p, p.tc, ks, sc.part, ctr)
    if (BN == 256) {
#if KT2N_TC_WIDE_MINK > 0
      if (p.K == 3) KT2N_TC(256, 3); else if (p.K == 7) KT2N_TC(256, 7); else KT2N_TC(256, 11);
#endif
    } else if (BN == 128) {
      if (p.K == 3) KT2N_TC(128, 3); else if (p.K == 7) KT2N_TC(128, 7); else KT2N_TC(128, 11);
    } else {
      if (p.K == 3) KT2N_TC(64, 3); else if (p.K == 7) KT2N_TC(64, 7); else KT2N_TC(64, 11);
    }
#undef KT2N_TC
    return true;
  }
  const int bt = (p.lout + BG_T - 1) / BG_T, bc = (p.cout + BG_CO - 1) / BG_CO;
  if (g_conv_mode >= 1 && g_conv_mode <= 3 && bt * bc >= g_big_min_tiles && p.cin % BG_CI == 0 && p.K <= WK_MAX &&
      p.dil * (p.K - 1) <= 50) {
    if (g_conv_mode == 1) {
      launch(conv1d_big_k, dim3(bt, bc, B), dim3(256), 0, sc.s, p);
    } else {
      const size_t smem = (BG_CI * MM_XW + BG_CI * (p.K * BG_CO + 8)) * sizeof(float);
      if (g_conv_mode == 2) launch(conv1d_mma_k<0>, dim3(bt, bc, B), dim3(256), smem, sc.s, p);
      else launch(conv1d_mma_k<1>, dim3(bt, bc, B), dim3(256), smem, sc.s, p);
    }
    return true;
  }
  const int tt = (p.lout + T_T - 1) / T_T, ct = (p.cout + CO_T - 1) / CO_T;
  int ks = choose_ks((long)tt * ct * B, p.cin);
  if (ks > 1 && (size_t)ks * B * p.cout * p.lout > sc.part_floats) ks = 1;
  unsigned* ctr = nullptr;
  if (ks > 1) {
    ctr = take_ctr(h, (size_t)tt * ct * B);
    if (!ctr) ks = 1;
  }
  launch(conv1d_k, dim3(tt, ct, B * ks), dim3(256), 0, sc.s, p, ks, sc.part, ctr);
  return true;
}

// v26: can this layer's conv run on the tensor-core path (launch_conv's condition, XC = 1 builds)?
bool tc_eligible(const Hift& h, const ConvP& p, int B) {
  const long tc_min = h.tc_min >= 0 ? h.tc_min : g_big_min_tiles;
  return g_conv_mode >= 4 && p.tc && (long)B * p.cout * p.lout >= tc_min * 1024 && p.cin % 8 == 0 &&
         p.cout % TC_BM == 0 && (p.K == 3 || p.K == 7 || p.K == 11) && p.dil * (p.K - 1) <= 50 && tc_xc(p.K) == 1;
}

// v26: one launch for the three resblock branches' layer-k convs (g.p[j] has K = kResK[j]), on sc[0]'s stream;
// branch j uses partial-sum buffer sc[j]. fill = 1: tiles / splits chosen exactly as launch_conv chooses them for
// ONE branch (so bitwise v21); fill = 3: chosen for the whole 3-branch grid (fewer splits, less partial-sum work).
bool launch_conv3(Hift& h, ConvP3& g, int B, StreamCtx* const sc[3], int mode) {
  const ConvP& p = g.p[0];
  for (int j = 0; j < 3; ++j)
    if (!tc_eligible(h, g.p[j], B) || g.p[j].K != kResK[j] || g.p[j].cout != p.cout || g.p[j].lout != p.lout ||
        g.p[j].cin != p.cin || g.p[j].dil != p.dil)
      return false;
  const int target = h.tc_target_mul * h.sms;
  const int nch = p.cin / 8;
  const long p128 = (long)B * (p.cout / TC_BM) * ((p.lout + 127) / 128);
  const long p64 = (long)B * (p.cout / TC_BM) * ((p.lout + 63) / 64);
  int BN = 128;
  int ks[3] = {1, 1, 1};
  if (mode == 2) {
    // Balanced: splits ~ taps (K = 3 / 7 / 11 -> u, 2u, 4u), so every block does about the same work (3 n / u,
    // 3.5 n / u, 2.75 n / u tap-chunks) and a layer ends without a long K = 11 tail; the same partial-sum traffic
    // as a uniform 2u split. No split when one tile per branch already fills the GPU.
    if (3 * p128 >= target) {
      BN = 128;
    } else if (3 * p64 >= target) {
      BN = 64;
    } else if (7 * p128 >= target && nch >= 8) {
      BN = 128; ks[1] = 2; ks[2] = 4;
    } else {
      BN = 64;
      int u = 1;
      while (7L * u * p64 < target && 8 * u <= nch / 2) u *= 2;   // 4u <= nch / 2 for the K = 11 branch
      if (nch >= 8) { ks[0] = u; ks[1] = 2 * u; ks[2] = 4 * u; }
    }
  } else {
    const int fill = mode == 1 ? 1 : 3;   // 1: v21's choice for ONE branch (bitwise); 3: for the 3-branch grid
    long per = p128;
    if (per * fill < KT2N_TC_BN64_MUL * h.sms && g_conv_mode == 4) {
      BN = 64;
      per = p64;
    }
    const long blocks = per * fill;
    int k1 = 1;
    if (blocks < target) {
      const int want = (int)((target + blocks - 1) / blocks);
      for (int c = std::min(want, nch / 2); c > 1; --c)
        if (nch % c == 0) { k1 = c; break; }
    }
    ks[0] = ks[1] = ks[2] = k1;
  }
  const int bt = (p.lout + BN - 1) / BN, bc = p.cout / TC_BM;
  for (int j = 0; j < 3; ++j) {
    if (ks[j] > 1 && (size_t)ks[j] * B * p.cout * p.lout > sc[j]->part_floats) ks[j] = 1;
    g.part[j] = sc[j]->part;
    g.ctr[j] = nullptr;
    if (ks[j] > 1) {
      g.ctr[j] = take_ctr(h, (size_t)bt * bc * B);
      if (!g.ctr[j]) ks[j] = 1;
    }
    g.ks[j] = ks[j];
  }
  size_t smem = 0;
  for (int j = 0; j < 3; ++j) smem = std::max(smem, tc_smem_bytes(g.p[j].K, p.dil, BN));
  const dim3 grid(bt, bc, B * (ks[0] + ks[1] + ks[2]));
  if (BN == 128) launch(conv1d_tc3_k<128>, grid, dim3(256), smem, sc[0]->s, g, B);
  else launch(conv1d_tc3_k<64>, grid, dim3(256), smem, sc[0]->s, g, B);
  return true;
}

void launch_convT_tc(Hift& h, const ConvTP& p, const StageW& w, int B, StreamCtx& sc) {
  const int lraw = p.lout - p.reflect_left;
  ConvP q{};
  q.x = p.x; q.cin = p.cin; q.lin = p.lin; q.y = p.y; q.cout = p.cout;
  q.lout = (lraw + p.pad + p.stride - 1) / p.stride + 1;   // u range [0, U)
  q.w = nullptr; q.bias = p.bias; q.K = w.up_kmax; q.dil = 1; q.pad = w.up_kmax - 1;
  q.pre = PRE_LEAKY; q.slope = p.slope; q.alpha = nullptr; q.inv = nullptr; q.res = nullptr; q.add_before = nullptr;
  q.tc = w.up_tc; q.out_stride = p.stride; q.out_shift = -p.pad; q.out_len = lraw; q.reflect = p.reflect_left;
  q.phase_words = w.up_phase_words;
  const int BN = ((long)B * p.stride * (q.cout / TC_BM) * ((q.lout + 127) / 128) >= 2L * h.sms) ? 128 : 64;
  const dim3 grid((q.lout + BN - 1) / BN, q.cout / TC_BM, B * p.stride);
  const size_t smem = tc_smem_bytes(q.K, 1, BN);
  if (BN == 128) {
    if (q.K == 2) launch(conv1d_tc_k<128, 2>, grid, dim3(256), smem, sc.s, q, q.tc, 1, sc.part, (unsigned*)nullptr);
    else launch(conv1d_tc_k<128, 3>, grid, dim3(256), smem, sc.s, q, q.tc, 1, sc.part, (unsigned*)nullptr);
  } else {
    if (q.K == 2) launch(conv1d_tc_k<64, 2>, grid, dim3(256), smem, sc.s, q, q.tc, 1, sc.part, (unsigned*)nullptr);
    else launch(conv1d_tc_k<64, 3>, grid, dim3(256), smem, sc.s, q, q.tc, 1, sc.part, (unsigned*)nullptr);
  }
}

void launch_convT(Hift& h, const ConvTP& p, int B, StreamCtx& sc) {
  const long big_blocks = (long)B * ((p.lout + CTB_T - 1) / CTB_T) * ((p.cout + CTB_CO - 1) / CTB_CO);
  if (g_convt_big && big_blocks >= 2L * h.sms && p.cin % CTB_CI == 0 && p.K <= CT_WK_MAX) {
    launch(convT_big_k, dim3((p.lout + CTB_T - 1) / CTB_T, (p.cout + CTB_CO - 1) / CTB_CO, B), dim3(256), 0, sc.s, p);
    return;
  }
  const int tt = (p.lout + T_T - 1) / T_T, ct = (p.cout + CO_T - 1) / CO_T;
  int ks = choose_ks((long)tt * ct * B, p.cin);
  if (ks > 1 && (size_t)ks * B * p.cout * p.lout > sc.part_floats) ks = 1;
  unsigned* ctr = nullptr;
  if (ks > 1) {
    ctr = take_ctr(h, (size_t)tt * ct * B);
    if (!ctr) ks = 1;
  }
  launch(convT_k, dim3(tt, ct, B * ks), dim3(256), 0, sc.s, p, ks, sc.part, ctr);
}

ConvP conv_params(const Conv& c, InView x, int lin, float* y, int lout) {
  ConvP p{};
  p.x = x; p.cin = c.cin; p.lin = lin; p.y = y; p.cout = c.cout; p.lout = lout;
  p.w = c.w; p.bias = c.b; p.K = c.K; p.dil = c.dil; p.pad = c.pad;
  p.pre = PRE_NONE; p.slope = 0.f; p.alpha = nullptr; p.inv = nullptr;
  p.res = nullptr; p.add_before = nullptr; p.tc = c.tc;
  p.y_act = nullptr; p.post = PRE_NONE; p.post_alpha = nullptr; p.post_inv = nullptr;
  p.out_stride = 0; p.out_shift = 0; p.out_len = 0; p.reflect = 0; p.phase_words = 0;
  return p;
}

// One resblock chain: three (snake -> conv(dil d) -> snake -> conv(1) -> + residual).
// conv_begin..conv_end selects a part of the chain (the source resblock's last conv runs on the main
// stream after the up-conv). Snake is applied once per element, in the producing conv's epilogue
// (y_act), except for the chain's first conv, whose input is shared by several chains (Snake in staging).
// Buffers: t1 (activated inner output), ra / rb (residual stream), sx (activated residual stream).
void run_resblock(Hift& h, const ResBlockW& rb, const float* in, int L, int B, float* t1, float* ra, float* rb_,
                  float* sx, StreamCtx& sc, int conv_begin, int conv_end, float* final_y, const float* final_add) {
  const float* cur = in;
  float* bufs[3] = {ra, rb_, ra};  // k = 0 -> ra, 1 -> rb, 2 -> ra (y)
  if (!g_fuse_snake) {  // v10 chain: Snake applied while each conv stages its input
    for (int k = 0; k < 3; ++k) {
      float* out = (k == 2 && final_y) ? final_y : bufs[k];
      if (2 * k >= conv_begin && 2 * k < conv_end) {
        ConvP p1 = conv_params(rb.c1[k], single(cur), L, t1, L);
        p1.pre = PRE_SNAKE; p1.alpha = rb.s1[k].a; p1.inv = rb.s1[k].inv;
        launch_conv(h, p1, B, sc);
      }
      if (2 * k + 1 >= conv_begin && 2 * k + 1 < conv_end) {
        ConvP p2 = conv_params(rb.c2[k], single(t1), L, out, L);
        p2.pre = PRE_SNAKE; p2.alpha = rb.s2[k].a; p2.inv = rb.s2[k].inv;
        p2.res = cur;
        if (k == 2) p2.add_before = final_add;
        launch_conv(h, p2, B, sc);
      }
      cur = out;
    }
    return;
  }
  for (int k = 0; k < 3; ++k) {
    float* out = (k == 2 && final_y) ? final_y : bufs[k];
    if (2 * k >= conv_begin && 2 * k < conv_end) {
      ConvP p1 = conv_params(rb.c1[k], single(k == 0 ? cur : sx), L, nullptr, L);
      if (k == 0) { p1.pre = PRE_SNAKE; p1.alpha = rb.s1[k].a; p1.inv = rb.s1[k].inv; }
      p1.y_act = t1; p1.post = PRE_SNAKE; p1.post_alpha = rb.s2[k].a; p1.post_inv = rb.s2[k].inv;
      launch_conv(h, p1, B, sc);
    }
    if (2 * k + 1 >= conv_begin && 2 * k + 1 < conv_end) {
      ConvP p2 = conv_params(rb.c2[k], single(t1), L, out, L);
      p2.res = cur;
      if (k == 2) p2.add_before = final_add;
      if (k < 2) { p2.y_act = sx; p2.post = PRE_SNAKE; p2.post_alpha = rb.s1[k + 1].a; p2.post_inv = rb.s1[k + 1].inv; }
      launch_conv(h, p2, B, sc);
    }
    cur = out;
  }
  return;
}

// v26: the three resblock branches of a stage, one launch per layer (launch_conv3), all on sc[0]'s stream. Same
// math as three run_resblock calls (v10 chain: Snake applied while each conv stages its input). Returns false (and
// launches nothing) when a layer cannot take the grouped tensor-core path; the caller then uses the 3-stream path.
bool run_resblock3(Hift& h, const ResBlockW* const rb[3], const float* in, int L, int B, float* const t1[3],
                   float* const ra[3], float* const rbuf[3], StreamCtx* const sc[3], int fill) {
  if (g_fuse_snake) return false;
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < 3; ++j) {
      if (!tc_eligible(h, conv_params(rb[j]->c1[k], single(in), L, nullptr, L), B)) return false;
      if (!tc_eligible(h, conv_params(rb[j]->c2[k], single(in), L, nullptr, L), B)) return false;
    }
  const float* cur[3] = {in, in, in};
  for (int k = 0; k < 3; ++k) {
    float* out[3];
    for (int j = 0; j < 3; ++j) out[j] = k == 1 ? rbuf[j] : ra[j];   // k = 0 -> ra, 1 -> rb, 2 -> ra (y)
    ConvP3 g1{};
    for (int j = 0; j < 3; ++j) {
      ConvP p1 = conv_params(rb[j]->c1[k], single(cur[j]), L, t1[j], L);
      p1.pre = PRE_SNAKE; p1.alpha = rb[j]->s1[k].a; p1.inv = rb[j]->s1[k].inv;
      g1.p[j] = p1;
    }
    if (!launch_conv3(h, g1, B, sc, fill)) return false;   // not reached: eligibility checked above
    ConvP3 g2{};
    for (int j = 0; j < 3; ++j) {
      ConvP p2 = conv_params(rb[j]->c2[k], single(t1[j]), L, out[j], L);
      p2.pre = PRE_SNAKE; p2.alpha = rb[j]->s2[k].a; p2.inv = rb[j]->s2[k].inv;
      p2.res = cur[j];
      g2.p[j] = p2;
    }
    if (!launch_conv3(h, g2, B, sc, fill)) return false;
    for (int j = 0; j < 3; ++j) cur[j] = out[j];
  }
  return true;
}

struct RunIO {
  int B, T;
  const float* mel;
  const float* phase_ext;   // [B][9] (stride 9) or null (row mode)
  const float* noise_ext;   // [B][9][L] or null (row mode: generated)
  SrcArgs src;
  float* wave;
  float* source;            // may be null: the internal buffer is used
  const Kt2nHiftRow* rows;
};

int run_impl(Hift& h, const RunIO& io, cudaStream_t main) {
  const int B = io.B, T = io.T;
  if (B > h.max_b || T > h.max_t || B < 1 || T < 1 || B * T > h.max_fr) {
    snprintf(g_err, sizeof g_err, "shape B=%d T=%d exceeds workspace (%d, %d, %d frame-rows)", B, T, h.max_b, h.max_t,
             h.max_fr);
    return 1;
  }
  const int L = 480 * T, F = 120 * T + 1;
  h.ctr_next = 0;
  StreamCtx& M = h.sc[0];
  M.s = main;
  {
    int prio = h.prio_least;
    if (cudaStreamGetPriority(main, &prio) != cudaSuccess) {
      (void)cudaGetLastError();
      prio = h.prio_least;
    }
    const int lvl = std::min(std::max(prio - h.prio_greatest, 0), h.n_prio - 1);
    for (int k = 1; k < kStreams; ++k) h.sc[k].s = h.side[lvl][k - 1];
  }
  StreamCtx& A = g_parallel ? h.sc[1] : M;
  StreamCtx& R1 = g_parallel ? h.sc[2] : M;
  StreamCtx& R2 = g_parallel ? h.sc[3] : M;
  auto rec = [&](cudaEvent_t e, StreamCtx& s) { if (g_parallel) cudaEventRecord(e, s.s); };
  auto wait = [&](StreamCtx& s, cudaEvent_t e) { if (g_parallel) cudaStreamWaitEvent(s.s, e, 0); };
  rec(h.ev_fork, M);
  wait(A, h.ev_fork);
  // --- native draws on R2 (idle until the first resblocks), joined before the source merge
  const float* noise = io.noise_ext;
  if (!noise) {
    wait(R2, h.ev_fork);
    const int numel = 9 * L;
    const int bps = h.max_threads_sm / 256;
    const int gx = std::min<long>((long)h.dev_sms * bps, (numel + 255) / 256);
    NoiseArgs na{io.rows, nullptr, io.mel, T};
    launch(noise_k, dim3(gx, B), dim3(256), 0, R2.s, na, numel, h.noise);
    rec(h.ev_noise, R2);
    noise = h.noise;
  }
  // --- source chain (stream A): f0 predictor, sines, merge, STFT, source branches
  float* fbuf[2] = {h.f0a, h.f0b};
  const float* cur = io.mel;
  for (int c = 0; c < 5; ++c) {
    ConvP p = conv_params(h.f0c[c], single(cur), T, fbuf[c & 1], T);
    if (c > 0) p.pre = PRE_ELU;
    launch_conv(h, p, B, A);
    cur = fbuf[c & 1];
  }
  launch(f0_head_k, dim3((B * T * 32 + 255) / 256), dim3(256), 0, A.s, cur, 512, T, h.f0_cls_w, h.f0_cls_b, h.f0, B);
  {
    const int lx = torch_scan_log_nx(9, L);
    const float* phase = io.phase_ext;
    int pstride = 9;
    if (!phase) {
      phase = reinterpret_cast<const float*>(reinterpret_cast<const char*>(io.rows) + offsetof(Kt2nHiftRow, phase));
      pstride = sizeof(Kt2nHiftRow) / sizeof(float);
    }
    launch(sine_rows_k, dim3(B * 9), dim3(1 << lx), (size_t)(2 << lx) * sizeof(float), A.s, (const float*)h.f0, T,
           phase, pstride, h.sines, lx, 9);
  }
  float* src = io.source ? io.source : h.s;
  if (!io.noise_ext) wait(A, h.ev_noise);
  launch(source_stft_k, dim3((F + 63) / 64, B), dim3(256), 0, A.s, (const float*)h.f0, T, (const float*)h.sines,
         noise, h.src_w, h.src_b, io.src, src, h.win, h.sst, F);
  int lens[3];
  {
    int lin = T;
    for (int s = 0; s < 3; ++s) {
      const StageW& w = h.st[s];
      const int lraw = (lin - 1) * w.up_s - 2 * w.up_p + w.up_k;
      lens[s] = s == 2 ? lraw + 1 : lraw;
      lin = lens[s];
    }
  }
  for (int s = 0; s < 3; ++s) {
    const StageW& w = h.st[s];
    const int lout = lens[s];
    const int lsd = (F + 2 * w.sd_p - w.sd_k) / w.sd_s + 1;
    if (lsd != lout) {
      snprintf(g_err, sizeof g_err, "stage %d: source length %d != %d", s, lsd, lout);
      return 2;
    }
    const long sd_blocks = (long)B * ((lout + 255) / 256) * (w.sd_cout / SD_CO);
    const dim3 sd_grid((lout + 255) / 256, w.sd_cout / SD_CO, B);
    if (KT2N_SD_SM && sd_blocks >= 2L * h.sms && w.sd_s == 15 && w.sd_k == 30) {
      launch(conv_strided_sm_k<15, 30>, sd_grid, dim3(256), sd_sm_bytes<15, 30>(18), A.s, (const float*)h.sst, 18,
             F, h.src_chain[s][0], w.sd_cout, lout, (const float*)w.sd_w, (const float*)w.sd_b, w.sd_p);
    } else if (KT2N_SD_SM && sd_blocks >= 2L * h.sms && w.sd_s == 3 && w.sd_k == 6) {
      launch(conv_strided_sm_k<3, 6>, sd_grid, dim3(256), sd_sm_bytes<3, 6>(18), A.s, (const float*)h.sst, 18,
             F, h.src_chain[s][0], w.sd_cout, lout, (const float*)w.sd_w, (const float*)w.sd_b, w.sd_p);
    } else if (KT2N_SD_SM && sd_blocks >= 2L * h.sms && w.sd_s == 1 && w.sd_k == 1) {
      launch(conv_strided_sm_k<1, 1>, sd_grid, dim3(256), sd_sm_bytes<1, 1>(18), A.s, (const float*)h.sst, 18,
             F, h.src_chain[s][0], w.sd_cout, lout, (const float*)w.sd_w, (const float*)w.sd_b, w.sd_p);
    } else if (sd_blocks >= 2L * h.sms) {
      launch(conv_strided_k, dim3((lout + 255) / 256, w.sd_cout / SD_CO, B), dim3(256),
             (size_t)SD_CO * 18 * w.sd_k * sizeof(float), A.s, (const float*)h.sst, 18, F, h.src_chain[s][0],
             w.sd_cout, lout, w.sd_w, w.sd_b, w.sd_k, w.sd_s, w.sd_p);
    } else {
      const size_t total = (size_t)B * w.sd_cout * lout;
      launch(conv_strided1_k, dim3((unsigned)std::min<size_t>((total + 255) / 256, 65535)), dim3(256), 0, A.s,
             (const float*)h.sst, 18, F, h.src_chain[s][0], w.sd_cout, lout, w.sd_w, w.sd_b, w.sd_k, w.sd_s, w.sd_p,
             B);
    }
    run_resblock(h, w.sres, h.src_chain[s][0], lout, B, h.src_chain[s][1], h.src_chain[s][2], h.src_chain[s][3],
                 h.src_sx[s], A, 0, 5, nullptr, nullptr);
    rec(h.ev_src[s], A);
  }
  // --- main chain
  {
    ConvP p = conv_params(h.pre, single(io.mel), T, h.xpre, T);
    launch_conv(h, p, B, M);
  }
  InView xin = single(h.xpre);
  int lin = T;
  for (int s = 0; s < 3; ++s) {
    const StageW& w = h.st[s];
    const int lout = lens[s];
    {
      ConvTP p{};
      p.x = xin; p.cin = w.up_cin; p.lin = lin; p.y = h.xup; p.cout = w.up_cout; p.lout = lout;
      p.w = w.up_w; p.bias = w.up_b; p.K = w.up_k; p.stride = w.up_s; p.pad = w.up_p; p.slope = 0.1f;
      p.reflect_left = s == 2;
      if (g_conv_mode >= 4 && w.up_tc &&
          (long)B * p.cout * p.lout >= (long)(h.tc_min >= 0 ? h.tc_min : g_big_min_tiles) * 1024)
        launch_convT_tc(h, p, w, B, M);
      else
        launch_convT(h, p, B, M);
    }
    // x = up + source_resblock(si): the source chain's last conv, on the main stream.
    wait(M, h.ev_src[s]);
    run_resblock(h, w.sres, h.src_chain[s][0], lout, B, h.src_chain[s][1], h.src_chain[s][2], h.src_chain[s][3],
                 h.src_sx[s], M, 5, 6, h.xup, h.xup);
    bool grouped = false;
    if (h.group > 0) {   // v26: one launch per layer for the three branches, on the main stream
      const ResBlockW* rbs[3] = {&w.res[0], &w.res[1], &w.res[2]};
      float* const t1s[3] = {h.chain[0][0], h.chain[1][0], h.chain[2][0]};
      float* const ras[3] = {h.chain[0][1], h.chain[1][1], h.chain[2][1]};
      float* const rbb[3] = {h.chain[0][2], h.chain[1][2], h.chain[2][2]};
      StreamCtx* const scs[3] = {&M, &h.sc[2], &h.sc[3]};   // partial-sum buffers per branch (as v21's streams)
      grouped = run_resblock3(h, rbs, h.xup, lout, B, t1s, ras, rbb, scs, h.group);
    }
    if (!grouped) {
      rec(h.ev_x[s], M);
      wait(R1, h.ev_x[s]);
      wait(R2, h.ev_x[s]);
      StreamCtx* bs[3] = {&M, &R1, &R2};
      for (int j = 0; j < 3; ++j)
        run_resblock(h, w.res[j], h.xup, lout, B, h.chain[j][0], h.chain[j][1], h.chain[j][2], h.chain_sx[j], *bs[j],
                     0, 6, nullptr, nullptr);
      rec(h.ev_y[s][0], R1);
      rec(h.ev_y[s][1], R2);
      wait(M, h.ev_y[s][0]);
      wait(M, h.ev_y[s][1]);
    }
    xin = InView{h.chain[0][1], h.chain[1][1], h.chain[2][1], 1.0f / 3.0f};
    lin = lout;
  }
  // --- fused tail
  {
    TailArgs ta{};
    ta.x = xin; ta.w = h.post_w; ta.bias = h.post.b; ta.win = h.win; ta.ham = h.ham;
    ta.F = F; ta.L = L; ta.wave = io.wave; ta.rows = io.rows;
    if ((long)B * ((L + 255) / 256) >= 2L * h.sms)
      launch(tail_k<256>, dim3((L + 255) / 256, B), dim3(256), TailGeo<256>::smem, M.s, ta);
    else
      launch(tail_k<128>, dim3((L + 127) / 128, B), dim3(256), TailGeo<128>::smem, M.s, ta);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    snprintf(g_err, sizeof g_err, "launch: %s", cudaGetErrorString(err));
    return 3;
  }
  return 0;
}

// torch CPU mt19937 (at::mt19937, seeded with the low 32 bits).
struct Mt19937 {
  uint32_t mt[624];
  int idx;
  explicit Mt19937(uint64_t seed) {
    mt[0] = (uint32_t)(seed & 0xffffffffu);
    for (int j = 1; j < 624; ++j) mt[j] = 1812433253u * (mt[j - 1] ^ (mt[j - 1] >> 30)) + (uint32_t)j;
    idx = 624;
  }
  uint32_t next() {
    if (idx >= 624) {
      for (int k = 0; k < 624; ++k) {
        const uint32_t y = (mt[k] & 0x80000000u) | (mt[(k + 1) % 624] & 0x7fffffffu);
        mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
      }
      idx = 0;
    }
    uint32_t y = mt[idx++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
  }
};

}  // namespace

extern "C" {

const char* kt2n_hift_last_error() { return g_err; }

// Per-handle work policy (v25). tc_min_k: convolutions with B * cout * lout >= tc_min_k * 1024 outputs run on the
// tensor cores, smaller ones on the SIMT kernels (lower latency for a lone tiny call, but several times more GPU
// work per output). -1 = the global default (128, first-commit latency). sms: size
// the split / tile heuristics for this many SMs (0 = the device's). Call after create, before any run / capture.
int kt2n_hift_set_policy(void* handle, int tc_min_k, int sms) {
  if (!handle) return 1;
  Hift& h = *static_cast<Hift*>(handle);
  h.tc_min = tc_min_k < 0 ? -1 : tc_min_k;
  h.sms = sms > 0 ? std::min(sms, h.dev_sms) : h.dev_sms;
  return 0;
}

int kt2n_hift_set_group(void* handle, int mode) {
  if (!handle || mode < 0 || mode > 3) return 1;
  static_cast<Hift*>(handle)->group = mode;
  return 0;
}

void kt2n_hift_config(int conv_mode, int big_min_tiles, int pdl, int parallel) {
  g_conv_mode = conv_mode;
  g_big_min_tiles = big_min_tiles;
  g_pdl = pdl;
  g_parallel = parallel;
}

void kt2n_hift_phases(uint64_t seed, float* out9) {
  Mt19937 g(seed);
  const float low = (float)-M_PI, high = (float)M_PI;
  volatile float range = high - low;
  for (int k = 0; k < 9; ++k) {
    const uint32_t r = g.next();
    const float u = (float)(r & ((1u << 24) - 1)) * (1.0f / 16777216.0f);
    volatile float prod = u * range;
    out9[k] = low + prod;
  }
  out9[0] = 0.f;
}

void kt2n_hift_plan(const Kt2nHiftPlanIn* in, Kt2nHiftPlanOut* out) {
  std::memset(out, 0, sizeof *out);
  const int W = 480 * in->frames;
  const bool first = in->held_len <= 0;
  if (first) {
    out->flags |= KT2N_HIFT_FIRST;
  } else {
    out->overlap = std::min(std::min(kSrcCache, W), in->held_len);
    if (in->short_first && out->overlap < kSrcCache) out->flags |= KT2N_HIFT_SHORT_XFADE;
  }
  int emitted = W;
  bool tail_now = false;
  if (!in->finalize) {
    const bool short_ok = in->short_first && in->commit_index == 0 && in->first_commit_tail && !in->publish_held_tail &&
                          !in->preserve_next_commit && W > 2 * in->row_fade;
    if (W <= kSrcCache && !short_ok) out->error = 1;
    out->flags |= KT2N_HIFT_HOLD;
    out->tail_len = std::min(kSrcCache, W);
    out->new_mel_cache = std::min(8, in->frames);
    out->new_src_len = std::min(kSrcCache, W);
    tail_now = in->first_commit_tail && !in->publish_held_tail && !in->preserve_next_commit && in->commit_index == 0;
    if (in->publish_held_tail) emitted = W;
    else if (tail_now) emitted = W - in->row_fade;
    else emitted = std::max(0, W - kSrcCache);
  }
  out->emit_len = emitted;
  out->out_len = emitted;
  if (in->pending_tail > 0 && !first) {
    out->flags |= KT2N_HIFT_SPLICE;
    out->skip = in->pending_tail;
    if (emitted <= in->pending_tail) {
      out->out_len = 0;
      out->splice_count = 0;
    } else {
      out->splice_count = std::max(0, std::min(std::min(in->row_fade, emitted - in->pending_tail), in->held_len));
      out->splice_start = std::max(0, in->held_len - in->row_fade);
      out->out_len = emitted - in->pending_tail;
    }
  }
  out->new_pending_tail = tail_now ? ((in->short_first ? out->tail_len : kSrcCache) - in->row_fade) : 0;
}

void kt2n_hift_tables(float* trim960, float* ham7680) {
  if (trim960) {
    // s3gen.py: trim_fade = zeros(960); trim_fade[480:] = (cos(linspace(pi, 0, 480)) + 1) / 2 (fp32).
    const int n = 480;
    const float start = (float)M_PI, end = 0.f;
    const float step = (end - start) / (float)(n - 1);
    for (int i = 0; i < 960; ++i) trim960[i] = 0.f;
    for (int i = 0; i < n; ++i) {
      const float x = i < n / 2 ? start + step * (float)i : end - step * (float)(n - 1 - i);
      trim960[n + i] = (cosf(x) + 1.0f) / 2.0f;
    }
  }
  if (ham7680) {
    const int M = 2 * kSrcCache;
    for (int i = 0; i < M; ++i)
      ham7680[i] = (float)(0.54 + 0.46 * std::cos(M_PI * (double)(2 * i + 1 - M) / (double)(M - 1)));
  }
}

void* kt2n_hift_create(const void* const* ptrs, int n_ptrs, int max_b, int max_t) {
  return kt2n_hift_create2(ptrs, n_ptrs, max_b, max_t, max_b * max_t);
}

void* kt2n_hift_create2(const void* const* ptrs, int n_ptrs, int max_b, int max_t, int max_frame_rows) {
  if (max_b < 1 || max_t < 1 || max_frame_rows < max_t) {
    snprintf(g_err, sizeof g_err, "workspace shape max_b=%d max_t=%d must be >= 1", max_b, max_t);
    return nullptr;
  }
  if (ptrs == nullptr || n_ptrs != kHiftWeightPtrs) {
    snprintf(g_err, sizeof g_err, "expected %d weight pointers, got %d", kHiftWeightPtrs, n_ptrs);
    return nullptr;
  }
  Hift* h = new Hift();
  int i = 0;
  auto next = [&]() -> const float* {
    const int at = i++;
    return at < n_ptrs ? static_cast<const float*>(ptrs[at]) : nullptr;
  };
  const int f0_cin[5] = {80, 512, 512, 512, 512};
  for (int c = 0; c < 5; ++c) {
    h->f0c[c].w = next(); h->f0c[c].b = next();
    h->f0c[c].cin = f0_cin[c]; h->f0c[c].cout = 512; h->f0c[c].K = 3; h->f0c[c].dil = 1; h->f0c[c].pad = 1;
  }
  h->f0_cls_w = next(); h->f0_cls_b = next();
  h->src_w = next(); h->src_b = next();
  h->pre.w = next(); h->pre.b = next(); h->pre.cin = 80; h->pre.cout = 512; h->pre.K = 7; h->pre.dil = 1; h->pre.pad = 3;
  auto read_res = [&](ResBlockW& rb, int ch, int K) {
    for (int k = 0; k < 3; ++k) {
      rb.s1[k].a = next(); rb.s1[k].inv = next();
      rb.c1[k].w = next(); rb.c1[k].b = next();
      rb.c1[k].cin = ch; rb.c1[k].cout = ch; rb.c1[k].K = K; rb.c1[k].dil = kDil[k]; rb.c1[k].pad = kDil[k] * (K - 1) / 2;
      rb.s2[k].a = next(); rb.s2[k].inv = next();
      rb.c2[k].w = next(); rb.c2[k].b = next();
      rb.c2[k].cin = ch; rb.c2[k].cout = ch; rb.c2[k].K = K; rb.c2[k].dil = 1; rb.c2[k].pad = (K - 1) / 2;
    }
  };
  const int sd_k[3] = {30, 6, 1}, sd_s[3] = {15, 3, 1}, sd_p[3] = {7, 1, 0};
  for (int s = 0; s < 3; ++s) {
    StageW& w = h->st[s];
    w.up_w = next(); w.up_b = next();
    w.up_cin = s == 0 ? 512 : kChan[s - 1]; w.up_cout = kChan[s]; w.up_k = kUpK[s]; w.up_s = kUpRates[s];
    w.up_p = (kUpK[s] - kUpRates[s]) / 2;
    w.sd_w = next(); w.sd_b = next(); w.sd_cout = kChan[s]; w.sd_k = sd_k[s]; w.sd_s = sd_s[s]; w.sd_p = sd_p[s];
    read_res(w.sres, kChan[s], kSrcK[s]);
    for (int j = 0; j < 3; ++j) read_res(w.res[j], kChan[s], kResK[j]);
  }
  h->post.w = next(); h->post.b = next(); h->post.cin = 64; h->post.cout = 18; h->post.K = 7; h->post.dil = 1; h->post.pad = 3;
  h->post_w = const_cast<float*>(h->post.w);
  h->win = next();
  if (i != n_ptrs) {
    snprintf(g_err, sizeof g_err, "internal: the weight parse reads %d pointers, not %d", i, n_ptrs);
    delete h;
    return nullptr;
  }
  if (const char* e = std::getenv("KT2N_HIFT_CONVT_BIG")) g_convt_big = std::atoi(e);
  if (const char* e = std::getenv("KT2N_HIFT_FUSE_SNAKE")) g_fuse_snake = KT2N_HIFT_FUSE_SNAKE_BUILD ? std::atoi(e) : 0;
  if (const char* e = std::getenv("KT2N_HIFT_TARGET_BLOCKS")) kTargetBlocks = std::max(1, std::atoi(e));
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&h->sms, cudaDevAttrMultiProcessorCount, dev);
  h->dev_sms = h->sms;
  if (KT2N_HIFT_SMS_DEFAULT > 0) h->sms = std::min(h->sms, KT2N_HIFT_SMS_DEFAULT);
  if (const char* e = std::getenv("KT2N_HIFT_SMS")) h->sms = std::max(1, std::min(h->sms, std::atoi(e)));
  if (const char* e = std::getenv("KT2N_HIFT_TC_MIN")) h->tc_min = std::atoi(e);   // per-process default policy
  if (const char* e = std::getenv("KT2N_HIFT_GROUP")) h->group = std::atoi(e);      // v26 resblock grouping
  if (const char* e = std::getenv("KT2N_HIFT_TC_TARGET_MUL")) h->tc_target_mul = std::max(1, std::atoi(e));
  cudaDeviceGetAttribute(&h->max_threads_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
  float cs[9][16], sn[9][16];
  for (int m = 0; m < 9; ++m)
    for (int k = 0; k < 16; ++k) {
      const double a = 2.0 * M_PI * m * k / 16.0;
      cs[m][k] = (float)std::cos(a);
      sn[m][k] = (float)std::sin(a);
    }
  cudaMemcpyToSymbol(c_cos, cs, sizeof cs);
  cudaMemcpyToSymbol(c_sin, sn, sizeof sn);
  std::vector<float> trim(kTrim), ham(2 * kSrcCache);
  kt2n_hift_tables(trim.data(), ham.data());
  cudaMemcpyToSymbol(c_trim, trim.data(), kTrim * sizeof(float));
  h->max_b = max_b; h->max_t = max_t; h->max_fr = max_frame_rows;
  const size_t FR = max_frame_rows, B = max_b;
  const size_t BF = 120 * FR + B;          // sum over rows of F = 120 T + 1
  const size_t big = std::max<size_t>(64 * BF + 64 * B, 512 * FR);
  const size_t st_len[3] = {256 * 8 * FR, 128 * 40 * FR, 64 * BF};
  bool failed = false;
  auto alloc = [&](size_t n) {
    void* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess || p == nullptr) {
      failed = true;
      return (float*)nullptr;
    }
    h->allocs.push_back(p);
    return (float*)p;
  };
  h->f0a = alloc(512 * FR); h->f0b = alloc(512 * FR); h->f0 = alloc(FR);
  h->noise = alloc(9 * 480 * FR); h->sines = alloc(9 * 480 * FR); h->s = alloc(480 * FR); h->sst = alloc(18 * BF);
  h->xpre = alloc(512 * FR); h->xup = alloc(big);
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) h->chain[j][k] = alloc(big);
    h->chain_sx[j] = alloc(big);
  }
  for (int s = 0; s < 3; ++s) {
    for (int k = 0; k < 4; ++k) h->src_chain[s][k] = alloc(st_len[s]);
    h->src_sx[s] = alloc(st_len[s]);
  }
  h->ham = alloc(2 * kSrcCache);
  h->ctr_n = 1u << 20;
  h->ctr = reinterpret_cast<unsigned*>(alloc(h->ctr_n));
  for (int k = 0; k < kStreams; ++k) {
    h->sc[k].part_floats = std::max<size_t>(B * 300000, 4u << 20);  // SIMT: per-row splits; TC: <= ~376 x 64 x 128
    h->sc[k].part = alloc(h->sc[k].part_floats);
  }
  if (failed) {
    (void)cudaGetLastError();
    snprintf(g_err, sizeof g_err, "cudaMalloc failed (workspace for max_b=%d max_t=%d)", max_b, max_t);
    for (void* p : h->allocs) cudaFree(p);
    delete h;
    return nullptr;
  }
  {
    std::vector<Conv*> convs;
    for (int c = 0; c < 5; ++c) convs.push_back(&h->f0c[c]);
    convs.push_back(&h->pre);
    for (int s = 0; s < 3; ++s) {
      ResBlockW* rbs[4] = {&h->st[s].sres, &h->st[s].res[0], &h->st[s].res[1], &h->st[s].res[2]};
      for (ResBlockW* rb : rbs)
        for (int k = 0; k < 3; ++k) { convs.push_back(&rb->c1[k]); convs.push_back(&rb->c2[k]); }
    }
    for (Conv* c : convs) {
      const size_t n = (size_t)c->cin * c->K * c->cout;
      uint32_t* buf = reinterpret_cast<uint32_t*>(alloc(n));
      if (!buf) break;
      tc_arrange_k<<<(unsigned)std::min<size_t>((n + 255) / 256, 4096), 256>>>(c->w, c->cin, c->K, c->cout, buf);
      c->tc = buf;
    }
    for (int st = 0; st < 3 && !failed; ++st) {
      StageW& w = h->st[st];
      const int kmax = (w.up_k + w.up_s - 1) / w.up_s;
      const size_t per = (size_t)w.up_cin * kmax * w.up_cout;
      float* tmp = alloc(per);
      uint32_t* buf = reinterpret_cast<uint32_t*>(alloc(per * w.up_s));
      if (!tmp || !buf) break;
      for (int r = 0; r < w.up_s; ++r) {
        convT_phase_w_k<<<(unsigned)std::min<size_t>((per + 255) / 256, 4096), 256>>>(w.up_w, w.up_cin, w.up_cout, w.up_k,
                                                                                       w.up_s, r, kmax, tmp);
        tc_arrange_k<<<(unsigned)std::min<size_t>((per + 255) / 256, 4096), 256>>>(tmp, w.up_cin, kmax, w.up_cout,
                                                                                   buf + r * per);
      }
      w.up_tc = buf; w.up_kmax = kmax; w.up_phase_words = per;
    }
    if (failed) {
      (void)cudaGetLastError();
      snprintf(g_err, sizeof g_err, "cudaMalloc failed (tensor-core weights)");
      for (void* p : h->allocs) cudaFree(p);
      delete h;
      return nullptr;
    }
  }
  cudaMemset(h->ctr, 0, h->ctr_n * sizeof(unsigned));
  cudaMemcpy(h->ham, ham.data(), ham.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceGetStreamPriorityRange(&h->prio_least, &h->prio_greatest);
  h->n_prio = std::min(Hift::kMaxPrio, std::max(1, h->prio_least - h->prio_greatest + 1));
  for (int l = 0; l < h->n_prio; ++l)
    for (int k = 0; k < kStreams - 1; ++k)
      cudaStreamCreateWithPriority(&h->side[l][k], cudaStreamNonBlocking, h->prio_greatest + l);
  for (int k = 1; k < kStreams; ++k) h->sc[k].s = h->side[h->n_prio - 1][k - 1];
  cudaEventCreateWithFlags(&h->ev_fork, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&h->ev_noise, cudaEventDisableTiming);
  for (int s = 0; s < 3; ++s) {
    cudaEventCreateWithFlags(&h->ev_src[s], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&h->ev_x[s], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&h->ev_y[s][0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&h->ev_y[s][1], cudaEventDisableTiming);
  }
  const int max_smem = (BG_CI * MM_XW + BG_CI * (WK_MAX * BG_CO + 8)) * sizeof(float);
  cudaFuncSetAttribute(conv1d_mma_k<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
  cudaFuncSetAttribute(conv1d_mma_k<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
#define KT2N_TC_ATTR(BNV, KV) cudaFuncSetAttribute(conv1d_tc_k<BNV, KV, (KV == 11 ? KT2N_TC_TG11 : KV == 7 ? KT2N_TC_TG7 : KV)>, \
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                  (int)tc_smem_bytes(KV, 5, BNV))
  KT2N_TC_ATTR(64, 2); KT2N_TC_ATTR(64, 3); KT2N_TC_ATTR(64, 7); KT2N_TC_ATTR(64, 11);
  KT2N_TC_ATTR(128, 2); KT2N_TC_ATTR(128, 3); KT2N_TC_ATTR(128, 7); KT2N_TC_ATTR(128, 11);
#if KT2N_TC_WIDE_MINK > 0
  KT2N_TC_ATTR(256, 3); KT2N_TC_ATTR(256, 7); KT2N_TC_ATTR(256, 11);
#endif
#undef KT2N_TC_ATTR
  {
    // Load every kernel now (CUDA's default lazy loading would do it at the first live launch of each:
    // L7 measured a first commit-0 launch 6.3 ms -> 4.7 ms after warming, 1.1 ms with eager loading).
    cudaFuncAttributes fa;
    const void* fns[] = {(const void*)conv1d_k, (const void*)convT_k, (const void*)conv_strided1_k,
                         (const void*)conv_strided_k, (const void*)f0_head_k, (const void*)sine_rows_k,
                         (const void*)source_stft_k, (const void*)noise_k, (const void*)pack_mel_k,
                         (const void*)tail_k<128>, (const void*)tail_k<256>};
    for (const void* f : fns) cudaFuncGetAttributes(&fa, f);
  }
  cudaFuncSetAttribute(conv_strided_sm_k<15, 30>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)sd_sm_bytes<15, 30>(18));
  cudaFuncSetAttribute(conv_strided_sm_k<3, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sd_sm_bytes<3, 6>(18));
  cudaFuncSetAttribute(conv_strided_sm_k<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sd_sm_bytes<1, 1>(18));
  cudaFuncSetAttribute(tail_k<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TailGeo<256>::smem);
  cudaFuncSetAttribute(tail_k<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TailGeo<128>::smem);
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    snprintf(g_err, sizeof g_err, "create: %s", cudaGetErrorString(err));
    for (void* p : h->allocs) cudaFree(p);
    delete h;
    return nullptr;
  }
  return h;
}

void kt2n_hift_destroy(void* handle) {
  Hift* h = static_cast<Hift*>(handle);
  if (!h) return;
  cudaDeviceSynchronize();
  for (int l = 0; l < Hift::kMaxPrio; ++l)
    for (int k = 0; k < kStreams - 1; ++k)
      if (h->side[l][k]) cudaStreamDestroy(h->side[l][k]);
  cudaEventDestroy(h->ev_fork);
  cudaEventDestroy(h->ev_noise);
  for (int s = 0; s < 3; ++s) {
    cudaEventDestroy(h->ev_src[s]);
    cudaEventDestroy(h->ev_x[s]);
    cudaEventDestroy(h->ev_y[s][0]);
    cudaEventDestroy(h->ev_y[s][1]);
  }
  for (void* p : h->allocs) cudaFree(p);
  delete h;
}

int kt2n_hift_run(void* handle, int B, int T, int cache_len, const float* mel, const float* cache,
                  const float* phase, const float* noise, float* wave, float* source, void* stream_ptr) {
  Hift& h = *static_cast<Hift*>(handle);
  RunIO io{};
  io.B = B; io.T = T; io.mel = mel; io.phase_ext = phase; io.noise_ext = noise;
  io.src = SrcArgs{nullptr, cache, cache ? cache_len : 0};
  io.wave = wave; io.source = source; io.rows = nullptr;
  return run_impl(h, io, static_cast<cudaStream_t>(stream_ptr));
}

int kt2n_hift_run_rows(void* handle, int B, int T, const float* mel, const Kt2nHiftRow* rows, float* wave,
                       float* source, void* stream_ptr) {
  Hift& h = *static_cast<Hift*>(handle);
  RunIO io{};
  io.B = B; io.T = T; io.mel = mel; io.phase_ext = nullptr; io.noise_ext = nullptr;
  io.src = SrcArgs{rows, nullptr, 0};
  io.wave = wave; io.source = source; io.rows = rows;
  return run_impl(h, io, static_cast<cudaStream_t>(stream_ptr));
}

int kt2n_hift_pack_mel(int B, int T, const float* frames_fm, const int32_t* frame_off, const float* const* mel_cache,
                       const int32_t* cache_frames, float* out, void* stream_ptr) {
  pack_mel_k<<<dim3((80 * T + 255) / 256, B), 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
      T, frames_fm, frame_off, mel_cache, cache_frames, out);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int kt2n_hift_noise(void* handle, int B, int L, const uint64_t* seeds_host, float* out, void* stream_ptr) {
  Hift& h = *static_cast<Hift*>(handle);
  cudaStream_t st = static_cast<cudaStream_t>(stream_ptr);
  uint64_t* d = nullptr;
  if (cudaMalloc(&d, B * sizeof(uint64_t)) != cudaSuccess) return 4;
  cudaMemcpy(d, seeds_host, B * sizeof(uint64_t), cudaMemcpyHostToDevice);
  const int numel = 9 * L;
  const int bps = h.max_threads_sm / 256;
  const int gx = std::min<long>((long)h.dev_sms * bps, (numel + 255) / 256);
  NoiseArgs na{nullptr, d, nullptr, 0};
  launch(noise_k, dim3(gx, B), dim3(256), 0, st, na, numel, out);
  cudaStreamSynchronize(st);
  cudaFree(d);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

}  // extern "C"
