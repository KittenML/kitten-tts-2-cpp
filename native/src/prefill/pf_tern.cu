// prof-prefill variant (used through KT2N_PF_FLAG_PROJ_TABLE): a register-lean, multi-stage ternary GEMM for the prefill
// projections (M > 16) with EXACTLY k1's per-row arithmetic (src/cuda/kernels/tern_proj.cu, tern_gemm_kernel):
//
//   * same K permutation and fragments: lane (g, t), MMA step j of a 256-chunk feeds physical
//     k = 256c + 64t + 4j + {0,1} as logical (2t, 2t+1) and + {2,3} as (2t+8, 2t+9), for A and B alike;
//   * same chains: K split w = CPW consecutive chunks; accumulator a of split w is the MMA chain over the steps
//     j = a (mod ACC) of the split's chunks in (chunk, j) order, starting from +0;
//   * same folds: p_w = ((acc0 + acc1) + acc2) + ...; out = ((p0 + p1) + p2) + ... in fp32, then k1's epilogue.
//
// What differs is only WHEN each chain runs: k1 keeps all ACC chains of a split live at once (ACC x 4 fp32
// registers per 16x8 output tile, plus the running total); this kernel runs the chains of a split P at a time
// (P = 1 or 2 chains in flight) and folds each into p_w as soon as it ends, so a 16x8 tile costs (P + 2) x 4
// registers instead of (ACC + 1) x 4.  The freed registers buy larger warp tiles (fewer shared-memory reads of
// activations per MMA, fewer weight decodes per MMA) and a deeper cp.async pipeline (ST stages of one K split).
// Each output element still sees the same operands in the same order: bitwise identical to kt2n_tern_proj.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "kt2n/kernels/tern_proj.h"
#include "kt2n/cuda_caps.h"
#include "pf_tern.h"
#include "tern_proj.cuh"

using bf16 = __nv_bfloat16;
using namespace kt2n_tern;

namespace {

// k1's epilogue (tern_proj.cu store_tile), verbatim.
template <int EPI>
__device__ __forceinline__ void store_tile(const float (&v)[4], int tile, int row0, int M, int g, int t, bf16* Y,
                                           int ldy, const bf16* R, int ldr) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = row0 + h * 8 + g;
    if (row >= M) continue;
    const float c0 = v[2 * h], c1 = v[2 * h + 1];
    if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL) {
      Y[(size_t)row * ldy + tile * 4 + t] = silu_mul_bf16(c0, c1);
    } else {
      const int col = tile * 8 + 2 * t;
      __nv_bfloat162 out;
      if constexpr (EPI == KT2N_TERN_EPI_RESIDUAL)
        out = residual_bf16x2(c0, c1, ld_act_bf16x2(R + (size_t)row * ldr + col));
      else
        out = __floats2bfloat162_rn(c0, c1);
      *reinterpret_cast<__nv_bfloat162*>(Y + (size_t)row * ldy + col) = out;
    }
  }
}

__device__ __forceinline__ int xunit(int u, int r) { return u ^ ((2 * (u >> 3) + (r & 1)) & 7); }

__device__ __forceinline__ uint2 lds64(const void* p) { return *reinterpret_cast<const uint2*>(p); }

template <int BM, int TB, int CPW>
struct LeanSmem {
  static constexpr int kX = BM * 512;  // one chunk of activations
  static constexpr int kW = TB * 512;  // one chunk of codes
  static constexpr int kS = TB * 32;   // one chunk of scales
  static constexpr int kStage = CPW * (kX + kW + kS);
};

// Block = WM x WN warps; warp tile = MT 16-row tiles x NT 8-col tiles; block tile BM = 16*MT*WM rows x TB = NT*WN
// column tiles.  One pipeline stage = one K split (CPW chunks).  P chains in flight (1, or 2 = adjacent steps
// j, j+1 sharing one 16-B activation read).
template <int C, int CPW, int ACC, int WM, int WN, int MT, int NT, int ST, int P, int EPI>
__global__ void __launch_bounds__(WM * WN * 32)
    lean_gemm_kernel(const bf16* X, int ldx, int M, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St,
                     bf16* Y, int ldy, const bf16* R, int ldr) {
  static_assert(16 % ACC == 0 && ACC % P == 0 && (P == 1 || P == 2), "chains");
  static_assert(C % CPW == 0 && ST >= 2, "pipeline");
  constexpr int NTHR = WM * WN * 32;
  constexpr int BM = 16 * MT * WM;
  constexpr int TB = NT * WN;
  constexpr int WKS = C / CPW;
  using L = LeanSmem<BM, TB, CPW>;
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wm = wid / WN, wn = wid % WN;
  const int g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.y * BM;
  const int tileb = blockIdx.x * TB;

  auto load_w = [&](int s, int stage) {
    unsigned char* base = smem + stage * L::kStage + CPW * L::kX;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
      const int ch = s * CPW + i;
      unsigned char* sw = base + i * L::kW;
      for (int k = tid; k < TB * 32; k += NTHR) {
        const int tl = k >> 5, ln = k & 31;
        cp_async16(sw + k * 16, Wt + ((size_t)(tileb + tl) * C + ch) * 32 + ln, 16);
      }
      unsigned char* ss = base + CPW * L::kW + i * L::kS;
      for (int k = tid; k < TB * 2; k += NTHR) {
        const int tl = k >> 1, h = k & 1;
        cp_async16(ss + k * 16, St + ((size_t)(tileb + tl) * C + ch) * 16 + h * 8, 16);
      }
    }
  };
  auto load_x = [&](int s, int stage) {
    unsigned char* base = smem + stage * L::kStage;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
      const int ch = s * CPW + i;
      unsigned char* sx = base + i * L::kX;
      for (int k = tid; k < BM * 32; k += NTHR) {
        const int r = k >> 5, u = k & 31;
        const int row = m0 + r;
        const bool ok = row < M;
        cp_async16(sx + r * 512 + xunit(u, r) * 16, X + (size_t)(ok ? row : 0) * ldx + ch * kChunk + u * 8,
                   ok ? 16 : 0);
      }
    }
  };

  // Prologue: the constant weights of the first ST-1 splits before griddepcontrol.wait, activations after.
#pragma unroll
  for (int s = 0; s < ST - 1; ++s)
    if (s < WKS) load_w(s, s);
  pdl_wait();
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < WKS) load_x(s, s);
    cp_async_commit();
  }

  float tot[MT][NT][4];
#pragma unroll 1
  for (int s = 0; s < WKS; ++s) {
    cp_async_wait<ST - 2>();
    __syncthreads();  // split s visible to all; every warp is done with split s - 1's stage
    {
      const int ns = s + ST - 1;
      if (ns < WKS) {
        load_w(ns, ns % ST);
        load_x(ns, ns % ST);
      }
      cp_async_commit();
    }
    const unsigned char* sx = smem + (s % ST) * L::kStage;
    const unsigned char* sw = sx + CPW * L::kX;
    const unsigned char* ss = sw + CPW * L::kW;
    float v[MT][NT][4];
#pragma unroll
    for (int a = 0; a < ACC; a += P) {
      float acc[P][MT][NT][4];
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) acc[p][mt][nt][0] = acc[p][mt][nt][1] = acc[p][mt][nt][2] = acc[p][mt][nt][3] = 0.f;
#pragma unroll
      for (int i = 0; i < CPW; ++i) {
        uint4 w[NT];
        uint32_t ta[NT], tb[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int tl = wn * NT + nt;
          w[nt] = lds128(sw + i * L::kW + (tl * 32 + lane) * 16);
          const uint32_t sc = *reinterpret_cast<const uint16_t*>(ss + i * L::kS + tl * 32 + (g * 2 + (t >> 1)) * 2);
          scale_tables(sc, ta[nt], tb[nt]);
        }
        const unsigned char* sxi = sx + i * L::kX;
#pragma unroll
        for (int j = a; j < 16; j += ACC) {
          // activations of steps j .. j + P - 1 (P = 2: j even, one 16-B unit holds both steps)
          uint32_t A[P][MT][4];
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            const int r0 = (wm * MT + mt) * 16 + g, r1 = r0 + 8;
            const int u = 8 * t + (j >> 1);
            if constexpr (P == 2) {
              const uint4 xa = lds128(sxi + r0 * 512 + xunit(u, r0) * 16);
              const uint4 xb = lds128(sxi + r1 * 512 + xunit(u, r1) * 16);
              A[0][mt][0] = xa.x; A[0][mt][2] = xa.y; A[0][mt][1] = xb.x; A[0][mt][3] = xb.y;
              A[P - 1][mt][0] = xa.z; A[P - 1][mt][2] = xa.w; A[P - 1][mt][1] = xb.z; A[P - 1][mt][3] = xb.w;
            } else {
              const int h = (j & 1) * 8;
              const uint2 xa = lds64(sxi + r0 * 512 + xunit(u, r0) * 16 + h);
              const uint2 xb = lds64(sxi + r1 * 512 + xunit(u, r1) * 16 + h);
              A[0][mt][0] = xa.x; A[0][mt][2] = xa.y; A[0][mt][1] = xb.x; A[0][mt][3] = xb.y;
            }
          }
#pragma unroll
          for (int p = 0; p < P; ++p) {
            const int jj = j + p;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
              uint32_t b0, b1;
              decode_step(word_of(w[nt], jj >> 2), jj & 3, ta[nt], tb[nt], b0, b1);
#pragma unroll
              for (int mt = 0; mt < MT; ++mt)
                mma_16816(acc[p][mt][nt], A[p][mt][0], A[p][mt][1], A[p][mt][2], A[p][mt][3], b0, b1);
            }
          }
        }
      }
      // fold the finished chains into this split's partial: ((acc0 + acc1) + acc2) + ...
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              if (a + p == 0)
                v[mt][nt][q] = acc[p][mt][nt][q];
              else
                v[mt][nt][q] += acc[p][mt][nt][q];
            }
    }
    // and the split's partial into the total: ((p0 + p1) + p2) + ...
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (s == 0)
            tot[mt][nt][q] = v[mt][nt][q];
          else
            tot[mt][nt][q] += v[mt][nt][q];
        }
  }
  cp_async_wait<0>();
  pdl_launch_dependents();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
      store_tile<EPI>(tot[mt][nt], tileb + wn * NT + nt, m0 + (wm * MT + mt) * 16, M, g, t, Y, ldy, R, ldr);
}

template <typename Kern, typename... Args>
cudaError_t launch(Kern kern, dim3 grid, dim3 block, size_t smem, bool pdl, cudaStream_t stream, Args... args) {
  pdl = kt2n::cuda::pdl_enabled(pdl);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kern, args...);
}

template <int C, int CPW, int ACC, int WM, int WN, int MT, int NT, int ST, int P, int EPI>
cudaError_t go(const kt2n_pf_tern_args& a) {
  constexpr int BM = 16 * MT * WM, TB = NT * WN;
  using L = LeanSmem<BM, TB, CPW>;
  constexpr int bytes = ST * L::kStage;
  if constexpr (bytes > 99 * 1024 || WM * WN * 32 > 1024) {
    return cudaErrorInvalidConfiguration;
  } else {
    auto kern = lean_gemm_kernel<C, CPW, ACC, WM, WN, MT, NT, ST, P, EPI>;
    static int attr_set[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if (!attr_set[dev]) {
      const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
      if (e != cudaSuccess) return e;
      attr_set[dev] = 1;
    }
    if (a.n % (8 * TB)) return cudaErrorInvalidConfiguration;
    const int mblocks = (a.m + BM - 1) / BM;
    if (mblocks > 65535) return cudaErrorInvalidConfiguration;
    return launch(kern, dim3(a.n / (8 * TB), mblocks), dim3(WM * WN * 32), (size_t)bytes, a.pdl, a.stream,
                  static_cast<const bf16*>(a.x), (int)a.ldx, a.m, static_cast<const uint4*>(a.codes),
                  static_cast<const uint16_t*>(a.scales), static_cast<bf16*>(a.y), (int)a.ldy,
                  static_cast<const bf16*>(a.residual), (int)a.ldr);
  }
}

// Shape code: WM, WN, MT, NT, ST, P packed as decimal digits (see pf_tern.h).
template <int C, int CPW, int ACC, int EPI>
cudaError_t shape(int code, const kt2n_pf_tern_args& a) {
#define KT2N_LEAN(WM_, WN_, MT_, NT_, ST_, P_)                                    \
  if (code == WM_ * 100000 + WN_ * 10000 + MT_ * 1000 + NT_ * 100 + ST_ * 10 + P_) \
    return go<C, CPW, ACC, WM_, WN_, MT_, NT_, ST_, P_, EPI>(a);
  // small M (17..64): one M block, many column blocks
  KT2N_LEAN(1, 4, 2, 2, 3, 2) KT2N_LEAN(1, 4, 2, 2, 4, 2) KT2N_LEAN(2, 2, 2, 2, 3, 2) KT2N_LEAN(1, 4, 4, 2, 3, 1)
  KT2N_LEAN(1, 4, 4, 2, 3, 2) KT2N_LEAN(2, 2, 2, 4, 3, 1) KT2N_LEAN(1, 4, 2, 4, 3, 1) KT2N_LEAN(1, 4, 2, 4, 3, 2)
  // large M
  KT2N_LEAN(2, 4, 2, 4, 3, 1) KT2N_LEAN(2, 4, 2, 4, 3, 2) KT2N_LEAN(2, 2, 2, 8, 3, 1) KT2N_LEAN(2, 2, 4, 4, 2, 1)
  KT2N_LEAN(1, 8, 4, 2, 3, 1) KT2N_LEAN(2, 4, 4, 2, 3, 1) KT2N_LEAN(4, 2, 2, 4, 3, 1) KT2N_LEAN(1, 4, 2, 8, 3, 1)
  KT2N_LEAN(2, 4, 2, 4, 4, 1) KT2N_LEAN(1, 8, 2, 4, 3, 1) KT2N_LEAN(2, 2, 2, 4, 4, 1) KT2N_LEAN(1, 4, 4, 4, 3, 1)
  // two-chunk splits (gate_up, down: CPW 2) fit shared memory with 16-row blocks or 2 stages
  KT2N_LEAN(1, 4, 1, 2, 3, 1) KT2N_LEAN(1, 4, 1, 2, 3, 2) KT2N_LEAN(1, 4, 2, 2, 2, 1) KT2N_LEAN(1, 4, 2, 2, 2, 2)
  KT2N_LEAN(1, 4, 2, 4, 2, 1) KT2N_LEAN(1, 4, 2, 4, 2, 2) KT2N_LEAN(1, 8, 1, 2, 3, 1) KT2N_LEAN(1, 8, 1, 2, 3, 2)
#undef KT2N_LEAN
  return cudaErrorInvalidConfiguration;
}

}  // namespace

extern "C" int kt2n_pf_tern_proj(const kt2n_pf_tern_args* a, int shape_code) {
  if (!a || a->family < 0 || a->family > 3 || a->m < 1 || !a->x || !a->codes || !a->scales || !a->y)
    return KT2N_TERN_EINVAL;
  if ((reinterpret_cast<uintptr_t>(a->x) | reinterpret_cast<uintptr_t>(a->codes) |
       reinterpret_cast<uintptr_t>(a->scales)) % 16 || a->ldx % 8)
    return KT2N_TERN_EINVAL;
  if (a->epi == KT2N_TERN_EPI_RESIDUAL && !a->residual) return KT2N_TERN_EINVAL;
  {  // the numerics constants below must be k1's (a build with other KT2N_TERN_*_WK/_ACC would differ)
    static const int want_wk[4] = {8, 8, 4, 12}, want_acc[4] = {4, 4, 4, 2};
    int wk = 0;
    if (kt2n_tern_family_split(a->family, &wk, nullptr) != KT2N_TERN_OK || wk != want_wk[a->family] ||
        kt2n_tern_family_acc(a->family) != want_acc[a->family])
      return KT2N_TERN_EINVAL;
  }
  cudaError_t e = cudaErrorInvalidConfiguration;
  // Family constants = k1's frozen table (tern_proj.cu family_cfg): (chunks, splits, accumulators).
  switch (a->family) {
    case KT2N_TERN_FAM_QKV:  // K 2048 = 8 chunks, WK 8 -> CPW 1, ACC 4
      if (a->epi == KT2N_TERN_EPI_STORE) e = shape<8, 1, 4, KT2N_TERN_EPI_STORE>(shape_code, *a);
      break;
    case KT2N_TERN_FAM_O:  // WK 8 -> CPW 1, ACC 4
      if (a->epi == KT2N_TERN_EPI_RESIDUAL) e = shape<8, 1, 4, KT2N_TERN_EPI_RESIDUAL>(shape_code, *a);
      break;
    case KT2N_TERN_FAM_GATE_UP:  // WK 4 -> CPW 2, ACC 4
      if (a->epi == KT2N_TERN_EPI_SILU_MUL) e = shape<8, 2, 4, KT2N_TERN_EPI_SILU_MUL>(shape_code, *a);
      break;
    default:  // down: K 6144 = 24 chunks, WK 12 -> CPW 2, ACC 2
      if (a->epi == KT2N_TERN_EPI_RESIDUAL) e = shape<24, 2, 2, KT2N_TERN_EPI_RESIDUAL>(shape_code, *a);
      break;
  }
  if (e == cudaErrorInvalidConfiguration) return KT2N_TERN_EINVAL;
  return e == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

// Prefill projection shape table (KT2N_PF_FLAG_PROJ_TABLE): per (family, M range) either a lean-GEMM shape code or an
// explicit k1 launch shape for kt2n_tern_proj_ex (same kernels as k1's automatic choice, another launch shape).
// Every entry is bitwise identical to kt2n_tern_proj (projection sweeps checked 3 layers per point, and
// the prefill suites run with the flag forced).  Ranges are inclusive; M outside every range keeps k1's choice.
// Measured on the RTX PRO 6000 (28-layer sequences, warm and L2-flushed).
namespace {
struct ShapeEntry {
  int family, m_lo, m_hi;
  int lean;        // lean-GEMM shape code, or 0
  int nt, wn, mt;  // explicit k1 shape when lean == 0
};
// From sweeps of every M in 17..200 and a 16-step grid in 208..1024 (28-layer sequences,
// warm and L2-flushed, entry kept only where BOTH are >= 1.05x k1's choice) + random-M checks inside the burst
// ranges.  Comments: the measured speed-up range of the projection over the entry's M (warm).
const ShapeEntry kShapeTable[] = {
    {KT2N_TERN_FAM_QKV, 45, 48, 142221, 0, 0, 0},       // 1.07-1.32x
    {KT2N_TERN_FAM_QKV, 65, 80, 141231, 0, 0, 0},       // 1.22-1.23x
    {KT2N_TERN_FAM_QKV, 161, 176, 141231, 0, 0, 0},     // 1.07x
    {KT2N_TERN_FAM_O, 81, 87, 141232, 0, 0, 0},         // 1.08-1.10x
    {KT2N_TERN_FAM_O, 88, 112, 181231, 0, 0, 0},        // 1.05-1.23x
    {KT2N_TERN_FAM_O, 129, 149, 181231, 0, 0, 0},       // 1.20-1.21x
    {KT2N_TERN_FAM_O, 161, 176, 181231, 0, 0, 0},       // 1.26-1.27x
    {KT2N_TERN_FAM_O, 489, 744, 182431, 0, 0, 0},       // 1.08-1.17x
    {KT2N_TERN_FAM_O, 873, 1024, 142421, 0, 0, 0},      // 1.05-1.13x
    {KT2N_TERN_FAM_GATE_UP, 28, 32, 0, 2, 4, -101},     // k1 GEMM 1x4 warps, 2 col tiles: 1.06-1.16x
    {KT2N_TERN_FAM_GATE_UP, 33, 48, 0, 4, 4, -1},       // k1 GEMM 16-row tiles: 1.18-1.49x
    {KT2N_TERN_FAM_GATE_UP, 97, 112, 0, 4, 4, -1},      // k1 GEMM 16-row tiles: 1.16-1.17x
    {KT2N_TERN_FAM_DOWN, 45, 80, 141232, 0, 0, 0},      // 1.11-1.50x
    {KT2N_TERN_FAM_DOWN, 81, 160, 142221, 0, 0, 0},     // 1.08-1.33x
    {KT2N_TERN_FAM_DOWN, 161, 176, 181231, 0, 0, 0},    // 1.38-1.39x
    {KT2N_TERN_FAM_DOWN, 265, 360, 142422, 0, 0, 0},    // 1.36-1.41x
    {KT2N_TERN_FAM_DOWN, 713, 1024, 142422, 0, 0, 0},   // 1.28-1.82x
};
}  // namespace

extern "C" int kt2n_pf_proj_choice(int family, int m, int* lean, int* nt, int* wn, int* mt) {
  for (const ShapeEntry& e : kShapeTable) {
    if (e.family != family || m < e.m_lo || m > e.m_hi) continue;
    *lean = e.lean;
    *nt = e.nt;
    *wn = e.wn;
    *mt = e.mt;
    return 1;
  }
  return 0;
}
