// Native MeanFlow estimator kernels (L5). See include/kt2n/render/estimator.h for the contract.
//
// Every GEMM-shaped op runs on tensor cores (mma.sync.m16n8k16 bf16 -> fp32) with fragments from
// ldmatrix; weights stream through shared memory with cp.async double buffering. Fused epilogues
// follow production's module boundaries and round to bf16 where production stores a bf16 tensor.
// All kernels are batch invariant (per-frame math; attention splits at fixed 256-key boundaries from
// key 0) and graph capturable (shapes in device memory). PDL: every kernel waits (griddepcontrol.wait)
// before its first read of a predecessor's output and then triggers its dependents ("wait -> trigger ->
// work", r15). So when a kernel launches, every kernel two or more places earlier has completed, and the
// code before the wait may read their outputs (est_prep's row metadata) and the weights: weight prefetch,
// bias / LN affine and row metadata are all read before the wait (r12).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cooperative_groups.h>

#include "kt2n/render/ops.h"

namespace kt2n::render::est {

using bf16 = __nv_bfloat16;

inline constexpr int kHid = 256;       // estimator channels
inline constexpr int kInner = 512;     // heads * head_dim
inline constexpr int kFfn = 1024;
inline constexpr int kIn = 320;
inline constexpr int kMelC = 80;
inline constexpr int kChunkFr = 50;
inline constexpr int kAttnSplit = 128;  // keys per attention split (fixed: batch invariant)
inline constexpr int kAlignFr = 16;     // RowDesc::fr_off must be a multiple of this (conv tiles never span rows)
inline constexpr int kNumResnets = 14;
inline constexpr int kNumConvs = 31;
inline constexpr int kNumAttn = 56;

// Compact per-row view written by the prep kernel from RowDesc.
struct EstRow {
  int32_t fr_off;    // first packed frame
  int32_t past;      // frames whose K/V are final in the pool
  int32_t suffix;    // frames computed this call
  int32_t valid;     // frames solved (suffix - 2 * lookahead); masked past it
  int32_t page_off;  // first page-table entry
  int32_t slot;      // StateView slot (-1 = padding row)
  int32_t shift;     // frames the stable end advances (tail save at shift - 2, shift - 1); 0 = no save
  int32_t full;      // kRowFullAttention
};

struct AttnItem {
  int32_t row;
  int32_t q0;  // first local query frame of this 64-query block
};

// ---------------------------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
  const int bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// torch mish (opmath fp32): x * tanh(softplus(x)), softplus = log1p(exp(x)).
__device__ __forceinline__ float mish_f(float x) { return x * tanhf(log1pf(expf(x))); }
// torch gelu (approximate='none'): 0.5 x (1 + erf(x / sqrt 2)).
__device__ __forceinline__ float gelu_f(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)); }

// ---------------------------------------------------------------------------------------------
// Optional in-kernel phase trace (build with -DKT2N_EST_TRACE; off by default, no code otherwise).
// Thread 0 of every CTA stamps %globaltimer at: entry, after the dependency wait, when the first
// operand tile is ready (after the LN prologue for LN GEMMs), after the main loop, and at exit.
// Records are appended in exit order; a kernel's CTAs all exit before its successor's (they wait).
// ---------------------------------------------------------------------------------------------
struct EstTraceRec {
  uint32_t kid;  // 10 + GemmMode, 20 + conv MODE, 30 attention v2, 1 prep, 2 assemble, 3 tail commit
  uint32_t cta;
  uint64_t t[5];
};
#if defined(KT2N_EST_TRACE)
__device__ EstTraceRec* g_est_trace = nullptr;
__device__ unsigned g_est_trace_n = 0;
__device__ unsigned g_est_trace_cap = 0;
__device__ __forceinline__ uint64_t est_gt() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define EST_TR_BEGIN                    \
  uint64_t est_tr_[4];                  \
  est_tr_[0] = est_gt();                \
  est_tr_[1] = est_tr_[2] = est_tr_[3] = est_tr_[0];
#define EST_TR(i) (est_tr_[i] = est_gt())
#define EST_TR_END(KID)                                                                                 \
  do {                                                                                                  \
    if (threadIdx.x == 0 && g_est_trace != nullptr) {                                                  \
      const uint64_t t4_ = est_gt();                                                                    \
      const unsigned i_ = atomicAdd(&g_est_trace_n, 1u);                                                \
      if (i_ < g_est_trace_cap) {                                                                       \
        EstTraceRec r_;                                                                                 \
        r_.kid = (KID);                                                                                 \
        r_.cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);                        \
        r_.t[0] = est_tr_[0]; r_.t[1] = est_tr_[1]; r_.t[2] = est_tr_[2]; r_.t[3] = est_tr_[3];         \
        r_.t[4] = t4_;                                                                                  \
        g_est_trace[i_] = r_;                                                                           \
      }                                                                                                 \
    }                                                                                                   \
  } while (0)
#else
#define EST_TR_BEGIN
#define EST_TR(i) ((void)0)
#define EST_TR_END(KID) ((void)0)
#endif

// ---------------------------------------------------------------------------------------------
// Prep: RowDesc -> EstRow, frame -> row map, attention work items.
// ---------------------------------------------------------------------------------------------
__global__ void est_prep_kernel(const RowDesc* __restrict__ rows, int rows_cap, int frames_cap,
                                EstRow* __restrict__ erows, int32_t* __restrict__ frame_row,
                                AttnItem* __restrict__ items, int32_t* __restrict__ n_items, int items_cap) {
  // Wait, then trigger (prep and assemble): a kernel can then launch only after the kernel two places before
  // it has completed, so the later kernels may read est_prep's outputs (row metadata) before their own
  // dependency wait (r12 hoists). With trigger-first here, the first conv could launch while est_prep was
  // still writing them.
  pdl_wait();
  pdl_trigger();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // frame -> row (16-aligned spans)
  for (int f = tid; f < frames_cap; f += gridDim.x * blockDim.x) {
    int r_found = -1;
    for (int r = 0; r < rows_cap; ++r) {
      const RowDesc& d = rows[r];
      if (d.slot < 0 || (d.flags & kRowCancelled)) continue;
      const int span = (d.fr_suffix + kAlignFr - 1) / kAlignFr * kAlignFr;
      if (f >= d.fr_off && f < d.fr_off + span) { r_found = r; break; }
    }
    frame_row[f] = r_found;
  }
  if (tid < rows_cap) {
    const RowDesc& d = rows[tid];
    EstRow e{};
    if (d.slot >= 0 && !(d.flags & kRowCancelled)) {
      e.fr_off = d.fr_off;
      e.past = d.fr_past;
      e.suffix = d.fr_suffix;
      const int la = (d.flags & kRowFinalize) ? 0 : d.lookahead;
      e.valid = max(0, d.fr_suffix - 2 * la);
      e.page_off = d.page_off;
      e.slot = d.slot;
      e.full = (d.flags & kRowFullAttention) ? 1 : 0;
      int shift = 0;
      if (!(d.flags & (kRowFinalize | kRowNoPoolWrite))) {
        const int end_abs = d.tok_keys - la;                       // stable end incl. this commit
        const int new_past = max(d.tok_past, (end_abs / kTokenChunk) * kTokenChunk);
        shift = 2 * (new_past - d.tok_past);
      }
      e.shift = shift;
    } else {
      e.slot = -1;
    }
    erows[tid] = e;
  }
  if (tid == 0) {
    int n = 0;
    for (int r = 0; r < rows_cap; ++r) {
      const RowDesc& d = rows[r];
      if (d.slot < 0 || (d.flags & kRowCancelled)) continue;
      for (int q0 = 0; q0 < d.fr_suffix && n < items_cap; q0 += 64) items[n++] = AttnItem{r, q0};
    }
    *n_items = n;
  }
}

// Assemble x | mu | spks | cond (bf16) for the suffix frames; zeros on padding frames.
__global__ void est_assemble_kernel(const EstRow* __restrict__ erows, const int32_t* __restrict__ frame_row,
                                    const float* __restrict__ x0, const float* __restrict__ mu,
                                    const float* __restrict__ cond, const float* __restrict__ spks,
                                    bf16* __restrict__ xin, int frames_cap) {
  pdl_wait();  // then trigger: see est_prep_kernel
  pdl_trigger();
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= frames_cap * kIn) return;
  const int f = idx / kIn, c = idx % kIn;
  const int r = frame_row[f];
  float v = 0.f;
  if (r >= 0) {
    const EstRow e = erows[r];
    const int t = f - e.fr_off;
    if (t < e.suffix) {
      if (c < 80) v = x0[size_t(f) * 80 + c];
      else if (c < 160) v = mu[size_t(f) * 80 + c - 80];
      else if (c < 240) v = spks[size_t(r) * 80 + c - 160];
      else v = cond[size_t(f) * 80 + c - 240];
    }
  }
  xin[idx] = __float2bfloat16_rn(v);
}

// ---------------------------------------------------------------------------------------------
// Fused causal conv (k = 3): a cluster of 4 CTAs covers 16 frames x 256 output channels, each CTA
// 64 channels (128 threads, 4 warps of 16 x 16), so every CTA streams a quarter of the weights; the
// LayerNorm over the 256 channels reduces its per-row partial sums across the cluster through
// distributed shared memory (DSMEM). The 16 frames always belong to one row (fr_off is 16-aligned),
// so the 2-frame left halo is the row's previous frames, its conv tail (fr_past > 0) or zeros. The
// input is masked (frames >= valid read 0), as production's conv(x * mask).
//   MODE 0 (block1):  out = bf16(mask * Mish(LN(conv + b))) + temb      (resnet block1 + mlp(t))
//   MODE 1 (block2):  out = bf16(mask * Mish(LN(conv + b))) + bf16(res_conv(x * mask) + b_res)
//   MODE 2 (plain):   out = bf16(conv + b)                              (down / up CausalConv1d)
//   MODE 3 (final):   out = bf16(mask * Mish(LN(conv + b)))             (final_block)
// in0/in1: the conv input (in1 = the up block's skip, channels C0..C0+C1); res_in0/res_in1 the
// resnet input (MODE 1; for block2 the conv input is block1's output).
// ---------------------------------------------------------------------------------------------
struct ConvArgs {
  const bf16* in0; int c0; int ld0;
  const bf16* in1; int c1; int ld1;     // concat (may be null)
  const bf16* w;                         // [256][3][Cin] (Cin = c0 + c1)
  const bf16* bias;                      // [256]
  const bf16* ln_g; const bf16* ln_b;    // [256]
  const bf16* temb;                      // [256] (MODE 0)
  const bf16* res_in0; int res_c0; int res_ld0;
  const bf16* res_in1; int res_c1; int res_ld1;
  const bf16* res_w;                     // [256][Cres]
  const bf16* res_b;
  bf16* out; int ld_out;
  const EstRow* erows; const int32_t* frame_row;
  const bf16* tail_state;                // est_tail + step * slots * tail_width + conv offset (index by slot)
  int tail_slot_stride;                  // elements per slot (tail_width)
  bf16* tail_stage;                      // stage + step * rows_cap * tail_width + conv offset (index by row)
  int stage_row_stride;
  int trig_early;                        // 1: PDL trigger right after the dependency wait (r15), 0: after the main loop
  bf16* portable_values;                 // [frames][256], borrowed from the unused FF workspace
  bf16* portable_residual;               // [frames][256], MODE 1 only
};

inline constexpr int kConvBM = 16;
inline constexpr int kConvBN = 64;       // per CTA; 4 CTAs per cluster cover 256
inline constexpr int kConvCluster = 4;
inline constexpr int kConvThreads = 128;
inline constexpr int kConvBK = 64;
inline constexpr int kConvStages = 4;
inline constexpr int kBPad = 8;  // smem row padding (elements) to dodge ldmatrix bank conflicts

// ST = weight pipeline stages (r19: 3 / 2 at large buckets so block2 / block1 fit 2 / 3 CTAs per SM; the
// k order and the math are the same for every ST).
__host__ __device__ constexpr size_t conv_smem_bytes(int cin, int cres, int st = kConvStages) {
  return size_t(kConvBM + 2) * (cin + kBPad) * 2 + size_t(kConvBM) * (cres + kBPad) * 2 +
         size_t(st) * kConvBN * (kConvBK + kBPad) * 2 + 4 * 16 * 4 + 2 * 16 * 4;
}

template <int MODE, int ST = kConvStages, bool PORTABLE = false>
__global__ void __launch_bounds__(kConvThreads) est_conv_kernel(ConvArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  namespace cg = cooperative_groups;
  const int cin = a.c0 + a.c1;
  const int cres = (MODE == 1) ? (a.res_c0 + a.res_c1) : 0;
  const int lda_s = cin + kBPad;
  const int ldr_s = cres + kBPad;
  constexpr int LDB = kConvBK + kBPad;
  bf16* win = reinterpret_cast<bf16*>(smem_raw);                 // [18][cin + pad]
  bf16* rwin = win + size_t(kConvBM + 2) * lda_s;                // [16][cres + pad]
  bf16* bst = rwin + size_t(kConvBM) * ldr_s;                    // [stages][64][BK + pad]

  EST_TR_BEGIN
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m0 = blockIdx.x * kConvBM;
  const int nb = blockIdx.y * kConvBN;  // this CTA's first output channel
  const int k_conv = 3 * cin;
  const int n_kt = k_conv / kConvBK;
  const int n_kt_res = cres / kConvBK;
  const int n_kt_all = n_kt + n_kt_res;

  auto load_b = [&](int kt, int stage) {
    const bf16* wsrc;
    int ldw, k0;
    if (kt < n_kt) { wsrc = a.w; ldw = k_conv; k0 = kt * kConvBK; }
    else { wsrc = a.res_w; ldw = cres; k0 = (kt - n_kt) * kConvBK; }
    bf16* dst = bst + size_t(stage) * kConvBN * LDB;
#pragma unroll
    for (int i = 0; i < (kConvBN * kConvBK / 8) / kConvThreads; ++i) {
      const int chunk = tid + i * kConvThreads;
      const int n = chunk / (kConvBK / 8), kc = (chunk % (kConvBK / 8)) * 8;
      cp_async16(dst + n * LDB + kc, wsrc + size_t(nb + n) * ldw + k0 + kc, true);
    }
  };
  // Weights do not depend on the predecessor: prefetch the first stages before the dependency wait.
#pragma unroll
  for (int s = 0; s < ST - 1; ++s)
    if (s < n_kt_all) load_b(s, s);
  // r12: row metadata (est_prep, start of the call) and the epilogue's per-channel constants do not depend
  // on the predecessor: read them before the wait instead of in a dependent chain after it.
  const int r = a.frame_row[m0];
  EstRow e{};
  if (r >= 0) e = a.erows[r];
  const int g = lane >> 2, tq = lane & 3;
  float cb[2][2], cg_[2][2], cbt[2][2], cte[2][2], crb[2][2];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt)
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int n = nb + warp * 16 + nt * 8 + 2 * tq + u;
      cb[nt][u] = __bfloat162float(a.bias[n]);
      cg_[nt][u] = cbt[nt][u] = cte[nt][u] = crb[nt][u] = 0.f;
      if constexpr (MODE != 2) {
        cg_[nt][u] = __bfloat162float(a.ln_g[n]);
        cbt[nt][u] = __bfloat162float(a.ln_b[n]);
      }
      if constexpr (MODE == 0) cte[nt][u] = __bfloat162float(a.temb[n]);
      if constexpr (MODE == 1) crb[nt][u] = __bfloat162float(a.res_b[n]);
    }
  pdl_wait();
  if (a.trig_early) pdl_trigger();  // r15: wait -> trigger -> work (dependents' prologues overlap this kernel)
  EST_TR(1);
  const int t0 = m0 - e.fr_off;  // local frame of the tile's first output frame
  // Input window: rows w = 0..17 hold local frames t0 - 2 + w (masked input).
  for (int idx = tid; idx < (kConvBM + 2) * (cin / 8); idx += kConvThreads) {
    const int w = idx / (cin / 8), c = (idx % (cin / 8)) * 8;
    const int t = t0 - 2 + w;
    bf16* dst = win + w * lda_s + c;
    if (r < 0 || t >= e.valid) {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    } else if (t >= 0) {
      const int f = e.fr_off + t;
      const bf16* src = (c < a.c0) ? a.in0 + size_t(f) * a.ld0 + c : a.in1 + size_t(f) * a.ld1 + (c - a.c0);
      cp_async16(dst, src, true);
    } else if (e.past > 0) {
      // conv tail: [cin][2] channel-major; t = -2 -> j = 0, t = -1 -> j = 1
      const bf16* tail = a.tail_state + size_t(e.slot) * a.tail_slot_stride;
      const int j = t + 2;
#pragma unroll
      for (int q = 0; q < 8; ++q) dst[q] = tail[(c + q) * 2 + j];
    } else {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
  }
  if constexpr (MODE == 1) {
    for (int idx = tid; idx < kConvBM * (cres / 8); idx += kConvThreads) {
      const int w = idx / (cres / 8), c = (idx % (cres / 8)) * 8;
      const int t = t0 + w;
      bf16* dst = rwin + w * ldr_s + c;
      if (r < 0 || t >= e.valid) {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
      } else {
        const int f = e.fr_off + t;
        const bf16* src = (c < a.res_c0) ? a.res_in0 + size_t(f) * a.res_ld0 + c
                                         : a.res_in1 + size_t(f) * a.res_ld1 + (c - a.res_c0);
        cp_async16(dst, src, true);
      }
    }
  }
  cp_async_commit();  // group 0 = the prefetched weight stages + the window
#pragma unroll
  for (int s = 1; s < ST - 1; ++s) cp_async_commit();  // empty groups keep the count uniform

  float acc[2][4];
  float accr[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) { acc[i][j] = 0.f; accr[i][j] = 0.f; }

  const int wn0 = warp * 16;  // this warp's 16 output channels within the CTA's 64
  for (int kt = 0; kt < n_kt_all; ++kt) {
    if (kt + ST - 1 < n_kt_all) load_b(kt + ST - 1, (kt + ST - 1) % ST);
    cp_async_commit();
    cp_async_wait<ST - 1>();
    __syncthreads();
    if (kt == 0) EST_TR(2);
    const bf16* bs = bst + size_t(kt % ST) * kConvBN * LDB;
    const bool is_res = kt >= n_kt;
#pragma unroll
    for (int kk = 0; kk < kConvBK; kk += 16) {
      uint32_t af[4];
      if (!is_res) {
        const int kg = kt * kConvBK + kk;  // global k = j * cin + c
        const int j = kg / cin, c = kg % cin;
        const bf16* p = win + (j + (lane & 15)) * lda_s + c + (lane >> 4) * 8;
        ldsm_x4(af[0], af[1], af[2], af[3], p);
      } else {
        const int c = (kt - n_kt) * kConvBK + kk;
        const bf16* p = rwin + (lane & 15) * ldr_s + c + (lane >> 4) * 8;
        ldsm_x4(af[0], af[1], af[2], af[3], p);
      }
      uint32_t b0, b1, b2, b3;
      const int mtx = lane >> 3;
      const bf16* p = bs + (wn0 + (mtx >> 1) * 8 + (lane & 7)) * LDB + kk + (mtx & 1) * 8;
      ldsm_x4(b0, b1, b2, b3, p);
      if (!is_res) {
        mma16816(acc[0], af, b0, b1);
        mma16816(acc[1], af, b2, b3);
      } else {
        mma16816(accr[0], af, b0, b1);
        mma16816(accr[1], af, b2, b3);
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();
  EST_TR(3);
  if (!a.trig_early) pdl_trigger();

  // Tail stage: the tile that owns local frame shift - 1 saves frames shift - 2, shift - 1 of the
  // masked conv input (window rows (shift - 2) - t0 + 2 and + 1). Rank 0 of the cluster writes.
  if (blockIdx.y == 0 && r >= 0 && a.tail_stage != nullptr && e.shift > 0) {
    const int last = e.shift - 1;
    if (last >= t0 && last < t0 + kConvBM) {
      const int w0 = (e.shift - 2) - t0 + 2;
      bf16* stg = a.tail_stage + size_t(r) * a.stage_row_stride;
      for (int c = tid; c < cin; c += kConvThreads) {
        stg[c * 2 + 0] = win[w0 * lda_s + c];
        stg[c * 2 + 1] = win[(w0 + 1) * lda_s + c];
      }
    }
  }

  // Epilogue. Thread rows g, g + 8; columns nb + wn0 + nt * 8 + 2 t, + 1.
  float v[2][4];
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    const float b0 = cb[nt][0], b1 = cb[nt][1];
    v[0][nt * 2 + 0] = bfr(acc[nt][0] + b0);
    v[0][nt * 2 + 1] = bfr(acc[nt][1] + b1);
    v[1][nt * 2 + 0] = bfr(acc[nt][2] + b0);
    v[1][nt * 2 + 1] = bfr(acc[nt][3] + b1);
  }
  float maskv[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) maskv[h] = (r >= 0 && t0 + g + 8 * h < e.valid) ? 1.f : 0.f;

  if constexpr (PORTABLE && MODE != 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int f = m0 + g + 8 * h;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int n = nb + wn0 + nt * 8 + 2 * tq;
        *reinterpret_cast<uint32_t*>(a.portable_values + size_t(f) * kHid + n) =
            pack_bf16(v[h][nt * 2], v[h][nt * 2 + 1]);
        if constexpr (MODE == 1) {
          const float r0 = bfr(accr[nt][h * 2] + crb[nt][0]);
          const float r1 = bfr(accr[nt][h * 2 + 1] + crb[nt][1]);
          *reinterpret_cast<uint32_t*>(a.portable_residual + size_t(f) * kHid + n) = pack_bf16(r0, r1);
        }
      }
    }
  }

  if constexpr (MODE != 2 && !PORTABLE) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cg::cluster_group cluster = cg::this_cluster();
    float* red = reinterpret_cast<float*>(bst + size_t(ST) * kConvBN * LDB);  // [4 warps][16]
    float* part = red + 4 * 16;  // [2][16]: this CTA's row sums (mean, var)
    // LayerNorm over the 256 channels of each frame (fp32 stats, two passes), reduced over the
    // 4 warps of this CTA, then over the 4 CTAs of the cluster in rank order (deterministic).
    auto row_reduce = [&](float s0, float s1, int which) {
      s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
      if (tq == 0) { red[warp * 16 + g] = s0; red[warp * 16 + g + 8] = s1; }
      __syncthreads();
      if (tid < 16) part[which * 16 + tid] = red[tid] + red[16 + tid] + red[32 + tid] + red[48 + tid];
      cluster.sync();
      float out[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float tot = 0.f;
#pragma unroll
        for (int rk = 0; rk < kConvCluster; ++rk) {
          const float* rp = cluster.map_shared_rank(part, rk);
          tot += rp[which * 16 + g + 8 * h];
        }
        out[h] = tot;
      }
      return make_float2(out[0], out[1]);
    };
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { s0 += v[0][i]; s1 += v[1][i]; }
    auto sum = row_reduce(s0, s1, 0);
    const float mean[2] = {sum.x * (1.0f / 256.0f), sum.y * (1.0f / 256.0f)};
    float q0 = 0.f, q1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float d0 = v[0][i] - mean[0], d1 = v[1][i] - mean[1];
      q0 += d0 * d0;
      q1 += d1 * d1;
    }
    __syncthreads();  // red is reused
    auto sq = row_reduce(q0, q1, 1);
    cluster.sync();   // every remote read of `part` is done before any CTA moves on / exits
    const float rstd[2] = {rsqrtf(sq.x * (1.0f / 256.0f) + 1e-5f), rsqrtf(sq.y * (1.0f / 256.0f) + 1e-5f)};
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const float gm = cg_[nt][u], bt = cbt[nt][u];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float y = bfr((v[h][nt * 2 + u] - mean[h]) * rstd[h] * gm + bt);
          y = bfr(mish_f(y));
          y = y * maskv[h];
          if constexpr (MODE == 0) y = bfr(y + cte[nt][u]);
          if constexpr (MODE == 1) {
            const float rv = bfr(accr[nt][h * 2 + u] + crb[nt][u]);
            y = bfr(y + rv);
          }
          v[h][nt * 2 + u] = y;
        }
      }
    }
#else
    asm volatile("trap;");  // Host dispatch must select the portable kernel on these devices.
#endif
  }
  if constexpr (!PORTABLE || MODE == 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int f = m0 + g + 8 * h;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int n = nb + wn0 + nt * 8 + 2 * tq;
        *reinterpret_cast<uint32_t*>(a.out + size_t(f) * a.ld_out + n) = pack_bf16(v[h][nt * 2], v[h][nt * 2 + 1]);
      }
    }
  }
  EST_TR_END(20 + MODE);
}

// Four logical ranks share one block. Keep the clustered kernel's reduction and BF16 rounding order.
template <int MODE>
__global__ void __launch_bounds__(512) est_conv_portable_epilogue(ConvArgs a) {
  static_assert(MODE == 0 || MODE == 1 || MODE == 3);
  __shared__ float red[4][64];
  __shared__ float part[4][16];
  const int rank = threadIdx.x / 128, tid = threadIdx.x % 128;
  const int warp = tid / 32, lane = tid % 32, g = lane / 4, tq = lane % 4;
  const int m0 = blockIdx.x * kConvBM;
  const int row = a.frame_row[m0];
  EstRow e{};
  if (row >= 0) e = a.erows[row];
  float v[2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int n = rank * 64 + warp * 16 + nt * 8 + tq * 2 + u;
        v[h][nt * 2 + u] = __bfloat162float(a.portable_values[size_t(m0 + g + 8 * h) * kHid + n]);
      }
  auto reduce = [&](float s0, float s1) {
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
    if (tq == 0) { red[rank][warp * 16 + g] = s0; red[rank][warp * 16 + g + 8] = s1; }
    __syncthreads();
    if (tid < 16) part[rank][tid] = red[rank][tid] + red[rank][16 + tid] + red[rank][32 + tid] + red[rank][48 + tid];
    __syncthreads();
    float sums[2] = {};
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int rk = 0; rk < 4; ++rk) sums[h] += part[rk][g + 8 * h];
    return make_float2(sums[0], sums[1]);
  };
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) { s0 += v[0][i]; s1 += v[1][i]; }
  const float2 sum = reduce(s0, s1);
  const float mean[2] = {sum.x * (1.f / 256.f), sum.y * (1.f / 256.f)};
  float q0 = 0.f, q1 = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float d0 = v[0][i] - mean[0], d1 = v[1][i] - mean[1];
    q0 += d0 * d0;
    q1 += d1 * d1;
  }
  __syncthreads();
  const float2 sq = reduce(q0, q1);
  const float rstd[2] = {rsqrtf(sq.x * (1.f / 256.f) + 1e-5f), rsqrtf(sq.y * (1.f / 256.f) + 1e-5f)};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int f = m0 + g + 8 * h;
    const float mask = row >= 0 && f - e.fr_off < e.valid ? 1.f : 0.f;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int n = rank * 64 + warp * 16 + nt * 8 + tq * 2;
      float out[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        float y = bfr((v[h][nt * 2 + u] - mean[h]) * rstd[h] * __bfloat162float(a.ln_g[n + u]) + __bfloat162float(a.ln_b[n + u]));
        y = bfr(mish_f(y));
        y *= mask;
        if constexpr (MODE == 0) y = bfr(y + __bfloat162float(a.temb[n + u]));
        if constexpr (MODE == 1) y = bfr(y + __bfloat162float(a.portable_residual[size_t(f) * kHid + n + u]));
        out[u] = y;
      }
      *reinterpret_cast<uint32_t*>(a.out + size_t(f) * a.ld_out + n) = pack_bf16(out[0], out[1]);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Transformer GEMMs: out[m, n] = A[m, :] . W[n, :] with A either LayerNorm(hidden) resident in shared
// memory (K = 256; LN1 -> fused QKV, LN3 -> FF1) or streamed (to_out K = 512, FF2 K = 1024,
// final_proj K = 256). Tile shapes depend on the bucket; the per-element reduction order (k ascending,
// one mma chain per output) does not, so rows are batch invariant.
//   GM_QKV:   n < 1024 -> K|V into the row's pool page (frames t < suffix), n >= 1024 -> q
//   GM_OUT:   hidden = bf16(hidden + bf16(acc + b))        (to_out + residual)
//   GM_FF1:   out = bf16(gelu(bf16(acc + b)))
//   GM_FF2:   hidden = bf16(hidden + bf16(acc + b))        (ff.net.2 + residual)
//   GM_FINAL: v = bf16(bf16(acc + b) * mask) for n < 80 (final_proj, W padded to 128 rows);
//             x <- bf16(x + bf16(dt * v)) in xin columns 0..79 (the Euler step)
// ---------------------------------------------------------------------------------------------
enum GemmMode : int { GM_QKV = 0, GM_OUT = 1, GM_FF1 = 2, GM_FF2 = 3, GM_FINAL = 4 };

struct GemmArgs {
  const bf16* a; int lda; int k;          // A (hidden for LN modes)
  const bf16* w; const bf16* bias;        // W [N][K]
  const bf16* ln_g; const bf16* ln_b;
  int n;
  bf16* out; int ld_out;                  // q / ff / hidden
  // QKV pool write
  const EstRow* erows; const int32_t* frame_row; const int32_t* page_table;
  bf16* pool_layer;                       // pool.est + ((step * 56 + layer) * n_pages) * 64 * 1024
  int frames_cap;
  // GM_FINAL
  bf16* xin; float dt; float* deriv_out; float* mel_out;
  int trig_early;                         // see ConvArgs::trig_early
};

template <int BM, int BN, int BK, int ST, int MODE>
__host__ __device__ constexpr size_t gemm_smem_bytes() {
  return (MODE == GM_QKV || MODE == GM_FF1)
             ? size_t(BM) * (256 + kBPad) * 2 + size_t(ST) * BN * (BK + kBPad) * 2
             : size_t(ST) * (BM + BN) * (BK + kBPad) * 2;
}

// KS > 1: the K range is cut into KS fixed slices, one CTA each, grouped in a (1, 1, KS) cluster; rank 0
// adds the partial tiles of ranks 1..KS-1 through DSMEM in rank order and runs the epilogue. The slices
// and the order are the same in every bucket, so rows stay batch invariant.
// NWP = warps per CTA (r16: 1-2-warp CTAs with 16-row tiles for the N = 256 GEMMs at small buckets, so FF2 /
// to_out get 4-8x more CTAs; each warp still owns a 16 x 16 tile with one k-ascending mma chain per element).
template <int BM, int BN, int WM, int BK, int ST, int MODE, int KS, int NWP = 8>
__global__ void __launch_bounds__(32 * NWP) est_gemm_kernel(GemmArgs a) {
  constexpr bool kLn = (MODE == GM_QKV || MODE == GM_FF1);
  constexpr int NTHR = 32 * NWP;
  constexpr int WN = NWP / WM;
  constexpr int TM = BM / WM;  // warp tile rows
  constexpr int TN = BN / WN;  // warp tile cols
  constexpr int MT = TM / 16;
  constexpr int NT = TN / 8;
  static_assert(MT >= 1 && NT >= 2 && NT % 2 == 0 && TM % 16 == 0, "tile config");
  constexpr int LDB = BK + kBPad;
  constexpr int LDA_RES = 256 + kBPad;
  EST_TR_BEGIN
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* as = reinterpret_cast<bf16*>(smem_raw);
  bf16* bs = kLn ? as + BM * LDA_RES : as + ST * BM * LDB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp % WN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int k_base = (KS > 1) ? int(blockIdx.z) * (a.k / KS) : 0;
  const int n_kt = (a.k / KS) / BK;

  auto load_b = [&](int kt, int stage) {
    bf16* dst = bs + stage * BN * LDB;
#pragma unroll
    for (int chunk = tid; chunk < BN * (BK / 8); chunk += NTHR) {
      const int n = chunk / (BK / 8), kc = (chunk % (BK / 8)) * 8;
      cp_async16(dst + n * LDB + kc, a.w + size_t(n0 + n) * a.k + k_base + kt * BK + kc, true);
    }
  };
  auto load_a = [&](int kt, int stage) {
    bf16* dst = as + stage * BM * LDB;
#pragma unroll
    for (int chunk = tid; chunk < BM * (BK / 8); chunk += NTHR) {
      const int m = chunk / (BK / 8), kc = (chunk % (BK / 8)) * 8;
      cp_async16(dst + m * LDB + kc, a.a + size_t(m0 + m) * a.lda + k_base + kt * BK + kc, true);
    }
  };

  // Weight prefetch (independent of the predecessor) before the dependency wait.
#pragma unroll
  for (int s = 0; s < ST - 1; ++s)
    if (s < n_kt) load_b(s, s);

  // r12: everything else that does not depend on the predecessor is also read before the wait, so no
  // dependent round trip is left in the prologue / epilogue: bias, LayerNorm affine, and the per-row
  // metadata (frame -> row -> pool page, written by est_prep at the start of the call; every GEMM is
  // >= 3 kernels later, so it is complete when this grid can launch). Same values, same arithmetic.
  const int g = lane >> 2, tq = lane & 3;
  float2 biasv[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int n = n0 + wn * TN + nt * 8 + 2 * tq;
    biasv[nt] = make_float2(0.f, 0.f);
    if constexpr (MODE != GM_QKV) {
      if (MODE != GM_FINAL || n < kMelC)
        biasv[nt] = make_float2(__bfloat162float(a.bias[n]), __bfloat162float(a.bias[n + 1]));
    }
  }
  uint4 lng = make_uint4(0, 0, 0, 0), lnb = make_uint4(0, 0, 0, 0);
  if constexpr (kLn) {
    lng = *reinterpret_cast<const uint4*>(a.ln_g + lane * 8);
    lnb = *reinterpret_cast<const uint4*>(a.ln_b + lane * 8);
  }
  // QKV: element offset of this thread's rows in the layer's pool (-1 = not written); FINAL: row mask.
  int64_t rowoff[MT][2];
  float rowmask[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      rowoff[mt][h] = -1;
      rowmask[mt][h] = 0.f;
      if constexpr (MODE == GM_QKV || MODE == GM_FINAL) {
        const int f = m0 + wm * TM + mt * 16 + g + 8 * h;
        const int r = (f < a.frames_cap) ? a.frame_row[f] : -1;
        if (r >= 0) {
          const EstRow e = a.erows[r];
          const int t = f - e.fr_off;
          if constexpr (MODE == GM_QKV) {
            if (t < e.suffix) {
              const int abs_f = e.past + t;
              const int page = a.page_table[e.page_off + abs_f / kPageFrames];
              rowoff[mt][h] = (int64_t(page) * kPageFrames + (abs_f % kPageFrames)) * (2 * kInner);
            }
          } else {
            rowmask[mt][h] = (t < e.valid) ? 1.f : 0.f;
          }
        }
      }
    }

  pdl_wait();
  if (a.trig_early) pdl_trigger();  // r15: wait -> trigger -> work (dependents' prologues overlap this kernel)
  EST_TR(1);
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if constexpr (!kLn) {
      if (s < n_kt) load_a(s, s);
    }
    cp_async_commit();
  }
  // OUT / FF2: the residual (this thread's epilogue elements of hidden) is read now, overlapping the
  // main loop, instead of after it. hidden was last written >= 2 kernels ago (complete after the wait).
  uint32_t resv[MT][2][NT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        resv[mt][h][nt] = 0u;
        if constexpr (MODE == GM_OUT || MODE == GM_FF2) {
          const int f = m0 + wm * TM + mt * 16 + g + 8 * h;
          const int n = n0 + wn * TN + nt * 8 + 2 * tq;
          if (f < a.frames_cap) resv[mt][h][nt] = *reinterpret_cast<const uint32_t*>(a.out + size_t(f) * a.ld_out + n);
        }
      }

  if constexpr (kLn) {
    // LayerNorm of BM rows (K = 256): warp w owns rows w, w + 8, ...; lane holds 8 channels. All of a
    // warp's row loads are issued before any reduction (independent loads in flight, not one round trip
    // per row).
    constexpr int RPW = BM / NWP;
    constexpr int BATCH = RPW < 8 ? RPW : 8;
    // r12: the BATCH rows' shuffle reductions are interleaved (independent chains in flight instead of
    // one row after another); each row's sums keep exactly the same order, so the bits are unchanged.
    const bf16* gv = reinterpret_cast<const bf16*>(&lng);
    const bf16* bv = reinterpret_cast<const bf16*>(&lnb);
#pragma unroll
    for (int b0 = 0; b0 < RPW; b0 += BATCH) {
      uint4 raw[BATCH];
#pragma unroll
      for (int i = 0; i < BATCH; ++i)
        raw[i] = *reinterpret_cast<const uint4*>(a.a + size_t(m0 + warp + NWP * (b0 + i)) * a.lda + lane * 8);
      float x[BATCH][8], s[BATCH], q[BATCH], mean[BATCH], rstd[BATCH];
#pragma unroll
      for (int i = 0; i < BATCH; ++i) {
        const bf16* hv = reinterpret_cast<const bf16*>(&raw[i]);
        s[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { x[i][j] = __bfloat162float(hv[j]); s[i] += x[i][j]; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < BATCH; ++i) s[i] += __shfl_xor_sync(0xffffffffu, s[i], o);
#pragma unroll
      for (int i = 0; i < BATCH; ++i) {
        mean[i] = s[i] * (1.0f / 256.0f);
        q[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { const float d = x[i][j] - mean[i]; q[i] += d * d; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < BATCH; ++i) q[i] += __shfl_xor_sync(0xffffffffu, q[i], o);
#pragma unroll
      for (int i = 0; i < BATCH; ++i) {
        const int m = warp + NWP * (b0 + i);
        rstd[i] = rsqrtf(q[i] * (1.0f / 256.0f) + 1e-5f);
        uint32_t packed[4];
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const float y0 = (x[i][j] - mean[i]) * rstd[i] * __bfloat162float(gv[j]) + __bfloat162float(bv[j]);
          const float y1 = (x[i][j + 1] - mean[i]) * rstd[i] * __bfloat162float(gv[j + 1]) + __bfloat162float(bv[j + 1]);
          packed[j / 2] = pack_bf16(y0, y1);
        }
        *reinterpret_cast<uint4*>(as + m * LDA_RES + lane * 8) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
      }
    }
    EST_TR(2);  // LN GEMMs: t2 = this thread's LayerNorm rows written (the k loop's first wait is in 'loop')
  }

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  for (int kt = 0; kt < n_kt; ++kt) {
    if (kt + ST - 1 < n_kt) {
      load_b(kt + ST - 1, (kt + ST - 1) % ST);
      if constexpr (!kLn) load_a(kt + ST - 1, (kt + ST - 1) % ST);
    }
    cp_async_commit();
    cp_async_wait<ST - 1>();
    __syncthreads();
    if (!kLn && kt == 0) EST_TR(2);
    const bf16* bsk = bs + (kt % ST) * BN * LDB;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int row = wm * TM + mt * 16 + (lane & 15);
        const bf16* p;
        if constexpr (kLn) p = as + row * LDA_RES + kt * BK + kk + (lane >> 4) * 8;
        else p = as + (kt % ST) * BM * LDB + row * LDB + kk + (lane >> 4) * 8;
        ldsm_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3], p);
      }
#pragma unroll
      for (int np = 0; np < NT / 2; ++np) {
        uint32_t b0, b1, b2, b3;
        const int mtx = lane >> 3;
        const bf16* p = bsk + (wn * TN + np * 16 + (mtx >> 1) * 8 + (lane & 7)) * LDB + kk + (mtx & 1) * 8;
        ldsm_x4(b0, b1, b2, b3, p);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(acc[mt][np * 2], af[mt], b0, b1);
          mma16816(acc[mt][np * 2 + 1], af[mt], b2, b3);
        }
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();
  EST_TR(3);
  if (!a.trig_early) pdl_trigger();
  if constexpr (KS > 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    // the pipeline buffers are free now: this CTA's partial tile, per-thread contiguous
    float* part = reinterpret_cast<float*>(smem_raw);
    constexpr int PER = MT * NT * 4;
    static_assert(size_t(NTHR) * PER * 4 <= gemm_smem_bytes<BM, BN, BK, ST, MODE>(), "partial tile must fit");
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) part[tid * PER + (mt * NT + nt) * 4 + q] = acc[mt][nt][q];
    cluster.sync();
    const bool lead = cluster.block_rank() == 0;
    if (lead) {
#pragma unroll
      for (int rk = 1; rk < KS; ++rk) {
        const float* rp = cluster.map_shared_rank(part, rk);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mt][nt][q] += rp[tid * PER + (mt * NT + nt) * 4 + q];
      }
    }
    cluster.sync();  // remote partials stay alive until rank 0 has read them
    if (!lead) return;
#else
    asm volatile("trap;");
#endif
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int f = m0 + wm * TM + mt * 16 + g + 8 * h;
      if (f >= a.frames_cap) continue;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int n = n0 + wn * TN + nt * 8 + 2 * tq;
        float y0 = acc[mt][nt][h * 2], y1 = acc[mt][nt][h * 2 + 1];
        if constexpr (MODE == GM_QKV) {
          if (n < 2 * kInner) {
            if (rowoff[mt][h] < 0) continue;
            bf16* dst = a.pool_layer + rowoff[mt][h] + n;
            *reinterpret_cast<uint32_t*>(dst) = pack_bf16(y0, y1);
          } else {
            *reinterpret_cast<uint32_t*>(a.out + size_t(f) * a.ld_out + (n - 2 * kInner)) = pack_bf16(y0, y1);
          }
        } else if constexpr (MODE == GM_FF1) {
          y0 = gelu_f(bfr(y0 + biasv[nt].x));
          y1 = gelu_f(bfr(y1 + biasv[nt].y));
          *reinterpret_cast<uint32_t*>(a.out + size_t(f) * a.ld_out + n) = pack_bf16(y0, y1);
        } else if constexpr (MODE == GM_FINAL) {
          if (n >= kMelC) continue;
          const float maskv = rowmask[mt][h];
          const float yy[2] = {y0, y1};
          const float bb[2] = {biasv[nt].x, biasv[nt].y};
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            const float v = bfr(bfr(yy[u] + bb[u]) * maskv);
            bf16* xp = a.xin + size_t(f) * kIn + n + u;
            const float xn = bfr(__bfloat162float(*xp) + bfr(a.dt * v));
            *xp = __float2bfloat16_rn(xn);
            if (a.deriv_out) a.deriv_out[size_t(f) * kMelC + n + u] = v;
            if (a.mel_out) a.mel_out[size_t(f) * kMelC + n + u] = xn;
          }
        } else {
          bf16* dst = a.out + size_t(f) * a.ld_out + n;
          const __nv_bfloat162 hv = *reinterpret_cast<const __nv_bfloat162*>(&resv[mt][h][nt]);
          y0 = __bfloat162float(hv.x) + bfr(y0 + biasv[nt].x);
          y1 = __bfloat162float(hv.y) + bfr(y1 + biasv[nt].y);
          *reinterpret_cast<uint32_t*>(dst) = pack_bf16(y0, y1);
        }
      }
    }
  }
  EST_TR_END(10 + MODE);
}

// ---------------------------------------------------------------------------------------------
// Pool attention: one CTA = (64-query block of a row, head, 256-key split); 4 warps x 16 queries,
// flash-style online softmax in registers, K/V tiles of 64 keys = one pool page. Chunk-causal on
// absolute frames: query q (absolute qa) sees keys < min(past + valid, (qa / 50 + 1) * 50) (full
// attention: keys < past + valid). Splits combine in-kernel: every split writes (m, l, O) partials;
// the last CTA to arrive (atomic counter) merges them in split order (deterministic) and resets the
// counter (graph-replay safe).
// ---------------------------------------------------------------------------------------------
struct AttnArgs {
  const bf16* q;          // [frames][512]
  bf16* out;              // [frames][512]
  const bf16* pool_layer; // as GemmArgs
  const int32_t* page_table;
  const EstRow* erows;
  const AttnItem* items;
  const int32_t* n_items;
  float* part_o;          // [items_cap][8][nsplit][64][64]
  float* part_ml;         // [items_cap][8][nsplit][64][2]
  int32_t* counters;      // [items_cap][8]
  int nsplit_cap;
  int split;              // keys per split (multiple of 64; fixed per engine, not per batch)
  float scale;
  int trig_early;         // see ConvArgs::trig_early (attention v2 only)
};

inline constexpr int kAttnLd = 64 + kBPad;
__host__ __device__ constexpr size_t attn_smem_bytes() { return size_t(64) * kAttnLd * 2 * 5; }

__device__ __forceinline__ int key_end_of(const EstRow& e, int t) {
  const int lim = e.past + e.valid;
  if (e.full) return lim;
  const int qa = e.past + t;
  return min(lim, (qa / kChunkFr + 1) * kChunkFr);
}

__global__ void __launch_bounds__(128) est_attn_kernel(AttnArgs a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* qs = reinterpret_cast<bf16*>(smem_raw);  // [64][72]
  bf16* ks = qs + 64 * kAttnLd;                   // [2][64][72]
  bf16* vs = ks + 2 * 64 * kAttnLd;               // [2][64][72]
  __shared__ int s_last;
  pdl_wait();
  const int item_idx = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
  if (item_idx >= *a.n_items) return;
  const AttnItem it = a.items[item_idx];
  const EstRow e = a.erows[it.row];
  const int qlast = min(it.q0 + 63, e.suffix - 1);
  const int kend_blk = key_end_of(e, qlast);
  const int nsplit = min(a.nsplit_cap, (kend_blk + a.split - 1) / a.split);
  if (sp >= nsplit) return;
  const int k_lo = sp * a.split;
  const int k_hi = min(k_lo + a.split, kend_blk);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  // Q tile
  for (int chunk = tid; chunk < 64 * 8; chunk += 128) {
    const int m = chunk >> 3, c = (chunk & 7) * 8;
    const int t = it.q0 + m;
    const bool ok = t < e.suffix;
    cp_async16(qs + m * kAttnLd + c, a.q + size_t(e.fr_off + (ok ? t : 0)) * kInner + h * 64 + c, ok);
  }
  auto load_kv = [&](int kt0, int stage) {
    const int page = a.page_table[e.page_off + kt0 / kPageFrames];
    const bf16* base = a.pool_layer + size_t(page) * kPageFrames * (2 * kInner);
    for (int chunk = tid; chunk < 64 * 8; chunk += 128) {
      const int k = chunk >> 3, c = (chunk & 7) * 8;
      const bool ok = kt0 + k < k_hi;
      const bf16* src = base + size_t(ok ? k : 0) * (2 * kInner) + h * 64 + c;
      cp_async16(ks + (stage * 64 + k) * kAttnLd + c, src, ok);
      cp_async16(vs + (stage * 64 + k) * kAttnLd + c, src + kInner, ok);
    }
  };
  load_kv(k_lo, 0);
  cp_async_commit();

  const int g = lane >> 2, tq = lane & 3;
  const int qrow0 = warp * 16 + g;  // local query rows qrow0, qrow0 + 8 within the block
  int kend[2];
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) kend[hh] = key_end_of(e, min(it.q0 + qrow0 + 8 * hh, e.suffix - 1));
  float m_i[2] = {-1e30f, -1e30f}, l_i[2] = {0.f, 0.f};
  float o[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
  uint32_t qf[4][4];
  const float sl2 = a.scale * 1.4426950408889634f;  // exp(x) = exp2(x * log2 e)

  const int n_tiles = (k_hi - k_lo + 63) / 64;
  for (int ti = 0; ti < n_tiles; ++ti) {
    const int kt0 = k_lo + ti * 64;
    if (ti + 1 < n_tiles) load_kv(kt0 + 64, (ti + 1) & 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    if (ti == 0) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const bf16* p = qs + (warp * 16 + (lane & 15)) * kAttnLd + kk * 16 + (lane >> 4) * 8;
        ldsm_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], p);
      }
    }
    const bf16* kst = ks + ((ti & 1) * 64) * kAttnLd;
    const bf16* vst = vs + ((ti & 1) * 64) * kAttnLd;
    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b0, b1, b2, b3;
        const int mtx = lane >> 3;
        const bf16* p = kst + (np * 16 + (mtx >> 1) * 8 + (lane & 7)) * kAttnLd + kk * 16 + (mtx & 1) * 8;
        ldsm_x4(b0, b1, b2, b3, p);
        mma16816(s[np * 2], qf[kk], b0, b1);
        mma16816(s[np * 2 + 1], qf[kk], b2, b3);
      }
    }
    // mask + online softmax (base 2)
    float mx[2] = {m_i[0], m_i[1]};
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int hh = u >> 1;
        const int key = kt0 + nt * 8 + 2 * tq + (u & 1);
        const float val = (key < kend[hh]) ? s[nt][u] * sl2 : -1e30f;
        s[nt][u] = val;
        mx[hh] = fmaxf(mx[hh], val);
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
    }
    float alpha[2], rs[2] = {0.f, 0.f};
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) alpha[hh] = exp2f(m_i[hh] - mx[hh]);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int hh = u >> 1;
        const float p = (s[nt][u] <= -1e29f) ? 0.f : exp2f(s[nt][u] - mx[hh]);
        s[nt][u] = p;
        rs[hh] += p;
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      rs[hh] += __shfl_xor_sync(0xffffffffu, rs[hh], 1);
      rs[hh] += __shfl_xor_sync(0xffffffffu, rs[hh], 2);
      l_i[hh] = l_i[hh] * alpha[hh] + rs[hh];
      m_i[hh] = mx[hh];
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      o[nt][0] *= alpha[0]; o[nt][1] *= alpha[0];
      o[nt][2] *= alpha[1]; o[nt][3] *= alpha[1];
    }
    // O += P V
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t pa[4];
      pa[0] = pack_bf16(s[2 * kk][0], s[2 * kk][1]);
      pa[1] = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
      pa[2] = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      pa[3] = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b0, b1, b2, b3;
        const int mtx = lane >> 3;
        const bf16* p = vst + (kk * 16 + (mtx & 1) * 8 + (lane & 7)) * kAttnLd + dp * 16 + (mtx >> 1) * 8;
        ldsm_x4_t(b0, b1, b2, b3, p);
        mma16816(o[dp * 2], pa, b0, b1);
        mma16816(o[dp * 2 + 1], pa, b2, b3);
      }
    }
    __syncthreads();
  }
  pdl_trigger();

  if (nsplit == 1) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int t = it.q0 + qrow0 + 8 * hh;
      if (t >= e.suffix) continue;
      const float inv = 1.0f / fmaxf(l_i[hh], 1e-30f);
      bf16* dst = a.out + size_t(e.fr_off + t) * kInner + h * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
        *reinterpret_cast<uint32_t*>(dst + nt * 8 + 2 * tq) = pack_bf16(o[nt][hh * 2] * inv, o[nt][hh * 2 + 1] * inv);
    }
    return;
  }
  // Partials.
  const size_t pbase = (size_t(item_idx) * 8 + h) * a.nsplit_cap;
  float* po = a.part_o + (pbase + sp) * 64 * 64;
  float* pml = a.part_ml + (pbase + sp) * 64 * 2;
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int row = qrow0 + 8 * hh;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      po[row * 64 + nt * 8 + 2 * tq] = o[nt][hh * 2];
      po[row * 64 + nt * 8 + 2 * tq + 1] = o[nt][hh * 2 + 1];
    }
    if (tq == 0) { pml[row * 2] = m_i[hh]; pml[row * 2 + 1] = l_i[hh]; }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(a.counters + item_idx * 8 + h, 1);
    s_last = (prev == nsplit - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  // Merge in split order: thread handles row (tid / 2) and 32 dims.
  const int row = tid >> 1, d0 = (tid & 1) * 32;
  const int t = it.q0 + row;
  float mm = -1e30f;
  for (int s2 = 0; s2 < nsplit; ++s2) mm = fmaxf(mm, __ldcg(a.part_ml + (pbase + s2) * 128 + row * 2));
  float L = 0.f;
  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = 0.f;
  for (int s2 = 0; s2 < nsplit; ++s2) {
    const float ms = __ldcg(a.part_ml + (pbase + s2) * 128 + row * 2);
    const float ls = __ldcg(a.part_ml + (pbase + s2) * 128 + row * 2 + 1);
    const float w = exp2f(ms - mm);
    L += ls * w;
    const float* src = a.part_o + (pbase + s2) * 4096 + row * 64 + d0;
#pragma unroll
    for (int i = 0; i < 32; ++i) acc[i] += __ldcg(src + i) * w;
  }
  if (t < e.suffix) {
    const float inv = 1.0f / fmaxf(L, 1e-30f);
    bf16* dst = a.out + size_t(e.fr_off + t) * kInner + h * 64 + d0;
#pragma unroll
    for (int i = 0; i < 32; i += 2) *reinterpret_cast<uint32_t*>(dst + i) = pack_bf16(acc[i] * inv, acc[i + 1] * inv);
  }
  if (tid == 0) a.counters[item_idx * 8 + h] = 0;
}

// ---------------------------------------------------------------------------------------------
// Pool attention v2 (the default): one CTA = (16-query group of a row, head), 4 warps. The visible
// keys are cut into 32-key tiles; warp w takes tiles w, w + 4, w + 8, ... (fixed partition, batch
// invariant), each warp with its own double-buffered K/V tiles in shared memory, so four key streams
// are in flight at once. Every warp keeps a flash-style (m, l, O) over its tiles; the four are merged
// in warp order through shared memory: no global partials, no atomics, one launch.
// ---------------------------------------------------------------------------------------------
inline constexpr int kA2Tile = 32;
inline constexpr int kA2Ld = 64 + kBPad;
inline constexpr int kA2MaxPages = 128;  // keys <= 8192 frames read their page list from shared memory
// SW = true (r17): K/V tiles are stored unpadded (64 elements per row) with the 16-byte chunk index XORed
// with (row & 7), so the 8 rows one ldmatrix phase reads still fall in 8 different bank groups. Dropping
// the 8-element padding is what lets 3 stages per warp fit in the 99 KB per-block limit.
template <bool SW>
__host__ __device__ constexpr int attn2_kv_ld() { return SW ? 64 : kA2Ld; }
template <bool SW>
__device__ __forceinline__ int attn2_kv_off(int r, int c) {  // element offset of (key row r, 8-element chunk c)
  if constexpr (SW) return r * 64 + ((c ^ (r & 7)) << 3);
  else return r * kA2Ld + (c << 3);
}
template <int STG, int MTQ = 1, bool SW = false>
__host__ __device__ constexpr size_t attn2_smem_bytes() {
  // Q [16 MTQ][72] + per warp STG stages x (K, V) x [32][72 | 64] bf16; the combine reuses the K/V region.
  return size_t(16 * MTQ) * kA2Ld * 2 + size_t(4) * STG * 2 * kA2Tile * attn2_kv_ld<SW>() * 2;
}

// STG = 2: each warp double-buffers its K/V tiles (small buckets: few CTAs, overlap inside the warp).
// STG = 1: half the shared memory, so 2 CTAs fit per SM (large buckets: overlap across CTAs).
// STG = 3 (r17, with SW): each warp keeps two tiles in flight instead of one. At B1 a CTA has one warp per
// scheduler, so nothing else hides a tile's load latency (r16 trace: ~0.8 us per 32-key tile).
// MTQ = queries per CTA / 16: with MTQ = 2 a CTA serves 32 queries, so every K/V tile read from the pool
// feeds twice the queries (half the L2 traffic at large batches). Each query row runs the same mma /
// softmax / merge sequence whatever STG, MTQ and SW are, so the choice never changes a bit of the result.
template <int STG, int MTQ = 1, bool SW = false>
__global__ void __launch_bounds__(128) est_attn2_kernel(AttnArgs a) {
  constexpr int QG = 16 * MTQ;
  constexpr int LDK = attn2_kv_ld<SW>();
  constexpr int GPI = 64 / QG;  // groups per 64-query item
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* qs = reinterpret_cast<bf16*>(smem_raw);
  bf16* kvbase = qs + QG * kA2Ld;
  // r12: the work item, its row and the row's page list come from est_prep / the host (complete before
  // this grid can launch), so they are read before the dependency wait: after it, only Q and K/V reads
  // remain, with no page-table round trip in front of each K/V tile.
  __shared__ int pages_s[kA2MaxPages];
  EST_TR_BEGIN
  const int item_idx = blockIdx.x / GPI, grp = blockIdx.x % GPI, h = blockIdx.y;
  if (item_idx >= *a.n_items) return;
  const AttnItem it = a.items[item_idx];
  const EstRow e = a.erows[it.row];
  const int qg0 = it.q0 + grp * QG;  // first local query of this group
  if (qg0 >= e.suffix) return;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kend_grp = key_end_of(e, min(qg0 + QG - 1, e.suffix - 1));
  const int n_tiles = (kend_grp + kA2Tile - 1) / kA2Tile;
  const int n_pg = (kend_grp + kPageFrames - 1) / kPageFrames;
  const bool pages_in_smem = n_pg <= kA2MaxPages;
  if (pages_in_smem)
    for (int i = tid; i < n_pg; i += 128) pages_s[i] = a.page_table[e.page_off + i];
  __syncthreads();
  pdl_wait();
  if (a.trig_early) pdl_trigger();  // r15: wait -> trigger -> work (dependents' prologues overlap this kernel)
  EST_TR(1);
  bf16* ks = kvbase + size_t(warp) * STG * 2 * kA2Tile * LDK;  // [stage][K|V][32][LDK]

  for (int chunk = tid; chunk < QG * 8; chunk += 128) {
    const int m = chunk >> 3, c = (chunk & 7) * 8;
    const int t = qg0 + m;
    const bool ok = t < e.suffix;
    cp_async16(qs + m * kA2Ld + c, a.q + size_t(e.fr_off + (ok ? t : qg0)) * kInner + h * 64 + c, ok);
  }
  cp_async_commit();
  auto load_kv = [&](int tile, int stage) {
    const int kt0 = tile * kA2Tile;
    const int page = pages_in_smem ? pages_s[kt0 / kPageFrames] : a.page_table[e.page_off + kt0 / kPageFrames];
    const bf16* base = a.pool_layer + (size_t(page) * kPageFrames + (kt0 % kPageFrames)) * (2 * kInner) + h * 64;
    bf16* dk = ks + size_t(stage) * 2 * kA2Tile * LDK;
    bf16* dv = dk + kA2Tile * LDK;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int chunk = lane + i * 32;  // 32 keys x 8 chunks
      const int k = chunk >> 3, c = (chunk & 7) * 8;
      const bool ok = kt0 + k < kend_grp;
      const bf16* src = base + size_t(ok ? k : 0) * (2 * kInner) + c;
      cp_async16(dk + attn2_kv_off<SW>(k, chunk & 7), src, ok);
      cp_async16(dv + attn2_kv_off<SW>(k, chunk & 7), src + kInner, ok);
    }
  };
  int my = warp;  // this warp's next tile
  // prologue: STG - 1 tiles in flight (one for STG = 1), each its own cp.async group after Q's
  constexpr int PRE = STG > 1 ? STG - 1 : 1;
#pragma unroll
  for (int j = 0; j < PRE; ++j) {
    if (my + 4 * j < n_tiles) load_kv(my + 4 * j, j);
    cp_async_commit();
  }
  cp_async_wait<PRE - 1>();  // Q and this warp's first tile have landed (later prologue tiles may not)
  __syncthreads();  // Q visible to every warp
  EST_TR(2);

  const int g = lane >> 2, tq = lane & 3;
  int kend[MTQ][2];
  uint32_t qf[MTQ][4][4];
  float m_i[MTQ][2], l_i[MTQ][2];
  float o[MTQ][8][4];
#pragma unroll
  for (int mt = 0; mt < MTQ; ++mt) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      kend[mt][hh] = key_end_of(e, min(qg0 + mt * 16 + g + 8 * hh, e.suffix - 1));
      m_i[mt][hh] = -1e30f;
      l_i[mt][hh] = 0.f;
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const bf16* p = qs + (mt * 16 + (lane & 15)) * kA2Ld + kk * 16 + (lane >> 4) * 8;
      ldsm_x4(qf[mt][kk][0], qf[mt][kk][1], qf[mt][kk][2], qf[mt][kk][3], p);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) o[mt][i][j] = 0.f;
  }
  const float sl2 = a.scale * 1.4426950408889634f;

  int stage = 0, stage_ld = STG - 1;  // stage computed / stage loaded this iteration (rotating)
  for (int it2 = 0; my < n_tiles; ++it2, my += 4) {
    if constexpr (STG >= 2) {
      // the stage loaded now was computed in the previous iteration; the __syncwarp at its end freed it
      if (my + 4 * (STG - 1) < n_tiles) load_kv(my + 4 * (STG - 1), stage_ld);
      cp_async_commit();
      cp_async_wait<STG - 1>();
    } else {
      if (it2 > 0) {
        load_kv(my, 0);
        cp_async_commit();
      }
      cp_async_wait<0>();
    }
    __syncwarp();
    const bf16* kst = ks + size_t(stage) * 2 * kA2Tile * LDK;
    const bf16* vst = kst + kA2Tile * LDK;
    const int kt0 = my * kA2Tile;
#pragma unroll
    for (int mt = 0; mt < MTQ; ++mt) {
      float s[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b0, b1, b2, b3;
          const int mtx = lane >> 3;
          const bf16* p = kst + attn2_kv_off<SW>(np * 16 + (mtx >> 1) * 8 + (lane & 7), kk * 2 + (mtx & 1));
          ldsm_x4(b0, b1, b2, b3, p);
          mma16816(s[np * 2], qf[mt][kk], b0, b1);
          mma16816(s[np * 2 + 1], qf[mt][kk], b2, b3);
        }
      }
      float mx[2] = {m_i[mt][0], m_i[mt][1]};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int hh = u >> 1;
          const int key = kt0 + nt * 8 + 2 * tq + (u & 1);
          const float val = (key < kend[mt][hh]) ? s[nt][u] * sl2 : -1e30f;
          s[nt][u] = val;
          mx[hh] = fmaxf(mx[hh], val);
        }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
      }
      float alpha[2], rs[2] = {0.f, 0.f};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) alpha[hh] = exp2f(m_i[mt][hh] - mx[hh]);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int hh = u >> 1;
          const float p = (s[nt][u] <= -1e29f) ? 0.f : exp2f(s[nt][u] - mx[hh]);
          s[nt][u] = p;
          rs[hh] += p;
        }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        rs[hh] += __shfl_xor_sync(0xffffffffu, rs[hh], 1);
        rs[hh] += __shfl_xor_sync(0xffffffffu, rs[hh], 2);
        l_i[mt][hh] = l_i[mt][hh] * alpha[hh] + rs[hh];
        m_i[mt][hh] = mx[hh];
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        o[mt][nt][0] *= alpha[0]; o[mt][nt][1] *= alpha[0];
        o[mt][nt][2] *= alpha[1]; o[mt][nt][3] *= alpha[1];
      }
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t pa[4];
        pa[0] = pack_bf16(s[2 * kk][0], s[2 * kk][1]);
        pa[1] = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
        pa[2] = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        pa[3] = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
        for (int dp = 0; dp < 4; ++dp) {
          uint32_t b0, b1, b2, b3;
          const int mtx = lane >> 3;
          const bf16* p = vst + attn2_kv_off<SW>(kk * 16 + (mtx & 1) * 8 + (lane & 7), dp * 2 + (mtx >> 1));
          ldsm_x4_t(b0, b1, b2, b3, p);
          mma16816(o[mt][dp * 2], pa, b0, b1);
          mma16816(o[mt][dp * 2 + 1], pa, b2, b3);
        }
      }
    }
    __syncwarp();
    if constexpr (STG >= 2) {
      stage = (stage + 1 == STG) ? 0 : stage + 1;
      stage_ld = (stage_ld + 1 == STG) ? 0 : stage_ld + 1;
    }
  }
  cp_async_wait<0>();
  EST_TR(3);
  if (!a.trig_early) pdl_trigger();
  // Merge the four warps in order through shared memory (each warp writes into its own K/V region).
  float* po = reinterpret_cast<float*>(ks);   // [QG][64]
  float* pml = po + QG * 64;                  // [QG][2]
  static_assert(size_t(QG) * 66 * 4 <= size_t(STG) * 2 * kA2Tile * LDK * 2, "merge buffer must fit the warp region");
#pragma unroll
  for (int mt = 0; mt < MTQ; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int row = mt * 16 + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        po[row * 64 + nt * 8 + 2 * tq] = o[mt][nt][hh * 2];
        po[row * 64 + nt * 8 + 2 * tq + 1] = o[mt][nt][hh * 2 + 1];
      }
      if (tq == 0) { pml[row * 2] = m_i[mt][hh]; pml[row * 2 + 1] = l_i[mt][hh]; }
    }
  __syncthreads();
  constexpr int TPR = 128 / QG;   // threads per row
  constexpr int DPT = 64 / TPR;   // dims per thread
  const int row = tid / TPR, d0 = (tid % TPR) * DPT;
  const int t = qg0 + row;
  if (t >= e.suffix) return;
  const size_t wstride = size_t(STG) * 2 * kA2Tile * LDK / 2;  // floats per warp region
  const float* wbase = reinterpret_cast<const float*>(kvbase);
  float mm = -1e30f;
#pragma unroll
  for (int w = 0; w < 4; ++w) mm = fmaxf(mm, wbase[w * wstride + QG * 64 + row * 2]);
  float L = 0.f, acc[DPT];
#pragma unroll
  for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float* wb = wbase + w * wstride;
    const float wt = exp2f(wb[QG * 64 + row * 2] - mm);
    L += wb[QG * 64 + row * 2 + 1] * wt;
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] += wb[row * 64 + d0 + i] * wt;
  }
  const float inv = 1.0f / fmaxf(L, 1e-30f);
#pragma unroll
  for (int i0 = 0; i0 < DPT; i0 += 8) {
    uint32_t pk[4];
#pragma unroll
    for (int i = 0; i < 8; i += 2) pk[i / 2] = pack_bf16(acc[i0 + i] * inv, acc[i0 + i + 1] * inv);
    *reinterpret_cast<uint4*>(a.out + size_t(e.fr_off + t) * kInner + h * 64 + d0 + i0) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
  EST_TR_END(30);
}

// Tail commit: stage -> StateView.est_tail for rows whose stable end advanced (16-byte copies).
__global__ void est_tail_commit_kernel(const EstRow* __restrict__ erows, const bf16* __restrict__ stage,
                                       bf16* __restrict__ state, int rows_cap, int slots, int width, int steps) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, s = blockIdx.y;
  if (r >= rows_cap || s >= steps) return;
  const EstRow e = erows[r];
  if (e.slot < 0 || e.shift <= 0) return;
  const uint4* src = reinterpret_cast<const uint4*>(stage + (size_t(s) * rows_cap + r) * width);
  uint4* dst = reinterpret_cast<uint4*>(state + (size_t(s) * slots + e.slot) * width);
  const int i = blockIdx.z * blockDim.x + threadIdx.x;
  if (i < width / 8) dst[i] = src[i];
}

}  // namespace kt2n::render::est
