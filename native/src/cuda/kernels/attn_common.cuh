// Shared device helpers for the kt2n attention kernels (lane k3-attn).
// Header-only, internal linkage: safe to include from several .cu files of one static lib.
//
// The arithmetic helpers here ARE the numerics contract documented in attn.h.  Decode and
// prefill call the same helpers with the same operands in the same order, which is what makes
// prefill bitwise equal to decode at every position.  Do not change them (fast math,
// contraction, a different MMA order, skipping a zero term) without bumping
// NUMERICS_REVISION.
//
// Tensor-core facts this relies on (probed on sm_120, tests/gpu/k3-attn/mma_*_probe.cu):
//  * an mma.sync.m16n8k16 output row depends only on that row of A, B and that row of C --
//    not on its position in the tile nor on the other 15 rows (zeros, random, huge, NaN);
//  * an exact-zero sum comes out as +0 (so p = 0 terms never change a result, even its sign).
#pragma once
#include "kt2n/cuda_caps.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#if __has_include("kt2n/kernels/attn.h")
#include "kt2n/kernels/attn.h"
#else
#include "../../../include/kt2n/kernels/attn.h"
#endif

namespace kt2n_attn {

constexpr int HQ = KT2N_ATTN_HEADS_Q;
constexpr int HKV = KT2N_ATTN_HEADS_KV;
constexpr int GQA = HQ / HKV;  // 2 query heads per K/V head
constexpr int D = KT2N_ATTN_HEAD_DIM;
constexpr int CHUNK = KT2N_ATTN_CHUNK;
constexpr int SPLIT_CHUNKS = KT2N_ATTN_SPLIT_CHUNKS;
constexpr int SPLIT_KEYS = KT2N_ATTN_SPLIT_KEYS;
constexpr int SEGS = D / 8;  // 16-byte segments per K/V row
constexpr int NUMERICS_REVISION = 2;
static_assert(HQ == 16 && HKV == 8 && D == 128 && CHUNK == 64, "fixed geometry");
static_assert(SPLIT_CHUNKS >= 1 && SPLIT_CHUNKS <= 64, "split size");

// 1/sqrt(128) rounded to fp32 (= float(128 ** -0.5), production's layer.self_attn.scaling).
constexpr float SCALE = 0.08838834764831845f;

__host__ __device__ __forceinline__ int64_t imin64(int64_t a, int64_t b) { return a < b ? a : b; }
__host__ __device__ __forceinline__ int64_t imax64(int64_t a, int64_t b) { return a > b ? a : b; }

// ---------------------------------------------------------------------------
// Scalar canonical arithmetic.
__device__ __forceinline__ float score_of(float dot) { return __fmul_rn(dot, SCALE); }
__device__ __forceinline__ float prob_of(float s, float m) { return expf(__fsub_rn(s, m)); }

struct FoldCoef {
  float a, e, m;
};
// Split fold F <- fold(F, P): mn = max; a = expf(F.m - mn); e = expf(P.m - mn).
__device__ __forceinline__ FoldCoef fold_coef(float fm, float pm) {
  FoldCoef c;
  c.m = fmaxf(fm, pm);
  c.a = expf(__fsub_rn(fm, c.m));
  c.e = expf(__fsub_rn(pm, c.m));
  return c;
}
__device__ __forceinline__ float fold_val(float f, float a, float p, float e) {
  return __fadd_rn(__fmul_rn(f, a), __fmul_rn(p, e));
}
__device__ __forceinline__ __nv_bfloat16 finish(float o, float l) {
  return __float2bfloat16_rn(__fdiv_rn(o, l));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t finish2(float o0, float o1, float l) {
  __nv_bfloat162 v;
  v.x = finish(o0, l);
  v.y = finish(o1, l);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// Exact 3-way bf16 split of fp32 p: p == hi + mid + lo (each step's subtraction is exact).
struct Split3 {
  float hi, mid, lo;
};
__device__ __forceinline__ Split3 split3(float p) {
  Split3 s;
  s.hi = __bfloat162float(__float2bfloat16_rn(p));
  const float r1 = __fsub_rn(p, s.hi);
  s.mid = __bfloat162float(__float2bfloat16_rn(r1));
  s.lo = __bfloat162float(__float2bfloat16_rn(__fsub_rn(r1, s.mid)));
  return s;
}

// ---------------------------------------------------------------------------
// Tensor-core fragments (mma.sync.m16n8k16, bf16 x bf16 -> fp32).
//   lane = 4*g + t.  A (16x16): a0 = A[g][2t..], a1 = A[g+8][2t..], a2 = A[g][8+2t..],
//   a3 = A[g+8][8+2t..].  B (16x8): b0 = B[2t..][g], b1 = B[8+2t..][g].
//   C (16x8): c0,c1 = C[g][2t, 2t+1]; c2,c3 = C[g+8][2t, 2t+1].
// Query-row convention (decode and prefill): tile row g = (query g, head 0 of the GQA pair),
// row g + 8 = (query g, head 1).
struct FragA {
  uint32_t x[4];
};
struct FragB {
  uint32_t x[2];
};
struct FragC {
  float x[4];
};

__device__ __forceinline__ void mma16816(FragC& d, const FragA& a, const FragB& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]), "r"(b.x[1]));
}
__device__ __forceinline__ void zero(FragC& c) { c.x[0] = c.x[1] = c.x[2] = c.x[3] = 0.f; }

__device__ __forceinline__ void ldsm_x4(uint32_t r[4], const void* smem) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t r[4], const void* smem) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// K / V chunk tiles in shared memory: 64 rows x 16 segments of 16 B, segment XOR-swizzled by
// (row & 7): ldmatrix row groups of 8 consecutive rows hit 8 distinct bank groups.
__device__ __forceinline__ int kswz(int row, int seg) { return row * SEGS + (seg ^ (row & 7)); }

// S tile nt (keys 8nt .. 8nt+7 of the chunk) = Q . K^T, chained over the 8 dim steps in order
// from +0.  qa[ks] = A fragment of dims 16ks .. 16ks+15.
__device__ __forceinline__ void qk_tile(FragC& s, const FragA (&qa)[8], const uint4* sK, int nt, int lane) {
  zero(s);
  const int mi = lane >> 3, r = lane & 7;
#pragma unroll
  for (int ks = 0; ks < 8; ks += 2) {
    uint32_t b[4];
    ldsm_x4(b, &sK[kswz(8 * nt + r, 2 * ks + mi)]);
    const FragB b0 = {{b[0], b[1]}}, b1 = {{b[2], b[3]}};
    mma16816(s, qa[ks], b0);
    mma16816(s, qa[ks + 1], b1);
  }
}

// Two dim steps (ks, ks+1) of S tiles 0..NT-1 = Q . K^T (callers loop ks = 0, 2, 4, 6 after
// zeroing s): per tile the same chain qk_tile computes.
template <int NT>
__device__ __forceinline__ void qk_step2(FragC (&s)[NT], const FragA& a0, const FragA& a1, const uint4* sK, int ks,
                                         int lane) {
  const int mi = lane >> 3, r = lane & 7;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    uint32_t b[4];
    ldsm_x4(b, &sK[kswz(8 * nt + r, ks + mi)]);
    const FragB b0 = {{b[0], b[1]}}, b1 = {{b[2], b[3]}};
    mma16816(s[nt], a0, b0);
    mma16816(s[nt], a1, b1);
  }
}

// P parts for one key step (16 keys): the A fragments of hi / mid / lo.
struct PFrag {
  FragA hi, mid, lo;
};
// Build from the fp32 probabilities this lane holds for that key step, in A-fragment order:
// p[0..1] = row g keys 2t, 2t+1; p[2..3] = row g+8 keys 2t, 2t+1; p[4..7] the same at +8 keys.
__device__ __forceinline__ PFrag make_pfrag(const float (&p)[8]) {
  PFrag f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const Split3 a = split3(p[2 * i]), b = split3(p[2 * i + 1]);
    f.hi.x[i] = pack_bf16x2(a.hi, b.hi);
    f.mid.x[i] = pack_bf16x2(a.mid, b.mid);
    f.lo.x[i] = pack_bf16x2(a.lo, b.lo);
  }
  return f;
}

// One key step ks (keys 16ks .. 16ks+15) of O += P . V for NT dim tiles starting at dt0
// (dims 8*dt0 ..), each tile chained hi, mid, lo.  Callers loop ks = 0..3 in order, so every
// O element is the chain (ks0: hi, mid, lo), (ks1: ...), ... from the value it entered with.
template <int NT>
__device__ __forceinline__ void pv_step(FragC (&o)[NT], const PFrag& pf, const uint4* sV, int ks, int dt0, int lane) {
  static_assert(NT % 2 == 0, "tile pairs");
  const int mi = lane >> 3, r = lane & 7;
#pragma unroll
  for (int i = 0; i < NT; i += 2) {
    uint32_t b[4];
    ldsm_x4_t(b, &sV[kswz(16 * ks + 8 * (mi & 1) + r, dt0 + i + (mi >> 1))]);
    const FragB v0 = {{b[0], b[1]}}, v1 = {{b[2], b[3]}};
    mma16816(o[i], pf.hi, v0);
    mma16816(o[i], pf.mid, v0);
    mma16816(o[i], pf.lo, v0);
    mma16816(o[i + 1], pf.hi, v1);
    mma16816(o[i + 1], pf.mid, v1);
    mma16816(o[i + 1], pf.lo, v1);
  }
}

// The same key step of l += P . 1 (every column of the result holds the row sum).
__device__ __forceinline__ void l_step(FragC& l, const PFrag& pf) {
  const FragB ones = {{0x3F803F80u, 0x3F803F80u}};
  mma16816(l, pf.hi, ones);
  mma16816(l, pf.mid, ones);
  mma16816(l, pf.lo, ones);
}

__device__ __forceinline__ void scale_frag(FragC& c, float a_row_g, float a_row_g8) {
  c.x[0] = __fmul_rn(c.x[0], a_row_g);
  c.x[1] = __fmul_rn(c.x[1], a_row_g);
  c.x[2] = __fmul_rn(c.x[2], a_row_g8);
  c.x[3] = __fmul_rn(c.x[3], a_row_g8);
}

// ---------------------------------------------------------------------------
// cp.async (16 bytes, zero-fill when !valid).  The source address is only dereferenced when
// valid; callers still pass an in-bounds address.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Copy rows [0, nvalid) of a 64-key chunk (row j at g + j*D) into a swizzled smem tile; rows
// >= nvalid are zero-filled.
template <int NT>
__device__ __forceinline__ void load_tile(uint4* sT, const __nv_bfloat16* g, int nvalid, int tid) {
#pragma unroll
  for (int i = tid; i < CHUNK * SEGS; i += NT) {
    const int row = i / SEGS, seg = i % SEGS;
    const bool ok = row < nvalid;
    cp_async16(&sT[kswz(row, seg)], g + (ok ? (static_cast<int64_t>(row) * D + seg * 8) : 0), ok);
  }
}

// Programmatic dependent launch: wait for the previous kernel's results (a no-op when the
// kernel was not launched with KT2N_ATTN_FLAG_PDL).
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

template <typename... KArgs, typename... Args>
inline cudaError_t launch_kernel(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                 bool pdl, Args... args) {
  pdl = kt2n::cuda::pdl_enabled(pdl);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = pdl ? attr : nullptr;
  cfg.numAttrs = pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

}  // namespace kt2n_attn
