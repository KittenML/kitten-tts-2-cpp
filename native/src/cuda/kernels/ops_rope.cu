// q/k RMSNorm over head_dim + RoPE (+ K/V append), plain K/V append, and
// HF-layout cos/sin tables. See include/kt2n/kernels/ops.h.
//
// One warp per (row, head), as production's `_qk_norm_rope_kv` (num_warps=1):
// lane l owns elements 2l, 2l+1 of each half (x1 = [0, 64), x2 = [64, 128)),
// which is Triton's layout for a 64-wide bf16 tensor on 32 lanes
// (sizePerThread = 2, checked in the TTGIR). Sum of squares: in-lane pair,
// butterfly, then (sum(x1^2) + sum(x2^2)) / 128 -- production's expression
// and order. A lane also owns both RoPE partners (i, i + 64): no shuffles.
//
// Latency matters more than throughput here (a decode step launches this 28
// times on 1..16 rows), so: the K/V destination indices are loaded first,
// every load is issued before any arithmetic, and the K/V addressing is
// specialised at compile time (none / affine + cursor / generic).
#include <cstring>

#include "kt2n/kernels/ops.h"
#include "ops_common.cuh"

namespace {

using namespace kt2n_ops_detail;

constexpr int kHeadDim = 128;
// Heads (warps) per CTA. Measured on the RTX PRO 6000 (in-graph, M=1): 1-2 warps
// per CTA are ~0.07 us faster per launch than 8 (spreads the K/V stores over
// SMs); 2 keeps CTAs from being tiny at prefill sizes.
#ifndef KT2N_OPS_QK_WARPS
#define KT2N_OPS_QK_WARPS 2
#endif
constexpr int kWarps = KT2N_OPS_QK_WARPS;
constexpr int kKvWarps = 1;  // plain K/V append: 1 warp per CTA measured fastest (1.08 vs 1.25 us)

// K/V addressing modes.
constexpr int kKvNone = 0;     // no append
constexpr int kKvAffine = 1;   // slot = base + r*step; col = base + r*step (+ cursor[0])
constexpr int kKvGeneric = 2;  // + row_slot / row_col device arrays

struct KvDst {
  __nv_bfloat16* k;
  __nv_bfloat16* v;
  int64_t slot_stride, head_stride, col_stride;
  const int64_t* row_slot;
  int64_t slot_base, slot_step;
  const int64_t* row_col;
  const int64_t* cursor;
  int64_t col_base, col_step, num_cols;
};

KvDst to_dst(const kt2n_ops_kv_target* t) {
  KvDst d{};
  if (t == nullptr) return d;
  d.k = t->k;
  d.v = t->v;
  d.slot_stride = t->slot_stride;
  d.head_stride = t->head_stride;
  d.col_stride = t->col_stride;
  d.row_slot = t->row_slot;
  d.slot_base = t->slot_base;
  d.slot_step = t->slot_step;
  d.row_col = t->row_col;
  d.cursor = t->cursor;
  d.col_base = t->col_base;
  d.col_step = t->col_step;
  d.num_cols = t->num_cols;
  return d;
}

int kv_mode(const kt2n_ops_kv_target* t) {
  if (t == nullptr || (t->k == nullptr && t->v == nullptr)) return kKvNone;
  return (t->row_slot || t->row_col) ? kKvGeneric : kKvAffine;
}

// K/V destination of a row in two halves, so the index loads (cursor,
// row_slot, row_col) are issued together with the data loads and the
// arithmetic that needs them runs last. GPUs issue in order: a data-dependent
// branch right after the cursor load would stall every later load behind it
// (measured +0.13 us per launch), hence the branch-free resolve.
struct KvIdx {
  int64_t slot;  // affine slot, or the row_slot value
  int64_t col;   // affine column part
  int64_t cur;   // cursor[0] (0 without a cursor)
  int64_t rcol;  // row_col[row] (0 without row_col)
};

// Loads only (no arithmetic on loaded values).
template <int KV>
__device__ __forceinline__ KvIdx kv_load(const KvDst& t, int row) {
  KvIdx i;
  i.slot = (KV == kKvGeneric && t.row_slot != nullptr) ? t.row_slot[row]
                                                       : t.slot_base + row * t.slot_step;
  i.col = t.col_base + row * t.col_step;
  i.cur = t.cursor != nullptr ? t.cursor[0] : int64_t{0};
  i.rcol = (KV == kKvGeneric && t.row_col != nullptr) ? t.row_col[row] : int64_t{0};
  return i;
}

// Element offset of (row, head 0, d 0), or -1 when the row is skipped.
__device__ __forceinline__ int64_t kv_resolve(const KvDst& t, const KvIdx& i) {
  const int64_t col = i.col + i.cur + i.rcol;
  const bool ok = (i.slot >= 0) & ((t.num_cols <= 0) | ((col >= 0) & (col < t.num_cols)));
  const int64_t off = i.slot * t.slot_stride + col * t.col_stride;
  return ok ? off : int64_t{-1};
}

struct QkParams {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  int64_t q_stride, k_stride, v_stride;
  const __nv_bfloat16* q_w;
  const __nv_bfloat16* k_w;
  float eps;
  const int64_t* positions;
  const float* inv_freq;
  const __nv_bfloat16* cos_tab;
  const __nv_bfloat16* sin_tab;
  int64_t cs_stride;
  __nv_bfloat16* q_out;
  int64_t q_out_row_stride, q_out_head_stride;
  __nv_bfloat16* k_out;
  int64_t k_out_row_stride, k_out_head_stride;
  KvDst kv;
  int q_heads, kv_heads;
};

// TABLE: read cos/sin from a per-row bf16 table (cos[r*cs_stride + i], i < 64,
// as written by kt2n_ops_embed_rmsnorm / kt2n_ops_rope_cos_sin, or HF's
// [rows, 128] buffers) instead of computing them; same bits either way.
template <bool TABLE, int KV, bool PDL>
__global__ void __launch_bounds__(kWarps * 32) qk_norm_rope_kernel(const QkParams p) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int hh = blockIdx.y * kWarps + (threadIdx.x >> 5);
  const int i0 = 2 * lane;  // this lane's pair in each half
  if (hh >= p.q_heads + p.kv_heads) {  // whole warp idle
    pdl_wait<PDL>();
    pdl_trigger<PDL>();
    return;
  }
  const bool is_q = hh < p.q_heads;
  const int head = is_q ? hh : hh - p.q_heads;

  // Constants.
  const __nv_bfloat16* wp = is_q ? p.q_w : p.k_w;
  const float2 w1 = load_pair(wp + i0);
  const float2 w2 = load_pair(wp + 64 + i0);
  float f0 = 0.f, f1 = 0.f;
  if (!TABLE) {
    f0 = inv_freq_at(p.inv_freq, i0);
    f1 = inv_freq_at(p.inv_freq, i0 + 1);
  }
  pdl_wait<PDL>();

  // All loads, longest dependency chain (index -> address -> store) first.
  KvIdx kvi{0, 0, 0, 0};
  if (KV != kKvNone && !is_q) kvi = kv_load<KV>(p.kv, row);
  const __nv_bfloat16* src = is_q ? p.q + row * p.q_stride + head * kHeadDim
                                  : p.k + row * p.k_stride + head * kHeadDim;
  const float2 x1 = load_pair(src + i0);
  const float2 x2 = load_pair(src + 64 + i0);
  float2 cs_c = make_float2(0.f, 0.f), cs_s = make_float2(0.f, 0.f);
  int64_t pos = 0;
  if (TABLE) {
    cs_c = load_pair(p.cos_tab + row * p.cs_stride + i0);
    cs_s = load_pair(p.sin_tab + row * p.cs_stride + i0);
  } else {
    pos = p.positions[row];
  }
  float2 vv1 = make_float2(0.f, 0.f), vv2 = make_float2(0.f, 0.f);
  if (KV != kKvNone && !is_q && p.kv.v != nullptr) {
    const __nv_bfloat16* vs = p.v + row * p.v_stride + head * kHeadDim;
    vv1 = load_pair(vs + i0);
    vv2 = load_pair(vs + 64 + i0);
  }
  pdl_trigger<PDL>();

  // RMSNorm over head_dim, production order; / 128 == * (1/128) exactly.
  float s1 = __fadd_rn(__fmul_rn(x1.x, x1.x), __fmul_rn(x1.y, x1.y));
  float s2 = __fadd_rn(__fmul_rn(x2.x, x2.x), __fmul_rn(x2.y, x2.y));
  s1 = warp_sum_bfly(s1);
  s2 = warp_sum_bfly(s2);
  const float var = __fmul_rn(__fadd_rn(s1, s2), 1.0f / static_cast<float>(kHeadDim));
  const float inv = rsqrtf(__fadd_rn(var, p.eps));
  const float n1a = rbf(__fmul_rn(rbf(__fmul_rn(x1.x, inv)), w1.x));
  const float n1b = rbf(__fmul_rn(rbf(__fmul_rn(x1.y, inv)), w1.y));
  const float n2a = rbf(__fmul_rn(rbf(__fmul_rn(x2.x, inv)), w2.x));
  const float n2b = rbf(__fmul_rn(rbf(__fmul_rn(x2.y, inv)), w2.y));

  // RoPE (HF rotate_half), cos/sin as Qwen3RotaryEmbedding produces them.
  float c0, sn0, c1, sn1;
  if (TABLE) {
    c0 = cs_c.x;
    c1 = cs_c.y;
    sn0 = cs_s.x;
    sn1 = cs_s.y;
  } else {
    rope_cs(f0, pos, c0, sn0);
    rope_cs(f1, pos, c1, sn1);
  }
  float lo_a, hi_a, lo_b, hi_b;
  rope_pair(n1a, n2a, c0, sn0, lo_a, hi_a);
  rope_pair(n1b, n2b, c1, sn1, lo_b, hi_b);

  if (is_q) {
    __nv_bfloat16* dst = p.q_out + row * p.q_out_row_stride + head * p.q_out_head_stride;
    store_pair(dst + i0, lo_a, lo_b);
    store_pair(dst + 64 + i0, hi_a, hi_b);
    return;
  }
  if (p.k_out != nullptr) {
    __nv_bfloat16* dst = p.k_out + row * p.k_out_row_stride + head * p.k_out_head_stride;
    store_pair(dst + i0, lo_a, lo_b);
    store_pair(dst + 64 + i0, hi_a, hi_b);
  }
  const int64_t kv_off = (KV != kKvNone) ? kv_resolve(p.kv, kvi) : int64_t{-1};
  if (KV != kKvNone && kv_off >= 0) {
    const int64_t off = kv_off + head * p.kv.head_stride;
    if (p.kv.k != nullptr) {
      store_pair(p.kv.k + off + i0, lo_a, lo_b);
      store_pair(p.kv.k + off + 64 + i0, hi_a, hi_b);
    }
    if (p.kv.v != nullptr) {
      // Raw copy: bf16 -> fp32 -> bf16 is exact.
      store_pair(p.kv.v + off + i0, vv1.x, vv1.y);
      store_pair(p.kv.v + off + 64 + i0, vv2.x, vv2.y);
    }
  }
}

// Plain K/V append: one warp per (row, kv head), 4 elements per lane.
template <int KV, bool PDL>
__global__ void __launch_bounds__(kKvWarps * 32)
kv_append_kernel(const __nv_bfloat16* k, int64_t k_stride, const __nv_bfloat16* v,
                 int64_t v_stride, KvDst kv, int kv_heads) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.y * kKvWarps + (threadIdx.x >> 5);
  pdl_wait<PDL>();
  if (head >= kv_heads) {
    pdl_trigger<PDL>();
    return;
  }
  const KvIdx kvi = kv_load<KV>(kv, row);
  const int i0 = 2 * lane;
  __nv_bfloat162 k1{}, k2{}, v1{}, v2{};
  if (kv.k != nullptr) {
    const __nv_bfloat16* ks = k + row * k_stride + head * kHeadDim;
    k1 = *reinterpret_cast<const __nv_bfloat162*>(ks + i0);
    k2 = *reinterpret_cast<const __nv_bfloat162*>(ks + 64 + i0);
  }
  if (kv.v != nullptr) {
    const __nv_bfloat16* vs = v + row * v_stride + head * kHeadDim;
    v1 = *reinterpret_cast<const __nv_bfloat162*>(vs + i0);
    v2 = *reinterpret_cast<const __nv_bfloat162*>(vs + 64 + i0);
  }
  pdl_trigger<PDL>();
  const int64_t base = kv_resolve(kv, kvi);
  if (base < 0) return;
  const int64_t off = base + head * kv.head_stride;
  if (kv.k != nullptr) {
    *reinterpret_cast<__nv_bfloat162*>(kv.k + off + i0) = k1;
    *reinterpret_cast<__nv_bfloat162*>(kv.k + off + 64 + i0) = k2;
  }
  if (kv.v != nullptr) {
    *reinterpret_cast<__nv_bfloat162*>(kv.v + off + i0) = v1;
    *reinterpret_cast<__nv_bfloat162*>(kv.v + off + 64 + i0) = v2;
  }
}

template <bool PDL>
__global__ void __launch_bounds__(64)
rope_cos_sin_kernel(const int64_t* positions, const float* inv_freq, __nv_bfloat16* cos_out,
                    __nv_bfloat16* sin_out, int64_t cs_stride) {
  const int64_t row = blockIdx.x;
  const int i = threadIdx.x;  // 0..63
  const float f = inv_freq_at(inv_freq, i);
  pdl_wait<PDL>();
  const int64_t pos = positions[row];
  pdl_trigger<PDL>();
  float c, s;
  rope_cs(f, pos, c, s);
  cos_out[row * cs_stride + i] = f2bf(c);
  sin_out[row * cs_stride + i] = f2bf(s);
  if (cs_stride >= 128) {  // HF layout: the second half repeats the first
    cos_out[row * cs_stride + 64 + i] = f2bf(c);
    sin_out[row * cs_stride + 64 + i] = f2bf(s);
  }
}

bool pair_ok(const void* p, int64_t stride) { return aligned_to(p, 4) && (stride % 2) == 0; }

bool kv_ok(const kt2n_ops_kv_target* t) {
  if (t == nullptr) return true;
  if (t->k && !aligned_to(t->k, 4)) return false;
  if (t->v && !aligned_to(t->v, 4)) return false;
  return (t->slot_stride % 2) == 0 && (t->head_stride % 2) == 0 && (t->col_stride % 2) == 0 &&
         t->num_cols >= 0;
}

using QkKernel = void (*)(const QkParams);

template <bool PDL>
QkKernel pick_qk_pdl(bool table, int mode) {
  if (table) {
    if (mode == kKvNone) return qk_norm_rope_kernel<true, kKvNone, PDL>;
    if (mode == kKvAffine) return qk_norm_rope_kernel<true, kKvAffine, PDL>;
    return qk_norm_rope_kernel<true, kKvGeneric, PDL>;
  }
  if (mode == kKvNone) return qk_norm_rope_kernel<false, kKvNone, PDL>;
  if (mode == kKvAffine) return qk_norm_rope_kernel<false, kKvAffine, PDL>;
  return qk_norm_rope_kernel<false, kKvGeneric, PDL>;
}

QkKernel pick_qk(bool table, int mode, int pdl) {
  return pdl ? pick_qk_pdl<true>(table, mode) : pick_qk_pdl<false>(table, mode);
}

}  // namespace

extern "C" cudaError_t kt2n_ops_kv_target_slotkv(kt2n_ops_kv_target* t, __nv_bfloat16* store,
                                                 int layer, int layers, int slots, int kv_heads,
                                                 int ctx, int head_dim, const int64_t* cursor) {
  if (!t || !store || layers <= 0 || layer < 0 || layer >= layers || slots <= 0 || kv_heads <= 0 ||
      ctx <= 0 || head_dim != kHeadDim)
    return cudaErrorInvalidValue;
  const int64_t col_stride = head_dim;
  const int64_t head_stride = static_cast<int64_t>(ctx) * col_stride;
  const int64_t slot_stride = static_cast<int64_t>(kv_heads) * head_stride;
  const int64_t layer_stride = static_cast<int64_t>(slots) * slot_stride;
  *t = kt2n_ops_kv_target{};
  t->k = store + static_cast<int64_t>(layer) * layer_stride;
  t->v = store + (static_cast<int64_t>(layers) + layer) * layer_stride;
  t->slot_stride = slot_stride;
  t->head_stride = head_stride;
  t->col_stride = col_stride;
  t->row_slot = nullptr;
  t->slot_base = 0;
  t->slot_step = 1;
  t->row_col = nullptr;
  t->cursor = cursor;
  t->col_base = 0;
  t->col_step = 0;
  t->num_cols = ctx;
  return cudaSuccess;
}

extern "C" cudaError_t kt2n_ops_qk_norm_rope(const __nv_bfloat16* q, int64_t q_stride,
                                             const __nv_bfloat16* k, int64_t k_stride,
                                             const __nv_bfloat16* v, int64_t v_stride,
                                             const __nv_bfloat16* q_w, const __nv_bfloat16* k_w,
                                             float eps, const int64_t* positions,
                                             const float* inv_freq, const __nv_bfloat16* cos_tab,
                                             const __nv_bfloat16* sin_tab, int64_t cs_stride,
                                             __nv_bfloat16* q_out, int64_t q_out_row_stride,
                                             int64_t q_out_head_stride, __nv_bfloat16* k_out,
                                             int64_t k_out_row_stride, int64_t k_out_head_stride,
                                             const kt2n_ops_kv_target* kv, int rows, int q_heads,
                                             int kv_heads, int head_dim, int pdl,
                                             cudaStream_t stream) {
  if (rows < 0 || head_dim != kHeadDim || q_heads <= 0 || kv_heads <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;  // before the buffer checks: an empty call may pass null buffers
  const bool table = cos_tab != nullptr || sin_tab != nullptr;
  if (!q || !k || !q_w || !k_w || !q_out) return cudaErrorInvalidValue;
  if (table && (!cos_tab || !sin_tab || cs_stride < 64 || (cs_stride % 2) != 0 ||
                !aligned_to(cos_tab, 4) || !aligned_to(sin_tab, 4)))
    return cudaErrorInvalidValue;
  if (!table && !positions) return cudaErrorInvalidValue;
  if (!pair_ok(q, q_stride) || !pair_ok(k, k_stride) || !aligned_to(q_w, 4) || !aligned_to(k_w, 4) ||
      !pair_ok(q_out, q_out_row_stride) || (q_out_head_stride % 2) != 0)
    return cudaErrorInvalidValue;
  if (k_out && (!pair_ok(k_out, k_out_row_stride) || (k_out_head_stride % 2) != 0))
    return cudaErrorInvalidValue;
  if (!kv_ok(kv)) return cudaErrorInvalidValue;
  if (kv && kv->v && (!v || !pair_ok(v, v_stride))) return cudaErrorInvalidValue;
  QkParams prm{};
  prm.q = q;
  prm.k = k;
  prm.v = v;
  prm.q_stride = q_stride;
  prm.k_stride = k_stride;
  prm.v_stride = v_stride;
  prm.q_w = q_w;
  prm.k_w = k_w;
  prm.eps = eps;
  prm.positions = positions;
  prm.inv_freq = inv_freq;
  prm.cos_tab = cos_tab;
  prm.sin_tab = sin_tab;
  prm.cs_stride = cs_stride;
  prm.q_out = q_out;
  prm.q_out_row_stride = q_out_row_stride;
  prm.q_out_head_stride = q_out_head_stride;
  prm.k_out = k_out;
  prm.k_out_row_stride = k_out_row_stride;
  prm.k_out_head_stride = k_out_head_stride;
  prm.kv = to_dst(kv);
  prm.q_heads = q_heads;
  prm.kv_heads = kv_heads;
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>((q_heads + kv_heads + kWarps - 1) / kWarps));
  return launch_k(pick_qk(table, kv_mode(kv), pdl), grid, dim3(kWarps * 32), 0, stream, pdl, prm);
}

extern "C" cudaError_t kt2n_ops_kv_append(const __nv_bfloat16* k, int64_t k_stride,
                                          const __nv_bfloat16* v, int64_t v_stride,
                                          const kt2n_ops_kv_target* kv, int rows, int kv_heads,
                                          int head_dim, int pdl, cudaStream_t stream) {
  if (rows < 0 || head_dim != kHeadDim || kv_heads <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!kv || !kv_ok(kv)) return cudaErrorInvalidValue;
  if (kv->k && (!k || !pair_ok(k, k_stride))) return cudaErrorInvalidValue;
  if (kv->v && (!v || !pair_ok(v, v_stride))) return cudaErrorInvalidValue;
  const KvDst dst = to_dst(kv);
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>((kv_heads + kKvWarps - 1) / kKvWarps));
  using KvKernel = void (*)(const __nv_bfloat16*, int64_t, const __nv_bfloat16*, int64_t, KvDst, int);
  const bool generic = kv_mode(kv) == kKvGeneric;
  const KvKernel kern = pdl ? (generic ? kv_append_kernel<kKvGeneric, true> : kv_append_kernel<kKvAffine, true>)
                            : (generic ? kv_append_kernel<kKvGeneric, false> : kv_append_kernel<kKvAffine, false>);
  return launch_k(kern, grid, dim3(kKvWarps * 32), 0, stream, pdl, k, k_stride, v, v_stride, dst,
                  kv_heads);
}

extern "C" cudaError_t kt2n_ops_rope_cos_sin(const int64_t* positions, int rows,
                                             const float* inv_freq, __nv_bfloat16* cos_out,
                                             __nv_bfloat16* sin_out, int64_t cs_stride, int pdl,
                                             cudaStream_t stream) {
  if (rows < 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!positions || !cos_out || !sin_out || cs_stride < 64) return cudaErrorInvalidValue;
  return launch_k(pdl ? rope_cos_sin_kernel<true> : rope_cos_sin_kernel<false>,
                  dim3(static_cast<unsigned>(rows)), dim3(64), 0, stream, pdl, positions, inv_freq,
                  cos_out, sin_out, cs_stride);
}

extern "C" void kt2n_ops_rope_inv_freq_default(float out64[64]) {
  static const uint32_t bits[64] = {KT2N_OPS_INV_FREQ_BITS};
  static_assert(sizeof(bits) == 64 * sizeof(float), "inv_freq table");
  std::memcpy(out64, bits, sizeof(bits));
}
