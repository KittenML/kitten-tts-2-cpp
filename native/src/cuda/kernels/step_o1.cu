// kt2n o1-opt: optimised variant of the k5 decode step (see include/kt2n/kernels/step_o1.h).
//
// Only launch-level changes.  The two kernels in this file are copies of k3's SlotKV decode-attention kernel
// (src/cuda/kernels/attn_decode.cu, decode_kernel<true>) and k4's audio head kernel (samp_head.cu) whose
// arithmetic is left untouched, operation for operation (tested bitwise against
// kt2n_step_forward).  What differs is WHEN things happen relative to griddepcontrol.wait, the CTA dispatch
// order, the PDL trigger, and L2 prefetches of weights (which change no value).
#include "kt2n/kernels/step_o1.h"

#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "attn_common.cuh"
#include "ops_common.cuh"
#include "kt2n/kernels/attn.h"
#include "kt2n/kernels/ops.h"
#include "kt2n/kernels/ops_norm_early.h"
#include "kt2n/kernels/samp.h"
#include "kt2n/kernels/step.h"
#include "kt2n/kernels/tern_proj.h"

namespace kt2n_o1 {

using kt2n_attn::CHUNK;
using kt2n_attn::D;
using kt2n_attn::FragA;
using kt2n_attn::FragC;
using kt2n_attn::GQA;
using kt2n_attn::HKV;
using kt2n_attn::PFrag;
using kt2n_attn::SEGS;
using kt2n_attn::SPLIT_CHUNKS;
using kt2n_attn::SPLIT_KEYS;

// Options per bucket (rows 1..16), measured on the RTX PRO 6000:
// every set gives the same bits; only speed differs.  Index 0 unused.
constexpr unsigned kSmall = KT2N_STEP_O1_ATTN | KT2N_STEP_O1_ATTN_TRIGGER | KT2N_STEP_O1_ATTN_GRID |
                            KT2N_STEP_O1_ATTN_KVPRE | KT2N_STEP_O1_FUSE_QK | KT2N_STEP_O1_L2PF_W |
                            KT2N_STEP_O1_PF_GU;  // 0x848f
constexpr unsigned kLarge = KT2N_STEP_O1_ATTN | KT2N_STEP_O1_ATTN_TRIGGER | KT2N_STEP_O1_ATTN_GRID |
                            KT2N_STEP_O1_ATTN_KVPRE | KT2N_STEP_O1_FUSE_QK | KT2N_STEP_O1_L2PF_LATE |
                            KT2N_STEP_O1_PF_GU | KT2N_STEP_O1_PF_QKV;  // 0x548f
// Per-step profiling: the residual + RMSNorm launches trigger their dependents at start
// (NORM_EARLY1/2, ops_norm_early.cu), so the following projection streams its weights during the o / down
// projection (and, through the PDL cascade, during attention).  Measured per bucket against the sets above
// (6 processes, L2 flushed and warm, contexts 330 / 1000 / 2000): rows 1-3 and 7-10
// gain; rows 4-6 (and the untimed 11-16) keep o1's set.  The PDL head (HEAD + HEAD_PF) helps once the last norm
// triggers early.  Every set was checked for the same bits against k5's step.
constexpr unsigned kNe1 = KT2N_STEP_O1_NORM_EARLY1, kNe2 = KT2N_STEP_O1_NORM_EARLY2;
constexpr unsigned kHeadPdl = KT2N_STEP_O1_HEAD | KT2N_STEP_O1_HEAD_PF;
constexpr unsigned kCore = KT2N_STEP_O1_ATTN | KT2N_STEP_O1_ATTN_TRIGGER | KT2N_STEP_O1_ATTN_GRID |
                           KT2N_STEP_O1_ATTN_KVPRE | KT2N_STEP_O1_FUSE_QK;  // 0x8f
constexpr unsigned kP1 = kSmall | kNe2 | kHeadPdl;         // 0x485af  row 1
constexpr unsigned kP2 = kSmall | kNe1 | kNe2 | kHeadPdl;  // 0x685af  row 2
constexpr unsigned kP3 = kSmall | kNe2;                     // 0x4848f  row 3
constexpr unsigned kP7 = kCore | kNe1 | kNe2;               // 0x6008f  rows 7-10 (the early launches replace the L2 prefetch)
// L3: the table covers the per-row-K/V buckets up to KT2N_STEP_MAX_ROWS_EXT (17..64 start as kLarge; the engine
// sets its measured sets at create).
// L3 sprint 7: buckets 11-16 had never been timed (o1's kLarge by default).  Measured with one voice per row at 600
// keys, quiet and beside a bulk-render stand-in at decode priority 2: kP7 (early-trigger norms instead of the L2 weight prefetch) is faster at 11-16 rows, most
// under load, as prof-step found for 7-10.  Same bits (every set does).
#ifndef KT2N_STEP_O1_B11_16
#define KT2N_STEP_O1_B11_16 kP7
#endif
unsigned g_bucket_options[KT2N_STEP_MAX_ROWS_EXT + 1] = {0,      kP1,    kP2,    kP3,    kLarge, kLarge,
                                                         kLarge, kP7,    kP7,    kP7,    kP7,    KT2N_STEP_O1_B11_16,
                                                         KT2N_STEP_O1_B11_16, KT2N_STEP_O1_B11_16, KT2N_STEP_O1_B11_16,
                                                         KT2N_STEP_O1_B11_16, KT2N_STEP_O1_B11_16};
struct BucketTail {  // rows 17..64: kLarge until set
  BucketTail() {
    for (int r = KT2N_STEP_MAX_ROWS + 1; r <= KT2N_STEP_MAX_ROWS_EXT; ++r) g_bucket_options[r] = kLarge;
    // L3 sprint 3 (nsys per-step medians): at 17-20 rows the PDL
    // head reads its 27 MB once (one 24-row launch) instead of twice: head 40.4 -> 25.2 us per step, the other
    // families unchanged.  Neutral at 24 rows, slower at 32 (24 + 8 rows vs 16 + 16), so only 17-20.
    for (int r = KT2N_STEP_MAX_ROWS + 1; r <= 20 && r <= KT2N_STEP_MAX_ROWS_EXT; ++r) g_bucket_options[r] = kLarge | kHeadPdl;
  }
} g_bucket_tail;
static_assert(kSmall == 0x848fu && kLarge == 0x548fu, "o1's measured sets");
static_assert(kP1 == 0x485afu && kP2 == 0x685afu && kP3 == 0x4848fu && kP7 == 0x6008fu, "prof-step's measured sets");
unsigned g_options = KT2N_STEP_O1_AUTO;
unsigned long long g_kv_next_budget = 48ull << 20;  // KT2N_STEP_O1_KV_NEXT: one layer's prefetched K + V

// Decode launch shapes {nt, wn} per (rows, family) used inside the step (0 = k1's own table).  A shape never
// changes a result bit (k1 contract: the per-row arithmetic is fixed per family); only speed differs.
// Measured in place (3 captures per shape): only the qkv projection at 3-5 rows
// gains (-3..-8%) from fewer, wider CTAs next to the fused attention's early K/V loads.
// L3: a third entry mt (k1's M-tiling above 16 rows; 0 = k1's choice) for the per-row-K/V buckets 17..64.
// L3 sprint 7: qkv {4, 2} at 9-10 rows (k1's table has {2, 2} there and {4, 2} at 8 and 11-16): the RC1 env runs
// 17-20-row steps as two 9-10-row branches, where {4, 2} measured faster.
#ifndef KT2N_STEP_O1_QKV910
#define KT2N_STEP_O1_QKV910 1
#endif
#if KT2N_STEP_O1_QKV910
#define KT2N_QKV910 {{4, 2}}
#else
#define KT2N_QKV910 {}
#endif
short g_shape[KT2N_STEP_MAX_ROWS_EXT + 1][4][3] = {
    {}, {}, {}, {{2, 2}}, {{4, 1}}, {{2, 2}}, {}, {}, {}, KT2N_QKV910, KT2N_QKV910, {}, {}, {}, {}, {}, {}};
#undef KT2N_QKV910

// lm-proj-wide (11 Oct, DEFAULT OFF, execution only): with env KT2N_LM_PROJ_WIDE=1 the 17..32-row decode projections
// run on kt2n_tern_proj_wide (cluster split-K, activations staged once per CTA in shared memory).  Each row's
// arithmetic is the family's, so every logit is the same bits as with the k1 kernels (checked by
// memcmp).  KT2N_LM_PROJ_WIDE_FAMS = family bit mask (1 qkv, 2 o, 4 gate_up, 8 down; default 15),
// KT2N_LM_PROJ_WIDE_SHAPE = one digit per family in that order, 0 = the family's default shape (e.g. "0000").
// Read once per process; unset or 0 = the k1 kernels, exactly as before.
struct WideCfg {
  bool on = false;
  unsigned fams = 15u;
  int shape[4] = {0, 0, 0, 0};
  WideCfg() {
    const char* v = getenv("KT2N_LM_PROJ_WIDE");
    on = v && *v && atoi(v) != 0;
    if (!on) return;
    if (const char* f = getenv("KT2N_LM_PROJ_WIDE_FAMS"); f && *f) fams = static_cast<unsigned>(atoi(f)) & 15u;
    if (const char* sh = getenv("KT2N_LM_PROJ_WIDE_SHAPE"); sh && *sh)
      for (int i = 0; i < 4 && sh[i]; ++i) shape[i] = (sh[i] >= '0' && sh[i] <= '9') ? sh[i] - '0' : 0;
    fprintf(stderr, "kt2n: KT2N_LM_PROJ_WIDE on (families mask %u, shapes %d%d%d%d) for 17..32-row decode steps\n", fams,
            shape[0], shape[1], shape[2], shape[3]);
  }
};
const WideCfg& wide_cfg() {
  static const WideCfg c;
  return c;
}

int proj(int fam, const void* x, int64_t ldx, int m, const void* codes, const void* scales, int n, int k, void* y,
         int64_t ldy, int epi, const void* r, int64_t ldr, unsigned flags, cudaStream_t s) {
  if (m > 16 && m <= 32) {
    const WideCfg& w = wide_cfg();
    if (w.on && ((w.fams >> fam) & 1u)) {
      const int rc = kt2n_tern_proj_wide(fam, x, ldx, m, codes, scales, n, k, y, ldy, epi, r, ldr, flags, w.shape[fam], s);
      if (rc != KT2N_TERN_EINVAL) return rc;
      // EINVAL: not available here (nothing was launched); the k1 kernel below runs instead.  Said once per family.
      static bool told[4] = {false, false, false, false};
      if (!told[fam]) {
        told[fam] = true;
        fprintf(stderr, "kt2n: KT2N_LM_PROJ_WIDE: family %d at %d rows not available, using the k1 kernel\n", fam, m);
      }
    }
  }
  const int nt = (m >= 1 && m <= KT2N_STEP_MAX_ROWS_EXT) ? g_shape[m][fam][0] : 0;
  const int wn = (m >= 1 && m <= KT2N_STEP_MAX_ROWS_EXT) ? g_shape[m][fam][1] : 0;
  const int mt = (m > 16 && m <= KT2N_STEP_MAX_ROWS_EXT) ? g_shape[m][fam][2] : 0;
  if (nt == 0 && wn == 0 && mt == 0)
    return kt2n_tern_proj(fam, x, ldx, m, codes, scales, n, k, y, ldy, epi, r, ldr, flags, s);
  return kt2n_tern_proj_ex(fam, x, ldx, m, codes, scales, n, k, y, ldy, epi, r, ldr, flags, nt, wn, mt, s);
}

unsigned options_for(int rows) {
  unsigned options = g_options != KT2N_STEP_O1_AUTO
                         ? g_options
                         : ((rows >= 1 && rows <= KT2N_STEP_MAX_ROWS_EXT) ? g_bucket_options[rows] : 0u);
#if defined(KT2N_ATTN_PRODUCTION_ORDER) && KT2N_ATTN_PRODUCTION_ORDER
  // This is also the public query: report exactly the paths that can run.
  options &= ~(KT2N_STEP_O1_ATTN | KT2N_STEP_O1_FUSE_QK | KT2N_STEP_O1_ATTN_TRIGGER |
               KT2N_STEP_O1_ATTN_GRID | KT2N_STEP_O1_ATTN_KVPRE | KT2N_STEP_O1_ATTN_DEEP |
               KT2N_STEP_O1_ATTN_DEEP_POST | KT2N_STEP_O1_L2PF | KT2N_STEP_O1_L2PF_HEAD |
               KT2N_STEP_O1_L2PF_LATE | KT2N_STEP_O1_L2PF_W | KT2N_STEP_O1_PF_MASK);
#endif
#if defined(KT2N_HEAD_PRODUCTION_ORDER) && KT2N_HEAD_PRODUCTION_ORDER
  options &= ~(KT2N_STEP_O1_HEAD | KT2N_STEP_O1_HEAD_PF);
#endif
  return options;
}

// ------------------------------------------------------------------------------------------------ L2 prefetch
constexpr int kPfMax = 10;
struct PfList {
  const char* p[kPfMax];
  unsigned long long n[kPfMax];  // bytes, multiple of 16; p 16-byte aligned
  int count;
};

__device__ __forceinline__ void l2_bulk_prefetch(const void* p, unsigned bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#endif
}

// CTA `cta` of `ncta` prefetches its share of every region (thread r issues region r).
__device__ __forceinline__ void l2_prefetch_share(const PfList& pf, int cta, int ncta, int tid) {
  if (tid >= pf.count) return;
  const unsigned long long units = pf.n[tid] >> 4;
  const unsigned long long u0 = units * cta / ncta, u1 = units * (cta + 1) / ncta;
  if (u1 > u0) l2_bulk_prefetch(pf.p[tid] + (u0 << 4), static_cast<unsigned>((u1 - u0) << 4));
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// ------------------------------------------------------------------------------------------------ attention
// k3's decode_kernel<true> (SlotKV), arithmetic copied verbatim.  Differences (opts):
//   ATTN_TRIGGER: launch_dependents at start; ATTN_GRID: blockIdx = (kv head, row, split);
//   ATTN_KVPRE: chunk-0 K/V cp.async before griddepcontrol.wait when chunk 0 cannot hold this step's key;
//   L2PF: L2 prefetch share of `pf`.
// Reading positions / cursor before the wait is safe: they are inputs written before the step (graph) starts and
// no kernel of the step writes them.  K/V columns other than *cursor are not written during the step.
constexpr int DEC_THREADS = 128;
constexpr int PSTR = 132;
constexpr size_t COUNTER_BYTES = 4096;
constexpr int kDeepSmem = 3 * KT2N_ATTN_CHUNK * (KT2N_ATTN_HEAD_DIM / 8) * 16;  // three 16 KB K/V tiles
// SPLIT64: rows up to this many keys use 64-key splits (A/B measurement on 10 Oct: 1 row -4.5% at
// 330 keys, +6.9% at 2,000; 2 rows -8.2% at 330, +2.4% at 1,000).
constexpr int64_t kSplit64MaxKeys = 640;

__host__ __device__ inline int splits_for(int64_t keys, int split_keys = SPLIT_KEYS) {
  return static_cast<int>((keys + split_keys - 1) / split_keys);
}

__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }

// FUSE: this kernel also does k2's q/k RMSNorm + RoPE + K/V append (kt2n_ops_qk_norm_rope, TABLE mode, affine
// SlotKV target: slot = row, column = *cursor), with k2's arithmetic (ops_common.cuh helpers, one warp per head,
// lane l owns pairs 2l and 64 + 2l).  Warps 0/1 normalise the CTA's two query heads into shared memory; in the
// CTA whose split holds the step's key (the last split), warp 2 normalises + ropes the key and warp 3 copies the
// value: both are written to the cache column (for later steps) and straight into this CTA's shared K/V tile row
// (that row is left out of the tile's cp.async, so nothing races).  `q` is then the qkv projection output.
struct QkFuse {
  const __nv_bfloat16* qkv;  // [rows, ld] q (16 heads) | k (8) | v (8)
  int64_t ld;
  const __nv_bfloat16* q_w;
  const __nv_bfloat16* k_w;
  float eps;
  const __nv_bfloat16* cos_t;  // [rows, cs]
  const __nv_bfloat16* sin_t;
  int64_t cs;
  int64_t num_cols;  // K/V columns: the append is skipped unless 0 <= *cursor < num_cols (k2's rule)
  // L3 shared voice prefix (ROW_KV, io->voice_kv): a split whose keys all lie in the slot's voice prefix reads them
  // from the voice store ([2][layers][8][p][128] at voff[slot]) instead of the slot's copy.  Same values, so the
  // same results; rows of one voice then share those reads through L2.  vkv == nullptr: off.
  const __nv_bfloat16* vkv;
  const int64_t* voff;  // [rows] element offset of the bucket row's voice K/V, or -1
  const int32_t* vlen;  // [rows] the voice's prefix length p
  int layer;
  int layers;
  // L3 sprint 5 (KT2N_STEP_O1_KV_NEXT): the next layer's slot K / V caches (nullptr: no prefetch) and the key cap
  // (a CTA whose first key is at or beyond pf_keys prefetches nothing).
  const __nv_bfloat16* nk = nullptr;
  const __nv_bfloat16* nv = nullptr;
  int64_t pf_keys = 0;
};

// k2 qk_norm_rope_kernel arithmetic for one head (TABLE mode); returns the bf16x2 words of pairs i0 and 64 + i0.
__device__ __forceinline__ void qk_head(const __nv_bfloat16* src, const __nv_bfloat16* wp, float eps,
                                        const __nv_bfloat16* cos_row, const __nv_bfloat16* sin_row, int lane,
                                        uint32_t& lo_word, uint32_t& hi_word) {
  using namespace kt2n_ops_detail;
  const int i0 = 2 * lane;
  const float2 w1 = load_pair(wp + i0);
  const float2 w2 = load_pair(wp + 64 + i0);
  const float2 x1 = load_pair(src + i0);
  const float2 x2 = load_pair(src + 64 + i0);
  const float2 cs_c = load_pair(cos_row + i0);
  const float2 cs_s = load_pair(sin_row + i0);
  float s1 = __fadd_rn(__fmul_rn(x1.x, x1.x), __fmul_rn(x1.y, x1.y));
  float s2 = __fadd_rn(__fmul_rn(x2.x, x2.x), __fmul_rn(x2.y, x2.y));
  s1 = warp_sum_bfly(s1);
  s2 = warp_sum_bfly(s2);
  const float var = __fmul_rn(__fadd_rn(s1, s2), 1.0f / 128.0f);
  const float inv = rsqrtf(__fadd_rn(var, eps));
  const float n1a = rbf(__fmul_rn(rbf(__fmul_rn(x1.x, inv)), w1.x));
  const float n1b = rbf(__fmul_rn(rbf(__fmul_rn(x1.y, inv)), w1.y));
  const float n2a = rbf(__fmul_rn(rbf(__fmul_rn(x2.x, inv)), w2.x));
  const float n2b = rbf(__fmul_rn(rbf(__fmul_rn(x2.y, inv)), w2.y));
  float lo_a, hi_a, lo_b, hi_b;
  rope_pair(n1a, n2a, cs_c.x, cs_s.x, lo_a, hi_a);
  rope_pair(n1b, n2b, cs_c.y, cs_s.y, lo_b, hi_b);
  const __nv_bfloat162 lo = __floats2bfloat162_rn(lo_a, lo_b);  // == store_pair's rounding
  const __nv_bfloat162 hi = __floats2bfloat162_rn(hi_a, hi_b);
  lo_word = *reinterpret_cast<const uint32_t*>(&lo);
  hi_word = *reinterpret_cast<const uint32_t*>(&hi);
}

// 32-bit word holding elements e, e+1 (e even) of row `row` of a swizzled K/V tile.
__device__ __forceinline__ uint32_t* tile_word(uint4* tile, int row, int e) {
  return reinterpret_cast<uint32_t*>(&tile[kt2n_attn::kswz(row, e >> 3)]) + ((e & 7) >> 1);
}

// load_tile without row `skip` (neither copied nor zero-filled: the caller stores it).
template <int NT>
__device__ __forceinline__ void load_tile_skip(uint4* sT, const __nv_bfloat16* g, int nvalid, int skip, int tid) {
  using namespace kt2n_attn;
#pragma unroll
  for (int i = tid; i < CHUNK * SEGS; i += NT) {
    const int row = i / SEGS, seg = i % SEGS;
    if (row == skip) continue;
    const bool ok = row < nvalid;
    cp_async16(&sT[kswz(row, seg)], g + (ok ? (static_cast<int64_t>(row) * D + seg * 8) : 0), ok);
  }
}

// load_tile from two sources: rows < rb from `a` (the voice store), the others from `g` (the row's slot); row `skip`
// (-1 = none) is left out as in load_tile_skip.  Same bytes as load_tile on the slot (the slot's prefix is a copy).
template <int NT>
__device__ __forceinline__ void load_tile2(uint4* sT, const __nv_bfloat16* a, const __nv_bfloat16* g, int rb,
                                           int nvalid, int skip, int tid) {
  using namespace kt2n_attn;
#pragma unroll
  for (int i = tid; i < CHUNK * SEGS; i += NT) {
    const int row = i / SEGS, seg = i % SEGS;
    if (row == skip) continue;
    const bool ok = row < nvalid;
    const __nv_bfloat16* src = row < rb ? a : g;
    cp_async16(&sT[kswz(row, seg)], src + (ok ? (static_cast<int64_t>(row) * D + seg * 8) : 0), ok);
  }
}

template <bool FUSE, bool DEEP>
__global__ void __launch_bounds__(DEC_THREADS)
    attn_kernel(const __nv_bfloat16* __restrict__ q, int64_t q_row_stride, __nv_bfloat16* kc_,
                __nv_bfloat16* vc_, int64_t slot_stride, int64_t head_stride,
                const int64_t* __restrict__ positions, const int64_t* __restrict__ cursor, int grid_splits,
                __nv_bfloat16* __restrict__ out, int64_t out_row_stride, int* __restrict__ counters,
                float* __restrict__ partials, unsigned opts, const PfList pf, const QkFuse fz,
                const int32_t* __restrict__ row_slot, int row_kv, int split64) {
  // split64 (KT2N_STEP_O1_SPLIT64, never with DEEP): a row with <= kSplit64MaxKeys keys uses one 64-key chunk per
  // split (more CTAs, no serial second tile); longer rows keep SPLIT_CHUNKS (fewer partials to merge).  The grid
  // and the partials are sized for 64-key splits.
  using namespace kt2n_attn;
  const __nv_bfloat16* kc = kc_;
  const __nv_bfloat16* vc = vc_;
  __shared__ __align__(16) __nv_bfloat16 sQ[FUSE ? GQA : 1][FUSE ? D : 2];
  static_assert(FUSE || !DEEP, "DEEP needs FUSE");
  __shared__ __align__(16) uint4 sKV[DEEP ? 1 : 2 * CHUNK * SEGS];  // K | V (swizzled); partials in the merge
  extern __shared__ __align__(16) uint4 dyn[];                       // DEEP: three K/V tiles
  __shared__ __align__(16) float sP[GQA][CHUNK];         // probabilities of head 0 / head 1
  __shared__ float sMax[DEC_THREADS / 32][GQA];
  __shared__ int sLast;

  uint4* sK = DEEP ? dyn : sKV;
  uint4* sV = (DEEP ? dyn : sKV) + CHUNK * SEGS;
  const bool gridx = (opts & KT2N_STEP_O1_ATTN_GRID) != 0;
  const int split = gridx ? blockIdx.z : blockIdx.x;
  const int kvh = gridx ? blockIdx.x : blockIdx.y;
  const int b = gridx ? blockIdx.y : blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  if (opts & KT2N_STEP_O1_ATTN_TRIGGER) pdl_trigger();
  if (opts & KT2N_STEP_O1_L2PF) {
    const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    l2_prefetch_share(pf, cta, gridDim.x * gridDim.y * gridDim.z, tid);
  }

  // share_voice_kv: the row's voice offset / prefix length (per bucket row), requested beside the cursor loads.
  int64_t vrow_off = -1, vrow_len = 0;
  if (FUSE && fz.vkv != nullptr && row_kv) {
    vrow_off = fz.voff[b];
    vrow_len = fz.vlen[b];
  }
  // row_kv (KT2N_STEP_FLAG_ROW_KV): per-row cursor and K/V slot (row_slot NULL = slot b).
  const int64_t cur_col = row_kv ? cursor[b] : *cursor;
  int64_t n, col0, slot;
  {
    const int64_t cur = cur_col;
    const int64_t pos = imin64(positions[b], cur);
    n = pos + 1;
    col0 = cur - pos;
    slot = (row_kv && row_slot != nullptr) ? static_cast<int64_t>(row_slot[b]) : static_cast<int64_t>(b);
  }
  __nv_bfloat16* orow = out + b * out_row_stride + kvh * GQA * D;
  // FUSE: the step's key/value words of this lane (warp 2: K, warp 3: V), computed after the wait.
  uint32_t kv_lo = 0u, kv_hi = 0u;
  auto fused_kv = [&](bool to_cache) {
    if (warp == 2) {
      qk_head(fz.qkv + b * fz.ld + (GQA * 8 + kvh) * D, fz.k_w, fz.eps, fz.cos_t + b * fz.cs, fz.sin_t + b * fz.cs,
              lane, kv_lo, kv_hi);
    } else if (warp == 3) {
      const __nv_bfloat16* vs = fz.qkv + b * fz.ld + (GQA * 8 + 8 + kvh) * D;
      kv_lo = *reinterpret_cast<const uint32_t*>(vs + 2 * lane);
      kv_hi = *reinterpret_cast<const uint32_t*>(vs + 64 + 2 * lane);
    }
    if (to_cache && warp >= 2 && cur_col >= 0 && (fz.num_cols <= 0 || cur_col < fz.num_cols)) {
      __nv_bfloat16* dst = (warp == 2 ? kc_ : vc_) + slot * slot_stride + kvh * head_stride + cur_col * D;
      *reinterpret_cast<uint32_t*>(dst + 2 * lane) = kv_lo;
      *reinterpret_cast<uint32_t*>(dst + 64 + 2 * lane) = kv_hi;
    }
  };
  const int sc = (split64 && n <= kSplit64MaxKeys) ? 1 : SPLIT_CHUNKS;
  const int sk = sc * CHUNK;
  if (n <= 0 || splits_for(n, sk) > grid_splits) {
    pdl_wait();
    if (FUSE && split == 0) fused_kv(true);
    if (split == 0) {
      const float fill = n <= 0 ? 0.f : __int_as_float(0x7fc00000);
      for (int i = tid; i < GQA * D; i += DEC_THREADS) orow[i] = __float2bfloat16_rn(fill);
    }
    return;
  }
  const int nsplit = splits_for(n, sk);
  if (split >= nsplit) return;  // nothing to read or write (k3: the same return, after the wait)

  const int64_t key0 = static_cast<int64_t>(split) * sk;
  const int64_t left = n - key0;
  const int nch = static_cast<int>(imin64(sc, (left + CHUNK - 1) / CHUNK));
  const int64_t base = slot * slot_stride + kvh * head_stride + (col0 + key0) * D;
  const __nv_bfloat16* kb = kc + base;
  const __nv_bfloat16* vb = vc + base;
  // share_voice_kv: keys below the voice prefix length p come from the voice store, row by row (kbv / vbv point at
  // this split's first key there; pb = prefix keys from key0 on).  The step's own key is never prefix.
  const __nv_bfloat16* kbv = nullptr;
  const __nv_bfloat16* vbv = nullptr;
  int64_t pb = 0;
  if (FUSE && fz.vkv != nullptr && row_kv && col0 == 0) {
    const int64_t vo = vrow_off;  // per bucket row: independent of row_slot, so no dependent load chain
    if (vo >= 0) {
      const int64_t p = vrow_len;
      if (key0 < p) {
        pb = p - key0;
        kbv = fz.vkv + vo + ((static_cast<int64_t>(fz.layer) * HKV + kvh) * p + key0) * D;
        vbv = fz.vkv + vo + ((static_cast<int64_t>(fz.layers + fz.layer) * HKV + kvh) * p + key0) * D;
      }
    }
  }
  auto nvalid = [&](int c) { return static_cast<int>(imin64(CHUNK, left - static_cast<int64_t>(c) * CHUNK)); };
  auto koff = [&](int c) { return static_cast<int64_t>(c) * CHUNK * D; };

  // FUSE: the step's key is row `rn` of chunk `cn` of the last split; that row is stored from registers.
  const bool last = split == nsplit - 1;
  const int cn = FUSE && last ? static_cast<int>((left - 1) / CHUNK) : -1;
  const int rn = FUSE && last ? static_cast<int>((left - 1) % CHUNK) : -1;
  // One K (isk) or V tile of chunk c into dst, leaving out the step's own row when skip.
  auto tile = [&](uint4* dst, bool isk, int c, bool skip) {
    const __nv_bfloat16* g = (isk ? kb : vb) + koff(c);
    const int64_t r = pb - static_cast<int64_t>(c) * CHUNK;
    const int rb = (kbv == nullptr || r <= 0) ? 0 : static_cast<int>(imin64(r, CHUNK));
    if (rb > 0) {
      load_tile2<DEC_THREADS>(dst, (isk ? kbv : vbv) + koff(c), g, rb, nvalid(c), skip ? rn : -1, tid);
    } else if (skip) {
      load_tile_skip<DEC_THREADS>(dst, g, nvalid(c), rn, tid);
    } else {
      load_tile<DEC_THREADS>(dst, g, nvalid(c), tid);
    }
  };
  float m0 = 0.f, m1 = 0.f;
  FragC o[4], lf;
  if constexpr (DEEP) {
    // ---- DEEP (needs FUSE, SPLIT_CHUNKS == 2): every K/V tile of the split is requested before
    // griddepcontrol.wait into three dynamic-smem tiles (K0 -> T0, V0 -> T1, K1 -> T2; V1 -> T0 once Q.K0 has
    // read it).  The step's key/value row (chunk cn, row rn) is never loaded: it is stored from registers after
    // the wait.  Same tiles, same arithmetic, same order as k3's loop below.
    static_assert(SPLIT_CHUNKS == 2, "DEEP is written for two chunks per split");
    uint4* T0 = dyn;
    uint4* T1 = dyn + CHUNK * SEGS;
    uint4* T2 = dyn + 2 * CHUNK * SEGS;
    auto ld = [&](uint4* dst, bool isk, int c) { tile(dst, isk, c, cn == c); };
    const bool post = (opts & KT2N_STEP_O1_ATTN_DEEP_POST) != 0;  // issue after the wait (no DRAM competition
    if (post) pdl_wait();                                          // with the predecessor's weight stream)
    ld(T0, true, 0);
    cp_async_commit();  // G1
    ld(T1, false, 0);
    cp_async_commit();  // G2
    if (nch > 1) ld(T2, true, 1);
    cp_async_commit();  // G3 (empty when nch == 1)
    if (!post) pdl_wait();
    if ((opts & KT2N_STEP_O1_L2PF_W) && b == 0) l2_prefetch_share(pf, split * HKV + kvh, nsplit * HKV, tid);
    if (warp < GQA) {
      uint32_t lo, hi;
      qk_head(fz.qkv + b * fz.ld + (kvh * GQA + warp) * D, fz.q_w, fz.eps, fz.cos_t + b * fz.cs,
              fz.sin_t + b * fz.cs, lane, lo, hi);
      *reinterpret_cast<uint32_t*>(&sQ[warp][2 * lane]) = lo;
      *reinterpret_cast<uint32_t*>(&sQ[warp][64 + 2 * lane]) = hi;
    } else if (last) {
      fused_kv(true);
      uint4* tile = cn == 0 ? (warp == 2 ? T0 : T1) : (warp == 2 ? T2 : nullptr);  // V1 row: after Q.K0
      if (tile != nullptr) {
        *tile_word(tile, rn, 2 * lane) = kv_lo;
        *tile_word(tile, rn, 64 + 2 * lane) = kv_hi;
      }
    }
    __syncthreads();  // sQ (and the stored rows) visible
    FragA qa[8];
    {
      const __nv_bfloat16* q0 = &sQ[0][0];
      const __nv_bfloat16* q1 = &sQ[GQA - 1][0];
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const bool on = g == 0;
        qa[ks].x[0] = on ? ld32(q0 + 16 * ks + 2 * t) : 0u;
        qa[ks].x[1] = on ? ld32(q1 + 16 * ks + 2 * t) : 0u;
        qa[ks].x[2] = on ? ld32(q0 + 16 * ks + 8 + 2 * t) : 0u;
        qa[ks].x[3] = on ? ld32(q1 + 16 * ks + 8 + 2 * t) : 0u;
      }
    }
    for (int c = 0; c < nch; ++c) {
      uint4* sK = c == 0 ? T0 : T2;
      uint4* sV = c == 0 ? T1 : T0;
      const int nv = nvalid(c);
      if (c == 0) cp_async_wait<2>(); else cp_async_wait<1>();
      __syncthreads();  // K(c) visible
      FragC s0, s1;
      qk_tile(s0, qa, sK, 2 * warp, lane);
      qk_tile(s1, qa, sK, 2 * warp + 1, lane);
      const int j0 = 16 * warp + 2 * t, j1 = j0 + 8;
      float sc[8] = {s0.x[0], s0.x[1], s0.x[2], s0.x[3], s1.x[0], s1.x[1], s1.x[2], s1.x[3]};
      const bool v[8] = {j0 < nv, j0 + 1 < nv, j0 < nv, j0 + 1 < nv, j1 < nv, j1 + 1 < nv, j1 < nv, j1 + 1 < nv};
#pragma unroll
      for (int i = 0; i < 8; ++i) sc[i] = v[i] ? score_of(sc[i]) : -INFINITY;
      const float w0 = quad_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[4], sc[5])));
      const float w1 = quad_max(fmaxf(fmaxf(sc[2], sc[3]), fmaxf(sc[6], sc[7])));
      if (lane == 0) {
        sMax[warp][0] = w0;
        sMax[warp][1] = w1;
      }
      __syncthreads();  // every Q.K read of sK done; sMax visible
      if (c == 0) {
        if (nch > 1) {
          ld(T0, false, 1);  // V1 into T0 (K0 is consumed)
          if (cn == 1 && warp == 3) {
            *tile_word(T0, rn, 2 * lane) = kv_lo;
            *tile_word(T0, rn, 64 + 2 * lane) = kv_hi;
          }
        }
        cp_async_commit();  // G4 (empty when nch == 1)
      }
      const float mc0 = fmaxf(fmaxf(sMax[0][0], sMax[1][0]), fmaxf(sMax[2][0], sMax[3][0]));
      const float mc1 = fmaxf(fmaxf(sMax[0][1], sMax[1][1]), fmaxf(sMax[2][1], sMax[3][1]));
      float a0 = 1.f, a1 = 1.f;
      if (c == 0) {
        m0 = mc0;
        m1 = mc1;
      } else {
        const float n0 = fmaxf(m0, mc0), n1 = fmaxf(m1, mc1);
        a0 = expf(__fsub_rn(m0, n0));
        a1 = expf(__fsub_rn(m1, n1));
        m0 = n0;
        m1 = n1;
      }
      if (g == 0) {
        sP[0][j0] = v[0] ? prob_of(sc[0], m0) : 0.f;
        sP[0][j0 + 1] = v[1] ? prob_of(sc[1], m0) : 0.f;
        sP[1][j0] = v[2] ? prob_of(sc[2], m1) : 0.f;
        sP[1][j0 + 1] = v[3] ? prob_of(sc[3], m1) : 0.f;
        sP[0][j1] = v[4] ? prob_of(sc[4], m0) : 0.f;
        sP[0][j1 + 1] = v[5] ? prob_of(sc[5], m0) : 0.f;
        sP[1][j1] = v[6] ? prob_of(sc[6], m1) : 0.f;
        sP[1][j1 + 1] = v[7] ? prob_of(sc[7], m1) : 0.f;
      }
      if (c == 0) cp_async_wait<2>(); else cp_async_wait<0>();
      __syncthreads();  // V(c) and sP visible
      if (c == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) zero(o[i]);
        zero(lf);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) scale_frag(o[i], a0, a1);
        scale_frag(lf, a0, a1);
      }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        float p[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (g == 0) {
          const int k0 = 16 * ks + 2 * t, k1 = k0 + 8;
          p[0] = sP[0][k0];
          p[1] = sP[0][k0 + 1];
          p[2] = sP[1][k0];
          p[3] = sP[1][k0 + 1];
          p[4] = sP[0][k1];
          p[5] = sP[0][k1 + 1];
          p[6] = sP[1][k1];
          p[7] = sP[1][k1 + 1];
        }
        const PFrag pf2 = make_pfrag(p);
        pv_step<4>(o, pf2, sV, ks, 4 * warp, lane);
        l_step(lf, pf2);
      }
      __syncthreads();  // every P.V read of sV / sP done
    }
  } else {
    // The step's own key (column *cursor, written by the predecessor) is key n-1, in chunk 0 iff left <= CHUNK.
    const bool pre = (opts & KT2N_STEP_O1_ATTN_KVPRE) && left > CHUNK;
    if (pre) {
      tile(sK, true, 0, false);
      cp_async_commit();
      tile(sV, false, 0, false);
      cp_async_commit();
    }
    pdl_wait();
    if (!pre) {
      const bool skip0 = FUSE && cn == 0;
      tile(sK, true, 0, skip0);
      cp_async_commit();
      tile(sV, false, 0, skip0);
      cp_async_commit();
    }
    if ((opts & KT2N_STEP_O1_L2PF_W) && b == 0) l2_prefetch_share(pf, split * HKV + kvh, nsplit * HKV, tid);
    if constexpr (FUSE) {
      if (warp < GQA) {
        uint32_t lo, hi;
        qk_head(fz.qkv + b * fz.ld + (kvh * GQA + warp) * D, fz.q_w, fz.eps, fz.cos_t + b * fz.cs,
                fz.sin_t + b * fz.cs, lane, lo, hi);
        *reinterpret_cast<uint32_t*>(&sQ[warp][2 * lane]) = lo;
        *reinterpret_cast<uint32_t*>(&sQ[warp][64 + 2 * lane]) = hi;
      } else if (last) {
        fused_kv(true);
        if (cn == 0) {
          uint4* tile = warp == 2 ? sK : sV;
          *tile_word(tile, rn, 2 * lane) = kv_lo;
          *tile_word(tile, rn, 64 + 2 * lane) = kv_hi;
        }
      }
      __syncthreads();  // sQ visible
    }

    // ---- from here on: k3's decode_kernel, unchanged (FUSE: q from sQ; the step's K/V row stored, not loaded) ----
    FragA qa[8];
    {
      const __nv_bfloat16* q0 = FUSE ? &sQ[0][0] : q + b * q_row_stride + kvh * GQA * D;
      const __nv_bfloat16* q1 = FUSE ? &sQ[GQA - 1][0] : q0 + D;
  #pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const bool on = g == 0;
        qa[ks].x[0] = on ? ld32(q0 + 16 * ks + 2 * t) : 0u;
        qa[ks].x[1] = on ? ld32(q1 + 16 * ks + 2 * t) : 0u;
        qa[ks].x[2] = on ? ld32(q0 + 16 * ks + 8 + 2 * t) : 0u;
        qa[ks].x[3] = on ? ld32(q1 + 16 * ks + 8 + 2 * t) : 0u;
      }
    }

    for (int c = 0; c < nch; ++c) {
      const int nv = nvalid(c);
      cp_async_wait<1>();
      __syncthreads();  // K(c) visible
      FragC s0, s1;
      qk_tile(s0, qa, sK, 2 * warp, lane);
      qk_tile(s1, qa, sK, 2 * warp + 1, lane);
      const int j0 = 16 * warp + 2 * t, j1 = j0 + 8;
      float sc[8] = {s0.x[0], s0.x[1], s0.x[2], s0.x[3], s1.x[0], s1.x[1], s1.x[2], s1.x[3]};
      const bool v[8] = {j0 < nv, j0 + 1 < nv, j0 < nv, j0 + 1 < nv, j1 < nv, j1 + 1 < nv, j1 < nv, j1 + 1 < nv};
  #pragma unroll
      for (int i = 0; i < 8; ++i) sc[i] = v[i] ? score_of(sc[i]) : -INFINITY;
      const float w0 = quad_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[4], sc[5])));
      const float w1 = quad_max(fmaxf(fmaxf(sc[2], sc[3]), fmaxf(sc[6], sc[7])));
      if (lane == 0) {
        sMax[warp][0] = w0;
        sMax[warp][1] = w1;
      }
      __syncthreads();
      if (c + 1 < nch) {
        if (FUSE && cn == c + 1) {
          tile(sK, true, c + 1, true);
          if (warp == 2) {
            *tile_word(sK, rn, 2 * lane) = kv_lo;
            *tile_word(sK, rn, 64 + 2 * lane) = kv_hi;
          }
        } else {
          tile(sK, true, c + 1, false);
        }
      }
      cp_async_commit();
      const float mc0 = fmaxf(fmaxf(sMax[0][0], sMax[1][0]), fmaxf(sMax[2][0], sMax[3][0]));
      const float mc1 = fmaxf(fmaxf(sMax[0][1], sMax[1][1]), fmaxf(sMax[2][1], sMax[3][1]));
      float a0 = 1.f, a1 = 1.f;
      if (c == 0) {
        m0 = mc0;
        m1 = mc1;
      } else {
        const float n0 = fmaxf(m0, mc0), n1 = fmaxf(m1, mc1);
        a0 = expf(__fsub_rn(m0, n0));
        a1 = expf(__fsub_rn(m1, n1));
        m0 = n0;
        m1 = n1;
      }
      if (g == 0) {
        sP[0][j0] = v[0] ? prob_of(sc[0], m0) : 0.f;
        sP[0][j0 + 1] = v[1] ? prob_of(sc[1], m0) : 0.f;
        sP[1][j0] = v[2] ? prob_of(sc[2], m1) : 0.f;
        sP[1][j0 + 1] = v[3] ? prob_of(sc[3], m1) : 0.f;
        sP[0][j1] = v[4] ? prob_of(sc[4], m0) : 0.f;
        sP[0][j1 + 1] = v[5] ? prob_of(sc[5], m0) : 0.f;
        sP[1][j1] = v[6] ? prob_of(sc[6], m1) : 0.f;
        sP[1][j1 + 1] = v[7] ? prob_of(sc[7], m1) : 0.f;
      }
      cp_async_wait<1>();
      __syncthreads();
      if (c == 0) {
  #pragma unroll
        for (int i = 0; i < 4; ++i) zero(o[i]);
        zero(lf);
      } else {
  #pragma unroll
        for (int i = 0; i < 4; ++i) scale_frag(o[i], a0, a1);
        scale_frag(lf, a0, a1);
      }
  #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        float p[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (g == 0) {
          const int k0 = 16 * ks + 2 * t, k1 = k0 + 8;
          p[0] = sP[0][k0];
          p[1] = sP[0][k0 + 1];
          p[2] = sP[1][k0];
          p[3] = sP[1][k0 + 1];
          p[4] = sP[0][k1];
          p[5] = sP[0][k1 + 1];
          p[6] = sP[1][k1];
          p[7] = sP[1][k1 + 1];
        }
        const PFrag pf2 = make_pfrag(p);
        pv_step<4>(o, pf2, sV, ks, 4 * warp, lane);
        l_step(lf, pf2);
      }
      __syncthreads();
      if (c + 1 < nch) {
        if (FUSE && cn == c + 1) {
          tile(sV, false, c + 1, true);
          if (warp == 3) {
            *tile_word(sV, rn, 2 * lane) = kv_lo;
            *tile_word(sV, rn, 64 + 2 * lane) = kv_hi;
          }
        } else {
          tile(sV, false, c + 1, false);
        }
      }
      cp_async_commit();
    }
  }
  cp_async_wait<0>();
  if constexpr (FUSE) {
    // KT2N_STEP_O1_KV_NEXT (L3 sprint 5): this CTA's own tiles have landed; ask the L2 for the same row / head / keys
    // of the next layer (voice-store part, then the slot part), so that DRAM read overlaps the projections between
    // this attention and the next.  A prefetch moves no value into any register or shared memory: results unchanged.
    if (fz.nk != nullptr && key0 < fz.pf_keys && tid < 4) {
      const int64_t nks = imin64(static_cast<int64_t>(sk), left);  // keys this CTA covers
      const int64_t rbv = kbv != nullptr ? imin64(pb, nks) : 0;   // ... of them read from the voice store
      const __nv_bfloat16* src = nullptr;
      int64_t keys = 0;
      if (tid < 2) {
        if (rbv > 0) {
          const int64_t lk = static_cast<int64_t>(tid == 0 ? 0 : fz.layers) + fz.layer + 1;
          src = fz.vkv + vrow_off + ((lk * HKV + kvh) * vrow_len + key0) * D;
          keys = rbv;
        }
      } else if (nks > rbv) {
        src = (tid == 2 ? fz.nk : fz.nv) + slot * slot_stride + kvh * head_stride + (col0 + key0 + rbv) * D;
        keys = nks - rbv;
      }
      if (keys > 0) l2_bulk_prefetch(src, static_cast<unsigned>(keys * D * 2));
    }
  }
  if ((opts & KT2N_STEP_O1_L2PF_LATE) && split == 0) {
    // after this CTA's own K/V reads: the split-0 CTAs (rows x 8, all working) share the prefetch
    const int rows_n = gridx ? gridDim.y : gridDim.z;
    l2_prefetch_share(pf, b * HKV + kvh, rows_n * HKV, tid);
  }
  const float l0 = lf.x[0], l1 = lf.x[2];

  if (nsplit == 1) {
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = 8 * (4 * warp + i) + 2 * t;
        *reinterpret_cast<uint32_t*>(orow + d) = finish2(o[i].x[0], o[i].x[1], l0);
        *reinterpret_cast<uint32_t*>(orow + D + d) = finish2(o[i].x[2], o[i].x[3], l1);
      }
    }
    return;
  }

  const size_t row_parts = (static_cast<size_t>(b) * HKV + kvh) * grid_splits;
  {
    float* p0 = partials + ((row_parts + split) * GQA + 0) * PSTR;
    float* p1 = p0 + PSTR;
    if (tid == 0) {
      *reinterpret_cast<float4*>(p0) = make_float4(m0, l0, 0.f, 0.f);
      *reinterpret_cast<float4*>(p1) = make_float4(m1, l1, 0.f, 0.f);
    }
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = 8 * (4 * warp + i) + 2 * t;
        *reinterpret_cast<float2*>(p0 + 4 + d) = make_float2(o[i].x[0], o[i].x[1]);
        *reinterpret_cast<float2*>(p1 + 4 + d) = make_float2(o[i].x[2], o[i].x[3]);
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(&counters[b * HKV + kvh], 1);
    sLast = (prev == nsplit - 1);
  }
  __syncthreads();
  if (!sLast) return;
  __threadfence();

  const int hh = tid >> 6, dp = tid & 63;
  constexpr int CAP = static_cast<int>(2 * CHUNK * SEGS * sizeof(uint4) / (GQA * PSTR * sizeof(float)));
  float* sPart = reinterpret_cast<float*>(DEEP ? dyn : sKV);
  const float* rp = partials + row_parts * GQA * PSTR;
  float Fm = 0.f, Fl = 0.f, Fo0 = 0.f, Fo1 = 0.f;
  for (int s0 = 0; s0 < nsplit; s0 += CAP) {
    const int cnt = min(CAP, nsplit - s0);
    const int n4 = cnt * GQA * PSTR / 4;
    const float4* src = reinterpret_cast<const float4*>(rp + static_cast<size_t>(s0) * GQA * PSTR);
    float4* dst = reinterpret_cast<float4*>(sPart);
    for (int i = tid; i < n4; i += DEC_THREADS) dst[i] = __ldcg(src + i);
    __syncthreads();
    for (int k = 0; k < cnt; ++k) {
      const float* P = sPart + (k * GQA + hh) * PSTR;
      const float pm = P[0], pl = P[1];
      const float2 po = *reinterpret_cast<const float2*>(P + 4 + 2 * dp);
      if (s0 + k == 0) {
        Fm = pm;
        Fl = pl;
        Fo0 = po.x;
        Fo1 = po.y;
      } else {
        const FoldCoef f = fold_coef(Fm, pm);
        Fl = fold_val(Fl, f.a, pl, f.e);
        Fo0 = fold_val(Fo0, f.a, po.x, f.e);
        Fo1 = fold_val(Fo1, f.a, po.y, f.e);
        Fm = f.m;
      }
    }
    __syncthreads();
  }
  *reinterpret_cast<uint32_t*>(orow + hh * D + 2 * dp) = finish2(Fo0, Fo1, Fl);
  if (tid == 0) counters[b * HKV + kvh] = 0;
}

int launch_attn(const void* q, int64_t q_row_stride, void* k_cache, void* v_cache, int64_t kv_slot_stride,
                int64_t kv_head_stride, const int64_t* positions, const int64_t* cursor, int rows, int max_ctx,
                void* out, int64_t out_row_stride, void* workspace, size_t workspace_bytes, bool pdl,
                unsigned opts, const PfList& pf, const QkFuse* fz, cudaStream_t stream,
                const int32_t* row_slot = nullptr, int row_kv = 0) {
  using kt2n_ops_detail::aligned_to;
  // SPLIT64 (L3): one 64-key chunk per split, so twice the CTAs and no serial second tile (not with DEEP).
  const int split64 = ((opts & KT2N_STEP_O1_SPLIT64) && !(fz != nullptr && (opts & KT2N_STEP_O1_ATTN_DEEP))) ? 1 : 0;
  // Grid: rows <= kSplit64MaxKeys need <= 10 64-key splits, longer rows <= ceil(max_ctx / SPLIT_KEYS) splits.
  const int gs_long = splits_for(max_ctx), gs_short = splits_for(kSplit64MaxKeys, CHUNK);
  const int gs = (split64 && gs_short > gs_long) ? gs_short : gs_long;
  // k3's launcher checks (attn_decode.cu launch_decode<SLOTKV>): the kernel moves K/V, q and out in 16-byte pieces.
  if (rows < 1 || rows > KT2N_ATTN_MAX_ROWS || max_ctx <= 0 || max_ctx > KT2N_ATTN_MAX_CTX) return KT2N_ATTN_EINVAL;
  if (!q || !k_cache || !v_cache || !out || !workspace || !positions || !cursor) return KT2N_ATTN_EINVAL;
  if (!aligned_to(q, 16) || !aligned_to(k_cache, 16) || !aligned_to(v_cache, 16) || !aligned_to(out, 16) ||
      !aligned_to(workspace, 16))
    return KT2N_ATTN_EINVAL;
  if ((q_row_stride % 8) || (out_row_stride % 8) || (kv_slot_stride % 8) || (kv_head_stride % 8))
    return KT2N_ATTN_EINVAL;
  if (workspace_bytes < kt2n_attn_decode_workspace_bytes(rows, gs * SPLIT_KEYS)) return KT2N_ATTN_EINVAL;
  static const bool carve = [] {
    cudaFuncSetAttribute(attn_kernel<false, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(attn_kernel<true, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(attn_kernel<true, true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(attn_kernel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDeepSmem);
    cudaGetLastError();
    return true;
  }();
  (void)carve;

  int* counters = static_cast<int*>(workspace);
  float* partials = reinterpret_cast<float*>(static_cast<char*>(workspace) + COUNTER_BYTES);
  const dim3 grid = (opts & KT2N_STEP_O1_ATTN_GRID) ? dim3(HKV, rows, gs) : dim3(gs, HKV, rows);
  const QkFuse none{};
  const bool deep = fz != nullptr && (opts & KT2N_STEP_O1_ATTN_DEEP);
  auto kern = fz ? (deep ? attn_kernel<true, true> : attn_kernel<true, false>) : attn_kernel<false, false>;
  const cudaError_t e = kt2n_attn::launch_kernel(
      kern, grid, dim3(DEC_THREADS), deep ? static_cast<size_t>(kDeepSmem) : 0, stream, pdl,
      static_cast<const __nv_bfloat16*>(q), q_row_stride, static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache), kv_slot_stride, kv_head_stride, positions, cursor, gs,
      static_cast<__nv_bfloat16*>(out), out_row_stride, counters, partials, opts, pf, fz ? *fz : none, row_slot,
      row_kv, split64);
  return static_cast<int>(e);
}

// ------------------------------------------------------------------------------------------------ audio head
// k4's samp_head_kernel, arithmetic copied verbatim; PDL: the warp's weight rows are prefetched to L2 and its
// first k-block loaded before griddepcontrol.wait, x (the predecessor's output) staged after it.
constexpr int kHeadMaxThreads = 576;

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
  const unsigned int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    f[2 * q] = __uint_as_float(w[q] << 16);
    f[2 * q + 1] = __uint_as_float(w[q] & 0xFFFF0000u);
  }
}

template <int M, int CPW>
__global__ void __launch_bounds__(kHeadMaxThreads, 1)
    head_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, const __nv_bfloat16* __restrict__ w, int64_t ldw,
                __nv_bfloat16* __restrict__ out, int64_t ldo, int N, int K, int prefetch) {
  extern __shared__ uint4 xs[];  // [M][K / 8]
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int kv = K >> 3;
  const int groups = (N + CPW - 1) / CPW;
  // Before the wait: only the (constant) weights.  First group of this warp: L2 prefetch of its CPW rows.
  if (prefetch) {
    const int g0 = blockIdx.x * warps + warp;
    if (g0 < groups && lane == 0) {
      const int n0 = g0 * CPW;
      const int nr = (n0 + CPW <= N) ? CPW : (N - n0);
      if (ldw == K) {
        l2_bulk_prefetch(w + static_cast<int64_t>(n0) * ldw, static_cast<unsigned>(nr) * K * 2u);
      } else {
        for (int c = 0; c < nr; ++c) l2_bulk_prefetch(w + static_cast<int64_t>(n0 + c) * ldw, K * 2u);
      }
    }
  }
  kt2n_attn::pdl_wait();
  for (int i = threadIdx.x; i < M * kv; i += blockDim.x) {
    const int m = i / kv, c = i - m * kv;
    xs[i] = *(reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * ldx) + c);  // coherent: written by
  }                                                                                      // the predecessor
  __syncthreads();
  for (int g = blockIdx.x * warps + warp; g < groups; g += gridDim.x * warps) {
    const int n0 = g * CPW;
    float acc[M][CPW];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int c = 0; c < CPW; ++c) acc[m][c] = 0.0f;

    const uint4* wcol[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) {
      const int n = (n0 + c < N) ? (n0 + c) : (N - 1);
      wcol[c] = reinterpret_cast<const uint4*>(w + static_cast<int64_t>(n) * ldw) + lane;
    }
    uint4 wcur[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) wcur[c] = __ldcs(wcol[c]);

    for (int kb = 0; kb < K; kb += 256) {
      uint4 wnext[CPW];
      if (kb + 256 < K) {
#pragma unroll
        for (int c = 0; c < CPW; ++c) wnext[c] = __ldcs(wcol[c] + ((kb + 256) >> 3));
      }
      float wf[CPW][8];
#pragma unroll
      for (int c = 0; c < CPW; ++c) unpack8(wcur[c], wf[c]);
      const int xi = (kb >> 3) + lane;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[8];
        unpack8(xs[m * kv + xi], xf);
#pragma unroll
        for (int c = 0; c < CPW; ++c)
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[m][c] = __fmaf_rn(xf[j], wf[c][j], acc[m][c]);
      }
      if (kb + 256 < K) {
#pragma unroll
        for (int c = 0; c < CPW; ++c) wcur[c] = wnext[c];
      }
    }

#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int c = 0; c < CPW; ++c) {
        float v = acc[m][c];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
        acc[m][c] = v;
      }

#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int c = 0; c < CPW; ++c) {
        const int e = m * CPW + c;
        if ((e & 31) == lane && n0 + c < N)
          out[static_cast<int64_t>(m) * ldo + n0 + c] = __float2bfloat16_rn(acc[m][c]);
      }
  }
}

template <int M>
int launch_head_m(const __nv_bfloat16* x, int64_t ldx, const __nv_bfloat16* w, int64_t ldw, __nv_bfloat16* out,
                  int64_t ldo, int N, int K, bool pdl, int prefetch, cudaStream_t stream) {
  constexpr int CPW = 2;  // k4: KT2N_SAMP_HEAD_CPW_SMALL (2) for M <= 8, 2 above
  const size_t smem = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
  if (smem > 96 * 1024) return KT2N_SAMP_EARG;
  // as k4: the dynamic shared-memory limit set to exactly this launch's need
  const cudaError_t err = cudaFuncSetAttribute(head_kernel<M, CPW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(smem));
  if (err != cudaSuccess) return static_cast<int>(err);
  const int groups = (N + CPW - 1) / CPW;
  int device = 0, sms = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms < 1)
    sms = 1;
  int warps = (groups + sms - 1) / sms;
  warps = warps < 4 ? 4 : (warps > kHeadMaxThreads / 32 ? kHeadMaxThreads / 32 : warps);
  int blocks = (groups + warps - 1) / warps;
  if (blocks > sms) blocks = sms;
  return static_cast<int>(kt2n_attn::launch_kernel(head_kernel<M, CPW>, dim3(blocks), dim3(warps * 32), smem, stream,
                                                   pdl, x, ldx, w, ldw, out, ldo, N, K, prefetch));
}

// L3: the o1 head takes up to 24 rows per launch (x staged in 24 x 2048 x 2 = 96 KB of shared memory), so the
// 17..64-row buckets read the 27 MB head ceil(rows / 24) times instead of ceil(rows / 16).  Per-row arithmetic is
// unchanged (each row's accumulators see the same FMAs in the same order for any M).
constexpr int kO1HeadMaxRows = 24;

int launch_head(const void* x, int64_t ldx, const void* w, int64_t ldw, void* out, int64_t ldo, int M, int N, int K,
                bool pdl, int prefetch, cudaStream_t stream) {
  if (!x || !w || !out || M < 1 || M > kO1HeadMaxRows || N < 1 || K < 256 || (K % 256) != 0 || ldx < K ||
      ldw < K || ldo < N)
    return KT2N_SAMP_EARG;
  if ((reinterpret_cast<uintptr_t>(x) & 15u) || (reinterpret_cast<uintptr_t>(w) & 15u) || (ldx % 8) || (ldw % 8))
    return KT2N_SAMP_EALIGN;
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const __nv_bfloat16*>(w);
  auto* ob = static_cast<__nv_bfloat16*>(out);
  switch (M) {
#define KT2N_O1_HEAD_CASE(R) \
  case R:                    \
    return launch_head_m<R>(xb, ldx, wb, ldw, ob, ldo, N, K, pdl, prefetch, stream);
    KT2N_O1_HEAD_CASE(1)
    KT2N_O1_HEAD_CASE(2)
    KT2N_O1_HEAD_CASE(3)
    KT2N_O1_HEAD_CASE(4)
    KT2N_O1_HEAD_CASE(5)
    KT2N_O1_HEAD_CASE(6)
    KT2N_O1_HEAD_CASE(7)
    KT2N_O1_HEAD_CASE(8)
    KT2N_O1_HEAD_CASE(9)
    KT2N_O1_HEAD_CASE(10)
    KT2N_O1_HEAD_CASE(11)
    KT2N_O1_HEAD_CASE(12)
    KT2N_O1_HEAD_CASE(13)
    KT2N_O1_HEAD_CASE(14)
    KT2N_O1_HEAD_CASE(15)
    KT2N_O1_HEAD_CASE(16)
    KT2N_O1_HEAD_CASE(17)
    KT2N_O1_HEAD_CASE(18)
    KT2N_O1_HEAD_CASE(19)
    KT2N_O1_HEAD_CASE(20)
    KT2N_O1_HEAD_CASE(21)
    KT2N_O1_HEAD_CASE(22)
    KT2N_O1_HEAD_CASE(23)
    KT2N_O1_HEAD_CASE(24)
#undef KT2N_O1_HEAD_CASE
    default:
      return KT2N_SAMP_EARG;
  }
}

// ------------------------------------------------------------------------------------------------ the step
constexpr int kH = KT2N_STEP_HIDDEN;
constexpr int kI = KT2N_STEP_INTER;
constexpr int kQKV = KT2N_STEP_QKV;
constexpr int kHQ = KT2N_STEP_Q_HEADS;
constexpr int kHK = KT2N_STEP_KV_HEADS;
constexpr int kD = KT2N_STEP_HEAD_DIM;
constexpr int kCS = 64;

int lane_err(int lane, long code) {
  if (code == 0) return KT2N_STEP_OK;
  const long c = code < 0 ? 50000 - code : code;
  return static_cast<int>(lane * KT2N_STEP_ELANE + (c % KT2N_STEP_ELANE));
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

void pf_add(PfList& pf, const void* p, size_t bytes) {
  if (!p || bytes < 16 || pf.count >= kPfMax) return;
  pf.p[pf.count] = static_cast<const char*>(p);
  pf.n[pf.count] = bytes & ~static_cast<size_t>(15);
  ++pf.count;
}

}  // namespace kt2n_o1

extern "C" {

void kt2n_step_o1_set_options(unsigned options) { kt2n_o1::g_options = options; }
unsigned kt2n_step_o1_options(void) { return kt2n_o1::g_options; }
int kt2n_step_o1_set_bucket_options(int rows, unsigned options) {
  if (rows < 1 || rows > KT2N_STEP_MAX_ROWS_EXT || options == KT2N_STEP_O1_AUTO) return KT2N_STEP_EINVAL;
  kt2n_o1::g_bucket_options[rows] = options;
  return KT2N_STEP_OK;
}
unsigned kt2n_step_o1_options_for(int rows) { return kt2n_o1::options_for(rows); }
void kt2n_step_o1_set_kv_next_budget(unsigned long long bytes) { kt2n_o1::g_kv_next_budget = bytes; }
int kt2n_step_o1_set_shape(int rows, int family, int nt, int wn) {
  if (rows < 1 || rows > KT2N_STEP_MAX_ROWS_EXT || family < 0 || family > 3 || nt < 0 || nt > 8 || wn < 0 || wn > 8)
    return KT2N_STEP_EINVAL;
  return kt2n_step_o1_set_shape_mt(rows, family, nt, wn, 0);
}
int kt2n_step_o1_set_shape_mt(int rows, int family, int nt, int wn, int mt) {
  if (rows < 1 || rows > KT2N_STEP_MAX_ROWS_EXT || family < 0 || family > 3 || nt < 0 || nt > 8 || wn < 0 || wn > 8 ||
      mt < -999 || (mt > 199 && mt < 1000) || mt > 1999 || (mt != 0 && rows <= KT2N_STEP_MAX_ROWS))
    return KT2N_STEP_EINVAL;
  kt2n_o1::g_shape[rows][family][0] = static_cast<short>(nt);
  kt2n_o1::g_shape[rows][family][1] = static_cast<short>(wn);
  kt2n_o1::g_shape[rows][family][2] = static_cast<short>(mt);
  return KT2N_STEP_OK;
}

int kt2n_step_forward_o1(const kt2n_step_model* m, const kt2n_step_io* io, cudaStream_t s) {
  using namespace kt2n_o1;
  // Argument checks: identical to kt2n_step_forward.
  if (!m || !io || m->abi != KT2N_STEP_ABI) return KT2N_STEP_EINVAL;
  if (m->layers < 1 || m->layers > KT2N_STEP_MAX_LAYERS || !m->layer || !m->embed || !m->final_norm) {
    return KT2N_STEP_EINVAL;
  }
  const int rows = io->rows;
  const bool row_kv = (io->flags & KT2N_STEP_FLAG_ROW_KV) != 0;
  const int max_rows = row_kv ? KT2N_STEP_MAX_ROWS_EXT : KT2N_STEP_MAX_ROWS;
  if (rows < 1 || rows > max_rows || rows > m->slots || m->max_ctx < 1) return KT2N_STEP_EINVAL;
  if (!io->tokens || !io->positions || !io->cursor) return KT2N_STEP_EINVAL;
  if (!io->scratch || io->scratch_bytes < kt2n_step_scratch_bytes(rows)) return KT2N_STEP_EINVAL;
  if (!io->attn_ws || io->attn_ws_bytes < kt2n_attn_decode_workspace_bytes(rows, m->max_ctx)) {
    return KT2N_STEP_EINVAL;
  }
  const bool head = !(io->flags & KT2N_STEP_FLAG_NO_HEAD);
  const bool sample = (io->flags & KT2N_STEP_FLAG_SAMPLE) != 0;
  if (head && (!m->head_w || m->head_n < 1 || !io->logits || io->ld_logits < m->head_n || !aligned16(m->head_w))) {
    return KT2N_STEP_EINVAL;
  }
  if (sample && (!head || m->head_n != KT2N_SAMP_AUDIO_N || !io->samp_params || !io->samp_history ||
                 !io->samp_history_len || !io->samp_seeds || !io->samp_offsets || !io->samp_tokens)) {
    return KT2N_STEP_EINVAL;
  }
  for (int l = 0; l < m->layers; ++l) {
    const kt2n_step_layer& L = m->layer[l];
    if (!L.qkv_codes || !L.qkv_scales || !L.o_codes || !L.o_scales || !L.gate_up_codes || !L.gate_up_scales ||
        !L.down_codes || !L.down_scales || !L.ln_in || !L.ln_post || !L.q_norm || !L.k_norm || !L.k_cache ||
        !L.v_cache) {
      return KT2N_STEP_EINVAL;
    }
  }

  kt2n_step_scratch_view v;
  if (kt2n_step_scratch_bind(io->scratch, io->scratch_bytes, rows, &v)) return KT2N_STEP_EINVAL;
  using bf = __nv_bfloat16;
  bf* hidden = static_cast<bf*>(v.hidden);
  bf* normed = static_cast<bf*>(v.normed);
  bf* qkv = static_cast<bf*>(v.qkv);
  bf* query = static_cast<bf*>(v.query);
  bf* attn = static_cast<bf*>(v.attn);
  bf* delta = static_cast<bf*>(v.delta);
  bf* act = static_cast<bf*>(v.act);
  bf* cos_t = static_cast<bf*>(v.cos);
  bf* sin_t = static_cast<bf*>(v.sin);

  const unsigned opts = options_for(rows);
  // Per-row K/V lives only in the fused attention kernel of this file.
  if (row_kv && !((opts & KT2N_STEP_O1_ATTN) && (opts & KT2N_STEP_O1_FUSE_QK))) return KT2N_STEP_EINVAL;
  const int pdl = (io->flags & KT2N_STEP_FLAG_PDL) ? 1 : 0;
  const bool norm_early1 = (opts & KT2N_STEP_O1_NORM_EARLY1) != 0;
  const bool fuse_norm = (opts & KT2N_STEP_O1_FUSE_NORM) != 0 && rows <= KT2N_STEP_MAX_ROWS;
  const bool norm_early2 = (opts & KT2N_STEP_O1_NORM_EARLY2) != 0;
  const unsigned tflags = pdl ? KT2N_TERN_FLAG_PDL : 0u;
  const unsigned aflags = pdl ? KT2N_ATTN_FLAG_PDL : 0u;
  const float eps = m->rms_eps;
  kt2n_step_weight_bytes_t wb;
  kt2n_step_weight_bytes(&wb);
  long rc;

  rc = kt2n_ops_embed_rmsnorm(io->tokens, rows, static_cast<const bf*>(m->embed), m->vocab, kH, nullptr, 0, nullptr,
                              -1, static_cast<const bf*>(m->layer[0].ln_in), eps, hidden, kH, normed, kH,
                              io->positions, nullptr, cos_t, sin_t, kCS, pdl, s);
  if (rc) return lane_err(2, rc);

  for (int l = 0; l < m->layers; ++l) {
    const kt2n_step_layer& L = m->layer[l];
    if (fuse_norm && l > 0) {  // layer 0 reads the embedding kernel's normed output
      rc = kt2n_tern_proj_norm(KT2N_TERN_FAM_QKV, hidden, kH, rows, L.ln_in, eps, L.qkv_codes, L.qkv_scales, kQKV,
                               kH, qkv, kQKV, KT2N_TERN_EPI_STORE, tflags, s);
    } else {
      rc = proj(KT2N_TERN_FAM_QKV, normed, kH, rows, L.qkv_codes, L.qkv_scales, kQKV, kH, qkv, kQKV,
                KT2N_TERN_EPI_STORE, nullptr, 0, tflags, s);
    }
    if (rc) return lane_err(1, rc);

    kt2n_ops_kv_target kv{};
    kv.k = static_cast<bf*>(L.k_cache);
    kv.v = static_cast<bf*>(L.v_cache);
    kv.slot_stride = m->kv_slot_stride;
    kv.head_stride = m->kv_head_stride;
    kv.col_stride = kD;
    kv.row_slot = nullptr;
    kv.slot_base = 0;
    kv.slot_step = 1;
    kv.row_col = nullptr;
    kv.cursor = io->cursor;
    kv.col_base = 0;
    kv.col_step = 0;
    kv.num_cols = m->max_ctx;
    const bool fuse = (opts & KT2N_STEP_O1_ATTN) && (opts & KT2N_STEP_O1_FUSE_QK);
    if (!fuse) {
      rc = kt2n_ops_qk_norm_rope(qkv, kQKV, qkv + kHQ * kD, kQKV, qkv + (kHQ + kHK) * kD, kQKV,
                                 static_cast<const bf*>(L.q_norm), static_cast<const bf*>(L.k_norm), eps, nullptr,
                                 nullptr, cos_t, sin_t, kCS, query, kHQ * kD, kD, nullptr, 0, 0, &kv, rows, kHQ, kHK,
                                 kD, pdl, s);
      if (rc) return lane_err(2, rc);
    } else {
      // The fused path skips kt2n_ops_qk_norm_rope, so apply its checks on the caller's buffers here (q/k norm
      // weights and K/V targets: 4-byte pairs, even strides), with its return code: same result as k5.
      using kt2n_ops_detail::aligned_to;
      if (!aligned_to(L.q_norm, 4) || !aligned_to(L.k_norm, 4) || !aligned_to(L.k_cache, 4) ||
          !aligned_to(L.v_cache, 4) || (m->kv_slot_stride % 2) || (m->kv_head_stride % 2)) {
        return lane_err(2, cudaErrorInvalidValue);
      }
    }
    QkFuse fz{qkv, kQKV, static_cast<const bf*>(L.q_norm), static_cast<const bf*>(L.k_norm), eps, cos_t, sin_t, kCS,
              m->max_ctx, nullptr, nullptr, nullptr, 0, 0};
    if (row_kv && io->voice_kv && io->voice_off && io->voice_len) {
      fz.vkv = static_cast<const bf*>(io->voice_kv);
      fz.voff = io->voice_off;
      fz.vlen = io->voice_len;
      fz.layer = l;
      fz.layers = m->layers;
    }

    if ((opts & KT2N_STEP_O1_KV_NEXT) && fuse && row_kv && l + 1 < m->layers) {
      fz.nk = static_cast<const bf*>(m->layer[l + 1].k_cache);
      fz.nv = static_cast<const bf*>(m->layer[l + 1].v_cache);
      fz.pf_keys = static_cast<int64_t>(g_kv_next_budget / (static_cast<unsigned long long>(rows) * kHK * kD * 2 * 2));
    }
    if (opts & KT2N_STEP_O1_ATTN) {
      PfList pf{};
      unsigned opts_l = opts;
      if (opts & (KT2N_STEP_O1_L2PF | KT2N_STEP_O1_L2PF_LATE | KT2N_STEP_O1_L2PF_W)) {
        const unsigned mask = (opts & KT2N_STEP_O1_PF_MASK) ? (opts & KT2N_STEP_O1_PF_MASK) : KT2N_STEP_O1_PF_MASK;
        if (mask & KT2N_STEP_O1_PF_O) {
          pf_add(pf, L.o_codes, wb.o_codes);
          pf_add(pf, L.o_scales, wb.o_scales);
        }
        if (mask & KT2N_STEP_O1_PF_GU) {
          pf_add(pf, L.gate_up_codes, wb.gate_up_codes);
          pf_add(pf, L.gate_up_scales, wb.gate_up_scales);
        }
        if (mask & KT2N_STEP_O1_PF_DOWN) {
          pf_add(pf, L.down_codes, wb.down_codes);
          pf_add(pf, L.down_scales, wb.down_scales);
        }
        if ((mask & KT2N_STEP_O1_PF_QKV) && l + 1 < m->layers) {
          pf_add(pf, m->layer[l + 1].qkv_codes, wb.qkv_codes);
          pf_add(pf, m->layer[l + 1].qkv_scales, wb.qkv_scales);
        }
      }
      if ((opts & KT2N_STEP_O1_L2PF_HEAD) && head && l + 1 == m->layers) {
        pf_add(pf, m->head_w, static_cast<size_t>(m->head_n) * kH * 2);
        if (!(opts & (KT2N_STEP_O1_L2PF | KT2N_STEP_O1_L2PF_LATE | KT2N_STEP_O1_L2PF_W))) opts_l |= KT2N_STEP_O1_L2PF;
      }
      if (pf.count == 0) opts_l &= ~(KT2N_STEP_O1_L2PF | KT2N_STEP_O1_L2PF_LATE | KT2N_STEP_O1_L2PF_W);
      rc = launch_attn(query, kHQ * kD, L.k_cache, L.v_cache, m->kv_slot_stride, m->kv_head_stride, io->positions,
                       io->cursor, rows, m->max_ctx, attn, kHQ * kD, io->attn_ws, io->attn_ws_bytes, pdl != 0, opts_l,
                       pf, fuse ? &fz : nullptr, s, row_kv ? io->samp_slot_of_row : nullptr, row_kv ? 1 : 0);
    } else {
      rc = kt2n_attn_decode_slotkv(query, kHQ * kD, L.k_cache, L.v_cache, m->kv_slot_stride, m->kv_head_stride,
                                   io->positions, io->cursor, rows, m->max_ctx, attn, kHQ * kD, io->attn_ws,
                                   io->attn_ws_bytes, aflags, s);
    }
    if (rc) return lane_err(3, rc);

    if (fuse_norm) {
      // h += o(attn) in the epilogue; gate_up normalises h itself; down adds into h; the next qkv normalises h.
      rc = proj(KT2N_TERN_FAM_O, attn, kH, rows, L.o_codes, L.o_scales, kH, kH, hidden, kH, KT2N_TERN_EPI_RESIDUAL,
                hidden, kH, tflags, s);
      if (rc) return lane_err(1, rc);
      rc = kt2n_tern_proj_norm(KT2N_TERN_FAM_GATE_UP, hidden, kH, rows, L.ln_post, eps, L.gate_up_codes,
                               L.gate_up_scales, 2 * kI, kH, act, kI, KT2N_TERN_EPI_SILU_MUL, tflags, s);
      if (rc) return lane_err(1, rc);
      rc = proj(KT2N_TERN_FAM_DOWN, act, kI, rows, L.down_codes, L.down_scales, kH, kI, hidden, kH,
                KT2N_TERN_EPI_RESIDUAL, hidden, kH, tflags, s);
      if (rc) return lane_err(1, rc);
      continue;
    }
    rc = proj(KT2N_TERN_FAM_O, attn, kH, rows, L.o_codes, L.o_scales, kH, kH, delta, kH,
                        KT2N_TERN_EPI_STORE, nullptr, 0, tflags, s);
    if (rc) return lane_err(1, rc);
    rc = norm_early1 ? kt2n_ops_residual_rmsnorm_early(hidden, kH, delta, kH, static_cast<const bf*>(L.ln_post), eps,
                                                      hidden, kH, normed, kH, rows, kH, pdl, s)
                    : kt2n_ops_residual_rmsnorm(hidden, kH, delta, kH, static_cast<const bf*>(L.ln_post), eps, hidden,
                                                kH, normed, kH, rows, kH, pdl, s);
    if (rc) return lane_err(2, rc);

    rc = proj(KT2N_TERN_FAM_GATE_UP, normed, kH, rows, L.gate_up_codes, L.gate_up_scales, 2 * kI, kH,
                        act, kI, KT2N_TERN_EPI_SILU_MUL, nullptr, 0, tflags, s);
    if (rc) return lane_err(1, rc);
    rc = proj(KT2N_TERN_FAM_DOWN, act, kI, rows, L.down_codes, L.down_scales, kH, kI, delta, kH,
                        KT2N_TERN_EPI_STORE, nullptr, 0, tflags, s);
    if (rc) return lane_err(1, rc);
    const void* next_w = (l + 1 < m->layers) ? m->layer[l + 1].ln_in : m->final_norm;
    rc = norm_early2 ? kt2n_ops_residual_rmsnorm_early(hidden, kH, delta, kH, static_cast<const bf*>(next_w), eps,
                                                      hidden, kH, normed, kH, rows, kH, pdl, s)
                    : kt2n_ops_residual_rmsnorm(hidden, kH, delta, kH, static_cast<const bf*>(next_w), eps, hidden,
                                                kH, normed, kH, rows, kH, pdl, s);
    if (rc) return lane_err(2, rc);
  }

  if (fuse_norm) {  // the final norm (the head reads `normed`)
    rc = kt2n_ops_rmsnorm(hidden, kH, static_cast<const bf*>(m->final_norm), eps, normed, kH, rows, kH, pdl, s);
    if (rc) return lane_err(2, rc);
  }
  if (head) {
    // > 16 rows (ROW_KV only): the head runs in pieces of 24 rows (o1 head) or 16 (k4 head); each piece re-reads
    // the 27 MB head.
    const int piece = (opts & KT2N_STEP_O1_HEAD) ? kO1HeadMaxRows : KT2N_SAMP_HEAD_MAX_ROWS;
    for (int r0 = 0; r0 < rows; r0 += piece) {
      const int mr = (rows - r0) < piece ? (rows - r0) : piece;
      const bf* x0 = normed + static_cast<int64_t>(r0) * kH;
      void* l0 = static_cast<char*>(io->logits) + static_cast<int64_t>(r0) * io->ld_logits * 2;
      if (opts & KT2N_STEP_O1_HEAD) {
        rc = launch_head(x0, kH, m->head_w, kH, l0, io->ld_logits, mr, m->head_n, kH, pdl != 0,
                         (opts & KT2N_STEP_O1_HEAD_PF) ? 1 : 0, s);
      } else {
        rc = kt2n_samp_head_bf16(x0, kH, m->head_w, kH, l0, io->ld_logits, mr, m->head_n, kH, s);
      }
      if (rc) return lane_err(4, rc);
    }
  }
  if (sample) {
    rc = kt2n_samp_sample(io->logits, io->ld_logits, io->samp_params, io->samp_history, io->samp_history_len,
                          io->samp_seeds, io->samp_offsets, io->samp_slot_of_row, nullptr, 0, io->samp_tokens,
                          nullptr, nullptr, rows, io->samp_flags, s);
    if (rc) return lane_err(4, rc);
  }
  return KT2N_STEP_OK;
}

}  // extern "C"
