// kt2n native conformer encoder: device tile functions (lane L4). Included by encoder.cu (Phase B
// kernels) and by any megakernel translation unit (MK-R) that dispatches encoder ops through
// enc::tiles::run_tile. Contract: include/kt2n/render/encoder.h and ops.h.
#pragma once

#include <cuda_runtime.h>

#include "kt2n/render/encoder.h"

// The GEMM tile configurations (BM, BN, TM, TN, BK, KG): the host picks one per (op, bucket) through
// gemm_cfg_for (BM, BN); every configuration gives the same bits (canonical reduction order).
#define KT2N_ENC_GEMM_CONFIGS(X) \
  X(16, 16, 2, 2, 64, 4)          \
  X(64, 16, 4, 1, 64, 1)          \
  X(16, 32, 2, 4, 64, 4)          \
  X(32, 64, 2, 4, 64, 1)          \
  X(64, 64, 4, 4, 64, 1)          \
  KT2N_ENC_GEMM_128X64_ROWMAJOR(X)
// (128x64 row-major: the v6 tile for M > 1024; since sprint 2 only KT2N_ENC_BIG_TILE selects it.)
// k-major large-M tiles (gemm_tile_t, double buffered, 16 x 16 threads): (BM, BN, BK). Same bits as
// every other tile. -DKT2N_ENC_T128X64 builds the 128x64 tile this way instead of row-major.
#ifdef KT2N_ENC_T128X64
#define KT2N_ENC_GEMM_128X64_ROWMAJOR(X)
#define KT2N_ENC_GEMM_T_CONFIGS(X) \
  X(128, 64, 8)                    \
  X(64, 128, 8)                    \
  X(128, 128, 8)
#else
#define KT2N_ENC_GEMM_128X64_ROWMAJOR(X) X(128, 64, 8, 4, 32, 1)
#define KT2N_ENC_GEMM_T_CONFIGS(X) \
  X(64, 128, 8)                    \
  X(128, 128, 8)
#endif

namespace kt2n::render::enc::tiles {

inline constexpr int kSplitChunk = 512;  // split-K structure is a function of K only (batch invariance)
inline constexpr int kThreads = kTileThreads;  // 256

enum Dom : int { kDomTok = 0, kDomExt = 1, kDomFr = 2, kDomExplicit = 3 };

static __device__ __forceinline__ void griddep_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
static __device__ __forceinline__ void griddep_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::);
#endif
}
static __device__ __forceinline__ float tf32_rna(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r);
}
static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

static __device__ __forceinline__ int op_domain(const OpDesc& op) {
  switch (static_cast<OpKind>(op.kind)) {
    case OpKind::kEncPreLookahead: return op.kernel == 4 ? kDomExt : kDomTok;
    case OpKind::kEncEmbedNorm: return (op.flags & kOpFrames) ? kDomFr : kDomExt;
    case OpKind::kEncUpLayer:
    case OpKind::kEncUpEmbed:
    case OpKind::kEncProj: return kDomFr;
    case OpKind::kEncPosTable: return kDomExplicit;
    default: return (op.flags & kOpFrames) ? kDomFr : kDomTok;
  }
}
static __device__ __forceinline__ int dom_count(int dom, const OpDesc& op, const CallDesc& c) {
  switch (dom) {
    case kDomTok: return c.total_tokens;
    case kDomExt: return c.total_tokens + 2 * c.n_rows;
    case kDomFr: return c.total_frames;
    default: return op.m;
  }
}
static __device__ __forceinline__ int row_off(const RowDesc& r, int ri, int dom) {
  return dom == kDomFr ? r.fr_off : (dom == kDomExt ? r.tok_off + 2 * ri : r.tok_off);
}
// Last row whose packed offset is <= m (rows are packed in order).
static __device__ __forceinline__ int find_row(const CallDesc& c, int m, int dom) {
  int lo = 0, hi = c.n_rows - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (row_off(c.rows[mid], mid, dom) <= m) lo = mid; else hi = mid - 1;
  }
  return lo;
}
static __device__ __forceinline__ float* kv_ptr(const CallDesc& c, const RowDesc& r, bool frames, int layer, int kv,
                                         int abs) {
  if (!frames) {
    const int phys = c.page_table[r.page_off + (abs >> 5)];
    return c.pool.enc1 + enc1_offset(c.pool, layer, kv, phys, abs & (kPageTokens - 1));
  }
  const int phys = c.page_table[r.page_off + (abs >> 6)];
  return c.pool.enc2 + enc2_offset(c.pool, layer, kv, phys, abs & (kPageFrames - 1));
}

static __device__ __forceinline__ bool op_has_ln(const OpDesc& op) {
  const auto k = static_cast<OpKind>(op.kind);
  return k == OpKind::kEncLnQkv || k == OpKind::kEncLnFf1 || k == OpKind::kEncProj;
}

// Pointer to A[m][k0 .. k0 + 16) for packed row m (row ri, local position loc), or nullptr = zeros.
static __device__ __forceinline__ const float* a_src(const OpDesc& op, const CallDesc& c, int m, int ri, int loc, int k0) {
  switch (static_cast<OpKind>(op.kind)) {
    case OpKind::kEncEmbed:
      return static_cast<const float*>(op.in[0]) + size_t(c.tokens[m]) * kEncDim + k0;
    case OpKind::kEncPreLookahead: {
      const RowDesc& r = c.rows[ri];
      const int tap = k0 / kEncDim, cc = k0 % kEncDim;
      const int src = loc + tap;  // tail-extended index
      if (op.kernel == 4 && src >= r.tok_suffix + 2) return nullptr;  // F.pad zeros past the row's end
      return static_cast<const float*>(op.in[0]) + size_t(r.tok_off + 2 * ri + src) * kEncDim + cc;
    }
    case OpKind::kEncUpLayer: {
      const RowDesc& r = c.rows[ri];
      const int tap = k0 / kEncDim, cc = k0 % kEncDim;
      const int idx = (loc + tap) >> 1;  // nearest x2: frame f of [tail | x1] repeated
      if (idx < 2) {
        if (r.tok_past == 0) return nullptr;  // sequence start: F.pad zeros
        return c.state.enc_up_tail + (size_t(r.slot) * 2 + idx) * kEncDim + cc;
      }
      return static_cast<const float*>(op.in[0]) + size_t(r.tok_off + idx - 2) * kEncDim + cc;
    }
    default:
      return static_cast<const float*>(op.in[0]) + size_t(m) * size_t(op.ld_in[0]) + k0;
  }
}

// ---------------------------------------------------------------------------------------------
// GEMM tile: out[m, n] = epilogue(sum_k A[m, k] * W[n, k]) in fp32 FFMA.
//
// Canonical per-element reduction order (identical in every tile configuration, so a row's bits
// never depend on the batch or the bucket):
//   * K is cut into 512-wide outer chunks (across blocks: fixed split-K, kSplitChunk);
//   * each outer chunk into four 128-wide inner chunks, each a k-ascending fmaf chain from 0;
//   * outer partial = ((i0 + i1) + i2) + i3; result = outer partials summed in order.
// Small tiles (KG = 4) give each inner chunk to its own 64-thread group (4x fewer shared-memory
// bytes per FMA than a 256-thread 16x32 tile); large tiles (KG = 1) walk the inner chunks in order.
// Shared memory: per group, A and W tiles row-major with a 4-float pad ([BM][BK+4], [BN][BK+4]) so the
// float4 stores and the float4 (LDS.128) reads along k are bank-conflict free.
// ---------------------------------------------------------------------------------------------
inline constexpr int kInnerChunk = 128;
// Arrival counters of wave-tail pieces sit in the upper half of the counter array (split-K output tiles and
// attention groups use the lower indices).
inline constexpr int kPieceCounterBase = kMaxSplitTiles / 2;

template <int BM, int BN, int BK, int KG>
constexpr uint32_t gemm_smem_bytes() {
  constexpr int tiles = KG * (BM + BN) * (BK + 4);
  constexpr int red = KG * BM * BN;
  return uint32_t(((tiles > red ? tiles : red) + 2 * BM) * sizeof(float) + 2 * BM * sizeof(int));
}

static __device__ __forceinline__ void gemm_epilogue(const OpDesc& op, const CallDesc& c, int m, int n, float v, int ri,
                                              int loc) {
  const auto kind = static_cast<OpKind>(op.kind);
  const float* bias = static_cast<const float*>(op.w[1]);
  float* out = static_cast<float*>(op.out[0]);
  const int64_t ldo = op.ld_out[0];
  if (bias != nullptr) v += bias[n];
  switch (kind) {
    case OpKind::kEncPreLookahead:
      if (op.kernel == 4) {
        v = v > 0.f ? v : v * kLeakySlope;
        if (loc < 2 && c.rows[ri].tok_past == 0) v = 0.f;  // chunk cache left_ok: sequence-start pad
      } else {
        const RowDesc& r = c.rows[ri];
        v += static_cast<const float*>(op.in[1])[size_t(r.tok_off + 2 * ri + loc + 2) * kEncDim + n];
      }
      out[size_t(m) * ldo + n] = v;
      break;
    case OpKind::kEncLnQkv:
      if (n < kEncDim) {
        out[size_t(m) * ldo + n] = v;
      } else {
        const RowDesc& r = c.rows[ri];
        const bool frames = (op.flags & kOpFrames) != 0;
        const int abs = (frames ? r.fr_past : r.tok_past) + loc;
        kv_ptr(c, r, frames, op.layer, n >= 2 * kEncDim ? 1 : 0, abs)[n & (kEncDim - 1)] = v;
      }
      break;
    case OpKind::kEncOutProj:
    case OpKind::kEncFf2:
      v = static_cast<const float*>(op.in[1])[size_t(m) * ldo + n] + v;
      out[size_t(m) * ldo + n] = v;
      break;
    case OpKind::kEncLnFf1:
      out[size_t(m) * ldo + n] = v / (1.f + expf(-v));  // SiLU as torch: x / (1 + exp(-x))
      break;
    default:
      out[size_t(m) * ldo + n] = v;
      break;
  }
}

// LayerNorm row statistics of rows m0 .. m0 + BM (fixed order, the same in every tile configuration):
// one warp per row, each lane sums 16 values, then an xor tree; mean first, then the centred squares.
template <int BM>
static __device__ __forceinline__ void ln_row_stats(const OpDesc& op, int m0, int M, float* s_mean, float* s_rstd) {
  const int t = threadIdx.x;
  // Row statistics, one warp per row, fixed order: lane sums 16 values, then an xor tree. A warp
  // issues the loads of RB rows (i, i + 8, ...) before reducing any of them, so their memory
  // latencies overlap (v6 paid one dependent round trip per row: 16 in a row for BM = 128). The
  // per-row arithmetic is unchanged, so the bits are too.
  constexpr int kWarps = kThreads / 32;
  constexpr int RPW = BM / kWarps > 0 ? BM / kWarps : 1;
  constexpr int RB = RPW >= 4 ? 4 : RPW;
  const int warp = t >> 5, lane = t & 31;
  for (int i0 = warp; i0 < BM; i0 += kWarps * RB) {
    float4 v[RB][4];
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      const int m = m0 + i0 + kWarps * r;
      if (i0 + kWarps * r < BM && m < M) {
        const float4* src = reinterpret_cast<const float4*>(static_cast<const float*>(op.in[0]) +
                                                            size_t(m) * size_t(op.ld_in[0]));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[r][j] = src[lane + 32 * j];
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) v[r][j] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      const int i = i0 + kWarps * r;
      if (i >= BM) continue;
      float mean = 0.f, rstd = 0.f;
      if (m0 + i < M) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) s += v[r][j].x + v[r][j].y + v[r][j].z + v[r][j].w;
        mean = warp_sum(s) / float(kEncDim);
        float q = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float a = v[r][j].x - mean, b = v[r][j].y - mean, cc = v[r][j].z - mean, d = v[r][j].w - mean;
          q += a * a + b * b + cc * cc + d * d;
        }
        const float var = warp_sum(q) / float(kEncDim);
        rstd = 1.f / sqrtf(var + op.scalar[0]);
      }
      if (lane == 0) { s_mean[i] = mean; s_rstd[i] = rstd; }
    }
  }
}

// A operand kinds: PLAIN = row-major activations (A[m][k] = in0[m * ld + k]: every LayerNorm GEMM, the
// out-projection, FF2, the up-embed, the projection, the pos table); otherwise a gather (token embedding,
// the pre-lookahead and up-layer convs: per-row tails, taps, zero pads) through a_src.
__host__ __device__ inline bool gemm_a_is_plain(const OpDesc& op) {
  const auto k = static_cast<OpKind>(op.kind);
  return k != OpKind::kEncEmbed && k != OpKind::kEncPreLookahead && k != OpKind::kEncUpLayer;
}

// Tiles that can run wave-tail pieces (plain A only): the KG = 1 tiles the host picks for M > 64.
template <int BM, int BN, int KG>
inline constexpr bool kGemmPieceTile = KG == 1 && BN == 64 && (BM == 32 || BM == 64);

template <int BM, int BN, int TM, int TN, int BK, int KG, bool PLAIN, bool PIECES = false>
static __device__ void gemm_tile_impl(const OpDesc& op, const CallDesc& c, int tile, float* smem, bool pdl);

template <int BM, int BN, int TM, int TN, int BK, int KG>
static __device__ void gemm_tile(const OpDesc& op, const CallDesc& c, int tile, float* smem, bool pdl = false) {
  // One uniform branch per block; the plain instantiation has no a_src switch, no per-load row lookup
  // and no TF32 / gather code in its main loop (sprint 3).
  if (gemm_a_is_plain(op)) {
    if constexpr (kGemmPieceTile<BM, BN, KG>) {
      if (op.stride > 0) { gemm_tile_impl<BM, BN, TM, TN, BK, KG, true, true>(op, c, tile, smem, pdl); return; }
    }
    gemm_tile_impl<BM, BN, TM, TN, BK, KG, true>(op, c, tile, smem, pdl);
  } else {
    gemm_tile_impl<BM, BN, TM, TN, BK, KG, false>(op, c, tile, smem, pdl);
  }
}

template <int BM, int BN, int TM, int TN, int BK, int KG, bool PLAIN, bool PIECES>
static __device__ void gemm_tile_impl(const OpDesc& op, const CallDesc& c, int tile, float* smem, bool pdl) {
  // pdl: launched with programmatic dependent launch. Everything that does not read the previous kernel's
  // output (row map from the host-written descriptors, the first W tile) runs BEFORE griddepcontrol.wait,
  // so it overlaps the previous kernel's tail and this kernel's launch.
  constexpr int GT = kThreads / KG;  // threads per K group
  constexpr int TX = BN / TN;
  constexpr int TY = BM / TM;
  static_assert(TX * TY == GT, "thread tile must cover the group");
  static_assert(kInnerChunk % BK == 0, "BK must divide the inner chunk");
  constexpr int LDS = BK + 4;
  constexpr int QK = BK / 4;
  constexpr int A_LOADS = (BM * QK + GT - 1) / GT;
  constexpr int W_LOADS = (BN * QK + GT - 1) / GT;
  constexpr int SPC = kInnerChunk / BK;  // steps per inner chunk
  constexpr int TILES = KG * (BM + BN) * LDS;
  constexpr int RED = KG * BM * BN;
  constexpr int REGION = TILES > RED ? TILES : RED;

  const int dom = op_domain(op);
  const int M = dom_count(dom, op, c);
  const int N = op.n, K = op.k;
  const int mtiles = (op.m + BM - 1) / BM;
  const int ntiles = (N + BN - 1) / BN;
  const int splits = op.scalar[3] > 1.f ? int(op.scalar[3]) : 1;
  // Wave-tail pieces (sprint 3, KG = 1 tiles only): with op.stride = P > 0, tile-blocks [0, P) run whole and
  // every later tile-block runs as 4 blocks, one per 128-wide inner chunk; the last of the 4 to arrive folds
  // the chunk sums in order, ((i0 + i1) + i2) + i3, exactly as a whole block's tot does. Used when a grid
  // overruns its last full wave by a few tile-blocks (e.g. 384 64x64 tiles on 376 slots: the 8 extra tiles
  // cost a whole second block time; as 32 quarter blocks they cost a quarter).
  // PIECES is a separate kernel instantiation, so the ops without pieces pay no registers for it.
  int tb = tile, piece = -1;
  if constexpr (PIECES) {
    static_assert(kGemmPieceTile<BM, BN, KG> && PLAIN, "pieces: plain-A KG = 1 tiles only");
    if (tile >= op.stride) {
      tb = op.stride + (tile - op.stride) / 4;
      piece = (tile - op.stride) % 4;
    }
  }
  const int split = tb / (mtiles * ntiles);
  const int tile_mn = tb % (mtiles * ntiles);
  const int mt = tile_mn % mtiles, nt = tile_mn / mtiles;
  const int m0 = mt * BM, n0 = nt * BN;
  const int kc = K / splits, kbeg = split * kc;
  if (m0 >= M) return;
  const int t = threadIdx.x;
  const int g = t / GT, gt = t % GT;
  float* As = smem + g * (BM + BN) * LDS;  // this group's tiles
  float* Ws = As + BM * LDS;
  float* red = smem;                       // reused after the main loop: [KG][BM][BN]
  float* s_mean = smem + REGION;
  float* s_rstd = s_mean + BM;
  int* s_row = reinterpret_cast<int*>(s_rstd + BM);
  int* s_loc = s_row + BM;
  const bool has_rows = dom != kDomExplicit;
  for (int i = t; i < BM; i += kThreads) {
    const int m = m0 + i;
    if (m < M && has_rows) {
      const int ri = find_row(c, m, dom);
      s_row[i] = ri;
      s_loc[i] = m - row_off(c.rows[ri], ri, dom);
    } else {
      s_row[i] = m < M ? 0 : -1;
      s_loc[i] = 0;
    }
  }
  const bool tf32 = !PLAIN && op.math == static_cast<uint8_t>(MathMode::kTf32);  // TF32 is a conv (gather) option
  const float* W = static_cast<const float*>(op.w[0]);
  const int ty = gt / TX, tx = gt % TX;
  const int n_inner = kc / kInnerChunk;          // 4
  const int c0 = piece >= 0 ? piece : 0;         // first inner chunk of this block
  const int steps = piece >= 0 ? SPC : (n_inner / KG) * SPC;  // per group
  auto k_of = [&](int s) { return kbeg + (g + KG * (c0 + s / SPC)) * kInnerChunk + (s % SPC) * BK; };
  // Per-thread load slots (sprint 3): load j of thread gt is float4 column qd0 of tile row i0 + j * RSTEP,
  // for A and for W alike, at every K step. So the row pointers and the in-range masks are computed once,
  // not per load (v6 recomputed the slot, the row map lookup and the a_src switch for every float4: about
  // as many instructions per K step as the FMAs themselves, ncu n22g / SASS). Same addresses, same values.
  static_assert(GT % QK == 0, "a thread keeps one float4 column of the tile");
  constexpr int RSTEP = GT / QK;
  const int i0 = gt / QK, qd0 = gt % QK;
  const float* w_row = nullptr;
  size_t w_rs = 0;
  uint32_t w_ok = 0;
  if constexpr (PLAIN) {
    w_row = W + size_t(n0 + i0) * size_t(K) + qd0 * 4;
    w_rs = size_t(RSTEP) * size_t(K);
#pragma unroll
    for (int j = 0; j < W_LOADS; ++j)
      if (i0 + j * RSTEP < BN && n0 + i0 + j * RSTEP < N) w_ok |= 1u << j;
  }
  float4 ra[A_LOADS], rw[W_LOADS];
  auto load_w = [&](int k0) {
#pragma unroll
    for (int j = 0; j < W_LOADS; ++j) {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (PLAIN) {
        if ((w_ok >> j) & 1u) v = __ldg(reinterpret_cast<const float4*>(w_row + j * w_rs + k0));
      } else {  // gather kernels: v6 indexing (fewer live registers next to the a_src state)
        const int jn = i0 + j * RSTEP;
        if (jn < BN && n0 + jn < N) v = __ldg(reinterpret_cast<const float4*>(W + size_t(n0 + jn) * K + k0 + qd0 * 4));
      }
      rw[j] = v;
    }
  };
  load_w(k_of(0));  // weights never depend on the previous kernel
  // Plain A: rows m0 + i0 + j * RSTEP of in0 (zeros past M, as the row map's -1 gave before).
  const float* a_row = nullptr;
  size_t a_rs = 0;
  uint32_t a_ok = 0;
  if constexpr (PLAIN) {
    a_row = static_cast<const float*>(op.in[0]) + size_t(m0 + i0) * size_t(op.ld_in[0]) + qd0 * 4;
    a_rs = size_t(RSTEP) * size_t(op.ld_in[0]);
#pragma unroll
    for (int j = 0; j < A_LOADS; ++j)
      if (i0 + j * RSTEP < BM && m0 + i0 + j * RSTEP < M) a_ok |= 1u << j;
  }
  if (pdl) {
    griddep_wait();
    griddep_launch_dependents();
  }
  const bool ln = PLAIN && op_has_ln(op);
  const float* ln_g = static_cast<const float*>(op.w[2]);
  const float* ln_b = static_cast<const float*>(op.w[3]);
  // gamma / beta as float4 when both are 16-byte aligned (always, for the converter's and the check
  // driver's layouts); the scalar path stays for any other layout. Same values either way.
  const bool ln_v4 = ln && ((reinterpret_cast<uintptr_t>(ln_g) | reinterpret_cast<uintptr_t>(ln_b)) & 15u) == 0;
  if (ln) ln_row_stats<BM>(op, m0, M, s_mean, s_rstd);
  __syncthreads();

  // Global loads of one K step into registers; nothing consumes them here, so they stay in flight
  // while the previous step's FMAs run (LN / TF32 transforms happen at the shared-memory store).
  auto load_a = [&](int k0) {
#pragma unroll
    for (int j = 0; j < A_LOADS; ++j) {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (PLAIN) {
        if ((a_ok >> j) & 1u) v = *reinterpret_cast<const float4*>(a_row + j * a_rs + k0);
      } else {
        const int i = i0 + j * RSTEP;
        if (i < BM) {
          const int ri = s_row[i];
          if (ri >= 0) {
            const float* p = a_src(op, c, m0 + i, ri, s_loc[i], k0);
            if (p != nullptr) v = *reinterpret_cast<const float4*>(p + qd0 * 4);
          }
        }
      }
      ra[j] = v;
    }
  };
  auto store_smem = [&](int k0) {
#pragma unroll
    for (int j = 0; j < A_LOADS; ++j) {
      const int i = i0 + j * RSTEP;
      if (i < BM) {
        float4 v = ra[j];
        if (ln && ((a_ok >> j) & 1u)) {
          const int kk = k0 + qd0 * 4;
          const float mu = s_mean[i], rs = s_rstd[i];
          float4 g4, b4;
          if (ln_v4) {
            g4 = __ldg(reinterpret_cast<const float4*>(ln_g + kk));
            b4 = __ldg(reinterpret_cast<const float4*>(ln_b + kk));
          } else {
            g4 = make_float4(ln_g[kk + 0], ln_g[kk + 1], ln_g[kk + 2], ln_g[kk + 3]);
            b4 = make_float4(ln_b[kk + 0], ln_b[kk + 1], ln_b[kk + 2], ln_b[kk + 3]);
          }
          v.x = (v.x - mu) * rs * g4.x + b4.x;
          v.y = (v.y - mu) * rs * g4.y + b4.y;
          v.z = (v.z - mu) * rs * g4.z + b4.z;
          v.w = (v.w - mu) * rs * g4.w + b4.w;
        }
        if (tf32) { v.x = tf32_rna(v.x); v.y = tf32_rna(v.y); v.z = tf32_rna(v.z); v.w = tf32_rna(v.w); }
        *reinterpret_cast<float4*>(As + i * LDS + qd0 * 4) = v;
      }
    }
#pragma unroll
    for (int j = 0; j < W_LOADS; ++j) {
      const int jn = i0 + j * RSTEP;
      if (jn < BN) {
        float4 v = rw[j];
        if (tf32) { v.x = tf32_rna(v.x); v.y = tf32_rna(v.y); v.z = tf32_rna(v.z); v.w = tf32_rna(v.w); }
        *reinterpret_cast<float4*>(Ws + jn * LDS + qd0 * 4) = v;
      }
    }
  };

  float acc[TM][TN], tot[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) { acc[i][j] = 0.f; tot[i][j] = 0.f; }
  load_a(k_of(0));
  for (int s = 0; s < steps; ++s) {
    store_smem(k_of(s));
    __syncthreads();
    if (s + 1 < steps) {  // next step's global loads overlap this step's FMAs
      load_a(k_of(s + 1));
      load_w(k_of(s + 1));
    }
#pragma unroll 4
    for (int kk = 0; kk < BK; kk += 4) {
      float4 a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = *reinterpret_cast<const float4*>(As + (ty + i * TY) * LDS + kk);
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = *reinterpret_cast<const float4*>(Ws + (tx + j * TX) * LDS + kk);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = fmaf(a[i].x, b[j].x, acc[i][j]);
          acc[i][j] = fmaf(a[i].y, b[j].y, acc[i][j]);
          acc[i][j] = fmaf(a[i].z, b[j].z, acc[i][j]);
          acc[i][j] = fmaf(a[i].w, b[j].w, acc[i][j]);
        }
    }
    __syncthreads();
    if ((s + 1) % SPC == 0) {  // end of an inner chunk: fold it in chunk order, restart the chain
      const bool first = (s + 1) == SPC;
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          tot[i][j] = first ? acc[i][j] : tot[i][j] + acc[i][j];
          acc[i][j] = 0.f;
        }
    }
  }
  // Group partials to shared memory, then per-element ownership e = t + 256 r for the finish.
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) red[(g * BM + ty + i * TY) * BN + tx + j * TX] = tot[i][j];
  __syncthreads();
  constexpr int PER = (BM * BN + kThreads - 1) / kThreads;
  float vals[PER];
#pragma unroll
  for (int r = 0; r < PER; ++r) {
    const int e = t + r * kThreads;
    float v = 0.f;
    if (e < BM * BN) {
      v = red[e];
#pragma unroll
      for (int gg = 1; gg < KG; ++gg) v = v + red[gg * BM * BN + e];  // ((g0 + g1) + g2) + g3
    }
    vals[r] = v;
  }
  if (PIECES && piece >= 0) {
    // Inner-chunk sums to [tile-block][chunk][BM * BN] after any outer split-K partials; the last piece folds.
    const int r_tb = tb - op.stride;
    const size_t pbase = splits > 1 ? size_t(splits) * size_t(op.m) * size_t(N) : 0;
    float* pp = static_cast<float*>(op.out[1]) + pbase + size_t(r_tb) * 4 * (BM * BN);
    unsigned int* pc = static_cast<unsigned int*>(const_cast<void*>(op.in[3])) + kPieceCounterBase + r_tb;
#pragma unroll
    for (int r = 0; r < PER; ++r) {
      const int e = t + r * kThreads;
      if (e < BM * BN) __stcg(pp + size_t(piece) * (BM * BN) + e, vals[r]);
    }
    __threadfence();
    __syncthreads();
    __shared__ int s_last_piece;
    if (t == 0) {
      const unsigned int prev = atomicAdd(pc, 1u);
      s_last_piece = prev == 3u;
      if (s_last_piece) *pc = 0u;
    }
    __syncthreads();
    if (!s_last_piece) return;
    __threadfence();
#pragma unroll
    for (int r = 0; r < PER; ++r) {
      const int e = t + r * kThreads;
      if (e < BM * BN) {
        float v = __ldcg(pp + e);
        v = v + __ldcg(pp + (BM * BN) + e);
        v = v + __ldcg(pp + 2 * (BM * BN) + e);
        v = v + __ldcg(pp + 3 * (BM * BN) + e);
        vals[r] = v;
      }
    }
  }
  if (splits > 1) {
    // Fixed split-K: every split writes its outer partial; the last block to arrive sums them in split
    // order, whichever block that is. The counter resets itself for the next op / graph replay.
    float* part = static_cast<float*>(op.out[1]);
    unsigned int* cnt = static_cast<unsigned int*>(const_cast<void*>(op.in[3]));
#pragma unroll
    for (int r = 0; r < PER; ++r) {
      const int e = t + r * kThreads;
      const int m = m0 + e / BN, n = n0 + e % BN;
      if (e < BM * BN && m < M && n < N) __stcg(part + (size_t(split) * op.m + m) * N + n, vals[r]);
    }
    __threadfence();
    __syncthreads();
    __shared__ int s_last;
    if (t == 0) {
      const unsigned int prev = atomicAdd(cnt + tile_mn, 1u);
      s_last = prev == unsigned(splits - 1);
      if (s_last) cnt[tile_mn] = 0u;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
#pragma unroll
    for (int r = 0; r < PER; ++r) {
      const int e = t + r * kThreads;
      const int m = m0 + e / BN, n = n0 + e % BN;
      if (e < BM * BN && m < M && n < N) {
        float v = __ldcg(part + size_t(m) * N + n);
        for (int sp = 1; sp < splits; ++sp) v += __ldcg(part + (size_t(sp) * op.m + m) * N + n);
        vals[r] = v;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < PER; ++r) {
    const int e = t + r * kThreads;
    if (e >= BM * BN) continue;
    const int li = e / BN;
    const int m = m0 + li, n = n0 + e % BN;
    if (m < M && n < N) gemm_epilogue(op, c, m, n, vals[r], s_row[li], s_loc[li]);
  }
}

// ---------------------------------------------------------------------------------------------
// Large-M GEMM tile (throughput path, B >= ~8 rows): same canonical reduction order as gemm_tile
// (one k-ascending fmaf chain per 128-wide inner chunk, folded in order; fixed split-K), so it gives
// the same bits as every other configuration. Differences from gemm_tile, for FMA throughput:
//   * shared tiles stored k-major (As[k][m], Ws[k][n], rows padded by 4 floats): a thread reads its
//     8 rows / 8 columns at one k as two float4 each (4 LDS.128 per 64 FMA);
//   * global loads: consecutive lanes read consecutive float4 along k of one row (coalesced; the
//     first version mapped lanes to rows, 32 rows per warp load); with BK = 8 (2 quads per row) the
//     transposing scalar stores land in banks 16 * quad + 4 * c + row: conflict free with the pad;
//   * two shared buffers: one __syncthreads per K step instead of two;
//   * the thread grid is 16 x 16; thread (ty, tx) owns rows {ty*4 + i, BM/2 + ty*4 + i} and columns
//     {tx*4 + j, BN/2 + tx*4 + j} (TM = BM / 16, TN = BN / 16; halves only when TM / TN = 8).
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int BK>
constexpr uint32_t gemm_t_smem_bytes() {
  constexpr int tiles = 2 * BK * (BM + 4 + BN + 4);
  constexpr int red = BM * BN;
  return uint32_t(((tiles > red ? tiles : red) + 2 * BM) * sizeof(float) + 2 * BM * sizeof(int));
}

template <int BM, int BN, int BK>
static __device__ void gemm_tile_t(const OpDesc& op, const CallDesc& c, int tile, float* smem, bool pdl = false) {
  constexpr int TM = BM / 16, TN = BN / 16;
  static_assert(TM == 4 || TM == 8, "TM");
  static_assert(TN == 4 || TN == 8, "TN");
  static_assert(kInnerChunk % BK == 0 && BK % 4 == 0, "BK");
  constexpr int QK = BK / 4;                      // float4 quads per row per K step
  constexpr int A_Q = BM * QK, W_Q = BN * QK;     // quads per step
  constexpr int A_LOADS = (A_Q + kThreads - 1) / kThreads;
  constexpr int W_LOADS = (W_Q + kThreads - 1) / kThreads;
  constexpr int SPC = kInnerChunk / BK;
  constexpr int SA = BM + 4, SW = BN + 4;         // padded k-major row strides
  constexpr int BUF = BK * (SA + SW);
  constexpr int TILES = 2 * BUF;
  constexpr int RED = BM * BN;
  constexpr int REGION = TILES > RED ? TILES : RED;
  constexpr int AH = TM == 8 ? BM / 2 : 0;        // row offset of the second float4 group
  constexpr int WH = TN == 8 ? BN / 2 : 0;

  const int dom = op_domain(op);
  const int M = dom_count(dom, op, c);
  const int N = op.n, K = op.k;
  const int mtiles = (op.m + BM - 1) / BM;
  const int ntiles = (N + BN - 1) / BN;
  const int splits = op.scalar[3] > 1.f ? int(op.scalar[3]) : 1;
  const int split = tile / (mtiles * ntiles);
  const int tile_mn = tile % (mtiles * ntiles);
  const int mt = tile_mn % mtiles, nt = tile_mn / mtiles;
  const int m0 = mt * BM, n0 = nt * BN;
  const int kc = K / splits, kbeg = split * kc;
  if (m0 >= M) return;
  const int t = threadIdx.x;
  float* red = smem;
  float* s_mean = smem + REGION;
  float* s_rstd = s_mean + BM;
  int* s_row = reinterpret_cast<int*>(s_rstd + BM);
  int* s_loc = s_row + BM;
  const bool has_rows = dom != kDomExplicit;
  for (int i = t; i < BM; i += kThreads) {
    const int m = m0 + i;
    if (m < M && has_rows) {
      const int ri = find_row(c, m, dom);
      s_row[i] = ri;
      s_loc[i] = m - row_off(c.rows[ri], ri, dom);
    } else {
      s_row[i] = m < M ? 0 : -1;
      s_loc[i] = 0;
    }
  }
  const bool tf32 = op.math == static_cast<uint8_t>(MathMode::kTf32);
  const float* W = static_cast<const float*>(op.w[0]);
  const int steps = (kc / kInnerChunk) * SPC;
  auto k_of = [&](int s) { return kbeg + s * BK; };  // KG = 1: inner chunks in order
  float4 ra[A_LOADS], rw[W_LOADS];
  // quad q of a step: row q / QK, k quad q % QK (consecutive lanes -> consecutive quads of a row)
  auto load_w = [&](int k0) {
#pragma unroll
    for (int j = 0; j < W_LOADS; ++j) {
      const int q = t + j * kThreads;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (q < W_Q) {
        const int jn = q / QK, qd = q % QK;
        if (n0 + jn < N) v = __ldg(reinterpret_cast<const float4*>(W + size_t(n0 + jn) * K + k0 + qd * 4));
      }
      rw[j] = v;
    }
  };
  load_w(k_of(0));
  if (pdl) {
    griddep_wait();
    griddep_launch_dependents();
  }
  const bool ln = op_has_ln(op);
  const float* ln_g = static_cast<const float*>(op.w[2]);
  const float* ln_b = static_cast<const float*>(op.w[3]);
  if (ln) ln_row_stats<BM>(op, m0, M, s_mean, s_rstd);
  __syncthreads();
  auto load_a = [&](int k0) {
#pragma unroll
    for (int j = 0; j < A_LOADS; ++j) {
      const int q = t + j * kThreads;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (q < A_Q) {
        const int i = q / QK, qd = q % QK;
        const int ri = s_row[i];
        if (ri >= 0) {
          const float* p = a_src(op, c, m0 + i, ri, s_loc[i], k0);
          if (p != nullptr) v = *reinterpret_cast<const float4*>(p + qd * 4);
        }
      }
      ra[j] = v;
    }
  };
  auto store_smem = [&](int k0, float* buf) {
    float* As = buf;            // [BK][SA]
    float* Ws = buf + BK * SA;  // [BK][SW]
#pragma unroll
    for (int j = 0; j < A_LOADS; ++j) {
      const int q = t + j * kThreads;
      if (q < A_Q) {
        const int i = q / QK, qd = q % QK;
        float4 v = ra[j];
        if (ln && s_row[i] >= 0) {
          const int kk = k0 + qd * 4;
          const float mu = s_mean[i], rs = s_rstd[i];
          v.x = (v.x - mu) * rs * ln_g[kk + 0] + ln_b[kk + 0];
          v.y = (v.y - mu) * rs * ln_g[kk + 1] + ln_b[kk + 1];
          v.z = (v.z - mu) * rs * ln_g[kk + 2] + ln_b[kk + 2];
          v.w = (v.w - mu) * rs * ln_g[kk + 3] + ln_b[kk + 3];
        }
        if (tf32) { v.x = tf32_rna(v.x); v.y = tf32_rna(v.y); v.z = tf32_rna(v.z); v.w = tf32_rna(v.w); }
        As[(qd * 4 + 0) * SA + i] = v.x;
        As[(qd * 4 + 1) * SA + i] = v.y;
        As[(qd * 4 + 2) * SA + i] = v.z;
        As[(qd * 4 + 3) * SA + i] = v.w;
      }
    }
#pragma unroll
    for (int j = 0; j < W_LOADS; ++j) {
      const int q = t + j * kThreads;
      if (q < W_Q) {
        const int jn = q / QK, qd = q % QK;
        float4 v = rw[j];
        if (tf32) { v.x = tf32_rna(v.x); v.y = tf32_rna(v.y); v.z = tf32_rna(v.z); v.w = tf32_rna(v.w); }
        Ws[(qd * 4 + 0) * SW + jn] = v.x;
        Ws[(qd * 4 + 1) * SW + jn] = v.y;
        Ws[(qd * 4 + 2) * SW + jn] = v.z;
        Ws[(qd * 4 + 3) * SW + jn] = v.w;
      }
    }
  };

  const int ty = t >> 4, tx = t & 15;
  float acc[TM][TN], tot[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) { acc[i][j] = 0.f; tot[i][j] = 0.f; }
  load_a(k_of(0));
  store_smem(k_of(0), smem);
  __syncthreads();
  for (int s = 0; s < steps; ++s) {
    const float* As = smem + (s & 1) * BUF;
    const float* Ws = As + BK * SA;
    if (s + 1 < steps) {  // next step's global loads overlap this step's FMAs
      load_a(k_of(s + 1));
      load_w(k_of(s + 1));
    }
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
      {
        const float4 a0 = *reinterpret_cast<const float4*>(As + kk * SA + ty * 4);
        a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
        if constexpr (TM == 8) {
          const float4 a1 = *reinterpret_cast<const float4*>(As + kk * SA + AH + ty * 4);
          a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
        }
        const float4 b0 = *reinterpret_cast<const float4*>(Ws + kk * SW + tx * 4);
        b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
        if constexpr (TN == 8) {
          const float4 b1 = *reinterpret_cast<const float4*>(Ws + kk * SW + WH + tx * 4);
          b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
        }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    if (s + 1 < steps) store_smem(k_of(s + 1), smem + ((s + 1) & 1) * BUF);
    __syncthreads();
    if ((s + 1) % SPC == 0) {  // end of an inner chunk: fold it in chunk order, restart the chain
      const bool first = (s + 1) == SPC;
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          tot[i][j] = first ? acc[i][j] : tot[i][j] + acc[i][j];
          acc[i][j] = 0.f;
        }
    }
  }
  // Results to shared memory (float4 rows), then per-element ownership e = t + 256 r for the finish.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = (i < 4 ? 0 : AH) + ty * 4 + (i & 3);
#pragma unroll
    for (int jh = 0; jh < TN / 4; ++jh) {
      float4 v;
      v.x = tot[i][jh * 4 + 0]; v.y = tot[i][jh * 4 + 1]; v.z = tot[i][jh * 4 + 2]; v.w = tot[i][jh * 4 + 3];
      *reinterpret_cast<float4*>(red + row * BN + jh * WH + tx * 4) = v;
    }
  }
  __syncthreads();
  constexpr int PER = (BM * BN + kThreads - 1) / kThreads;
  if (splits > 1) {
    float* part = static_cast<float*>(op.out[1]);
    unsigned int* cnt = static_cast<unsigned int*>(const_cast<void*>(op.in[3]));
#pragma unroll 4
    for (int r = 0; r < PER; ++r) {
      const int e = t + r * kThreads;
      const int m = m0 + e / BN, n = n0 + e % BN;
      if (e < BM * BN && m < M && n < N) __stcg(part + (size_t(split) * op.m + m) * N + n, red[e]);
    }
    __threadfence();
    __syncthreads();
    __shared__ int s_last;
    if (t == 0) {
      const unsigned int prev = atomicAdd(cnt + tile_mn, 1u);
      s_last = prev == unsigned(splits - 1);
      if (s_last) cnt[tile_mn] = 0u;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
#pragma unroll 4
    for (int r = 0; r < PER; ++r) {
      const int e = t + r * kThreads;
      const int li = e / BN;
      const int m = m0 + li, n = n0 + e % BN;
      if (e < BM * BN && m < M && n < N) {
        float v = __ldcg(part + size_t(m) * N + n);
        for (int sp = 1; sp < splits; ++sp) v += __ldcg(part + (size_t(sp) * op.m + m) * N + n);
        gemm_epilogue(op, c, m, n, v, s_row[li], s_loc[li]);
      }
    }
    return;
  }
#pragma unroll 4
  for (int r = 0; r < PER; ++r) {
    const int e = t + r * kThreads;
    if (e >= BM * BN) continue;
    const int li = e / BN;
    const int m = m0 + li, n = n0 + e % BN;
    if (m < M && n < N) gemm_epilogue(op, c, m, n, red[e], s_row[li], s_loc[li]);
  }
}

// ---------------------------------------------------------------------------------------------
// Rel-pos chunk-causal attention tile: (row, head, 16-query tile), keys in tiles of 64 from the
// pool pages, online softmax in fp32, keys ascending (fixed order per element).
// ---------------------------------------------------------------------------------------------
inline constexpr int kKs = kAttnKTile + 4;  // padded K / P row stride: float4-aligned; LDS.128 over 8 rows is conflict free
inline constexpr int kAttnPartStride = 68;  // per-query partial state: acc[64], m, l (+2 pad keeps float4 alignment)
inline constexpr int kPRows = kAttnQTile + kAttnKTile - 1;
// Shared memory: Qu, Qv [16][64]; one K-then-V buffer [64][68]; one P-rows-then-probabilities buffer
// [79][68]. 47,088 bytes, so two blocks fit in sm_120's 100 KB per SM (v6 kept K, V, P rows and the
// probabilities apart: 67,568 bytes, one block per SM, 8 warps; ncu n12a: occupancy 16.7%, est. +52%).
inline constexpr uint32_t kAttnSmemBytes = uint32_t((2 * kAttnQTile * 64 + kAttnKTile * kKs + kPRows * kKs) * sizeof(float));
// (Padding the probability rows to 80 floats removes a 2-way store conflict ncu flagged, but measured no
// gain: attention 1,345 vs 1,349 us at B32 first, p22o vs p22n. Kept unpadded.)
inline constexpr int kPpStride = kAttnKTile;
static_assert(kAttnQTile * kPpStride <= kPRows * kKs, "probabilities alias the P rows");
// Score schedule. 1 (default, sprint 2): the scores are two small register-tiled GEMMs (Qv . P rows -> raw
// [16][80], Qu . K -> ac [16][64]) and s = (ac + raw[q][q - k + 63]) / 8 is gathered through shared memory;
// every score is the same pair of d-ascending fmaf chains as schedule 0, so the bits are identical. Schedule 0
// (v6): one query x 4 keys per thread, K and P rows read per score (~2x the shared-memory wavefronts per FMA).
#ifndef KT2N_ENC_ATTN_V2
#define KT2N_ENC_ATTN_V2 1
#endif
inline constexpr int kRawStride = 88;  // raw rows: the 2 query pairs of a warp land 16 banks apart
inline constexpr int kSStride = 80;    // score / probability rows
static_assert(kAttnQTile * (kRawStride + kSStride) <= kPRows * kKs, "raw + scores alias the P rows");
static_assert(kAttnKTile * 64 <= kAttnKTile * kKs, "V aliases K");

// INBLOCK = false: a per-split-only instantiation (ops with op.stride == 0), with no in-block / mixed code; the
// in-block bookkeeping cost the per-split ops 1-3% (first B16 / B8 steady / first B20 attention, `p38`).
template <bool INBLOCK = true>
static __device__ void attn_tile(const OpDesc& op, const CallDesc& c, int tile, float* smem, bool pdl = false) {
  const bool frames = (op.flags & kOpFrames) != 0;
  const int chunk = frames ? kMelChunk : kTokenChunk;
  const int qtiles = op.tile_n;
  // Key splits: split g of a (row, head, query tile) takes key tiles g, g + G, g + 2G, ... and the G
  // partial softmax states are merged in g order (fixed structure: G is a constant of the op, so the
  // bits do not depend on the batch or on timing). Two schedules give the same bits:
  //   * op.stride == 0: one block per split; the last block to finish merges (small grids);
  //   * op.stride == 1 (kAttnSplitsInBlock): one block runs the G splits one after the other and
  //     merges them itself (large grids: G x fewer blocks, Q and the first P rows loaded once, no
  //     arrival counters).
  //   * mixed (sprint 3, op.stride == 1 and op.kernel = F > 0): groups [0, F) run in-block, the groups after F
  //     (the last, partial wave of in-block groups) one block per split, so that tail costs a few 1/G block
  //     times instead of one whole in-block block time. Both schedules merge the same partials in g order.
  const int G = op.dilation > 1 ? op.dilation : 1;
  const int F = (INBLOCK && op.stride == 1) ? (op.kernel > 0 ? op.kernel : 0x7fffffff) : 0;  // groups run in-block
  const bool inblock = tile < F;
  const int tq = inblock ? tile : F + (tile - F) / G;
  const int g_lo = inblock ? 0 : (tile - F) % G;
  const int g_hi = inblock ? G : g_lo + 1;
  const int ri = tq / (kEncHeads * qtiles);
  const int rem = tq % (kEncHeads * qtiles);
  const int h = rem / qtiles, qt = rem % qtiles;
  if (ri >= c.n_rows) return;
  const RowDesc& r = c.rows[ri];
  const int S = frames ? r.fr_suffix : r.tok_suffix;
  const int past = frames ? r.fr_past : r.tok_past;
  const int off = frames ? r.fr_off : r.tok_off;
  const int q0 = qt * kAttnQTile;
  if (q0 >= S) return;
  const int nq = min(kAttnQTile, S - q0);
  const bool full = (r.flags & kRowFullAttention) != 0;
  const int keys_total = past + S;
  const int q_last = past + q0 + nq - 1;
  const int kend = full ? keys_total : min(keys_total, (q_last / chunk + 1) * chunk);
  const int maxd = op.k;
  const float* q = static_cast<const float*>(op.in[0]);
  const float* ptab = static_cast<const float*>(op.w[0]);
  const float* pu = static_cast<const float*>(op.w[1]) + h * kEncHeadDim;
  const float* pv = static_cast<const float*>(op.w[2]) + h * kEncHeadDim;

  float* Qu = smem;                        // [16][64]
  float* Qv = Qu + kAttnQTile * 64;        // [16][64]
  float* Ks = Qv + kAttnQTile * 64;        // [64][68] K tile for the scores ...
  float* Vs = Ks;                          // ... then [64][64] V tile for the PV loop
  float* Ps = Ks + kAttnKTile * kKs;       // [79][68] P rows for the scores ...
  [[maybe_unused]] float* Pp = Ps;                          // ... then [16][64] probabilities
  const int t = threadIdx.x;
  // Relative-position rows of one key tile (the P table is a load-time constant).
  auto load_p = [&](int k0) {
    const int dmin = past + q0 - (k0 + kAttnKTile - 1);
    for (int e = t; e < kPRows * 16; e += kThreads) {
      const int pr = e >> 4, q4 = e & 15;
      const int d = dmin + pr;
      float4 pvv = make_float4(0.f, 0.f, 0.f, 0.f);
      if (d > -maxd && d < maxd)
        pvv = *reinterpret_cast<const float4*>(ptab + size_t(d + maxd - 1) * kEncDim + h * kEncHeadDim + q4 * 4);
      *reinterpret_cast<float4*>(Ps + pr * kKs + q4 * 4) = pvv;
    }
  };
  // K tile to shared memory; the V tile stays in registers until the scores are done (its loads overlap
  // the score loop), then goes into the same buffer.
  constexpr int kKvPer = kAttnKTile * 16 / kThreads;  // 4 float4 per thread
  float4 vreg[kKvPer];
  auto load_kv = [&](int k0) {
#pragma unroll
    for (int j = 0; j < kKvPer; ++j) {
      const int e = t + j * kThreads;
      const int key = e >> 4, q4 = e & 15;
      const int kabs = k0 + key;
      float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
      if (kabs < kend) {
        kv = *reinterpret_cast<const float4*>(kv_ptr(c, r, frames, op.layer, 0, kabs) + h * kEncHeadDim + q4 * 4);
        vv = *reinterpret_cast<const float4*>(kv_ptr(c, r, frames, op.layer, 1, kabs) + h * kEncHeadDim + q4 * 4);
      }
      *reinterpret_cast<float4*>(Ks + key * kKs + q4 * 4) = kv;
      vreg[j] = vv;
    }
  };
  if (g_lo * kAttnKTile < kend) load_p(g_lo * kAttnKTile);  // before the PDL wait: constants and host descriptors only
  // Keys before the row's past were written by earlier calls, never by the kernel this one waits for (that
  // one writes positions >= past), so a first key tile that lies entirely in the past loads before the wait.
  const bool kv_early = g_lo * kAttnKTile + kAttnKTile <= past;
  if (kv_early) load_kv(g_lo * kAttnKTile);
  if (pdl) {
    griddep_wait();
    griddep_launch_dependents();
  }
  for (int e = t; e < kAttnQTile * 64; e += kThreads) {
    const int qi = e >> 6, d = e & 63;
    float u = 0.f, v = 0.f;
    if (qi < nq) {
      const float qq = q[size_t(off + q0 + qi) * kEncDim + h * kEncHeadDim + d];
      u = qq + pu[d];
      v = qq + pv[d];
    }
    Qu[e] = u;
    Qv[e] = v;
  }
  const int qi = t >> 4;   // query of this thread (scores and output)
  const int kl = t & 15;   // key lane (scores) / dim group (output)
  [[maybe_unused]] const int q_abs = past + q0 + qi;
  float m_run = -INFINITY, l_run = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int g = g_lo; g < g_hi; ++g) {
  const int k_first = g * kAttnKTile;
  m_run = -INFINITY;
  l_run = 0.f;
  acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
  for (int k0 = k_first; k0 < kend; k0 += G * kAttnKTile) {
    __syncthreads();  // the previous tile's PV loop is done with Vs / Pp
    if (!(kv_early && g == g_lo && k0 == k_first)) load_kv(k0);
    [[maybe_unused]] const int dmin = past + q0 - (k0 + kAttnKTile - 1);
    if (g != g_lo || k0 != k_first) load_p(k0);
    __syncthreads();
    float s[4];
    float tmax = -INFINITY;
#if KT2N_ENC_ATTN_V2
    float* RAW = Ps;                            // [16][88] once the P rows are consumed
    float* S = Ps + kAttnQTile * kRawStride;    // [16][80] scores, then the probabilities in place
    {
      const int tt = t & 127;
      const int qa = (tt >> 4) * 2;             // this thread's queries qa, qa + 1
      const int lg = tt & 15;                   // rows lg + 16 v (raw) or keys lg + 16 u (ac)
      float r2[2][5], a2[2][4];
      if (t < 128) {                            // warps 0-3: raw = Qv . P rows (the bd chains)
        const float* pr[5];
#pragma unroll
        for (int v = 0; v < 5; ++v) pr[v] = Ps + min(lg + 16 * v, kPRows - 1) * kKs;
#pragma unroll
        for (int v = 0; v < 5; ++v) { r2[0][v] = 0.f; r2[1][v] = 0.f; }
#pragma unroll 4
        for (int d = 0; d < 64; d += 4) {
          const float4 v0 = *reinterpret_cast<const float4*>(Qv + qa * 64 + d);
          const float4 v1 = *reinterpret_cast<const float4*>(Qv + (qa + 1) * 64 + d);
#pragma unroll
          for (int v = 0; v < 5; ++v) {
            const float4 p4 = *reinterpret_cast<const float4*>(pr[v] + d);
            r2[0][v] = fmaf(v0.x, p4.x, r2[0][v]); r2[0][v] = fmaf(v0.y, p4.y, r2[0][v]);
            r2[0][v] = fmaf(v0.z, p4.z, r2[0][v]); r2[0][v] = fmaf(v0.w, p4.w, r2[0][v]);
            r2[1][v] = fmaf(v1.x, p4.x, r2[1][v]); r2[1][v] = fmaf(v1.y, p4.y, r2[1][v]);
            r2[1][v] = fmaf(v1.z, p4.z, r2[1][v]); r2[1][v] = fmaf(v1.w, p4.w, r2[1][v]);
          }
        }
      } else {                                  // warps 4-7: ac = Qu . K (the ac chains)
        const float* kr[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) kr[u] = Ks + (lg + 16 * u) * kKs;
#pragma unroll
        for (int u = 0; u < 4; ++u) { a2[0][u] = 0.f; a2[1][u] = 0.f; }
#pragma unroll 4
        for (int d = 0; d < 64; d += 4) {
          const float4 u0 = *reinterpret_cast<const float4*>(Qu + qa * 64 + d);
          const float4 u1 = *reinterpret_cast<const float4*>(Qu + (qa + 1) * 64 + d);
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float4 k4 = *reinterpret_cast<const float4*>(kr[u] + d);
            a2[0][u] = fmaf(u0.x, k4.x, a2[0][u]); a2[0][u] = fmaf(u0.y, k4.y, a2[0][u]);
            a2[0][u] = fmaf(u0.z, k4.z, a2[0][u]); a2[0][u] = fmaf(u0.w, k4.w, a2[0][u]);
            a2[1][u] = fmaf(u1.x, k4.x, a2[1][u]); a2[1][u] = fmaf(u1.y, k4.y, a2[1][u]);
            a2[1][u] = fmaf(u1.z, k4.z, a2[1][u]); a2[1][u] = fmaf(u1.w, k4.w, a2[1][u]);
          }
        }
      }
      __syncthreads();  // P rows and the K tile are consumed
      if (t < 128) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int v = 0; v < 5; ++v)
            if (lg + 16 * v < kPRows) RAW[(qa + i) * kRawStride + lg + 16 * v] = r2[i][v];
      }
#pragma unroll
      for (int j = 0; j < kKvPer; ++j) {        // V from registers into the K buffer
        const int e = t + j * kThreads;
        *reinterpret_cast<float4*>(Vs + (e >> 4) * 64 + (e & 15) * 4) = vreg[j];
      }
      __syncthreads();
      if (t >= 128) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int q = qa + i;
          const int qa_abs = past + q0 + q;
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int kidx = lg + 16 * u;
            const int kabs = k0 + kidx;
            const bool vis = q < nq && kabs < kend && (full || kabs < past || (kabs / chunk) <= (qa_abs / chunk));
            S[q * kSStride + kidx] = vis ? (a2[i][u] + RAW[q * kRawStride + q - kidx + kAttnKTile - 1]) * 0.125f : -INFINITY;
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        s[u] = S[qi * kSStride + kl + 16 * u];
        tmax = fmaxf(tmax, s[u]);
      }
    }
#else
    // Scores of this thread's query against keys kl + 16u (u = 0..3): float4 shared-memory reads, d ascending.
    {
      bool vis[4];
      const float* kr[4];
      const float* prr[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int kidx = kl + 16 * u;
        const int kabs = k0 + kidx;
        vis[u] = qi < nq && kabs < kend && (full || kabs < past || (kabs / chunk) <= (q_abs / chunk));
        kr[u] = Ks + kidx * kKs;
        prr[u] = Ps + (vis[u] ? (q_abs - kabs - dmin) : 0) * kKs;
      }
      float ac[4] = {0.f, 0.f, 0.f, 0.f}, bd[4] = {0.f, 0.f, 0.f, 0.f};
      const float* qu = Qu + qi * 64;
      const float* qv = Qv + qi * 64;
#pragma unroll 4
      for (int d = 0; d < 64; d += 4) {
        const float4 u4 = *reinterpret_cast<const float4*>(qu + d);
        const float4 v4 = *reinterpret_cast<const float4*>(qv + d);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const float4 k4 = *reinterpret_cast<const float4*>(kr[u] + d);
          const float4 p4 = *reinterpret_cast<const float4*>(prr[u] + d);
          ac[u] = fmaf(u4.x, k4.x, ac[u]); ac[u] = fmaf(u4.y, k4.y, ac[u]);
          ac[u] = fmaf(u4.z, k4.z, ac[u]); ac[u] = fmaf(u4.w, k4.w, ac[u]);
          bd[u] = fmaf(v4.x, p4.x, bd[u]); bd[u] = fmaf(v4.y, p4.y, bd[u]);
          bd[u] = fmaf(v4.z, p4.z, bd[u]); bd[u] = fmaf(v4.w, p4.w, bd[u]);
        }
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        s[u] = vis[u] ? (ac[u] + bd[u]) * 0.125f : -INFINITY;  // / sqrt(64), exact
        tmax = fmaxf(tmax, s[u]);
      }
    }
#endif
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    const float m_new = fmaxf(m_run, tmax);
    float psum = 0.f;
    float pr4[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float p = (m_new == -INFINITY || s[u] == -INFINITY) ? 0.f : expf(s[u] - m_new);
      pr4[u] = p;
      psum += p;
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
    const float scale = (m_run == -INFINITY) ? 0.f : expf(m_run - m_new);
    l_run = l_run * scale + psum;
    m_run = m_new;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) acc[cc] *= scale;
#if KT2N_ENC_ATTN_V2
    // Probabilities in place of the scores: each thread rewrites exactly the entries it read. V is stored.
#pragma unroll
    for (int u = 0; u < 4; ++u) S[qi * kSStride + kl + 16 * u] = pr4[u];
    __syncthreads();
    const float* prow = S + qi * kSStride;
#else
    __syncthreads();  // every thread is done reading Ks and Ps: reuse them for V and the probabilities
#pragma unroll
    for (int u = 0; u < 4; ++u) Pp[qi * kPpStride + kl + 16 * u] = pr4[u];
#pragma unroll
    for (int j = 0; j < kKvPer; ++j) {
      const int e = t + j * kThreads;
      *reinterpret_cast<float4*>(Vs + (e >> 4) * 64 + (e & 15) * 4) = vreg[j];
    }
    __syncthreads();
    const float* prow = Pp + qi * kPpStride;
#endif
#pragma unroll 8
    for (int kk = 0; kk < kAttnKTile; ++kk) {
      const float p = prow[kk];
      const float4 vv = *reinterpret_cast<const float4*>(Vs + kk * 64 + kl * 4);
      acc[0] = fmaf(p, vv.x, acc[0]);
      acc[1] = fmaf(p, vv.y, acc[1]);
      acc[2] = fmaf(p, vv.z, acc[2]);
      acc[3] = fmaf(p, vv.w, acc[3]);
    }
  }
  if (G > 1) {  // this split's partial state (acc, m, l per query)
    float* part = static_cast<float*>(op.out[1]) + (size_t(tq) * G + g) * (kAttnQTile * kAttnPartStride);
    float* pq = part + qi * kAttnPartStride;
    __stcg(reinterpret_cast<float4*>(pq + kl * 4), make_float4(acc[0], acc[1], acc[2], acc[3]));
    if (kl == 0) { __stcg(pq + 64, m_run); __stcg(pq + 65, l_run); }
  }
  }  // splits
  if (G > 1) {
    if (inblock) {
      __syncthreads();  // block-local: the partials this block wrote are visible to all its threads
    } else {
      unsigned int* cnt = static_cast<unsigned int*>(const_cast<void*>(op.in[3]));
      __threadfence();
      __syncthreads();
      __shared__ int s_last;
      if (threadIdx.x == 0) {
        const unsigned int prev = atomicAdd(cnt + tq, 1u);
        s_last = prev == unsigned(G - 1);
        if (s_last) cnt[tq] = 0u;
      }
      __syncthreads();
      if (!s_last) return;
      __threadfence();
    }
    const float* base = static_cast<const float*>(op.out[1]) + size_t(tq) * G * (kAttnQTile * kAttnPartStride) + qi * kAttnPartStride;
    float mmax = -INFINITY;
    for (int gg = 0; gg < G; ++gg) mmax = fmaxf(mmax, __ldcg(base + size_t(gg) * kAttnQTile * kAttnPartStride + 64));
    float l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int gg = 0; gg < G; ++gg) {
      const float* pg = base + size_t(gg) * kAttnQTile * kAttnPartStride;
      const float mg = __ldcg(pg + 64);
      const float wgt = (mg == -INFINITY) ? 0.f : expf(mg - mmax);
      const float4 ag = __ldcg(reinterpret_cast<const float4*>(pg + kl * 4));
      l = fmaf(__ldcg(pg + 65), wgt, l);
      a0 = fmaf(ag.x, wgt, a0); a1 = fmaf(ag.y, wgt, a1); a2 = fmaf(ag.z, wgt, a2); a3 = fmaf(ag.w, wgt, a3);
    }
    l_run = l;
    acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
  }
  if (qi < nq) {
    float* out = static_cast<float*>(op.out[0]) + size_t(off + q0 + qi) * kEncDim + h * kEncHeadDim + kl * 4;
    const float inv = 1.f / l_run;
    float4 o;
    o.x = acc[0] * inv; o.y = acc[1] * inv; o.z = acc[2] * inv; o.w = acc[3] * inv;
    *reinterpret_cast<float4*>(out) = o;
  }
}

// ---------------------------------------------------------------------------------------------
// Row ops: LayerNorm rows (embed / up_embed), tails, speaker affine.
// ---------------------------------------------------------------------------------------------
static __device__ void embed_norm_tile(const OpDesc& op, const CallDesc& c, int tile) {
  const int dom = op_domain(op);
  const int M = dom_count(dom, op, c);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m = tile * (kThreads / 32) + warp;
  if (m >= M) return;
  float* dst = static_cast<float*>(op.out[0]) + size_t(m) * kEncDim;
  const float* src;
  if (dom == kDomExt) {
    const int ri = find_row(c, m, dom);
    const RowDesc& r = c.rows[ri];
    const int loc = m - (r.tok_off + 2 * ri);
    if (loc < 2) {  // the slot's pre-lookahead tail (zeros at the sequence start)
      const float* tail = c.state.enc_pre_tail + (size_t(r.slot) * 2 + loc) * kEncDim;
      for (int e = lane; e < kEncDim / 4; e += 32) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r.tok_past > 0) v = reinterpret_cast<const float4*>(tail)[e];
        reinterpret_cast<float4*>(dst)[e] = v;
      }
      return;
    }
    src = static_cast<const float*>(op.in[0]) + size_t(r.tok_off + loc - 2) * kEncDim;
  } else {
    src = static_cast<const float*>(op.in[0]) + size_t(m) * kEncDim;
  }
  const float* g = static_cast<const float*>(op.w[0]);
  const float* b = static_cast<const float*>(op.w[1]);
  float4 v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = reinterpret_cast<const float4*>(src)[lane + 32 * j];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) s += v[j].x + v[j].y + v[j].z + v[j].w;
  const float mean = warp_sum(s) / float(kEncDim);
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float a = v[j].x - mean, bb = v[j].y - mean, cc = v[j].z - mean, d = v[j].w - mean;
    q += a * a + bb * bb + cc * cc + d * d;
  }
  const float rstd = 1.f / sqrtf(warp_sum(q) / float(kEncDim) + op.scalar[0]);
  const float xs = op.scalar[1];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int k = 4 * (lane + 32 * j);
    float4 o;
    o.x = ((v[j].x - mean) * rstd * g[k + 0] + b[k + 0]) * xs;
    o.y = ((v[j].y - mean) * rstd * g[k + 1] + b[k + 1]) * xs;
    o.z = ((v[j].z - mean) * rstd * g[k + 2] + b[k + 2]) * xs;
    o.w = ((v[j].w - mean) * rstd * g[k + 3] + b[k + 3]) * xs;
    reinterpret_cast<float4*>(dst)[lane + 32 * j] = o;
  }
}

static __device__ void tail_save_tile(const OpDesc& op, const CallDesc& c, int tile) {
  if (tile >= c.n_rows) return;
  const RowDesc& r = c.rows[tile];
  if (r.slot < 0 || (r.flags & (kRowFinalize | kRowNoPoolWrite | kRowFullAttention))) return;
  const int end_abs = r.tok_past + r.tok_suffix - r.lookahead;
  const int new_past = max(r.tok_past, (end_abs / kTokenChunk) * kTokenChunk);
  const int shift = new_past - r.tok_past;
  const float* xext = static_cast<const float*>(op.in[0]);
  const float* x1 = static_cast<const float*>(op.in[1]);
  const int ext = r.tok_off + 2 * tile;
  constexpr int kPer = 4 * kEncDim / kThreads;  // 8
  float vals[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int e = threadIdx.x + j * kThreads;
    const int which = e / (2 * kEncDim), rem = e % (2 * kEncDim);
    const int idx = shift + rem / kEncDim, cc = rem % kEncDim;
    float v;
    if (which == 0) {
      v = xext[size_t(ext + idx) * kEncDim + cc];
    } else if (idx < 2) {
      v = r.tok_past == 0 ? 0.f : c.state.enc_up_tail[(size_t(r.slot) * 2 + idx) * kEncDim + cc];
    } else {
      v = x1[size_t(r.tok_off + idx - 2) * kEncDim + cc];
    }
    vals[j] = v;
  }
  __syncthreads();  // every old up-tail value is read before any is overwritten
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int e = threadIdx.x + j * kThreads;
    const int which = e / (2 * kEncDim), rem = e % (2 * kEncDim);
    float* dst = which == 0 ? c.state.enc_pre_tail : c.state.enc_up_tail;
    dst[size_t(r.slot) * 2 * kEncDim + rem] = vals[j];
  }
}

static __device__ void spk_affine_tile(const OpDesc& op, int tile, float* smem) {
  if (tile >= op.m) return;
  const float* x = static_cast<const float*>(op.in[0]) + size_t(tile) * kSpkDim;
  float* xn = smem;  // [192]
  __shared__ float s_den;
  if (threadIdx.x < 32) {
    float s = 0.f;
    for (int cidx = threadIdx.x; cidx < kSpkDim; cidx += 32) s = fmaf(x[cidx], x[cidx], s);
    s = warp_sum(s);
    if (threadIdx.x == 0) s_den = fmaxf(sqrtf(s), 1e-12f);  // F.normalize: x / max(||x||, eps)
  }
  __syncthreads();
  for (int cidx = threadIdx.x; cidx < kSpkDim; cidx += kThreads) xn[cidx] = x[cidx] / s_den;
  __syncthreads();
  const float* W = static_cast<const float*>(op.w[0]);
  const float* b = static_cast<const float*>(op.w[1]);
  if (threadIdx.x < kMel) {
    const int n = threadIdx.x;
    float acc = 0.f;
    for (int k = 0; k < kSpkDim; ++k) acc = fmaf(xn[k], W[n * kSpkDim + k], acc);
    static_cast<float*>(op.out[0])[size_t(tile) * kMel + n] = acc + b[n];
  }
}


// One entry point for every encoder op (MK-R: a persistent block calls it per claimed ticket).
static __device__ void run_tile(const OpDesc& op, const CallDesc& c, int32_t tile, TileCtx& ctx) {
  float* smem = static_cast<float*>(ctx.smem);
  switch (static_cast<OpKind>(op.kind)) {
    case OpKind::kEncRelPosAttn: attn_tile(op, c, tile, smem); return;
    case OpKind::kEncEmbedNorm: embed_norm_tile(op, c, tile); return;
    case OpKind::kEncTailSave: tail_save_tile(op, c, tile); return;
    case OpKind::kEncSpkAffine: spk_affine_tile(op, tile, smem); return;
    default: break;
  }
#define KT2N_ENC_GEMM_CASE(BM, BN, TM, TN, BK, KG) \
  if (op.tile_m == BM && op.tile_n == BN) { gemm_tile<BM, BN, TM, TN, BK, KG>(op, c, tile, smem); return; }
  KT2N_ENC_GEMM_CONFIGS(KT2N_ENC_GEMM_CASE)
#undef KT2N_ENC_GEMM_CASE
#define KT2N_ENC_GEMM_T_CASE(BM, BN, BK) \
  if (op.tile_m == BM && op.tile_n == BN) { gemm_tile_t<BM, BN, BK>(op, c, tile, smem); return; }
  KT2N_ENC_GEMM_T_CONFIGS(KT2N_ENC_GEMM_T_CASE)
#undef KT2N_ENC_GEMM_T_CASE
}

}  // namespace kt2n::render::enc::tiles
