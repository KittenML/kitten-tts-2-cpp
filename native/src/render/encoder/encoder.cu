// kt2n native conformer encoder: fused fp32 SIMT tile functions + Phase B launcher (lane L4).
// Contract and production references: include/kt2n/render/encoder.h.
#include "kt2n/render/encoder.h"
#include "kt2n/cuda_caps.h"
#include "enc_tiles.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace kt2n::render::enc {
namespace {

using namespace kt2n::render::enc::tiles;

// Tiles up to 64x64 are held to <= 128 registers so two blocks fit per SM (16 warps to hide latency);
// the 128x64 tile needs ~250 and keeps one block (-DKT2N_ENC_LB2_MAX_TILE=8192 forces it to 128: experiment).
#ifndef KT2N_ENC_LB2_MAX_TILE
#define KT2N_ENC_LB2_MAX_TILE 4096
#endif
// One kernel per (tile, A kind): the plain-A and gather-A instantiations get their own register allocation
// (as one kernel with a runtime branch, the gather path paid for the plain path's hoisted pointers: embed,
// pre-lookahead and up-layer convs 12-17% slower, p30n vs p30o).
template <int BM, int BN, int TM, int TN, int BK, int KG, bool PLAIN, bool PIECES = false>
__global__ void __launch_bounds__(kThreads, (BM * BN <= KT2N_ENC_LB2_MAX_TILE ? 2 : 1)) gemm_kernel(const OpDesc op, const CallDesc* call) {
  extern __shared__ float4 smem4[];
  gemm_tile_impl<BM, BN, TM, TN, BK, KG, PLAIN, PIECES>(op, *call, blockIdx.x, reinterpret_cast<float*>(smem4),
                                                        (op.flags & kOpPdl) != 0);
}


template <int BM, int BN, int BK>
__global__ void __launch_bounds__(kThreads, (BM * BN <= 128 * 64 ? 2 : 1)) gemm_t_kernel(const OpDesc op, const CallDesc* call) {
  extern __shared__ float4 smem4[];
  gemm_tile_t<BM, BN, BK>(op, *call, blockIdx.x, reinterpret_cast<float*>(smem4), (op.flags & kOpPdl) != 0);
}


template <bool INBLOCK>
__global__ void __launch_bounds__(kThreads, 2) attn_kernel(const OpDesc op, const CallDesc* call) {
  extern __shared__ float4 smem4[];
  attn_tile<INBLOCK>(op, *call, blockIdx.x, reinterpret_cast<float*>(smem4), (op.flags & kOpPdl) != 0);
}


__global__ void __launch_bounds__(kThreads) rows_kernel(const OpDesc op, const CallDesc* call) {
  extern __shared__ float4 smem4[];
  if (op.flags & kOpPdl) {
    griddep_wait();
    griddep_launch_dependents();
  }
  switch (static_cast<OpKind>(op.kind)) {
    case OpKind::kEncEmbedNorm: embed_norm_tile(op, *call, blockIdx.x); break;
    case OpKind::kEncTailSave: tail_save_tile(op, *call, blockIdx.x); break;
    case OpKind::kEncSpkAffine: spk_affine_tile(op, blockIdx.x, reinterpret_cast<float*>(smem4)); break;
    default: break;
  }
}


// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------
using KernelFn = void (*)(const OpDesc, const CallDesc*);

struct GemmEntry { int bm, bn; KernelFn fn, fn_gather, fn_pieces; uint32_t smem; };
template <int BM, int BN, int TM, int TN, int BK, int KG>
constexpr KernelFn pieces_kernel() {
  if constexpr (kGemmPieceTile<BM, BN, KG>) return gemm_kernel<BM, BN, TM, TN, BK, KG, true, true>;
  else return nullptr;
}
#define KT2N_ENC_GEMM_ENTRY(BM, BN, TM, TN, BK, KG)                                                   \
  {BM, BN, gemm_kernel<BM, BN, TM, TN, BK, KG, true>, gemm_kernel<BM, BN, TM, TN, BK, KG, false>, \
   pieces_kernel<BM, BN, TM, TN, BK, KG>(), gemm_smem_bytes<BM, BN, BK, KG>()},
#define KT2N_ENC_GEMM_T_ENTRY(BM, BN, BK) \
  {BM, BN, gemm_t_kernel<BM, BN, BK>, gemm_t_kernel<BM, BN, BK>, nullptr, gemm_t_smem_bytes<BM, BN, BK>()},
const GemmEntry kGemmTable[] = {KT2N_ENC_GEMM_CONFIGS(KT2N_ENC_GEMM_ENTRY) KT2N_ENC_GEMM_T_CONFIGS(KT2N_ENC_GEMM_T_ENTRY)};
#undef KT2N_ENC_GEMM_ENTRY
#undef KT2N_ENC_GEMM_T_ENTRY

const GemmEntry& gemm_entry(int bm, int bn) {
  for (const auto& e : kGemmTable)
    if (e.bm == bm && e.bn == bn) return e;
  throw std::runtime_error("encoder: no GEMM instantiation for " + std::to_string(bm) + "x" + std::to_string(bn));
}

bool is_gemm(OpKind k) {
  switch (k) {
    case OpKind::kEncEmbed:
    case OpKind::kEncPreLookahead:
    case OpKind::kEncUpLayer:
    case OpKind::kEncLnQkv:
    case OpKind::kEncOutProj:
    case OpKind::kEncLnFf1:
    case OpKind::kEncFf2:
    case OpKind::kEncUpEmbed:
    case OpKind::kEncProj:
    case OpKind::kEncPosTable: return true;
    default: return false;
  }
}

OpDesc blank_op(OpKind kind) {
  OpDesc op;
  std::memset(&op, 0, sizeof(op));
  op.kind = static_cast<uint16_t>(kind);
  op.math = static_cast<uint8_t>(MathMode::kFp32Simt);
  op.layer = -1;
  op.step = -1;
  return op;
}

OpDesc gemm_op(OpKind kind, uint32_t flags, int32_t m_bucket, int32_t n, int32_t k, const void* w, const void* b,
               const void* g, const void* beta, const void* in0, int64_t ld_in0, const void* in1, void* out0,
               int64_t ld_out0) {
  OpDesc op = blank_op(kind);
  op.flags = flags;
  op.math = static_cast<uint8_t>(MathMode::kFp32Simt);  // encoder Linear: fp32 FFMA, no TF32
  op.m = m_bucket;
  op.n = n;
  op.k = k;
  const GemmCfg cfg = gemm_cfg_for(m_bucket, n, k);
  op.tile_m = cfg.bm;
  op.tile_n = cfg.bn;
  op.n_tiles = ((m_bucket + cfg.bm - 1) / cfg.bm) * ((n + cfg.bn - 1) / cfg.bn);
  op.smem_bytes = gemm_entry(cfg.bm, cfg.bn).smem;
  op.w[0] = w;
  op.w[1] = b;
  op.w[2] = g;
  op.w[3] = beta;
  op.in[0] = in0;
  op.ld_in[0] = ld_in0;
  op.in[1] = in1;
  op.out[0] = out0;
  op.ld_out[0] = ld_out0;
  if (k % kSplitChunk != 0) throw std::runtime_error("encoder: GEMM K must be a multiple of 512");
  const int splits = k / kSplitChunk;  // canonical outer chunks (one per block)
  op.scalar[3] = float(splits);
  op.n_tiles *= splits;
  return op;
}

}  // namespace

WorkspaceLayout workspace_layout(int32_t rows_bucket, int32_t suffix_bucket) {
  const size_t T = size_t(rows_bucket) * size_t(suffix_bucket);
  const size_t E = T + 2 * size_t(rows_bucket);
  const size_t F = 2 * T;
  WorkspaceLayout l{};
  size_t o = 0;
  auto take = [&](size_t n) { const size_t at = o; o += (n + 63) / 64 * 64; return at; };
  // The split-K / attention arrival counters come FIRST, at a bucket-independent offset: a workspace
  // shared by graphs of several buckets must keep one zeroed counter array (another bucket's layout
  // would otherwise put activations where this bucket's counters live).
  l.counters = take(kMaxSplitTiles);
  l.e = take(T * kEncDim);
  l.xext = take(E * kEncDim);
  l.hext = take(E * kEncDim);
  l.x = take(T * kEncDim);
  l.q = take(F * kEncDim);
  l.ctx = take(F * kEncDim);
  l.ff = take(F * kEncFfn);
  l.u = take(F * kEncDim);
  l.e2 = take(F * kEncDim);
  l.y = take(F * kEncDim);
  l.mu = take(F * kMel);
  // Split-K partials (the largest op: up conv, 5 splits x F x 512) + one counter per output tile.
  l.splitk = take(size_t(5) * F * kEncDim);
  l.attnp = take(size_t(rows_bucket) * kEncHeads * ((2 * size_t(suffix_bucket) + kAttnQTile - 1) / kAttnQTile) *
                 kAttnSplitsFr * kAttnQTile * kAttnPartStride);
  l.floats = o;
  return l;
}

static bool attn_tail_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("KT2N_ENC_ATTN_TAIL");
    return e == nullptr || std::atoi(e) != 0;
  }();
  return v;
}
// Block slots of the wave-tail rules: 2 blocks per SM x SMs. Two is fixed by the kernels themselves: the
// attention kernel and the 32x64 / 64x64 GEMM tiles are built with __launch_bounds__(256, 2) and use 128
// registers, so 65,536 registers per SM hold exactly two blocks (and their 26-47 KB of shared memory fits).
// No occupancy API on purpose: build_ops runs inside the renderer's stream capture (L7 flow_enqueue), where a
// capture-unsafe call would invalidate the graph; cudaGetDevice / cudaDeviceGetAttribute are already used
// there since sprint 2. KT2N_ENC_WAVE_SLOTS=<n> overrides (bitwise checks of the tail paths on small calls).
static long wave_slots() {
  static const long v = [] {
    const char* e = std::getenv("KT2N_ENC_WAVE_SLOTS");
    if (e != nullptr && std::atol(e) > 0) return long(std::atol(e));
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0)
      sms = 188;
    return 2L * sms;
  }();
  return v;
}

// Attention (row, head, query tile) groups from which one block runs all key splits.
// Default: two full waves (752 on the 188-SM part, the sprint-2 value). Sprint 3 adds (build_ops): from ONE
// wave when the overrun groups can run per split (the wave-tail rule below), which made stage-2 attention
// -9% at first B16 and -5% at steady B8 (512 groups, `p42b33`); one-wave in-block WITHOUT that tail lost 8% at
// late B20 stage 1 (640 groups, G = 4: 234 vs 216 us, `p42b33`), so that case stays per split. The env
// variable forces a plain threshold (experiments).
static int32_t attn_inblock_env() {
  static int32_t v = [] {
    const char* e = std::getenv("KT2N_ENC_ATTN_INBLOCK_MIN");
    return e != nullptr ? int32_t(std::atoi(e)) : int32_t(-1);
  }();
  return v;
}
// Groups past the last full wave of an in-block grid run per split when their split blocks fit in at most
// G / 2 waves (half an in-block block time). Returns the groups run in-block (0 = no tail split).
static long attn_tail_full(long groups, long G) {
  const long slots = wave_slots();
  const long full = slots > 0 ? (groups / slots) * slots : 0, rem = groups - full;
  if (!attn_tail_enabled() || full == 0 || rem == 0 || (rem * G + slots - 1) / slots > G / 2) return 0;
  return full;
}
static bool attn_use_inblock(long groups, long G) {
  const int32_t env = attn_inblock_env();
  if (env >= 0) return groups >= env;
  return groups >= 2 * wave_slots() || (groups >= wave_slots() && attn_tail_full(groups, G) > 0);
}

GemmCfg gemm_cfg_for(int32_t m_bucket, int32_t n, int32_t k) {
  if (n % 32 != 0) return m_bucket <= 64 ? GemmCfg{16, 16} : GemmCfg{64, 16};
  if (m_bucket <= 64) return {16, 32};
  if (m_bucket <= 256) return {32, 64};
  // M > 256: 64x64 tiles (two blocks per SM). They beat the 128x64 tile (one block per SM, ~250
  // registers) on every large op: nsys per-op minima at B32, GEMMs 4.07 -> 3.58 ms (first commit) and
  // 7.92 -> 6.31 ms (steady) (`p20A` vs `p20C`). When 64x64 gives fewer tiles than SMs (less than half
  // a wave), 32x64 doubles the parallelism (B32 first, out-proj 28.3 -> 25.1 us, `p20E`).
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0)
      sms = 188;
  }
  const long tiles64 = long((m_bucket + 63) / 64) * long((n + 63) / 64) * long(k / kSplitChunk);
  GemmCfg best = tiles64 < sms ? GemmCfg{32, 64} : GemmCfg{64, 64};
  // Experiment only: force one large-M tile for every GEMM with M > 256 (e.g. "128x64"); every
  // tile gives the same bits, so this changes speed only.
  static const char* force = std::getenv("KT2N_ENC_BIG_TILE");
  if (force != nullptr) {
    int bm = 0, bn = 0;
    if (std::sscanf(force, "%dx%d", &bm, &bn) == 2) best = {bm, bn};
  }
  return best;
}

// Wave-tail pieces (sprint 3; enc_tiles.cuh gemm_tile_impl): a GEMM whose tile-blocks overrun its last full
// wave (slots = blocks per SM x SMs) by rem <= 1/4 of a wave runs those rem tile-blocks as 4 inner-chunk
// pieces each, so the tail costs one quarter block time instead of one whole block time.
// Same canonical chunk fold, so the same bits. Only the 32x64 / 64x64 tiles (KG = 1), only when the pieces'
// partials fit the split-K region after the op's own outer partials. KT2N_ENC_WAVE_PIECES=0 turns it off;
// KT2N_ENC_WAVE_SLOTS=<n> pretends the GPU has n slots (bitwise checks of the piece path on small calls).
static bool wave_tail_pieces(OpDesc& op, size_t part_floats) {
  static const bool enabled = [] {
    const char* e = std::getenv("KT2N_ENC_WAVE_PIECES");
    return e == nullptr || std::atoi(e) != 0;
  }();
  if (!enabled || !is_gemm(static_cast<OpKind>(op.kind)) || op.kind == uint16_t(OpKind::kEncPosTable)) return false;
  if (!((op.tile_m == 64 && op.tile_n == 64) || (op.tile_m == 32 && op.tile_n == 64)) || !gemm_a_is_plain(op))
    return false;
  if (op.k / (op.scalar[3] > 1.f ? int(op.scalar[3]) : 1) != kSplitChunk) return false;
  if (gemm_entry(op.tile_m, op.tile_n).fn_pieces == nullptr) return false;
  const long slots = wave_slots();
  const long blocks = op.n_tiles;  // tile-blocks: (outer split, output tile)
  const long full = (blocks / slots) * slots, rem = blocks - full;
  // Only when all pieces fit in ONE quarter wave (4 rem <= slots): at first B16 the stage-2 QKV (384 tiles,
  // rem 8 -> 32 pieces) went 73.8 -> 59.0 us, but FF1 (512 tiles, rem 136 -> 544 pieces, two quarter waves)
  // went 75.7 -> 82.3 us: a piece repeats the block's fixed work (row map, LayerNorm statistics, partial
  // write and fold), so two quarter waves cost more than the half block time they save (`p35b27` vs `p35b26`).
  if (full == 0 || rem == 0 || 4 * rem > slots) return false;
  const int splits = op.scalar[3] > 1.f ? int(op.scalar[3]) : 1;
  const size_t need = (splits > 1 ? size_t(splits) * size_t(op.m) * size_t(op.n) : 0) +
                      size_t(rem) * 4 * size_t(op.tile_m) * size_t(op.tile_n);
  if (need > part_floats || rem > kMaxSplitTiles / 2) return false;
  op.stride = int32_t(full);
  op.n_tiles = int32_t(full + 4 * rem);
  return true;
}

std::vector<OpDesc> build_ops(const Weights& w, void* workspace, int32_t rows_bucket, int32_t suffix_bucket,
                              MathMode conv_math) {
  const WorkspaceLayout L = workspace_layout(rows_bucket, suffix_bucket);
  float* ws = static_cast<float*>(workspace);
  const int32_t T = rows_bucket * suffix_bucket;
  const int32_t E = T + 2 * rows_bucket;
  const int32_t F = 2 * T;
  const uint32_t pdl = kOpPdl;
  struct Ops : std::vector<OpDesc> {
    float* part;
    float* cnt;
    size_t part_floats;
    void push_back(OpDesc op) {
      if (op.scalar[3] > 1.f) {
        op.out[1] = part;
        op.in[3] = cnt;
      }
      if (wave_tail_pieces(op, part_floats)) {
        op.out[1] = part;
        op.in[3] = cnt;
      }
      std::vector<OpDesc>::push_back(op);
    }
  };
  Ops ops;
  ops.part = ws + L.splitk;
  ops.cnt = ws + L.counters;
  ops.part_floats = L.attnp - L.splitk;
  ops.reserve(64);

  // Stage 1 input.
  ops.push_back(gemm_op(OpKind::kEncEmbed, pdl, T, kEncDim, kEncDim, w.embed_w, w.embed_b, nullptr, nullptr,
                        w.emb_table, kEncDim, nullptr, ws + L.e, kEncDim));
  {
    OpDesc op = blank_op(OpKind::kEncEmbedNorm);
    op.flags = pdl;
    op.m = E;
    op.n_tiles = (E + 7) / 8;
    op.w[0] = w.embed_ln_g;
    op.w[1] = w.embed_ln_b;
    op.in[0] = ws + L.e;
    op.out[0] = ws + L.xext;
    op.scalar[0] = kEmbedLnEps;
    op.scalar[1] = kXScale;
    ops.push_back(op);
  }
  {
    OpDesc op = gemm_op(OpKind::kEncPreLookahead, pdl, E, kEncDim, 4 * kEncDim, w.pre1_w, w.pre1_b, nullptr,
                        nullptr, ws + L.xext, kEncDim, nullptr, ws + L.hext, kEncDim);
    op.kernel = 4;
    op.pad_right = kLookahead;
    op.math = static_cast<uint8_t>(conv_math);
    ops.push_back(op);
  }
  {
    OpDesc op = gemm_op(OpKind::kEncPreLookahead, pdl, T, kEncDim, 3 * kEncDim, w.pre2_w, w.pre2_b, nullptr,
                        nullptr, ws + L.hext, kEncDim, ws + L.xext, ws + L.x, kEncDim);
    op.kernel = 3;
    op.pad_left = 2;
    op.math = static_cast<uint8_t>(conv_math);
    ops.push_back(op);
  }
  auto layers = [&](const LayerW* lw, int n_layers, bool frames, int32_t M, float* xbuf, int32_t maxd,
                    int32_t max_suffix) {
    const uint32_t fl = pdl | (frames ? uint32_t(kOpFrames) : 0u);
    for (int l = 0; l < n_layers; ++l) {
      const LayerW& W = lw[l];
      OpDesc qkv = gemm_op(OpKind::kEncLnQkv, fl | kOpWritesPool, M, kLayerQkvN, kEncDim, W.w_qkv, W.b_qkv,
                           W.ln_mha_g, W.ln_mha_b, xbuf, kEncDim, nullptr, ws + L.q, kEncDim);
      qkv.layer = l;
      qkv.scalar[0] = kLayerLnEps;
      ops.push_back(qkv);
      OpDesc at = blank_op(OpKind::kEncRelPosAttn);
      at.flags = fl | kOpReadsPool;
      at.layer = l;
      at.k = maxd;
      at.tile_m = kAttnQTile;
      at.tile_n = (max_suffix + kAttnQTile - 1) / kAttnQTile;
      at.dilation = frames ? kAttnSplitsFr : kAttnSplitsTok;  // key splits G
      // Large grids run the G splits inside one block (same bits; see attn_tile).
      const int32_t groups = rows_bucket * kEncHeads * at.tile_n;
      at.stride = attn_use_inblock(groups, at.dilation) ? kAttnSplitsInBlock : 0;
      at.n_tiles = at.stride == kAttnSplitsInBlock ? groups : groups * at.dilation;
      if (at.stride == kAttnSplitsInBlock) {
        // Wave tail (sprint 3): when the in-block groups overrun their last full wave, run the overrun
        // groups one block per split (at.kernel = groups run in-block), if their split blocks fit in at
        // most G / 2 waves, i.e. cost at most half an in-block block time (same bits: same partials, same
        // merge order). KT2N_ENC_ATTN_TAIL=0 turns it off.
        const long full = attn_tail_full(groups, at.dilation);
        if (full > 0) {
          at.kernel = int32_t(full);
          at.n_tiles = int32_t(full + (groups - full) * at.dilation);
        }
      }
      at.out[1] = ws + L.attnp;
      at.in[3] = ws + L.counters;
      at.smem_bytes = kAttnSmemBytes;
      at.w[0] = W.ptab;
      at.w[1] = W.pos_u;
      at.w[2] = W.pos_v;
      at.in[0] = ws + L.q;
      at.out[0] = ws + L.ctx;
      ops.push_back(at);
      ops.push_back(gemm_op(OpKind::kEncOutProj, fl, M, kEncDim, kEncDim, W.w_out, W.b_out, nullptr, nullptr,
                            ws + L.ctx, kEncDim, xbuf, xbuf, kEncDim));
      OpDesc f1 = gemm_op(OpKind::kEncLnFf1, fl, M, kEncFfn, kEncDim, W.w1, W.b1, W.ln_ff_g, W.ln_ff_b, xbuf,
                          kEncDim, nullptr, ws + L.ff, kEncFfn);
      f1.act = static_cast<uint8_t>(Activation::kSilu);
      f1.scalar[0] = kLayerLnEps;
      ops.push_back(f1);
      ops.push_back(gemm_op(OpKind::kEncFf2, fl, M, kEncDim, kEncFfn, W.w2, W.b2, nullptr, nullptr, ws + L.ff,
                            kEncFfn, xbuf, xbuf, kEncDim));
    }
  };
  layers(w.enc1, kEnc1Layers, false, T, ws + L.x, w.maxd_tok, suffix_bucket);

  // Stage 2.
  {
    OpDesc op = gemm_op(OpKind::kEncUpLayer, pdl, F, kEncDim, 5 * kEncDim, w.up_w, w.up_b, nullptr, nullptr,
                        ws + L.x, kEncDim, nullptr, ws + L.u, kEncDim);
    op.kernel = 5;
    op.pad_left = 4;
    op.math = static_cast<uint8_t>(conv_math);
    ops.push_back(op);
  }
  {
    OpDesc op = blank_op(OpKind::kEncTailSave);
    op.flags = pdl;
    op.n_tiles = rows_bucket;
    op.in[0] = ws + L.xext;
    op.in[1] = ws + L.x;
    ops.push_back(op);
  }
  ops.push_back(gemm_op(OpKind::kEncUpEmbed, pdl, F, kEncDim, kEncDim, w.up_embed_w, w.up_embed_b, nullptr,
                        nullptr, ws + L.u, kEncDim, nullptr, ws + L.e2, kEncDim));
  {
    OpDesc op = blank_op(OpKind::kEncEmbedNorm);
    op.flags = pdl | kOpFrames;
    op.m = F;
    op.n_tiles = (F + 7) / 8;
    op.w[0] = w.up_ln_g;
    op.w[1] = w.up_ln_b;
    op.in[0] = ws + L.e2;
    op.out[0] = ws + L.y;
    op.scalar[0] = kEmbedLnEps;
    op.scalar[1] = kXScale;
    ops.push_back(op);
  }
  layers(w.enc2, kEnc2Layers, true, F, ws + L.y, w.maxd_fr, 2 * suffix_bucket);
  {
    OpDesc op = gemm_op(OpKind::kEncProj, pdl, F, kMel, kEncDim, w.proj_w, w.proj_b, w.after_g, w.after_b,
                        ws + L.y, kEncDim, nullptr, ws + L.mu, kMel);
    op.scalar[0] = kAfterNormEps;
    ops.push_back(op);
  }
  (void)E;
  for (const OpDesc& op : ops) {
    const int tiles_mn = op.n_tiles / (op.scalar[3] > 1.f ? int(op.scalar[3]) : 1);
    if (op.scalar[3] > 1.f && tiles_mn > kMaxSplitTiles) throw std::runtime_error("encoder: too many split-K tiles");
  }
  return std::vector<OpDesc>(ops.begin(), ops.end());
}

int launch_ops(const std::vector<OpDesc>& ops, const CallDesc* call, CUstream_st* stream, bool pdl) {
  pdl = kt2n::cuda::pdl_enabled(pdl);
  int device = -1;
  const cudaError_t device_err = cudaGetDevice(&device);
  if (device_err != cudaSuccess) return int(device_err);
  static thread_local std::array<bool, 64> attrs_done{};
  if (device >= int(attrs_done.size()) || !attrs_done[device]) {
    for (KernelFn f : {attn_kernel<true>, attn_kernel<false>}) {
      cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kAttnSmemBytes));
      // Two 47 KB attention blocks per SM need the largest shared-memory carveout (100 KB on sm_120).
      cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, int(cudaSharedmemCarveoutMaxShared));
    }
    for (const auto& e : kGemmTable) {
      cudaFuncSetAttribute(e.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(e.smem));
      cudaFuncSetAttribute(e.fn_gather, cudaFuncAttributeMaxDynamicSharedMemorySize, int(e.smem));
      if (e.fn_pieces != nullptr)
        cudaFuncSetAttribute(e.fn_pieces, cudaFuncAttributeMaxDynamicSharedMemorySize, int(e.smem));
    }
    if (device < int(attrs_done.size())) attrs_done[device] = true;
  }
  for (const OpDesc& o : ops) {
    OpDesc op = o;
    if (!pdl) op.flags &= ~uint32_t(kOpPdl);
    const auto kind = static_cast<OpKind>(op.kind);
    KernelFn fn;
    if (is_gemm(kind)) {
      const GemmEntry& e = gemm_entry(op.tile_m, op.tile_n);
      fn = !gemm_a_is_plain(op) ? e.fn_gather : (op.stride > 0 && e.fn_pieces != nullptr) ? e.fn_pieces : e.fn;
    }
    else if (kind == OpKind::kEncRelPosAttn) fn = op.stride == kAttnSplitsInBlock ? attn_kernel<true> : attn_kernel<false>;
    else fn = rows_kernel;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(unsigned(op.n_tiles));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = op.smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = (op.flags & kOpPdl) ? 1 : 0;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, fn, op, call);
    if (err != cudaSuccess) return int(err);
  }
  return int(cudaGetLastError());
}

std::vector<float> host_pe_table(int32_t maxd) {
  // EspnetRelPositionalEncoding.extend_pe: div_term = exp(arange(0, 512, 2) * -(ln(10000) / 512)) in fp32;
  // pe(+p) = [sin(p * div), cos(p * div)] interleaved (even sin, odd cos); pe(-p) = [sin(-p * div), cos(...)].
  const int rows = 2 * maxd - 1;
  std::vector<float> pe(size_t(rows) * kEncDim);
  const float coef = float(-(std::log(10000.0) / double(kEncDim)));
  float div[kEncDim / 2];
  for (int i = 0; i < kEncDim / 2; ++i) div[i] = float(std::exp(double(float(2 * i) * coef)));
  for (int r = 0; r < rows; ++r) {
    const int d = r - (maxd - 1);
    const float p = float(d);
    for (int i = 0; i < kEncDim / 2; ++i) {
      const float arg = p * div[i];  // fp32 product, as torch
      pe[size_t(r) * kEncDim + 2 * i] = float(std::sin(double(arg)));
      pe[size_t(r) * kEncDim + 2 * i + 1] = float(std::cos(double(arg)));
    }
  }
  return pe;
}

// Load-time ops (kEncPosTable, kEncSpkAffine) read no call state; they get an all-zero descriptor.
const CallDesc* zero_call() {
  static CallDesc* d = nullptr;
  if (d == nullptr) {
    if (cudaMalloc(&d, sizeof(CallDesc)) != cudaSuccess) return nullptr;
    cudaMemset(d, 0, sizeof(CallDesc));
  }
  return d;
}

int build_pos_table(const float* d_pe, int32_t maxd, const float* d_linear_pos, float* d_ptab, CUstream_st* stream) {
  const int32_t rows = 2 * maxd - 1;
  OpDesc op = gemm_op(OpKind::kEncPosTable, 0, rows, kEncDim, kEncDim, d_linear_pos, nullptr, nullptr, nullptr,
                      d_pe, kEncDim, nullptr, d_ptab, kEncDim);
  std::vector<OpDesc> ops{op};
  return launch_ops(ops, zero_call(), stream, false);
}

int spk_affine(const Weights& w, const float* d_embedding, int32_t n, float* d_spk_cond, CUstream_st* stream) {
  OpDesc op = blank_op(OpKind::kEncSpkAffine);
  op.m = n;
  op.n_tiles = n;
  op.smem_bytes = kSpkDim * sizeof(float);
  op.w[0] = w.spk_w;
  op.w[1] = w.spk_b;
  op.in[0] = d_embedding;
  op.out[0] = d_spk_cond;
  std::vector<OpDesc> ops{op};
  return launch_ops(ops, zero_call(), stream, false);
}

}  // namespace kt2n::render::enc
