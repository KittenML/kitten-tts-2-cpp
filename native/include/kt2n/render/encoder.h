// kt2n native conformer encoder (lane L4, 10 Oct 2026).
//
// Production reference (kittenml-api ebcdd3f):
//   kitten_tts_09/chunk_cache_20260927.py ChunkCache._core (encoder part) and _conformer:
//   the live steady/first-commit path (G_CHUNK_CACHE=1, G_CHUNK_FIRST=1);
//   kitten_tts_09/incremental_s3gen.py _flow: the eager path, whole renders
//   (DEMO_FULL_ATTENTION=1 -> full attention) and the look-ahead crop;
//   S3Gen transformer/upsample_encoder.py (UpsampleConformerEncoder), attention.py
//   (RelPositionMultiHeadedAttention), embedding.py (EspnetRelPositionalEncoding).
//
// What one encoder pass computes for every row of a call (ragged rows, RowDesc):
//   stage 1 (token rate, the row's suffix tokens [tok_past, tok_past + tok_suffix)):
//     e   = input_embedding[token] . embed.out.0^T + b          (kEncEmbed)
//     x   = LN_1e-5(e) * sqrt(512)                               (kEncEmbedNorm, tail-extended layout:
//                                                                 2 rows of the slot's pre_tail first)
//     h   = leaky_relu_0.01(conv1_k4(x, 3 zeros past the row's end))  (kEncPreLookahead kernel 4;
//                                                                 the 2 tail positions are 0 at tok_past 0)
//     x1  = conv2_k3(h) + x                                      (kEncPreLookahead kernel 3)
//     6 x conformer layer (5 fused ops each, below), chunk-causal on 25-token blocks
//   stage 2 (frame rate, frames [fr_past, fr_past + fr_suffix), fr = 2 * tok):
//     u   = conv_k5(nearest_x2([up_tail | x1]))                  (kEncUpLayer)
//     y   = LN_1e-5(u . up_embed.out.0^T + b) * sqrt(512)        (kEncUpEmbed + kEncEmbedNorm)
//     4 x conformer layer, chunk-causal on 50-frame blocks
//     mu  = LN_1e-5(y) . encoder_proj^T + b   -> fp32 [frames][80] (kEncProj; packed by fr_off)
//   conformer layer (normalize_before, no macaron, no conv module, ff_scale 1):
//     q|k|v = LN_1e-12(x) . W_qkv^T + b, K/V written to the pool pages   (kEncLnQkv)
//     ctx   = softmax(((q + u) . K^T + rel_shift((q + v) . P^T)) / 8) . V  (kEncRelPosAttn)
//     x     = x + ctx . W_out^T + b                                        (kEncOutProj)
//     h     = SiLU(LN_1e-12(x) . W_1^T + b_1)                              (kEncLnFf1)
//     x     = x + h . W_2^T + b_2                                          (kEncFf2)
//   with P = linear_pos(sinusoid(distance)) precomputed per layer at load (kEncPosTable), the
//   same values production's chunk cache precomputes (_precompute_rel_pos).
//
// Chunk cache contract (what the encoder reads and writes in the pool / state):
//   * Keys [0, tok_past) (stage 1) and [0, fr_past) (stage 2) are read from the row's pages; the
//     suffix's own K/V are written to the pages at absolute positions before attention reads them.
//   * Visibility (production _conformer): key < past, or key < past + suffix and key / chunk <=
//     query / chunk; kRowFullAttention rows (whole renders, past 0): key < suffix.
//   * Tails: after a call, a row's stage-1 "new past" is
//       new_past = max(tok_past, floor((tok_past + tok_suffix - lookahead) / 25) * 25)
//     (chunk_cache run(): the end of the last complete block before the look-ahead) and the
//     encoder saves the pre-lookahead tail (the 2 embed outputs before new_past) and the up-layer
//     tail (the 2 stage-1 outputs before new_past) into StateView at that point. The NEXT call of
//     the row must therefore use tok_past = new_past (fr_past = 2 * new_past). Rows with
//     kRowFinalize, kRowNoPoolWrite or kRowFullAttention save no tails.
//     Note: production's first commit (run_transient) does not advance its slot (the second commit
//     re-solves from the voice prefix). Advancing is the same math on the same values (complete
//     blocks are final: their look-ahead exists and no later key is visible to them), so the
//     native renderer may keep the first commit's blocks; it is an execution-only change.
//   * At tok_past == 0 both tails read as zeros, whatever the slot holds (production zeroes them in
//     _load_row / _build_prefix; the conv1 outputs at the 2 tail positions are masked as left_ok).
//
// Precision (frozen policy): fp32 FFMA everywhere (no TF32: S3GEN_TF32=0) with one sequential
// k-ascending FMA chain per output element, so a row's results are bitwise identical in any batch
// and any tile configuration. The 3 convs run in production under cuDNN's TF32 default; OpDesc::math
// = kTf32 rounds their operands to TF32 (round-to-nearest, cvt.rna) before the fp32 FMA (an
// emulation of the rounding class, not of cuDNN's accumulation order); kFp32Simt keeps fp32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kt2n/render/ops.h"

namespace kt2n::render::enc {

inline constexpr int32_t kLayerQkvN = 3 * kEncDim;   // 1536
inline constexpr float kXScale = 22.627416997969522f;  // math.sqrt(512) as float32
inline constexpr float kEmbedLnEps = 1e-5f;
inline constexpr float kLayerLnEps = 1e-12f;
inline constexpr float kAfterNormEps = 1e-5f;
inline constexpr float kLeakySlope = 0.01f;   // F.leaky_relu default
inline constexpr int32_t kAttnQTile = 16;     // queries per attention tile
inline constexpr int32_t kAttnKTile = 64;     // keys per inner tile
inline constexpr int32_t kMaxSplitTiles = 65536;  // split-K output tiles per op (counter slots)
inline constexpr int32_t kAttnSplitsTok = 4;      // attention key splits (interleaved 64-key tiles), stage 1
inline constexpr int32_t kAttnSplitsFr = 8;       // stage 2
// kEncRelPosAttn OpDesc::stride: 0 = one block per key split (last block merges), kAttnSplitsInBlock =
// one block runs every split and merges them (large grids); both give the same bits.
inline constexpr int32_t kAttnSplitsInBlock = 1;

// One conformer layer's device weights (fp32, derived layouts built at load).
struct LayerW {
  const float* ln_mha_g;
  const float* ln_mha_b;
  const float* w_qkv;  // [1536][512]: linear_q | linear_k | linear_v rows
  const float* b_qkv;  // [1536]
  const float* pos_u;  // [8][64]
  const float* pos_v;  // [8][64]
  const float* w_out;  // [512][512]
  const float* b_out;
  const float* ln_ff_g;
  const float* ln_ff_b;
  const float* w1;     // [2048][512]
  const float* b1;
  const float* w2;     // [512][2048]
  const float* b2;
  const float* ptab;   // [2 * maxd - 1][512]: linear_pos(pe(d)), row d + maxd - 1 (d = query - key)
};

struct Weights {
  const float* emb_table;  // [6561][512]
  const float* embed_w;    // [512][512]
  const float* embed_b;
  const float* embed_ln_g;
  const float* embed_ln_b;
  const float* pre1_w;     // [512][4 * 512], tap-major: [n][i][c] = conv1.weight[n][c][i]
  const float* pre1_b;
  const float* pre2_w;     // [512][3 * 512]
  const float* pre2_b;
  LayerW enc1[kEnc1Layers];
  const float* up_w;       // [512][5 * 512]
  const float* up_b;
  const float* up_embed_w;
  const float* up_embed_b;
  const float* up_ln_g;
  const float* up_ln_b;
  LayerW enc2[kEnc2Layers];
  const float* after_g;
  const float* after_b;
  const float* proj_w;     // [80][512]
  const float* proj_b;
  const float* spk_w;      // [80][192]
  const float* spk_b;
  int32_t maxd_tok;        // P table half size, stage 1 (distances |d| < maxd_tok)
  int32_t maxd_fr;         // stage 2
};

// Activation scratch of one encoder pass, as float offsets into CallDesc::workspace.
// T = rows_bucket * suffix_bucket tokens, E = T + 2 * rows_bucket (tail-extended), F = 2 * T.
struct WorkspaceLayout {
  size_t e, xext, hext, x;          // stage 1: [T|E][512]
  size_t q, ctx, ff, u, e2, y, mu;  // [F][512] / [F][2048] / mu [F][80]
  size_t splitk;                    // split-K partials of the K > 512 GEMMs
  size_t counters;                  // uint32 arrival counters, always at offset 0: ZERO once at allocation
                                    // (self-resetting; one workspace may serve every bucket)
  size_t attnp;                     // attention key-split partial softmax states
  size_t floats;
  size_t bytes() const { return floats * sizeof(float); }
};
WorkspaceLayout workspace_layout(int32_t rows_bucket, int32_t suffix_bucket);

// The tile configuration a GEMM op uses at a given M bucket, N and K (any choice gives the same bits).
struct GemmCfg { int32_t bm, bn; };
GemmCfg gemm_cfg_for(int32_t m_bucket, int32_t n, int32_t k = 512);

// Host: the op list of one encoder pass at (rows_bucket, suffix_bucket tokens). Pointers in
// OpDesc refer to `w` and to `workspace` (device). conv_math: kFp32Simt or kTf32.
std::vector<OpDesc> build_ops(const Weights& w, void* workspace, int32_t rows_bucket, int32_t suffix_bucket,
                              MathMode conv_math);

// Launch every op as its own kernel on `stream` (Phase B: one launch per fused op), optionally
// with programmatic dependent launch. `call` is the device CallDesc. Returns cudaError_t as int.
int launch_ops(const std::vector<OpDesc>& ops, const CallDesc* call, CUstream_st* stream, bool pdl);

// Load-time helpers (device pointers):
//   sinusoid table rows for distances d in (-maxd, maxd): pe[d + maxd - 1][512] (fp32, host-built
//   exactly as EspnetRelPositionalEncoding.extend_pe: fp32 products, even = sin, odd = cos);
std::vector<float> host_pe_table(int32_t maxd);
//   ptab = pe . linear_pos^T for one layer (GEMM on `stream`).
int build_pos_table(const float* d_pe, int32_t maxd, const float* d_linear_pos, float* d_ptab, CUstream_st* stream);
//   spk_cond[v] = spk_embed_affine_layer(l2normalize(embedding[v])) for n voices.
int spk_affine(const Weights& w, const float* d_embedding, int32_t n, float* d_spk_cond, CUstream_st* stream);

}  // namespace kt2n::render::enc
