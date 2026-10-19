// kt2n renderer op-library contract (G0 interface freeze, 10 Oct 2026).
//
// One op library serves three execution modes of the token -> mel -> PCM
// renderer (conformer encoder, 2-step MeanFlow estimator, HiFT + tail):
//   Phase A  correctness backbone: library GEMMs/convs + handwritten pieces,
//            one kernel launch per op;
//   Phase B  fused handwritten kernels, one launch per fused op;
//   Phase C  persistent render megakernel (MK-R): blocks claim (op, tile)
//            tickets in topological order and call the SAME tile functions.
// Every op is written once as a tile function (`run_tile`, below) over a
// ragged batch whose shapes live in device memory (RowDesc table), so CUDA
// graphs exist only per (phase, rows bucket, suffix bucket) -- not per shape.
//
// Production reference (kittenml-api ebcdd3f): kitten_tts_09/incremental_s3gen.py,
// kitten_tts_09/chunk_cache_20260927.py (the live chunk cache: G_CHUNK_CACHE=1),
// kitten_tts_09/acoustic_graphs.py, kitten_tts_09/runtime.py; module definitions in
// the S3Gen package (flow.py, decoder.py, flow_matching.py, hifigan.py,
// transformer/upsample_encoder.py, f0_predictor.py). Weights: the container format in tools/kt2n_assets/format.py
// (s3gen.kt2n). The precision policy is recorded per op in OpDesc::math.
//
// Rules every op implementation follows:
//   1. Batch invariance. A row's outputs are bitwise identical whether it runs
//      alone or inside any batch / bucket: fixed per-element reduction order, no
//      split counts that depend on the batch, no atomics in accumulations.
//      Phase A library calls (cuBLASLt / cuDNN) cannot promise this across
//      buckets (the library may pick another kernel per shape): they pin one
//      algorithm per (op, bucket) with splitK = 1, and the quality gates, not
//      this rule, judge them. Phase B / MK-R tile functions follow the rule.
//   2. A tile writes only its own output region and reads only its op's inputs
//      (OpDesc::in / w / the pool through CallDesc). No grid-wide sync inside a
//      tile; block-level sync only. Every tile runs with kTileThreads threads and
//      at most OpDesc::smem_bytes (<= kTileSmemBytes) of dynamic shared memory.
//   3. No device allocation, no host sync, no printf in serving builds.
//   4. Padding rows (index >= CallDesc::n_rows, slot == -1) produce nothing and
//      read nothing past their (zero) lengths.
#pragma once

#include <cstddef>
#include <cstdint>

// Forward declarations: cudaStream_t / cudaEvent_t are pointers to these, so
// this header compiles without the CUDA headers (host lanes, CPU tests).
struct CUstream_st;
struct CUevent_st;

#if defined(__CUDACC__)
#define KT2N_HD __host__ __device__
#else
#define KT2N_HD
#endif

namespace kt2n::render {

// ---------------------------------------------------------------------------
// Frozen model geometry (checked against s3gen.kt2n "config" at load).
// ---------------------------------------------------------------------------
inline constexpr int32_t kVocab = 6561;          // speech-token vocabulary (flow.input_embedding)
inline constexpr int32_t kSilenceToken = 4299;   // S3GEN_SIL: appended 3x by every final flush
inline constexpr int32_t kTokenChunk = 25;       // encoder chunk-causal block (S3GEN_BLOCK_TOKENS=25)
inline constexpr int32_t kMelPerToken = 2;
inline constexpr int32_t kMelChunk = 50;         // estimator chunk-causal block, frames
inline constexpr int32_t kLookahead = 3;         // PRELOOKAHEAD_TOKENS (frozen: look-ahead 3)
inline constexpr int32_t kMel = 80;
inline constexpr int32_t kSpkDim = 192;          // x-vector width (flow.spk_embed_affine_layer input)

inline constexpr int32_t kEncDim = 512;
inline constexpr int32_t kEncHeads = 8;
inline constexpr int32_t kEncHeadDim = 64;
inline constexpr int32_t kEncFfn = 2048;
inline constexpr int32_t kEnc1Layers = 6;        // flow.encoder.encoders (per token)
inline constexpr int32_t kEnc2Layers = 4;        // flow.encoder.up_encoders (per frame, after the 2x up layer)

inline constexpr int32_t kEstIn = 320;           // x | mu | spks | cond, 80 each
inline constexpr int32_t kEstDim = 256;
inline constexpr int32_t kEstHeads = 8;
inline constexpr int32_t kEstHeadDim = 64;
inline constexpr int32_t kEstInner = 512;        // heads * head_dim
inline constexpr int32_t kEstFfn = 1024;
inline constexpr int32_t kEstTimeDim = 1024;
inline constexpr int32_t kEstMidBlocks = 12;
inline constexpr int32_t kEstTfPerBlock = 4;
inline constexpr int32_t kEstAttnLayers = 56;    // (1 down + 12 mid + 1 up) x 4 transformer blocks
inline constexpr int32_t kEstCausalConvs = 31;   // k=3 causal convs with a 2-frame tail per flow step
inline constexpr int32_t kFlowSteps = 2;         // frozen: 2 mean-flow steps (preview: 1)

inline constexpr int32_t kSampleRate = 24000;
inline constexpr int32_t kSamplesPerFrame = 480; // 8 * 5 * 3 upsampling * 4 iSTFT hop
inline constexpr int32_t kIstftNfft = 16;
inline constexpr int32_t kIstftHop = 4;
inline constexpr int32_t kHarmonics = 9;         // SineGen harmonic_num 8 + fundamental
inline constexpr int32_t kMelCacheFrames = 8;    // MEL_CACHE_FRAMES
inline constexpr int32_t kSourceCacheSamples = 3840;  // SOURCE_CACHE_SAMPLES = 8 * 480 (160 ms)
inline constexpr int32_t kTailFadeSamples = 480;      // Q_FIRST_TAIL_FADE_MS = 20 ms

// ---------------------------------------------------------------------------
// Phases and buckets
// ---------------------------------------------------------------------------
enum class Phase : uint16_t {
  // A row's first render. Pool holds only the voice prefix (shared pages); the
  // suffix is the prompt's open block + the first commit + look-ahead (or the
  // look-ahead the row actually has: S3GEN_NO_LOOKAHEAD=first). Urgent class.
  kFirstCommit = 0,
  // A later commit: suffix = the row's open block + new tokens + 3 look-ahead.
  kSteadyCommit = 1,
  // Segment end: suffix + 3 x kSilenceToken, finalize (no crop), publishes the
  // held tail.
  kFinalFlush = 2,
  // stream:false / DEMO "s2" row: the whole chunk at once with FULL attention
  // (DEMO_FULL_ATTENTION=1: static_chunk_size 0 for encoder and estimator),
  // tokens + 3 x 4299. Lengths in kWholeBucketTokens steps. Bulk class, cut
  // into <= 5 ms pieces. Starts at position 0 (tok_past = fr_past = 0): under
  // full attention every K/V depends on the whole row, so no voice prefix
  // applies. Noise from the row's numpy tables (prompt region = the voice's
  // prompt noise); HiFT with no mel / source cache, trim-fade head, publishes
  // everything (incremental_s3gen.py launch: demo_full -> _flow, finalize).
  kWholeRender = 3,
  // The 1-step stream preview (STREAM_PREVIEW_TOKENS=8, FLOW_STEPS=1,
  // FINALIZE=0): fires when commit 0 has not rendered by token 8.
  // Production renders it with the STOCK flow (runtime.py:15916 ->
  // _decode_cuda_preview_batch :14254 -> s3gen.flow_inference): prompt + the
  // 8 tokens solved from position 0 (chunk masks on), one step at (t=0, r=1),
  // torch-RNG noise over the prompt region too (see RNG). So the voice prefix
  // does NOT apply to the estimator (its K/V are for 2 steps at t=0/0.5 over
  // the voice's numpy prompt noise): a preview row has tok_past = fr_past = 0,
  // writes its own K/V into scratch pages owned by the call (released after
  // it), and never writes the row's slot state (kRowNoPoolWrite). Estimator
  // valid frames = 2 * (prompt + 8) - 6 (finalize=False crops the look-ahead
  // before the decoder). Output: HiFT of the 2 * 8 - 6 = 10 target frames with
  // no caches, trim-fade head and a 480-sample linear fade-out at the end
  // (runtime.py _preview_waveforms_to_numpy); commit 0 later drops the
  // preview's samples on the host (_trim_incremental_preview).
  kPreview = 4,
};
// A voice-prefix build (startup for built-ins, first use for clones;
// chunk_cache_20260927.py _build_prefix) is a kSteadyCommit row from position 0
// over prompt[:stable + 3] with lookahead 3 and an empty output window
// (out_fr_count = pcm_count = 0, no HiFT). Its complete pages become the
// voice's shared prefix pages and its conv tails the voice's prefix tails
// (VoiceView). stable = acoustic_prefix.stable_prefix_tokens(prompt tokens).
inline constexpr int32_t kPhaseCount = 5;

enum RowFlags : uint16_t {
  kRowFinalize = 1u << 0,          // no look-ahead crop (finals, whole renders)
  kRowFullAttention = 1u << 1,     // no chunk mask (whole render)
  kRowNoPoolWrite = 1u << 2,       // preview: K/V only into the call's scratch pages; no slot state,
                                   // no shared page, no tail is written
  kRowFirstCommitTail = 1u << 3,   // S3GEN_FIRST_COMMIT_TAIL=1: publish the held tail too
  kRowLookaheadPadZero = 1u << 4,  // X_LOOKAHEAD_PAD_ZERO=1: pre-lookahead conv reads zeros past the row's end
  kRowSeamCrossfade = 1u << 5,     // crossfade the head with the previous held tail (Q_SEAM_WINDOW=hamming)
  kRowTrimFade = 1u << 6,          // apply the stream-start trim_fade (20 ms zero + 20 ms cosine rise)
  kRowCancelled = 1u << 7,         // row dropped after planning: compute nothing, publish nothing
};

// Graph buckets. A call is captured / replayed per (phase, rows bucket, suffix
// bucket); per-row lengths, keys and pages are data in the RowDesc table.
inline constexpr int32_t kRowBuckets[] = {1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64};
inline constexpr int32_t kFirstSuffixBuckets[] = {16, 32, 48};          // tokens incl. look-ahead (FIRST_S_BUCKETS)
inline constexpr int32_t kSteadySuffixBuckets[] = {16, 32, 48, 64, 80}; // S_BUCKETS (steady commits and finals)
// Whole renders, and any suffix above the largest bucket (a row starting at
// position 0 behind a 250-token prompt needs up to 272 tokens), round up to
// multiples of kWholeBucketTokens.
inline constexpr int32_t kWholeBucketTokens = 64;
inline constexpr int32_t kPreviewTokens = 8;
// Largest row (decoder prompt + segment tokens + 3) the renderer must serve. NOT production's
// G_CHUNK_CAP_TOKENS=1088: that is the chunk cache's per-slot capacity, and production renders
// longer rows by its eager path. Live rows above 1088 exist: a streaming segment may decode up to
// budget x SILENCE_FREE_BUDGET_CEILING (2) LM tokens (runtime.py:14694), a tagged segment's budget
// reaches 2,341 tokens (tagged_segment_budget, 4.4 x 532), a DEMO chunk 1,000 (+ a prompt <= 250).
// The LM's K/V ceiling (CUDA_GRAPH_MAX_CTX=4096, which also holds the ~330-id LM prefix and the
// text) bounds every row, so 4096 covers them all. Paging makes a long row cost pages, not a slot.
inline constexpr int32_t kMaxRowTokens = 4096;

// Smallest bucket >= n, or -1 if n exceeds the largest.
template <size_t N>
constexpr int32_t bucket_of(const int32_t (&buckets)[N], int32_t n) {
  for (size_t i = 0; i < N; ++i)
    if (n <= buckets[i]) return buckets[i];
  return -1;
}
constexpr int32_t whole_bucket(int32_t tokens) {
  return ((tokens + kWholeBucketTokens - 1) / kWholeBucketTokens) * kWholeBucketTokens;
}

// ---------------------------------------------------------------------------
// Math modes (recorded per op)
// ---------------------------------------------------------------------------
enum class MathMode : uint8_t {
  kFp32Simt = 0,        // fp32 FFMA, no TF32 (encoder Linear: production S3GEN_TF32=0)
  kTf32 = 1,            // TF32 tensor-core inputs, fp32 accumulate (convs: production's cuDNN default)
  kBf16F32Acc = 2,      // bf16 weights/activations, fp32 accumulate (estimator)
  kFp32 = 3,            // fp32 elementwise / reductions (LayerNorm stats, softmax, HiFT tail)
};

enum class Activation : uint8_t {
  kNone = 0, kRelu, kSilu, kGeluErf, kMish, kSnake, kLeakyRelu, kTanh,
};

// ---------------------------------------------------------------------------
// Op kinds. Values are append-only within each lane's range; a lane registers
// its fused ops by adding them here (one line each) in its own PR.
// ---------------------------------------------------------------------------
enum class OpKind : uint16_t {
  // 0x000 generic (any lane)
  kNop = 0x000,
  kGemm = 0x001,         // out[m, n] = epilogue(sum_k in[m, k] * W[n, k])  (W row-major [N, K])
  kConv1d = 0x002,       // implicit-GEMM conv, causal / look-ahead padding, stride, dilation
  kConvTranspose1d = 0x003,
  kLayerNorm = 0x004,    // row LayerNorm (fp32 stats) with optional residual-in / activation-out
  kGather = 0x005,       // embedding lookup / row gather
  kPack = 0x006,         // ragged pack / unpack between layouts
  kCopy = 0x007,
  // 0x100 encoder (L4)
  kEncEmbed = 0x100,         // input_embedding * mask -> embed.out (Linear+LN) * sqrt(512) + rel-pos table
  kEncPreLookahead = 0x101,  // conv1 (k=4, look-ahead 3) + leaky_relu + conv2 (k=3 causal) + residual
  kEncRelPosAttn = 0x102,    // chunk-causal rel-pos MHA (pos_bias_u/v, linear_pos), K/V from pool pages
  kEncUpLayer = 0x103,       // nearest 2x + causal pad 4 + conv k=5
  kEncConformerFfn = 0x104,
  // L4 fused ops (src/render/encoder/, include/kt2n/render/encoder.h). kEncPreLookahead is conv1
  // (OpDesc::kernel 4, writes the tail-extended layout) or conv2 (kernel 3, + residual).
  kEncEmbedNorm = 0x105,     // LayerNorm (eps 1e-5) * sqrt(512) rows; stage 1 into the tail-extended layout
  kEncLnQkv = 0x106,         // norm_mha LN prologue + linear_q|k|v GEMM; K/V written to pool pages
  kEncOutProj = 0x107,       // linear_out + bias + residual
  kEncLnFf1 = 0x108,         // norm_ff LN prologue + w_1 + bias + SiLU
  kEncFf2 = 0x109,           // w_2 + bias + residual
  kEncUpEmbed = 0x10A,       // up_embed Linear (frame rate)
  kEncProj = 0x10B,          // after_norm LN prologue + encoder_proj -> mu [frames][80]
  kEncTailSave = 0x10C,      // pre-lookahead / up-layer conv tails at the row's new stage-1 past
  kEncSpkAffine = 0x10D,     // l2-normalize + spk_embed_affine_layer (voice load, not per render)
  kEncPosTable = 0x10E,      // load time: linear_pos over the sinusoid table (no bias)
  // 0x200 estimator (L5a resnets / U-Net, L5b transformer + pool attention)
  kEstTimeEmbed = 0x200,     // sinusoid(t), sinusoid(r) -> time_mlp -> time_embed_mixer (once per step)
  kEstInputAssemble = 0x201, // x | mu | spks | cond for the suffix frames
  kEstCausalBlock = 0x202,   // conv k=3 (tail from state) -> LN -> Mish -> mask (+ time add)
  kEstResnetTail = 0x203,    // mlp(time) add, block2, res_conv residual
  kEstQkv = 0x204,           // fused to_k | to_v | to_q GEMM, K/V written to pool pages
  kEstPoolAttn = 0x205,      // chunk-causal attention over pool pages (split keys + combine)
  kEstFeedForward = 0x206,   // LN -> GELU proj -> out, residual
  kEstFinal = 0x207,         // final_block + final_proj, mask
  kEstFlowUpdate = 0x208,    // x += (r - t) * v on the suffix; output the step's mel
  // 0x300 HiFT + tail (L6)
  kHiftF0 = 0x300,           // f0_predictor condnet + classifier
  kHiftSource = 0x301,       // f0 upsample, SineGen (Philox noise, mt19937 phases), l_linear + tanh
  kHiftStft = 0x302,
  kHiftConvPre = 0x303,
  kHiftUp = 0x304,           // leaky_relu + ups[i] + source_downs/source_resblocks fusion
  kHiftResblock = 0x305,     // Snake + dilated convs
  kHiftTail = 0x306,         // conv_post -> exp (clip 1e2) / sin -> iSTFT + overlap-add / envelope ->
                             // audio_limit clamp (0.99) -> seam crossfade or trim-fade -> held-tail cut
                             // -> splice of the previous first-commit tail (publish_skip, linear
                             // tail_fade) -> clip +-1 -> float for post (+ PCM16) ->
                             // one pinned copy. Order as production (incremental_s3gen.py _hift_rows,
                             // launch, collect): the clamp comes BEFORE the crossfade.
  // 0x400 render state / RNG / packing (L7)
  kStateNoiseGather = 0x400, // noise for the suffix frames from the row's tables
  kStateTailSave = 0x401,    // conv tails / HiFT caches of the last complete block
  kStatePrefixCopy = 0x402,  // copy-on-admit of a voice prefix's partial last page + its conv tails
                             // (VoiceView prefix_*_tails -> the row's StateView slot)
  // 0x500 megakernel control (L12)
  kMkBarrier = 0x500,
};

// ---------------------------------------------------------------------------
// Paged K/V pool. One physical page = 64 frames = 32 tokens. A row's page
// table maps its logical page i (frames [64i, 64i + 64)) to a physical page;
// the same physical index addresses every layer of all three domains, so one
// table serves the whole renderer. Voice prefixes are shared read-only pages
// (full pages only; a prefix's partial last page is copied into the row's
// first private page at admission: kStatePrefixCopy). Allocation never calls
// cudaMalloc: pages come from the device arena; no free page = admission
// back-pressure.
// ---------------------------------------------------------------------------
inline constexpr int32_t kPageFrames = 64;
inline constexpr int32_t kPageTokens = kPageFrames / kMelPerToken;  // 32

// Bytes of one physical page in each domain (all layers, K and V).
inline constexpr size_t kEnc1PageBytes = size_t(kEnc1Layers) * 2 * kPageTokens * kEncDim * sizeof(float);
inline constexpr size_t kEnc2PageBytes = size_t(kEnc2Layers) * 2 * kPageFrames * kEncDim * sizeof(float);
inline constexpr size_t kEstPageBytes = size_t(kFlowSteps) * kEstAttnLayers * kPageFrames * 2 * kEstInner * 2;
inline constexpr size_t kPageBytes = kEnc1PageBytes + kEnc2PageBytes + kEstPageBytes;  // 16,515,072
static_assert(kPageBytes == 16515072, "page geometry changed: update the container format in tools/kt2n_assets/format.py and the pool sizing");

// Device views. Layouts (row-major, innermost last):
//   enc1: fp32 [kEnc1Layers][2 (K, V)][n_pages][kPageTokens][kEncDim]   (post-projection K / V)
//   enc2: fp32 [kEnc2Layers][2 (K, V)][n_pages][kPageFrames][kEncDim]
//   est:  bf16 [kFlowSteps][kEstAttnLayers][n_pages][kPageFrames][2 * kEstInner]  (K | V side by side,
//         as production's chunk-cache est_kv)
struct PoolView {
  float* enc1;
  float* enc2;
  uint16_t* est;  // bf16 bits
  int32_t n_pages;
  int32_t reserved;
};

KT2N_HD inline size_t enc1_offset(const PoolView& p, int32_t layer, int32_t kv, int32_t page, int32_t tok) {
  return ((size_t(layer * 2 + kv) * size_t(p.n_pages) + size_t(page)) * kPageTokens + size_t(tok)) * kEncDim;
}
KT2N_HD inline size_t enc2_offset(const PoolView& p, int32_t layer, int32_t kv, int32_t page, int32_t frame) {
  return ((size_t(layer * 2 + kv) * size_t(p.n_pages) + size_t(page)) * kPageFrames + size_t(frame)) * kEncDim;
}
KT2N_HD inline size_t est_offset(const PoolView& p, int32_t step, int32_t layer, int32_t page, int32_t frame) {
  return ((size_t(step * kEstAttnLayers + layer) * size_t(p.n_pages) + size_t(page)) * kPageFrames + size_t(frame)) *
         (2 * kEstInner);
}

// Host-side page allocator (implemented by L7). Thread-compatible: only the
// render dispatcher thread calls it.
class PagePool {
 public:
  virtual ~PagePool() = default;
  virtual int32_t pages_total() const = 0;
  virtual int32_t pages_free() const = 0;
  // All-or-nothing: false (nothing allocated) when fewer than n pages are free.
  virtual bool alloc(int32_t n, int32_t* out_pages) = 0;
  // Reference counts (shared voice-prefix pages): retain +1, release -1, freed at 0.
  virtual void retain(const int32_t* pages, int32_t n) = 0;
  virtual void release(const int32_t* pages, int32_t n) = 0;
  virtual PoolView view() const = 0;
};

// ---------------------------------------------------------------------------
// Per-slot render state (not paged) and per-voice tables
// ---------------------------------------------------------------------------
struct StateView {
  float* target_noise;     // fp32 [slots][kMel][noise_cap] (row stride noise_cap, target frames from 0).
                           // Frames [0, noise_w) = the 4,000,000 + seed draw (value(c, t) = draw c * noise_w
                           // + t); frames past noise_w = the 5,000,000 + seed extension appended per channel
                           // (runtime.py:15444), so extending never re-lays the table
  float* enc_pre_tail;     // fp32 [slots][2][kEncDim]   pre-lookahead conv tail (chunk-cache pre_tail: the
                           // 2 embed-output tokens before the stable end)
  float* enc_up_tail;      // fp32 [slots][2][kEncDim]   up-layer conv tail (chunk-cache up_tail)
  uint16_t* est_tail;      // bf16 [kFlowSteps][slots][est_tail_width]: per causal conv (forward order) its
                           // [in_ch][2] tail, channel-major, as chunk_cache_20260927.py tail_views
  float* hift_mel_cache;   // fp32 [slots][kMel][kMelCacheFrames] (RowDesc::mel_cache frames are valid)
  float* hift_src_cache;   // fp32 [slots][kSourceCacheSamples] (RowDesc::src_cache samples are valid)
  float* hift_held_tail;   // fp32 [slots][kSourceCacheSamples]: the held-back tail for the next crossfade
                           // (same length as the source cache: both are the last <= 3840 samples)
  float* hift_phase;       // fp32 [slots][kHarmonics] (phase continuity; Q_HIFT_PHASE_CONT=0 live: unused)
  int32_t slots;
  int32_t noise_cap;       // target-noise frames per slot; 2 * kMaxRowTokens covers every row (production
                           // draws (budget + 3) * 2, then extends to (budget * SILENCE_FREE_BUDGET_CEILING
                           // + 3) * 2 when a row passes its budget)
  int32_t est_tail_width;  // sum over the 31 convs of 2 * in_channels
  int32_t reserved;
};

struct VoiceView {
  const float* prompt_feat;     // fp32 [voices][max_prompt_frames][kMel]
  const float* prompt_noise;    // fp32 [voices][kMel][max_prompt_frames]: prompt-region noise (G_PROMPT_NOISE=voice:
                                // the seed-0 draw, default_rng(4,000,000), shape [80, prompt_frames])
  const float* spk_cond;        // fp32 [voices][kMel]: spk_embed_affine_layer(l2normalize(embedding))
  const int32_t* prompt_tokens; // [voices][max_prompt_frames / 2]
  const int32_t* prompt_frames; // [voices]
  const int32_t* prefix_pages;  // [voices][max_prefix_pages]: shared pages of the prompt's complete blocks
  const int32_t* prefix_tokens; // [voices]: tokens covered by those pages' final K/V (the prefix's `stable`)
  // The prefix's conv tails at its stable end (chunk_cache_20260927.py _build_prefix / _load_prefix):
  // copied into the row's StateView slot at admission together with the partial last page.
  const float* prefix_enc_tails;     // fp32 [voices][2 (pre, up)][2][kEncDim]
  const uint16_t* prefix_est_tails;  // bf16 [voices][kFlowSteps][est_tail_width] (StateView layout)
  int32_t n_voices;                  // capacity: built-ins + inline LRU + loaded saved voices
  int32_t max_prompt_frames;
  int32_t max_prefix_pages;
  int32_t reserved;
};

// ---------------------------------------------------------------------------
// Ragged row descriptors (device memory, one per row of a call)
// ---------------------------------------------------------------------------
struct alignas(16) RowDesc {
  int32_t slot;            // render-state slot (StateView index); -1 = padding row
  int32_t voice;           // VoiceView index
  uint16_t phase;          // Phase. Equal to CallDesc::phase, except kFirstCommit rows inside a
                           // kSteadyCommit call: live production batches commit-0 rows with steady rows
                           // (acoustic_batch_key is empty with every cache off, BATCHED_S3GEN_COMMITS=1,
                           // CROSS_VOICE_S3GEN=1), and the RC makes the same grouping decisions. Tiles
                           // key per-row behaviour on the fields and flags below, never on this value
  uint16_t flags;          // RowFlags
  int16_t lookahead;       // trailing look-ahead tokens cropped from the output (0..3; 0 with kRowFinalize).
                           // The estimator does not solve them either (Q_FLOW_LOOKAHEAD=0 live): its valid
                           // frames are fr_suffix - 2 * lookahead (chunk_cache_20260927.py est_valid)
  int16_t flow_steps;      // 2, or 1 for kPreview

  // Token domain: encoder stage 1 (kTokenChunk blocks).
  int32_t tok_past;        // tokens (prompt included) whose stage-1 K/V are final in the pool
  int32_t tok_suffix;      // tokens computed this call: open block + new tokens + look-ahead (+3 silence on finals)
  int32_t tok_keys;        // keys visible to the last suffix token: tok_past + tok_suffix
  int32_t tok_off;         // first token of this row in CallDesc::tokens (packed, no padding)

  // Frame domain: encoder stage 2 + estimator (kMelChunk blocks). frames = 2 * tokens.
  int32_t fr_past;
  int32_t fr_suffix;
  int32_t fr_keys;
  int32_t fr_off;          // first frame of this row in the packed frame activations
  int32_t prompt_frames;   // cond = prompt mel on [0, prompt_frames); published output starts after it

  // Output window.
  int32_t out_fr_begin;    // absolute frame of the first playable mel frame of this call
  int32_t out_fr_count;    // playable frames (after the look-ahead crop)
  int32_t pcm_off;         // first sample of this row in the call's packed PCM output
  int32_t pcm_count;       // samples this row publishes this call
  int32_t publish_skip;    // head samples already published (first-commit tail) and skipped now

  // Paged K/V.
  int32_t page_off;        // first entry of this row in CallDesc::page_table
  int32_t n_pages;         // logical pages mapped (covers fr_keys frames)
  int32_t n_shared_pages;  // leading pages that belong to the voice prefix (read-only)

  // RNG and HiFT continuity.
  int32_t noise_w;         // W of the row's target-noise table: value(c, t) = PCG64 draw c * W + t
  int32_t committed_tokens;// tokens published before this call (HiFT seed input, seams)
  int32_t src_cache;       // HiFT source-cache samples carried in, 0..kSourceCacheSamples. Also the held
                           // tail's length and the seam overlap's upper bound. Below 3840 after a 3-4 token
                           // first commit (G_SHORT_FIRST_COMMIT=1 live): that seam then uses a whole
                           // np.hamming(2 * overlap) window, not the head of the 2 x 3840 one
                           // (incremental_s3gen.py seam_rise_fall / _short_crossfade)
  int32_t mel_cache;       // HiFT mel-cache frames carried in, 0..kMelCacheFrames (6 after a 3-token commit)
  int32_t tail_fade;       // samples of the first-commit tail splice (kTailFadeSamples; linear ramp)
  float t_mid;             // mean-flow split point (Q_MF_TMID_PERMILLE=500 -> 0.5)
  uint64_t seed;           // segment seed = request salt + ordinal
  uint64_t philox_seed;    // HiFT source noise (and preview flow noise): torch CUDA generator seed
  uint64_t philox_offset;  // torch PhiloxCudaState offset at this call's first draw
  int32_t reserved[4];
};
static_assert(sizeof(RowDesc) == 160, "RowDesc layout is frozen (G0)");

struct OpDesc;

// Call descriptor: one per launch / graph replay, in device memory.
struct alignas(16) CallDesc {
  uint16_t phase;          // Phase
  uint16_t flags;
  int32_t n_rows;          // real rows; rows[n_rows .. rows_bucket) are padding
  int32_t rows_bucket;
  int32_t suffix_bucket;   // tokens (graph key)
  int32_t total_tokens;    // packed tokens over real rows
  int32_t total_frames;
  int32_t total_pcm;
  int32_t flow_step;       // the step this launch computes (Phase A/B per-step graphs); -1 = all
  const RowDesc* rows;     // [rows_bucket]
  const int32_t* tokens;   // packed speech-token ids [total_tokens]
  const int32_t* page_table;  // physical page ids
  PoolView pool;
  StateView state;
  VoiceView voices;
  void* workspace;         // activation scratch, sized for (rows_bucket, suffix_bucket) at capture
  uint64_t workspace_bytes;
  float* mel_out;          // fp32 [total_frames][kMel] (checks / oracle comparison; may be null)
  float* pcm_f32;          // fp32 packed PCM for the post pool
  int16_t* pcm16;          // packed PCM16
  uint32_t* op_done;       // MK-R: per-op completed-tile counters (generation-tagged)
  uint32_t* next_ticket;   // MK-R: global ticket counter
  uint32_t generation;     // MK-R: this call's generation (counters compare against it)
  int32_t n_ops;
  const OpDesc* ops;       // [n_ops] in topological order
  const int32_t* deps;       // dependency lists (op indices), see OpDesc::dep_begin
  int32_t reserved[4];
};

// ---------------------------------------------------------------------------
// Per-op descriptor and the tile function
// ---------------------------------------------------------------------------
inline constexpr int32_t kTileThreads = 256;           // every tile, every mode
inline constexpr uint32_t kTileSmemBytes = 5 * 16384;  // MK-R shared-memory ring: 5 x 16 KB

enum OpFlags : uint32_t {
  kOpBias = 1u << 0,
  kOpResidual = 1u << 1,      // out += in[1]
  kOpMaskRows = 1u << 2,      // zero frames past each row's length
  kOpTimeAdd = 1u << 3,       // + per-row time projection (estimator resnets)
  kOpWritesPool = 1u << 4,    // writes K/V into pool pages (layer = OpDesc::layer)
  kOpReadsPool = 1u << 5,
  kOpCausalPad = 1u << 6,     // left pad (k - 1) * dilation from the row's state tail
  kOpLookaheadPad = 1u << 7,  // right pad (pre-lookahead conv1)
  kOpPdl = 1u << 8,           // Phase A/B: launched with programmatic dependent launch
  kOpFrames = 1u << 9,        // L4: the op runs at frame rate (encoder stage 2), else token rate
};

struct alignas(16) OpDesc {
  uint16_t kind;          // OpKind
  uint8_t math;           // MathMode
  uint8_t act;            // Activation of the epilogue
  uint32_t flags;         // OpFlags
  int32_t layer;          // pool layer within its domain (K/V reads / writes), -1 = none
  int32_t step;           // flow step 0 / 1, -1 = not a flow op
  int32_t n_tiles;        // tiles of this op for the bucket (Phase A/B: gridDim.x)
  int32_t tile_m;         // output rows (frames / tokens) per tile
  int32_t tile_n;         // output columns per tile
  int32_t m, n, k;        // logical GEMM / conv sizes at the bucket (m = packed rows)
  int32_t kernel, stride, dilation, pad_left, pad_right;  // conv geometry
  int32_t dep_begin;      // CallDesc::deps[dep_begin .. dep_begin + dep_count): ops that must finish first
  int32_t dep_count;
  int32_t ticket_begin;   // MK-R: global ticket of tile 0 (tickets are op-major, tiles ascending)
  uint32_t smem_bytes;    // dynamic shared memory a tile needs (<= kTileSmemBytes)
  float scalar[4];        // eps, slope, scale, ... (op-defined)
  const void* w[4];       // weights in the s3gen arena: weight, bias, gamma, beta / alpha
  const void* in[4];      // activations (packed ragged layout)
  void* out[2];
  int64_t ld_in[4];       // leading dimensions, elements
  int64_t ld_out[2];
  int32_t reserved[4];
};
static_assert(sizeof(OpDesc) == 240, "OpDesc layout is frozen (G0)");

// Per-block context handed to a tile.
struct TileCtx {
  void* smem;             // dynamic shared memory base
  uint32_t smem_bytes;
  int32_t block;          // Phase A/B: blockIdx.x; MK-R: the persistent block's index
  uint64_t t_start_ns;    // %globaltimer when the tile started (traces)
};

#if defined(__CUDACC__)
// The one entry point every op implements (a switch on OpDesc::kind in
// src/render/dispatch.cuh calls the owning lane's tile function). Contract:
//   * called by all kTileThreads threads of a block with the same arguments;
//   * `tile` in [0, op.n_tiles);
//   * Phase A/B: `op_kernel<<<op.n_tiles, kTileThreads, op.smem_bytes, stream>>>(op, call)` with
//     tile = blockIdx.x; with kOpPdl the tile calls griddepcontrol.wait (cudaGridDependencySynchronize)
//     before its first read of a predecessor's output;
//   * MK-R: a persistent block claims ticket = atomicAdd(call.next_ticket, 1), maps it to (op, tile)
//     through ticket_begin, waits (acquire) until op_done[d] == n_tiles(d) for every dependency d of
//     this call's generation, runs the tile, then __threadfence() and increments op_done[op]
//     (release). Tickets are claimed in topological order, so a waiting block only ever waits on
//     tickets already claimed by running blocks: no deadlock, no grid.sync. The grid is capped
//     (64-96 blocks or an SM partition) so LM decode is never locked out.
__device__ void run_tile(const OpDesc& op, const CallDesc& call, int32_t tile, TileCtx& ctx);
#endif

// ---------------------------------------------------------------------------
// RNG contract (exact ports; L7 owns them, L6 the HiFT draws)
// ---------------------------------------------------------------------------
// Flow noise, every streaming / final / whole-render row (incremental_s3gen.py:88
// draw_state_noise; G_PROMPT_NOISE=voice live):
//   rng = numpy default_rng(4,000,000 + seed)            (SeedSequence -> PCG64)
//   rng.standard_normal((80, prompt_frames), float32)    drawn and DISCARDED under voice mode
//                                                        (it still advances rng)
//   prompt noise = default_rng(4,000,000).standard_normal((80, prompt_frames), float32)
//   target noise = rng.standard_normal((80, (budget + 3) * 2), float32)  -> W = (budget + 3) * 2
//   extension when a row outgrows W (runtime.py:15444 _extend_incremental_noise):
//     default_rng(5,000,000 + seed).standard_normal((80, frames - W), float32), appended per channel
// numpy fills C order, so value(c, t) is draw c * W + t of its stream: the table is
// materialised at admission (as production does), never generated per frame.
// float32 standard_normal = numpy's float ziggurat over PCG64 next_uint32 (which
// buffers the high half of a 64-bit output: Pcg64State::has_uint32 / uinteger).
//
// HiFT source noise and phases, every incremental render row
// (incremental_s3gen.py:3561 / :3962, _hift_rows :3031):
//   s = 3,000,000 + seed * 10,000 + committed_tokens        (uint64 arithmetic)
//   phases: torch CPU generator after manual_seed(s) (mt19937): rand((1, 9, 1)) -> U(-pi, pi),
//           harmonic 0 phase forced to 0;
//   noise:  torch CUDA generator after manual_seed(s) (Philox4x32-10, offset 0): randn((1, 9, L));
//           the stock code's final unused randn_like(uv) is skipped.
//
// 1-step preview (runtime.py:15916 / _decode_cuda_preview_batch :14254): global torch RNG
// after manual_seed(group[0].seed + 1,000,000): the batch's FIRST row's seed for the whole group;
// draws in order (P = prompt frames, w = padded preview tokens):
//   flow noised_mels = randn(1, 80, 2w) (s3gen.flow_inference; shared by every row), then
//   z = randn_like(mu) [B, 80, P + 2w - 6] (finalize=False crops 6 frames BEFORE the decoder), then
//   z[..., P - 6:] = noised_mels (flow_matching.py: prompt_len = mu.size(2) - noised_mels.size(2)),
//   so the last 6 prompt frames take noised_mels' first 6 columns; then HiFT as the stock
//   HiFTGenerator.inference draws (CPU rand (B, 9, 1) for phases, CUDA randn_like for noise, and the
//   final unused randn_like(uv)) on the same generators. Rows depend on their place in the group:
//   the host gives each row its own philox_offset / phase draw so tiles stay batch-invariant.
//
// Segment seed (runtime.py:2946 _request_seed_salt, DC_REQUEST_SEED=request):
//   salt = (big-endian u32 of sha256(request_id)[0:4] & 0x3FFFFFFF) * 1024; seed = salt + ordinal.
inline constexpr uint64_t kFlowNoiseSeedBase = 4000000ull;
inline constexpr uint64_t kFlowNoiseExtSeedBase = 5000000ull;
inline constexpr uint64_t kHiftSeedBase = 3000000ull;
inline constexpr uint64_t kHiftSeedMul = 10000ull;
inline constexpr uint64_t kPreviewSeedOffset = 1000000ull;
inline constexpr uint64_t kSeedSaltUnit = 1024ull;

constexpr uint64_t hift_seed(uint64_t seed, int64_t committed_tokens) {
  return kHiftSeedBase + seed * kHiftSeedMul + uint64_t(committed_tokens);
}
constexpr int32_t target_noise_width(int32_t budget_tokens) { return (budget_tokens + kLookahead) * kMelPerToken; }

struct Pcg64State {      // numpy bit_generator state after SeedSequence seeding
  uint64_t state_hi, state_lo;
  uint64_t inc_hi, inc_lo;
  uint32_t has_uint32;   // numpy's buffered half word
  uint32_t uinteger;
};
struct PhiloxState {     // torch PhiloxCudaState
  uint64_t seed;
  uint64_t offset;       // advanced per draw exactly as torch's distribution kernels do
};
struct Mt19937State {    // torch CPUGeneratorImpl (host only)
  uint32_t mt[624];
  int32_t index;
  uint32_t seed_lo;
};

// ---------------------------------------------------------------------------
// Streams and events (pipelining contract)
// ---------------------------------------------------------------------------
// Five streams per engine, created once at startup (priorities from
// cudaDeviceGetStreamPriorityRange; "high" = greatest priority):
//   kSLm      high  LM prefill + decode steps (one step queued ahead; tokens / stop flags return one
//                   step late through a device ring). Nothing else ever runs here.
//   kSUrgent  high  first commits (commit 0), second commits (commit 1) and previews.
//   kSBulk    low   later commits, final flushes, whole renders (cut into <= 5 ms pieces so urgent work
//                   interleaves), prefix builds.
//   kSCopy    high  device -> pinned PCM copies, pinned -> device staging (tokens, descriptors, noise).
//   kSAux     low   voice builds, warm-ups, non-urgent uploads.
// If the sm_120 probe shows priorities do not separate the classes, the same roles map to SM
// partitions (green contexts); the roles and joins below do not change.
// Joins: only cudaEventRecord + cudaStreamWaitEvent (events created with cudaEventDisableTiming,
// except the trace pair of each render). HiFT of commit k may overlap the flow of commit k + 1.
// The host learns completion by cudaEventQuery or a host callback on kSCopy that wakes the post
// pool; the conductor and dispatcher threads never call a blocking stream / device sync.
// Queue bound: per render stream at most 1 batch executing + 1 queued, so the driver queue never
// fills. The RC dispatcher makes production's launch decisions (T9 deferral, ACOUSTIC_LAZY,
// commit coalescing, D_SCHEDULE_ALONE, first commit by sharing, startup hold); streams change only
// how fast and how overlapped they execute.
enum class StreamRole : int32_t { kSLm = 0, kSUrgent = 1, kSBulk = 2, kSCopy = 3, kSAux = 4 };
inline constexpr int32_t kStreamRoles = 5;
inline constexpr int32_t kMaxQueuedBatchesPerStream = 2;  // 1 executing + 1 queued
inline constexpr double kBulkPieceMaxMs = 5.0;

struct StreamSet {
  CUstream_st* s[kStreamRoles];
};

// Which stream a render of this phase / commit index goes to. A mixed call passes its most urgent
// row (kFirstCommit, or the smallest commit index).
constexpr StreamRole stream_for(Phase phase, int32_t commit_index) {
  return (phase == Phase::kPreview || phase == Phase::kFirstCommit ||
          (phase == Phase::kSteadyCommit && commit_index <= 1))
             ? StreamRole::kSUrgent
             : StreamRole::kSBulk;
}

}  // namespace kt2n::render
