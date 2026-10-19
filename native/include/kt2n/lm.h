// kt2n::lm -- the torch-free native Kitten TTS 2 LM (lane L3, 10 Oct 2026).
//
// What it does: per-voice prefix K/V, batched suffix prefill, decode with one CUDA graph per row bucket
// (1..16, 20, 24, 32, 40, 48, 64), the production sampler chain per row, production's stop rules on the device,
// and host-free run-ahead (step k+1 is queued before step k's tokens are read; tokens and stop flags come back
// through a ring in mapped host memory).  No torch, no Python: CUDA runtime + the kt2n kernel lanes only.
//
// Memory: the engine's weights (one allocation, kt2n_pf_model_load) and ONE device arena (K/V slots, step
// scratch, sampler and stop state, voice store, logits) plus one mapped host ring, all allocated in create().
// Nothing is allocated while serving: a full slot table or voice store is back-pressure (submit / build_voice
// return an error), never an allocation.  The voice store is a first-fit free list inside the arena:
// release_voice() returns a voice's range once no row reads it (112 KB per prefix token).
//
// K/V layout ("flat row", the prefill lane's layout): slot s holds [2][layers][8][max_ctx][128] bf16; logical
// position t of the row is column t.  A row keeps its slot from admission to finish; the decode step reads each
// row's slot through a row -> slot table (KT2N_STEP_FLAG_ROW_KV), so rows join and leave without moving K/V.
//
// Threading: one driver thread calls pump(); submit(), cancel() and stats() may be called from any thread.
// build_voice() / drop_voice() / release_voice() run on the driver thread (or before the first pump()).
//
// Numerics (statistical equality, not bit identity): the kernel lanes' arithmetic (bf16
// activations, fp32 accumulation, fp32 softmax/norm statistics).  Checked against PyTorch with next-token logits
// relative L2 and a sampled-distribution test.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kt2n::lm {

// Token ids of the Kitten TTS 2 vocabulary that the LM loop needs (runtime.py @ ebcdd3f).
inline constexpr int32_t kSpeechStart = 151669;  // column 0 of the audio slice
inline constexpr int32_t kSpeechEnd = 151670;    // -> finish "eos"
inline constexpr int32_t kStopToken = 151674;    // -> finish "stop" (named apart from Finish::kStop: -Wshadow)
inline constexpr int32_t kAudioIdBase = 151675;  // first audio code token
inline constexpr int32_t kAudioVocab = 6561;
inline constexpr int kMaxRows = 64;              // largest decode bucket
inline constexpr int kMaxPrefillRows = 16;       // rows per admission prefill call (the head's limit)

// Decode buckets with a captured graph (the plan's list).
inline constexpr int kBuckets[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 40, 48, 64};

// ---------------------------------------------------------------------------------------------- sampler settings
// The public sampler surface (kitten_tts/sampler_params.py @ ebcdd3f): presets and the advanced fields.
enum class SamplerMode : uint8_t { kRelease, kStable, kExpressive, kAdvanced };

struct SamplerSettings {
  float temperature = 0.9f;
  float top_p = 0.9f;
  int32_t top_k = 50;
  float min_p = 0.0f;
  int32_t max_new_tokens = 0;  // 0 = none (SMP cap: every LM step counts)
};

// Preset values: release 0.7/0.9/100/0.05, stable 0.8/0.8/50/0, expressive 0.9/0.9/50/0 (the default mode).
SamplerSettings preset(SamplerMode mode);

// Port of sampler_params.parse (ranges, integer fields) + resolve: the `mode` preset when sent, else the
// documented mode default (expressive) when any of temperature / top_p / top_k / min_p was sent, else
// `no_settings_default`; then the explicit fields.  Returns "" when valid, otherwise production's 400 message
// ("temperature must be a number between 0.3 and 2.0.") with *param naming the field.
struct SamplerRequest {
  std::optional<std::string> mode;  // "stable" | "expressive" (case-insensitive, trimmed)
  std::optional<double> temperature, top_p, top_k, min_p, max_new_tokens;
};
std::string resolve_sampler(const SamplerRequest& req, SamplerMode no_settings_default, SamplerSettings* out,
                            std::string* param);

// ---------------------------------------------------------------------------------------------- engine config
struct Config {
  std::string engine_path;           // engine.kt2n
  std::string expect_source_sha256;  // "" = do not check (otherwise the converter's recorded source sha)
  int device = 0;
  int max_rows = 64;                 // concurrent rows = K/V slots (<= kMaxRows)
  int max_ctx = 4096;                // K/V columns per slot (production CUDA_GRAPH_MAX_CTX default 4096)
  int ctx_limit = 4096;              // production's CUDA_GRAPH_MAX_CTX for the demo-chunked ceiling rule
  int max_prefill_tokens = 4096;     // new tokens per admission prefill call (all rows)
  size_t voice_store_bytes = size_t{256} << 20;  // per-voice prefix K/V (112 KB per prefix token)
  int run_ahead = 2;                 // decode steps in flight (>= 1)
  // While an admission (prefill + first token) is in flight, keep at most this many decode steps queued (0 = run_ahead),
  // so the new rows join the bucket one step sooner and the prefill shares the GPU with less queued decode work
  // (execution only).  Sprint 4, arrivals bench, 6 interleaved rounds at 1 / 10 / 20 rows: the new row's first token
  // 8-27% sooner, its 10th token 1-15% sooner, other rows' token gaps unchanged (lane log).
  int admission_run_ahead = 1;
  // A blocking pump() also wakes when an in-flight admission finishes (not only when a decode step does), so the new
  // rows are joined at once.  Measured mixed (sprint 4), off.
  bool join_poll = false;
  bool admission_priority = true;    // the admission stream gets the highest stream priority
  // Sprint 5: the decode stream's CUDA priority, in levels above the device's lowest (0 = lowest, the default and
  // the level of L7's bulk renders; clamped below the highest, which L7's urgent renders use).  A higher level lets
  // the block scheduler start pending decode CTAs before pending bulk-render CTAs; it never preempts running ones.
  // Execution only.  Sprint 6: the process env KT2N_LM_DECODE_PRIORITY=<level>, when set, overrides this field at
  // create (a lab A/B knob for engine-level runs; Stats::decode_priority reports the priority used).
  int decode_priority_level = 0;
  bool async_admission = true;       // prefill + first draw on a second stream, beside decode (false = on the
                                     // decode stream, which then stalls for the prefill)
  bool graphs = true;                // decode through captured graphs (false = eager launches, for tests)
  bool pdl = true;                   // programmatic dependent launch between the step's kernels
  bool silence_armed = true;         // SKIP_OPENING_SILENCE_ARMED: sub-gate ids are not charged to the budget
  int ceiling_factor = 2;            // SILENCE_FREE_BUDGET_CEILING
  int tagged_slot_margin = 16;       // TAGGED_SLOT_MARGIN_TOKENS
  int fuse_norm_max_rows = 0;        // EXPERIMENTAL (measured slower, lane log): fused residual + RMSNorm up to N rows
  int split64_max_rows = 2;          // buckets up to this many rows: 64-key attention splits for rows <= 640 keys
  // Sprint 5: buckets from kv_next_min_rows rows up (0 = off) let each layer's attention L2-prefetch the next layer's
  // K/V for its own rows (KT2N_STEP_O1_KV_NEXT; execution only, logits bit-identical), up to kv_next_budget_mb of K + V
  // per layer over all rows.  The step benches decide the default (lane log, sprint 5).
  int kv_next_min_rows = 0;
  int kv_next_budget_mb = 48;
  // Sprint 5: buckets of split_min_rows..split_max_rows rows (split_min_rows 0 = off) run each decode step as two
  // half-batch steps side by side (two branches of the step's graph on two streams, joined before the ring post).
  // Each row's arithmetic is the same as in the full bucket (k1 / k3 per-row contracts): logits and sampled tokens
  // are bit-identical (checked); execution only.  Measured (nsys step-span medians, 600 keys, one voice per row,
  // lane log sprint 5): 14 rows 1.59 -> 1.46 ms, 15 rows 1.69 -> 1.53, 16 rows 1.77 -> 1.61; 12-13 rows 2-5% slower,
  // 20 / 32 rows within 1.5%: hence 14..16 by default.  Costs one more scratch + attention workspace (~70 MB at
  // max_ctx 4,096).
  int split_min_rows = 14;
  int split_max_rows = 16;
  // split_branch_rows = N > 0: instead of two halves, branches of at most N rows each (up to 8 branches).  Measured
  // slower at 20-32 rows (every branch streams all weights; lane log sprint 5).  0 = two halves (the default).
  int split_branch_rows = 0;
  // Sprint 6: split_first_rows = F > 0: a split bucket of B rows runs as two uneven branches, F rows (on the decode
  // stream) + B - F, instead of two halves (e.g. 20 = 16 + 4: the 16-row branch on the 16-row decode kernel, the 4
  // rows on the lean <= 8-row one), when B - F >= 3 (a 1-2-row branch would take the 1-row arithmetic); else two
  // halves.  Ignores split_branch_rows.  Measured not faster (lane log sprint 6).  0 = off (the default).
  int split_first_rows = 0;
  // Decode attention reads a row's voice-prefix keys from the voice store instead of the row's own copy (same
  // values, same results), so rows of one voice share those reads through L2.  A dropped or released voice's K/V
  // stays valid for rows still decoding (release_voice frees the range only after their slots are released).
  bool share_voice_kv = true;
  // ... only in buckets of at least this many rows.  A 1-row step has no other row to share with, and the two-source
  // tile loads cost 1-3% there (sprint 3, nsys medians: 1 x 2,000 keys 722 -> 740 us); same results either way.
  int share_voice_kv_min_rows = 2;
  bool debug_logits = false;         // keep every step's audio logits in host memory (tests only)
  std::vector<int> buckets;          // empty = kBuckets (entries above max_rows dropped)
};

// ---------------------------------------------------------------------------------------------- rows
enum class Finish : uint8_t {
  kNone = 0,
  kEos = 1,           // SPEECH_END
  kStop = 2,          // STOP
  kBudget = 3,        // budget (speech-only count, or every step for DEMO rows)
  kMaxNewTokens = 4,  // the request's max_new_tokens
  kCeiling = 5,       // 2 x budget, demo-chunked slot margin, or the slot is full
  kCancelled = 6,
};
const char* finish_name(Finish f);  // "eos", "stop", "budget", "max_new_tokens", "ceiling", "cancelled"

struct RowRequest {
  uint64_t tag = 0;                // caller's id, echoed in every event (unique among live rows)
  int voice = -1;                  // build_voice() id, or -1 = no prefix (the suffix is the whole prompt)
  std::vector<int64_t> suffix;     // prompt ids after the voice prefix (>= 1)
  // Without a voice, position 0 of the suffix takes this projected speaker vector (bf16 bits [2048]) if given.
  std::vector<uint16_t> speaker_bf16;
  SamplerSettings sampler;
  uint64_t seed = 0;               // the segment generator's seed (torch.Generator("cuda").manual_seed)
  int32_t budget = 200;            // job.budget, LM tokens
  bool demo = false;               // DEMO rows: every step counts against the budget
  bool demo_chunked = false;       // demo-chunked rows: ceiling at ctx_limit - tagged_slot_margin
  std::vector<int32_t> forced;     // tests only: force these sampled ids (LAB_FORCE_IDS analogue)
};

struct TokenEvent {
  uint64_t tag;
  int32_t token;       // full-vocabulary id
  int32_t index;       // 1-based generated-token count (new_tokens after this token)
  Finish finish;       // kNone, or why the row ended with this token (then it is the row's last event)
};

struct Stats {
  uint64_t steps = 0;           // decode steps launched
  uint64_t admissions = 0;      // prefill calls
  uint64_t rows_admitted = 0;
  uint64_t rows_finished = 0;
  int active_rows = 0;
  int pending_rows = 0;
  int bucket = 0;               // rows of the bucket the last step ran
  double last_step_ms = 0;      // host-observed launch -> tokens visible, last collected step
  size_t arena_bytes = 0;
  size_t weight_bytes = 0;
  int voices_live = 0;               // built and not dropped / released
  size_t voice_store_used = 0;       // bytes held by voices (incl. released ones that rows still read)
  size_t voice_store_free_max = 0;   // largest free range: a voice of up to this many bytes fits now
  int decode_priority = 0;           // the decode stream's CUDA priority (sprint 5), and the device's range
  int priority_lo = 0;               // ... lowest (largest number)
  int priority_hi = 0;               // ... highest
  int64_t priority_switches = 0;     // sprint 6: set_decode_priority switches applied
};

class Engine {
 public:
  // Loads the engine file, allocates the arena, captures the bucket graphs.  nullptr + *err on failure.
  static std::unique_ptr<Engine> create(const Config& cfg, std::string* err);
  virtual ~Engine() = default;

  // Raw speaker embedding (fp32 [spk_in], production's job.ref["speaker"] before spk_proj) -> projected bf16
  // [2048] through the engine file's spk_proj (Linear + LayerNorm eps 1e-5).  Synchronous.
  virtual int project_speaker(const float* speaker, int dim, std::vector<uint16_t>* out_bf16, std::string* err) = 0;

  // Per-voice prefix K/V: prefill of `prefix` (position 0 = the projected speaker vector) into the voice store.
  // Returns the voice id (>= 0) or -1 + *err.  Synchronous (startup / clone path, not per request).
  virtual int build_voice(const std::vector<uint16_t>& speaker_bf16, const std::vector<int64_t>& prefix,
                          std::string* err) = 0;
  virtual int voice_prefix_len(int voice) const = 0;
  // Marks the voice dead (no new rows).  Its store space is kept (not reused), so rows still decoding keep valid
  // prefix K/V (their own copy, and the store with share_voice_kv).
  virtual void drop_voice(int voice) = 0;
  // Marks the voice dead (no new rows, like drop_voice) AND gives its store space back: later build_voice calls may
  // reuse it once no row reads it any more.  Rows of the voice that are pending or decoding keep it until their
  // slot is released (after every step that ran them has been read), so a caller may release a voice at any time,
  // e.g. when its last request finishes or when it evicts the voice.  Voice ids are never reused.  Driver thread.
  // Returns 0, or -1 for an unknown or already released id.
  virtual int release_voice(int voice) = 0;

  // Queue a row; it is admitted (prefill + first token) at the next pump().  Returns 0, or -1 + *err when the
  // request is invalid or the engine is full (back-pressure: the caller retries later).
  virtual int submit(const RowRequest& row, std::string* err) = 0;
  // The row ends with a kCancelled event at the next pump() (no more tokens after it).
  virtual void cancel(uint64_t tag) = 0;

  // Drive the LM: admit pending rows, keep `run_ahead` decode steps in flight, and append the tokens of every
  // completed step to *events.  block = wait until at least one event is available (or nothing is running).
  // Returns the number of events appended, or -1 + *err on a CUDA failure (the engine is then unusable).
  virtual int pump(std::vector<TokenEvent>* events, bool block, std::string* err) = 0;

  // Rows that are admitted or pending.
  virtual int live_rows() const = 0;
  virtual Stats stats() const = 0;

  // Diagnostics: the host row table (active rows, slots, generations, flags) to stderr.
  virtual void debug_dump() const = 0;

  // Tests (debug_logits): audio-slice logits (bf16 bits [6568]) the row's index-th token was sampled from.
  virtual bool debug_logits(uint64_t tag, int index, std::vector<uint16_t>* out) const = 0;

  // Sprint 6: change the decode stream's priority level at run time (the same levels as
  // Config::decode_priority_level: 0 = lowest = L7's bulk renders, clamped below the top level).  Applied at the
  // start of the next pump(): later decode work goes to a decode stream of that priority, ordered after the earlier
  // work (an event), so results and token order are unchanged; only scheduling changes.  For a conductor policy,
  // e.g. a high level while first audio is at stake and level 0 when bulk renders fall behind.  Driver thread, or
  // any thread (the request is stored atomically).  Returns the CUDA priority that will be used, or -1 when the
  // implementation does not support it.
  virtual int set_decode_priority(int level) {
    (void)level;
    return -1;
  }
};

}  // namespace kt2n::lm
