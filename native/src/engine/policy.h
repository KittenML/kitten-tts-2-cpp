// Scheduling and playback decisions as pure host functions.
// Keeping policy separate from execution lets unit tests check each decision.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kt2n::engine::policy {

inline constexpr int kSampleRate = 24000;
inline constexpr int kSamplesPerToken = 960;  // 2 mel frames x 480
inline constexpr int kLookahead = 3;          // acoustic look-ahead tokens
inline constexpr int kSilenceToken = 4299;    // final flush appends three silence tokens

// Default timing and scheduling settings.
struct Levers {
  // Admission capacity and per-request segment concurrency.
  int request_batch_size = 10;
  int lane_inflight = 1;
  // Commit widths in acoustic tokens; the final width repeats.
  std::vector<int> helper_schedule{4, 5, 5, 10, 15, 25, 50};
  std::vector<int> alone_schedule{4, 5, 5, 10, 15, 25, 50};
  int d_decide_at_tokens = 4;
  // Delay ready acoustic work while playback has enough buffered audio.
  bool acoustic_deferral = true;
  bool acoustic_lazy = true;
  bool defer_starting = true;
  double defer_margin_s = 0.3;
  int lazy_min_batch = 3;
  int lazy_backlog_tokens = 50;
  bool t9_true_reservoir = true;
  // Combine compatible ready rows when partners are close to readiness.
  int coalesce_tokens = 12;
  double coalesce_min_reservoir_s = 0.3;
  // Group first commits separately and order ready groups by urgency.
  bool ragged_groups = true;
  int chunk_first_slots = 12;
  bool urgency_order = true;
  double urgency_ms = 80.0;
  // Minimum decoded rows and commit index for asynchronous rendering.
  bool async_steady = true;
  int async_min_rows = 2;
  int async_from_commit = 0;
  // Final-flush buckets limit both row count and relative sequence length.
  int final_bucket_rows = 8;
  double final_bucket_ratio = 2.0;
  // Optional early preview and adaptive startup hold.
  bool preview_enabled = true;
  int preview_tokens = 8;
  double hold_cap = 2.0, hold_base = 0.06, hold_per_listener = 0.006;
  // Whole-response rendering waits for playback runway, with bounded delays.
  bool dw_gate = true;
  double dw_runway_ms = 1300.0;
  double dw_max_wait_ms = 10000.0;
  double dw_min_gap_ms = 200.0;
  double dw_hard_s = 12.0;
  double dw_max_wait_low_ms = 250.0;
  // Optional clone work may interleave after first audio, with bounded delays.
  bool cf_interleave = true;
  double cf_max_wait_ms = 1500.0;
  double cf_min_gap_ms = 250.0;
  // Limit opening-silence removal and retain context before speech.
  bool skip_opening = true;
  int skip_cap = 48;
  int skip_keep = 1;
  int skip_run = 2;
  // Prefetch after an estimated fraction of the current segment has played.
  bool prefetch = true;
  double prefetch_fraction = 0.8;
  double prefetch_chars_per_s = 20.0;
  // Normalize one segment ahead after 1.5 seconds of audio or completion.
  double normalize_start_audio_s = 1.5;
  double normalize_wait_max_s = 20.0;
  int normalize_lookahead = 1;
  // Concurrent whole-response requests.
  int complete_slots = 3;
};

// Compatibility profiles vary whole-response capacity and rendering wait limits.
Levers live_levers(const std::string& app);

// ---------------------------------------------------------------- opening skip
bool opening_quiet_id(int s3_token);  // "no_speech" set: the 93 sub-gate ids minus {1466, 3653}
struct OpeningSkip {
  std::vector<int32_t> skipped, pending;
  bool closed = false;
  int skipped_tokens = 0;  // set when the opening closes
  int kept_tokens = 0;     // context retained before speech
};
// Append one sampled S3Gen token to `audio_tokens` through the skip. `armed` = segment 0, the skip
// is on for the request (not DEMO), and `audio_tokens` was empty when the skip started.
void opening_admit(OpeningSkip& s, bool armed, int32_t value, std::vector<int32_t>& audio_tokens,
                   const Levers& lv);
// A generation that never closed the opening keeps all its tokens.
void opening_restore(OpeningSkip& s, std::vector<int32_t>& audio_tokens);

// ---------------------------------------------------------------- commit sizing
struct CommitState {
  int committed = 0;
  int next = 5;              // next_commit_tokens (cumulative)
  int commit_index = 0;
  bool held = false;         // DEMO s2 row: never commits (next = 2**30), only its final flush
  bool inflight = false;     // an ASYNC launch for this row is not yet collected
  bool d_decided = false;
  bool alone = false;        // row selected for the single-request schedule
};
CommitState new_commit_state(const Levers& lv, bool demo_held);
// Ready when token_count >= next + 3, no launch is in flight, and the row is not held.
bool commit_ready(const CommitState& c, int token_count);
// On the first turn with >= 4 tokens, read `sharing` once.
// Returns true when this call made the decision.
bool d_prepare(CommitState& c, int token_count, int sharing, const Levers& lv);
// After a commit launch (sync or async): committed = next (commit k rendered ids[:next+3] and
// cropped the look-ahead), commit_index++, next = committed + width (helper; alone rows: D override).
void after_commit(CommitState& c, const Levers& lv);
int schedule_width(const std::vector<int>& schedule, int commit_index);

// ---------------------------------------------------------------- startup sizes
// min(cap, base + per * (sharing - 1)), never below 0.
double adaptive_hold_seconds(int sharing, const Levers& lv);
// Playback start target: max(0, hold, reserve); a realtime phrase has hold 0; reserve is zero by default.
double start_target_seconds(int sharing_rows, bool realtime_phrase, const Levers& lv);

// ---------------------------------------------------------------- T9
struct T9Row {
  bool finished = false;          // finished on this turn's decode step
  bool held_short = false;        // DEMO s2 row (never selected for commits)
  int segment_index = 0;
  bool preview_ready = false;     // the job has published audio
  double published_s = 0;         // job.published_samples / 24000
  double start_target_s = 0;      // start_target_seconds(len(active), ...)
  std::optional<double> reservoir_s;  // _reservoir_estimate(job, now); nullopt before first publication
  double dropped_s = 0;           // dropped audio included in reservoir accounting
  bool row_ready = false;         // commit ready or preview due
  int backlog = 0;                // len(audio_tokens) - next_commit_tokens - 3
};
struct T9Result {
  std::vector<int> selected;  // indices into the input, ascending
  std::vector<int> deferred_rows;  // rows whose t9_deferred_turns grows this turn
  int deferred = 0;
  bool starting = false;
  bool trigger = false;
};
T9Result t9_acoustic_rows(const std::vector<T9Row>& rows, const Levers& lv);

// ---------------------------------------------------------------- preview
// Preview is blocked when commit 0 needs >= 8 tokens (c0 + 3 >= 8).
bool preview_due(bool preview_ready, int token_count, int committed, bool held_short, int next_commit,
                 int commit_index, const Levers& lv);

// ---------------------------------------------------------------- groups, coalescing, order
// Ragged first commits use ("ragged","first",part); steady commits use ("ragged","steady",0).
// Coalescing uses the geometry owner, which never equals a ragged key, so default ragged groups are not held.
struct GroupKey {
  int kind = 0;        // 0 = ragged first, 1 = ragged steady, 2 = geometry owner
  int part = 0;        // ragged first-commit part (cut at chunk_first_slots)
  int64_t geometry = 0;  // geometry owner: prompt tokens | prompt frames | embedding width
  bool operator==(const GroupKey&) const = default;
  // Python tuple order: ("ragged","first",p) < ("ragged","steady",0); geometry tuples (ints)
  // never mix with ragged ones with ragged groups enabled.
  bool operator<(const GroupKey& o) const {
    if (kind != o.kind) return kind < o.kind;
    if (part != o.part) return part < o.part;
    return geometry < o.geometry;
  }
};
int64_t geometry_owner(int prompt_tokens, int prompt_frames, int embedding_width);

struct CommitRow {      // one ready (or not-ready partner candidate) row of this turn
  int id = 0;           // caller's index
  bool first = false;   // commit_index == 0 && committed == 0
  int commit_index = 0;
  int token_count = 0;
  int next = 0;
  int64_t geometry = 0;
  std::optional<double> reservoir_s;
  std::optional<double> d_buffer_ms;  // the request's simulated client buffer (None before release)
};
struct Group {
  GroupKey key;
  std::vector<int> rows;  // CommitRow ids
};
// Groups the ready rows (in `ready` order), cuts ragged first groups at chunk_first_slots,
// applies coalescing against `partners` (not-ready rows: no launch in flight, no pending preview,
// not held, not cancelled), sorts by key, then by urgency. Returns groups to render this turn in
// order; *held_rows counts rows coalescing held back.
std::vector<Group> plan_commit_groups(const std::vector<CommitRow>& ready, const std::vector<CommitRow>& partners,
                                      const Levers& lv, int* held_rows = nullptr);
// Experiment arm X-SCHED (never the RC; KT2N_CFG_X_SCHED): every ready commit renders on its turn (no T9
// deferral, no coalescing). Rows split into an urgent class (commit_index <= 1: first audio and the commit that
// must follow it) and the rest; urgent groups first, then the rest by underrun deadline (lowest client buffer
// minus a 60 ms margin; a row that published nothing counts as most urgent).
std::vector<Group> plan_commit_groups_xsched(const std::vector<CommitRow>& ready, const Levers& lv);
// Urgency key: (0, B) if lowest B < 80 ms; (1, 0) if any row's request published nothing; (2, B).
std::pair<int, double> d_buffer_key(const std::vector<std::optional<double>>& buffers_ms, const Levers& lv);
// Asynchronous launch rule: >= 2 rows decoded this turn and every row's commit_index >= 0.
bool launch_async(int active_rows_this_turn, const Levers& lv);

// Final buckets: sort by row length, cut at 8 rows or when a row is
// > 2.0 x the bucket's shortest. Input: (id, length) pairs; output: buckets of ids.
std::vector<std::vector<int>> final_flush_buckets(std::vector<std::pair<int, int>> rows, const Levers& lv);

// ---------------------------------------------------------------- DeliveryClock
class DeliveryClock {
 public:
  void publish(double now, int64_t samples);
  void hold();
  void release(double now);
  std::optional<double> buffer_ms(double now) const;
  bool started() const { return play_end_.has_value(); }
  std::optional<double> first_at() const { return first_at_; }
  int64_t published_samples() const { return published_; }
  int underruns() const { return underruns_; }
  double underrun_ms() const { return underrun_ms_; }

 private:
  void play(double now, int64_t samples);
  std::optional<double> first_at_, play_end_;
  int64_t published_ = 0, pending_ = 0;
  bool held_ = false;
  int underruns_ = 0;
  double underrun_ms_ = 0;
};

// ---------------------------------------------------------------- DW gate
struct DwInput {
  bool any_realtime = false;      // active rows that are not whole-render rows
  bool any_resting = false;       // resting rows (zero by default)
  double now_s = 0;
  std::optional<double> oldest_queued_s;
  std::optional<double> last_whole_render_s;
  bool waiting = false;           // a realtime row's request has no published audio, or a segment-0 row has no preview_ready
  std::optional<double> low_buffer_ms;  // min d_buffer over realtime rows (None -> treated as 0)
};
enum class DwReason { kNone, kIdle, kHard, kRunway, kMaxWait };
DwReason dw_due(const DwInput& in, const Levers& lv);
const char* dw_reason_name(DwReason r);

// ---------------------------------------------------------------- clone interleave
bool cf_interleave_due(bool any_active, bool any_before_first_audio, double waited_ms,
                       std::optional<double> since_last_build_ms, const Levers& lv);

// ---------------------------------------------------------------- admission
// Round robin over lanes from `cursor`; a lane yields its next job only while inflight < 1.
// `lanes[i]` = {has_next_job, inflight}. Returns the lane indices taken (<= capacity) and advances cursor.
struct LaneView {
  bool has_job = false;
  int inflight = 0;
};
std::vector<int> admission_take(const std::vector<LaneView>& lanes, int capacity, size_t* cursor, const Levers& lv);
// Capacity mode only (L8, the engine has more seats than decode rows; production never runs this way): lanes whose
// request already had a segment admitted (`continuing`) are offered first, with production's round-robin among them
// (cursor_cont); new requests share the rest with production's round-robin and cursor. Indexes as admission_take.
std::vector<int> admission_take_capacity(const std::vector<LaneView>& lanes, const std::vector<bool>& continuing,
                                         int capacity, size_t* cursor_cont, size_t* cursor, const Levers& lv);

// ---------------------------------------------------------------- prefetch
bool segment_prefetch_due(bool last_segment, bool segment_done, int64_t received_samples, size_t segment_chars,
                          int inflight_segments, const Levers& lv);

// ---------------------------------------------------------------- seeds
uint64_t request_seed_salt(const std::string& request_id);

}  // namespace kt2n::engine::policy
