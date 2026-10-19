// L7 render calls: ragged row descriptors (host build) and the render CALL API the conductor uses.
//
//   conductor (L8)                       this header
//   -------------                        -----------
//   RowCommitState::plan_commit()   ->   RowRender {geometry, slot, voice, ids, pages, seed}
//   group of rows (production's          build_call(): buckets, RowDesc table, packed tokens,
//   grouping decision)                   page table, PCM packing; pages allocated (all-or-nothing)
//                                        RenderCaller::launch(plan, stream role):
//                                          pinned staging -> one H2D copy -> captured graph for
//                                          (role, phase, rows bucket, suffix bucket) -> one D2H copy of
//                                          the call's PCM into pinned host memory -> event
//   RenderCaller::poll(ticket)      <-   done? then pcm(ticket) = packed float PCM of every row
//
// The chunk-cache geometry is production's (chunk_cache_20260927.py run :607): per row
//   pt = prompt tokens, past = the row's pool position (voice prefix `stable`, then the last
//   complete 25-token block before the committed end), suffix = prompt[past:] + ids (+ 3 x 4299 on a
//   final) when past < pt else ids[past - pt:], new_past = floor((pt + committed_after) / 25) * 25.
// Whole renders (kWholeRender) and previews (kPreview) start at position 0 and use call-owned
// scratch pages, released by the caller after the call completes.
#pragma once

#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include "kt2n/render/commit_plan.h"
#include "kt2n/render/ops.h"
#include "kt2n/render/state.h"

struct CUctx_st;

namespace kt2n::render {

struct RowRender {
  int32_t slot = -1;                 // StateView slot (whole renders / previews: a slot is still needed
                                     // for the noise table; they write no slot state)
  const VoicePrefix* voice = nullptr;
  RenderGeometry geom;
  const int32_t* ids = nullptr;      // the row's S3Gen ids (>= geom.render_tokens - geom.silence_tokens)
  RowPages* pages = nullptr;         // streaming rows; nullptr for kWholeRender / kPreview
  uint64_t seed = 0;                 // segment seed
  int32_t noise_w = 0;               // W of the row's target-noise table (target_noise_width(budget))
  uint16_t extra_flags = 0;          // e.g. kRowLookaheadPadZero (live: on), kRowCancelled
  uint64_t philox_offset = 0;        // preview rows: the row's offset in the group's draws
};

struct CallPlan {
  Phase phase = Phase::kSteadyCommit;
  int32_t n_rows = 0;
  int32_t rows_bucket = 0;
  int32_t suffix_bucket = 0;          // tokens (graph key)
  int32_t keys_bucket = 0;            // frames: max fr_keys rounded up to kKeysBucketFrames (graph key: the
                                      // estimator's attention split count); passed to stages in CallDesc::reserved[0]
  int32_t flow_steps = kFlowSteps;
  int32_t total_tokens = 0;
  int32_t total_frames = 0;
  int32_t total_pcm = 0;
  int32_t min_commit_index = 0;       // stream_for()
  std::vector<RowDesc> rows;          // rows_bucket entries; padding rows have slot -1
  // The estimator's copy of `rows`: identical except fr_off, which is 16-aligned with each row owning
  // round_up(fr_suffix, 16) frames (L5 contract: a conv tile never crosses rows). The encoder (L4) needs the
  // contiguous packing fr_off = 2 x tok_off instead (it maps every packed frame to a row, so gaps would be
  // computed as frames of the previous row). est_rows live right after `rows` in the device descriptor arena
  // (CallDesc::rows + rows_bucket); x0 / mu / cond / mel_out of the estimator use est_rows' fr_off.
  std::vector<RowDesc> est_rows;
  int32_t est_total_frames = 0;       // sum of round_up(fr_suffix, 16) over real rows (<= 2 x rows x suffix bucket)
  std::vector<int32_t> tokens;        // packed suffix tokens (real rows only)
  std::vector<int32_t> page_table;    // packed per-row logical -> physical pages
  std::vector<int32_t> new_tok_past;  // per real row: the row's pool position after this call
  std::vector<int32_t> scratch_pages; // pages the call owns (whole renders / previews): release after it
  std::vector<int32_t> prefix_copy_rows;  // real rows whose first private page needs the prefix's partial
                                          // page copied in (kStatePrefixCopy) before this call
};

inline constexpr int32_t kEstRowAlign = 16;  // estimator rows: fr_off multiple of 16 (L5 contract)

enum class BuildStatus : int32_t { kOk = 0, kNoPages = 1, kTooLong = 2, kBadInput = 3 };
const char* build_status_name(BuildStatus s);

inline constexpr int32_t kKeysBucketFrames = 512;
constexpr int32_t keys_bucket_for(int32_t max_fr_keys) {
  return (max_fr_keys + kKeysBucketFrames - 1) / kKeysBucketFrames * kKeysBucketFrames;
}

// The suffix bucket of a call whose longest row suffix is `max_suffix` tokens.
int32_t suffix_bucket_for(Phase call_phase, int32_t max_suffix);

// Build one call. All rows must share `call_phase` (kFirstCommit rows may ride in a kSteadyCommit call).
// On kNoPages nothing is allocated (pages already held by rows stay). Row pages are grown to cover each
// row's keys; RowPages::tok_past is NOT advanced here: apply plan.new_tok_past once the call is queued
// (production advances its state at launch).
BuildStatus build_call(Phase call_phase, const RowRender* rows, int32_t n_rows, HostPagePool& pool, CallPlan& out);

// Several built calls as ONE call (RenderService call merging, execution only): the parts' rows concatenated in
// order, their packed offsets (tokens, frames, estimator frames, PCM, page table) shifted, the buckets recomputed
// from the rows. The result equals build_call() over all the rows at once (gtest MergedPlanEqualsOneBuild); pages
// and row state are not touched again (each part allocated and applied them when it was built). Commit calls only:
// one phase stays that phase; a mix of kFirstCommit / kSteadyCommit / kFinalFlush is a kSteadyCommit call (the graph
// key is canonical and only per-row flags differ); false for whole renders, previews, or more rows than the largest
// rows bucket.
bool merge_plans(const CallPlan* const* parts, int32_t n_parts, CallPlan& out);

// Apply new_tok_past to the rows' pages (after a successful launch).
void apply_call_state(const CallPlan& plan, RowRender* rows, int32_t n_rows);

// The phase a render graph is keyed on. The flow program differs between phases only in its step count (a 1-step
// preview vs 2 steps; program.cu: layouts, op lists and grids depend on rows / suffix / keys buckets only, and every
// per-row phase rule is data in the RowDesc table), so first commits, steady commits, finals and whole renders of the
// same buckets share one graph. Before 15:00 (sprint 4) the call's own phase was part of the key, which captured up
// to four identical graphs per bucket and missed live shapes that another phase had already captured.
constexpr Phase graph_phase(Phase p) { return p == Phase::kPreview ? Phase::kPreview : Phase::kSteadyCommit; }

// A render graph's key (RenderCaller captures one graph per key; `phase` is graph_phase() of the call's phase).
struct GraphKey {
  int32_t role = 0;  // StreamRole value
  int32_t phase = 0;
  int32_t rows_bucket = 0, suffix_bucket = 0, keys_bucket = 0;
  bool operator<(const GraphKey& o) const {
    return std::tie(role, phase, rows_bucket, suffix_bucket, keys_bucket) <
           std::tie(o.role, o.phase, o.rows_bucket, o.suffix_bucket, o.keys_bucket);
  }
  bool operator==(const GraphKey& o) const {
    return std::tie(role, phase, rows_bucket, suffix_bucket, keys_bucket) ==
           std::tie(o.role, o.phase, o.rows_bucket, o.suffix_bucket, o.keys_bucket);
  }
};
// Every graph key calls of a voice's rows can produce up to `row_tokens` tokens per row, for ANY mix of rows in a
// call: a host-only replay through build_call on a bookkeeping page pool (no device memory) of one row on each
// schedule (alone and shared) gives every row shape (suffix, keys, commit index) of first commits, later commits,
// finals at every token count (up to 64 tokens past the state's next boundary: late finals) and 1-row whole renders;
// a call of B >= 2 rows takes every pair of shapes (max suffix, max keys) and the stream of its most urgent row
// (stream_for, as L8's conductor picks it). An unbuilt voice is enumerated with the prefix geometry its build will
// have. Before sprint 4 only rows moving in lockstep on one schedule were replayed, which missed (L8, 13:37) a
// shared-schedule row alone in its call, rows at different positions in one call and late finals.
std::set<GraphKey> enumerate_graph_keys(const VoicePrefix& voice, const CommitPlanConfig& commit, int32_t max_rows,
                                        int32_t row_tokens);

// ---- execution (src/render/render_call.cu; CUDA builds only) ------------------------------------------
// A lane's piece of the render program. The caller captures every stage of a call into one CUDA graph
// per (stream role, phase, rows bucket, suffix bucket). `d_call` is the device CallDesc (fixed address
// per role: the graph reads every per-call value -- rows, tokens, pages, lengths -- from it).
struct RenderStage {
  const char* name = "";
  void* ctx = nullptr;
  // Workspace bytes this stage needs at the bucket (the caller sums them and hands each stage its
  // own aligned piece through CallDesc::workspace + the offset passed to enqueue).
  // `h` carries the bucket (phase, rows_bucket, suffix_bucket; keys bucket in reserved[0]).
  uint64_t (*workspace_bytes)(void* ctx, const CallDesc& h) = nullptr;
  // Enqueue the stage's kernels / library calls on `stream` (being captured).
  void (*enqueue)(void* ctx, const CallDesc& h_call, const CallDesc* d_call, uint64_t workspace_offset,
                  CUstream_st* stream) = nullptr;
};

// An eager step after the call's graph, on the same stream, before the PCM copy (e.g. the HiFT library,
// whose launch shape depends on the rows' lengths). Receives the host plan and the device CallDesc.
struct RenderPostStage {
  const char* name = "";
  void* ctx = nullptr;
  // Runs after the call's flow graph on `stream`. `post_stream` (may equal `stream`): when different, the stage
  // may finish its work there so the role's next flow call can start on `stream` meanwhile (pipelining: HiFT of
  // call k overlaps the flow of call k + 1). Returns the stream on which the call's PCM is complete.
  CUstream_st* (*run)(void* ctx, const CallPlan& plan, const CallDesc& h_call, const CallDesc* d_call, int32_t role,
                      CUstream_st* stream, CUstream_st* post_stream) = nullptr;
};

struct RenderCallerConfig {
  int32_t device = 0;
  int32_t max_rows = 64;                 // largest rows bucket served
  int32_t max_call_tokens = 64 * 320;    // packed suffix tokens per call (desc staging)
  int32_t max_call_pages = 64 * 130;     // page-table entries per call
  int32_t max_call_pcm = 64 * 24000 * 3; // packed PCM samples per call (D2H staging)
  size_t workspace_bytes = 64ull << 20;  // per stream role (urgent, bulk)
  bool use_graphs = true;                // false = run the stages eagerly (debug)
  // Each role gets a second stream (same priority) for the post stage, so the HiFT of call k overlaps the flow
  // of call k + 1 on the same role; the call's PCM copy and completion event follow the post stage. Execution
  // only: PCM bitwise identical. Measured (same-process interleaved A/Bs, shared GPU):
  //   + back-to-back calls on one stream: wall -2.5 to -13 % (B1-B12, 25/50-token commits; tput_ab_s3e_r*.json);
  //   - a first commit on S_urgent while S_bulk runs back-to-back commits: 66-138 ms instead of 18-25 ms
  //     (urgent_ab_s3h_r*.json): two concurrent bulk streams leave the urgent chain of ~700 small kernels
  //     waiting for free SM slots at every kernel.
  // First audio is the gate, so OFF by default; KT2N_RENDER_PIPELINE=1 (Renderer::create) turns it on.
  bool pipeline_post = false;
  // > 0: S_bulk runs in its own SM partition of this many SMs (CUDA green context; the driver rounds to 8-SM
  // steps), S_urgent stays in the primary context (every SM). Keeps SMs free for first commits while bulk renders
  // saturate the rest. 0 = off (EXPERIMENT, sprint 3; KT2N_RENDER_BULK_SMS in Renderer::create).
  int32_t bulk_sms = 0;
  // > 0: a third render role on StreamRole::kSAux, at S_urgent's priority, with its own descriptors, workspace,
  // graphs and HiFT handle, for calls of at most this many rows (EXPERIMENT, sprint 4, L8's proposal 13:55: L8
  // sends a first commit there when S_urgent already runs a call, so two first commits that became ready a turn
  // apart render at the same time instead of one after the other). 0 = off (no memory spent; a kSAux call then
  // runs on S_urgent). KT2N_RENDER_AUX_ROWS in Renderer::create. NEEDS CUDA_DEVICE_MAX_CONNECTIONS=32 in the process
  // environment before CUDA initialises: with the default 8 hardware queues S_aux shared a queue with bulk work and the
  // arm was 2-3x slower under load; with 32 it cut the slowest of 2-3 near-simultaneous first commits under bulk load
  // 9.0 -> 4.3 / 12.6 -> 6.5 ms.
  int32_t aux_max_rows = 0;
  size_t aux_workspace_bytes = 128ull << 20;  // first commits of <= aux_max_rows rows need ~16-60 MB
  // S_aux's CUDA priority, in levels below the top (0 = the top, as S_urgent). L8 arm: a middle level for commits of
  // playing streams whose buffer is low (KT2N_RENDER_AUX_PRIO_LEVEL); clamped to the lowest level.
  int32_t aux_priority_level = 0;
};

// RenderCaller's index of a stream role (RenderTicket::role): S_urgent 0, S_bulk 1, S_aux 2; -1 for others.
inline constexpr int32_t kCallerRoles = 3;
constexpr int32_t caller_role(StreamRole r) {
  return r == StreamRole::kSUrgent ? 0 : r == StreamRole::kSBulk ? 1 : r == StreamRole::kSAux ? 2 : -1;
}

struct RenderTicket {
  int32_t role = -1;
  int32_t call_slot = -1;
  uint64_t serial = 0;
};

struct RenderTiming {
  double host_launch_us = 0;  // build is the caller's; this is staging + H2D + graph launch + D2H + event
  float gpu_ms = -1;          // event pair around the call (when timing is on)
  bool graph_captured = false;
  int32_t graph_nodes = 0;    // nodes (kernels + memsets) of the call's graph
};

class RenderCaller {
 public:
  RenderCaller(const RenderCallerConfig& cfg, const PoolView& pool, const StateView& state,
               const VoiceView& voices, std::vector<RenderStage> program);
  ~RenderCaller();
  RenderCaller(const RenderCaller&) = delete;
  RenderCaller& operator=(const RenderCaller&) = delete;

  // Queue a call. Returns false (nothing queued) when the role already has kMaxQueuedBatchesPerStream
  // calls in flight: the dispatcher retries on its next turn (the driver queue never fills).
  // `waits`: events (other streams' earlier calls of the same rows, admission uploads) the call must
  // follow; each costs one cudaStreamWaitEvent, no host wait.
  // `internal_slot`: use the role's extra slot reserved for internal work (voice-prefix builds) instead of
  // the conductor's two.
  bool launch(const CallPlan& plan, StreamRole role, RenderTicket* ticket, RenderTiming* timing = nullptr,
              CUevent_st* const* waits = nullptr, int32_t n_waits = 0, bool internal_slot = false);
  // True when the role can take another call now (a call slot is free; `internal_slot`: the extra one).
  bool can_launch(StreamRole role, bool internal_slot = false) const;
  // True when calls of one role can complete out of their stream's order of work (pipeline_post): a row's next
  // call must then wait on its previous call's event even on the same role.
  bool pipelined() const;
  // Switch pipeline_post at run time (waits for both roles' streams; for same-process A/B, call when idle).
  void set_pipeline_post(bool on);
  // bulk_sms: the role's green context (nullptr = primary context) and the bulk partition's SM count (0 = off).
  // Work a caller issues itself on a role's stream must run with that context current.
  CUctx_st* role_context(StreamRole role) const;
  int32_t bulk_partition_sms() const;
  // The event recorded after the ticket's call (valid until the ticket is released).
  CUevent_st* done_event(const RenderTicket& t) const;
  // Non-blocking completion check (cudaEventQuery).
  bool done(const RenderTicket& t) const;
  // Blocking wait (tests and the CLI only; serving threads poll).
  void wait(const RenderTicket& t) const;
  // Pinned host PCM of a finished call (plan.total_pcm floats, rows at RowDesc::pcm_off).
  const float* pcm(const RenderTicket& t) const;
  float gpu_ms(const RenderTicket& t) const;
  // Free the call slot (after the PCM has been consumed).
  void release(const RenderTicket& t);

  CUstream_st* stream(StreamRole role) const;
  // The role's CallDesc::mel_out buffer (checks / oracle comparison; valid after a call completes).
  const float* device_mel_out(StreamRole role) const;
  size_t graphs_cached() const;
  // Capture (record + instantiate, nothing runs) the graph of one bucket ahead of its first call, so no live
  // call pays the 3-6 ms host capture. True when a graph was captured (false: cached already / graphs off).
  bool prewarm(StreamRole role, Phase phase, int32_t rows_bucket, int32_t suffix_bucket, int32_t keys_bucket);
  // Workspace bytes the program needs for one bucket (a call needing more than workspace_bytes() fails cleanly).
  uint64_t workspace_need(Phase phase, int32_t rows_bucket, int32_t suffix_bucket, int32_t keys_bucket) const;
  uint64_t workspace_bytes() const;
  // The role's workspace (S_aux has its own, smaller one) and the aux role's rows cap (0 = no aux role).
  uint64_t workspace_bytes(StreamRole role) const;
  int32_t aux_max_rows() const;
  int32_t max_rows() const;  // RenderCallerConfig::max_rows
  // The plan fits the role's staging (rows, tokens, page-table entries, PCM samples): launch() would not refuse it.
  bool fits_staging(const CallPlan& plan, StreamRole role) const;
  // A graph for this key is captured already (a live call with it pays no host capture).
  bool has_graph(StreamRole role, Phase phase, int32_t rows_bucket, int32_t suffix_bucket, int32_t keys_bucket) const;
  void set_timing(bool on) { timing_ = on; }
  void set_post_stage(const RenderPostStage& post);

 private:
  struct Impl;
  Impl* impl_;
  bool timing_ = false;
};

}  // namespace kt2n::render
