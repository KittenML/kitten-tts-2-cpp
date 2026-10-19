// L7 RenderService: the renderer the conductor (L8) drives. Same calls as L8's
// kt2n::engine::RenderBackend (src/engine/backend.h on lab/l8-conductor-20261010):
//   add_voice  -> voice tables (prompt tokens, prompt mel, speaker conditioning, numpy prompt noise)
//   open_row   -> render-state slot + pool pages on the voice prefix + the row's numpy noise table
//                 (drawn on the host, uploaded on the urgent stream; later calls wait on that upload)
//   launch     -> geometry from the conductor's decision (commit boundary / final / whole render),
//                 ragged descriptors, pages, then RenderCaller (one captured graph per bucket); never
//                 waits for the GPU: calls beyond the per-stream bound are queued on the host
//   collect    -> finished calls' PCM per row (float): every finished call (not only the oldest ones), in launch
//                 order per row; calls of different rows may come back out of launch order
// Cross-stream order: a row's call on another stream than its previous call waits on that call's
// event (cudaStreamWaitEvent), so commit k (urgent) always runs before commit k + 1 (bulk).
// CUDA builds only (src/render/service.cu).
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kt2n/render/arena.h"
#include "kt2n/render/call.h"
#include "kt2n/render/commit_plan.h"
#include "kt2n/render/state.h"

namespace kt2n::render {

struct ServiceRowReq {        // == engine::RenderRowReq
  uint64_t row = 0;
  int32_t voice = 0;
  Phase phase = Phase::kSteadyCommit;
  int32_t commit_index = 0;
  int32_t committed_tokens = 0;   // tokens published before this call
  int32_t commit_tokens = 0;      // the commit boundary after this call; finals: all tokens (+3 silence)
  bool finalize = false;
  std::vector<int32_t> tokens;    // commit: ids[:next + 3]; final / whole: ids + [4299] x 3; preview: ids[:8]
  uint64_t seq = 0;
};
struct ServiceCall {          // == engine::RenderCall
  uint64_t call_id = 0;
  StreamRole stream = StreamRole::kSUrgent;
  std::vector<ServiceRowReq> rows;
};
struct ServiceOut {           // == engine::RenderOut
  uint64_t call_id = 0;
  int32_t status = 0;
  std::vector<std::vector<float>> pcm;  // per row of the call
  double gpu_ms = 0;
  double launch_ms = 0;
  int32_t graph_nodes = 0;  // nodes of the call's captured graph (kernel count of the flow part)
  bool graph_captured = false;
};

struct RenderServiceConfig {
  ArenaConfig arena;
  RenderCallerConfig caller;
  CommitPlanConfig commit;
  bool timing = false;  // per-call GPU timing events
  int32_t noise_workers = 2;  // host threads drawing the rows' numpy noise (0 = on the caller's thread)
  // build_voice_prefix returns once the build is queued on S_urgent (the GPU orders every reader after it);
  // false = also wait for it on the host (the old behaviour, for A/B).
  bool async_prefix_build = true;
  // With the aux role on (RenderCallerConfig::aux_max_rows > 0): a first-commit call the conductor sends to S_urgent
  // while S_urgent already has a call queued or running goes to S_aux instead, so the arm needs no conductor change
  // (an explicit kSAux call is honoured either way). Off: only explicit kSAux calls use the role.
  bool aux_auto = true;
  // Call merging (EXPERIMENT, sprint 5, execution only): conductor calls that wait on the host for the same role
  // (its 2 call slots are busy: 1 running + 1 queued on the stream) are launched as ONE call of all their rows when a
  // slot frees, instead of one after the other. What the conductor decided is unchanged: the same rows, geometry,
  // tokens and order per row; each conductor call still gets its own ServiceOut. Only a contiguous run of waiting calls
  // of the role is merged (no call overtakes an earlier one), never two calls of one row, never whole renders or
  // previews, only up to `merge_*_rows` rows, only when the merged call fits the workspace and its graph is captured
  // already (no live capture). Why: at N=20 the conductor makes ~1.7 rows per call, S_bulk is busy ~all the time with
  // B1 calls (L8's 15:29 window: 1,100 of 1,487 bulk calls were B1, ~11 ms each under load; a B3 took ~15.5 ms), and
  // the GPU time those calls take is what slows the LM step 1.9 -> 3.1 ms. 0 = off. KT2N_RENDER_MERGE_BULK /
  // KT2N_RENDER_MERGE_URGENT in Renderer::create.
  int32_t merge_bulk_rows = 0;
  int32_t merge_urgent_rows = 0;
  // With merging on: at most this many unfinished calls on the role's stream before later calls wait on the host
  // (where they can merge). 2 = the call slots' own limit (no added wait); 1 = hold the next call until the running
  // one finished and a collect()/launch() noticed (more merging, a short gap on the stream). KT2N_RENDER_MERGE_DEPTH.
  int32_t merge_depth = 2;
  // L8 arm KT2N_HOLD_URGENT (execution only, off by default): the conductor sends a steady commit of a request whose
  // startup hold has not released to S_urgent, so prewarm also captures the S_urgent twin of every single-row S_bulk
  // commit graph up to `hold_urgent_twin_max_suffix` tokens (commits 2-4 of a row: 33 / 58 / 108 tokens with the
  // look-ahead). Without the twins such a call would capture its graph live on the conductor thread (2-6 ms).
  bool hold_urgent_twins = false;
  int32_t hold_urgent_twin_max_suffix = 128;
  // L8 arm KT2N_BUFFER_ROLE=aux (with KT2N_BUFFER_URGENT_MS): commits of playing streams with a low buffer go to S_aux,
  // which then runs at a middle priority (RenderCallerConfig::aux_priority_level). Prewarm S_aux twins of the
  // single-row S_bulk commit graphs (same suffix cap); an explicit S_aux steady call that does not fit S_aux runs on
  // S_bulk (its natural stream), never on S_urgent.
  bool aux_steady_twins = false;
};

class RenderService {
 public:
  RenderService(const RenderServiceConfig& cfg, std::vector<RenderStage> program);
  ~RenderService();
  RenderService(const RenderService&) = delete;
  RenderService& operator=(const RenderService&) = delete;

  // prompt_feat: [frames][80] fp32 (frames = 2 x tokens); spk_cond: [80] (spk_embed_affine_layer output;
  // may be empty until the encoder lane's spk_affine runs). Returns the voice index or -1.
  int32_t add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                    const std::vector<float>& spk_cond, std::string* err);
  // Solve the voice's prompt prefix once (chunk_cache_20260927.py _build_prefix): a kSteadyCommit row from
  // position 0 over prompt[:stable + 3] with no output; its complete pages become the voice's shared,
  // read-only prefix pages and its conv tails the voice's prefix tails. Queued on S_urgent and returns without
  // waiting for the GPU (RenderServiceConfig::async_prefix_build); calls already in flight are untouched and
  // their outputs still come back from collect(). Rows opened afterwards start at `stable` (225 for a 250-token
  // prompt) instead of 0, ordered after the build on the GPU.
  int32_t build_voice_prefix(int32_t voice, std::string* err);
  // Release a voice (L8, 13:13: voices were append-only, so a long-running server refused its 33rd distinct custom
  // voice): its prefix pages go back to the pool and its table index is free for the next add_voice. Only when
  // nothing uses it any more: returns -2 (nothing changed; retry after the voice's rows are closed and their calls
  // collected) while an open or closing row, or the voice's own prefix build, still reads it; -1 for a bad index.
  int32_t remove_voice(int32_t voice, std::string* err);
  int32_t voices_in_use() const;
  int32_t open_row(uint64_t row, int32_t voice, uint64_t seed, int32_t budget, bool demo, std::string* err);
  // Exact pool pages a row of `tokens` LM ids on `voice` can allocate (state.h row_pages_needed), for admission
  // reservation: reserve row_pages_needed(budget) at open_row, and the difference to 2 x budget + 3 when the
  // noise extension starts. Call after the voice prefix is built (an unbuilt prefix shares nothing).
  int32_t pages_needed(int32_t voice, int32_t tokens, bool demo) const;
  // Capture ahead of time every flow graph the voice's rows hit up to `row_tokens` tokens (alone and shared
  // schedules at each rows bucket <= max_rows: commits, finals, whole renders), so no live call pays a host
  // capture (3-6 ms on the conductor thread). Recording only; nothing runs. Call at startup after
  // build_voice_prefix; voices with the same prompt geometry share every key (the later ones capture ~0).
  // Returns the number of graphs captured, -1 on error.
  // Keys whose workspace need exceeds the role's workspace are skipped and counted (`skipped`), never an error: such a
  // call would fail live with a clean error too (a lab-sized workspace; L8 15:5x: 403 MB keys vs a 384 MB knob).
  struct PrewarmStats {
    int32_t keys = 0, captured = 0;
    double ms = 0;
    int32_t skipped = 0;
  };
  int32_t prewarm_voice(int32_t voice, int32_t max_rows, int32_t row_tokens, PrewarmStats* stats, std::string* err);
  // The 5,000,000 + seed extension (the conductor calls it when the LM's new tokens reach the budget).
  int32_t extend_noise(uint64_t row, std::string* err);
  void close_row(uint64_t row);
  // Never throws: a call refused before it is queued returns -1 (+ err); a call that fails later (e.g. a
  // capacity limit met while launching it) comes back from collect() with status -1 and empty PCM.
  int32_t launch(ServiceCall&& call, std::string* err);
  int64_t failed_calls() const;
  // Calls the aux role ran (ServiceCall::stream == kSAux and the call fit it; others ran on S_urgent).
  int64_t aux_calls() const;
  int64_t aux_fallback_bulk() const;
  void set_aux_auto(bool on);  // RenderServiceConfig::aux_auto at run time (A/B benches)
  // RenderServiceConfig::merge_* at run time (A/B benches; takes effect for calls launched to the GPU from now on).
  void set_merge(int32_t bulk_rows, int32_t urgent_rows, int32_t depth);
  // Merged GPU calls and the conductor calls they carried (a merge of 3 calls counts 1 and 3).
  int64_t merges() const;
  int64_t merged_calls() const;
  // Whole-render calls rendered as several calls because one call of all their rows would not fit the workspace.
  int64_t split_calls() const;
  void collect(std::vector<ServiceOut>* out, bool wait);
  int32_t inflight() const;

  RenderArena& arena();
  RenderCaller& caller();
  HostPagePool& pool();

 private:
  int32_t launch_impl(ServiceCall&& call, std::string* err);
  void collect_impl(std::vector<ServiceOut>* out, bool wait);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kt2n::render
