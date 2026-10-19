// L8: the two execution backends the conductor drives. The conductor makes production's
// decisions (policy.h); a backend only executes them.
//   LmBackend      the torch-free LM (L3 kt2n::lm::Engine), or a CPU stub for tests.
//   RenderBackend  the token -> PCM renderer (L7 render state + L4/L5/L6 op library through
//                  include/kt2n/render/ops.h), or a CPU stub until L7 lands.
// Both are called only from the conductor thread (except stats()).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kt2n/lm.h"
#include "kt2n/render/ops.h"

namespace kt2n::engine {

// ------------------------------------------------------------------------------------- LM
class LmBackend {
 public:
  virtual ~LmBackend() = default;
  // Per-voice LM prefix (voices.kt2n lm_prefix + projected speaker). Returns the LM voice id.
  virtual int add_voice(const std::vector<float>& projected_speaker, const std::vector<int32_t>& lm_prefix,
                        std::string* err) = 0;
  virtual void drop_voice(int lm_voice) = 0;
  // Gives the voice's store space back (L3 release_voice): no new rows; rows still reading it keep it until their
  // slots are released. Conductor thread. 0, or -1 for an unknown / already released id.
  virtual int release_voice(int lm_voice) = 0;
  virtual int submit(const lm::RowRequest& row, std::string* err) = 0;
  virtual void cancel(uint64_t tag) = 0;
  // Appends the tokens of every completed step. block: wait for at least one event when rows are live.
  virtual int pump(std::vector<lm::TokenEvent>* events, bool block, std::string* err) = 0;
  // Execution-only lab arm. The L3 driver orders a priority switch after earlier decode work.
  virtual int set_decode_priority(int level) {
    (void)level;
    return -1;
  }
  virtual int live_rows() const = 0;
  virtual uint64_t steps() const = 0;
  virtual const char* name() const = 0;
};

// The real LM (L3). Only with CUDA.
std::unique_ptr<LmBackend> make_l3_lm(const lm::Config& cfg, std::string* err);

// CPU stub: deterministic audio tokens from the seed (a short opening of sub-gate ids, then
// "speech" ids), EOS after ~3.4 tokens per suffix id or at the budget; step_us of simulated
// decode time per step. For runtime tests and plumbing only; never in the RC.
struct StubLmConfig {
  int step_us = 500;
  double tokens_per_suffix_id = 3.4;
  int opening_quiet_tokens = 3;
};
std::unique_ptr<LmBackend> make_stub_lm(const StubLmConfig& cfg);

// ------------------------------------------------------------------------------------- render
struct RenderRowReq {
  uint64_t row = 0;           // conductor row key (owns a render-state slot from open_row to close_row)
  int voice = 0;              // engine voice index
  render::Phase phase = render::Phase::kSteadyCommit;
  int commit_index = 0;
  int committed_tokens = 0;   // tokens published before this call
  int commit_tokens = 0;      // the commit boundary after this call (next_commit_tokens); finals: all tokens
  bool finalize = false;
  std::vector<int32_t> tokens;  // commit: ids[:next+3]; final: ids + [4299] x 3; preview: ids[:8]
  uint64_t seq = 0;           // per-row publication order
};
struct RenderCall {
  uint64_t call_id = 0;
  render::StreamRole stream = render::StreamRole::kSUrgent;
  bool async = true;          // production's launch class (BS rule); native executes both overlapped
  std::vector<RenderRowReq> rows;
};
struct RenderOut {
  uint64_t call_id = 0;
  int status = 0;                       // 0 ok, else kt2n_status
  std::vector<std::vector<float>> pcm;  // per row of the call, the samples it publishes (float, [-1, 1])
  double gpu_ms = 0;
  double launch_ms = 0;
  std::string error;  // a failed call's reason (internal; logs / lab)
  int graph_nodes = 0;          // L7: nodes of the call's captured graph (flow kernels)
  bool graph_captured = false;  // L7: this call captured its bucket's graph (capture time is in its wall)
};

// open_row: the renderer cannot take the row yet (no render-state slot, or its pool pages cannot be reserved).
// The conductor puts the job back and retries on a later turn (admission back-pressure, never a failure).
inline constexpr int kRenderBackpressure = 1;

class RenderBackend {
 public:
  virtual ~RenderBackend() = default;
  virtual int add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                        const std::vector<float>& embedding, std::string* err) = 0;
  // Admission: a render-state slot + the row's noise tables (seed, budget). 0 ok, kRenderBackpressure = retry
  // later, < 0 = failure.
  virtual int open_row(uint64_t row, int voice, uint64_t seed, int budget, bool demo, std::string* err) = 0;
  virtual void close_row(uint64_t row) = 0;
  // Give an evicted voice's table row and prefix pages back (L7 RenderService::remove_voice). 0 ok, -2 busy (rows or
  // a prefix build of it still in flight: call again later), -1 failure / not supported (never retried).
  virtual int remove_voice(int voice, std::string* err) {
    (void)voice;
    if (err) *err = "remove_voice: not supported";
    return -1;
  }
  // The 5,000,000 + seed noise extension (runtime.py:15421): called once when a row's LM steps reach its
  // budget without finishing, so frames past the budget get their own stream (L7 RenderService::extend_noise).
  virtual int extend_noise(uint64_t row, std::string* err) {
    (void)row;
    (void)err;
    return 0;
  }
  // Never waits for the GPU (bounded queue: at most kMaxQueuedBatchesPerStream per stream).
  virtual int launch(RenderCall&& call, std::string* err) = 0;
  // Finished calls. wait: block until at least one finishes (or none is in flight).
  virtual void collect(std::vector<RenderOut>* out, bool wait) = 0;
  virtual int inflight() const = 0;
  virtual const char* name() const = 0;
  // Backend facts for kt2n_stats_json (a JSON object; "{}" when none).
  virtual std::string stats_json() const { return "{}"; }
};

// CPU stub: a quiet token-keyed tone with production's sample accounting (960 samples per
// committed token; finals publish the rest); simulated GPU time a_us + b_us x tokens per call on
// one worker per stream role. Plumbing and timing-port tests only: NOT the model's audio.
struct StubRenderConfig {
  int call_us = 2000;
  double token_us = 5.0;
  float amplitude = 0.25f;
  int fail_every = 0;  // tests: every n-th call returns a failure (0 = never)
  int max_open_rows = 0;  // tests: open_row back-pressures beyond this many open rows (0 = unlimited)
};
std::unique_ptr<RenderBackend> make_stub_render(const StubRenderConfig& cfg);

// L7's RenderService with L7's stand-in stages (noise gather + stub PCM): the real render state and call
// path with placeholder audio (lab arm "l7-standin"; built only with L7's CUDA target).
std::unique_ptr<RenderBackend> make_l7_standin_render(int device, int slots, int pool_pages, int voices,
                                                      std::string* err);

// The real renderer (L7 Renderer over s3gen.kt2n: L4 encoder + L5 estimator + L6 HiFT). nullptr + err when it
// cannot be built (or the library was built without L7's flow target).
struct L7RenderOptions {
  std::string s3gen_path;
  int device = 0;
  int slots = 16;             // render-state slots (rows open at once)
  int pool_pages = 256;       // 64-frame K/V pages, 16.5 MB each
  int voices = 64;            // voice table rows (catalogue + loaded / built voices)
  int max_rows = 16;          // largest rows bucket a call may use
  int workspace_mb = 1024;    // per stream role (urgent, bulk)
  int hift_max_frame_rows = 0;  // 0 = L7's default (2048; ~600 KB of HiFT workspace per frame-row)
  int hift_bulk_max_t = 0;           // S_bulk HiFT handle (finals + whole renders); 0 = L7's default (4096 frames,
  int hift_bulk_max_frame_rows = 0;  // ~3.1 GB, one 2,045-token stream:false row per launch). Lab: lower it.
  // Graph prewarm (L7 RenderService::prewarm_voice): capture every flow graph a voice's rows can hit, up to
  // prewarm_rows rows per call and prewarm_tokens tokens per row, at create, so no live call captures a graph on
  // the conductor thread (2.7-5.8 ms each). With lazy prefixes only the first voice is built + prewarmed at
  // create (voices of the same prompt geometry share the keys); with all prefixes every voice is.
  bool prewarm = true;
  int prewarm_rows = 0;      // 0 = min(max_rows, 20)
  int prewarm_tokens = 320;
  bool lazy_prefix = true;    // build a voice's prefix on its first use (production) instead of at create
  bool reserve_pages = true;  // admission reserves a row's pool pages (back-pressure instead of no_pages)
  bool timing = true;         // per-call GPU timing events (s3gen_gpu_ms)
};
std::unique_ptr<RenderBackend> make_l7_render(const L7RenderOptions& opts, std::string* err);

}  // namespace kt2n::engine
