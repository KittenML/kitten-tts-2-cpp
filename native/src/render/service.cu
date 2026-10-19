// RenderService (include/kt2n/render/service.h).
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <thread>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "kt2n/render/rng.h"
#include "kt2n/render/service.h"

namespace kt2n::render {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

int role_of(StreamRole r) { return caller_role(r); }  // RenderTicket::role

}  // namespace

struct RenderService::Impl {
  RenderServiceConfig cfg;
  std::unique_ptr<RenderArena> arena;
  std::unique_ptr<HostPagePool> pool;
  std::unique_ptr<SlotPool> slots;
  std::unique_ptr<RenderCaller> caller;
  std::vector<VoicePrefix> voices;
  float* pinned_noise = nullptr;           // [slots][80][noise_cap]
  std::vector<cudaEvent_t> admit_events;   // per slot: noise upload / prefix copy done
  std::vector<cudaEvent_t> open_events;    // per slot: open_row's work on S_urgent (noise clear, prefix tails) done
  int32_t build_slot = -1;                 // reserved slot for voice-prefix builds
  cudaStream_t aux = nullptr;              // voice-table uploads (non-blocking: never waits for render / LM work)
  // Outputs of calls that finished while build_voice_prefix waited for a call slot: returned by the next
  // collect(), ahead of anything newer (they are older), so no call's audio is ever lost.
  std::vector<ServiceOut> stash;
  // Noise draws off the conductor thread (production: G_ASYNC_NOISE=1, a 2-worker pool). The upload is
  // issued at the row's first launch, after the draw (usually long done: the first commit needs >= 7
  // LM steps after prefill, the draw ~0.5 ms).
  std::vector<std::thread> workers;
  std::mutex qmu;
  std::condition_variable qcv;
  std::deque<std::function<void()>> tasks;
  bool stop = false;
  void submit(std::function<void()> f) {
    {
      std::lock_guard<std::mutex> g(qmu);
      tasks.push_back(std::move(f));
    }
    qcv.notify_one();
  }
  void worker_loop() {
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock<std::mutex> l(qmu);
        qcv.wait(l, [&] { return stop || !tasks.empty(); });
        if (stop && tasks.empty()) return;
        f = std::move(tasks.front());
        tasks.pop_front();
      }
      f();
    }
  }

  struct Row {
    int32_t slot = -1;
    int32_t voice = -1;
    uint64_t seed = 0;
    int32_t budget = 0;
    int32_t noise_w = 0;
    bool demo = false;
    bool extended = false;
    bool admit_fresh = true;  // the next call on a non-urgent stream must wait for the admission event
    bool closing = false;
    RowPages pages;
    RowCommitState commit;
    RenderTicket last{};
    bool has_last = false;
    std::shared_future<void> noise_ready;  // the host draw into the slot's pinned buffer
    bool noise_uploaded = false;
    int32_t queued = 0;       // calls built but not launched yet
    explicit Row(const CommitPlanConfig* c) : commit(c) {}
  };
  std::unordered_map<uint64_t, std::unique_ptr<Row>> rows;

  struct Pending {
    uint64_t call_id = 0;
    StreamRole role = StreamRole::kSUrgent;
    CallPlan plan;
    std::vector<uint64_t> row_keys;
    double build_ms = 0;
    // A part of a whole-render call split to fit the workspace: its rows' indices in the conductor's call.
    int32_t n_parts = 1;
    std::vector<int32_t> parent_rows;
    // A merged call (RenderServiceConfig::merge_*): the conductor calls it carries, in row order, and their rows.
    std::vector<uint64_t> part_ids;
    std::vector<int32_t> part_rows;
  };
  // Split whole-render calls: the parts' outputs are joined into the conductor's one ServiceOut per call.
  struct SplitOut {
    int32_t parts_left = 0;
    ServiceOut out;
  };
  std::unordered_map<uint64_t, SplitOut> splits;
  int64_t split_calls = 0;
  std::deque<Pending> pending;  // arrival order
  struct Inflight {
    uint64_t call_id = 0;
    RenderTicket ticket{};
    std::vector<int32_t> pcm_off, pcm_count;
    std::vector<int32_t> scratch_pages;
    double launch_ms = 0;
    int32_t graph_nodes = 0;
    bool graph_captured = false;
    bool failed = false;  // the launch threw (e.g. a workspace / HiFT capacity limit): no GPU work, status -1
    bool internal = false;  // a voice-prefix build: no output; its scratch pages are released when it is done
    int32_t voice = -1;     // internal: the voice whose prefix it builds (remove_voice waits for it)
    std::vector<uint64_t> row_keys;  // rows of the call (per-row output order in collect)
    std::string error;
    int32_t n_parts = 1;              // > 1: one part of a split whole-render call (Pending::parent_rows)
    std::vector<int32_t> parent_rows;
    std::vector<uint64_t> part_ids;   // a merged call: one ServiceOut per conductor call (Pending::part_ids)
    std::vector<int32_t> part_rows;
  };
  std::deque<Inflight> inflight;
  // prewarm_voice: graph keys per (prompt tokens, prefix stable tokens or -1, rows cap, row tokens).
  std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, std::set<GraphKey>> key_sets;
  int64_t failed_calls = 0;
  int64_t aux_calls = 0;  // calls that ran on the aux role
  int64_t aux_fallback_bulk = 0;  // explicit S_aux steady commits that did not fit S_aux and ran on S_bulk
  int64_t merges = 0, merged_calls = 0;  // merged GPU calls / the conductor calls they carried
  std::string last_error;

  // S_urgent has a conductor call or an internal build queued or still running (auto aux dispatch).
  bool urgent_busy() const {
    for (const Pending& q : pending)
      if (q.role == StreamRole::kSUrgent) return true;
    for (const Inflight& f : inflight)
      if (!f.failed && f.ticket.role == caller_role(StreamRole::kSUrgent) && !caller->done(f.ticket)) return true;
    return false;
  }

  Row* find(uint64_t key) {
    auto it = rows.find(key);
    return it == rows.end() ? nullptr : it->second.get();
  }

  // The row's noise upload, issued (once) before its first call, on S_urgent, or on S_aux when the row's first call
  // runs there: on S_urgent it would queue behind the urgent render S_aux is meant to overlap, and the aux call
  // would wait for it through the admission event, so the aux arm serialised anyway (lane log, sprint 4 15:04).
  void ensure_noise(Row& r, StreamRole role = StreamRole::kSUrgent) {
    if (r.noise_uploaded) return;
    if (r.noise_ready.valid()) r.noise_ready.get();  // rethrows a failed draw
    cudaStream_t s = caller->stream(role);
    if (role != StreamRole::kSUrgent) check(cudaStreamWaitEvent(s, open_events[size_t(r.slot)], 0), "open wait");
    float* host = pinned_noise + size_t(r.slot) * kMel * size_t(cfg.arena.noise_cap);
    float* dst = arena->state().target_noise + size_t(r.slot) * kMel * size_t(cfg.arena.noise_cap);
    check(cudaMemcpy2DAsync(dst, size_t(cfg.arena.noise_cap) * sizeof(float), host, size_t(cfg.arena.noise_cap) * sizeof(float),
                            size_t(r.noise_w) * sizeof(float), kMel, cudaMemcpyHostToDevice, s),
          "noise upload");
    check(cudaEventRecord(admit_events[size_t(r.slot)], s), "admit record");
    r.noise_uploaded = true;
    r.admit_fresh = true;
  }

  static bool shares(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    for (uint64_t x : a)
      for (uint64_t y : b)
        if (x == y) return true;
    return false;
  }
  int32_t merge_cap(StreamRole role) const {
    const int32_t c = role == StreamRole::kSBulk ? cfg.merge_bulk_rows
                      : role == StreamRole::kSUrgent ? cfg.merge_urgent_rows
                                                     : 0;
    return std::min(c, caller->max_rows());
  }
  // Calls of the role still running or queued on its stream.
  int32_t unfinished_on(StreamRole role) const {
    int32_t k = 0;
    for (const Inflight& f : inflight)
      if (!f.failed && f.ticket.role == caller_role(role) && !caller->done(f.ticket)) ++k;
    return k;
  }
  static bool mergeable(const Pending& p) {
    return p.n_parts == 1 && p.part_ids.empty() && p.plan.scratch_pages.empty() && p.plan.phase != Phase::kWholeRender &&
           p.plan.phase != Phase::kPreview;
  }

  // Call merging (RenderServiceConfig::merge_*): launch pending[i] together with the contiguous run of later pending
  // calls of its role that can join it (see the config comment). False when nothing joins (launch it alone).
  bool try_merge(size_t i, int32_t cap) {
    const Pending& p = pending[i];
    std::vector<size_t> take{i};
    std::vector<const CallPlan*> plans{&p.plan};
    std::vector<uint64_t> keys = p.row_keys;  // rows of the merged call
    // Rows of calls that must launch first: every call waiting AHEAD of pending[i] (any role; e.g. a row's commit 1
    // still waiting for S_urgent while its commit 2 waits for S_bulk) and other roles' calls between the merged ones.
    // A later call of one of these rows must not overtake them (stress test, sprint 5: without the calls ahead in this
    // set, a row's bulk commit ran before its urgent commit and 6 of 8 rows rendered wrong audio).
    std::vector<uint64_t> between;
    for (size_t j = 0; j < i; ++j) between.insert(between.end(), pending[j].row_keys.begin(), pending[j].row_keys.end());
    int32_t rows_n = p.plan.n_rows;
    CallPlan merged, trial;
    bool have = false;
    for (size_t j = i + 1; j < pending.size(); ++j) {
      const Pending& q = pending[j];
      if (q.role != p.role) {  // stays in its place; a later call of one of its rows must not overtake it
        between.insert(between.end(), q.row_keys.begin(), q.row_keys.end());
        continue;
      }
      if (!mergeable(q) || rows_n + q.plan.n_rows > cap || shares(q.row_keys, keys) || shares(q.row_keys, between)) break;
      plans.push_back(&q.plan);
      if (!merge_plans(plans.data(), int32_t(plans.size()), trial) || !caller->fits_staging(trial, p.role) ||
          caller->workspace_need(trial.phase, trial.rows_bucket, trial.suffix_bucket, trial.keys_bucket) >
              caller->workspace_bytes(p.role) ||
          !caller->has_graph(p.role, trial.phase, trial.rows_bucket, trial.suffix_bucket, trial.keys_bucket))
        break;
      merged = std::move(trial);
      have = true;
      take.push_back(j);
      keys.insert(keys.end(), q.row_keys.begin(), q.row_keys.end());
      rows_n += q.plan.n_rows;
    }
    if (!have) return false;
    Pending m;
    m.call_id = p.call_id;
    m.role = p.role;
    m.plan = std::move(merged);
    m.row_keys = std::move(keys);
    for (size_t t : take) {
      m.part_ids.push_back(pending[t].call_id);
      m.part_rows.push_back(pending[t].plan.n_rows);
      m.build_ms += pending[t].build_ms;
    }
    ++merges;
    merged_calls += int64_t(take.size());
    launch_now(m);
    for (size_t k = take.size(); k-- > 0;) pending.erase(pending.begin() + std::ptrdiff_t(take[k]));
    return true;
  }

  // Launch every pending call that may go now: its role has a free call slot, no earlier pending call
  // of the same role, and no earlier pending call that shares a row. With merging on for the role, waiting calls
  // that can join it go in the same GPU call (try_merge).
  void pump() {
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t i = 0; i < pending.size(); ++i) {
        Pending& p = pending[i];
        bool blocked = !caller->can_launch(p.role);
        for (size_t j = 0; j < i && !blocked; ++j)
          if (pending[j].role == p.role || shares(pending[j].row_keys, p.row_keys)) blocked = true;
        const int32_t cap = merge_cap(p.role);
        if (!blocked && cap > 1 && cfg.merge_depth < kMaxQueuedBatchesPerStream &&
            unfinished_on(p.role) >= std::max(1, cfg.merge_depth))
          blocked = true;  // merge_depth 1: the next call waits on the host until the role's stream is idle
        if (blocked) continue;
        if (!(cap > 1 && mergeable(p) && try_merge(i, cap))) {
          launch_now(p);
          pending.erase(pending.begin() + std::ptrdiff_t(i));
        }
        progress = true;
        break;
      }
    }
  }

  void launch_now(Pending& p) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<CUevent_st*> waits;
    for (uint64_t key : p.row_keys) {
      Row* r = find(key);
      if (r == nullptr) continue;
      if (r->admit_fresh && p.role != StreamRole::kSUrgent) waits.push_back(admit_events[size_t(r->slot)]);
      // Another stream's earlier call of this row; with the post stage pipelined also the same role's (its HiFT
      // may still run on the post stream and the next HiFT reads the row's mel cache it writes).
      if (r->has_last && (r->last.role != role_of(p.role) || caller->pipelined()) && !caller->done(r->last))
        waits.push_back(caller->done_event(r->last));
    }
    RenderTicket t;
    RenderTiming tm;
    Inflight f;
    f.call_id = p.call_id;
    try {
      if (!caller->launch(p.plan, p.role, &t, &tm, waits.data(), int32_t(waits.size())))
        throw std::logic_error("RenderService: launch refused after can_launch");
    } catch (const std::exception& e) {
      // Never throw into the conductor: the call completes as failed (status -1, empty PCM) through collect(),
      // in its rows' order, and the request fails cleanly (L8: speech_generation_failed).
      f.failed = true;
      f.error = e.what();
      ++failed_calls;
      last_error = f.error;
      std::fprintf(stderr, "[l7 render] call %llu failed at launch: %s\n", (unsigned long long)p.call_id, e.what());
    }
    f.ticket = t;
    for (int32_t i = 0; i < p.plan.n_rows; ++i) {
      f.pcm_off.push_back(p.plan.rows[size_t(i)].pcm_off);
      f.pcm_count.push_back(p.plan.rows[size_t(i)].pcm_count);
    }
    f.scratch_pages = p.plan.scratch_pages;
    f.row_keys = p.row_keys;
    f.n_parts = p.n_parts;
    f.parent_rows = p.parent_rows;
    f.part_ids = p.part_ids;
    f.part_rows = p.part_rows;
    f.launch_ms = p.build_ms + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    f.graph_nodes = tm.graph_nodes;
    f.graph_captured = tm.graph_captured;
    for (uint64_t key : p.row_keys) {
      Row* r = find(key);
      if (r == nullptr) continue;
      if (!f.failed) {
        r->last = t;
        r->has_last = true;
        r->admit_fresh = false;
      }
      --r->queued;
    }
    inflight.push_back(std::move(f));
  }

  void release_closed() {
    for (auto it = rows.begin(); it != rows.end();) {
      Row& r = *it->second;
      if (r.closing && r.queued == 0 && (!r.has_last || caller->done(r.last))) {
        if (r.noise_ready.valid()) r.noise_ready.wait();  // the worker may still write the slot's pinned buffer
        r.pages.release(*pool);
        slots->release(r.slot);
        it = rows.erase(it);
      } else {
        ++it;
      }
    }
  }
};

RenderService::RenderService(const RenderServiceConfig& cfg, std::vector<RenderStage> program) : impl_(new Impl) {
  Impl& m = *impl_;
  m.cfg = cfg;
  m.arena = std::make_unique<RenderArena>(cfg.arena);
  m.pool = std::make_unique<HostPagePool>(m.arena->pool());
  m.slots = std::make_unique<SlotPool>(cfg.arena.slots);
  m.build_slot = m.slots->acquire();  // reserved for voice-prefix builds
  m.caller = std::make_unique<RenderCaller>(cfg.caller, m.arena->pool(), m.arena->state(), m.arena->voices(),
                                            std::move(program));
  m.caller->set_timing(cfg.timing);
  check(cudaHostAlloc(reinterpret_cast<void**>(&m.pinned_noise),
                      size_t(cfg.arena.slots) * kMel * size_t(cfg.arena.noise_cap) * sizeof(float), cudaHostAllocDefault),
        "pinned noise");
  m.admit_events.resize(size_t(cfg.arena.slots));
  for (auto& e : m.admit_events) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
  m.open_events.resize(size_t(cfg.arena.slots));
  for (auto& e : m.open_events) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
  m.voices.resize(size_t(cfg.arena.voices));
  check(cudaStreamCreateWithFlags(&m.aux, cudaStreamNonBlocking), "aux stream");
  for (int i = 0; i < cfg.noise_workers; ++i) m.workers.emplace_back([&m] { m.worker_loop(); });
}

RenderService::~RenderService() {
  Impl& m = *impl_;
  std::vector<ServiceOut> drop;
  while (!m.inflight.empty() || !m.pending.empty()) collect(&drop, true);
  if (m.merges > 0)  // experiment evidence for engine runs (the engine's stats JSON does not carry it)
    std::fprintf(stderr, "[l7 render] call merging: %lld merged GPU calls carried %lld conductor calls\n",
                 (long long)m.merges, (long long)m.merged_calls);
  {
    std::lock_guard<std::mutex> g(m.qmu);
    m.stop = true;
  }
  m.qcv.notify_all();
  for (auto& t : m.workers) t.join();
  for (auto& e : m.admit_events) cudaEventDestroy(e);
  for (auto& e : m.open_events) cudaEventDestroy(e);
  if (m.aux != nullptr) cudaStreamDestroy(m.aux);
  cudaFreeHost(m.pinned_noise);
}

int32_t RenderService::add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                                 const std::vector<float>& spk_cond, std::string* err) {
  Impl& m = *impl_;
  int32_t v = -1;
  for (size_t i = 0; i < m.voices.size(); ++i)
    if (m.voices[i].voice < 0) {
      v = int32_t(i);
      break;
    }
  const int32_t pt = int32_t(prompt_tokens.size()), pf = 2 * pt;
  if (v < 0 || pf > m.cfg.arena.max_prompt_frames || prompt_feat.size() != size_t(pf) * kMel ||
      !(spk_cond.empty() || spk_cond.size() == size_t(kMel))) {
    if (err) *err = v < 0 ? "voice table full" : "voice reference shape";
    return -1;
  }
  std::vector<float> noise(size_t(kMel) * size_t(pf));
  draw_voice_prompt_noise(pf, noise.data(), pf);
  m.arena->set_voice(v, pf, noise.data(), prompt_tokens.data(), prompt_feat.data(),
                     spk_cond.empty() ? nullptr : spk_cond.data(), m.aux);
  VoicePrefix& p = m.voices[size_t(v)];
  p = VoicePrefix{};
  p.voice = v;
  p.prompt_tokens = pt;
  p.prompt_frames = pf;
  p.prompt_ids = prompt_tokens;
  // The shared prefix (complete blocks solved once per voice) needs the encoder / estimator stages;
  // until it is built rows start at position 0 (same values, longer first suffix).
  return v;
}

int32_t RenderService::build_voice_prefix(int32_t voice, std::string* err) {
  Impl& m = *impl_;
  if (voice < 0 || size_t(voice) >= m.voices.size() || m.voices[size_t(voice)].voice < 0) {
    if (err) *err = "build_voice_prefix: bad voice";
    return -1;
  }
  VoicePrefix& vp = m.voices[size_t(voice)];
  if (vp.built) return 0;
  const int32_t stable = stable_prefix_tokens(vp.prompt_tokens);
  if (stable < kTokenChunk) return 0;  // no complete block before the look-ahead: rows start at 0
  // The build row sees the prompt cut at stable + 3: prompt frames = its whole suffix (cond, prompt noise).
  VoicePrefix cut = vp;
  cut.prompt_tokens = stable + kLookahead;
  cut.prompt_frames = kMelPerToken * cut.prompt_tokens;
  cut.prompt_ids.assign(vp.prompt_ids.begin(), vp.prompt_ids.begin() + cut.prompt_tokens);
  cut.built = false;
  cudaStream_t s = m.caller->stream(StreamRole::kSUrgent);
  m.arena->clear_slot(m.build_slot, s);
  RowPages pages;
  pages.admit(*m.pool, nullptr);
  RowRender row;
  row.slot = m.build_slot;
  row.voice = &cut;
  row.geom.phase = Phase::kSteadyCommit;
  row.geom.lookahead = kLookahead;  // the estimator solves 2 x stable frames; tails saved at `stable`
  row.pages = &pages;
  row.extra_flags = kRowLookaheadPadZero;
  CallPlan plan;
  const BuildStatus bs = build_call(Phase::kSteadyCommit, &row, 1, *m.pool, plan);
  if (bs != BuildStatus::kOk) {
    pages.release(*m.pool);
    if (err) *err = std::string("build_voice_prefix: ") + build_status_name(bs);
    return -1;
  }
  // S_urgent's internal call slot (free unless a previous build is still running). Calls already in flight
  // keep running; any that finish while we wait are kept in `stash` for the next collect(). (Before: every
  // in-flight call was drained and its output DROPPED, so a voice's first use while other rows were rendering
  // lost those rows' audio.)
  try {
    while (!m.caller->can_launch(StreamRole::kSUrgent, true)) {
      if (m.inflight.empty()) throw std::logic_error("no free urgent call slot and nothing in flight");
      collect_impl(&m.stash, true);
    }
  } catch (const std::exception& e) {
    pages.release(*m.pool);
    if (err) *err = std::string("build_voice_prefix: ") + e.what();
    return -1;
  }
  RenderTicket t;
  try {
    if (!m.caller->launch(plan, StreamRole::kSUrgent, &t, nullptr, nullptr, 0, true)) throw std::logic_error("no call slot");
    m.arena->save_prefix_tails(m.build_slot, voice, s);
  } catch (const std::exception& e) {
    pages.release(*m.pool);
    if (err) *err = std::string("build_voice_prefix: ") + e.what();
    return -1;
  }
  // Keep the complete pages (and the partial page holding frames [64 n, 2 stable)) for the voice.
  const int32_t full = kMelPerToken * stable / kPageFrames;
  const bool partial = (kMelPerToken * stable) % kPageFrames != 0;
  const auto& pg = pages.pages();
  vp.shared_pages.assign(pg.begin(), pg.begin() + full);
  vp.partial_page = partial ? pg[size_t(full)] : -1;
  std::vector<int32_t> keep(vp.shared_pages);
  if (partial) keep.push_back(vp.partial_page);
  if (!keep.empty()) m.pool->retain(keep.data(), int32_t(keep.size()));
  vp.stable_tokens = stable;
  vp.built = true;
  // The build runs on S_urgent without the host waiting for it (execution only; production builds its prefix
  // synchronously). Everything that reads the prefix is ordered after it on the GPU: a row opened from now on
  // loads the prefix tails and copies the partial page on S_urgent, and its first call on another stream
  // waits on the row's admission event recorded there. The build row's own references (the pages it wrote
  // beyond the kept ones) are released only when the build is done, so no other row can take such a page
  // while the build still writes it.
  Impl::Inflight f;
  f.ticket = t;
  f.internal = true;
  f.voice = voice;
  f.scratch_pages = pg;
  m.inflight.push_back(std::move(f));
  if (!m.cfg.async_prefix_build) m.caller->wait(t);
  return 0;
}

int32_t RenderService::remove_voice(int32_t voice, std::string* err) {
  Impl& m = *impl_;
  if (voice < 0 || size_t(voice) >= m.voices.size() || m.voices[size_t(voice)].voice < 0) {
    if (err) *err = "remove_voice: bad voice";
    return -1;
  }
  m.release_closed();  // closed rows whose last call is done let go of the prefix pages now
  for (const auto& kv : m.rows)
    if (kv.second->voice == voice) {
      if (err) *err = "remove_voice: a row still uses the voice";
      return -2;
    }
  // Rows are released only after their last call is done (release_closed), so no queued or running call reads the
  // voice's tables or pages now, except its own prefix build.
  for (const auto& f : m.inflight)
    if (f.internal && f.voice == voice && !m.caller->done(f.ticket)) {
      if (err) *err = "remove_voice: the voice's prefix build is still running";
      return -2;
    }
  VoicePrefix& vp = m.voices[size_t(voice)];
  std::vector<int32_t> keep(vp.shared_pages);
  if (vp.partial_page >= 0) keep.push_back(vp.partial_page);
  if (!keep.empty()) m.pool->release(keep.data(), int32_t(keep.size()));
  // The device table rows (prompt tokens / mel / noise / speaker, prefix tails) are rewritten by the next
  // add_voice / build_voice_prefix that takes this index; nothing reads them before that.
  vp = VoicePrefix{};
  return 0;
}

int32_t RenderService::voices_in_use() const {
  int32_t n = 0;
  for (const auto& v : impl_->voices) n += v.voice >= 0 ? 1 : 0;
  return n;
}

int32_t RenderService::open_row(uint64_t key, int32_t voice, uint64_t seed, int32_t budget, bool demo, std::string* err) {
  Impl& m = *impl_;
  if (m.find(key) != nullptr || voice < 0 || size_t(voice) >= m.voices.size() || m.voices[size_t(voice)].voice < 0) {
    if (err) *err = "open_row: bad row or voice";
    return -1;
  }
  const int32_t w = target_noise_width(budget);
  // The budget's noise extension may pass the table (budgets of 2,047+ tokens); draw_row_noise_extension keeps the
  // frames a row can reach (its tokens are bounded by the LM context), so only W must fit here.
  if (w > m.cfg.arena.noise_cap) {
    if (err) *err = "open_row: budget exceeds the noise table";
    return -1;
  }
  const int32_t slot = m.slots->acquire();
  if (slot < 0) {
    if (err) *err = "open_row: no render-state slot";
    return -2;
  }
  auto row = std::make_unique<Impl::Row>(&m.cfg.commit);
  row->slot = slot;
  row->voice = voice;
  row->seed = seed;
  row->budget = budget;
  row->noise_w = w;
  row->demo = demo;
  const VoicePrefix& vp = m.voices[size_t(voice)];
  cudaStream_t s = m.caller->stream(StreamRole::kSUrgent);
  float* host = m.pinned_noise + size_t(slot) * kMel * size_t(m.cfg.arena.noise_cap);
  cudaEvent_t prev = m.admit_events[size_t(slot)];
  const int32_t pf = vp.prompt_frames, cap = m.cfg.arena.noise_cap;
  auto draw = [prev, host, seed, pf, budget, cap] {
    // The previous owner's upload of this slot's pinned buffer must be done before it is overwritten.
    check(cudaEventSynchronize(prev), "admit event");
    draw_row_target_noise(seed, pf, budget, host, cap);
  };
  if (m.workers.empty()) {
    draw();
  } else {
    auto task = std::make_shared<std::packaged_task<void()>>(draw);
    row->noise_ready = task->get_future().share();
    m.submit([task] { (*task)(); });
  }
  // Frames past W read as production's zero padding.
  float* dst = m.arena->state().target_noise + size_t(slot) * kMel * size_t(m.cfg.arena.noise_cap);
  check(cudaMemsetAsync(dst, 0, size_t(kMel) * size_t(m.cfg.arena.noise_cap) * sizeof(float), s), "noise memset");
  if (!demo) {
    row->pages.admit(*m.pool, &vp);
    if (vp.built) m.arena->load_prefix_tails(voice, slot, s);
    else m.arena->clear_slot(slot, s);
  } else {
    m.arena->clear_slot(slot, s);
  }
  check(cudaEventRecord(m.open_events[size_t(slot)], s), "open record");
  if (m.workers.empty()) {
    m.rows.emplace(key, std::move(row));
    m.ensure_noise(*m.rows[key]);
  } else {
    m.rows.emplace(key, std::move(row));
  }
  return 0;
}

int32_t RenderService::prewarm_voice(int32_t voice, int32_t max_rows, int32_t row_tokens, PrewarmStats* stats,
                                     std::string* err) {
  Impl& m = *impl_;
  if (voice < 0 || size_t(voice) >= m.voices.size() || m.voices[size_t(voice)].voice < 0) {
    if (err) *err = "prewarm_voice: bad voice";
    return -1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  // The keys depend on the voice's prompt geometry only, so voices of the same prompt length reuse the set (the
  // host replay takes ~3 ms; a voice added while serving would pay it on the conductor thread).
  const VoicePrefix& vp = m.voices[size_t(voice)];
  const int32_t rows_cap = std::min(max_rows, m.cfg.caller.max_rows);
  const auto memo_key = std::make_tuple(vp.prompt_tokens, vp.built ? vp.stable_tokens : -1, rows_cap, row_tokens);
  auto mit = m.key_sets.find(memo_key);
  if (mit == m.key_sets.end())
    mit = m.key_sets.emplace(memo_key, enumerate_graph_keys(vp, m.cfg.commit, rows_cap, row_tokens)).first;
  const std::set<GraphKey>& keys = mit->second;
  int32_t captured = 0, n_keys = int32_t(keys.size()), skipped = 0;
  try {
    for (const auto& k : keys) {
      if (m.caller->workspace_need(Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket) >
          m.caller->workspace_bytes(StreamRole(k.role))) {
        ++skipped;  // cannot be captured (nor run) with this workspace: not an error here
        continue;
      }
      if (m.caller->prewarm(StreamRole(k.role), Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket))
        ++captured;
    }
    // L8 arm KT2N_HOLD_URGENT: the S_urgent twin of each single-row S_bulk commit graph up to the twin suffix cap.
    if (m.cfg.hold_urgent_twins)
      for (const auto& k : keys) {
        if (k.role != int32_t(StreamRole::kSBulk) || k.rows_bucket != 1 || k.suffix_bucket > m.cfg.hold_urgent_twin_max_suffix ||
            keys.count(GraphKey{int32_t(StreamRole::kSUrgent), k.phase, k.rows_bucket, k.suffix_bucket, k.keys_bucket}) ||
            m.caller->workspace_need(Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket) >
                m.caller->workspace_bytes(StreamRole::kSUrgent))
          continue;
        ++n_keys;
        if (m.caller->prewarm(StreamRole::kSUrgent, Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket)) ++captured;
      }
    // L8 two-level arm: the S_aux twin of each single-row S_bulk commit graph up to the twin suffix cap.
    if (m.cfg.aux_steady_twins && m.caller->aux_max_rows() > 0)
      for (const auto& k : keys) {
        if (k.role != int32_t(StreamRole::kSBulk) || k.rows_bucket != 1 || k.suffix_bucket > m.cfg.hold_urgent_twin_max_suffix ||
            keys.count(GraphKey{int32_t(StreamRole::kSAux), k.phase, k.rows_bucket, k.suffix_bucket, k.keys_bucket}) ||
            m.caller->workspace_need(Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket) >
                m.caller->workspace_bytes(StreamRole::kSAux))
          continue;
        ++n_keys;
        if (m.caller->prewarm(StreamRole::kSAux, Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket)) ++captured;
      }
    // The aux role (when on) takes S_urgent's commit calls up to its rows cap that fit its workspace.
    const int32_t ar = m.caller->aux_max_rows();
    for (const auto& k : keys) {
      if (ar <= 0 || k.role != int32_t(StreamRole::kSUrgent) || k.rows_bucket > ar ||
          m.caller->workspace_need(Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket) >
              m.caller->workspace_bytes(StreamRole::kSAux))
        continue;
      ++n_keys;
      if (m.caller->prewarm(StreamRole::kSAux, Phase(k.phase), k.rows_bucket, k.suffix_bucket, k.keys_bucket)) ++captured;
    }
  } catch (const std::exception& e) {
    if (err) *err = std::string("prewarm_voice: ") + e.what();
    return -1;
  }
  if (stats != nullptr) {
    stats->keys = n_keys;
    stats->skipped = skipped;
    stats->captured = captured;
    stats->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  return captured;
}

int32_t RenderService::pages_needed(int32_t voice, int32_t tokens, bool demo) const {
  const Impl& m = *impl_;
  if (voice < 0 || size_t(voice) >= m.voices.size() || m.voices[size_t(voice)].voice < 0) return -1;
  const VoicePrefix& vp = m.voices[size_t(voice)];
  return row_pages_needed(&vp, vp.prompt_tokens, tokens, demo);
}

int32_t RenderService::extend_noise(uint64_t key, std::string* err) {
  Impl& m = *impl_;
  Impl::Row* r = m.find(key);
  if (r == nullptr || r->extended) {
    if (err) *err = r == nullptr ? "extend_noise: no row" : "extend_noise: already extended";
    return -1;
  }
  cudaStream_t s = m.caller->stream(StreamRole::kSUrgent);
  m.ensure_noise(*r);
  check(cudaEventSynchronize(m.admit_events[size_t(r->slot)]), "admit event");
  float* host = m.pinned_noise + size_t(r->slot) * kMel * size_t(m.cfg.arena.noise_cap);
  const int32_t f = draw_row_noise_extension(r->seed, r->budget, host, m.cfg.arena.noise_cap);
  m.arena->upload_row_noise_range(r->slot, host, r->noise_w, f, s);
  check(cudaEventRecord(m.admit_events[size_t(r->slot)], s), "admit record");
  r->extended = true;
  r->admit_fresh = true;
  return 0;
}

void RenderService::close_row(uint64_t key) {
  Impl& m = *impl_;
  Impl::Row* r = m.find(key);
  if (r == nullptr) return;
  r->closing = true;
  m.release_closed();
}

int32_t RenderService::launch(ServiceCall&& call, std::string* err) {
  try {
    return launch_impl(std::move(call), err);
  } catch (const std::exception& e) {  // never throw into the conductor
    if (err) *err = std::string("launch: ") + e.what();
    ++impl_->failed_calls;
    impl_->last_error = e.what();
    return -1;
  }
}

int64_t RenderService::failed_calls() const { return impl_->failed_calls; }
int64_t RenderService::aux_calls() const { return impl_->aux_calls; }
int64_t RenderService::aux_fallback_bulk() const { return impl_->aux_fallback_bulk; }
void RenderService::set_aux_auto(bool on) { impl_->cfg.aux_auto = on; }
void RenderService::set_merge(int32_t bulk_rows, int32_t urgent_rows, int32_t depth) {
  impl_->cfg.merge_bulk_rows = bulk_rows;
  impl_->cfg.merge_urgent_rows = urgent_rows;
  impl_->cfg.merge_depth = depth;
}
int64_t RenderService::merges() const { return impl_->merges; }
int64_t RenderService::merged_calls() const { return impl_->merged_calls; }
int64_t RenderService::split_calls() const { return impl_->split_calls; }

int32_t RenderService::launch_impl(ServiceCall&& call, std::string* err) {
  Impl& m = *impl_;
  const auto t0 = std::chrono::steady_clock::now();
  const int32_t n = int32_t(call.rows.size());
  if (n <= 0) {
    if (err) *err = "launch: empty call";
    return -1;
  }
  // Call phase: the rows' common phase; first commits ride in a steady call.
  Phase phase = call.rows[0].phase;
  for (const auto& rq : call.rows) {
    if (rq.phase == phase) continue;
    const bool fs = (rq.phase == Phase::kFirstCommit || rq.phase == Phase::kSteadyCommit) &&
                    (phase == Phase::kFirstCommit || phase == Phase::kSteadyCommit);
    if (!fs) {
      if (err) *err = "launch: mixed phases";
      return -1;
    }
    phase = Phase::kSteadyCommit;
  }
  std::vector<RowRender> rr(static_cast<size_t>(n));
  std::vector<Impl::Row*> rows(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const ServiceRowReq& q = call.rows[size_t(i)];
    Impl::Row* r = m.find(q.row);
    if (r == nullptr || r->closing) {
      if (err) *err = "launch: unknown row";
      return -1;
    }
    rows[size_t(i)] = r;
    RenderGeometry g;
    const int32_t nt = int32_t(q.tokens.size());
    try {
      if (q.phase == Phase::kWholeRender) {
        g = plan_whole_render(nt - 3);
      } else if (q.phase == Phase::kPreview) {
        g = plan_preview(m.cfg.commit);
      } else if (q.finalize || q.phase == Phase::kFinalFlush) {
        g = r->commit.plan_final(nt - 3);
      } else {
        if (q.committed_tokens != r->commit.committed_tokens()) throw std::logic_error("committed tokens disagree");
        g = r->commit.plan_commit_at(q.commit_tokens);
        g.phase = q.phase == Phase::kFirstCommit ? Phase::kFirstCommit : g.phase;
      }
    } catch (const std::exception& e) {
      if (err) *err = std::string("launch: ") + e.what();
      return -1;
    }
    if (nt < g.render_tokens) {
      if (err) *err = "launch: fewer tokens than the render needs";
      return -1;
    }
    RowRender& x = rr[size_t(i)];
    x.slot = r->slot;
    x.voice = &m.voices[size_t(r->voice)];
    x.geom = g;
    x.ids = q.tokens.data();
    x.pages = (phase == Phase::kWholeRender || phase == Phase::kPreview) ? nullptr : &r->pages;
    x.seed = r->seed;
    x.noise_w = r->noise_w;
    x.extra_flags = kRowLookaheadPadZero;  // X_LOOKAHEAD_PAD_ZERO=1 (live)
  }
  Impl::Pending p;
  p.call_id = call.call_id;
  p.role = call.stream;
  const BuildStatus bs = build_call(phase, rr.data(), n, *m.pool, p.plan);
  if (bs != BuildStatus::kOk) {
    if (err) *err = std::string("launch: ") + build_status_name(bs);
    return bs == BuildStatus::kNoPages ? -2 : -1;
  }
  if (p.role == StreamRole::kSUrgent && phase == Phase::kFirstCommit && m.cfg.aux_auto && m.caller->aux_max_rows() > 0 &&
      m.urgent_busy())
    p.role = StreamRole::kSAux;  // auto dispatch; the fit check below may still send it back to S_urgent
  if (p.role == StreamRole::kSAux) {
    // The aux role (experiment arm) serves small commits only: without it, or for a call that does not fit its
    // rows cap or workspace, the call runs on S_urgent as it would without the arm (never a failure). An explicit
    // S_aux steady commit (L8's two-level arm) that does not fit runs on S_bulk, its natural stream.
    const int32_t ar = m.caller->aux_max_rows();
    if (ar <= 0 || p.plan.rows_bucket > ar || phase == Phase::kFinalFlush || phase == Phase::kWholeRender ||
        m.caller->workspace_need(p.plan.phase, p.plan.rows_bucket, p.plan.suffix_bucket, p.plan.keys_bucket) >
            m.caller->workspace_bytes(StreamRole::kSAux)) {
      p.role = (call.stream == StreamRole::kSAux && phase == Phase::kSteadyCommit) ? StreamRole::kSBulk : StreamRole::kSUrgent;
      if (p.role == StreamRole::kSBulk) ++m.aux_fallback_bulk;
    } else {
      ++m.aux_calls;
    }
  }
  // Admission work (noise upload, prefix page copy) on S_urgent, or on S_aux for an aux call (see ensure_noise).
  const StreamRole adm = p.role == StreamRole::kSAux ? StreamRole::kSAux : StreamRole::kSUrgent;
  cudaStream_t adm_s = m.caller->stream(adm);
  for (int32_t i = 0; i < n; ++i) m.ensure_noise(*rows[size_t(i)], adm);
  // Copy-on-admit of the voice prefix's partial page (before the row's first call, same stream order).
  for (int32_t i : p.plan.prefix_copy_rows) {
    const RowPages& pg = *rr[size_t(i)].pages;
    if (adm != StreamRole::kSUrgent)  // the prefix (built on S_urgent) and open_row's work come first
      check(cudaStreamWaitEvent(adm_s, m.open_events[size_t(rr[size_t(i)].slot)], 0), "open wait");
    m.arena->copy_prefix_page(pg.prefix_copy_src(), pg.pages()[size_t(pg.n_shared())], adm_s);
    check(cudaEventRecord(m.admit_events[size_t(rr[size_t(i)].slot)], adm_s), "admit record");
    rows[size_t(i)]->admit_fresh = true;
  }
  // stream:false whole renders of several long rows can need more workspace than the role has (L8 batches up to 8
  // DEMO rows per call, production's final_flush_buckets: 8 rows of 1,000 tokens need 2,070 MB of workspace, measured,
  // vs 1 GB per role, and the call would FAIL). Such a call is rendered as several calls of fewer rows, joined into
  // one output. Execution only: whole-render rows share nothing (scratch pages, full attention from position 0) and
  // L4 / L5 are batch-invariant, so every row's PCM is the same as in one call.
  if (phase == Phase::kWholeRender && n > 1 &&
      m.caller->workspace_need(p.plan.phase, p.plan.rows_bucket, p.plan.suffix_bucket, p.plan.keys_bucket) >
          m.caller->workspace_bytes(p.role)) {
    m.pool->release(p.plan.scratch_pages.data(), int32_t(p.plan.scratch_pages.size()));
    auto fits = [&](int32_t rows_n, int32_t suffix) {
      return m.caller->workspace_need(Phase::kWholeRender, bucket_of(kRowBuckets, rows_n),
                                      suffix_bucket_for(Phase::kWholeRender, suffix), keys_bucket_for(kMelPerToken * suffix)) <=
             m.caller->workspace_bytes(p.role);
    };
    std::vector<std::vector<int32_t>> groups(1);
    int32_t smax = 0;
    for (int32_t i = 0; i < n; ++i) {
      const int32_t s = rr[size_t(i)].voice->prompt_tokens + rr[size_t(i)].geom.render_tokens;  // suffix from position 0
      if (!groups.back().empty() && !fits(int32_t(groups.back().size()) + 1, std::max(smax, s))) {
        groups.emplace_back();
        smax = 0;
      }
      groups.back().push_back(i);
      smax = std::max(smax, s);
    }
    std::vector<Impl::Pending> parts(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {  // every part's scratch pages first (all-or-nothing)
      std::vector<RowRender> sub;
      for (int32_t i : groups[g]) sub.push_back(rr[size_t(i)]);
      const BuildStatus b2 = build_call(phase, sub.data(), int32_t(sub.size()), *m.pool, parts[g].plan);
      if (b2 != BuildStatus::kOk) {
        for (size_t h = 0; h < g; ++h)
          m.pool->release(parts[h].plan.scratch_pages.data(), int32_t(parts[h].plan.scratch_pages.size()));
        if (err) *err = std::string("launch: ") + build_status_name(b2);
        return b2 == BuildStatus::kNoPages ? -2 : -1;
      }
    }
    Impl::SplitOut& so = m.splits[call.call_id];
    so.parts_left = int32_t(groups.size());
    so.out = ServiceOut{};
    so.out.call_id = call.call_id;
    so.out.pcm.resize(size_t(n));
    const double bms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t g = 0; g < groups.size(); ++g) {
      Impl::Pending& q = parts[g];
      q.call_id = call.call_id;
      q.role = p.role;
      q.n_parts = int32_t(groups.size());
      q.parent_rows = groups[g];
      q.build_ms = g == 0 ? bms : 0.0;
      for (int32_t i : groups[g]) {
        q.row_keys.push_back(call.rows[size_t(i)].row);
        ++rows[size_t(i)]->queued;
      }
      m.pending.push_back(std::move(q));
    }
    ++m.split_calls;
    m.pump();
    return 0;
  }
  // Production advances the row state at launch time.
  apply_call_state(p.plan, rr.data(), n);
  for (int32_t i = 0; i < n; ++i) {
    if (phase != Phase::kWholeRender && phase != Phase::kPreview) rows[size_t(i)]->commit.commit_launched(rr[size_t(i)].geom);
    p.row_keys.push_back(call.rows[size_t(i)].row);
    ++rows[size_t(i)]->queued;
  }
  p.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  m.pending.push_back(std::move(p));
  m.pump();
  return 0;
}

void RenderService::collect(std::vector<ServiceOut>* out, bool wait) {
  Impl& m = *impl_;
  if (!m.stash.empty()) {  // older than anything still in flight
    for (auto& o : m.stash) out->push_back(std::move(o));
    m.stash.clear();
    wait = false;  // something is returned: do not block for more
  }
  collect_impl(out, wait);
}

void RenderService::collect_impl(std::vector<ServiceOut>* out, bool wait) {
  Impl& m = *impl_;
  m.pump();
  bool any = false;
  for (;;) {
    // One pass over the in-flight calls, oldest first, returning EVERY finished call, not only a finished prefix
    // of the queue. (Before 13:40 collection stopped at the oldest unfinished call, so a first commit on S_urgent
    // that finished in 9 ms waited for the older S_bulk calls ahead of it in the queue: measured 37-115 ms from
    // launch to PCM under bulk load vs 7-9 ms idle.)
    // Per-row order is kept: a call is held back while an older call sharing one of its rows is unfinished. (A
    // row's later call waits on its earlier call's event, so it cannot finish first; this guards a pass that sees
    // the older call unfinished and, a moment later, the newer one finished.)
    std::vector<uint64_t> blocked;
    bool progressed = false;
    for (auto it = m.inflight.begin(); it != m.inflight.end();) {
      Impl::Inflight& f = *it;
      bool held = false;
      for (uint64_t k : f.row_keys)
        if (std::find(blocked.begin(), blocked.end(), k) != blocked.end()) held = true;
      const bool ready = !held && (f.failed || m.caller->done(f.ticket));
      if (!ready) {
        blocked.insert(blocked.end(), f.row_keys.begin(), f.row_keys.end());
        ++it;
        continue;
      }
      if (f.internal) {  // a voice-prefix build: release its scratch references once it is done, no output
        m.caller->release(f.ticket);
      } else if (f.n_parts > 1) {  // a part of a split whole-render call: joined into the call's one output
        Impl::SplitOut& so = m.splits[f.call_id];
        so.out.launch_ms += f.launch_ms;
        if (f.failed) {
          so.out.status = -1;
        } else {
          const float g = m.caller->gpu_ms(f.ticket);
          if (g >= 0) so.out.gpu_ms += double(g);
          so.out.graph_nodes = f.graph_nodes;
          so.out.graph_captured = so.out.graph_captured || f.graph_captured;
          const float* pcm = m.caller->pcm(f.ticket);
          for (size_t i = 0; i < f.pcm_off.size() && i < f.parent_rows.size(); ++i)
            so.out.pcm[size_t(f.parent_rows[i])].assign(pcm + f.pcm_off[i], pcm + f.pcm_off[i] + f.pcm_count[i]);
          m.caller->release(f.ticket);
        }
        if (--so.parts_left == 0) {
          out->push_back(std::move(so.out));
          m.splits.erase(f.call_id);
          any = true;
        }
      } else {
        // One ServiceOut per conductor call (a merged call carries several, rows in part order).
        const size_t nparts = f.part_ids.empty() ? 1 : f.part_ids.size();
        const float* pcm = f.failed ? nullptr : m.caller->pcm(f.ticket);
        const float gms = f.failed ? 0.f : m.caller->gpu_ms(f.ticket);
        size_t r0 = 0;
        for (size_t k = 0; k < nparts; ++k) {
          const size_t nr = f.part_ids.empty() ? f.pcm_off.size() : size_t(f.part_rows[k]);
          ServiceOut o;
          o.call_id = f.part_ids.empty() ? f.call_id : f.part_ids[k];
          o.launch_ms = f.launch_ms;
          if (f.failed) {
            o.status = -1;
            o.pcm.resize(nr);
          } else {
            o.gpu_ms = gms;
            o.graph_nodes = f.graph_nodes;
            o.graph_captured = f.graph_captured;
            for (size_t i = r0; i < r0 + nr && i < f.pcm_off.size(); ++i)
              o.pcm.emplace_back(pcm + f.pcm_off[i], pcm + f.pcm_off[i] + f.pcm_count[i]);
          }
          r0 += nr;
          out->push_back(std::move(o));
        }
        if (!f.failed) m.caller->release(f.ticket);
        any = true;
      }
      if (!f.scratch_pages.empty()) m.pool->release(f.scratch_pages.data(), int32_t(f.scratch_pages.size()));
      it = m.inflight.erase(it);
      progressed = true;
    }
    if (progressed) m.pump();  // freed call slots take queued calls
    if (any || !wait || m.inflight.empty()) break;
    if (!progressed) {
      // Wait for whichever call finishes first: block on the only one, else poll (CUDA has no wait-any).
      if (m.inflight.size() == 1 && !m.inflight.front().failed) m.caller->wait(m.inflight.front().ticket);
      else std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }
  m.release_closed();
}

int32_t RenderService::inflight() const {
  int32_t n = int32_t(impl_->pending.size() + impl_->stash.size());
  for (const auto& f : impl_->inflight) n += f.part_ids.empty() ? 1 : int32_t(f.part_ids.size());
  return n;  // conductor calls (a merged GPU call counts each call it carries)
}
RenderArena& RenderService::arena() { return *impl_->arena; }
RenderCaller& RenderService::caller() { return *impl_->caller; }
HostPagePool& RenderService::pool() { return *impl_->pool; }

}  // namespace kt2n::render
