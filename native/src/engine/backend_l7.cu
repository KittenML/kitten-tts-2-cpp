// L8: the L7 renderer behind the conductor's RenderBackend (kt2n::render::RenderService, lane L7).
// Two arms:
//   "l7"         the real renderer: kt2n::render::Renderer::create(s3gen.kt2n) = L4 encoder + L5 estimator (one
//                captured graph per bucket) + L6 HiFT, add_voice(prompt tokens, prompt mel, raw x-vector) with
//                L4's GPU spk_affine. Voice prefixes are built on a voice's first use (production builds its
//                chunk-cache prefix on first use too) or all at create (KT2N_L7_PREFIXES=all).
//   "l7-standin" (lab) L7's stand-in stages (noise gather + stub PCM): the REAL render state with placeholder
//                audio. It tests the conductor <-> renderer path on the GPU; it is never the RC.
// Page reservation (admission back-pressure): a row reserves the pool pages its token budget can need before it
// is admitted (open_row returns kRenderBackpressure when the pool cannot cover it, and the conductor retries the
// job on a later turn), so a launch within the budget never finds the pool empty mid-request. A row that runs
// past its budget (the 5,000,000 + seed noise extension, up to 2 x budget + 3) reserves the extra pages when the
// extension starts; if the pool cannot cover them it continues on the pool's real free pages (counted as
// `extension_overcommit`). Reservations are released at close_row.
// Voice handles (sprint 5): the engine holds a stable handle per voice; the backend maps it to L7's table index.
// When a first-use prefix or a row's reservation can never fit (nothing reserved, so nothing will free pages),
// the backend evicts IDLE voice prefixes, least recently used first, instead of failing the request: L7
// remove_voice (refused with -2 while a row, call or prefix build still reads the voice) then add_voice again
// with the same prompt data, so the handle stays valid and its prefix is rebuilt on its next use (production
// builds its chunk-cache prefix on first use too). Before this, a pool smaller than the voices in use failed
// every later request (lab slot knobs: 61 of 72 requests at N=4 with 38 voices on a 112-page pool).
#include "build_flags.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "kt2n/render/service.h"
#if KT2N_ENGINE_WITH_RENDERER
#include "kt2n/render/renderer.h"
#endif

namespace kt2n::engine {
namespace {

class L7Render final : public RenderBackend {
 public:
  // Stand-in arm: the service owns its stage list.
  L7Render(const render::RenderServiceConfig& cfg, std::vector<render::RenderStage> program, const char* name)
      : owned_(std::make_unique<render::RenderService>(cfg, std::move(program))), svc_(owned_.get()), name_(name) {}
#if KT2N_ENGINE_WITH_RENDERER
  // Real arm.
  L7Render(std::unique_ptr<render::Renderer> r, const L7RenderOptions& o)
      : renderer_(std::move(r)), svc_(&renderer_->service()), name_("l7"), lazy_prefix_(o.lazy_prefix),
        reserve_(o.reserve_pages), prewarm_(o.prewarm),
        prewarm_rows_(o.prewarm_rows > 0 ? o.prewarm_rows : std::min(20, std::max(1, o.max_rows))),
        prewarm_tokens_(std::max(16, o.prewarm_tokens)) {}
#endif

  int add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                const std::vector<float>& embedding, std::string* err) override {
    int l7v;
#if KT2N_ENGINE_WITH_RENDERER
    if (renderer_) {
      // Raw 192-d x-vector: the renderer applies L4's spk_affine on the GPU.
      l7v = renderer_->add_voice(prompt_tokens, prompt_feat, embedding, err);
      if (l7v >= 0) {
        const int v = next_handle_++;
        l7_of_[v] = l7v;
        prompt_toks_[v] = int(prompt_tokens.size());
        vdata_[v] = VoiceData{prompt_tokens, prompt_feat, embedding};  // for a re-add after prefix eviction
        // Prefix (+ graph prewarm) at create: every voice with KT2N_L7_PREFIXES=all; else only the first voice,
        // whose prewarm captures the keys voices of the same prompt geometry share (L7's statement; L8's runs
        // still saw a few live captures at N>=4 on the prewarmed voice itself: lane log 13:37).
        // Never prewarm on a lazy first use (that request would pay the captures).
        bool now = !lazy_prefix_ || (prewarm_ && !prewarm_tried_);
        // A pool too small for every prefix at create (lab knobs with KT2N_L7_PREFIXES=all): leave the rest lazy
        // (built on first use, with LRU eviction) instead of failing the voice; L7 18:35 saw requests fail on
        // voices whose create-time build did not fit.
        if (now && reserve_ && prefix_pages_bound(v) > avail_pages()) {
          if (create_lazy_fallback_++ == 0)
            std::fprintf(stderr, "[l7 render] render pool too small to build every voice prefix at create; the "
                                 "remaining voices build on first use\n");
          now = false;
        }
        if (now) {
          prewarm_tried_ = true;  // lazy mode: only the first voice pays at create, even when its prewarm fails
          if (build_prefix(v, err) != 0) return -1;
          // Prewarm is an optimisation: a failure (e.g. a key whose capture does not fit a lab-sized pool or
          // workspace) must not lose the voice. The voice stays usable; missing graphs are captured live.
          if (prewarm_) {
            std::string perr;
            if (prewarm(l7v, &perr) != 0) {
              if (prewarm_errors_++ == 0) std::fprintf(stderr, "[l7 render] graph prewarm failed (voice kept): %s\n", perr.c_str());
              prewarm_error_ = perr;
            }
          }
        }
        return v;
      }
      return l7v;
    }
#endif
    // The stand-in program reads no speaker conditioning.
    l7v = svc_->add_voice(prompt_tokens, prompt_feat, embedding.size() == 80 ? embedding : std::vector<float>{}, err);
    if (l7v < 0) return l7v;
    const int v = next_handle_++;
    l7_of_[v] = l7v;
    return v;
  }
  int open_row(uint64_t row, int voice, uint64_t seed, int budget, bool demo, std::string* err) override {
    int need = 0;
#if KT2N_ENGINE_WITH_RENDERER
    if (renderer_) {
      if (l7(voice) < 0) {
        if (err)
          *err = voice < 0 ? std::string("voice has no render state (its render add_voice failed; see render_voice_errors)")
                           : "render voice handle " + std::to_string(voice) + " has no table row (lost in a prefix eviction re-add)";
        return -1;
      }
      last_use_[voice] = ++use_tick_;
      if (prompt_toks_.count(voice) && !built_[voice]) {
        // A first-use prefix takes pool pages for good: with reservations on it must not take pages that open
        // rows have reserved (they could then meet an empty pool mid-request). Wait while rows hold reservations;
        // when nothing is reserved, evict idle prefixes (LRU); refuse only if it still cannot fit.
        if (reserve_) {
          const int pneed = prefix_pages_bound(voice);
          if (pneed > avail_pages() && reserved_ == 0) evict_idle_prefixes(pneed, voice);
          const int avail = avail_pages();
          if (pneed > avail) {
            if (reserved_ == 0) {
              if (err) *err = "voice prefix needs up to " + std::to_string(pneed) + " pool pages, the pool has " +
                              std::to_string(avail);
              return -1;
            }
            backpressure_++;
            return kRenderBackpressure;
          }
        }
        if (build_prefix(voice, err) != 0) return -1;
      }
      if (reserve_) {
        need = svc_->pages_needed(l7(voice), budget, demo);  // L7's exact count (gtest RowPagesNeededIsExact)
        if (need > avail_pages() && reserved_ == 0) evict_idle_prefixes(need, voice);
        const int avail = avail_pages();
        if (need > avail) {
          if (reserved_ == 0) {  // nothing to wait for: this row can never fit
            if (err) *err = "open_row: budget needs " + std::to_string(need) + " pool pages, the pool has " +
                            std::to_string(avail);
            return -1;
          }
          backpressure_++;
          return kRenderBackpressure;
        }
      }
    }
#endif
    const int rc = svc_->open_row(row, l7(voice), seed, budget, demo, err);
    if (rc == -2) return kRenderBackpressure;  // no render-state slot: wait, as for pages
    if (rc < 0) return -1;
    note_pool();
    if (need > 0) {
      reserved_ += need;
      row_need_[row] = need;
      row_voice_budget_[row] = {voice, budget, demo};
    }
    return 0;
  }
  // L7 frees a closed row's pages once that row's in-flight calls are collected (release_closed), so its
  // reservation is held until the row's own last call is collected. Keyed per row: since L7 537f19b collect()
  // returns finished calls in launch order per row only, not globally (the count-based release before 13:50
  // assumed global order and could free a reservation a few ms early).
  // L7 RenderService::remove_voice (L8's patch for L7, 10 Oct): the voice's prefix pages leave the prefix count, and
  // its table index may be reused by a later add_voice, so the index's lazy-prefix state is reset.
  int remove_voice(int voice, std::string* err) override {
    const int l7v = l7(voice);
    if (l7v >= 0) {
      const int rc = svc_->remove_voice(l7v, err);
      if (rc != 0) return rc == -2 ? -2 : -1;
    } else if (!l7_of_.count(voice)) {
      if (err) *err = "unknown render voice handle";
      return -1;
    }
    forget_prefix(voice);
    l7_of_.erase(voice);
    prompt_toks_.erase(voice);
    vdata_.erase(voice);
    last_use_.erase(voice);
    note_pool();
    voices_removed_++;
    return 0;
  }
  void close_row(uint64_t row) override {
    svc_->close_row(row);
    note_pool();
    auto it = row_need_.find(row);
    if (it != row_need_.end()) {
      auto c = row_calls_.find(row);
      if (c == row_calls_.end() || c->second == 0)
        reserved_ -= it->second;
      else
        closed_pending_[row] += it->second;
      row_need_.erase(it);
    }
    row_voice_budget_.erase(row);
  }
  int extend_noise(uint64_t row, std::string* err) override {
#if KT2N_ENGINE_WITH_RENDERER
    auto it = row_need_.find(row);
    auto vb = row_voice_budget_.find(row);
    if (it != row_need_.end() && vb != row_voice_budget_.end()) {
      const int full = svc_->pages_needed(l7(vb->second.voice), 2 * vb->second.budget + 3, vb->second.demo);
      const int extra = std::max(0, full - it->second);
      if (extra > svc_->pool().pages_total() - prefix_pages_ - reserved_) extension_overcommit_++;
      reserved_ += extra;
      it->second += extra;
    }
#endif
    return svc_->extend_noise(row, err);
  }
  int launch(RenderCall&& call, std::string* err) override {
    render::ServiceCall sc;
    sc.call_id = call.call_id;
    sc.stream = call.stream;
    sc.rows.reserve(call.rows.size());
    for (auto& r : call.rows) {
      render::ServiceRowReq q;
      q.row = r.row;
      q.voice = l7(r.voice);
      q.phase = r.phase;
      q.commit_index = r.commit_index;
      q.committed_tokens = r.committed_tokens;
      q.commit_tokens = r.commit_tokens;
      q.finalize = r.finalize;
      q.tokens = std::move(r.tokens);
      q.seq = r.seq;
      sc.rows.push_back(std::move(q));
    }
    std::vector<uint64_t> rows;
    rows.reserve(sc.rows.size());
    for (const auto& q : sc.rows) rows.push_back(q.row);
    const uint64_t id = sc.call_id;
    if (svc_->launch(std::move(sc), err) < 0) return -1;  // refused before it was queued: never collected
    for (uint64_t r : rows) row_calls_[r]++;
    call_rows_[id] = std::move(rows);
    return 0;
  }
  void collect(std::vector<RenderOut>* out, bool wait) override {
    std::vector<render::ServiceOut> so;
    svc_->collect(&so, wait);
    if (!so.empty()) {
      for (const auto& o : so) {
        auto cr = call_rows_.find(o.call_id);
        if (cr == call_rows_.end()) continue;
        for (uint64_t r : cr->second) {
          auto c = row_calls_.find(r);
          if (c == row_calls_.end() || --c->second > 0) continue;
          row_calls_.erase(c);
          auto pend = closed_pending_.find(r);
          if (pend != closed_pending_.end()) {
            reserved_ -= pend->second;
            closed_pending_.erase(pend);
          }
        }
        call_rows_.erase(cr);
      }
      note_pool();
    }
    for (auto& o : so) {
      if (o.graph_captured) live_captures_++;
      if (o.status != 0) failed_calls_++;
      RenderOut r;
      r.call_id = o.call_id;
      r.status = o.status;
      r.pcm = std::move(o.pcm);
      r.gpu_ms = o.gpu_ms;
      r.launch_ms = o.launch_ms;
      r.graph_nodes = o.graph_nodes;
      r.graph_captured = o.graph_captured;
      out->push_back(std::move(r));
    }
  }
  int inflight() const override { return svc_->inflight(); }
  const char* name() const override { return name_; }
  // Called from the stats thread: only atomics (the pool itself is the conductor's).
  std::string stats_json() const override {
    return "{\"pool_pages\":" + std::to_string(pool_total_.load()) +
           ",\"pool_free\":" + std::to_string(pool_free_.load()) + ",\"pool_free_min\":" + std::to_string(pool_free_min_.load()) +
           ",\"prefix_pages\":" + std::to_string(prefix_pages_.load()) + ",\"prefixes_built\":" +
           std::to_string(prefixes_built_.load()) + ",\"prefix_build_ms\":" + std::to_string(prefix_build_ms_.load()) +
           ",\"voices_removed\":" + std::to_string(voices_removed_.load()) +
           ",\"prefixes_evicted\":" + std::to_string(prefixes_evicted_.load()) +
           ",\"readd_failures\":" + std::to_string(readd_failures_.load()) +
           ",\"create_lazy_fallback\":" + std::to_string(create_lazy_fallback_.load()) +
           ",\"reserved_pages\":" + std::to_string(reserved_.load()) +
           ",\"page_backpressure\":" + std::to_string(backpressure_.load()) +
           ",\"extension_overcommit\":" + std::to_string(extension_overcommit_.load()) +
           ",\"prewarm\":{\"voices\":" + std::to_string(prewarmed_voices_.load()) + ",\"errors\":" +
           std::to_string(prewarm_errors_.load()) + ",\"keys\":" +
           std::to_string(prewarm_keys_.load()) + ",\"captured\":" + std::to_string(prewarm_captured_.load()) +
           ",\"ms\":" + std::to_string(prewarm_ms_.load()) + "}" + ",\"live_captures\":" +
           std::to_string(live_captures_.load()) + ",\"failed_calls\":" + std::to_string(failed_calls_.load()) + "}";
  }

 private:
#if KT2N_ENGINE_WITH_RENDERER
  // Upper bound on the pages a voice's prefix keeps (its complete 64-frame pages of 2 frames per prompt token).
  int prefix_pages_bound(int voice) const {
    auto it = prompt_toks_.find(voice);
    const int p = it != prompt_toks_.end() ? it->second : 0;
    return (2 * p + 63) / 64 + 1;
  }
  // Unbuild one idle voice's prefix: L7 remove_voice (refused while anything reads the voice), then add_voice with
  // the same prompt data under the same handle. 0 = its prefix pages are back in the pool.
  int unbuild_prefix(int voice) {
    auto d = vdata_.find(voice);
    const int l7v = l7(voice);
    if (d == vdata_.end() || l7v < 0) return -1;
    std::string err;
    if (svc_->remove_voice(l7v, &err) != 0) return -1;  // -2: a row, call or prefix build still reads it
    forget_prefix(voice);
    const int nv = renderer_->add_voice(d->second.toks, d->second.feat, d->second.emb, &err);
    l7_of_[voice] = nv;  // -1: the handle is dead; its next open_row fails with a clear error
    if (nv < 0) {
      readd_failures_++;
      std::fprintf(stderr, "[l7 render] re-add after prefix eviction failed (voice %d): %s\n", voice, err.c_str());
    }
    prefixes_evicted_++;
    return 0;
  }
  // Evict idle built prefixes, least recently used first (never `keep`), until `want` pages are available.
  void evict_idle_prefixes(int want, int keep) {
    std::vector<std::pair<uint64_t, int>> lru;
    for (const auto& b : built_)
      if (b.second && b.first != keep) lru.push_back({last_use_.count(b.first) ? last_use_[b.first] : 0, b.first});
    std::sort(lru.begin(), lru.end());
    for (const auto& c : lru) {
      if (avail_pages() >= want) break;
      unbuild_prefix(c.second);
    }
    note_pool();
  }
  int prewarm(int voice, std::string* err) {
    render::RenderService::PrewarmStats st;
    if (svc_->prewarm_voice(voice, prewarm_rows_, prewarm_tokens_, &st, err) < 0) return -1;
    prewarmed_voices_++;
    prewarm_keys_ += st.keys;
    prewarm_captured_ += st.captured;
    prewarm_ms_ = prewarm_ms_.load() + st.ms;
    return 0;
  }
  int build_prefix(int voice, std::string* err) {
    const int free0 = svc_->pool().pages_free();
    const auto t0 = std::chrono::steady_clock::now();
    if (svc_->build_voice_prefix(l7(voice), err) != 0) return -1;
    prefix_build_ms_ = prefix_build_ms_.load() + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int took = std::max(0, free0 - svc_->pool().pages_free());
    prefix_pages_ += took;
    voice_prefix_pages_[voice] = took;
    note_pool();
    built_[voice] = true;
    prefixes_built_++;
    return 0;
  }
  std::unique_ptr<render::Renderer> renderer_;
#endif
  std::unique_ptr<render::RenderService> owned_;
  render::RenderService* svc_ = nullptr;
  const char* name_;
  bool lazy_prefix_ = true;
  bool reserve_ = false;
  bool prewarm_ = false;
  int prewarm_rows_ = 20, prewarm_tokens_ = 320;
  std::atomic<int> prewarmed_voices_{0};
  std::atomic<long> prewarm_errors_{0};
  bool prewarm_tried_ = false;
  std::string prewarm_error_;  // conductor / create thread only (stats print the count)
  std::atomic<long> prewarm_keys_{0}, prewarm_captured_{0};
  std::atomic<double> prewarm_ms_{0};
  std::atomic<long> live_captures_{0}, failed_calls_{0};
  // Voice handles (engine side) -> L7 table index; -1 = lost in a failed re-add.
  int l7(int voice) const {
    auto it = l7_of_.find(voice);
    return it != l7_of_.end() ? it->second : -1;
  }
  int avail_pages() const { return svc_->pool().pages_total() - prefix_pages_ - reserved_; }
  void forget_prefix(int voice) {
    auto it = voice_prefix_pages_.find(voice);
    if (it != voice_prefix_pages_.end()) {
      prefix_pages_ -= it->second;
      voice_prefix_pages_.erase(it);
    }
    built_.erase(voice);
  }
  struct VoiceData {
    std::vector<int32_t> toks;
    std::vector<float> feat, emb;
  };
  int next_handle_ = 0;
  std::unordered_map<int, int> l7_of_;
  std::unordered_map<int, VoiceData> vdata_;   // real renderer only: the add_voice inputs, for a re-add
  std::unordered_map<int, uint64_t> last_use_;  // handle -> open_row tick (LRU for prefix eviction)
  uint64_t use_tick_ = 0;
  std::atomic<long> prefixes_evicted_{0}, readd_failures_{0};
  std::atomic<int> create_lazy_fallback_{0};  // voices left lazy at create because the pool could not hold their prefix
  std::unordered_map<int, int> prompt_toks_;
  std::unordered_map<int, bool> built_;
  std::unordered_map<int, int> voice_prefix_pages_;  // pool pages counted in prefix_pages_ per voice (remove_voice)
  std::unordered_map<uint64_t, int> row_need_;
  struct RowBudget {
    int voice = -1, budget = 0;
    bool demo = false;
  };
  std::unordered_map<uint64_t, RowBudget> row_voice_budget_;
  std::unordered_map<uint64_t, std::vector<uint64_t>> call_rows_;  // call id -> its rows (in flight)
  std::unordered_map<uint64_t, int> row_calls_;                    // row -> its calls in flight
  std::unordered_map<uint64_t, int> closed_pending_;               // closed row -> pages to release at its last call
  std::atomic<int> reserved_{0};
  std::atomic<int> prefix_pages_{0};
  std::atomic<int> prefixes_built_{0};
  std::atomic<long> voices_removed_{0};
  std::atomic<double> prefix_build_ms_{0};
  std::atomic<long> backpressure_{0};
  std::atomic<long> extension_overcommit_{0};
  std::atomic<int> pool_total_{0}, pool_free_{0}, pool_free_min_{1 << 30};
  void note_pool() {
    const int f = svc_->pool().pages_free();
    pool_total_ = svc_->pool().pages_total();
    pool_free_ = f;
    if (f < pool_free_min_.load()) pool_free_min_ = f;
  }
};

render::RenderServiceConfig service_config(const L7RenderOptions& o) {
  render::RenderServiceConfig cfg;
  cfg.arena.device = o.device;
  cfg.arena.slots = o.slots;
  cfg.arena.pool_pages = o.pool_pages;
  cfg.arena.voices = o.voices;
  cfg.caller.device = o.device;
  cfg.caller.max_rows = std::max(1, std::min(64, o.max_rows));
  cfg.caller.workspace_bytes = size_t(std::max(64, o.workspace_mb)) << 20;
  cfg.timing = o.timing;
  return cfg;
}

}  // namespace

std::unique_ptr<RenderBackend> make_l7_standin_render(int device, int slots, int pool_pages, int voices,
                                                      std::string* err) {
  try {
    L7RenderOptions o;
    o.device = device;
    o.slots = slots;
    o.pool_pages = pool_pages;
    o.voices = voices;
    o.workspace_mb = 64;
    return std::make_unique<L7Render>(service_config(o), std::vector<render::RenderStage>{render::noise_gather_stage(),
                                                                                          render::stub_pcm_stage()},
                                      "l7-standin");
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
    return nullptr;
  }
}

std::unique_ptr<RenderBackend> make_l7_render(const L7RenderOptions& o, std::string* err) {
#if KT2N_ENGINE_WITH_RENDERER
  try {
    render::RendererConfig rc;
    rc.service = service_config(o);
    rc.build_prefixes = false;  // the adapter builds them (at create, or on first use)
    if (o.hift_max_frame_rows > 0) rc.hift_max_frame_rows = o.hift_max_frame_rows;
    if (o.hift_bulk_max_t > 0) rc.hift_bulk_max_t = o.hift_bulk_max_t;
    if (o.hift_bulk_max_frame_rows > 0) rc.hift_bulk_max_frame_rows = o.hift_bulk_max_frame_rows;
    rc.hift_prewarm_rows = std::min(rc.hift_prewarm_rows, std::max(1, o.max_rows));
    auto r = render::Renderer::create(o.s3gen_path, rc, err);
    if (!r) return nullptr;
    return std::make_unique<L7Render>(std::move(r), o);
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
    return nullptr;
  }
#else
  (void)o;
  if (err) *err = "built without L7's renderer (kt2n_render_flow: L4/L5/L6)";
  return nullptr;
#endif
}

}  // namespace kt2n::engine
