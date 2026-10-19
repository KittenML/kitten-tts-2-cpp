// kt2n::lm engine: torch-free native LM (lane L3, 10 Oct 2026).  See include/kt2n/lm.h.
#include "kt2n/lm.h"
#include "kt2n/cuda_caps.h"

#include <cuda_runtime.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "kt2n/engine_file.h"
#include "kt2n/kernels/ops.h"
#include "kt2n/kernels/samp.h"
#include "kt2n/kernels/step.h"
#include "kt2n/kernels/step_o1.h"
#include "kt2n/prefill.h"
#include "lm_device.cuh"

namespace kt2n::lm {

// ------------------------------------------------------------------------------------------------ sampler surface
SamplerSettings preset(SamplerMode mode) {
  SamplerSettings s;
  switch (mode) {
    case SamplerMode::kRelease: s.temperature = 0.7f; s.top_p = 0.9f; s.top_k = 100; s.min_p = 0.05f; break;
    case SamplerMode::kStable: s.temperature = 0.8f; s.top_p = 0.8f; s.top_k = 50; s.min_p = 0.0f; break;
    case SamplerMode::kExpressive:
    case SamplerMode::kAdvanced: s.temperature = 0.9f; s.top_p = 0.9f; s.top_k = 50; s.min_p = 0.0f; break;
  }
  return s;
}

namespace {

std::string py_float(double v) {
  // Python's str(float) for the range ends used in the messages (0.3, 2.0, 0.5, 1.0, 0.0).
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.1f", v);
  return buf;
}

bool check_range(const char* name, double v, double lo, double hi, bool integer, std::string* msg) {
  const bool ok = std::isfinite(v) && v >= lo && v <= hi && (!integer || v == std::floor(v));
  if (!ok) {
    if (integer) {
      *msg = std::string(name) + " must be an integer between " + std::to_string(static_cast<long long>(lo)) +
             " and " + std::to_string(static_cast<long long>(hi)) + ".";
    } else {
      *msg = std::string(name) + " must be a number between " + py_float(lo) + " and " + py_float(hi) + ".";
    }
  }
  return ok;
}

}  // namespace

std::string resolve_sampler(const SamplerRequest& req, SamplerMode no_settings_default, SamplerSettings* out,
                            std::string* param) {
  std::string msg;
  auto fail = [&](const char* p) {
    if (param) *param = p;
    return msg;
  };
  SamplerMode base = no_settings_default;
  if (req.mode) {
    std::string m = *req.mode;
    m.erase(0, m.find_first_not_of(" \t\r\n"));
    m.erase(m.find_last_not_of(" \t\r\n") + 1);
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (m == "stable") {
      base = SamplerMode::kStable;
    } else if (m == "expressive") {
      base = SamplerMode::kExpressive;
    } else {
      msg = "mode must be one of: stable, expressive.";
      return fail("mode");
    }
  } else if (req.temperature || req.top_p || req.top_k || req.min_p) {
    base = SamplerMode::kExpressive;  // DEFAULT_MODE (g8.3c)
  }
  if (req.temperature && !check_range("temperature", *req.temperature, 0.3, 2.0, false, &msg)) return fail("temperature");
  if (req.top_p && !check_range("top_p", *req.top_p, 0.5, 1.0, false, &msg)) return fail("top_p");
  if (req.top_k && !check_range("top_k", *req.top_k, 0, 200, true, &msg)) return fail("top_k");
  if (req.min_p && !check_range("min_p", *req.min_p, 0.0, 0.5, false, &msg)) return fail("min_p");
  if (req.max_new_tokens && !check_range("max_new_tokens", *req.max_new_tokens, 500, 12000, true, &msg)) {
    return fail("max_new_tokens");
  }
  SamplerSettings s = preset(base);
  if (req.temperature) s.temperature = static_cast<float>(*req.temperature);
  if (req.top_p) s.top_p = static_cast<float>(*req.top_p);
  if (req.top_k) s.top_k = static_cast<int32_t>(*req.top_k);
  if (req.min_p) s.min_p = static_cast<float>(*req.min_p);
  s.max_new_tokens = req.max_new_tokens ? static_cast<int32_t>(*req.max_new_tokens) : 0;
  if (out) *out = s;
  return "";
}

const char* finish_name(Finish f) {
  switch (f) {
    case Finish::kNone: return "none";
    case Finish::kEos: return "eos";
    case Finish::kStop: return "stop";
    case Finish::kBudget: return "budget";
    case Finish::kMaxNewTokens: return "max_new_tokens";
    case Finish::kCeiling: return "ceiling";
    case Finish::kCancelled: return "cancelled";
  }
  return "?";
}

namespace {

// SUB_GATE_AUDIO_IDS (runtime.py @ ebcdd3f, pinned to S3Gen d65cb687...): audio ids in the codec's quiet corner.
constexpr int kSubGateIds[] = {
    331,  729,  735,  736,  770,  979,  1458, 1463, 1465, 1466, 1490, 1701, 1704, 1788, 1789, 1809, 1947, 1950, 2005,
    2028, 2031, 2109, 2112, 2139, 2933, 2948, 2957, 2960, 3003, 3246, 3247, 3408, 3653, 3662, 3676, 3677, 3680, 3685,
    3704, 3713, 3729, 3732, 3888, 3891, 3894, 3945, 3969, 3972, 3975, 3976, 4053, 4056, 4057, 4134, 4137, 4138, 4215,
    4216, 4218, 4219, 4227, 4296, 4299, 4300, 4308, 5109, 5190, 5352, 5891, 5900, 5919, 6078, 6081, 6082, 6159, 6162,
    6163, 6168, 6243, 6252, 6321, 6324, 6325, 6333, 6399, 6402, 6405, 6406, 6414, 6483, 6486, 6487, 6495,
};

// runtime.sampler_row: {t, float32(1)/float32(t), k, float32(1 - top_p) in double (-1 when top_p >= 1), min_p}.
void sampler_row(const SamplerSettings& s, float out[5]) {
  const float t = s.temperature;
  out[0] = t;
  out[1] = 1.0f / t;
  out[2] = static_cast<float>(s.top_k);
  const double top_p = static_cast<double>(s.top_p);
  out[3] = top_p < 1.0 ? static_cast<float>(1.0 - top_p) : -1.0f;
  out[4] = s.min_p;
}

constexpr int kH = 2048;
constexpr int kAudioN = KT2N_SAMP_AUDIO_N;
constexpr size_t kAlign = 256;
constexpr int kRing = 16;

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

#define LM_CUDA(expr)                                                                         \
  do {                                                                                        \
    const cudaError_t e_ = (expr);                                                            \
    if (e_ != cudaSuccess) {                                                                  \
      if (err) *err = std::string(#expr) + ": " + cudaGetErrorString(e_);                     \
      return -1;                                                                              \
    }                                                                                         \
  } while (0)

struct HostRow {
  uint64_t tag = 0;
  uint64_t gen = 0;        // admission generation: a slot reused by a later row gets a new one
  int slot = -1;
  bool finished = false;   // a finish event was emitted (device knows too)
  bool cancelled = false;  // cancel requested; a kCancelled event is emitted at the next pump
  bool cancel_emitted = false;
  int prompt_len = 0;
};

struct Pending {
  RowRequest req;
};

// An in-flight launch whose ring entry the host has not read yet.
struct Inflight {
  uint32_t seq;                 // the post launch's sequence number
  std::vector<int> slots;       // row r -> slot (padding rows absent)
  std::vector<uint64_t> gens;   // row r -> the slot's admission generation at launch (slots are reused)
  bool admission = false;
  std::chrono::steady_clock::time_point t0;
  int bucket = 0;
};

struct Voice {
  size_t off = 0;    // byte offset in the voice store
  size_t bytes = 0;
  int len = 0;       // prefix tokens
  bool live = false;      // new rows may use it
  bool released = false;  // release_voice(): give the range back once refs == 0
  bool freed = false;     // the range is back in the free list
  int refs = 0;           // rows that read it: pending, admitted, or finished but their slot not released yet
};

class EngineImpl final : public Engine {
 public:
  ~EngineImpl() override { release(); }

  int init(const Config& cfg, std::string* err);

  int project_speaker(const float* speaker, int dim, std::vector<uint16_t>* out_bf16, std::string* err) override;
  int build_voice(const std::vector<uint16_t>& speaker_bf16, const std::vector<int64_t>& prefix,
                  std::string* err) override;
  int voice_prefix_len(int voice) const override {
    std::lock_guard<std::mutex> lk(mu_);
    return voice_len_locked(voice);
  }
  void drop_voice(int voice) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (voice >= 0 && voice < static_cast<int>(voices_.size())) voices_[voice].live = false;
  }
  int release_voice(int voice) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (voice < 0 || voice >= static_cast<int>(voices_.size()) || voices_[voice].released) return -1;
    Voice& v = voices_[voice];
    v.live = false;
    v.released = true;
    if (v.refs == 0) free_voice_locked(voice);
    return 0;
  }
  int submit(const RowRequest& row, std::string* err) override;
  void cancel(uint64_t tag) override {
    std::lock_guard<std::mutex> lk(mu_);
    cancels_.push_back(tag);
  }
  int pump(std::vector<TokenEvent>* events, bool block, std::string* err) override;
  int set_decode_priority(int level) override;
  int live_rows() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(pending_.size()) + live_count_.load();
  }
  Stats stats() const override {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = stats_;
    s.pending_rows = static_cast<int>(pending_.size());
    s.active_rows = live_count_.load();
    for (const Voice& v : voices_) s.voices_live += v.live ? 1 : 0;
    s.voice_store_used = voice_used_;
    for (const auto& r : vfree_) s.voice_store_free_max = std::max(s.voice_store_free_max, r.second);
    return s;
  }
  void debug_dump() const override {
    std::fprintf(stderr, "kt2n::lm rows: active %zu inflight %zu adm %d pending_free %zu free %zu next_seq %u bucket %d\n",
                 active_.size(), inflight_.size(), adm_inflight_ ? 1 : 0, pending_free_.size(), free_slots_.size(),
                 next_seq_, cur_bucket_);
    for (int s : active_) {
      const HostRow& h = slot_rows_[s];
      std::fprintf(stderr, "  slot %d tag %llu gen %llu finished %d cancelled %d prompt %d\n", s,
                   static_cast<unsigned long long>(h.tag), static_cast<unsigned long long>(h.gen), h.finished ? 1 : 0,
                   h.cancelled ? 1 : 0, h.prompt_len);
    }
    if (!inflight_.empty()) {
      const Inflight& f = inflight_.front();
      std::fprintf(stderr, "  front record seq %u admission %d rows %zu ring seq %u\n", f.seq, f.admission ? 1 : 0,
                   f.slots.size(), ring_host_[f.seq % kRing].seq);
    }
  }
  bool debug_logits(uint64_t tag, int index, std::vector<uint16_t>* out) const override {
    auto it = dbg_.find({tag, index});
    if (it == dbg_.end()) return false;
    *out = it->second;
    return true;
  }

 private:
  void release();
  int bucket_for(int n) const {
    for (int b : buckets_)
      if (b >= n) return b;
    return -1;
  }
  int capture_graphs(std::string* err);
  int enqueue_step(int bucket, std::string* err);
  int admit(std::vector<Pending>& batch, std::string* err);
  int join_admission(std::vector<TokenEvent>* events, std::string* err);
  void release_slots();
  int quiesce_admission(std::string* err);
  int rebucket(std::string* err);
  int collect(std::vector<TokenEvent>* events, bool block, std::string* err);
  int wait_seq(const Inflight& f, bool block);
  // Voice store (first-fit free list over [0, voice_store_bytes)); *_locked: caller holds mu_.
  int voice_len_locked(int voice) const {
    return (voice >= 0 && voice < static_cast<int>(voices_.size()) && voices_[voice].live) ? voices_[voice].len : -1;
  }
  bool valloc_locked(size_t bytes, size_t* off);
  void vfree_locked(size_t off, size_t bytes);
  void free_voice_locked(int voice) {
    Voice& v = voices_[voice];
    if (v.freed) return;
    vfree_locked(v.off, v.bytes);
    v.freed = true;
  }
  void unref_voice_locked(int voice) {
    if (voice < 0) return;
    Voice& v = voices_[voice];
    if (--v.refs == 0 && v.released) free_voice_locked(voice);
  }

  Config cfg_;
  int slots_ = 0;  // max_rows + 1 (scratch)
  int scratch_ = 0;
  cudaStream_t stream_ = nullptr;     // decode (and the admission join: ring posts stay in order)
  // Sprint 6: runtime decode priority (set_decode_priority).  One decode stream per level, created on first use;
  // stream_ is the current one.  A switch makes the new stream wait for the old one's work (prio_ev_) before any
  // later decode work, so everything on the decode side stays in order.
  static constexpr int kPrioLevels = 8;
  cudaStream_t prio_stream_[kPrioLevels] = {};
  int prio_level_ = 0;
  int prio_lo_ = 0, prio_hi_ = 0;
  std::atomic<int> want_level_{-1};
  cudaEvent_t prio_ev_ = nullptr;
  int apply_decode_priority(int level, std::string* err);
  cudaStream_t pf_stream_ = nullptr;  // admissions (prefill + first-token sample), overlapping decode
  // split_min_rows: the extra branches of a split decode step (branch 0 runs on stream_), forked from stream_ by
  // fork_ev_ and joined back by bjoin_ev_[i] before the ring post; each with its own scratch / attention workspace.
  static constexpr int kMaxBranches = 8;
  int nbr_ = 1;  // branches allocated (1 = no split)
  cudaStream_t bstream_[kMaxBranches] = {};
  cudaEvent_t bjoin_ev_[kMaxBranches] = {};
  void* bscratch_[kMaxBranches] = {};
  void* bws_[kMaxBranches] = {};
  size_t bscratch_bytes_ = 0;  // sized for the largest extra branch (bmax rows)
  size_t bws_bytes_ = 0;
  cudaEvent_t fork_ev_ = nullptr;
  kt2n_pf_model* model_ = nullptr;
  kt2n_pf_ctx* pf_ = nullptr;
  const kt2n_pf_weights* w_ = nullptr;
  std::vector<kt2n_step_layer> layers_;
  kt2n_step_model step_model_{};

  // arena
  void* arena_ = nullptr;
  size_t arena_bytes_ = 0;
  char* kv_ = nullptr;
  int64_t slot_elems_ = 0;
  void* scratch_buf_ = nullptr;
  size_t scratch_bytes_ = 0;
  void* attn_ws_ = nullptr;
  size_t attn_ws_bytes_ = 0;
  void* logits_ = nullptr;      // [64][6568] bf16
  void* adm_logits_ = nullptr;  // [16][6568] bf16
  void* speakers_dev_ = nullptr;  // [16][2048] bf16
  char* voice_store_ = nullptr;
  size_t voice_used_ = 0;              // bytes held by voices (mu_)
  std::map<size_t, size_t> vfree_;     // free ranges of the voice store: offset -> bytes (mu_)
  std::vector<int> slot_voice_;        // by slot: the voice its row reads (-1 none); unref at slot release
  void* spk_w_ = nullptr;       // spk_proj tensors (bf16)
  void* spk_b_ = nullptr;
  void* spk_lnw_ = nullptr;
  void* spk_lnb_ = nullptr;
  int spk_in_ = 0;
  float* spk_in_dev_ = nullptr;  // [spk_in] fp32 staging
  dev::State st_{};

  // host
  dev::RingEntry* ring_host_ = nullptr;  // mapped
  uint16_t* spk_stage_ = nullptr;       // pinned [16][2048]
  int32_t* forced_stage_ = nullptr;     // pinned
  uint16_t* dbg_host_ = nullptr;        // pinned [64][6568]
  uint16_t* dbg_adm_host_ = nullptr;    // pinned [16][6568]
  cudaEvent_t stage_ev_ = nullptr;
  cudaEvent_t adm_ev_ = nullptr;      // the in-flight admission's work on pf_stream_ is done
  cudaEvent_t join_ev_ = nullptr;     // the last join's post (reads adm_slot / adm_tok) ran on the decode stream
  bool adm_inflight_ = false;
  bool joining_ = false;              // inside join_admission (join_poll off for its waits)
  Inflight adm_rec_;                  // its rows (ring post at the join)
  std::vector<std::pair<int, uint32_t>> pending_free_;  // (slot, every ring entry with seq < this must be read)
  std::vector<cudaGraphExec_t> graphs_;  // by bucket index
  std::vector<int> buckets_;
  std::vector<Voice> voices_;

  uint32_t next_seq_ = 0;   // sequence number of the next post launch
  uint64_t next_gen_ = 1;   // admission generations
  std::deque<Inflight> inflight_;
  std::vector<HostRow> slot_rows_;  // by slot
  std::vector<int> free_slots_;
  std::vector<int> active_;         // slots in bucket order
  bool rows_dirty_ = false;
  int cur_bucket_ = 0;
  std::unordered_map<uint64_t, int> tag_slot_;
  std::map<std::pair<uint64_t, int>, std::vector<uint16_t>> dbg_;

  mutable std::mutex mu_;
  std::deque<Pending> pending_;
  std::vector<uint64_t> cancels_;
  std::atomic<int> live_count_{0};
  Stats stats_;
  bool broken_ = false;
};

void EngineImpl::release() {
  if (stream_) cudaStreamSynchronize(stream_);
  for (auto g : graphs_)
    if (g) cudaGraphExecDestroy(g);
  graphs_.clear();
  if (pf_) kt2n_pf_ctx_free(pf_);
  pf_ = nullptr;
  if (model_) kt2n_pf_model_free(model_);
  model_ = nullptr;
  if (arena_) cudaFree(arena_);
  arena_ = nullptr;
  if (ring_host_) cudaFreeHost(ring_host_);
  ring_host_ = nullptr;
  if (spk_stage_) cudaFreeHost(spk_stage_);
  spk_stage_ = nullptr;
  if (forced_stage_) cudaFreeHost(forced_stage_);
  forced_stage_ = nullptr;
  if (dbg_host_) cudaFreeHost(dbg_host_);
  dbg_host_ = nullptr;
  if (dbg_adm_host_) cudaFreeHost(dbg_adm_host_);
  dbg_adm_host_ = nullptr;
  if (pf_stream_) cudaStreamSynchronize(pf_stream_);
  if (stage_ev_) cudaEventDestroy(stage_ev_);
  stage_ev_ = nullptr;
  if (adm_ev_) cudaEventDestroy(adm_ev_);
  adm_ev_ = nullptr;
  if (join_ev_) cudaEventDestroy(join_ev_);
  join_ev_ = nullptr;
  for (int i = 0; i < kPrioLevels; ++i) {  // sprint 6: decode streams of other priority levels (stream_ is one of them)
    if (prio_stream_[i] && prio_stream_[i] != stream_) {
      cudaStreamSynchronize(prio_stream_[i]);
      cudaStreamDestroy(prio_stream_[i]);
    }
    prio_stream_[i] = nullptr;
  }
  if (prio_ev_) cudaEventDestroy(prio_ev_);
  prio_ev_ = nullptr;
  if (stream_) cudaStreamDestroy(stream_);
  stream_ = nullptr;
  if (pf_stream_) cudaStreamDestroy(pf_stream_);
  pf_stream_ = nullptr;
  if (fork_ev_) cudaEventDestroy(fork_ev_);
  fork_ev_ = nullptr;
  for (int i = 1; i < kMaxBranches; ++i) {
    if (bstream_[i]) cudaStreamSynchronize(bstream_[i]);
    if (bjoin_ev_[i]) cudaEventDestroy(bjoin_ev_[i]);
    bjoin_ev_[i] = nullptr;
    if (bstream_[i]) cudaStreamDestroy(bstream_[i]);
    bstream_[i] = nullptr;
  }
  nbr_ = 1;
}

int EngineImpl::init(const Config& cfg, std::string* err) {
  cfg_ = cfg;
  if (cfg_.max_rows < 1 || cfg_.max_rows > kMaxRows) {
    *err = "max_rows must be 1..64";
    return -1;
  }
  if (cfg_.max_ctx < 64 || cfg_.max_ctx % 64) {
    *err = "max_ctx must be a positive multiple of 64";
    return -1;
  }
  if (cfg_.run_ahead < 1) cfg_.run_ahead = 1;
  if (cfg_.debug_logits) cfg_.run_ahead = 1;  // one fixed dump buffer
  if (cfg_.run_ahead > kRing - 4) cfg_.run_ahead = kRing - 4;
  slots_ = cfg_.max_rows + 1;
  scratch_ = cfg_.max_rows;
  LM_CUDA(cudaSetDevice(cfg_.device));
  cfg_.pdl = cfg_.pdl && kt2n::cuda::device_caps(cfg_.device).pdl();
  {
    int lo = 0, hi = 0;  // lower number = higher priority: hi <= lo
    LM_CUDA(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    // Sprint 6: lab A/B knob for engine-level runs without a conductor change: KT2N_LM_DECODE_PRIORITY=<level>
    // (process env) overrides Config::decode_priority_level.  Unset = the Config value.  Execution only.
    if (const char* v = std::getenv("KT2N_LM_DECODE_PRIORITY"); v && *v) cfg_.decode_priority_level = std::atoi(v);
    const int want = lo - std::max(0, cfg_.decode_priority_level);
    const int prio = std::max(want, hi < lo ? hi + 1 : hi);  // never the top level (urgent renders keep it)
    LM_CUDA(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, prio));
    prio_lo_ = lo;
    prio_hi_ = hi;
    prio_level_ = std::min(kPrioLevels - 1, lo - prio);
    prio_stream_[prio_level_] = stream_;
    LM_CUDA(cudaEventCreateWithFlags(&prio_ev_, cudaEventDisableTiming));
    stats_.decode_priority = prio;
    stats_.priority_lo = lo;
    stats_.priority_hi = hi;
  }
  {
    // Admissions on the highest-priority stream: the block scheduler prefers its CTAs, so a newcomer's prefill is not
    // slowed by the decode steps it overlaps (lower number = higher priority).
    int lo = 0, hi = 0;
    LM_CUDA(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    LM_CUDA(cudaStreamCreateWithPriority(&pf_stream_, cudaStreamNonBlocking, cfg_.admission_priority ? hi : lo));
  }
  nbr_ = 1;
  if (cfg_.split_min_rows > 0 && cfg_.split_max_rows >= cfg_.split_min_rows && cfg_.split_max_rows >= 2) {
    const int top = std::min(cfg_.max_rows, cfg_.split_max_rows);
    nbr_ = cfg_.split_branch_rows > 0 && cfg_.split_first_rows <= 0
               ? (top + cfg_.split_branch_rows - 1) / cfg_.split_branch_rows
               : 2;
    nbr_ = std::max(2, std::min(nbr_, kMaxBranches));
    LM_CUDA(cudaEventCreateWithFlags(&fork_ev_, cudaEventDisableTiming));
    for (int i = 1; i < nbr_; ++i) {
      LM_CUDA(cudaStreamCreateWithPriority(&bstream_[i], cudaStreamNonBlocking, stats_.decode_priority));
      LM_CUDA(cudaEventCreateWithFlags(&bjoin_ev_[i], cudaEventDisableTiming));
    }
  }

  // weights (one allocation, kernel layout)
  if (int rc = kt2n_pf_model_load(cfg_.engine_path.c_str(), cfg_.device,
                                  cfg_.expect_source_sha256.empty() ? nullptr : cfg_.expect_source_sha256.c_str(),
                                  &model_)) {
    *err = std::string("kt2n_pf_model_load: ") + kt2n_pf_last_error() + " (" + std::to_string(rc) + ")";
    return -1;
  }
  w_ = kt2n_pf_model_weights(model_);
  stats_.weight_bytes = kt2n_pf_model_bytes(model_);
  const int L = w_->layers;

  // spk_proj from the engine file (dense bf16)
  std::vector<std::pair<const char*, void**>> spk = {{"spk_proj.0.weight", &spk_w_},
                                                     {"spk_proj.0.bias", &spk_b_},
                                                     {"spk_proj.1.weight", &spk_lnw_},
                                                     {"spk_proj.1.bias", &spk_lnb_}};
  std::vector<std::pair<std::vector<uint8_t>, void**>> spk_bytes;
  size_t spk_total = 0;
  try {
    EngineFile ef = EngineFile::open(cfg_.engine_path);
    for (auto& [name, dst] : spk) {
      const TensorInfo* t = ef.find(name);
      if (!t || t->kind != TensorKind::kDense) throw FormatError(std::string("missing dense tensor ") + name);
      const auto segs = t->segments();
      if (segs.empty()) throw FormatError(std::string("no data segment for ") + name);
      auto b = ef.bytes(*segs[0].second);
      if (std::string(name) == "spk_proj.0.weight") spk_in_ = static_cast<int>(t->cols());
      spk_bytes.emplace_back(std::vector<uint8_t>(b.begin(), b.end()), dst);
      spk_total += align_up(b.size());
    }
  } catch (const std::exception& e) {
    *err = std::string("engine file: ") + e.what();
    return -1;
  }

  // arena layout
  slot_elems_ = static_cast<int64_t>(2) * L * 8 * cfg_.max_ctx * 128;
  const size_t kv_bytes = static_cast<size_t>(slot_elems_) * 2 * slots_;
  scratch_bytes_ = kt2n_step_scratch_bytes(kMaxRows);
  attn_ws_bytes_ = kt2n_step_attn_ws_bytes(kMaxRows, 2 * cfg_.max_ctx);  // room for 64-key splits
  const int fcap = 4096;
  struct Part {
    size_t bytes;
    void** dst;
  };
  int64_t *tok, *pos, *row_tok, *row_pos;
  int32_t *new_tokens, *silent, *budget, *cap, *prompt_len, *forced_len, *forced, *history, *history_len, *row_slot,
      *samp_tok, *adm_slot, *adm_tok;
  uint32_t *flags, *silent_bits, *seq;
  float* params;
  uint64_t *seeds, *offsets;
  int64_t *voff, *row_voff;
  int32_t *vlen, *row_vlen;
  void* kvp = nullptr;
  void* vs = nullptr;
  std::vector<Part> parts = {
      {kv_bytes, &kvp},
      {scratch_bytes_, &scratch_buf_},
      {attn_ws_bytes_, &attn_ws_},
      {static_cast<size_t>(kMaxRows) * kAudioN * 2, &logits_},
      {static_cast<size_t>(kMaxPrefillRows) * kAudioN * 2, &adm_logits_},
      {static_cast<size_t>(kMaxPrefillRows) * kH * 2, &speakers_dev_},
      {cfg_.voice_store_bytes, &vs},
      {sizeof(int64_t) * slots_, reinterpret_cast<void**>(&tok)},
      {sizeof(int64_t) * slots_, reinterpret_cast<void**>(&pos)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&new_tokens)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&silent)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&budget)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&cap)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&prompt_len)},
      {sizeof(uint32_t) * slots_, reinterpret_cast<void**>(&flags)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&forced_len)},
      {sizeof(int32_t) * slots_ * fcap, reinterpret_cast<void**>(&forced)},
      {sizeof(float) * slots_ * 5, reinterpret_cast<void**>(&params)},
      {sizeof(int32_t) * slots_ * 50, reinterpret_cast<void**>(&history)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&history_len)},
      {sizeof(uint64_t) * slots_, reinterpret_cast<void**>(&seeds)},
      {sizeof(uint64_t) * slots_, reinterpret_cast<void**>(&offsets)},
      {sizeof(int64_t) * kMaxRows, reinterpret_cast<void**>(&row_tok)},
      {sizeof(int64_t) * kMaxRows, reinterpret_cast<void**>(&row_pos)},
      {sizeof(int32_t) * kMaxRows, reinterpret_cast<void**>(&row_slot)},
      {sizeof(int32_t) * kMaxRows, reinterpret_cast<void**>(&samp_tok)},
      {sizeof(int32_t) * kMaxPrefillRows, reinterpret_cast<void**>(&adm_slot)},
      {sizeof(int32_t) * kMaxPrefillRows, reinterpret_cast<void**>(&adm_tok)},
      {sizeof(uint32_t) * dev::kSilentWords, reinterpret_cast<void**>(&silent_bits)},
      {sizeof(uint32_t), reinterpret_cast<void**>(&seq)},
      {sizeof(float) * 4096, reinterpret_cast<void**>(&spk_in_dev_)},
      {sizeof(int64_t) * slots_, reinterpret_cast<void**>(&voff)},
      {sizeof(int32_t) * slots_, reinterpret_cast<void**>(&vlen)},
      {sizeof(int64_t) * kMaxRows, reinterpret_cast<void**>(&row_voff)},
      {sizeof(int32_t) * kMaxRows, reinterpret_cast<void**>(&row_vlen)},
  };
  if (nbr_ > 1) {  // extra branches never carry more than bmax rows (branch 0 uses the main buffers)
    const int top = std::min(cfg_.max_rows, cfg_.split_max_rows);
    int bmax = std::min(kMaxRows, cfg_.split_branch_rows > 0 ? cfg_.split_branch_rows : (top + 1) / 2);
    // split_first_rows: branch 1 carries B - min(F, B - 1) rows, largest at the top bucket
    if (cfg_.split_first_rows > 0) bmax = std::min(kMaxRows, std::max(bmax, top - cfg_.split_first_rows));
    bscratch_bytes_ = kt2n_step_scratch_bytes(bmax);
    // keys: as the main workspace (2 x max_ctx, room for 64-key splits), and at least 2,048 so a branch of <= 2 rows
    // (64-key splits up to 640 keys: 10 splits of the launcher's 128-key grid unit) fits even when max_ctx is tiny.
    bws_bytes_ = kt2n_step_attn_ws_bytes(bmax, std::max(2 * cfg_.max_ctx, 2048));
  }
  for (int i = 1; i < nbr_; ++i) {
    parts.push_back({bscratch_bytes_, &bscratch_[i]});
    parts.push_back({bws_bytes_, &bws_[i]});
  }
  for (auto& sb : spk_bytes) parts.push_back({sb.first.size(), sb.second});
  size_t total = 0;
  for (auto& p : parts) total += align_up(p.bytes);
  LM_CUDA(cudaMalloc(&arena_, total));
  arena_bytes_ = total;
  stats_.arena_bytes = total;
  LM_CUDA(cudaMemsetAsync(arena_, 0, total, stream_));
  {
    char* p = static_cast<char*>(arena_);
    for (auto& part : parts) {
      *part.dst = p;
      p += align_up(part.bytes);
    }
  }
  kv_ = static_cast<char*>(kvp);
  voice_store_ = static_cast<char*>(vs);
  vfree_.clear();
  vfree_[0] = cfg_.voice_store_bytes;
  voice_used_ = 0;
  for (auto& sb : spk_bytes) LM_CUDA(cudaMemcpyAsync(*sb.second, sb.first.data(), sb.first.size(), cudaMemcpyHostToDevice, stream_));
  {
    std::vector<uint32_t> bits(dev::kSilentWords, 0u);
    for (int id : kSubGateIds) bits[id >> 5] |= 1u << (id & 31);
    LM_CUDA(cudaMemcpyAsync(silent_bits, bits.data(), bits.size() * 4, cudaMemcpyHostToDevice, stream_));
    const uint32_t done = dev::kDone;
    LM_CUDA(cudaMemcpyAsync(flags + scratch_, &done, 4, cudaMemcpyHostToDevice, stream_));
    LM_CUDA(cudaMemsetAsync(voff, 0xff, sizeof(int64_t) * slots_, stream_));  // -1: no voice (incl. scratch)
    LM_CUDA(cudaMemsetAsync(row_voff, 0xff, sizeof(int64_t) * kMaxRows, stream_));
    LM_CUDA(cudaMemsetAsync(row_vlen, 0, sizeof(int32_t) * kMaxRows, stream_));
  }
  if (int rc = kt2n_step_attn_ws_init(attn_ws_, attn_ws_bytes_, stream_)) {
    *err = "kt2n_step_attn_ws_init failed: " + std::to_string(rc);
    return -1;
  }

  // mapped ring + pinned staging
  LM_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&ring_host_), sizeof(dev::RingEntry) * kRing, cudaHostAllocMapped));
  std::memset(ring_host_, 0, sizeof(dev::RingEntry) * kRing);
  dev::RingEntry* ring_dev = nullptr;
  LM_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ring_dev), ring_host_, 0));
  LM_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&spk_stage_), kMaxPrefillRows * kH * 2, cudaHostAllocDefault));
  LM_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&forced_stage_), sizeof(int32_t) * fcap, cudaHostAllocDefault));
  if (cfg_.debug_logits) {
    LM_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&dbg_host_), kMaxRows * kAudioN * 2, cudaHostAllocDefault));
    LM_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&dbg_adm_host_), kMaxPrefillRows * kAudioN * 2, cudaHostAllocDefault));
  }
  LM_CUDA(cudaEventCreateWithFlags(&stage_ev_, cudaEventDisableTiming));
  LM_CUDA(cudaEventCreateWithFlags(&adm_ev_, cudaEventDisableTiming));
  LM_CUDA(cudaEventCreateWithFlags(&join_ev_, cudaEventDisableTiming));

  st_ = dev::State{tok,        pos,        new_tokens, silent,     budget,      cap,     prompt_len, flags,
                   forced_len, forced,     fcap,       params,     history,     history_len, seeds,  offsets,
                   row_tok,    row_pos,    row_slot,   samp_tok,   adm_slot,    adm_tok, silent_bits, seq,
                   ring_dev,   kRing,      cfg_.max_ctx, cfg_.ctx_limit, cfg_.tagged_slot_margin, cfg_.ceiling_factor,
                   cfg_.silence_armed ? 1 : 0, voff, vlen, row_voff, row_vlen};

  // prefill context (its own workspace, allocated once here)
  if (int rc = kt2n_pf_ctx_create(w_, cfg_.device, kMaxPrefillRows, cfg_.max_prefill_tokens, cfg_.max_ctx, &pf_)) {
    *err = std::string("kt2n_pf_ctx_create: ") + kt2n_pf_last_error() + " (" + std::to_string(rc) + ")";
    return -1;
  }

  // decode step model over the slot arena (flat-row layout)
  layers_.resize(L);
  const int64_t plane = static_cast<int64_t>(8) * cfg_.max_ctx * 128;  // one layer's K (or V) of one slot
  for (int l = 0; l < L; ++l) {
    const kt2n_pf_layer& s = w_->layer[l];
    kt2n_step_layer& d = layers_[l];
    d.qkv_codes = s.qkv_codes;
    d.qkv_scales = s.qkv_scales;
    d.o_codes = s.o_codes;
    d.o_scales = s.o_scales;
    d.gate_up_codes = s.gu_codes;
    d.gate_up_scales = s.gu_scales;
    d.down_codes = s.down_codes;
    d.down_scales = s.down_scales;
    d.ln_in = s.input_norm;
    d.ln_post = s.post_norm;
    d.q_norm = s.q_norm;
    d.k_norm = s.k_norm;
    d.k_cache = kv_ + static_cast<size_t>(l * plane) * 2;
    d.v_cache = kv_ + static_cast<size_t>((L + l) * plane) * 2;
  }
  step_model_.abi = KT2N_STEP_ABI;
  step_model_.layers = L;
  step_model_.vocab = w_->vocab;
  step_model_.rms_eps = w_->rms_eps;
  step_model_.head_n = kAudioN;
  step_model_.embed = w_->embed;
  step_model_.final_norm = w_->final_norm;
  step_model_.head_w = static_cast<const char*>(w_->embed) + static_cast<size_t>(KT2N_SAMP_AUDIO_LOW) * kH * 2;
  step_model_.slots = slots_;
  step_model_.max_ctx = cfg_.max_ctx;
  step_model_.kv_slot_stride = slot_elems_;
  step_model_.kv_head_stride = static_cast<int64_t>(cfg_.max_ctx) * 128;
  step_model_.layer = layers_.data();

  // host row table
  slot_rows_.assign(slots_, HostRow{});
  slot_voice_.assign(slots_, -1);
  for (int s = cfg_.max_rows - 1; s >= 0; --s) free_slots_.push_back(s);
  std::vector<int> want = cfg_.buckets.empty() ? std::vector<int>(std::begin(kBuckets), std::end(kBuckets)) : cfg_.buckets;
  for (int b : want)
    if (b >= 1 && b <= cfg_.max_rows) buckets_.push_back(b);
  std::sort(buckets_.begin(), buckets_.end());
  buckets_.erase(std::unique(buckets_.begin(), buckets_.end()), buckets_.end());
  if (buckets_.empty() || buckets_.back() < cfg_.max_rows) buckets_.push_back(cfg_.max_rows);
  // Per-bucket step options (process-wide o1 table; baked into the graphs at capture): 64-key attention splits
  // for the small buckets.
  for (int r = 1; r <= KT2N_STEP_MAX_ROWS_EXT; ++r) {
    unsigned o = kt2n_step_o1_options_for(r) & ~(KT2N_STEP_O1_SPLIT64 | KT2N_STEP_O1_FUSE_NORM | KT2N_STEP_O1_KV_NEXT);
    if (r <= cfg_.split64_max_rows) o |= KT2N_STEP_O1_SPLIT64;
    if (r <= cfg_.fuse_norm_max_rows) o |= KT2N_STEP_O1_FUSE_NORM;
    if (cfg_.kv_next_min_rows > 0 && r >= cfg_.kv_next_min_rows) o |= KT2N_STEP_O1_KV_NEXT;
    kt2n_step_o1_set_bucket_options(r, o);
  }
  kt2n_step_o1_set_kv_next_budget(static_cast<unsigned long long>(std::max(0, cfg_.kv_next_budget_mb)) << 20);
  LM_CUDA(cudaStreamSynchronize(stream_));
  if (cfg_.graphs && capture_graphs(err)) return -1;
  return 0;
}

// The decode step of `bucket` rows + the post kernel (one graph per bucket, or eager when graphs are off).
int EngineImpl::enqueue_step(int bucket, std::string* err) {
  // split_min_rows: the bucket's rows run as nb branches of the step's graph (branch 0 on stream_, branch i on
  // bstream_[i]; two halves by default, or branches of at most split_branch_rows rows), forked from stream_ and joined
  // back before the ring post.  Every per-row array is offset by the branch's first row; each branch has its own
  // scratch and attention workspace; slots (K/V, sampler state) are disjoint, so nothing is shared but the weights.
  const bool split = nbr_ > 1 && bucket >= cfg_.split_min_rows && bucket <= cfg_.split_max_rows && bucket >= 2;
  int nb = 1;
  // split_first_rows: only when the second branch keeps >= 3 rows (a 1-2-row branch would run the 1-row CUDA-core
  // projections / 64-key attention splits, whose bits differ from the batched path); otherwise two halves.
  const int first = split && cfg_.split_first_rows > 0 && bucket - cfg_.split_first_rows >= 3 ? cfg_.split_first_rows : 0;
  if (split) {
    nb = cfg_.split_branch_rows > 0 && first == 0 ? (bucket + cfg_.split_branch_rows - 1) / cfg_.split_branch_rows : 2;
    nb = std::max(2, std::min(std::min(nb, nbr_), bucket));
  }
  auto step = [&](int r0, int n, void* scratch, size_t scratch_bytes, void* ws, size_t ws_bytes, cudaStream_t s) -> int {
    kt2n_step_io io{};
    io.rows = n;
    io.flags = KT2N_STEP_FLAG_SAMPLE | KT2N_STEP_FLAG_ROW_KV | (cfg_.pdl ? KT2N_STEP_FLAG_PDL : 0u);
    io.tokens = st_.row_tok + r0;
    io.positions = st_.row_pos + r0;
    io.cursor = st_.row_pos + r0;  // per-row cursor = logical position (flat-row layout)
    io.scratch = scratch;
    io.scratch_bytes = scratch_bytes;
    io.attn_ws = ws;
    io.attn_ws_bytes = ws_bytes;
    io.logits = static_cast<char*>(logits_) + static_cast<size_t>(r0) * kAudioN * 2;
    io.ld_logits = kAudioN;
    io.samp_params = st_.params;
    io.samp_history = st_.history;
    io.samp_history_len = st_.history_len;
    io.samp_seeds = st_.seeds;
    io.samp_offsets = st_.offsets;
    io.samp_slot_of_row = st_.row_slot + r0;
    io.samp_tokens = st_.samp_tok + r0;
    io.samp_flags = KT2N_SAMP_UPDATE_STATE;
    if (cfg_.share_voice_kv && n >= cfg_.share_voice_kv_min_rows) {  // prefix keys from the voice store (same values)
      io.voice_kv = voice_store_;
      io.voice_off = st_.row_voff + r0;  // per bucket row (rowmap), so the kernel needs no row_slot load first
      io.voice_len = st_.row_vlen + r0;
    }
    if (int rc = kt2n_step_forward_o1(&step_model_, &io, s)) {
      if (err) *err = std::string("kt2n_step_forward_o1: ") + kt2n_step_error_string(rc) + " (" + std::to_string(rc) + ")";
      return -1;
    }
    return 0;
  };
  if (nb > 1) {
    LM_CUDA(cudaEventRecord(fork_ev_, stream_));
    for (int i = 1; i < nb; ++i) LM_CUDA(cudaStreamWaitEvent(bstream_[i], fork_ev_, 0));
  }
  // rows split as evenly as possible: the first bucket % nb branches take one row more (split_first_rows: branch 0
  // takes `first` rows, branch 1 the rest)
  for (int i = 0, r0 = 0; i < nb; ++i) {
    const int n = first > 0 ? (i == 0 ? first : bucket - first) : bucket / nb + (i < bucket % nb ? 1 : 0);
    if (i == 0) {
      if (step(r0, n, scratch_buf_, scratch_bytes_, attn_ws_, attn_ws_bytes_, stream_)) return -1;
    } else {
      if (step(r0, n, bscratch_[i], bscratch_bytes_, bws_[i], bws_bytes_, bstream_[i])) return -1;
      LM_CUDA(cudaEventRecord(bjoin_ev_[i], bstream_[i]));
    }
    r0 += n;
  }
  for (int i = 1; i < nb; ++i) LM_CUDA(cudaStreamWaitEvent(stream_, bjoin_ev_[i], 0));
  if (cfg_.debug_logits) {
    LM_CUDA(cudaMemcpyAsync(dbg_host_, logits_, static_cast<size_t>(bucket) * kAudioN * 2, cudaMemcpyDeviceToHost,
                            stream_));
  }
  LM_CUDA(dev::launch_post(st_, bucket, st_.row_slot, st_.samp_tok, 1, stream_));
  return 0;
}

int EngineImpl::capture_graphs(std::string* err) {
  graphs_.assign(buckets_.size(), nullptr);
  for (size_t i = 0; i < buckets_.size(); ++i) {
    cudaGraph_t g = nullptr;
    LM_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    const int rc = enqueue_step(buckets_[i], err);
    const cudaError_t ce = cudaStreamEndCapture(stream_, &g);
    if (rc) {
      if (g) cudaGraphDestroy(g);
      return -1;
    }
    LM_CUDA(ce);
    const cudaError_t ie = cudaGraphInstantiate(&graphs_[i], g, 0);
    cudaGraphDestroy(g);
    LM_CUDA(ie);
  }
  // Warm every graph once with an all-padding bucket (loads modules; the scratch slot is always done).
  for (size_t i = 0; i < buckets_.size(); ++i) {
    dev::RowMap m{};
    m.n = 0;
    m.b = buckets_[i];
    m.scratch = scratch_;
    LM_CUDA(dev::launch_rowmap(st_, m, stream_));
    LM_CUDA(cudaGraphLaunch(graphs_[i], stream_));
    ++next_seq_;
  }
  LM_CUDA(cudaStreamSynchronize(stream_));
  return 0;
}

int EngineImpl::project_speaker(const float* speaker, int dim, std::vector<uint16_t>* out, std::string* err) {
  if (!speaker || dim != spk_in_ || dim > 4096) {
    if (err) *err = "speaker dim must be " + std::to_string(spk_in_);
    return -1;
  }
  if (quiesce_admission(err)) return -1;  // speakers_dev_ is shared with admissions
  LM_CUDA(cudaMemcpyAsync(spk_in_dev_, speaker, sizeof(float) * dim, cudaMemcpyHostToDevice, pf_stream_));
  auto* o = static_cast<__nv_bfloat16*>(speakers_dev_);
  LM_CUDA(kt2n_ops_spk_proj(spk_in_dev_, 1, dim, static_cast<const __nv_bfloat16*>(spk_w_),
                            static_cast<const __nv_bfloat16*>(spk_b_), static_cast<const __nv_bfloat16*>(spk_lnw_),
                            static_cast<const __nv_bfloat16*>(spk_lnb_), 1e-5f, o, kH, 1, dim, kH, pf_stream_));
  out->assign(kH, 0);
  LM_CUDA(cudaMemcpyAsync(out->data(), o, kH * 2, cudaMemcpyDeviceToHost, pf_stream_));
  LM_CUDA(cudaStreamSynchronize(pf_stream_));
  return 0;
}

int EngineImpl::build_voice(const std::vector<uint16_t>& speaker_bf16, const std::vector<int64_t>& prefix,
                            std::string* err) {
  if (speaker_bf16.size() != static_cast<size_t>(kH) || prefix.empty() ||
      prefix.size() > static_cast<size_t>(std::min(cfg_.max_prefill_tokens, cfg_.max_ctx - 1))) {
    if (err) *err = "build_voice: speaker must be 2048 bf16 values and the prefix 1..max_prefill_tokens ids";
    return -1;
  }
  const int p = static_cast<int>(prefix.size());
  const size_t bytes = align_up(static_cast<size_t>(2) * w_->layers * 8 * p * 128 * 2);
  size_t off = 0;
  {
    // A released voice's range is in the free list only after the last row that read it had its slot released
    // (every decode step and the admission that touched it are complete), so writing it here races with nothing.
    std::lock_guard<std::mutex> lk(mu_);
    if (!valloc_locked(bytes, &off)) {
      if (err) *err = "voice store full";
      return -1;
    }
  }
  auto give_back = [&] {
    std::lock_guard<std::mutex> lk(mu_);
    vfree_locked(off, bytes);
  };
  // Startup / clone-build path: runs on the admission stream (the prefill context is shared), beside decode.
  if (quiesce_admission(err)) {
    give_back();
    return -1;
  }
  LM_CUDA(cudaMemcpyAsync(speakers_dev_, speaker_bf16.data(), kH * 2, cudaMemcpyHostToDevice, pf_stream_));
  const int32_t nt = p, past = 0;
  char* dst = voice_store_ + off;
  const int64_t row_stride = static_cast<int64_t>(2) * w_->layers * 8 * p * 128;
  if (int rc = kt2n_pf_run(pf_, 1, &nt, prefix.data(), &past, nullptr, speakers_dev_, dst, row_stride, p, nullptr,
                           nullptr, nullptr, 0, KT2N_PF_FLAGS_FAST | (cfg_.pdl ? 0u : KT2N_PF_FLAG_NO_PDL), pf_stream_)) {
    if (err) *err = std::string("kt2n_pf_run (voice): ") + kt2n_pf_last_error() + " (" + std::to_string(rc) + ")";
    give_back();
    return -1;
  }
  LM_CUDA(cudaStreamSynchronize(pf_stream_));
  Voice v;
  v.off = off;
  v.bytes = bytes;
  v.len = p;
  v.live = true;
  std::lock_guard<std::mutex> lk(mu_);
  voices_.push_back(v);
  return static_cast<int>(voices_.size()) - 1;
}

bool EngineImpl::valloc_locked(size_t bytes, size_t* off) {
  for (auto it = vfree_.begin(); it != vfree_.end(); ++it) {  // first fit (lowest offset)
    if (it->second < bytes) continue;
    *off = it->first;
    const size_t rest = it->second - bytes;
    vfree_.erase(it);
    if (rest > 0) vfree_[*off + bytes] = rest;
    voice_used_ += bytes;
    return true;
  }
  return false;
}

void EngineImpl::vfree_locked(size_t off, size_t bytes) {
  voice_used_ -= bytes;
  auto next = vfree_.lower_bound(off);
  if (next != vfree_.end() && off + bytes == next->first) {  // merge with the range after
    bytes += next->second;
    next = vfree_.erase(next);
  }
  if (next != vfree_.begin()) {  // merge with the range before
    auto prev = std::prev(next);
    if (prev->first + prev->second == off) {
      prev->second += bytes;
      return;
    }
  }
  vfree_[off] = bytes;
}

int EngineImpl::submit(const RowRequest& row, std::string* err) {
  if (row.suffix.empty()) {
    if (err) *err = "suffix must not be empty";
    return -1;
  }
  std::lock_guard<std::mutex> lk(mu_);
  int plen = 0;
  if (row.voice >= 0) {
    plen = voice_len_locked(row.voice);
    if (plen < 0) {
      if (err) *err = "unknown voice";
      return -1;
    }
  } else if (!row.speaker_bf16.empty() && row.speaker_bf16.size() != static_cast<size_t>(kH)) {
    if (err) *err = "speaker_bf16 must hold 2048 values";
    return -1;
  }
  const int64_t prompt = plen + static_cast<int64_t>(row.suffix.size());
  if (prompt >= cfg_.max_ctx || static_cast<int64_t>(row.suffix.size()) > cfg_.max_prefill_tokens) {
    if (err) *err = "prompt longer than the slot / prefill limit";
    return -1;
  }
  if (row.budget < 1 || row.forced.size() > 4096) {
    if (err) *err = "budget must be >= 1 (and at most 4096 forced ids)";
    return -1;
  }
  if (static_cast<int>(pending_.size()) + live_count_.load() >= cfg_.max_rows) {
    if (err) *err = "engine full";
    return -1;
  }
  if (row.voice >= 0) voices_[row.voice].refs += 1;  // until the row's slot is released (or it is cancelled pending)
  pending_.push_back(Pending{row});
  return 0;
}

// Admit up to 16 rows: init their slot state, prefill (prefix copy + suffix), sample the first token, post.
int EngineImpl::admit(std::vector<Pending>& batch, std::string* err) {
  const int n = static_cast<int>(batch.size());
  dev::AdmInit a{};
  a.n = n;
  std::vector<int32_t> new_tokens(n), past_len(n);
  std::vector<int64_t> ids;
  std::vector<const void*> past_kv(n, nullptr);
  bool any_spk = false;
  Inflight rec;
  rec.admission = true;
  for (int i = 0; i < n; ++i) {
    const RowRequest& r = batch[i].req;
    const int s = free_slots_.back();
    free_slots_.pop_back();
    const int plen = r.voice >= 0 ? voices_[r.voice].len : 0;
    a.slot[i] = s;
    slot_voice_[s] = r.voice;  // the row holds a voice ref (taken at submit) until this slot is released
    a.prompt_len[i] = plen + static_cast<int>(r.suffix.size());
    a.budget[i] = r.budget;
    a.cap[i] = r.sampler.max_new_tokens;
    a.flags[i] = (r.demo ? dev::kDemo : 0u) | (r.demo_chunked ? dev::kChunked : 0u);
    a.forced_len[i] = static_cast<int32_t>(r.forced.size());
    a.seed[i] = r.seed;
    sampler_row(r.sampler, a.params[i]);
    new_tokens[i] = static_cast<int32_t>(r.suffix.size());
    past_len[i] = plen;
    if (r.voice >= 0) past_kv[i] = voice_store_ + voices_[r.voice].off;
    a.voff[i] = r.voice >= 0 ? static_cast<int64_t>(voices_[r.voice].off / 2) : -1;
    a.vlen[i] = plen;
    ids.insert(ids.end(), r.suffix.begin(), r.suffix.end());
    if (r.voice < 0 && !r.speaker_bf16.empty()) any_spk = true;
    HostRow h;
    h.tag = r.tag;
    h.gen = next_gen_++;
    h.slot = s;
    h.prompt_len = a.prompt_len[i];
    slot_rows_[s] = h;
    tag_slot_[r.tag] = s;
    rec.slots.push_back(s);
    rec.gens.push_back(h.gen);
  }
  // staging buffers are reused: wait for the previous admission's copies
  cudaStream_t as = cfg_.async_admission ? pf_stream_ : stream_;
  LM_CUDA(cudaEventSynchronize(stage_ev_));
  for (int i = 0; i < n; ++i) {
    const RowRequest& r = batch[i].req;
    if (!r.forced.empty()) {  // tests only: synchronous copy into the slot's forced row
      std::memcpy(forced_stage_, r.forced.data(), r.forced.size() * 4);
      LM_CUDA(cudaMemcpyAsync(st_.forced + static_cast<int64_t>(a.slot[i]) * st_.forced_cap, forced_stage_,
                              r.forced.size() * 4, cudaMemcpyHostToDevice, as));
      LM_CUDA(cudaStreamSynchronize(as));
    }
    if (any_spk) {
      if (r.voice < 0 && !r.speaker_bf16.empty())
        std::memcpy(spk_stage_ + static_cast<size_t>(i) * kH, r.speaker_bf16.data(), kH * 2);
      else
        std::memset(spk_stage_ + static_cast<size_t>(i) * kH, 0, kH * 2);  // rows with a past ignore it
    }
  }
  if (any_spk) {
    LM_CUDA(cudaMemcpyAsync(speakers_dev_, spk_stage_, static_cast<size_t>(n) * kH * 2, cudaMemcpyHostToDevice, as));
  }
  LM_CUDA(cudaEventRecord(stage_ev_, as));
  // adm_slot / adm_tok are read by the previous admission's post on the decode stream, which may still be queued
  // behind decode steps: this admission's writes wait for it on the GPU (the host does not block).
  LM_CUDA(cudaStreamWaitEvent(as, join_ev_, 0));
  // The slots were idle (released only after every ring entry that touched them was read), and only the join below
  // adds them to the decode bucket, so this work can run beside decode on its own stream.
  LM_CUDA(dev::launch_adm_init(st_, a, as));
  std::vector<int32_t> dest(a.slot, a.slot + n);
  if (int rc = kt2n_pf_run(pf_, n, new_tokens.data(), ids.data(), past_len.data(), past_kv.data(),
                           any_spk ? speakers_dev_ : nullptr, kv_, slot_elems_, cfg_.max_ctx, dest.data(), nullptr,
                           adm_logits_, kAudioN, KT2N_PF_FLAGS_FAST | (cfg_.pdl ? 0u : KT2N_PF_FLAG_NO_PDL), as)) {
    if (err) *err = std::string("kt2n_pf_run: ") + kt2n_pf_last_error() + " (" + std::to_string(rc) + ")";
    return -1;
  }
  if (int rc = kt2n_samp_sample(adm_logits_, kAudioN, st_.params, st_.history, st_.history_len, st_.seeds, st_.offsets,
                                st_.adm_slot, nullptr, 0, st_.adm_tok, nullptr, nullptr, n, KT2N_SAMP_UPDATE_STATE,
                                as)) {
    if (err) *err = "kt2n_samp_sample (admission): " + std::to_string(rc);
    return -1;
  }
  if (cfg_.debug_logits) {
    LM_CUDA(cudaMemcpyAsync(dbg_adm_host_, adm_logits_, static_cast<size_t>(n) * kAudioN * 2, cudaMemcpyDeviceToHost,
                            as));
  }
  LM_CUDA(cudaEventRecord(adm_ev_, as));
  rec.t0 = std::chrono::steady_clock::now();
  rec.bucket = n;
  adm_rec_ = std::move(rec);
  adm_inflight_ = true;
  live_count_ += n;
  std::lock_guard<std::mutex> lk(mu_);
  stats_.admissions += 1;
  stats_.rows_admitted += n;
  return 0;
}

// The admission's prefill + first draw are done (adm_ev_): post its first tokens on the decode stream (ring order)
// and add the rows to the decode bucket.
int EngineImpl::join_admission(std::vector<TokenEvent>* events, std::string* err) {
  joining_ = true;  // these collects must really wait for a decode step (join_poll would return at once)
  while (inflight_.size() + 1 >= static_cast<size_t>(kRing)) {  // keep the ring from wrapping over unread entries
    if (collect(events, true, err) < 0) {
      joining_ = false;
      return -1;
    }
  }
  joining_ = false;
  LM_CUDA(cudaStreamWaitEvent(stream_, adm_ev_, 0));
  const int n = static_cast<int>(adm_rec_.slots.size());
  LM_CUDA(dev::launch_post(st_, n, st_.adm_slot, st_.adm_tok, 0, stream_));
  LM_CUDA(cudaEventRecord(join_ev_, stream_));
  Inflight rec = std::move(adm_rec_);
  rec.seq = next_seq_++;
  for (int s : rec.slots) active_.push_back(s);
  inflight_.push_back(std::move(rec));
  adm_inflight_ = false;
  rows_dirty_ = true;
  return 0;
}

// Slots of removed rows become reusable once every ring entry that may have touched them has been read.
void EngineImpl::release_slots() {
  const uint32_t done = inflight_.empty() ? next_seq_ : inflight_.front().seq;  // all seq < done are read
  for (auto it = pending_free_.begin(); it != pending_free_.end();) {
    if (static_cast<int32_t>(it->second - done) <= 0) {
      const int s = it->first;
      if (slot_voice_[s] >= 0) {  // no queued or running work reads this row's voice any more
        std::lock_guard<std::mutex> lk(mu_);
        unref_voice_locked(slot_voice_[s]);
        slot_voice_[s] = -1;
      }
      free_slots_.push_back(s);
      it = pending_free_.erase(it);
    } else {
      ++it;
    }
  }
}

int EngineImpl::quiesce_admission(std::string* err) {
  if (adm_inflight_) LM_CUDA(cudaEventSynchronize(adm_ev_));  // joined at the next pump
  return 0;
}

int EngineImpl::rebucket(std::string* err) {
  rows_dirty_ = false;
  const int n = static_cast<int>(active_.size());
  if (n == 0) {
    cur_bucket_ = 0;
    return 0;
  }
  const int b = bucket_for(n);
  dev::RowMap m{};
  m.n = n;
  m.b = b;
  m.scratch = scratch_;
  for (int r = 0; r < n; ++r) m.slot[r] = active_[r];
  LM_CUDA(dev::launch_rowmap(st_, m, stream_));
  cur_bucket_ = b;
  return 0;
}

// Wait (spin, then sleep) until the ring entry of `f` is complete.  Returns 1 when complete, 0 when not (block off).
int EngineImpl::wait_seq(const Inflight& f, bool block) {
  const volatile dev::RingEntry* e = ring_host_ + (f.seq % kRing);
  auto done = [&] { return e->seq == f.seq + 1u; };
  if (done()) return 1;
  if (!block) return 0;
  // join_poll: an admission that finished while we wait for a decode step is joined now (the caller goes round its
  // loop on 0), so its rows enter the next launched step instead of the one after.
  auto adm_ready = [&] {
    return cfg_.join_poll && adm_inflight_ && !joining_ && cudaEventQuery(adm_ev_) == cudaSuccess;
  };
  for (int i = 0; i < 20000; ++i) {
    if (done()) return 1;
    if ((i & 255) == 255 && adm_ready()) return 0;
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
  }
  const timespec ts{0, 20000};
  const auto t0 = std::chrono::steady_clock::now();
  bool traced = false;
  while (!done()) {
    if (adm_ready()) return 0;
    if (cudaStreamQuery(stream_) != cudaErrorNotReady && !done()) {
      // the stream drained without completing the entry: a launch failed
      if (!done()) return -1;
    }
    if (!traced && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1)) {
      traced = true;  // a stall: say what the engine is waiting for (stderr, once per wait)
      std::fprintf(stderr,
                   "kt2n::lm stall: waiting seq %u (entry seq %u), next_seq %u, inflight %zu, active %zu, "
                   "adm_inflight %d (event %s), decode stream %s, prefill stream %s, free %zu, pending_free %zu\n",
                   f.seq, e->seq, next_seq_, inflight_.size(), active_.size(), adm_inflight_ ? 1 : 0,
                   cudaGetErrorName(cudaEventQuery(adm_ev_)), cudaGetErrorName(cudaStreamQuery(stream_)),
                   cudaGetErrorName(cudaStreamQuery(pf_stream_)), free_slots_.size(), pending_free_.size());
    }
    nanosleep(&ts, nullptr);
  }
  return 1;
}

int EngineImpl::collect(std::vector<TokenEvent>* events, bool block, std::string* err) {
  int got = 0;
  while (!inflight_.empty()) {
    const Inflight& f = inflight_.front();
    const int w = wait_seq(f, block && got == 0);
    if (w < 0) {
      if (err) *err = std::string("decode stream failed: ") + cudaGetErrorString(cudaStreamQuery(stream_));
      broken_ = true;
      return -1;
    }
    if (w == 0) break;
    const dev::RingEntry* e = ring_host_ + (f.seq % kRing);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (!f.admission) {
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f.t0).count();
      std::lock_guard<std::mutex> lk(mu_);
      stats_.last_step_ms = ms;
    }
    for (size_t r = 0; r < f.slots.size(); ++r) {
      const int s = f.slots[r];
      const int32_t token = e->token[r];
      if (token < 0) continue;
      HostRow& h = slot_rows_[s];
      if (h.gen != f.gens[r]) continue;  // the row this step ran is gone (cancelled / finished), slot reused
      if (h.finished || h.cancelled) continue;
      TokenEvent ev{h.tag, token, e->index[r], static_cast<Finish>(e->finish[r])};
      if (cfg_.debug_logits) {
        const uint16_t* src = (f.admission ? dbg_adm_host_ : dbg_host_) + r * kAudioN;
        dbg_[{h.tag, ev.index}] = std::vector<uint16_t>(src, src + kAudioN);
      }
      events->push_back(ev);
      ++got;
      if (ev.finish != Finish::kNone) {
        h.finished = true;
        rows_dirty_ = true;
      }
    }
    inflight_.pop_front();
  }
  return got;
}

// Sprint 6: requested from any thread; applied by the driver thread at the start of the next pump().
int EngineImpl::set_decode_priority(int level) {
  level = std::max(0, std::min(level, kPrioLevels - 1));
  want_level_.store(level);
  return std::max(prio_lo_ - level, prio_hi_ < prio_lo_ ? prio_hi_ + 1 : prio_hi_);
}

int EngineImpl::apply_decode_priority(int level, std::string* err) {
  const int prio = std::max(prio_lo_ - level, prio_hi_ < prio_lo_ ? prio_hi_ + 1 : prio_hi_);
  const int eff = std::min(kPrioLevels - 1, prio_lo_ - prio);  // the clamped level
  if (eff == prio_level_) return 0;
  if (!prio_stream_[eff]) LM_CUDA(cudaStreamCreateWithPriority(&prio_stream_[eff], cudaStreamNonBlocking, prio));
  // everything already queued on the old decode stream comes first (steps in flight, posts, row maps, joins)
  LM_CUDA(cudaEventRecord(prio_ev_, stream_));
  LM_CUDA(cudaStreamWaitEvent(prio_stream_[eff], prio_ev_, 0));
  stream_ = prio_stream_[eff];
  prio_level_ = eff;
  std::lock_guard<std::mutex> lk(mu_);
  stats_.decode_priority = prio;
  stats_.priority_switches += 1;
  return 0;
}

int EngineImpl::pump(std::vector<TokenEvent>* events, bool block, std::string* err) {
  if (broken_) {
    if (err) *err = "engine failed earlier";
    return -1;
  }
  if (const int want = want_level_.exchange(-1); want >= 0 && apply_decode_priority(want, err)) {
    broken_ = true;
    return -1;
  }
  const size_t before = events->size();
  std::vector<Pending> batch_all;
  std::vector<uint64_t> cancels;
  {
    std::lock_guard<std::mutex> lk(mu_);
    cancels.swap(cancels_);
    // cancels of rows not admitted yet: drop them from the queue (+ event)
    for (auto c = cancels.begin(); c != cancels.end();) {
      auto p = std::find_if(pending_.begin(), pending_.end(), [&](const Pending& q) { return q.req.tag == *c; });
      if (p != pending_.end()) {
        events->push_back(TokenEvent{*c, -1, 0, Finish::kCancelled});
        unref_voice_locked(p->req.voice);
        pending_.erase(p);
        c = cancels.erase(c);
      } else {
        ++c;
      }
    }
  }
  // cancels of admitted rows: event now, removal at the boundary below
  for (uint64_t tag : cancels) {
    auto it = tag_slot_.find(tag);
    if (it == tag_slot_.end()) continue;
    HostRow& h = slot_rows_[it->second];
    if (!h.finished && !h.cancelled) {
      h.cancelled = true;
      events->push_back(TokenEvent{tag, -1, 0, Finish::kCancelled});
    }
  }
  for (long iter = 0;; ++iter) {
    if (iter == 200000) {  // a pump that cannot make progress: report the state once
      std::fprintf(stderr, "kt2n::lm pump spinning: active %zu inflight %zu adm %d pending_free %zu free %zu next_seq %u "
                   "bucket %d dirty %d\n", active_.size(), inflight_.size(), adm_inflight_ ? 1 : 0, pending_free_.size(),
                   free_slots_.size(), next_seq_, cur_bucket_, rows_dirty_ ? 1 : 0);
      for (int s : active_) {
        const HostRow& h = slot_rows_[s];
        std::fprintf(stderr, "  slot %d tag %llu gen %llu finished %d cancelled %d\n", s,
                     static_cast<unsigned long long>(h.tag), static_cast<unsigned long long>(h.gen), h.finished ? 1 : 0,
                     h.cancelled ? 1 : 0);
      }
    }
    if (collect(events, false, err) < 0) return -1;
    release_slots();
    if (adm_inflight_) {
      const cudaError_t q = cudaEventQuery(adm_ev_);
      if (q == cudaSuccess) {
        if (join_admission(events, err)) {
          broken_ = true;
          return -1;
        }
      } else if (q != cudaErrorNotReady) {
        if (err) *err = std::string("admission failed: ") + cudaGetErrorString(q);
        broken_ = true;
        return -1;
      }
    }
    // boundary: drop finished / cancelled rows; their slots are released once the steps that ran them are read
    {
      std::vector<int> keep;
      for (int s : active_) {
        HostRow& h = slot_rows_[s];
        if (h.finished || h.cancelled) {
          tag_slot_.erase(h.tag);
          h = HostRow{};
          pending_free_.emplace_back(s, next_seq_);
          live_count_ -= 1;
          std::lock_guard<std::mutex> lk(mu_);
          stats_.rows_finished += 1;
        } else {
          keep.push_back(s);
        }
      }
      if (keep.size() != active_.size()) rows_dirty_ = true;
      active_.swap(keep);
    }
    // one admission batch (<= 16 rows, <= max_prefill_tokens) at a time
    if (!adm_inflight_) {
      std::vector<Pending> batch;
      {
        std::lock_guard<std::mutex> lk(mu_);
        int64_t toks = 0;
        while (!pending_.empty() && static_cast<int>(batch.size()) < kMaxPrefillRows &&
               static_cast<int>(free_slots_.size()) > static_cast<int>(batch.size())) {
          const int64_t t = static_cast<int64_t>(pending_.front().req.suffix.size());
          if (!batch.empty() && toks + t > cfg_.max_prefill_tokens) break;
          toks += t;
          batch.push_back(std::move(pending_.front()));
          pending_.pop_front();
        }
      }
      if (!batch.empty()) {
        while (cfg_.debug_logits && !inflight_.empty()) {  // debug: one admission dump buffer, read it first
          if (collect(events, true, err) < 0) return -1;
        }
        if (admit(batch, err)) {
          broken_ = true;
          return -1;
        }
        if (!cfg_.async_admission && join_admission(events, err)) {  // serial mode: join on the same stream now
          broken_ = true;
          return -1;
        }
      }
    }
    if (rows_dirty_ && rebucket(err)) {
      broken_ = true;
      return -1;
    }
    // keep run_ahead steps in flight (fewer while an admission runs: admission_run_ahead)
    const int ra = (adm_inflight_ && cfg_.admission_run_ahead > 0) ? std::min(cfg_.run_ahead, cfg_.admission_run_ahead)
                                                                    : cfg_.run_ahead;
    while (cur_bucket_ > 0 && !active_.empty() && static_cast<int>(inflight_.size()) < ra) {
      const size_t bi = std::find(buckets_.begin(), buckets_.end(), cur_bucket_) - buckets_.begin();
      if (cfg_.graphs) {
        const cudaError_t e = cudaGraphLaunch(graphs_[bi], stream_);
        if (e != cudaSuccess) {
          if (err) *err = std::string("cudaGraphLaunch: ") + cudaGetErrorString(e);
          broken_ = true;
          return -1;
        }
      } else if (enqueue_step(cur_bucket_, err)) {
        broken_ = true;
        return -1;
      }
      Inflight rec;
      rec.seq = next_seq_++;
      rec.slots = active_;
      rec.gens.reserve(active_.size());
      for (int s : active_) rec.gens.push_back(slot_rows_[s].gen);
      rec.t0 = std::chrono::steady_clock::now();
      rec.bucket = cur_bucket_;
      inflight_.push_back(std::move(rec));
      std::lock_guard<std::mutex> lk(mu_);
      stats_.steps += 1;
      stats_.bucket = cur_bucket_;
    }
    if (!block || events->size() > before) break;
    if (!inflight_.empty()) {
      if (collect(events, true, err) < 0) return -1;
      continue;
    }
    if (adm_inflight_) {
      LM_CUDA(cudaEventSynchronize(adm_ev_));
      continue;
    }
    break;  // nothing running (live rows can only be pending ones waiting for a slot release)
  }
  return static_cast<int>(events->size() - before);
}

}  // namespace

std::unique_ptr<Engine> Engine::create(const Config& cfg, std::string* err) {
  auto e = std::make_unique<EngineImpl>();
  std::string local;
  if (e->init(cfg, err ? err : &local)) return nullptr;
  return e;
}

}  // namespace kt2n::lm
