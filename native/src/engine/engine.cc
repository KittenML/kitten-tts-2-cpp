// L8: the native engine runtime behind include/kt2n/engine.h.
//
// Threads (plan budget: production's 4 physical cores):
//   text pool (threads_text)   L9a/L9b front end: plan + normalize segments, tokens, budgets
//   conductor (1)              production's owner loop (TIMING_POLICIES §1): collect renders, DW gate,
//                              admit, LM steps (L3 pump, run-ahead), T9 / D / groups / async / finals,
//                              publication accounting (DeliveryClock, prefetch)
//   post pool (threads_post)   L10 ResponsePost -> StartupHold -> level_publication -> PCM16/WAV bytes
// The conductor makes production's decisions (policy.h); renders run on the render backend's streams
// while the LM keeps decoding (execution overlap only).
#include "build_flags.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <sys/eventfd.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "kt2n/engine.h"
#include "kt2n/post/encoder.h"
#include "kt2n/post/post.h"
#include "kt2n/post/response.h"
#include "kt2n/sha256.h"
#include "policy.h"
#include "text_front.h"
#include "json.h"
#include "voice_blob.h"
#if KT2N_ENGINE_WITH_VOICE
#include "clone_weights.h"
#include "voice_service.h"
#endif

// Lab diagnostic (KT2N_DEBUG_MEM=1): device memory in use (all processes) after each create phase, to stderr.
static void debug_mem(const char* phase) {
  static const bool on = [] {
    const char* v = std::getenv("KT2N_DEBUG_MEM");
    return v && *v == '1';
  }();
  if (!on) return;
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
    std::fprintf(stderr, "[mem] %-28s used %zu MiB\n", phase, (total_b - free_b) >> 20);
}

namespace kt2n::engine {
namespace pol = kt2n::engine::policy;

namespace {

// Timed condition waits. Production builds wait on the steady clock (pthread_cond_clockwait); the TSan build
// (KT2N_TSAN_COMPAT, GCC 11's TSan does not intercept pthread_cond_clockwait and then reports false races)
// waits on the system clock through pthread_cond_timedwait, which it does intercept.
template <class Rep, class Period>
void cv_wait_for(std::condition_variable& cv, std::unique_lock<std::mutex>& lk, std::chrono::duration<Rep, Period> d) {
#if KT2N_TSAN_COMPAT
  cv.wait_until(lk, std::chrono::system_clock::now() + d);
#else
  cv.wait_for(lk, d);
#endif
}
template <class Rep, class Period, class Pred>
bool cv_wait_for(std::condition_variable& cv, std::unique_lock<std::mutex>& lk, std::chrono::duration<Rep, Period> d,
                 Pred pred) {
#if KT2N_TSAN_COMPAT
  return cv.wait_until(lk, std::chrono::system_clock::now() + d, pred);
#else
  return cv.wait_for(lk, d, pred);
#endif
}

int64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}
double mono_s() { return double(mono_ns()) / 1e9; }

std::string jnum(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.3f", v);
  return b;
}

// Custom voices that can be loaded at the same time, for the LM voice store and the render voice table: the
// engine's inline LRU + the adapter's saved-voice cache (production's KITTENTTS_09_CUSTOM_VOICE_CACHE_ENTRIES, 32,
// read from the same process environment) + 8 in transit (consent voices, loads racing an eviction). Before 15:10
// both stores held 32, so a server with a full saved-voice cache refused its first inline clone. Evicted voices
// give their space back (release_voices), so this bounds what is live at once, not a lifetime total.
int custom_voice_headroom(int inline_lru) {
  int cache = 32;
  if (const char* v = std::getenv("KITTENTTS_09_CUSTOM_VOICE_CACHE_ENTRIES")) cache = std::max(1, std::atoi(v));
  return std::max(1, inline_lru) + cache + 8;
}

// Rows the engine serves at once: LM decode rows, render-state slots and the render pool's per-row pages.
// Every seat gets its own row while the GPU can hold them, as before: the release's 10 / 20 seats are unchanged
// (this function returns max_streams there and never queries the GPU). With more seats than that, the rows are
// capped and the admitted requests beyond them wait in admission, in arrival order; that wait counts in their
// first audio (plan, owner add-on: "requests the GPU memory cannot hold wait in the engine's admission queue").
// Before this cap, 300 seats (the owner's 1-200+ users graph) sized the render pool at 40 pages x 300 = 12,000
// pages x 16.5 MB = 198 GB and engine create failed in cudaMalloc(render arena).
// KT2N_ENGINE_ROWS sets the cap (capacity only, never a timing decision). Otherwise, above 32 seats, the cap is
// the smaller of
//  - the throughput knee, 32 rows: prior measurements with the RC1 settings gave the same
//    chars/s at 32 and 48 concurrent rows (3,738-3,837 vs 3,807-3,836) but 0 underruns at 32 and underruns in
//    351-411 of 530 requests (38-65 s of stalls) at 48. Past the knee more rows only add stalls; queued requests
//    wait for a row instead, before their first audio;
//  - the GPU memory fit: per row one LM K/V slot (max_ctx x 112 KiB) + 40 pool pages (kPageBytes, render/ops.h);
//    a fixed 36 GB for everything else (l0-rc 21:54: 54.2 GB at 20 seats, minus 22 LM slots and 800 row pages,
//    plus ~4 GB of lazy growth after create); 10% of the card kept free (G-C20's memory headroom). A 96 GB
//    RTX PRO 6000 fits 49 rows (48 rows measured at 82.3 GB peak); an 80 GB H100 36.
int engine_rows(int max_streams, int device, bool gpu, std::string* why) {
  constexpr int kMaxRows = 60;   // LM rows (rows + 2) and render slots (rows + 4) stop at 64
  constexpr int kKneeRows = 32;  // see above
  int cap = kMaxRows;
  *why = "row limit";
  if (const char* r = std::getenv("KT2N_ENGINE_ROWS"); r && *r && std::atoi(r) > 0) {
    cap = std::min(kMaxRows, std::atoi(r));
    *why = "KT2N_ENGINE_ROWS";
  } else if (max_streams > kKneeRows && gpu) {  // gpu: a real LM or renderer (the CPU stubs hold no device memory)
    int lm_max_ctx = 4096;
    if (const char* v = std::getenv("KT2N_LM_MAX_CTX")) lm_max_ctx = std::max(1024, std::atoi(v) / 64 * 64);
    size_t free_b = 0, total_b = 0;
    if (cudaSetDevice(device) == cudaSuccess && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      const double per_row = double(lm_max_ctx) * 114688.0 + 40.0 * 16515072.0;
      const double room = 0.9 * double(total_b) - 36e9;
      const int fit = std::max(4, std::min(kMaxRows, int(room / per_row)));
      cap = std::min(fit, kKneeRows);
      *why = fit < kKneeRows ? "GPU memory fit" : "throughput knee";
    }
  }
  if (max_streams <= cap) *why = "seats";
  return std::min(max_streams, cap);
}
// ------------------------------------------------------------------------------------- errors
struct ErrRow {
  int32_t status;
  int32_t http;
  const char* code;
  const char* type;
};
const ErrRow kErrors[] = {
    {KT2N_OK, 0, "", ""},
    {KT2N_ERR_INVALID_ARGUMENT, 500, "internal_error", "server_error"},
    {KT2N_ERR_INVALID_HANDLE, 500, "internal_error", "server_error"},
    {KT2N_ERR_ABI_MISMATCH, 500, "internal_error", "server_error"},
    {KT2N_ERR_AGAIN, 0, "", ""},
    {KT2N_ERR_CANCELLED, 0, "", ""},
    {KT2N_ERR_NOT_READY, 503, "service_unavailable", "server_error"},
    {KT2N_ERR_UNSUPPORTED, 500, "internal_error", "server_error"},
    {KT2N_ERR_TOO_SMALL, 0, "", ""},
    {KT2N_ERR_INVALID_REQUEST, 400, "invalid_request_error", "invalid_request_error"},
    {KT2N_ERR_INVALID_INPUT, 400, "invalid_input", "invalid_request_error"},
    {KT2N_ERR_INVALID_VOICE, 400, "invalid_voice", "invalid_request_error"},
    {KT2N_ERR_INVALID_RESPONSE_FORMAT, 400, "invalid_response_format", "invalid_request_error"},
    {KT2N_ERR_INVALID_STREAM_FORMAT, 400, "invalid_stream_format", "invalid_request_error"},
    {KT2N_ERR_REQUEST_TOO_LARGE, 413, "request_too_large", "invalid_request_error"},
    {KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED, 400, "custom_voice_realtime_not_supported", "invalid_request_error"},
    {KT2N_ERR_INPUT_ALREADY_CLOSED, 400, "input_already_closed", "invalid_request_error"},
    {KT2N_ERR_INVALID_SESSION_CONFIGURATION, 400, "invalid_session_configuration", "invalid_request_error"},
    {KT2N_ERR_MODEL_NOT_FOUND, 404, "model_not_found", "invalid_request_error"},
    {KT2N_ERR_VOICE_NOT_FOUND, 404, "voice_not_found", "invalid_request_error"},
    {KT2N_ERR_INVALID_AUDIO, 400, "invalid_audio", "invalid_request_error"},
    {KT2N_ERR_INVALID_AUDIO_DURATION, 400, "invalid_audio_duration", "invalid_request_error"},
    {KT2N_ERR_INVALID_AUDIO_SIGNAL, 400, "invalid_audio_signal", "invalid_request_error"},
    {KT2N_ERR_INVALID_REFERENCE_AUDIO, 400, "invalid_reference_audio", "invalid_request_error"},
    {KT2N_ERR_REFERENCE_ALIGNMENT_FAILED, 422, "invalid_reference_audio", "invalid_request_error"},
    {KT2N_ERR_SPEAKER_EMBEDDING_FAILED, 422, "speaker_embedding_failed", "invalid_request_error"},
    {KT2N_ERR_SPEAKER_VERIFICATION_FAILED, 422, "speaker_verification_failed", "invalid_request_error"},
    {KT2N_ERR_CONSENT_PHRASE_MISMATCH, 400, "consent_phrase_mismatch", "invalid_request_error"},
    {KT2N_ERR_UNSUPPORTED_CONSENT_LANGUAGE, 400, "unsupported_consent_language", "invalid_request_error"},
    {KT2N_ERR_CONSENT_REQUIRES_SPEAKER_VERIFICATION, 409, "consent_requires_speaker_verification",
     "invalid_request_error"},
    {KT2N_ERR_VOICE_ARTIFACT_VERSION_MISMATCH, 409, "voice_artifact_version_mismatch", "invalid_request_error"},
    {KT2N_ERR_VOICE_ARTIFACT_REBUILDING, 503, "voice_artifact_rebuilding", "server_error"},
    {KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY, 429, "custom_voice_creation_busy", "rate_limit_error"},
    {KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR, 503, "custom_voice_service_error", "server_error"},
    {KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, 503, "custom_voice_service_unavailable", "server_error"},
    {KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT, 400, "invalid_reference_transcript", "invalid_request_error"},
    {KT2N_ERR_INVALID_CONSENT_TRANSCRIPT, 400, "invalid_consent_transcript", "invalid_request_error"},
    {KT2N_ERR_SPEECH_GENERATION_FAILED, 503, "speech_generation_failed", "server_error"},
    {KT2N_ERR_DEADLINE_EXCEEDED, 503, "speech_generation_failed", "server_error"},
    {KT2N_ERR_SERVICE_UNAVAILABLE, 503, "service_unavailable", "server_error"},
    {KT2N_ERR_INTERNAL, 500, "internal_error", "server_error"},
};

// ------------------------------------------------------------------------------------- levers
struct LiveLever {
  const char* key;
  bool checked;  // TIMING / MODEL / AUDIO rows: the engine must reproduce the live value
  const char* us;
  const char* ap;
  const char* fq;
};
const LiveLever kLiveLevers[] = {
#include "live_levers.inc"
};

bool lever_equal(std::string a, std::string b) {
  auto trim = [](std::string& x) {
    while (!x.empty() && std::isspace(static_cast<unsigned char>(x.back()))) x.pop_back();
    size_t i = 0;
    while (i < x.size() && std::isspace(static_cast<unsigned char>(x[i]))) ++i;
    x.erase(0, i);
    for (auto& c : x) c = char(std::tolower(static_cast<unsigned char>(c)));
  };
  trim(a);
  trim(b);
  if (a == b) return true;
  auto truthy = [](const std::string& x) { return x == "1" || x == "true" || x == "yes" || x == "on"; };
  auto falsy = [](const std::string& x) { return x == "0" || x == "false" || x == "no" || x == "off"; };
  if ((truthy(a) && truthy(b)) || (falsy(a) && falsy(b))) return true;
  char* ea = nullptr;
  char* eb = nullptr;
  const double da = std::strtod(a.c_str(), &ea), db = std::strtod(b.c_str(), &eb);
  return !a.empty() && !b.empty() && *ea == 0 && *eb == 0 && da == db;
}

// levers_json against the selected compatibility profile. A checked lever at another value fails closed,
// except the few the runtime applies (DW waits, COMPLETE_SLOTS). Unknown keys are listed, not fatal (the
// additional configuration can contain keys that do not belong to the inference engine).
bool apply_levers(const char* json_text, const std::string& app, policy::Levers& lv, std::string* bad_key,
                  std::string* message, std::vector<std::string>* unknown) {
  kt2n::json::Value doc;
  try {
    doc = kt2n::json::parse(json_text);
  } catch (const std::exception& ex) {
    *bad_key = "levers_json";
    *message = std::string("levers_json is not valid JSON: ") + ex.what();
    return false;
  }
  if (!doc.is_object()) {
    *bad_key = "levers_json";
    *message = "levers_json must be an object";
    return false;
  }
  for (const auto& [key, val] : doc.members()) {
    std::string v = val.is_string() ? val.as_string() : val.dump();
    const LiveLever* lev = nullptr;
    for (const auto& l : kLiveLevers)
      if (key == l.key) lev = &l;
    if (!lev) {
      unknown->push_back(key);
      continue;
    }
    if (!lev->checked) continue;
    const char* live = app == "ap" ? lev->ap : app == "fq" ? lev->fq : lev->us;
    if (lever_equal(v, live)) continue;
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    const bool num = !v.empty() && *end == 0;
    if (key == "KITTENTTS_09_DW_RUNWAY_MS" && num) {
      lv.dw_runway_ms = d;
      continue;
    }
    if (key == "KITTENTTS_09_DW_MAX_WAIT_MS" && num) {
      lv.dw_max_wait_ms = d;
      continue;
    }
    if (key == "KITTENTTS_09_COMPLETE_SLOTS" && num) {
      lv.complete_slots = int(d);
      continue;
    }
    *bad_key = key;
    *message = key + "=" + v + " differs from the live value " + live + "; the native engine implements only the live value";
    return false;
  }
  return true;
}

void set_err(kt2n_error* err, int32_t status, const std::string& message, const std::string& param = "") {
  if (!err) return;
  const kt2n_error_info* info = kt2n_error_lookup(status);
  err->status = status;
  err->http_status = info->http_status;
  std::snprintf(err->code, sizeof err->code, "%s", info->code);
  std::snprintf(err->param, sizeof err->param, "%s", param.c_str());
  std::snprintf(err->message, sizeof err->message, "%s", message.c_str());
}

// ------------------------------------------------------------------------------------- state
struct Voice {
  std::string id;
  std::vector<std::string> names;  // id + public names
  std::string language = "en";
  int prompt_tokens = 0;
  int prompt_frames = 0;
  int lm_prefix_ids = 0;
  bool hidden = false;
  double peak_gain_db = 0.0;
  std::vector<float> raw_speaker;
  std::atomic<int> lm_voice{-1};  // -1 until the conductor built its LM prefix (saved voices: asynchronously)
  int render_voice = -1;
  int64_t geometry = 0;
  std::atomic<int> refs{0};
  int kind = KT2N_VOICE_BUILTIN;
  std::atomic<bool> evicted{false};
  bool lm_released = false;      // conductor only: its LM store space was given back (release_voices)
  bool render_released = false;  // conductor only: its render table row + prefix pages were given back
  std::vector<uint8_t> artifact;  // saved voices: the one-voice voices.kt2n blob (kt2n_voice_export)
  bool consent_only = false;      // a consent recording's speaker embedding: never speaks
};

struct Job {  // one LM chunk (production _CudaSegmentJob); segment_index = ordinal
  int ordinal = 0;
  int raw_segment = 0;
  std::u32string text;
  std::vector<int32_t> suffix;
  int budget = 200;
  bool charged = false, acknowledged = false;
  bool lm_done = false, done = false;
  bool preview_ready = false;
  int64_t published_samples = 0;
  std::optional<double> first_pub;
};

struct PostItem {
  int job = -1;
  std::vector<float> pcm;
  bool final = false;   // the job's last publication
  bool has_more = false;
  int lane_jobs = 0;    // jobs the lane holds now (SeamContext)
  bool lane_sealed = false;
  bool end = false;     // response complete (after the last job's final)
  int status = 0;       // end with a failure
};

struct Request;

struct EventRec {
  kt2n_event ev{};
  std::string data, name, fields;
  kt2n_error error{};
};

struct Request {
  uint64_t id = 0;
  uint64_t user_data = 0;
  uint32_t flags = 0;
  bool timeline = false;
  Voice* voice = nullptr;
  std::string raw;
  uint64_t salt = 0;
  lm::SamplerSettings sampler;
  double speed = 1.0;
  int format = KT2N_FORMAT_PCM;
  int output = KT2N_OUTPUT_STREAM;
  bool demo = false;  // speech mode DEMO (stream:false)
  int64_t deadline_ns = 0;
  double started_s = 0;

  // Text (conductor-owned after hand-off).
  std::unique_ptr<PlannedRequest> plan;
  std::vector<std::unique_ptr<Job>> jobs;      // conductor only (absorbed from new_jobs)
  std::vector<std::unique_ptr<Job>> new_jobs;  // text thread -> conductor, under e->mu
  size_t jobs_created = 0;                     // under e->mu
  bool c_sealed = false;                       // conductor's snapshot of `sealed` (taken with the jobs)
  size_t next_index = 0;
  int inflight = 0;                // lane.inflight
  size_t segments_normalized = 0;  // planned segments turned into jobs
  bool normalizing = false;
  bool sealed = false;             // every planned segment normalized
  int total_chunks = 0;            // control.q_demo_chunks
  int text_status = 0;
  std::string text_error;

  // Lifecycle.
  std::atomic<bool> cancelled{false};
  int cancel_status = KT2N_ERR_CANCELLED;
  std::atomic<bool> release_requested{false};
  std::string fail_reason;  // set by the conductor before the failure's post item (happens-before via q->mu)
  bool in_lanes = false;  // conductor
  int text_tasks = 0;     // under e->mu: text-pool tasks queued or running for this request
  int live_rows = 0;      // conductor: rows (decoding or rendering) that point at this request
  bool generation_complete = false;  // conductor sent the end item
  std::atomic<bool> terminal{false};  // DONE emitted
  int opening_skipped = 0, kept_tokens = 0;
  // True from create until the startup hold releases (a streamed response with a hold). The post thread clears it,
  // the conductor reads it (arm KT2N_HOLD_URGENT: this request's commits are on its first-audio path).
  std::atomic<bool> awaiting_hold{false};

  // Post side (guarded by mu).
  std::mutex mu;
  std::deque<PostItem> post_q;
  bool post_scheduled = false;
  std::mutex clock_mu;            // DeliveryClock only: the conductor never waits on post processing
  pol::DeliveryClock clock;       // under clock_mu
  std::atomic<int64_t> dropped_samples{0};  // the onset trimmer's dropped samples (post writes, conductor reads)
  std::unique_ptr<post::ResponsePost> rpost;
  std::unique_ptr<post::StartupHold> hold;
  std::optional<post::PeakNorm> peak;
  post::PeakNorm* peak_ptr = nullptr;  // a WS phrase levels with its session's state
#if KT2N_ENGINE_WITH_AV
  std::unique_ptr<post::StreamEncoder> enc;  // mp3 / aac / opus / flac (L10, production's CLI settings)
#endif
  std::vector<float> whole;  // COMPLETE / STREAM_FILE: the whole utterance
  int64_t samples_out = 0;
  int chunk_index = 0;
  bool header_sent = false;
  bool headers_sent = false;
  int post_begun_job = -1;
  // Event identity: own id/seq for an HTTP request; a WS phrase emits as its session.
  uint64_t ev_id = 0;
  std::mutex own_ev_lock;
  uint64_t own_seq = 0;
  std::mutex* ev_lock = &own_ev_lock;
  uint64_t* ev_seq = &own_seq;
  int phrase_index = -1;
  struct Session* session = nullptr;
  bool realtime = false;  // a WS phrase: startup hold pinned 0, T9 start target without the hold
  bool skip_opening = true;  // control skip_opening_silence (DEMO: off)
  bool normalize = true;
  std::optional<bool> emotion_preset;
  int post_cur_job = 0;                         // the consumer reads jobs in order
  std::map<int, std::deque<PostItem>> post_later;  // publications of later jobs, held until their turn
  std::optional<PostItem> post_end;
  bool hold_released_marked = false;
  int hold_chunks = 0;
  // Metrics (conductor writes before the final publication; read by the post stage after it).
  int64_t m_audio_tokens = 0, m_lm_steps = 0, m_segments = 0;
  std::optional<double> hold_pinned;  // control_json startup_hold_seconds (a WS phrase: 0)
};

// A realtime WebSocket session (api.py receive_input + batch_input_sentences): the L9b WsInputGate
// and WsPlanner, one generation slot; each phrase is an internal request emitting as the session.
struct Session {
  bool normalize = true;
  uint64_t id = 0;
  uint64_t user_data = 0;
  uint32_t flags = 0;
  Voice* voice = nullptr;
  std::string request_id;
  uint64_t salt = 0;
  bool seeded = false;
  lm::SamplerSettings sampler;
  double speed = 1.0;
  int format = KT2N_FORMAT_PCM;
  int64_t deadline_ns = 0;
  double opened_s = 0;
  std::mutex ev_lock;
  uint64_t seq = 0;
  // Planner state (under kt2n_engine::sess_mu).
  textplan::WsInputGate gate;
  textplan::WsPlanner planner;
  std::atomic<bool> phrase_running{false};  // set by the post thread without sess_mu (lock order)
  int phrases = 0;
  double wake_ms = 0;  // next planner step (ms since opened); 0 = now
  std::atomic<bool> dirty{true};
  bool planner_done = false;
  bool terminal = false;
  bool cancelled = false;
  int cancel_status = KT2N_ERR_CANCELLED;
  std::vector<uint64_t> phrase_ids;
  std::optional<post::PeakNorm> peak;  // one streaming peak-norm state per session, across phrases
};

void session_phrase_done(kt2n_engine* e, Request* q);

// A queued clone build (kt2n_voice_build -> L11 VoiceService on a voice thread).
struct BuildJob {
  uint64_t id = 0;
  uint64_t user_data = 0;
  uint32_t flags = 0;
  std::vector<uint8_t> audio;
  std::string audio_format;
  std::vector<float> pcm;
  int pcm_rate = 24000;
  std::vector<float> consent;
  std::string transcript, language, identity_key;
  int64_t deadline_ns = 0;
  double queued_s = 0;
  std::atomic<bool> cancelled{false};
  bool terminal = false;
  std::mutex ev_lock;
  uint64_t seq = 0;
};

struct Row {
  uint64_t key = 0;
  Request* req = nullptr;
  int job = 0;
  std::vector<int32_t> audio;  // S3Gen tokens (vocoder view)
  pol::OpeningSkip skip;
  bool skip_armed = false;
  pol::CommitState c;
  bool finished = false;
  lm::Finish finish = lm::Finish::kNone;
  int lm_steps = 0;
  bool in_active = true;
  bool final_launched = false;
  bool closed = false;
  uint64_t next_seq = 0, pub_seq = 0;
  struct Done {
    std::vector<float> pcm;
    bool final = false;
    bool failed = false;  // the render failed: nothing is published, but the row's order advances
    int commit_tokens = 0;
    double gpu_ms = 0, launch_ms = 0, launched_s = 0;
    size_t batch = 0;
    int graph_nodes = 0;
    bool graph_captured = false;
  };
  std::map<uint64_t, Done> reorder;  // seq -> finished render output, published in seq order
  int async_inflight = 0;
  double admitted_s = 0;
  bool first_token_marked = false;
  int t9_deferred_turns = 0;
  bool noise_extended = false;
  std::vector<int32_t> lm_head;  // first 64 generated ids (gpu.n_opening_skip lm_head)
};

struct Inflight {  // what the conductor keeps of a launched call (not its token lists)
  struct RowRef {
    uint64_t row = 0;
    uint64_t seq = 0;
    int commit_tokens = 0;
  };
  std::vector<RowRef> rows;
  std::vector<bool> finals;
  std::vector<bool> asyncs;
  double launched_s = 0;
  render::StreamRole stream = render::StreamRole::kSUrgent;
};

}  // namespace
}  // namespace kt2n::engine

// =====================================================================================================
struct kt2n_engine {
  // Config.
  kt2n::engine::policy::Levers lv;
  uint32_t flags = 0;
  uint64_t device_reserved_bytes = 0;
  int max_streams = 10;
  int row_cap = 10;  // rows served at once (engine_rows); <= max_streams
  std::string row_cap_why = "seats";
  std::unique_ptr<kt2n::engine::LmBackend> lm;
  std::unique_ptr<kt2n::engine::RenderBackend> render;
  std::unique_ptr<kt2n::engine::TextFront> text;
  std::vector<std::unique_ptr<kt2n::engine::Voice>> voices;  // append-only; id = index + 1 (under voices_mu)
  std::mutex voices_mu;
  std::optional<kt2n::voice::Catalogue> catalogue;  // built-ins (L11 load gate, nearest-flagship)
  std::deque<std::function<void()>> admin_q;         // conductor jobs (TIMING_POLICIES §1 step 2), under mu

  // Shared state.
  std::mutex mu;
  std::condition_variable cv;  // conductor wake-up
  std::unordered_map<uint64_t, std::unique_ptr<kt2n::engine::Request>> requests;
  std::deque<kt2n::engine::Request*> incoming;  // submitted, waiting for the conductor
  uint64_t next_id = 1;
  std::atomic<bool> stop{false};
  std::atomic<int> sharing{1};      // startup_sharing()
  std::atomic<int> steady{0};       // d_steady_requests()

  // Events.
  std::mutex ev_mu;
  std::condition_variable ev_cv;
  std::condition_variable ev_drained;
  size_t pollers = 0;  // entered polls, under ev_mu; destroy drains them before freeing ev_cv
  std::deque<kt2n::engine::EventRec*> events;
  std::map<std::pair<uint64_t, uint64_t>, std::unique_ptr<kt2n::engine::EventRec>> outstanding;
  int efd = -1;

  // Pools.
  std::mutex text_mu, post_mu;
  std::condition_variable text_cv, post_cv;
  std::deque<std::function<void()>> text_q;
  std::deque<kt2n::engine::Request*> post_q;
  std::vector<std::thread> threads;

  // Conductor-owned.
  std::vector<kt2n::engine::Request*> lanes;
  size_t lane_cursor = 0;
  size_t lane_cursor_cont = 0;  // capacity mode: the round-robin cursor over continuing requests
  bool capacity_order = false;  // KT2N_CAPACITY_ORDER=1, capacity mode only (seats > rows): continuing requests first
  std::unordered_map<uint64_t, std::unique_ptr<kt2n::engine::Row>> rows;
  std::vector<uint64_t> active;
  std::unordered_map<uint64_t, kt2n::engine::Inflight> inflight;
  std::deque<std::pair<uint64_t, double>> dw_queue;  // (row key, queued at)
  std::optional<double> last_whole_render;
  uint64_t next_row = 1, next_call = 1;
  int inflight_segments = 0;

  // Clone builds (voice threads; the conductor grants each build its turn: CF_BUILD_INTERLEAVE gate).
#if KT2N_ENGINE_WITH_VOICE
  std::unique_ptr<kt2n::voice::VoiceBuilder> vbuilder;
  std::unique_ptr<kt2n::voice::VoiceService> vservice;
  // Optional host executor; an empty executor selects the concrete VoiceService.
  std::function<kt2n::voice::VoiceBuildOutcome(const kt2n::voice::VoiceBuildRequest&)> voice_build_exec;
  // identity_key -> voice id: production's ActiveSafeLRU (inline_voice_lru entries, in-use voices never evicted;
  // L11 InlineVoiceIndex). Constructed with the configured size in kt2n_engine_create.
  std::unique_ptr<kt2n::voice::InlineVoiceIndex> inline_index;
#endif
  std::mutex build_mu;
  std::condition_variable build_cv;
  std::deque<std::shared_ptr<kt2n::engine::BuildJob>> build_q;
  std::unordered_map<uint64_t, std::shared_ptr<kt2n::engine::BuildJob>> builds;
  // Registered voices awaiting conductor initialization, under mu. Shared owners
  // survive worker return and let fatal closure finish every accepted build.
  std::unordered_map<uint64_t, std::vector<std::shared_ptr<kt2n::engine::BuildJob>>> pending_voice_builds;
  int builds_running = 0;
  bool build_waiting = false;    // a voice thread waits for the gate
  double build_wait_since = 0;
  bool build_go = false;
  std::optional<double> last_build_done_s;
  int clone_slots = 4, clone_queue = 16;
  double clone_timeout_s = 120.0;

  // Realtime sessions (session thread).
  std::mutex sess_mu;
  std::condition_variable sess_cv;
  std::unordered_map<uint64_t, std::unique_ptr<kt2n::engine::Session>> sessions;

  // Stats.
  std::atomic<uint64_t> done{0}, failed{0}, cancelled{0}, renders{0}, render_rows{0}, lm_steps{0};
  std::atomic<int> lm_rows{0}, render_inflight{0}, active_requests{0}, queued_requests{0};
  std::atomic<uint64_t> previews_due{0};     // 1-step preview condition met (L1: no live path fires it)
  std::atomic<uint64_t> lm_backpressure{0};  // LM submits refused (slot release lag); the job retries
  std::atomic<uint64_t> render_backpressure{0};  // open_row refused (render slot / pool pages); the job retries
  bool poll = true;                      // conductor poll mode (KT2N_CONDUCTOR_POLL=0 turns it off; execution only)
  // Arm (execution only, off by default): a commit group that mixes second commits (commit 1, which the startup hold
  // waits for) with later commits is launched as two calls, the commit-1 rows on S_urgent and the rest on S_bulk,
  // instead of one call that stream_for() puts on S_urgent because of its most urgent row. Same rows, same commit
  // boundaries, same turn. KT2N_SPLIT_URGENT=1.
  bool split_urgent = false;
  std::atomic<uint64_t> split_urgent_calls{0};
  // Arm (execution only, off by default; verifier finding 19:00 on the RC1 env): a commit of a request whose startup
  // hold has not released yet (its first audio waits for it) runs on S_urgent even when its commit index is >= 2.
  // stream_for() sends commit 2+ to S_bulk; with the RC1 env (decode priority 2 above S_bulk, bulk call merging) those
  // renders took 156-190 ms wall p50 for 39-51 ms of GPU work in the 18:40 window, and the ~4% of requests whose hold
  // waited for them set the N=20 p99 (270-299 ms vs 176-201 ms with production defaults). A group that mixes such
  // rows with later commits of released requests (and has no commit-0/1 row) is launched as two calls, as in
  // split_urgent. Same rows, same commit boundaries, same turn. KT2N_HOLD_URGENT=1 (the render service also
  // prewarms the urgent twins of single-row steady graphs, so the promoted calls capture no graph live).
  // Level 2 (KT2N_HOLD_URGENT=2, sprint 6 20:10, measured separately): also the FINAL flush of a row whose request is
  // still held (a one- or two-word input whose hold can only release on its final render).
  int hold_urgent = 0;
  // KT2N_BUFFER_URGENT_MS (execution-only arm, 0 = off): a commit or final of a playing stream whose delivery clock
  // (production's DeliveryClock, d_buffer) holds less than this much audio runs on S_urgent.
  double buffer_urgent_ms = 0;
  std::atomic<uint64_t> buffer_urgent_rows{0}, buffer_urgent_finals{0};
  // KT2N_BUFFER_ROLE=aux (two-level arm, L0 23:04): those commits go to S_aux, which the renderer then runs at a
  // middle priority (KT2N_RENDER_AUX_PRIO_LEVEL), instead of S_urgent; finals stay on their natural stream.
  bool buffer_to_aux = false;
  std::atomic<uint64_t> buffer_mid_calls{0};
  std::atomic<uint64_t> hold_urgent_calls{0}, hold_urgent_rows{0}, hold_urgent_splits{0}, hold_urgent_finals{0};
  // Default-off execution experiment: briefly let an already-launched bulk render compete at the LM's priority.
  // It never changes commits, T9, admission order, the startup hold, or any model setting.
  bool lm_render_yield = false;
  int lm_render_yield_base_priority = 0;
  bool lm_render_yield_active = false;  // conductor only
  uint64_t lm_render_yield_call = 0;
  double lm_render_yield_since = 0, lm_render_yield_cooldown = 0, lm_render_yield_checked = 0;
  std::atomic<uint64_t> lm_render_yield_episodes{0}, lm_render_yield_switches{0};
  int text_nice = 0;                     // nice of the text threads (KT2N_TEXT_NICE; lab arm, execution only)
  std::atomic<uint64_t> poll_idle{0};    // poll-mode passes that found no finished LM step
  int render_voice_errors = 0;               // render add_voice failures at create (stats)
  std::atomic<long> voice_store_full{0};     // loaded / built voices refused by a full LM or render voice store
  std::string render_voice_error;
  std::atomic<uint64_t> turn_ns{0}, turns{0}, turn_ns_max{0};  // conductor CPU time per turn (host on the token path)
  std::atomic<uint64_t> turns_over_1ms{0}, turns_over_5ms{0}, turn_max_decide_ns{0};
  // Render calls per turn (diagnostic, sprint 5): turns that launched commit groups, the groups, and the groups beyond
  // the first that went to the same stream in the same turn (could have been one bigger call); the same for finals.
  std::atomic<uint64_t> commit_turns{0}, commit_groups{0}, commit_same_stream_extra{0}, commit_extra_rows{0};
  std::atomic<uint64_t> commit_mixed_groups{0}, commit_mixed_later_rows{0};  // commit-1 rows + later rows in one group
  std::atomic<uint64_t> final_turns{0}, final_calls{0}, final_extra_calls{0};
  std::atomic<int> turn_max_rows{0};
  // LM step interval while rows stay live (diagnostic: the LM's ms per step under the engine's real load, prefills and
  // GPU sharing included, idle gaps excluded), split by rows per step: 1-4, 5-8, 9-12, 13-16, 17-20, 21+.
  int64_t lm_prev_ev_ns = 0;  // conductor only
  std::atomic<uint64_t> lm_iv_ns[8]{}, lm_iv_steps[8]{};  // 1-4 ... 17-20, 21-32, 33-48, 49+ rows
  std::atomic<bool> backend_failed{false};  // latched under mu with request admission
  std::string fatal;  // under mu; diagnostic only
  std::vector<std::string> levers_unknown;
  std::atomic<bool> release_pending{false};
  std::atomic<bool> voice_sweep{false};       // an evicted voice may have no request left: release_voices()
  std::atomic<long> lm_voices_released{0};    // LM voice store ranges given back (L3 release_voice)
  std::atomic<long> render_voices_released{0};  // render table rows + prefix pages given back (L7 remove_voice)
};

namespace kt2n::engine {
namespace {

using Engine = kt2n_engine;

// ------------------------------------------------------------------------------------- events
void push_event(Engine* e, EventRec* r) {
  {
    std::lock_guard<std::mutex> lk(e->ev_mu);
    e->events.push_back(r);
  }
  e->ev_cv.notify_all();
  const uint64_t one = 1;
  if (e->efd >= 0) (void)!write(e->efd, &one, sizeof one);
}

EventRec* new_event(Request* q, int kind) {
  auto* r = new EventRec();
  r->ev.kind = kind;
  r->ev.user_data = q->user_data;
  r->ev.at_ns = mono_ns();
  r->ev.segment_index = -1;
  r->ev.phrase_index = q->phrase_index;
  r->ev.sample_begin = -1;
  r->ev.engine_chunk_index = -1;
  return r;
}

// Assigns the per-id sequence number and queues the event under the id's emit lock, so seq order
// is queue order even when the conductor and a post thread emit for the same id.
void emit(Engine* e, Request* q, EventRec* r) {
  std::lock_guard<std::mutex> lk(*q->ev_lock);
  if (q->terminal.load() && r->ev.kind != KT2N_EVENT_DONE && r->ev.kind != KT2N_EVENT_PHRASE_DONE) {
    delete r;
    return;
  }
  r->ev.id = q->ev_id;
  r->ev.seq = (*q->ev_seq)++;
  push_event(e, r);
}

void timeline(Engine* e, Request* q, const char* name, const std::string& fields, int segment = -1) {
  if (!q->timeline) return;
  EventRec* r = new_event(q, KT2N_EVENT_TIMELINE);
  r->ev.segment_index = segment;
  r->name = name;
  r->fields = "{" + fields + "}";
  emit(e, q, r);
}

// ------------------------------------------------------------------------------------- post
void schedule_post(Engine* e, Request* q) {
  // q->mu held by the caller.
  if (q->post_scheduled) return;
  q->post_scheduled = true;
  {
    std::lock_guard<std::mutex> lk(e->post_mu);
    e->post_q.push_back(q);
  }
  e->post_cv.notify_one();
}

bool encoded_format(int f);
const char* format_name(int f);

void emit_audio(Engine* e, Request* q, const std::string& bytes, int64_t samples, bool final) {
  // Post thread.
  if (!q->headers_sent) {
    // Response metadata before the first audio (engine.h KT2N_EVENT_HEADERS).
    q->headers_sent = true;
    EventRec* h = new_event(q, KT2N_EVENT_HEADERS);
    h->name = "headers";
    h->fields = std::string("{\"sample_rate\":24000,\"format\":\"") + format_name(q->format) +
                "\",\"speech_mode\":" + (q->demo ? "\"demo\"" : "null") +
                (q->output == KT2N_OUTPUT_COMPLETE ? ",\"complete_bytes\":" + std::to_string(bytes.size()) : "") + "}";
    emit(e, q, h);
  }
  EventRec* r = new_event(q, KT2N_EVENT_AUDIO);
  r->data = bytes;
  r->ev.sample_begin = q->samples_out;
  r->ev.samples = int32_t(samples);
  r->ev.engine_chunk_index = q->chunk_index++;
  r->ev.is_final_chunk = final ? 1 : 0;
  // engine.h opening_skipped_tokens (L8b 9e948c97): production's control.n_opening_skipped_tokens, final before
  // the first audio (the conductor sets it when segment 0's opening closes, before the vocoder sees a token;
  // post_push gave this thread happens-before). api.py's realtime reserve reads it at the first chunk (the
  // skip-aware per-listener term, kitten_tts/api.py:1631). Set on every AUDIO event.
  r->ev.opening_skipped_tokens = q->opening_skipped;
  q->samples_out += samples;
  emit(e, q, r);
}

void emit_done(Engine* e, Request* q, int status, const std::string& message) {
  // Post thread.
  if (q->terminal) return;
  q->terminal = true;
  EventRec* r = new_event(q, q->session ? KT2N_EVENT_PHRASE_DONE : KT2N_EVENT_DONE);
  r->ev.status = status;
  if (status != KT2N_OK) {
    set_err(&r->error, status, message);
    r->ev.error = &r->error;
  }
  if (status == KT2N_OK)
    e->done++;
  else if (status == KT2N_ERR_CANCELLED || status == KT2N_ERR_DEADLINE_EXCEEDED)
    e->cancelled++;
  else
    e->failed++;
  e->active_requests--;
  emit(e, q, r);
  if (q->session) session_phrase_done(e, q);
}

bool encoded_format(int f) { return f == KT2N_FORMAT_MP3 || f == KT2N_FORMAT_OPUS || f == KT2N_FORMAT_AAC || f == KT2N_FORMAT_FLAC; }
const char* format_name(int f) {
  switch (f) {
    case KT2N_FORMAT_MP3: return "mp3";
    case KT2N_FORMAT_OPUS: return "opus";
    case KT2N_FORMAT_AAC: return "aac";
    case KT2N_FORMAT_FLAC: return "flac";
    case KT2N_FORMAT_WAV: return "wav";
    case KT2N_FORMAT_F32: return "f32";
    default: return "pcm";
  }
}

// One levelled publication -> the response's bytes (PCM16 LE; streaming WAV header first; or the encoder's
// header first, then whatever it finalises now).
std::string bytes_of(Request* q, const post::Pcm& lvl) {
  std::string bytes;
#if KT2N_ENGINE_WITH_AV
  if (q->enc) {
    if (!q->header_sent) bytes = q->enc->take_header();
    q->header_sent = true;
    std::string error;
    q->enc->push(lvl, bytes, &error);
    return bytes;
  }
#endif
  if (q->format == KT2N_FORMAT_WAV && !q->header_sent) bytes = post::streaming_wav_header();
  q->header_sent = true;
  post::pcm16_bytes_from_float(lvl, bytes);
  return bytes;
}
std::string bytes_end(Request* q) {
  std::string bytes;
#if KT2N_ENGINE_WITH_AV
  if (q->enc) {
    if (!q->header_sent) bytes = q->enc->take_header();
    q->header_sent = true;
    std::string error;
    q->enc->finish(bytes, &error);
    return bytes;
  }
#endif
  if (q->format == KT2N_FORMAT_WAV && !q->header_sent) bytes = post::streaming_wav_header();
  q->header_sent = true;
  return bytes;
}

// Startup hold release event: held/target ms, chunks, reached, mode + sharing (adaptive).
void mark_hold(Engine* e, Request* q, int64_t held_samples, bool reached) {
  if (!q->timeline || q->realtime) return;
  timeline(e, q, KT2N_TL_STARTUP_HOLD_RELEASED,
           "\"held_ms\":" + jnum(double(held_samples) * 1000.0 / 24000.0) + ",\"target_ms\":" +
               jnum(double(q->hold->target_samples()) * 1000.0 / 24000.0) + ",\"chunks\":" +
               std::to_string(q->hold_chunks) + ",\"reached\":" + (reached ? "true" : "false") +
               ",\"mode\":\"adaptive\",\"sharing\":" + std::to_string(q->hold->peak_sharing()));
}

std::string f32_bytes(const std::vector<float>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

void publish_streamed(Engine* e, Request* q, std::vector<float> pub) {
  // One production publication (after ResponsePost): startup hold -> level -> bytes.
  if (pub.empty()) return;
  if (q->format == KT2N_FORMAT_F32 && q->output == KT2N_OUTPUT_COMPLETE) {
    // KT2N_FORMAT_F32 (L8b adapter): production's engine -> api.py boundary. A complete file leaves the
    // engine unheld and unlevelled; api.py takes the whole-file peak and writes the header.
    emit_audio(e, q, f32_bytes(pub), int64_t(pub.size()), false);
    return;
  }
  if (q->format != KT2N_FORMAT_F32 && (q->output == KT2N_OUTPUT_COMPLETE || q->output == KT2N_OUTPUT_STREAM_FILE)) {
    q->whole.insert(q->whole.end(), pub.begin(), pub.end());
    return;
  }
  post::HoldObservation obs;
  obs.listeners = std::max(1, e->sharing.load());
  obs.steady_joiner = e->steady.load() > 0;
  obs.trimmed_ms_so_far = q->rpost ? q->rpost->trimmed_ms() : std::nullopt;
  obs.opening_skipped_tokens = q->opening_skipped;
  q->hold_chunks += 1;
  post::Pcm out = q->hold->feed(pub, obs);
  // KT2N_HOLD_URGENT: the first audio has left the hold, so later commits no longer gate first audio. This also
  // covers a pinned-0 hold (WS phrases, startup_hold_seconds=0), which is a passthrough and never reports
  // released() (RC1 fix 21:08: before it, every WS phrase commit was promoted to S_urgent).
  if (!out.empty()) q->awaiting_hold.store(false, std::memory_order_relaxed);
  if (q->hold->released() && !q->hold_released_marked) {
    q->hold_released_marked = true;
    q->awaiting_hold.store(false, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> ck(q->clock_mu);
      q->clock.release(mono_s());
    }
    mark_hold(e, q, int64_t(out.size()), true);
  }
  if (out.empty()) return;
  if (q->format == KT2N_FORMAT_F32) {
    // Held but not levelled: api.py's normalize_audio + peak norm + PCM16 / encoders run as in production.
    emit_audio(e, q, f32_bytes(out), int64_t(out.size()), false);
    return;
  }
  post::Pcm lvl;
  if (post::level_publication(out, (q->peak_ptr ? q->peak_ptr : (q->peak ? &*q->peak : nullptr)), lvl) != post::NormalizeStatus::kOk) return;
  emit_audio(e, q, bytes_of(q, lvl), int64_t(lvl.size()), false);
}

// KT2N_EVENT_METRICS (production control.metrics_snapshot, the scalars api.py reads / logs).
void emit_metrics(Engine* e, Request* q) {
  // Post thread.
  EventRec* r = new_event(q, KT2N_EVENT_METRICS);
  r->name = "metrics";
  int und;
  double und_ms;
  {
    std::lock_guard<std::mutex> ck(q->clock_mu);
    und = q->clock.underruns();
    und_ms = q->clock.underrun_ms();
  }
  std::ostringstream o;
  o << "{\"producer_seconds\":" << jnum(mono_s() - q->started_s) << ",\"audio_tokens\":" << q->m_audio_tokens
    << ",\"lm_steps\":" << q->m_lm_steps << ",\"segments\":" << q->m_segments
    << ",\"opening_silence_tokens_skipped\":" << std::max(0, q->opening_skipped - q->kept_tokens)
    << ",\"emo_applied\":" << ((q->plan && q->plan->decision.emotion) ? 1 : 0)
    << ",\"audio_seconds\":" << jnum(double(q->samples_out) / 24000.0)
    << ",\"d_underruns\":" << und << ",\"d_underrun_ms\":" << jnum(und_ms) << "}";
  r->fields = o.str();
  emit(e, q, r);
}

void finish_response(Engine* e, Request* q, int status) {
  // Post thread.
  if (status != KT2N_OK) {
    emit_done(e, q, status, status == KT2N_ERR_CANCELLED ? "request_cancelled" : "speech generation failed");
    return;
  }
  emit_metrics(e, q);
  if (q->format == KT2N_FORMAT_F32) {
    std::string bytes;
    int64_t n = 0;
    if (q->output != KT2N_OUTPUT_COMPLETE) {
      post::Pcm rest = q->hold->finish();
      if (!q->hold_released_marked && !rest.empty()) {
        q->hold_released_marked = true;
        q->awaiting_hold.store(false, std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> ck(q->clock_mu);
          q->clock.release(mono_s());
        }
        mark_hold(e, q, int64_t(rest.size()), false);
      }
      bytes = f32_bytes(rest);
      n = int64_t(rest.size());
    }
    emit_audio(e, q, bytes, n, true);  // the final marker (possibly empty)
  } else if (q->output == KT2N_OUTPUT_COMPLETE || q->output == KT2N_OUTPUT_STREAM_FILE) {
    // The whole-file levelling (api.py _peak_norm_whole_wav / _whole_file_levelled, demo peak formula).
    const float peak = post::demo_peak(q->whole);
    std::string bytes;
    int64_t n = int64_t(q->whole.size());
    if (encoded_format(q->format)) {
#if KT2N_ENGINE_WITH_AV
      post::Pcm lv;
      if (q->format == KT2N_FORMAT_FLAC)
        post::whole_file_flac(q->whole, peak, lv);
      else
        post::whole_file_lossy(q->whole, peak, 0.891, lv);
      bytes = bytes_of(q, lv);  // two statements: the push must come before the finish
      bytes += bytes_end(q);
#endif
    } else {
      std::vector<int16_t> pcm16;
      post::demo_int16_with_peak(q->whole, peak, pcm16);
      if (q->format == KT2N_FORMAT_WAV) bytes = post::complete_wav_header(uint64_t(pcm16.size()) * 2);
      bytes.append(reinterpret_cast<const char*>(pcm16.data()), pcm16.size() * 2);
    }
    emit_audio(e, q, bytes, n, true);
  } else {
    post::Pcm rest = q->hold->finish();
    if (!q->hold_released_marked && !rest.empty()) {
      q->hold_released_marked = true;
      q->awaiting_hold.store(false, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> ck(q->clock_mu);
        q->clock.release(mono_s());
      }
      mark_hold(e, q, int64_t(rest.size()), false);
    }
    std::string bytes;
    int64_t n = 0;
    if (!rest.empty()) {
      post::Pcm lvl;
      if (post::level_publication(rest, (q->peak_ptr ? q->peak_ptr : (q->peak ? &*q->peak : nullptr)), lvl) ==
          post::NormalizeStatus::kOk) {
        bytes = bytes_of(q, lvl);
        n = int64_t(lvl.size());
      }
    }
    bytes += bytes_end(q);
    emit_audio(e, q, bytes, n, true);
  }
  emit_done(e, q, KT2N_OK, "");
}

// Conductor (L3's driver thread, and the thread that drives L7): give the LM store space, the render table row and
// the prefix pages of evicted voices back once no request holds them (refs 0). Requests hold a ref from
// register_request to their release, so every row of the voice was submitted and closed by then; L3 keeps the LM
// range until those rows' slots are released, and L7 refuses (-2, retried on a later turn) while a row or a prefix
// build of the voice is still in flight. lm_voice / render_voice are set to -1 so a request that raced the eviction
// (looked the voice up just before it) fails at admission (custom_voice_service_error) instead of using a released
// id.
void release_voices(Engine* e) {
  bool retry = false;
  std::lock_guard<std::mutex> vlk(e->voices_mu);
  for (auto& v : e->voices) {
    if (!v->evicted.load() || v->refs.load() != 0) continue;
    if (!v->lm_released) {
      const int lmv = v->lm_voice.load();
      if (lmv >= 0) {  // < 0: never built (store full / failed build), or still building in an admin job
        if (e->lm->release_voice(lmv) == 0) e->lm_voices_released++;
        v->lm_voice = -1;
        v->lm_released = true;
      }
    }
    if (!v->render_released && v->render_voice >= 0) {
      std::string err;
      const int rc = e->render->remove_voice(v->render_voice, &err);
      if (rc == -2) {
        retry = true;  // busy: a row or a prefix build of it is still in flight
      } else {
        if (rc == 0) e->render_voices_released++;
        v->render_voice = -1;
        v->render_released = true;
      }
    }
  }
  if (retry) e->voice_sweep = true;
}

void post_item(Engine* e, Request* q, PostItem it) {
  // Post thread (one per request at a time); it.job == q->post_cur_job.
  if (q->post_begun_job < 0 && q->kept_tokens > 0) q->rpost->set_kept_opening_tokens(q->kept_tokens);
  if (it.job > q->post_begun_job) {
    q->post_begun_job = it.job;
    post::SeamContext seam;
    seam.demo_request = q->demo;
    seam.demo_chunked = true;
    seam.dc_mode = q->plan ? q->plan->decision.dc_mode : 0;
    seam.lane_sealed = it.lane_sealed;
    seam.lane_jobs = it.lane_jobs;
    seam.lane_known = true;
    q->rpost->begin_segment(it.job, seam);
  }
  std::vector<post::Pcm> pubs;
  if (!it.pcm.empty()) pubs.push_back(q->rpost->push(it.pcm));
  if (it.final) {
    const post::SeamEdgeTrimmer* seam = q->rpost->seam();
    const bool seam_finishing = seam && !seam->finished();
    for (auto& p : q->rpost->end_segment(it.has_more)) pubs.push_back(std::move(p));
    if (seam_finishing && q->timeline) {
      // runtime.py _dc_note_seam (:19663, called right after seam.finish at :20908): what this segment's edges
      // lost, on a traced request. seam_released_ms only when Q_HOLD_RELEASE_MS released audio (0 live: never).
      timeline(e, q, KT2N_TL_SEAM_TRIMMED,
               "\"segment_index\":" + std::to_string(it.job) +
                   ",\"seam_head_trimmed_ms\":" + jnum(double(seam->head_trimmed_samples()) * 1000.0 / 24000.0) +
                   ",\"seam_tail_trimmed_ms\":" + jnum(double(seam->tail_trimmed_samples()) * 1000.0 / 24000.0),
               it.job);
    }
    if (!it.has_more) pubs.push_back(q->rpost->finish());
    q->post_cur_job += 1;
  }
  q->dropped_samples.store(q->rpost->dropped_samples());
  for (auto& p : pubs) publish_streamed(e, q, std::move(p));
}

void post_run(Engine* e, Request* q) {
  // Items are popped under q->mu and processed without it: post state is touched only by the one post thread
  // that holds this request (post_scheduled), so the conductor's post_push never waits on trims / encoders.
  for (;;) {
    PostItem it;
    {
      std::lock_guard<std::mutex> lk(q->mu);
      if (q->post_q.empty()) {
        q->post_scheduled = false;
        return;
      }
      it = std::move(q->post_q.front());
      q->post_q.pop_front();
    }
    if (q->terminal) continue;
    if (q->cancelled.load()) {
      emit_done(e, q, q->cancel_status,
                q->cancel_status == KT2N_ERR_CANCELLED ? "request_cancelled"
                : q->fail_reason.empty()               ? "speech generation failed"
                                                       : "speech generation failed: " + q->fail_reason);
      continue;
    }
    if (it.end && it.status != KT2N_OK) {
      finish_response(e, q, it.status);
      continue;
    }
    if (it.end) {
      q->post_end = std::move(it);
    } else if (it.job != q->post_cur_job) {
      q->post_later[it.job].push_back(std::move(it));
    } else {
      post_item(e, q, std::move(it));
    }
    // Drain the held publications of the jobs whose turn came.
    for (;;) {
      auto later = q->post_later.find(q->post_cur_job);
      if (later == q->post_later.end() || later->second.empty()) break;
      PostItem next = std::move(later->second.front());
      later->second.pop_front();
      if (later->second.empty()) q->post_later.erase(later);
      post_item(e, q, std::move(next));
    }
    if (q->post_end && q->post_cur_job >= q->post_end->lane_jobs) {
      q->post_end.reset();
      finish_response(e, q, KT2N_OK);
    }
  }
}

void post_worker(Engine* e) {
  for (;;) {
    Request* q;
    {
      std::unique_lock<std::mutex> lk(e->post_mu);
      e->post_cv.wait(lk, [&] { return e->stop.load() || !e->post_q.empty(); });
      if (e->stop.load() && e->post_q.empty()) return;
      q = e->post_q.front();
      e->post_q.pop_front();
    }
    post_run(e, q);
  }
}

// ------------------------------------------------------------------------------------- text
void text_worker(Engine* e) {
  // Lab arm: text threads below the conductor / post threads under CPU contention (a thread's own nice).
  if (e->text_nice > 0) setpriority(PRIO_PROCESS, id_t(syscall(SYS_gettid)), e->text_nice);
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lk(e->text_mu);
      e->text_cv.wait(lk, [&] { return e->stop.load() || !e->text_q.empty(); });
      if (e->stop.load() && e->text_q.empty()) return;
      task = std::move(e->text_q.front());
      e->text_q.pop_front();
    }
    task();
  }
}

// A text task for request q: counted so a released request is never freed under a running task.
void text_enqueue(Engine* e, std::function<void()> f) {
  {
    std::lock_guard<std::mutex> lk(e->text_mu);
    e->text_q.push_back(std::move(f));
  }
  e->text_cv.notify_one();
}

// e->mu held by the caller.
void text_task_locked(Engine* e, Request* q, std::function<void()> f) {
  q->text_tasks += 1;
  text_enqueue(e, [e, q, f = std::move(f)] {
    if (!q->cancelled.load() && !e->backend_failed.load()) f();
    {
      std::lock_guard<std::mutex> lk(e->mu);
      q->text_tasks -= 1;
    }
    e->release_pending = true;
    e->cv.notify_all();
  });
}

// Normalize planned segment `index` into jobs (text thread). Appends under e->mu.
void normalize_into_jobs(Engine* e, Request* q, size_t index) {
  if (q->cancelled.load() || e->backend_failed.load()) return;
  std::vector<std::u32string> chunks;
  std::string error;
  try {
    chunks = e->text->normalize_segment(*q->plan, index);
  } catch (const std::exception& ex) {
    error = ex.what();
  }
  std::vector<std::unique_ptr<Job>> made;
  if (error.empty()) {
    const bool demo = q->demo;
    std::vector<size_t> lens;
    for (auto& c : chunks) lens.push_back(c.size());
    std::vector<int> budgets;
    std::unique_lock<std::mutex> lk(e->mu);
    if (q->cancelled.load() || e->backend_failed.load()) return;
    if (index == 0) q->total_chunks = int(q->plan->raw_segments.size()) - 1 + int(chunks.size());
    const int start = int(q->jobs_created);
    q->jobs_created += chunks.size();
    lk.unlock();
    if (demo) budgets = textplan::demo_budgets(lens, q->total_chunks, start);
    for (size_t i = 0; i < chunks.size(); ++i) {
      auto j = std::make_unique<Job>();
      j->ordinal = start + int(i);
      j->raw_segment = int(index);
      j->text = chunks[i];
      j->suffix = e->text->suffix(chunks[i], q->plan->decision.emotion);
      j->budget = demo ? budgets[i] : textplan::tagged_segment_budget(chunks[i].size());
      made.push_back(std::move(j));
    }
  }
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (q->cancelled.load() || e->backend_failed.load()) return;
    if (!error.empty()) {
      q->text_status = KT2N_ERR_SPEECH_GENERATION_FAILED;
      q->text_error = error;
    }
    for (auto& j : made) q->new_jobs.push_back(std::move(j));
    q->segments_normalized = index + 1;
    q->sealed = q->segments_normalized >= q->plan->raw_segments.size();
    q->normalizing = false;
  }
  e->cv.notify_all();
}

void plan_request(Engine* e, Request* q) {
  if (q->cancelled.load() || e->backend_failed.load()) return;
  std::unique_ptr<PlannedRequest> plan;
  std::string error;
  try {
    plan = std::make_unique<PlannedRequest>(e->text->plan(q->raw, q->voice->language, q->emotion_preset, q->normalize));
  } catch (const std::exception& ex) {
    error = ex.what();
  }
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (q->cancelled.load() || e->backend_failed.load()) return;
    if (!plan || plan->raw_segments.empty()) {
      q->text_status = KT2N_ERR_SPEECH_GENERATION_FAILED;
      q->text_error = error.empty() ? "empty plan" : error;
      q->sealed = true;
    } else {
      q->plan = std::move(plan);
      q->normalizing = true;
    }
  }
  if (q->plan) {
    timeline(e, q, KT2N_TL_TEXT_SEGMENTED,
             "\"characters\":" + std::to_string(q->raw.size()) +
                 ",\"segments\":" + std::to_string(q->plan->raw_segments.size()));
    normalize_into_jobs(e, q, 0);  // segment 0 on the request's own text task
    timeline(e, q, KT2N_TL_TEXT_NORMALIZED, "\"segment_index\":0", 0);
  } else {
    e->cv.notify_all();
  }
}

// ------------------------------------------------------------------------------------- conductor
std::optional<double> d_buffer_ms(Request* q, double now) {
  std::optional<double> b;
  {
    std::lock_guard<std::mutex> ck(q->clock_mu);
    b = q->clock.buffer_ms(now);
  }
  if (!b) return b;
  return *b - double(q->dropped_samples.load()) * 1000.0 / pol::kSampleRate;
}

void acknowledge(Engine* e, Request* q, Job* j) {
  if (j->acknowledged) return;
  j->acknowledged = true;
  if (j->charged) {
    q->inflight -= 1;
    e->inflight_segments -= 1;
  }
}

void post_push(Engine* e, Request* q, PostItem it) {
  std::lock_guard<std::mutex> lk(q->mu);
  q->post_q.push_back(std::move(it));
  schedule_post(e, q);
}

void fail_request(Engine* e, Request* q, int status, const std::string& reason = "") {
  if (q->generation_complete) return;
  q->generation_complete = true;
  q->fail_reason = reason;  // internal detail for logs / the lab (the public message stays production's)
  // Its rows stop decoding and rendering (the next turn cancels them); DONE carries this status.
  q->cancel_status = status;
  q->cancelled = true;
  PostItem it;
  it.end = true;
  it.status = status;
  post_push(e, q, std::move(it));
}

// Publication of one row's render output, in the row's order (conductor).
void publish_row(Engine* e, Row* row, std::vector<float> pcm, bool final) {
  Request* q = row->req;
  Job* job = q->jobs[size_t(row->job)].get();
  const double now = mono_s();
  if (q->cancelled.load()) return;
  if (!pcm.empty()) {
    if (!job->first_pub) job->first_pub = now;
    job->published_samples += int64_t(pcm.size());
    job->preview_ready = true;
    {
      std::lock_guard<std::mutex> ck(q->clock_mu);
      q->clock.publish(now, int64_t(pcm.size()));
    }
  }
  const bool last_job = q->c_sealed && size_t(job->ordinal) + 1 >= q->jobs.size();
  PostItem it;
  it.lane_jobs = int(q->jobs.size());
  it.lane_sealed = q->c_sealed;
  it.job = job->ordinal;
  it.pcm = std::move(pcm);
  it.final = final;
  it.has_more = !last_job;
  post_push(e, q, std::move(it));
  if (final) {
    job->done = true;
    acknowledge(e, q, job);
    if (last_job) {
      q->generation_complete = true;
      PostItem end;
      end.end = true;
      end.lane_jobs = int(q->jobs.size());
      post_push(e, q, std::move(end));
    }
  } else if (!job->acknowledged && job->charged) {
    // Consider the next text segment, on every engine chunk.
    if (pol::segment_prefetch_due(last_job, job->done, job->published_samples, job->text.size(),
                                  e->inflight_segments, e->lv)) {
      acknowledge(e, q, job);
      timeline(e, q, KT2N_TL_SEGMENT_PREFETCH_RELEASED,
               "\"segment_index\":" + std::to_string(job->ordinal) +
                   ",\"received_ms\":" + jnum(double(job->published_samples) * 1000.0 / 24000.0),
               job->ordinal);
    }
  }
}

void close_row_if_done(Engine* e, Row* row) {
  if (row->in_active || !row->final_launched || row->async_inflight > 0 || !row->reorder.empty()) return;
  if (row->pub_seq < row->next_seq) return;
  if (!row->closed) {
    e->render->close_row(row->key);
    row->closed = true;
  }
}

void drop_closed_rows(Engine* e) {
  for (auto it = e->rows.begin(); it != e->rows.end();) {
    if (it->second->closed) {
      it->second->req->live_rows -= 1;
      it = e->rows.erase(it);
    } else {
      ++it;
    }
  }
}

void process_outs(Engine* e, std::vector<RenderOut>& outs, bool drop_closed) {
  for (RenderOut& o : outs) {
    auto it = e->inflight.find(o.call_id);
    if (it == e->inflight.end()) continue;
    Inflight& inf = it->second;
    e->render_inflight--;
    for (size_t i = 0; i < inf.rows.size(); ++i) {
      const Inflight::RowRef& rr = inf.rows[i];
      auto rit = e->rows.find(rr.row);
      if (rit == e->rows.end()) continue;
      Row* row = rit->second.get();
      if (inf.asyncs[i]) {
        row->async_inflight -= 1;
        if (row->async_inflight == 0) row->c.inflight = false;
      }
      std::vector<float> pcm = i < o.pcm.size() ? std::move(o.pcm[i]) : std::vector<float>{};
      Row::Done d;
      if (o.status != 0) {
        fail_request(e, row->req, KT2N_ERR_SPEECH_GENERATION_FAILED,
                     "render call failed (status " + std::to_string(o.status) + (o.error.empty() ? "" : ": " + o.error) + ")");
        d.failed = true;
      }
      d.pcm = std::move(pcm);
      d.final = bool(inf.finals[i]);
      d.commit_tokens = rr.commit_tokens;
      d.gpu_ms = o.gpu_ms;
      d.launch_ms = o.launch_ms;
      d.launched_s = inf.launched_s;
      d.batch = inf.rows.size();
      d.graph_nodes = o.graph_nodes;
      d.graph_captured = o.graph_captured;
      row->reorder[rr.seq] = std::move(d);
      while (!row->reorder.empty() && row->reorder.begin()->first == row->pub_seq) {
        auto node = std::move(row->reorder.begin()->second);
        row->reorder.erase(row->reorder.begin());
        row->pub_seq += 1;
        const bool fin = node.final;
        const size_t n = node.pcm.size();
        if (node.failed) continue;
        if (row->req->timeline) {
          timeline(e, row->req, fin ? KT2N_TL_S3GEN_FINAL_PUBLICATION : KT2N_TL_S3GEN_PUBLICATION,
                   "\"s3gen_wall_ms\":" + jnum((mono_s() - node.launched_s) * 1000.0) +
                       ",\"s3gen_gpu_ms\":" + jnum(node.gpu_ms) + ",\"s3gen_launch_ms\":" + jnum(node.launch_ms) +
                       ",\"s3gen_queue_depth\":" + std::to_string(e->render_inflight.load()) +
                       ",\"batch_size\":" + std::to_string(node.batch) +
                       ",\"commit_tokens\":" + std::to_string(node.commit_tokens) +
                       ",\"graph_nodes\":" + std::to_string(node.graph_nodes) +
                       ",\"graph_captured\":" + (node.graph_captured ? "true" : "false") +
                       ",\"audio_ms\":" + jnum(double(n) * 1000.0 / 24000.0) +
                       ",\"audio_samples\":" + std::to_string(n) +
                       ",\"segment_index\":" + std::to_string(row->job),
                   row->job);
        }
        publish_row(e, row, std::move(node.pcm), fin);  // after its timeline mark: DONE may follow at once
      }
      if (!e->backend_failed.load()) close_row_if_done(e, row);
    }
    e->inflight.erase(it);
  }
  if (drop_closed) drop_closed_rows(e);  // not inside a launch: the caller may still hold Row pointers
}

void collect_renders(Engine* e, bool wait) {
  std::vector<RenderOut> outs;
  e->render->collect(&outs, wait);
  process_outs(e, outs, true);
}

// KT2N_BUFFER_URGENT_MS: true when the request is already playing (its first audio has left the hold) and its
// delivery clock's buffer (d_buffer: published audio not yet played on the listener's timeline) is below the arm's
// threshold. The PREREG audible-pause rule counts a >= 250 ms gap between chunks while < 1 s is buffered (l0-perf
// 22:45: RC1 N=20, 706 of 6,457 streams, 697 of them a 2nd-publication render queued behind ~690 ms of bulk work).
bool buffer_low(Engine* e, Request* q) {
  if (e->buffer_urgent_ms <= 0 || q->awaiting_hold.load(std::memory_order_relaxed)) return false;
  std::lock_guard<std::mutex> ck(q->clock_mu);
  if (!q->clock.started()) return false;
  const auto b = q->clock.buffer_ms(mono_s());
  return b && *b < e->buffer_urgent_ms;
}

void launch(Engine* e, std::vector<Row*> rows, render::Phase phase, bool async, bool finals,
            bool force_urgent = false, int force_role = -1) {
  if (rows.empty()) return;
  RenderCall call;
  call.call_id = e->next_call++;
  call.async = async;
  Inflight inf;
  int min_commit = 1 << 30;
  render::Phase call_phase = phase;
  for (Row* row : rows) {
    RenderRowReq rr;
    rr.row = row->key;
    rr.voice = row->req->voice->render_voice;
    rr.commit_index = row->c.commit_index;
    rr.committed_tokens = row->c.committed;
    rr.seq = row->next_seq++;
    if (finals) {
      rr.phase = row->req->demo ? render::Phase::kWholeRender : render::Phase::kFinalFlush;
      rr.finalize = true;
      rr.tokens = row->audio;
      rr.tokens.insert(rr.tokens.end(), 3, pol::kSilenceToken);
      rr.commit_tokens = int(row->audio.size());
      call_phase = rr.phase;
    } else {
      const bool first = row->c.commit_index == 0 && row->c.committed == 0;
      rr.phase = first ? render::Phase::kFirstCommit : render::Phase::kSteadyCommit;
      rr.commit_tokens = row->c.next;
      rr.tokens.assign(row->audio.begin(), row->audio.begin() + (row->c.next + pol::kLookahead));
      if (!first) call_phase = render::Phase::kSteadyCommit;
    }
    min_commit = std::min(min_commit, rr.commit_index);
    inf.finals.push_back(finals);
    inf.asyncs.push_back(async);
    if (async) {
      row->async_inflight += 1;
      row->c.inflight = !finals || row->c.inflight;
    }
    if (row->req->timeline)
      timeline(e, row->req, KT2N_TL_S3GEN_STARTED,
               "\"commit_tokens\":" + std::to_string(rr.commit_tokens) +
                   ",\"prefix_tokens\":" + std::to_string(rr.committed_tokens) +
                   ",\"segment_index\":" + std::to_string(row->job) +
                   ",\"batch_size\":" + std::to_string(rows.size()) +
                   ",\"async_launch\":" + (async ? "true" : "false"),
               row->job);
    call.rows.push_back(std::move(rr));
    if (!finals) pol::after_commit(row->c, e->lv);  // the launch advances the row's state at once
  }
  call.stream = render::stream_for(call_phase, min_commit);
  if (force_urgent) {  // KT2N_HOLD_URGENT: commits (level 1) or finals (level 2) a startup hold still waits for
    call.stream = render::StreamRole::kSUrgent;
    e->hold_urgent_calls++;
  }
  if (force_role >= 0) call.stream = render::StreamRole(force_role);  // KT2N_BUFFER_ROLE=aux
  inf.stream = call.stream;
  inf.launched_s = mono_s();
  std::string err;
  const uint64_t id = call.call_id;
  for (const auto& r : call.rows) inf.rows.push_back({r.row, r.seq, r.commit_tokens});
  e->inflight.emplace(id, std::move(inf));
  e->render_inflight++;
  e->renders++;
  e->render_rows += call.rows.size();
  if (e->render->launch(std::move(call), &err) != 0) {
    std::vector<RenderOut> failed(1);
    failed[0].call_id = id;
    failed[0].status = KT2N_ERR_SPEECH_GENERATION_FAILED;
    failed[0].error = "launch: " + err;
    process_outs(e, failed, false);
  }
}

int startup_sharing(Engine* e) {
  // admission snapshot queued_requests: requests holding a lane, not complete.
  int n = 0;
  for (Request* q : e->lanes)
    if (!q->generation_complete && !q->cancelled.load()) ++n;
  return std::max(1, n);
}

void dw_service(Engine* e) {
  if (e->dw_queue.empty()) return;
  const double now = mono_s();
  pol::DwInput in;
  in.now_s = now;
  in.oldest_queued_s = e->dw_queue.front().second;
  in.last_whole_render_s = e->last_whole_render;
  std::optional<double> low;
  for (uint64_t k : e->active) {
    Row* r = e->rows[k].get();
    if (r->req->demo) continue;
    in.any_realtime = true;
    auto b = d_buffer_ms(r->req, now);
    if (!b) in.waiting = true;
    Job* j = r->req->jobs[size_t(r->job)].get();
    if (j->ordinal == 0 && !j->preview_ready) in.waiting = true;
    const double v = b.value_or(0.0);
    low = low ? std::min(*low, v) : v;
  }
  in.low_buffer_ms = low;
  const pol::DwReason why = pol::dw_due(in, e->lv);
  if (why == pol::DwReason::kNone) return;
  std::vector<Row*> batch;
  if (why == pol::DwReason::kIdle) {
    for (auto& [k, t] : e->dw_queue) batch.push_back(e->rows[k].get());
    e->dw_queue.clear();
  } else {
    // The oldest + the queued renders of the same voice in its length bucket (_final_flush_buckets).
    Row* oldest = e->rows[e->dw_queue.front().first].get();
    std::vector<std::pair<int, int>> same;
    std::unordered_map<int, Row*> by_id;
    int idx = 0;
    for (auto& [k, t] : e->dw_queue) {
      Row* r = e->rows[k].get();
      if (r->req->voice == oldest->req->voice) {
        same.push_back({idx, int(r->audio.size())});
        by_id[idx] = r;
      }
      ++idx;
    }
    for (const auto& b : pol::final_flush_buckets(same, e->lv)) {
      bool has_oldest = false;
      for (int i : b) has_oldest |= by_id[i] == oldest;
      if (!has_oldest) continue;
      for (int i : b) batch.push_back(by_id[i]);
    }
    std::set<uint64_t> taken;
    for (Row* r : batch) taken.insert(r->key);
    std::deque<std::pair<uint64_t, double>> rest;
    for (auto& p : e->dw_queue)
      if (!taken.count(p.first)) rest.push_back(p);
    e->dw_queue.swap(rest);
  }
  for (Row* r : batch)
    timeline(e, r->req, KT2N_TL_DW_RENDER, "\"reason\":\"" + std::string(pol::dw_reason_name(why)) + "\"", r->job);
  launch(e, batch, render::Phase::kWholeRender, true, true);
  e->last_whole_render = mono_s();
}

void admit(Engine* e) {
  // Lanes in arrival order; prune completed ones.
  {
    std::lock_guard<std::mutex> lk(e->mu);
    while (!e->incoming.empty()) {
      e->lanes.push_back(e->incoming.front());
      e->lanes.back()->in_lanes = true;
      e->incoming.pop_front();
    }
  }
  std::vector<Request*> keep;
  for (Request* q : e->lanes) {
    bool live;
    {
      std::lock_guard<std::mutex> lk(e->mu);
      for (auto& j : q->new_jobs) q->jobs.push_back(std::move(j));
      q->new_jobs.clear();
      q->c_sealed = q->sealed;
      const bool voice_dead = q->voice->evicted.load() && q->voice->lm_voice.load() < 0;
      live = !q->generation_complete && !q->cancelled.load() && q->text_status == 0 && !voice_dead;
      if (voice_dead && !q->generation_complete)
        fail_request(e, q, KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR);  // its LM prefix could not be built
      else if (q->cancelled.load() && !q->generation_complete)
        fail_request(e, q, q->cancel_status);  // cancelled before or while waiting for a row
      else if (q->text_status != 0 && !q->generation_complete)
        fail_request(e, q, q->text_status);
    }
    if (live)
      keep.push_back(q);
    else
      q->in_lanes = false;
  }
  e->lanes.swap(keep);
  e->sharing = startup_sharing(e);
  int active_count = int(e->active.size());
  const int capacity = std::min(e->max_streams, e->row_cap) - active_count;
  if (capacity <= 0 || e->lanes.empty()) return;
  std::vector<pol::LaneView> views;
  std::vector<bool> continuing;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    for (Request* q : e->lanes) {
      views.push_back({q->next_index < q->jobs.size() && q->voice->lm_voice.load() >= 0, q->inflight});
      continuing.push_back(q->next_index > 0);
    }
  }
  std::vector<int> taken;
  if (e->capacity_order && e->row_cap < e->max_streams) {
    // Experiment arm (KT2N_CAPACITY_ORDER=1; off by default = production's single round-robin), capacity mode
    // only (more seats than rows; production never runs this way: its seats = its decode rows). With many
    // queued requests, round-robin makes a request that is already playing wait for its NEXT segment behind
    // the queued new ones (slot smoke 22:08 with a 96-page lab pool, ~1 row: 5 takes ran dry before their next
    // segment was admitted). Continuing requests are offered first, with production's round-robin among them;
    // new requests share what is left with production's round-robin and cursor. Slot A/B 22:18 (128 pages,
    // 3 rows, N=32): 0 dry takes in BOTH arms, so not shown to help yet; measured only in an exclusive window.
    taken = pol::admission_take_capacity(views, continuing, capacity, &e->lane_cursor_cont, &e->lane_cursor, e->lv);
  } else {
    taken = pol::admission_take(views, capacity, &e->lane_cursor, e->lv);
  }
  for (int li : taken) {
    Request* q = e->lanes[size_t(li)];
    Job* j;
    {
      std::lock_guard<std::mutex> lk(e->mu);
      j = q->jobs[q->next_index].get();
      q->next_index += 1;
    }
    if (!j->acknowledged) {
      q->inflight += 1;
      j->charged = true;
      e->inflight_segments += 1;
    }
    auto row = std::make_unique<Row>();
    row->key = e->next_row++;
    row->req = q;
    row->job = j->ordinal;
    row->c = pol::new_commit_state(e->lv, q->demo);
    row->skip_armed = j->ordinal == 0 && !q->demo && q->skip_opening;  // SKIP_OPENING_SILENCE: segment 0, not DEMO
    row->admitted_s = mono_s();
    lm::RowRequest rr;
    rr.tag = row->key;
    rr.voice = q->voice->lm_voice;
    rr.suffix.assign(j->suffix.begin(), j->suffix.end());
    rr.sampler = q->sampler;
    rr.seed = q->salt + uint64_t(j->ordinal);
    rr.budget = j->budget;
    rr.demo = q->demo;
    rr.demo_chunked = true;  // every live request is demo-chunked
    std::string err;
    auto put_back = [&] {
      std::lock_guard<std::mutex> lk(e->mu);
      q->next_index -= 1;
      if (j->charged) {
        q->inflight -= 1;
        j->charged = false;
        e->inflight_segments -= 1;
      }
    };
    // Render state first: a row the renderer cannot take yet (no slot, or its pool pages cannot be reserved)
    // waits in admission instead of failing mid-request (native back-pressure; production has no page pool).
    const int orc = e->render->open_row(row->key, q->voice->render_voice, rr.seed, j->budget, q->demo, &err);
    if (orc == kRenderBackpressure) {
      put_back();
      e->render_backpressure++;
      break;  // keep arrival order: nothing behind it is admitted this turn
    }
    if (orc != 0) {
      fail_request(e, q, KT2N_ERR_SPEECH_GENERATION_FAILED, "render open_row: " + err);
      continue;
    }
    if (e->lm->submit(rr, &err) != 0) {
      // LM back-pressure (a slot not yet released after read-back): put the job back, retry next loop.
      e->render->close_row(row->key);
      put_back();
      e->lm_backpressure++;
      continue;
    }
    timeline(e, q, KT2N_TL_SCHEDULER_ADMITTED,
             "\"admission_batch\":" + std::to_string(taken.size()) + ",\"active_before\":" +
                 std::to_string(active_count) + ",\"queue_ms\":" + jnum((mono_s() - q->started_s) * 1000.0) +
                 ",\"segment_index\":" + std::to_string(j->ordinal),
             j->ordinal);
    e->active.push_back(row->key);
    q->live_rows += 1;
    e->rows.emplace(row->key, std::move(row));
    ++active_count;
  }
}

// Normaliser thread trigger: segment k >= 1 after segment 0 published >= 1.5 s (or is done),
// at most 1 normalized segment ahead of admission.
void maybe_normalize_next(Engine* e, Request* q) {
  std::lock_guard<std::mutex> lk(e->mu);
  if (!q->plan || q->sealed || q->normalizing || q->cancelled.load() || q->jobs.empty()) return;
  Job* j0 = q->jobs[0].get();
  const bool first_ok = j0->done || j0->published_samples >= int64_t(e->lv.normalize_start_audio_s * 24000) ||
                        (mono_s() - q->started_s) >= e->lv.normalize_wait_max_s;
  if (!first_ok) return;
  if (q->jobs_created - q->next_index > size_t(e->lv.normalize_lookahead)) return;
  q->normalizing = true;
  const size_t index = q->segments_normalized;
  text_task_locked(e, q, [e, q, index] { normalize_into_jobs(e, q, index); });
}

void finalize_rows(Engine* e, const std::vector<Row*>& done, int decoded) {
  std::vector<Row*> stream_finals;
  for (Row* r : done) {
    pol::opening_restore(r->skip, r->audio);
    r->in_active = false;
    r->final_launched = true;
    if (r->req->timeline && !r->audio.empty()) {
      std::string bytes(r->audio.size() * 4, '\0');
      for (size_t i = 0; i < r->audio.size(); ++i) {
        const uint32_t v = uint32_t(r->audio[i]);
        for (int k = 0; k < 4; ++k) bytes[i * 4 + size_t(k)] = char((v >> (8 * k)) & 0xff);
      }
      timeline(e, r->req, KT2N_TL_LM_TOKENS_FINALIZED,
               "\"segment_index\":" + std::to_string(r->job) + ",\"audio_token_count\":" +
                   std::to_string(r->audio.size()) + ",\"audio_token_sha256\":\"" +
                   kt2n::sha256_hex(bytes.data(), bytes.size()) + "\"",
               r->job);
    }
    if (r->audio.empty()) {
      // "KittenTTS 0.9 generated no audio tokens" fails the request.
      fail_request(e, r->req, KT2N_ERR_SPEECH_GENERATION_FAILED, "generated no audio tokens");
      close_row_if_done(e, r);
      continue;
    }
    Job* j = r->req->jobs[size_t(r->job)].get();
    j->lm_done = true;
    r->req->m_audio_tokens += int64_t(r->audio.size());
    r->req->m_lm_steps += r->lm_steps;
    r->req->m_segments += 1;
    timeline(e, r->req, KT2N_TL_LM_SEGMENT_FINISHED,
             "\"lm_steps\":" + std::to_string(r->lm_steps) + ",\"segment_index\":" + std::to_string(r->job) +
                 ",\"finish_reason\":\"" + lm::finish_name(r->finish) + "\",\"audio_tokens\":" +
                 std::to_string(r->audio.size()),
             r->job);
    if (r->req->demo) {
      e->dw_queue.push_back({r->key, mono_s()});
      // Prefetch at language-model completion: acknowledge at LM end so the next chunk may be admitted.
      Request* q = r->req;
      const bool last = q->c_sealed && size_t(j->ordinal) + 1 >= q->jobs.size();
      if (!last && e->inflight_segments + 1 + 1 <= e->lv.request_batch_size) acknowledge(e, q, j);
    } else {
      stream_finals.push_back(r);
    }
  }
  if (stream_finals.empty()) return;
  // Group by geometry owner, bucket by length.
  std::map<int64_t, std::vector<Row*>> by_owner;
  for (Row* r : stream_finals) by_owner[r->req->voice->geometry].push_back(r);
  const bool async = pol::launch_async(decoded, e->lv);
  uint64_t calls = 0;
  for (auto& [owner, rows] : by_owner) {
    std::vector<std::pair<int, int>> items;
    for (size_t i = 0; i < rows.size(); ++i) items.push_back({int(i), int(rows[i]->audio.size())});
    for (const auto& b : pol::final_flush_buckets(items, e->lv)) {
      std::vector<Row*> batch;
      for (int i : b) batch.push_back(rows[size_t(i)]);
      if (e->hold_urgent >= 2 || e->buffer_urgent_ms > 0) {
        // KT2N_HOLD_URGENT=2: finals of requests whose startup hold has not released go to S_urgent (one call), the
        // rest of the bucket stays one S_bulk call. Same rows, same buckets, same turn. KT2N_BUFFER_URGENT_MS: the
        // same for finals of playing streams whose buffer is low.
        std::vector<Row*> held, rest;
        int low = 0;
        for (Row* r : batch) {
          const bool h = e->hold_urgent >= 2 && r->req->awaiting_hold.load(std::memory_order_relaxed);
          const bool lowbuf = !h && !e->buffer_to_aux && buffer_low(e, r->req);
          low += lowbuf;
          (h || lowbuf ? held : rest).push_back(r);
        }
        if (low) {
          e->buffer_urgent_rows += uint64_t(low);
          e->buffer_urgent_finals += uint64_t(low);
        }
        if (!held.empty()) {
          e->hold_urgent_rows += uint64_t(held.size()) - uint64_t(low);
          e->hold_urgent_finals += uint64_t(held.size()) - uint64_t(low);
          if (!rest.empty()) e->hold_urgent_splits++;
          launch(e, held, render::Phase::kFinalFlush, async, true, true);
          ++calls;
          if (rest.empty()) continue;
          batch.swap(rest);
        }
      }
      launch(e, batch, render::Phase::kFinalFlush, async, true);
      ++calls;
    }
  }
  if (calls) {
    e->final_turns++;
    e->final_calls += calls;
    e->final_extra_calls += calls - 1;
  }
}

// One owner turn after one decode step (TIMING_POLICIES §1 steps 10-13).
void turn(Engine* e, const std::vector<lm::TokenEvent>& step) {
  const double now = mono_s();
  int decoded = 0;
  for (const auto& ev : step) {
    auto it = e->rows.find(ev.tag);
    if (it == e->rows.end()) continue;
    Row* r = it->second.get();
    if (!r->in_active) continue;
    if (ev.finish != lm::Finish::kCancelled) ++decoded;  // _bs_active_rows: rows decoded this turn
    r->lm_steps = ev.index;
    if (!r->first_token_marked) {
      r->first_token_marked = true;
      timeline(e, r->req, KT2N_TL_LM_PREFILL_FINISHED,
               "\"batch_size\":1,\"segment_index\":" + std::to_string(r->job) +
                   ",\"lm_wall_ms\":" + jnum((now - r->admitted_s) * 1000.0),
               r->job);
    }
    if (ev.finish == lm::Finish::kCancelled) {
      r->finished = true;
      r->finish = ev.finish;
      continue;
    }
    if (r->lm_head.size() < 64) r->lm_head.push_back(ev.token - lm::kAudioIdBase);
    if (ev.token >= lm::kAudioIdBase && ev.token < lm::kAudioIdBase + lm::kAudioVocab) {
      const int32_t s3 = ev.token - lm::kAudioIdBase;
      const bool was_closed = r->skip.closed;
      const bool traced = !was_closed && r->req->timeline && r->skip_armed && r->audio.empty();
      std::vector<int32_t> skipped_before, pending_before;
      if (traced) {
        skipped_before = r->skip.skipped;
        pending_before = r->skip.pending;
      }
      pol::opening_admit(r->skip, r->skip_armed, s3, r->audio, e->lv);
      if (traced && r->skip.closed) {
        auto list = [](const std::vector<int32_t>& v) {
          std::string o = "[";
          for (size_t i = 0; i < v.size(); ++i) o += (i ? "," : "") + std::to_string(v[i]);
          return o + "]";
        };
        std::string in_set = "[";
        for (size_t i = 0; i < skipped_before.size(); ++i)
          in_set += std::string(i ? "," : "") + (pol::opening_quiet_id(skipped_before[i]) || skipped_before[i] == 1466 ||
                                                         skipped_before[i] == 3653
                                                     ? "true"
                                                     : "false");
        in_set += "]";
        timeline(e, r->req, KT2N_TL_N_OPENING_SKIP,
                 "\"cap\":" + std::to_string(e->lv.skip_cap) + ",\"skipped\":" + list(skipped_before) +
                     ",\"pending\":" + list(pending_before) + ",\"closing\":" + std::to_string(s3) +
                     ",\"skipped_in_set\":" + in_set + ",\"lm_head\":" + list(r->lm_head) +
                     ",\"segment_index\":" + std::to_string(r->job),
                 r->job);
      }
      if (!was_closed && r->skip.closed && r->job == 0) {
        // Known before any audio of the request exists (the opening closes before the vocoder sees a token);
        // the post stage applies it at its first item (post_item), after a post_push gave happens-before.
        r->req->opening_skipped = r->skip.skipped_tokens;
        r->req->kept_tokens = r->skip.kept_tokens;
      }
    }
    if (ev.finish != lm::Finish::kNone) {
      r->finished = true;
      r->finish = ev.finish;
    } else if (!r->noise_extended && r->lm_steps >= r->req->jobs[size_t(r->job)]->budget) {
      // a row that outlives its budget gets flow noise for the extra frames.
      r->noise_extended = true;
      std::string err;
      e->render->extend_noise(r->key, &err);
    }
  }
  e->lm_steps++;
  // Active rows in admission order; cancelled rows leave (step 6).
  std::vector<Row*> act;
  for (uint64_t k : e->active) {
    Row* r = e->rows[k].get();
    // Production adds an admitted row to `active` after its prefill (which samples its first token); L3
    // prefills beside decode, so a row joins the turn's decisions with its first token.
    if (!r->first_token_marked && !r->req->cancelled.load()) continue;
    if (r->req->cancelled.load() && !r->finished) {
      e->lm->cancel(r->key);
      r->finished = true;
      r->finish = lm::Finish::kCancelled;
    }
    if (r->finish == lm::Finish::kCancelled) continue;  // step 6: cancelled rows leave before any decision
    act.push_back(r);
  }
  // D steady count (one-commit steady age): live requests (a job not done) whose
  // listener has played for >= D_JOIN_WINDOW_MS = 1000 ms (d_clock_age_ms).
  {
    int steady = 0;
    for (Request* q : e->lanes) {
      if (q->cancelled.load()) continue;
      std::lock_guard<std::mutex> ck(q->clock_mu);
      const auto first = q->clock.first_at();
      if (first && (now - *first) * 1000.0 >= 1000.0) ++steady;
    }
    e->steady = steady;
  }
  // Step 12: T9 rows.
  std::vector<pol::T9Row> t9;
  const int sharing_rows = int(act.size());
  for (Row* r : act) {
    Job* j = r->req->jobs[size_t(r->job)].get();
    pol::T9Row t;
    t.finished = r->finished;
    t.held_short = r->c.held;
    t.segment_index = j->ordinal;
    t.preview_ready = j->preview_ready;
    t.published_s = double(j->published_samples) / 24000.0;
    t.start_target_s = r->req->hold_pinned ? std::max(0.0, *r->req->hold_pinned)
                                           : pol::start_target_seconds(sharing_rows, r->req->realtime, e->lv);
    if (j->first_pub) t.reservoir_s = double(j->published_samples) / 24000.0 - (now - *j->first_pub);
    t.dropped_s = double(r->req->dropped_samples.load()) / 24000.0;
    const int n = int(r->audio.size());
    const bool prev = pol::preview_due(j->preview_ready, n, r->c.committed, r->c.held, r->c.next, r->c.commit_index,
                                       e->lv);
    t.row_ready = prev || pol::commit_ready(r->c, n);
    t.backlog = n - r->c.next - pol::kLookahead;
    t9.push_back(t);
  }
  const bool xsched = (e->flags & KT2N_CFG_X_SCHED) != 0;  // experiment arm: no T9 deferral
  pol::T9Result sel;
  if (xsched) {
    for (int i = 0; i < int(t9.size()); ++i) sel.selected.push_back(i);
  } else {
    sel = pol::t9_acoustic_rows(t9, e->lv);
  }
  for (int i : sel.deferred_rows) act[size_t(i)]->t9_deferred_turns += 1;
  std::vector<Row*> rows;
  for (int i : sel.selected) {
    Row* r = act[size_t(i)];
    rows.push_back(r);
    if (r->t9_deferred_turns) {
      const auto& t = t9[size_t(i)];
      timeline(e, r->req, KT2N_TL_T9_DEFERRAL_ENDED,
               "\"segment_index\":" + std::to_string(r->job) + ",\"deferred_turns\":" +
                   std::to_string(r->t9_deferred_turns) + ",\"reservoir_s\":" +
                   (t.reservoir_s ? jnum(*t.reservoir_s) : std::string("null")),
               r->job);
      r->t9_deferred_turns = 0;
    }
  }
  // Previews: ported condition; L1 found no live path that fires it. Native renders it
  // as a 1-step kPreview call once L7 serves that phase; until then the condition is counted.
  for (Row* r : rows) {
    Job* j = r->req->jobs[size_t(r->job)].get();
    if (pol::preview_due(j->preview_ready, int(r->audio.size()), r->c.committed, r->c.held, r->c.next,
                         r->c.commit_index, e->lv))
      e->previews_due++;
  }
  // Step 12b: choose the row schedule on the rows passed to the acoustic step, sharing read once.
  const int sharing = startup_sharing(e);
  for (Row* r : rows) pol::d_prepare(r->c, int(r->audio.size()), sharing, e->lv);
  // Step 12c: ready commits -> groups -> order -> launch.
  std::vector<pol::CommitRow> ready, partners;
  std::unordered_map<int, Row*> by_id;
  int id = 0;
  for (Row* r : rows) {
    if (r->finish == lm::Finish::kCancelled) continue;
    pol::CommitRow cr;
    cr.id = id;
    by_id[id] = r;
    ++id;
    Job* j = r->req->jobs[size_t(r->job)].get();
    cr.first = r->c.commit_index == 0 && r->c.committed == 0;
    cr.commit_index = r->c.commit_index;
    cr.token_count = int(r->audio.size());
    cr.next = r->c.next;
    cr.geometry = r->req->voice->geometry;
    if (j->first_pub) cr.reservoir_s = double(j->published_samples) / 24000.0 - (now - *j->first_pub);
    cr.d_buffer_ms = d_buffer_ms(r->req, now);
    if (pol::commit_ready(r->c, cr.token_count))
      ready.push_back(cr);
    else if (!r->c.inflight && !r->c.held)
      partners.push_back(cr);
  }
  int held = 0;
  const auto groups = xsched ? pol::plan_commit_groups_xsched(ready, e->lv)
                             : pol::plan_commit_groups(ready, partners, e->lv, &held);
  const bool async = pol::launch_async(decoded, e->lv);
  if (!groups.empty()) {
    int per_stream[render::kStreamRoles] = {};
    for (const auto& g : groups) {
      bool all_first = true;
      int min_ci = 1 << 30, n_urgent = 0, n_later = 0;
      for (int rid : g.rows) {
        const Row* r = by_id[rid];
        const bool first = r->c.commit_index == 0 && r->c.committed == 0;
        all_first &= first;
        min_ci = std::min(min_ci, r->c.commit_index);
        (first || r->c.commit_index <= 1 ? n_urgent : n_later) += 1;
      }
      if (n_urgent && n_later) {
        e->commit_mixed_groups++;
        e->commit_mixed_later_rows += uint64_t(n_later);
      }
      const int s = int(render::stream_for(all_first ? render::Phase::kFirstCommit : render::Phase::kSteadyCommit, min_ci));
      if (per_stream[s]++ > 0) {
        e->commit_same_stream_extra++;
        e->commit_extra_rows += g.rows.size();
      }
    }
    e->commit_turns++;
    e->commit_groups += groups.size();
  }
  for (const auto& g : groups) {
    std::vector<Row*> batch;
    for (int rid : g.rows) batch.push_back(by_id[rid]);
    if (e->buffer_to_aux && e->buffer_urgent_ms > 0) {
      // Two-level arm: low-buffer rows of playing streams (not commit 0 / 1, not held) leave the group as one S_aux
      // call at the middle priority; the rest of the group launches as it would without them.
      std::vector<Row*> mid, rest;
      for (Row* r : batch) {
        const bool natural = (r->c.commit_index == 0 && r->c.committed == 0) || r->c.commit_index <= 1;
        const bool in_hold = e->hold_urgent > 0 && r->req->awaiting_hold.load(std::memory_order_relaxed);
        (!natural && !in_hold && buffer_low(e, r->req) ? mid : rest).push_back(r);
      }
      if (!mid.empty()) {
        e->buffer_urgent_rows += uint64_t(mid.size());
        e->buffer_mid_calls++;
        launch(e, mid, render::Phase::kSteadyCommit, async, false, false, int(render::StreamRole::kSAux));
        batch.swap(rest);
        if (batch.empty()) continue;
      }
    }
    if (e->split_urgent || e->hold_urgent > 0 || e->buffer_urgent_ms > 0) {
      // urgent = commit 0 / 1 rows (stream_for) + with KT2N_HOLD_URGENT the rows of requests still under the startup
      // hold, + with KT2N_BUFFER_URGENT_MS the rows of playing streams whose buffer is low (promoted); later = the rest.
      std::vector<Row*> urgent, later;
      int natural = 0, promoted = 0, low = 0;
      for (Row* r : batch) {
        const bool first = r->c.commit_index == 0 && r->c.committed == 0;
        if (first || r->c.commit_index <= 1) {
          ++natural;
          urgent.push_back(r);
        } else if (e->hold_urgent > 0 && r->req->awaiting_hold.load(std::memory_order_relaxed)) {
          ++promoted;
          urgent.push_back(r);
        } else if (!e->buffer_to_aux && buffer_low(e, r->req)) {
          ++promoted;
          ++low;
          urgent.push_back(r);
        } else {
          later.push_back(r);
        }
      }
      if (low) e->buffer_urgent_rows += uint64_t(low);
      if (promoted > low) e->hold_urgent_rows += uint64_t(promoted - low);
      // Split when the arm asks for it (split_urgent), or when only promoted rows make the group urgent (a commit-0/1
      // row already puts the whole call on S_urgent, as without the arm).
      if (!urgent.empty() && !later.empty() && (e->split_urgent || natural == 0)) {
        if (e->split_urgent) e->split_urgent_calls++;
        if (natural == 0) e->hold_urgent_splits++;
        launch(e, urgent, render::Phase::kSteadyCommit, async, false, promoted > 0 && natural == 0);
        launch(e, later, render::Phase::kSteadyCommit, async, false);
        continue;
      }
      if (later.empty() && natural == 0 && promoted > 0) {
        launch(e, batch, render::Phase::kSteadyCommit, async, false, true);
        continue;
      }
    }
    launch(e, batch, render::Phase::kSteadyCommit, async, false);
  }
  // Step 13: finished rows -> finals; they leave `active`.
  std::vector<Row*> done;
  std::vector<uint64_t> still;
  for (uint64_t k : e->active) {
    Row* r = e->rows[k].get();
    if (r->finished) {
      if (r->finish == lm::Finish::kCancelled) {
        r->in_active = false;
        r->final_launched = true;
        close_row_if_done(e, r);
      } else {
        done.push_back(r);
      }
    } else {
      still.push_back(r->key);
    }
  }
  e->active.swap(still);
  finalize_rows(e, done, decoded);
  e->lm_rows = int(e->active.size());
}

void build_emit_locked(Engine* e, BuildJob* j, int status, uint64_t voice, const std::string& code,
                       const std::string& param, const std::string& message);

// Close admission atomically with the snapshot of accepted requests. The conductor
// stays alive after an LM failure to drain renders and reclaim released owners.
void fail_engine(Engine* e, const std::string& reason) {
  {
    std::lock_guard<std::mutex> lk(e->mu);
    e->fatal = reason;
    e->backend_failed = true;
    for (auto& [id, q] : e->requests) {
      fail_request(e, q.get(), KT2N_ERR_SERVICE_UNAVAILABLE, reason);
      q->in_lanes = false;
    }
    e->incoming.clear();
    for (const auto& [voice, jobs] : e->pending_voice_builds)
      for (const auto& job : jobs)
        build_emit_locked(e, job.get(), KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "", reason);
    e->pending_voice_builds.clear();
    e->admin_q.clear();
  }
  e->lanes.clear();
  e->active.clear();
  e->dw_queue.clear();
  e->lm_rows = 0;
  e->queued_requests = 0;
  // These admission paths take their wait mutex before mu, so never nest them
  // inside mu here. Waking with the wait mutex prevents a missed notification.
  {
    std::lock_guard<std::mutex> lk(e->build_mu);
    e->build_cv.notify_all();
  }
  {
    std::lock_guard<std::mutex> lk(e->sess_mu);
    e->sess_cv.notify_all();
  }
}

void drain_failed_rows(Engine* e) {
  // Keep a row until every already launched render call has been collected.
  // Cancelled publications need no ordering or final flush after LM failure.
  std::set<uint64_t> rendering;
  for (const auto& [id, call] : e->inflight)
    for (const auto& row : call.rows) rendering.insert(row.row);
  for (auto& [id, row] : e->rows) {
    row->in_active = false;
    row->final_launched = true;
    row->reorder.clear();
    row->pub_seq = row->next_seq;
    if (!rendering.count(id)) close_row_if_done(e, row.get());
  }
  drop_closed_rows(e);
}

// A saturated GPU has no spare throughput: this only tests whether bounded LM priority relief reduces the long
// bulk-render waits. Existing runtime switching orders later work with an event; it does not preempt queued work.
// The eight-millisecond limit is a host-request interval, not a guarantee of eight milliseconds on the GPU.
void maybe_yield_lm_to_render(Engine* e) {
  if (!e->lm_render_yield) return;
  const double now = mono_s();
  if (e->lm_render_yield_active) {
    if (e->inflight.count(e->lm_render_yield_call) && now - e->lm_render_yield_since < 0.008) return;
    e->lm->set_decode_priority(e->lm_render_yield_base_priority);
    e->lm_render_yield_active = false;
    e->lm_render_yield_cooldown = now + 0.008;
    ++e->lm_render_yield_switches;
    return;
  }
  if (now < e->lm_render_yield_cooldown || now - e->lm_render_yield_checked < 0.001) return;
  e->lm_render_yield_checked = now;
  uint64_t oldest_id = 0;
  double oldest_s = now;
  for (const auto& [id, call] : e->inflight) {
    if (call.stream != render::StreamRole::kSBulk || now - call.launched_s < 0.080) continue;
    bool low_buffer = false;
    for (const auto& ref : call.rows) {
      const auto row = e->rows.find(ref.row);
      if (row == e->rows.end()) continue;
      Request* q = row->second->req;
      if (q->demo || q->awaiting_hold.load(std::memory_order_relaxed)) continue;
      const auto b = d_buffer_ms(q, now);
      if (b && *b < 1000.0) {
        low_buffer = true;
        break;
      }
    }
    if (low_buffer && call.launched_s < oldest_s) {
      oldest_id = id;
      oldest_s = call.launched_s;
    }
  }
  if (!oldest_id) return;
  e->lm->set_decode_priority(0);  // equal to S_bulk; first/held renders and prefill keep their existing priorities
  e->lm_render_yield_active = true;
  e->lm_render_yield_call = oldest_id;
  e->lm_render_yield_since = now;
  ++e->lm_render_yield_episodes;
  ++e->lm_render_yield_switches;
}

void conductor(Engine* e) {
  std::vector<lm::TokenEvent> events;
  while (!e->stop.load()) {
    // Step 1: collect finished renders (publish in row order).
    collect_renders(e, false);
    if (e->backend_failed.load()) drain_failed_rows(e);
    // Free released requests the conductor and the post stage no longer touch.
    if (e->release_pending.exchange(false)) {
      std::lock_guard<std::mutex> lk(e->mu);
      bool again = false;
      for (auto it = e->requests.begin(); it != e->requests.end();) {
        Request* q = it->second.get();
        bool idle = false;
        if (q->release_requested.load() && !q->in_lanes && q->live_rows == 0 && q->text_tasks == 0) {
          std::lock_guard<std::mutex> qlk(q->mu);
          idle = !q->post_scheduled && q->post_q.empty();
        }
        if (idle) {
          if (--q->voice->refs == 0 && q->voice->evicted.load()) e->voice_sweep = true;
          it = e->requests.erase(it);
        } else {
          again |= q->release_requested.load();
          ++it;
        }
      }
      if (again) e->release_pending = true;
    }
    if (e->backend_failed.load()) {
      // No more LM calls, admissions or builds; retain the release sweep above.
      std::unique_lock<std::mutex> lk(e->mu);
      cv_wait_for(e->cv, lk, std::chrono::milliseconds(5));
      continue;
    }
    if (e->voice_sweep.exchange(false)) release_voices(e);
    // Deadlines and cancels of requests without rows.
    const int64_t now_ns = mono_ns();
    for (Request* q : e->lanes) {
      if (q->deadline_ns && now_ns > q->deadline_ns && !q->cancelled.load()) {
        q->cancel_status = KT2N_ERR_DEADLINE_EXCEEDED;
        q->cancelled = true;
      }
      if (q->cancelled.load() && !q->generation_complete) {
        q->generation_complete = true;
        PostItem it;
        it.end = true;
        it.status = q->cancel_status;
        post_push(e, q, std::move(it));
      }
    }
    // Step 2: non-blocking admin jobs (voice preparation: LM prefix builds of loaded voices).
    {
      std::deque<std::function<void()>> jobs;
      {
        std::lock_guard<std::mutex> lk(e->mu);
        jobs.swap(e->admin_q);
      }
      for (auto& j : jobs) j();
    }
    // Step 3: the interleaved clone build, if due.
    {
      std::lock_guard<std::mutex> lk(e->build_mu);
      if (e->build_waiting && !e->build_go) {
        const double now = mono_s();
        bool before_first = false;
        for (uint64_t k : e->active) {
          Row* r = e->rows[k].get();
          if (r->req->demo) continue;  // realtime_rows_waiting excludes whole-render rows
          if (!r->req->jobs[size_t(r->job)]->preview_ready) before_first = true;
        }
        std::optional<double> since;
        if (e->last_build_done_s) since = (now - *e->last_build_done_s) * 1000.0;
        if (pol::cf_interleave_due(!e->active.empty(), before_first, (now - e->build_wait_since) * 1000.0, since,
                                   e->lv)) {
          e->build_go = true;
          e->build_cv.notify_all();
        }
      }
    }
    // Step 4: whole-render gate.
    dw_service(e);
    // Step 8-9: admit (the LM prefills beside decode).
    admit(e);
    for (Request* q : e->lanes) maybe_normalize_next(e, q);
    // Step 10: LM. Block for a step only when rows are live; else wait for renders or new work.
    events.clear();
    std::string err;
    // One production decision pass per LM step: a tag seen twice starts the next step.
    auto run_steps = [&] {
      size_t begin = 0;
      std::set<uint64_t> seen;
      for (size_t i = 0; i <= events.size(); ++i) {
        if (i == events.size() || seen.count(events[i].tag)) {
          std::vector<lm::TokenEvent> step(events.begin() + long(begin), events.begin() + long(i));
          if (!step.empty()) {
            const int64_t t0 = mono_ns();
            turn(e, step);
            const int64_t t1 = mono_ns();
            collect_renders(e, false);
            const uint64_t dt = uint64_t(mono_ns() - t0);
            e->turn_ns += dt;
            e->turns++;
            // Stall diagnostics: how often a turn (every stream's next LM step waits on it) is long, and for the
            // longest one, how it split between the decision pass + launches (turn) and the collect.
            if (dt > 1000000) e->turns_over_1ms++;
            if (dt > 5000000) e->turns_over_5ms++;
            if (dt > e->turn_ns_max.load()) {
              e->turn_ns_max = dt;
              e->turn_max_decide_ns = uint64_t(t1 - t0);
              e->turn_max_rows = int(step.size());
            }
          }
          seen.clear();
          begin = i;
        }
        if (i < events.size()) seen.insert(events[i].tag);
      }
    };
    const bool lm_live = e->lm->live_rows() > 0 || !e->active.empty();
    if (!lm_live) e->lm_prev_ev_ns = 0;
    maybe_yield_lm_to_render(e);
    // Steps in `events` (a tag seen twice starts the next step) -> the LM step-interval stats.
    auto note_lm_interval = [&] {
      size_t steps = 1;
      std::set<uint64_t> seen;
      for (const auto& ev : events) {
        if (seen.count(ev.tag)) {
          ++steps;
          seen.clear();
        }
        seen.insert(ev.tag);
      }
      const int64_t now = mono_ns();
      if (e->lm_prev_ev_ns > 0) {
        const double rows = double(events.size()) / double(steps);
        const int b = rows <= 4    ? 0
                      : rows <= 8  ? 1
                      : rows <= 12 ? 2
                      : rows <= 16 ? 3
                      : rows <= 20 ? 4
                      : rows <= 32 ? 5
                      : rows <= 48 ? 6
                                   : 7;
        e->lm_iv_ns[b] += uint64_t(now - e->lm_prev_ev_ns);
        e->lm_iv_steps[b] += steps;
      }
      e->lm_prev_ev_ns = now;
    };
    if (e->poll && (lm_live || e->render->inflight() > 0)) {
      // Poll mode (execution only): never block inside one backend. The LM is pumped without waiting, and when
      // no step finished the loop goes round again after a short pause, so a finished render is published and a
      // new request admitted within ~25 us instead of after the next LM step (or after a blocking render wait).
      if (lm_live && e->lm->pump(&events, false, &err) < 0) {
        fail_engine(e, err);
        continue;
      }
      if (!events.empty()) note_lm_interval();
      if (events.empty()) {
        e->poll_idle++;
        for (int i = 0; i < 64; ++i) {  // ~5-10 us of pause, then a 20 us sleep
#if defined(__x86_64__)
          __builtin_ia32_pause();
#endif
        }
        const timespec ts{0, 20000};
        nanosleep(&ts, nullptr);
        continue;
      }
      run_steps();
    } else if (lm_live) {
      if (e->lm->pump(&events, true, &err) < 0) {
        fail_engine(e, err);
        continue;
      }
      if (!events.empty()) note_lm_interval();
      if (events.empty()) {
        // Nothing decoded (rows still prefilling, or none live after all): never spin.
        if (e->render->inflight() > 0) {
          collect_renders(e, true);
        } else {
          std::unique_lock<std::mutex> lk(e->mu);
          cv_wait_for(e->cv, lk, std::chrono::microseconds(200));
        }
        continue;
      }
      run_steps();
    } else if (e->render->inflight() > 0) {
      collect_renders(e, true);
    } else {
      std::unique_lock<std::mutex> lk(e->mu);
      bool work = !e->incoming.empty() || !e->admin_q.empty();
      for (Request* q : e->lanes)
        work |= q->next_index < q->jobs.size() || !q->new_jobs.empty() || q->cancelled.load();
      if (!work && !e->dw_queue.empty()) work = true;
      if (!work) cv_wait_for(e->cv, lk, std::chrono::milliseconds(5));
    }
  }
}

}  // namespace
}  // namespace kt2n::engine

namespace kt2n::engine {
namespace {

Request* register_request(Engine* e, std::unique_ptr<Request> q, uint64_t* id = nullptr) {
  Request* raw = q.get();
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (e->backend_failed.load()) return nullptr;
    q->id = e->next_id++;
    if (!q->session) q->ev_id = q->id;
    e->requests.emplace(q->id, std::move(q));
    e->incoming.push_back(raw);
    if (id) *id = raw->id;
    e->active_requests++;
    raw->voice->refs++;
    timeline(e, raw, KT2N_TL_GENERATE_STREAM_ENTERED, "");
    text_task_locked(e, raw, [e, raw] { plan_request(e, raw); });
  }
  e->cv.notify_all();
  return raw;
}

// mu is the common fatal/admission/publication boundary. The caller retains a
// shared owner through asynchronous voice preparation and the final callback.
void build_emit_locked(Engine* e, BuildJob* j, int status, uint64_t voice, const std::string& code,
                       const std::string& param, const std::string& message = "") {
  std::lock_guard<std::mutex> lk(j->ev_lock);
  if (j->terminal) return;
  if (status == KT2N_OK && e->backend_failed.load()) {
    status = KT2N_ERR_SERVICE_UNAVAILABLE;
    voice = 0;
  }
  auto* r = new EventRec();
  r->ev.kind = KT2N_EVENT_VOICE_BUILT;
  r->ev.status = status;
  r->ev.user_data = j->user_data;
  r->ev.at_ns = mono_ns();
  r->ev.segment_index = -1;
  r->ev.phrase_index = -1;
  r->ev.sample_begin = -1;
  r->ev.engine_chunk_index = -1;
  r->ev.voice = voice;
  if (status != KT2N_OK) {
    set_err(&r->error, status, message.empty() ? code : message, param);
    std::snprintf(r->error.code, sizeof r->error.code, "%s", code.empty() ? kt2n_error_lookup(status)->code : code.c_str());
    r->ev.error = &r->error;
  }
  j->terminal = true;
  r->ev.id = j->id;
  r->ev.seq = j->seq++;
  push_event(e, r);
}

void build_emit(Engine* e, BuildJob* j, int status, uint64_t voice, const std::string& code,
                const std::string& param, const std::string& message = "") {
  std::lock_guard<std::mutex> lk(e->mu);
  build_emit_locked(e, j, status, voice, code, param, message);
}

// mu held. Loaded voices return an id immediately; builds complete only after
// both model stores are initialized. Fatal closure finishes all pending owners.
uint64_t register_voice_locked(Engine* e, std::unique_ptr<Voice> v, std::vector<int32_t> prompt_token,
                               std::vector<float> prompt_feat, std::vector<float> embedding,
                               std::vector<float> projected, std::vector<int32_t> lm_prefix,
                               const std::shared_ptr<BuildJob>& job = {}) {
  if (e->backend_failed.load()) return 0;
  Voice* raw = v.get();
  uint64_t id;
  {
    std::lock_guard<std::mutex> lk(e->voices_mu);
    e->voices.push_back(std::move(v));
    id = e->voices.size();
  }
  if (job) e->pending_voice_builds[id].push_back(job);
  e->admin_q.push_back([e, raw, id, pt = std::move(prompt_token), pf = std::move(prompt_feat),
                        em = std::move(embedding), pj = std::move(projected), lp = std::move(lm_prefix)] {
    std::string error;
    const int render_voice = e->render->add_voice(pt, pf, em, &error);
    const int lm_voice = render_voice >= 0 ? e->lm->add_voice(pj, lp, &error) : -1;
    std::lock_guard<std::mutex> lk(e->mu);
    raw->render_voice = render_voice;
    raw->lm_voice = lm_voice;
    if (lm_voice < 0) {
      raw->evicted = true;
      e->voice_store_full++;
    }
    if (raw->evicted.load()) e->voice_sweep = true;
    auto pending = e->pending_voice_builds.find(id);
    if (pending != e->pending_voice_builds.end()) {
      const int status = e->backend_failed.load() ? KT2N_ERR_SERVICE_UNAVAILABLE
                         : lm_voice < 0 ? KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR : KT2N_OK;
      for (const auto& owner : pending->second)
        build_emit_locked(e, owner.get(), status, status == KT2N_OK ? id : 0, "", "", error);
      e->pending_voice_builds.erase(pending);
    }
  });
  e->cv.notify_all();
  return id;
}

uint64_t register_voice(Engine* e, std::unique_ptr<Voice> v, std::vector<int32_t> prompt_token,
                        std::vector<float> prompt_feat, std::vector<float> embedding,
                        std::vector<float> projected, std::vector<int32_t> lm_prefix) {
  std::lock_guard<std::mutex> lk(e->mu);
  return register_voice_locked(e, std::move(v), std::move(prompt_token), std::move(prompt_feat),
                               std::move(embedding), std::move(projected), std::move(lm_prefix));
}

#if KT2N_ENGINE_WITH_VOICE
// Marks voices the inline index dropped (kt2n_voice_evict semantics: requests holding them finish with them).
void evict_inline(Engine* e, const std::vector<uint64_t>& ids) {
  std::lock_guard<std::mutex> lk(e->voices_mu);
  for (uint64_t id : ids)
    if (id >= 1 && id <= e->voices.size()) e->voices[size_t(id - 1)]->evicted = true;
  if (!ids.empty()) e->voice_sweep = true;
}

void run_build(Engine* e, const std::shared_ptr<BuildJob>& owner) {
  BuildJob* j = owner.get();
  namespace kv = kt2n::voice;
  const bool inline_key = (j->flags & KT2N_VB_INLINE) && !j->identity_key.empty();
  // The engine index may contain a voice whose initialization is still pending.
  // Such hits join its completion owners rather than publishing an unusable id.
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (e->backend_failed.load()) {
      build_emit_locked(e, j, KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
      return;
    }
    if (inline_key && e->inline_index) {
      const uint64_t hit = e->inline_index->lookup(j->identity_key);
      bool live = false;
      if (hit) {
        std::lock_guard<std::mutex> vlk(e->voices_mu);
        live = !e->voices[size_t(hit - 1)]->evicted.load();
      }
      if (live) {
        auto pending = e->pending_voice_builds.find(hit);
        if (pending != e->pending_voice_builds.end()) pending->second.push_back(owner);
        else build_emit_locked(e, j, KT2N_OK, hit, "", "");
        return;
      }
      if (hit) e->inline_index->erase_voice(hit);
    }
  }
  kv::VoiceBuildRequest req;
  req.flags = j->flags;
  req.audio = j->audio.empty() ? nullptr : j->audio.data();
  req.audio_len = j->audio.size();
  req.pcm = j->pcm.empty() ? nullptr : j->pcm.data();
  req.pcm_samples = j->pcm.size();
  req.pcm_rate = j->pcm_rate;
  req.consent_speaker = j->consent.empty() ? nullptr : j->consent.data();
  req.reference_transcript = j->transcript;
  req.language = j->language;  // consent create: production's language check (208) needs it
  req.identity_key = j->identity_key;
  if (j->deadline_ns) {
    const int64_t left = j->deadline_ns - mono_ns();
    req.deadline = kv::Clock::now() + std::chrono::nanoseconds(std::max<int64_t>(0, left));
  }
  req.cancelled = [e, j] { return e->backend_failed.load() || j->cancelled.load(); };
  // CF_BUILD_INTERLEAVE: the conductor's grant covers the GPU work only (production: admission and the decode
  // happen before it). L11 calls gpu_turn after admission + decode, under the builder's GPU mutex.
  req.gpu_turn = [e, j]() -> bool {
    std::unique_lock<std::mutex> lk(e->build_mu);
    e->build_waiting = true;
    e->build_wait_since = mono_s();
    e->build_go = false;
    e->cv.notify_all();
    e->build_cv.wait(lk, [&] {
      return e->build_go || e->stop.load() || e->backend_failed.load() || j->cancelled.load();
    });
    e->build_waiting = false;
    e->build_go = false;
    return !(e->stop.load() || e->backend_failed.load() || j->cancelled.load());
  };
  req.gpu_done = [e] {
    std::lock_guard<std::mutex> lk(e->build_mu);
    e->last_build_done_s = mono_s();
  };
  kv::VoiceBuildOutcome out;
  try {
    out = e->voice_build_exec ? e->voice_build_exec(req) : e->vservice->build(req);
  } catch (const std::exception& ex) {
    out.status = KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR;
    out.code = "custom_voice_service_error";
  }
  std::lock_guard<std::mutex> completion(e->mu);
  if (e->backend_failed.load()) {
    build_emit_locked(e, j, KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
    return;
  }
  if (out.status == KT2N_ERR_CANCELLED) {
    build_emit_locked(e, j, KT2N_ERR_CANCELLED, 0, "", "");
    return;
  }
  if (out.status != 0 || !out.result) {
    build_emit_locked(e, j, out.status ? out.status : KT2N_ERR_CUSTOM_VOICE_SERVICE_ERROR, 0, out.code, out.param,
               out.message);
    return;
  }
  const kv::CloneResult& r = *out.result;
  auto v = std::make_unique<Voice>();
  v->kind = (j->flags & KT2N_VB_SAVED) ? KT2N_VOICE_SAVED : KT2N_VOICE_INLINE;
  v->id = j->identity_key.empty() ? ("clone-" + std::to_string(j->id)) : j->identity_key;
  v->names = {v->id};
  v->peak_gain_db = kv::kCustomPeakGainDb;
  v->prompt_tokens = int(r.decoder.prompt_token.size());
  v->prompt_frames = int(r.decoder.prompt_feat.size() / 80);
  v->lm_prefix_ids = int(r.lm_prefix.size());
  v->raw_speaker = r.raw_speaker;
  v->geometry = policy::geometry_owner(v->prompt_tokens, v->prompt_frames, int(r.decoder.xvector.size()));
  v->artifact = out.artifact;
  std::vector<float> projected;
  if (out.projected_speaker) projected.assign(out.projected_speaker, out.projected_speaker + 2048);
  uint64_t vid;
  if (j->flags & KT2N_VB_CONSENT) {
    // Consent recordings carry only the speaker embedding (runtime.py:12277): no LM prefix, no render state.
    std::lock_guard<std::mutex> lk(e->voices_mu);
    v->consent_only = true;
    e->voices.push_back(std::move(v));
    vid = e->voices.size();
  } else {
    // A concurrent build of the same inline key may have registered first: serve that one (production's
    // single flight), and do not register a duplicate.
    if (inline_key && e->inline_index) {
      const uint64_t first = e->inline_index->lookup(j->identity_key);
      if (first) {
        auto pending = e->pending_voice_builds.find(first);
        if (pending != e->pending_voice_builds.end()) pending->second.push_back(owner);
        else build_emit_locked(e, j, KT2N_OK, first, "", "");
        return;
      }
    }
    vid = register_voice_locked(e, std::move(v), r.decoder.prompt_token, r.decoder.prompt_feat, r.decoder.xvector,
                         std::move(projected), r.lm_prefix, owner);
  }
  if (!vid) {
    build_emit_locked(e, j, KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
    return;
  }
  if (inline_key && e->inline_index) {
    std::vector<uint64_t> evict;
    auto in_use = [e](uint64_t id) {
      std::lock_guard<std::mutex> lk(e->voices_mu);
      return e->pending_voice_builds.count(id) ||
             (id >= 1 && id <= e->voices.size() && e->voices[size_t(id - 1)]->refs.load() > 0);
    };
    vid = e->inline_index->insert(j->identity_key, vid, in_use, &evict);
    evict_inline(e, evict);
  }
  if (j->flags & KT2N_VB_CONSENT) build_emit_locked(e, j, KT2N_OK, vid, "", "");
}
#endif

void voice_worker(Engine* e) {
  for (;;) {
    std::shared_ptr<BuildJob> j;
    {
      std::unique_lock<std::mutex> lk(e->build_mu);
      e->build_cv.wait(lk, [&] { return e->stop.load() || !e->build_q.empty(); });
      if (e->stop.load()) return;
      j = e->build_q.front();
      e->build_q.pop_front();
      e->builds_running += 1;
    }
#if KT2N_ENGINE_WITH_VOICE
    if (e->backend_failed.load())
      build_emit(e, j.get(), KT2N_ERR_SERVICE_UNAVAILABLE, 0, "", "");
    else if (j->cancelled.load())
      build_emit(e, j.get(), KT2N_ERR_CANCELLED, 0, "", "");
    else
      run_build(e, j);
#else
    build_emit(e, j.get(), KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, 0, "", "");
#endif
    std::lock_guard<std::mutex> lk(e->build_mu);
    e->builds_running -= 1;
  }
}

void session_emit(Engine* e, Session* s, int kind, int status, const std::string& message) {
  auto* r = new EventRec();
  r->ev.kind = kind;
  r->ev.status = status;
  r->ev.user_data = s->user_data;
  r->ev.at_ns = mono_ns();
  r->ev.segment_index = -1;
  r->ev.phrase_index = -1;
  r->ev.sample_begin = -1;
  r->ev.engine_chunk_index = -1;
  if (status != KT2N_OK) {
    set_err(&r->error, status, message);
    r->ev.error = &r->error;
  }
  std::lock_guard<std::mutex> lk(s->ev_lock);
  r->ev.id = s->id;
  r->ev.seq = s->seq++;
  push_event(e, r);
}

// A phrase from the planner -> one internal generation emitting as the session (sess_mu held).
void start_phrase(Engine* e, Session* s, const textplan::WsPlanner::Phrase& ph) {
  auto q = std::make_unique<Request>();
  q->user_data = s->user_data;
  q->flags = s->flags & ~uint32_t(KT2N_REQ_SEED);
  q->timeline = (s->flags & KT2N_REQ_TIMELINE) != 0;
  q->voice = s->voice;
  q->raw = textplan::to_utf8(ph.text);
  q->sampler = s->sampler;
  q->speed = s->speed;
  q->format = s->format;
  q->output = KT2N_OUTPUT_STREAM;
  q->deadline_ns = s->deadline_ns;
  q->started_s = mono_s();
  q->realtime = true;
  q->normalize = s->normalize;
  q->emotion_preset = ph.emotion;
  q->session = s;
  q->phrase_index = s->phrases + 1;  // production's phrase index counts from 1 (api.py:6248)
  q->ev_id = s->id;
  q->ev_lock = &s->ev_lock;
  q->ev_seq = &s->seq;
  // Phrase seed: production's phrase control has request_id "<session request id>:phrase:<index>"
  // (api.py:6066) and the salt is _request_seed_salt of it. Lab pairing (KT2N_REQ_SEED): seed + 1000 x (index-1).
  q->salt = s->seeded ? s->salt + uint64_t(s->phrases) * 1000ull
                      : policy::request_seed_salt(s->request_id + ":phrase:" + std::to_string(q->phrase_index));
  kt2n::post::ResponsePostConfig rc;
  rc.speed = s->speed;
  q->rpost = std::make_unique<kt2n::post::ResponsePost>(rc);
  kt2n::post::HoldPolicy hp;
  hp.pinned = 0.0;  // a realtime phrase pins startup_hold_seconds = 0; the WS reserve lives in the wrapper
  q->hold = std::make_unique<kt2n::post::StartupHold>(hp);
  q->awaiting_hold.store(true, std::memory_order_relaxed);  // until its first publication releases the pinned-0 hold
  q->peak_ptr = &*s->peak;
  s->phrases += 1;
  s->phrase_running = true;
  Request* raw = register_request(e, std::move(q));
  if (!raw) {
    s->phrase_running = false;
    s->cancelled = true;
    s->cancel_status = KT2N_ERR_SERVICE_UNAVAILABLE;
    return;
  }
  s->phrase_ids.push_back(raw->id);
  timeline(e, raw, KT2N_TL_API_PHRASE_GENERATION_STARTED,
           "\"phrase_index\":" + std::to_string(raw->phrase_index) + ",\"characters\":" + std::to_string(ph.text.size()));
}

void session_phrase_done(Engine* e, Request* q) {
  // Called from emit_done with q->mu held: never take sess_mu here (sess_mu -> e->mu -> q->mu order).
  q->session->phrase_running = false;
  q->session->dirty = true;
  e->sess_cv.notify_all();
}

void session_thread(Engine* e) {
  std::unique_lock<std::mutex> lk(e->sess_mu);
  while (!e->stop.load()) {
    double next_wake = 50.0;  // ms
    for (auto& [id, sp] : e->sessions) {
      Session* s = sp.get();
      if (s->terminal) continue;
      if (e->backend_failed.load()) {
        s->cancelled = true;
        s->cancel_status = KT2N_ERR_SERVICE_UNAVAILABLE;
      }
      const double now_ms = (mono_s() - s->opened_s) * 1000.0;
      if (!s->cancelled && s->deadline_ns && mono_ns() > s->deadline_ns) {
        // The session deadline (production session_control) ends it: deadline_exceeded.
        s->cancelled = true;
        s->cancel_status = KT2N_ERR_DEADLINE_EXCEEDED;
        std::lock_guard<std::mutex> elk(e->mu);
        for (uint64_t pid : s->phrase_ids) {
          auto it = e->requests.find(pid);
          if (it != e->requests.end() && !it->second->cancelled.load()) {
            it->second->cancel_status = KT2N_ERR_DEADLINE_EXCEEDED;
            it->second->cancelled = true;
          }
        }
        e->cv.notify_all();
      }
      if (s->cancelled) {
        if (!s->phrase_running) {
          s->terminal = true;
          session_emit(e, s, KT2N_EVENT_DONE, s->cancel_status,
                       s->cancel_status == KT2N_ERR_INPUT_ALREADY_CLOSED ? "Realtime TTS input is already closed"
                       : s->cancel_status == KT2N_ERR_SPEECH_GENERATION_FAILED ? "Speech generation failed"
                                                                               : "request_cancelled");
        }
        continue;
      }
      if (!s->dirty && now_ms < s->wake_ms) {
        next_wake = std::min(next_wake, s->wake_ms - now_ms);
        continue;
      }
      s->dirty = false;
      if (s->planner_done) {
        if (!s->phrase_running) {
          s->terminal = true;
          session_emit(e, s, KT2N_EVENT_DONE, KT2N_OK, "");
        }
        continue;
      }
      for (int guard = 0; guard < 64; ++guard) {
        auto act = s->planner.step(now_ms, !s->phrase_running);
        if (act.kind == textplan::WsPlanner::Action::Kind::kPhrase) {
          start_phrase(e, s, act.phrase);
          if (s->cancelled) break;
          continue;  // the next step waits for the slot
        }
        if (act.kind == textplan::WsPlanner::Action::Kind::kDone) {
          s->planner_done = true;
          if (!s->phrase_running) {
            s->terminal = true;
            session_emit(e, s, KT2N_EVENT_DONE, KT2N_OK, "");
          }
          break;
        }
        s->wake_ms = std::isinf(act.until_ms) ? 1e18 : act.until_ms;
        if (!std::isinf(act.until_ms)) next_wake = std::min(next_wake, std::max(0.0, act.until_ms - now_ms));
        break;
      }
    }
    cv_wait_for(e->sess_cv, lk, std::chrono::microseconds(int64_t(std::max(0.2, next_wake) * 1000.0)));
  }
}

Session* session_of(Engine* e, uint64_t id) {
  auto it = e->sessions.find(id);
  return it == e->sessions.end() ? nullptr : it->second.get();
}

}  // namespace
}  // namespace kt2n::engine

// =====================================================================================================
// C ABI
// =====================================================================================================
using namespace kt2n::engine;

extern "C" {

KT2N_API const kt2n_error_info* kt2n_error_lookup(int32_t status) {
  static kt2n_error_info table[sizeof(kErrors) / sizeof(kErrors[0])];
  static std::once_flag once;
  std::call_once(once, [] {
    for (size_t i = 0; i < sizeof(kErrors) / sizeof(kErrors[0]); ++i)
      table[i] = {kErrors[i].status, kErrors[i].http, kErrors[i].code, kErrors[i].type};
  });
  for (const auto& t : table)
    if (t.status == status) return &t;
  for (const auto& t : table)
    if (t.status == KT2N_ERR_INTERNAL) return &t;
  return &table[0];
}

KT2N_API uint32_t kt2n_abi_version(void) { return KT2N_ABI_VERSION; }

KT2N_API const char* kt2n_build_info(void) {
  // Capability tokens (engine.h): the adapter refuses the live skip-aware WS hold without the first one.
  return "Kitten TTS 2 native " KT2N_VERSION_STRING " " KT2N_CAP_AUDIO_OPENING_SKIP
#if KT2N_ENGINE_WITH_VOICE
      " " KT2N_CAP_VOICE_CLONING
#endif
      ;
}

static int text_threads(const kt2n_engine_config* cfg) { return cfg->threads_text > 0 ? cfg->threads_text : 4; }

KT2N_API int32_t kt2n_engine_create(const kt2n_engine_config* cfg, kt2n_engine** out, kt2n_error* err) {
  if (!cfg || !out || cfg->struct_size < offsetof(kt2n_engine_config, voice_build_dir)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "config is NULL or too small");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (cfg->abi_version != KT2N_ABI_VERSION) {
    set_err(err, KT2N_ERR_ABI_MISMATCH, "abi_version mismatch");
    return KT2N_ERR_ABI_MISMATCH;
  }
  *out = nullptr;
  const bool inference_only = (cfg->flags & KT2N_CFG_INFERENCE_ONLY) != 0;
  const int custom_capacity = cfg->struct_size >= offsetof(kt2n_engine_config, max_custom_voices) + sizeof(int32_t)
                                  ? cfg->max_custom_voices : 0;
  if (custom_capacity < 0 || custom_capacity > 4096) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "max_custom_voices must be 0..4096", "max_custom_voices");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  const bool bounded_capacity = inference_only || custom_capacity > 0;
  bool require_assets = inference_only;
#ifndef KT2N_ALLOW_TEST_STUBS
  require_assets = true;
#endif
  if (require_assets && (!cfg->lm_file || !cfg->s3gen_file || !cfg->voices_file ||
                         !cfg->tokenizer_json || !cfg->text_data_dir ||
                         std::string(cfg->s3gen_file) == "l7-standin")) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "inference requires complete model, voice, tokenizer and text assets");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (cfg->max_streams < 0 || cfg->max_streams > 64 || cfg->render_pages < 0) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "max_streams must be 0..64 and render_pages must be nonnegative");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (cfg->pinned_arena_bytes) {
    set_err(err, KT2N_ERR_UNSUPPORTED, "custom pinned-memory limits are not supported", "pinned_arena_bytes");
    return KT2N_ERR_UNSUPPORTED;
  }
  // Load every CUDA module at context creation instead of at a kernel's first launch (L7, 12:13: the first live
  // render of a process otherwise pays lazy module loading inside launch, 6.3 ms vs 1.1 ms). Execution only.
  // Takes effect only if this process has not initialised CUDA yet; an explicit CUDA_MODULE_LOADING wins.
  // KT2N_CUDA_EAGER=0 keeps the driver's lazy default.
  if (const char* v = std::getenv("KT2N_CUDA_EAGER"); !(v && std::atoi(v) == 0)) setenv("CUDA_MODULE_LOADING", "EAGER", 0);
  size_t free_before = 0, device_total = 0;
  if (bounded_capacity) {
    cudaError_t status = cudaSetDevice(cfg->device);
    if (status == cudaSuccess) status = cudaFree(nullptr);
    if (status == cudaSuccess) status = cudaMemGetInfo(&free_before, &device_total);
    if (status != cudaSuccess) {
      set_err(err, KT2N_ERR_UNSUPPORTED, std::string("CUDA initialization: ") + cudaGetErrorString(status), "device");
      return KT2N_ERR_UNSUPPORTED;
    }
  }
  auto e = std::make_unique<kt2n_engine>();
  e->flags = cfg->flags;
  // Conductor poll mode (execution only), on by default since the 11:47 A/B (first audio lower at N=1/4/8 in 10
  // of 12 interleaved pairs, ~25% less process CPU; lane log "Win 2"). KT2N_CONDUCTOR_POLL=0 restores the blocking
  // waits.
  e->poll = true;
  if (const char* v = std::getenv("KT2N_CONDUCTOR_POLL")) e->poll = std::atoi(v) != 0;
  if (const char* v = std::getenv("KT2N_SPLIT_URGENT")) e->split_urgent = std::atoi(v) != 0;
  if (const char* v = std::getenv("KT2N_HOLD_URGENT")) e->hold_urgent = std::max(0, std::min(2, std::atoi(v)));
  if (const char* v = std::getenv("KT2N_BUFFER_URGENT_MS")) e->buffer_urgent_ms = std::max(0.0, std::atof(v));
  if (const char* v = std::getenv("KT2N_BUFFER_ROLE")) e->buffer_to_aux = std::string(v) == "aux";
  if (const char* v = std::getenv("KT2N_TEXT_NICE")) e->text_nice = std::max(0, std::min(19, std::atoi(v)));
  const std::string app = cfg->app ? cfg->app : "default";
  if (app != "default" && app != "us" && app != "ap" && app != "fq") {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "unknown inference profile", "app");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  e->lv = policy::live_levers(app);
  if (cfg->levers_json) {
    std::string bad, why;
    if (!apply_levers(cfg->levers_json, app, e->lv, &bad, &why, &e->levers_unknown)) {
      set_err(err, KT2N_ERR_UNSUPPORTED, why, bad);
      return KT2N_ERR_UNSUPPORTED;
    }
  }
  e->max_streams = cfg->max_streams > 0 ? cfg->max_streams : (inference_only ? 1 : 10);
  e->row_cap = engine_rows(e->max_streams, cfg->device, cfg->lm_file || cfg->s3gen_file, &e->row_cap_why);
  // REQUEST_BATCH_SIZE = production's decode rows (= its seats). Here the decode rows are row_cap: the same as
  // max_streams unless the seats exceed the GPU's row fit (capacity mode).
  e->lv.request_batch_size = e->row_cap;
  if (const char* v = std::getenv("KT2N_CAPACITY_ORDER")) e->capacity_order = std::atoi(v) != 0;
  if (e->row_cap < e->max_streams)
    std::fprintf(stderr, "[kt2n engine] seats %d, rows %d (%s): admitted requests beyond %d rows wait in admission\n",
                 e->max_streams, e->row_cap, e->row_cap_why.c_str(), e->row_cap);
  std::string msg;
  // Text.
  // Text threads = grammar copies. Default 4 = production's N_POOL_WORKERS (L9a 11:20: with 2 copies the N_SPLIT
  // budget is 2 and 1 of 4,054 segments joins its pieces differently from production; 3-4 match all). engine.h's
  // comment still says 2 (G0-frozen header; noted for G0 review in the lane log).
  e->text = TextFront::create(cfg->text_data_dir, cfg->tokenizer_json, text_threads(cfg),
                              &msg);
  if (!e->text) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "text front end: " + msg, "text_data_dir");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  debug_mem("text front end");
  // Voices file first: the LM's voice store is sized from the prefixes it must hold.
  std::optional<kt2n::voice::VoicesBlob> vblob;
  size_t prefix_tokens = 0;
  if (cfg->voices_file) {
    try {
      std::ifstream f(cfg->voices_file, std::ios::binary);
      std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      if (blob.empty()) throw std::runtime_error("cannot read voices_file");
      vblob = kt2n::voice::parse_voices_blob(blob.data(), blob.size());
      for (const auto& rec : vblob->voices) prefix_tokens += rec.lm_prefix.size();
    } catch (const std::exception& ex) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, std::string("voices: ") + ex.what(), "voices_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
  }
  // LM.
  if (cfg->lm_file) {
    kt2n::lm::Config lc;
    lc.engine_path = cfg->lm_file;
    lc.device = cfg->device;
    // +2 slots: L3 releases a finished row's slot only after its last ring entry is read (lm.h), so a new
    // row admitted in the same loop may need one more slot for a moment.
    lc.max_rows = std::min(64, std::max(e->row_cap + 2, cfg->max_lm_rows > 0 ? cfg->max_lm_rows : 0));
    lc.graphs = !(cfg->flags & KT2N_CFG_NO_GRAPHS);
    // 112 KB of bf16 K/V per prefix token ([2][28][8][128]), plus 1/8 slack for alignment.
    // Headroom for loaded saved / inline voices: custom_voice_headroom() prefixes of 512 ids (72 by default, ~4.8 GB).
    // Lab knobs for the 12 GB GPU-slot rule (defaults unchanged): KT2N_LM_VOICE_HEADROOM = loaded-voice prefixes
    // of headroom (32); KT2N_LM_MAX_CTX = K/V columns per LM slot (4096; capacity only, the ceiling rule uses
    // ctx_limit, which stays 4096).
    size_t headroom = inference_only ? 0 : size_t(custom_capacity > 0 ? custom_capacity :
        custom_voice_headroom(cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32));
    if (custom_capacity == 0)
      if (const char* v = std::getenv("KT2N_LM_VOICE_HEADROOM")) headroom = size_t(std::max(0, std::atoi(v)));
    lc.voice_store_bytes = std::max<size_t>(size_t{256} << 20, (prefix_tokens + headroom * 512) * 114688 * 9 / 8);
    if (const char* v = std::getenv("KT2N_LM_MAX_CTX")) lc.max_ctx = std::max(1024, std::atoi(v) / 64 * 64);
    // A/B knob (L3 16:28, execution only): decode steps of split_min_rows..16 rows run as two half-batch graph
    // branches (L3's default 14; 0 = off, the pre-sprint-5 step). Unset = L3's default.
    if (const char* v = std::getenv("KT2N_LM_SPLIT_MIN_ROWS"); v && *v) lc.split_min_rows = std::max(0, std::atoi(v));
    // ... and the largest split bucket (L3 16:40: 78% of N=20 steps have 17-20 rows; 20 = bucket 20 as 10 + 10).
    if (const char* v = std::getenv("KT2N_LM_SPLIT_MAX_ROWS"); v && *v) lc.split_max_rows = std::max(0, std::atoi(v));
    // ... and the decode stream's CUDA priority level above the lowest (L3 46f00816, execution only; 0 = L3's default).
    if (const char* v = std::getenv("KT2N_LM_DECODE_PRIORITY"); v && *v) lc.decode_priority_level = std::max(0, std::atoi(v));
    // Graph replay takes the launch stream's priority. Eager split branches keep their original stream priorities,
    // so this experiment is restricted to the existing graph path and a positive restore priority.
    if (const char* v = std::getenv("KT2N_LM_RENDER_YIELD"); v && std::atoi(v) != 0) {
      if (!lc.graphs || lc.decode_priority_level <= 0) {
        set_err(err, KT2N_ERR_INVALID_ARGUMENT,
                "LM render-yield experiment requires graphs and a positive LM decode priority");
        return KT2N_ERR_INVALID_ARGUMENT;
      }
      e->lm_render_yield = true;
      e->lm_render_yield_base_priority = lc.decode_priority_level;
    }
    e->lm = make_l3_lm(lc, &msg);
    if (!e->lm) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "LM: " + msg, "lm_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    if (e->lm_render_yield) {
      // Create the alternate stream before any request exists, so a live relief episode allocates no CUDA stream.
      std::vector<kt2n::lm::TokenEvent> warm;
      for (int level : {0, e->lm_render_yield_base_priority}) {
        e->lm->set_decode_priority(level);
        if (e->lm->pump(&warm, false, &msg) < 0) {
          set_err(err, KT2N_ERR_INVALID_ARGUMENT, "LM render-yield prewarm: " + msg, "lm_file");
          return KT2N_ERR_INVALID_ARGUMENT;
        }
      }
    }
  } else {
    StubLmConfig lm_cfg;  // stub timing knobs (lab simulation only; the real LM ignores them)
    if (const char* v = std::getenv("KT2N_STUB_LM_STEP_US")) lm_cfg.step_us = std::atoi(v);
    e->lm = make_stub_lm(lm_cfg);
  }
  debug_mem("lm create");
  // Render. NULL = the CPU stub. "l7-standin" (lab) = L7's RenderService with its stand-in stages: the real
  // render state and call path, placeholder audio. A path = the real renderer (L7 Renderer over s3gen.kt2n).
  if (cfg->s3gen_file && std::string(cfg->s3gen_file) == "l7-standin") {
#if KT2N_ENGINE_WITH_L7
    // Pool pages: lab knob KT2N_L7_POOL_PAGES (16.5 MB each; voice prefixes take shared pages too).
    int pages = 160;  // 2.6 GB; 64 ran out at N=10 (no_pages, 10:40 run)
    if (const char* v = std::getenv("KT2N_L7_POOL_PAGES")) pages = std::max(8, std::atoi(v));
    e->render = make_l7_standin_render(cfg->device, std::min(64, e->row_cap + 4), pages, 64, &msg);
    if (!e->render) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "L7 render service: " + msg, "s3gen_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
#else
    set_err(err, KT2N_ERR_UNSUPPORTED, "built without L7's render service", "s3gen_file");
    return KT2N_ERR_UNSUPPORTED;
#endif
  } else if (cfg->s3gen_file) {
    // The real renderer: L7 Renderer over s3gen.kt2n (L4 encoder + L5 estimator + L6 HiFT).
    L7RenderOptions ro;
    ro.s3gen_path = cfg->s3gen_file;
    ro.device = cfg->device;
    ro.slots = std::min(64, e->row_cap + 4);
    ro.max_rows = std::min(64, e->row_cap + 4);
    // Voice table and pool. Voices are append-only in L7 (and L3) today: a loaded / built voice keeps its table
    // row and prefix pages for the process lifetime (lane log 13:30, request to L3/L7). Headroom: 32 voices
    // (the inline LRU size), the same as the LM voice store below.
    // Pool: every catalogue voice's prefix (built on first use, kept) + the headroom voices' prefixes + 40 pages
    // per stream (~640 tokens per row; a row reserves its budget's pages at admission and waits when they are
    // not free). Lab knobs override (the 12 GB GPU-slot rule). Before 13:30 the default was 64 + 40 x streams,
    // so a mix of more than ~8 voices let first-use prefixes eat the rows' pages.
    // Table rows for custom voices: custom_voice_headroom() (72 by default; KT2N_L7_VOICE_HEADROOM = lab / test
    // knob). The pool keeps pages for 32 custom prefixes (below): more live custom prefixes than that take pages
    // from the rows' share, which is back-pressure (a first-use prefix waits for reservations), never a failure.
    const int extra_voices = inference_only ? 0 : custom_capacity > 0 ? custom_capacity : std::max(1, [cfg] {
      const char* v = std::getenv("KT2N_L7_VOICE_HEADROOM");
      return v && *v ? std::atoi(v) : custom_voice_headroom(cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32);
    }());
    const int pool_extra_voices = std::min(extra_voices, 32);
    int prefix_pages = 0, n_cat = 0;
    if (vblob)
      for (const auto& rec : vblob->voices) {
        prefix_pages += (2 * int(rec.prompt_token.size()) + 63) / 64 + 1;
        ++n_cat;
      }
    const int per_voice = n_cat ? (prefix_pages + n_cat - 1) / n_cat : 10;
    ro.voices = n_cat + extra_voices;
    ro.pool_pages = prefix_pages + pool_extra_voices * per_voice + 40 * e->row_cap;
    auto env_int = [](const char* k, int d) {
      const char* v = std::getenv(k);
      return v && *v ? std::atoi(v) : d;
    };
    ro.pool_pages = std::max(16, cfg->render_pages > 0 ? cfg->render_pages : env_int("KT2N_L7_POOL_PAGES", ro.pool_pages));
    if (ro.pool_pages < prefix_pages + 40 * e->row_cap) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "render_pages cannot hold voice prefixes and requested streams", "render_pages");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    ro.workspace_mb = env_int("KT2N_L7_WORKSPACE_MB", ro.workspace_mb);
    ro.hift_max_frame_rows = env_int("KT2N_L7_HIFT_FRAME_ROWS", 0);
    ro.hift_bulk_max_t = env_int("KT2N_L7_HIFT_BULK_T", 0);                    // lab: the 12 GB slot rule
    ro.hift_bulk_max_frame_rows = env_int("KT2N_L7_HIFT_BULK_FRAME_ROWS", 0);
    ro.prewarm = env_int("KT2N_L7_PREWARM", 1) != 0;
    ro.prewarm_rows = env_int("KT2N_L7_PREWARM_ROWS", 0);
    ro.prewarm_tokens = env_int("KT2N_L7_PREWARM_TOKENS", ro.prewarm_tokens);
    ro.reserve_pages = env_int("KT2N_L7_RESERVE", 1) != 0;
    if (bounded_capacity) ro.lazy_prefix = false;
    if (const char* v = std::getenv("KT2N_L7_PREFIXES")) ro.lazy_prefix = std::string(v) != "all";
#if KT2N_ENGINE_WITH_L7
    e->render = make_l7_render(ro, &msg);
#else
    msg = "built without L7's render service";
#endif
    if (!e->render) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "L7 renderer: " + msg, "s3gen_file");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
  }
  if (!e->render) {
    StubRenderConfig stub_cfg;
    // Tests only (the stub renderer): every n-th render call fails, to exercise the failure paths.
    if (const char* f = std::getenv("KT2N_TEST_STUB_RENDER_FAIL_EVERY")) stub_cfg.fail_every = std::atoi(f);
    if (const char* f = std::getenv("KT2N_TEST_STUB_RENDER_MAX_ROWS")) stub_cfg.max_open_rows = std::atoi(f);
    if (const char* v = std::getenv("KT2N_STUB_RENDER_CALL_US")) stub_cfg.call_us = std::atoi(v);
    if (const char* v = std::getenv("KT2N_STUB_RENDER_TOKEN_US")) stub_cfg.token_us = std::atof(v);
    e->render = make_stub_render(stub_cfg);
  }
  debug_mem("render create");
  // Voices.
  try {
    if (vblob) {
      e->catalogue = kt2n::voice::Catalogue::from_blob(*vblob);
      for (auto& rec : vblob->voices) {
        auto v = std::make_unique<Voice>();
        v->id = rec.id();
        v->names.push_back(v->id);
        if (const auto* pn = rec.meta.get("public_names"))
          for (const auto& n : pn->items()) v->names.push_back(n.as_string());
        // The voice's language: production's gradio_demo/voices.json language_code at demo 5f9e5561, casefolded,
        // before the first '-' (runtime.py _reference_locked; built in by `python -m kt2n.voices catalogue
        // --voices-json`). An empty value (the L11 check catalogue 88c68b26 left every voice blank) falls back to
        // "en", and that fallback is NOT neutral: normalizer.cc prepare() takes code "en" as English before
        // NORMALIZE_LANGUAGE=text ever looks at the text, so a blank catalogue normalizes Hindi or other non-English
        // text with the English WFST.
        if (const auto* l = rec.meta.get("language"); l && l->is_string() && !l->as_string().empty())
          v->language = l->as_string();
        if (const auto* h = rec.meta.get("hidden"); h && h->is_bool()) v->hidden = h->as_bool();
        if (const auto* g = rec.meta.get("peak_gain_db"); g && !g->is_null()) v->peak_gain_db = g->as_double();
        v->prompt_tokens = int(rec.prompt_token.size());
        v->prompt_frames = int(rec.prompt_feat.size() / 80);
        v->lm_prefix_ids = int(rec.lm_prefix.size());
        v->raw_speaker = rec.raw_speaker;
        v->geometry = policy::geometry_owner(v->prompt_tokens, v->prompt_frames, int(rec.embedding.size()));
        v->lm_voice = e->lm->add_voice(rec.projected_speaker, rec.lm_prefix, &msg);
        if (v->lm_voice.load() < 0) throw std::runtime_error("LM voice " + v->id + ": " + msg);
        debug_mem(("lm voice " + std::to_string(rec.lm_prefix.size())).c_str());
        v->render_voice = e->render->add_voice(rec.prompt_token, rec.prompt_feat, rec.embedding, &msg);
        if (v->render_voice < 0) {
          if (require_assets) throw std::runtime_error("render voice " + v->id + ": " + msg);
          e->render_voice_errors++;
          e->render_voice_error = msg;
        }
        e->voices.push_back(std::move(v));
      }
    } else {
      // Stub catalogue (tests): two fake voices.
      for (const char* name : {"stub_voice_a", "stub_voice_b"}) {
        auto v = std::make_unique<Voice>();
        v->id = name;
        v->names = {name};
        v->prompt_tokens = 200;
        v->prompt_frames = 400;
        v->geometry = policy::geometry_owner(200, 400, 192);
        v->lm_voice = e->lm->add_voice(std::vector<float>(2048, 0.f), std::vector<int32_t>(10, 1), &msg);
        v->render_voice = e->render->add_voice({}, {}, {}, &msg);
        e->voices.push_back(std::move(v));
      }
    }
  } catch (const std::exception& ex) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, std::string("voices: ") + ex.what(), "voices_file");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  debug_mem("voices");
  // Clone builds (L11): VoiceBuilder (its device arena is allocated here) + VoiceService on voice threads.
  e->clone_slots = cfg->clone_slots > 0 ? cfg->clone_slots : 4;
  e->clone_queue = cfg->clone_queue > 0 ? cfg->clone_queue : 16;
  e->clone_timeout_s = cfg->clone_timeout_ms > 0 ? cfg->clone_timeout_ms / 1000.0 : 120.0;
  const char* vbd = cfg->struct_size >= offsetof(kt2n_engine_config, voice_build_dir) + sizeof(const char*)
                        ? cfg->voice_build_dir
                        : nullptr;
  // Clone weights: from s3gen.kt2n when it carries the four clone groups (L11 FORMAT amendment), else from the
  // lab voice_build_dir (three files). voice_build_dir wins when both are given.
  bool kt2n_clone = false;
#if KT2N_ENGINE_WITH_VOICE
  if (!inference_only && !vbd && cfg->s3gen_file && std::string(cfg->s3gen_file) != "l7-standin") {
    try {
      kt2n::voice::CloneWeightsFile::open(cfg->s3gen_file);
      kt2n_clone = true;
    } catch (const std::exception&) {
      // an s3gen.kt2n without the clone groups: no clone service (kt2n_voice_build answers 214), as before
    }
  }
#endif
  if (!inference_only && (vbd || kt2n_clone)) {
#if KT2N_ENGINE_WITH_VOICE
    if (!e->catalogue || e->catalogue->size() == 0 || !e->text->real_tokenizer()) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "clone builds need a raw speaker catalogue and tokenizer_json", "voice_build_dir");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    try {
      if (kt2n_clone) {
        e->vbuilder = std::make_unique<kt2n::voice::VoiceBuilder>(kt2n::voice::S3genKt2n{cfg->s3gen_file});
      } else {
        const std::string d = vbd;
        e->vbuilder = std::make_unique<kt2n::voice::VoiceBuilder>(d + "/s3gen.safetensors",
                                                                  d + "/voicebuild_consts.safetensors",
                                                                  d + "/voicebuild_pyannote.safetensors");
      }
      e->vbuilder->warmup();
      kt2n::engine::TextFront* tf = e->text.get();
      e->vservice = std::make_unique<kt2n::voice::VoiceService>(
          *e->vbuilder,
          [tf](const std::string& t) { return tf->encode(t); },
          e->clone_slots, e->clone_queue, e->clone_timeout_s, cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32);
      e->vservice->set_catalogue(&*e->catalogue);
      e->inline_index = std::make_unique<kt2n::voice::InlineVoiceIndex>(
          cfg->inline_voice_lru > 0 ? cfg->inline_voice_lru : 32);
    } catch (const std::exception& ex) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, std::string("voice build: ") + ex.what(), "voice_build_dir");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
#else
    set_err(err, KT2N_ERR_UNSUPPORTED, "built without the clone build service (L11, KT2N_VOICE_AV)", "voice_build_dir");
    return KT2N_ERR_UNSUPPORTED;
#endif
  }
  debug_mem("clone service");
  if (bounded_capacity) {
    size_t free_after = 0, total_after = 0;
    const auto status = cudaMemGetInfo(&free_after, &total_after);
    if (status != cudaSuccess) {
      set_err(err, KT2N_ERR_INTERNAL, "cannot query initialized device memory");
      return KT2N_ERR_INTERNAL;
    }
    e->device_reserved_bytes = free_before > free_after ? free_before - free_after : 0;
    if ((cfg->device_arena_bytes && e->device_reserved_bytes > cfg->device_arena_bytes) ||
        free_after < device_total / 10) {
      set_err(err, KT2N_ERR_TOO_SMALL,
              "model, streams and optional cloning exceed the memory budget; reduce capacity", "max_streams");
      return KT2N_ERR_TOO_SMALL;
    }
  }
  e->efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  // cpuset ("0-7,12"): every engine thread is pinned to it (plan: the 8-vCPU budget of production).
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  bool pin = false;
  if (cfg->cpuset && *cfg->cpuset) {
    std::stringstream ss(cfg->cpuset);
    std::string part;
    while (std::getline(ss, part, ',')) {
      const auto dash = part.find('-');
      const int lo = std::atoi(part.substr(0, dash).c_str());
      const int hi = dash == std::string::npos ? lo : std::atoi(part.substr(dash + 1).c_str());
      for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) {
        CPU_SET(c, &cpus);
        pin = true;
      }
    }
    if (!pin) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "cpuset is empty or malformed", "cpuset");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
  }
  kt2n_engine* raw = e.get();
  const int nt = text_threads(cfg);
  const int np = cfg->threads_post > 0 ? cfg->threads_post : 2;
  for (int i = 0; i < nt; ++i) raw->threads.emplace_back([raw] { text_worker(raw); });
  for (int i = 0; i < np; ++i) raw->threads.emplace_back([raw] { post_worker(raw); });
  raw->threads.emplace_back([raw] { conductor(raw); });
  raw->threads.emplace_back([raw] { session_thread(raw); });
  const int nv = cfg->threads_voice > 0 ? cfg->threads_voice : 1;
  for (int i = 0; i < nv; ++i) raw->threads.emplace_back([raw] { voice_worker(raw); });
  if (pin)
    for (auto& t : raw->threads) pthread_setaffinity_np(t.native_handle(), sizeof cpus, &cpus);
  *out = e.release();
  return KT2N_OK;
}

KT2N_API void kt2n_engine_destroy(kt2n_engine* e) {
  if (!e) return;
  {
    // Publish shutdown under the same mutex as the event wait predicate: no lost wake-up.
    std::lock_guard<std::mutex> lk(e->ev_mu);
    e->stop = true;
  }
  e->ev_cv.notify_all();
  // Pair every indefinite wait with its mutex. Otherwise a worker can observe stop=false,
  // miss the notification just before entering wait, and keep destroy's join blocked.
  auto wake = [](std::mutex& mu, std::condition_variable& cv) {
    std::lock_guard<std::mutex> lk(mu);
    cv.notify_all();
  };
  wake(e->mu, e->cv);
  wake(e->text_mu, e->text_cv);
  wake(e->post_mu, e->post_cv);
  wake(e->sess_mu, e->sess_cv);
  wake(e->build_mu, e->build_cv);
  for (auto& t : e->threads) t.join();
  e->render.reset();
  e->lm.reset();
  {
    std::unique_lock<std::mutex> lk(e->ev_mu);
    e->ev_drained.wait(lk, [&] { return e->pollers == 0; });
    for (EventRec* r : e->events) delete r;
    e->events.clear();
  }
  if (e->efd >= 0) close(e->efd);
  delete e;
}

static Voice* voice_of(kt2n_engine* e, kt2n_voice_id id) {
  std::lock_guard<std::mutex> lk(e->voices_mu);
  if (id == KT2N_INVALID_ID || id > e->voices.size()) return nullptr;
  Voice* v = e->voices[size_t(id - 1)].get();
  return v->evicted.load() ? nullptr : v;
}

KT2N_API int32_t kt2n_voice_lookup(kt2n_engine* e, const char* name, kt2n_voice_id* out, kt2n_error* err) {
  if (!e || !name || !out) return KT2N_ERR_INVALID_ARGUMENT;
  {
    std::lock_guard<std::mutex> lk(e->voices_mu);
    for (size_t i = 0; i < e->voices.size(); ++i)
      if (e->voices[i]->kind == KT2N_VOICE_BUILTIN)
        for (const auto& n : e->voices[i]->names)
          if (n == name) {
            *out = i + 1;
            return KT2N_OK;
          }
  }
  set_err(err, KT2N_ERR_INVALID_VOICE, std::string("Voice '") + name + "' is not available.", "voice");
  return KT2N_ERR_INVALID_VOICE;
}

KT2N_API int32_t kt2n_voice_info_get(kt2n_engine* e, kt2n_voice_id id, kt2n_voice_info* out) {
  if (!e || !out) return KT2N_ERR_INVALID_ARGUMENT;
  Voice* v = voice_of(e, id);
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  kt2n_voice_info info{};
  info.struct_size = sizeof info;
  info.kind = v->kind;
  std::snprintf(info.name, sizeof info.name, "%s", v->id.c_str());
  std::snprintf(info.language, sizeof info.language, "%s", v->language.c_str());
  info.prompt_tokens = v->prompt_tokens;
  info.prompt_frames = v->prompt_frames;
  info.lm_prefix_ids = v->lm_prefix_ids;
  info.hidden = v->hidden;
  info.peak_gain_db = float(v->peak_gain_db);
  info.refs = v->refs.load();
  std::memcpy(out, &info, std::min<size_t>(out->struct_size ? out->struct_size : sizeof info, sizeof info));
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_speaker(kt2n_engine* e, kt2n_voice_id id, float out[512]) {
  Voice* v = e ? voice_of(e, id) : nullptr;
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  if (v->raw_speaker.size() != 512) return KT2N_ERR_UNSUPPORTED;
  std::memcpy(out, v->raw_speaker.data(), 512 * sizeof(float));
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_build(kt2n_engine* e, const kt2n_voice_build_params* p, kt2n_job_id* out,
                                  kt2n_error* err) {
  if (!e || !p || !out) return KT2N_ERR_INVALID_ARGUMENT;
#if !KT2N_ENGINE_WITH_VOICE
  set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, "built without the clone build service (L11)");
  return KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE;
#else
  if (!e->vservice && !e->voice_build_exec) {
    set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, "clone builds need voice_build_dir");
    return KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE;
  }
  // A keyless KT2N_VB_INLINE build is never cached (L11 VoiceService) and never enters the inline LRU, so the caller
  // owns its eviction (kt2n_voice_evict; the conductor then releases its store space). Production's adapter always
  // sends a key. L11's engine_paths_check exercises the keyless path (builds succeed, the caller evicts).
  // Inline: transcript rules first, synchronously (runtime.py:11029, resolve_inline_voice checks before the slot).
  // Saved and consent creates take the slot first in production (create_voice / create_voice_consent run inside
  // the management slot), so L11's VoiceService checks their transcript after admission (429 before 400).
  std::string stripped;
  const std::string why = (p->flags & KT2N_VB_INLINE) && !(p->flags & KT2N_VB_CONSENT)
                              ? kt2n::voice::VoiceService::validate_transcript(
                                    p->reference_transcript ? p->reference_transcript : "", &stripped)
                              : std::string();
  if (!why.empty()) {
    const bool consent = (p->flags & KT2N_VB_CONSENT) != 0;
    set_err(err, consent ? KT2N_ERR_INVALID_CONSENT_TRANSCRIPT : KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT, why,
            consent ? "consent_transcript" : "reference_transcript");
    return consent ? KT2N_ERR_INVALID_CONSENT_TRANSCRIPT : KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT;
  }
  if ((p->flags & KT2N_VB_VERIFY_SPEAKER) && (p->consent_speaker_dim != 512 || !p->consent_speaker)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "consent_speaker must be 512 floats");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  auto j = std::make_shared<BuildJob>();
  j->user_data = p->user_data;
  j->flags = p->flags;
  if (p->flags & KT2N_VB_DECODED_PCM) {
    if (!p->pcm || !p->pcm_samples) {
      set_err(err, KT2N_ERR_INVALID_AUDIO, "no audio", "file");
      return KT2N_ERR_INVALID_AUDIO;
    }
    j->pcm.assign(p->pcm, p->pcm + p->pcm_samples);
    j->pcm_rate = p->pcm_rate;
  } else {
    if (!p->audio || !p->audio_len) {
      set_err(err, KT2N_ERR_INVALID_AUDIO, "no audio", "file");
      return KT2N_ERR_INVALID_AUDIO;
    }
    j->audio.assign(static_cast<const uint8_t*>(p->audio), static_cast<const uint8_t*>(p->audio) + p->audio_len);
  }
  if (p->flags & KT2N_VB_VERIFY_SPEAKER) j->consent.assign(p->consent_speaker, p->consent_speaker + 512);
  j->transcript = p->reference_transcript ? p->reference_transcript : "";
  j->language = p->language ? p->language : "";
  j->identity_key = p->identity_key ? p->identity_key : "";
  j->deadline_ns = p->deadline_ns ? p->deadline_ns : mono_ns() + int64_t(e->clone_timeout_s * 1e9);
  j->queued_s = mono_s();
  {
    std::lock_guard<std::mutex> lk(e->build_mu);
    // Admission (production: 4 running + 16 waiting, else custom_voice_creation_busy).
    if (e->builds_running + int(e->build_q.size()) >= e->clone_slots + e->clone_queue) {
      set_err(err, KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY, "Custom voice creation is busy. Retry shortly.");
      return KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY;
    }
    {
      std::lock_guard<std::mutex> ilk(e->mu);
      if (e->backend_failed.load()) {
        set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
        return KT2N_ERR_SERVICE_UNAVAILABLE;
      }
      j->id = e->next_id++;
    }
    *out = j->id;
    e->build_q.push_back(j);
    e->builds.emplace(j->id, std::move(j));
  }
  e->build_cv.notify_all();
  return KT2N_OK;
#endif
}
KT2N_API int32_t kt2n_voice_load(kt2n_engine* e, const void* artifact, size_t len, kt2n_voice_id* out,
                                 kt2n_error* err) {
  if (!e || !artifact || !out) return KT2N_ERR_INVALID_ARGUMENT;
  if (!e->catalogue) {
    set_err(err, KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE, "no built-in catalogue loaded (voices_file)");
    return KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE;
  }
  // production _load_custom_reference gate (runtime.py:13175) via L11 check_saved_voice.
  kt2n::voice::VoicesBlob parsed;
  const auto verdict = kt2n::voice::check_saved_voice(artifact, len, *e->catalogue, &parsed);
  if (verdict.status != 0) {
    std::string why = verdict.code;
    for (const auto& m : verdict.mismatched) why += " " + m;
    set_err(err, verdict.status, why, verdict.param);
    return verdict.status;
  }
  const auto& rec = parsed.voices[0];
  auto v = std::make_unique<Voice>();
  v->kind = KT2N_VOICE_SAVED;
  v->id = rec.id();
  v->names = {v->id};
  v->peak_gain_db = kt2n::voice::kCustomPeakGainDb;  // custom voices: the catalogue median
  if (const auto* l = rec.meta.get("language"); l && l->is_string() && !l->as_string().empty()) v->language = l->as_string();
  v->prompt_tokens = int(rec.prompt_token.size());
  v->prompt_frames = int(rec.prompt_feat.size() / 80);
  v->lm_prefix_ids = int(rec.lm_prefix.size());
  v->raw_speaker = rec.raw_speaker;
  v->geometry = policy::geometry_owner(v->prompt_tokens, v->prompt_frames, int(rec.embedding.size()));
  v->artifact.assign(static_cast<const uint8_t*>(artifact), static_cast<const uint8_t*>(artifact) + len);
  *out = register_voice(e, std::move(v), rec.prompt_token, rec.prompt_feat, rec.embedding, rec.projected_speaker,
                        rec.lm_prefix);
  if (!*out) {
    set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
    return KT2N_ERR_SERVICE_UNAVAILABLE;
  }
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_export(kt2n_engine* e, kt2n_voice_id id, void* buf, size_t cap, size_t* len) {
  if (!e || !len) return KT2N_ERR_INVALID_ARGUMENT;
  Voice* v = voice_of(e, id);
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  if (v->artifact.empty()) return KT2N_ERR_INVALID_ARGUMENT;  // built-ins are not exported
  *len = v->artifact.size();
  if (!buf || cap < v->artifact.size()) return KT2N_ERR_TOO_SMALL;
  std::memcpy(buf, v->artifact.data(), v->artifact.size());
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_evict(kt2n_engine* e, kt2n_voice_id id) {
  Voice* v = e ? voice_of(e, id) : nullptr;
  if (!v) return KT2N_ERR_INVALID_HANDLE;
  if (v->kind == KT2N_VOICE_BUILTIN) return KT2N_ERR_INVALID_ARGUMENT;
  v->evicted = true;  // new submits fail; requests holding it finish with it
  e->voice_sweep = true;  // the conductor gives its LM store space back once no request holds it
  e->cv.notify_all();
  return KT2N_OK;
}

KT2N_API int32_t kt2n_submit(kt2n_engine* e, const kt2n_request_params* p, kt2n_request_id* out, kt2n_error* err) {
  if (!e || !p || !out || p->struct_size < offsetof(kt2n_request_params, user_data)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "params");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  std::string text(p->text ? p->text : "", p->text ? p->text_len : 0);
  const auto u32 = kt2n::textplan::from_utf8(text);
  if (kt2n::textplan::py_strip(u32).empty()) {
    set_err(err, KT2N_ERR_INVALID_INPUT, "input must not be blank", "input");
    return KT2N_ERR_INVALID_INPUT;
  }
  if (u32.size() > 40000) {
    set_err(err, KT2N_ERR_REQUEST_TOO_LARGE, "input must be at most 40000 characters", "input");
    return KT2N_ERR_REQUEST_TOO_LARGE;
  }
  Voice* v = voice_of(e, p->voice);
  if (!v || v->consent_only) {
    set_err(err, KT2N_ERR_INVALID_VOICE, "voice is not available", "voice");
    return KT2N_ERR_INVALID_VOICE;
  }
  if (!(p->speed >= 0.25 && p->speed <= 4.0)) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, "speed must be between 0.25 and 4.0", "speed");
    return KT2N_ERR_INVALID_REQUEST;
  }
  if (p->format < KT2N_FORMAT_PCM || (p->format > KT2N_FORMAT_FLAC && p->format != KT2N_FORMAT_F32)) {
    set_err(err, KT2N_ERR_INVALID_RESPONSE_FORMAT, "unsupported response_format", "response_format");
    return KT2N_ERR_INVALID_RESPONSE_FORMAT;
  }
#if !KT2N_ENGINE_WITH_AV
  if (encoded_format(p->format)) {
    set_err(err, KT2N_ERR_UNSUPPORTED, "built without the in-process encoders (KT2N_AV)", "response_format");
    return KT2N_ERR_UNSUPPORTED;
  }
#endif
  if (p->speech_mode == KT2N_SPEECH_MODE_FAST || p->speech_mode == KT2N_SPEECH_MODE_QUALITY ||
      (p->flags & KT2N_REQ_DEMO_CHECK)) {
    // Live SPEECH_MODES=0: only DEFAULT and DEMO (stream:false) are reachable; the others fail closed.
    set_err(err, KT2N_ERR_UNSUPPORTED, "speech mode not served live (SPEECH_MODES=0)", "speech_mode");
    return KT2N_ERR_UNSUPPORTED;
  }
  if ((p->flags & KT2N_REQ_TIMELINE) && !(e->flags & KT2N_CFG_TIMELINE_ALLOWED)) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, "timeline not allowed on this engine", "timeline");
    return KT2N_ERR_INVALID_REQUEST;
  }
  // Sampler (sampler_params.resolve with SAMPLER_DEFAULT = expressive).
  kt2n::lm::SamplerRequest sr;
  const kt2n_sampler& s = p->sampler;
  if (s.present & KT2N_SAMPLER_HAS_MODE) sr.mode = s.mode == KT2N_SAMPLER_STABLE ? "stable" : "expressive";
  if (s.present & KT2N_SAMPLER_HAS_TEMPERATURE) sr.temperature = s.temperature;
  if (s.present & KT2N_SAMPLER_HAS_TOP_P) sr.top_p = s.top_p;
  if (s.present & KT2N_SAMPLER_HAS_TOP_K) sr.top_k = double(s.top_k);
  if (s.present & KT2N_SAMPLER_HAS_MIN_P) sr.min_p = s.min_p;
  if (s.present & KT2N_SAMPLER_HAS_MAX_NEW_TOKENS) sr.max_new_tokens = double(s.max_new_tokens);
  kt2n::lm::SamplerSettings settings;
  std::string param;
  const std::string msg = kt2n::lm::resolve_sampler(sr, kt2n::lm::SamplerMode::kExpressive, &settings, &param);
  if (!msg.empty()) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, msg, param);
    return KT2N_ERR_INVALID_REQUEST;
  }
  // control_json (L8b, appended): the request's GenerationControl policy fields. Known keys are applied as
  // runtime.py applies them; any other key fails closed so a request is never served under another policy.
  std::optional<double> hold_pinned;
  std::optional<bool> emo_preset, skip_open, phrase, normalize;
  const char* cj = p->struct_size >= offsetof(kt2n_request_params, control_json) + sizeof(const char*)
                       ? p->control_json
                       : nullptr;
  if (cj && *cj) {
    kt2n::json::Value doc;
    try {
      doc = kt2n::json::parse(cj);
    } catch (const std::exception&) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "control_json is not valid JSON", "control_json");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    if (!doc.is_object()) {
      set_err(err, KT2N_ERR_INVALID_ARGUMENT, "control_json must be an object", "control_json");
      return KT2N_ERR_INVALID_ARGUMENT;
    }
    for (const auto& [k, cv] : doc.members()) {
      bool ok = true;
      if (cv.is_null()) {
        ok = true;  // "not set"
      } else if (k == "startup_hold_seconds") {
        ok = cv.is_number();
        if (ok) hold_pinned = cv.as_double();
      } else if (k == "initial_audio_buffer_seconds" || k == "concurrent_initial_audio_buffer_seconds") {
        ok = cv.is_number() && cv.as_double() == 0.0;  // live: 0 (STREAM_*INITIAL_BUFFER_SECONDS=0)
      } else if (k == "normalize") {
        ok = cv.is_bool();
        if (ok) normalize = cv.as_bool();
      } else if (k == "emotion") {
        ok = cv.is_bool();
        if (ok) emo_preset = cv.as_bool();
      } else if (k == "emotion_carry_tag") {
        ok = cv.is_string();  // read only with EMO_CARRY_LEADING_TAG (0 live)
      } else if (k == "skip_opening_silence") {
        ok = cv.is_bool();
        if (ok) skip_open = cv.as_bool();
      } else if (k == "q_skip_limiter") {
        ok = cv.is_bool();  // the internal-silence limiter is off live either way
      } else if (k == "demo_check") {
        ok = cv.is_bool() && !cv.as_bool();  // DEMO+CHECK is unreachable live
      } else if (k == "phrase") {
        ok = cv.is_bool();
        if (ok) phrase = cv.as_bool();
      } else {
        ok = false;
      }
      if (!ok) {
        set_err(err, KT2N_ERR_UNSUPPORTED, "control field not served by the native engine: " + k, k);
        return KT2N_ERR_UNSUPPORTED;
      }
    }
  }
  auto q = std::make_unique<Request>();
  q->user_data = p->user_data;
  q->flags = p->flags;
  q->hold_pinned = hold_pinned;
  q->emotion_preset = emo_preset;
  q->realtime = phrase.value_or(false);
  q->skip_opening = skip_open.value_or(true);
  q->normalize = normalize.value_or((p->flags & KT2N_REQ_NORMALIZE_OFF) == 0);
  q->timeline = (p->flags & KT2N_REQ_TIMELINE) != 0;
  q->voice = v;
  q->raw = text;
  q->sampler = settings;
  q->speed = p->speed;
  q->format = p->format;
  q->output = p->output;
  q->demo = p->output == KT2N_OUTPUT_COMPLETE || p->speech_mode == KT2N_SPEECH_MODE_DEMO;
  q->deadline_ns = p->deadline_ns;
  q->started_s = mono_s();
  q->salt = (p->flags & KT2N_REQ_SEED) ? p->seed : policy::request_seed_salt(p->request_id ? p->request_id : "");
  kt2n::post::ResponsePostConfig rc;
  rc.speed = p->speed;
  rc.onset_trim = !q->demo;
  rc.withhold = !q->demo;
  q->rpost = std::make_unique<kt2n::post::ResponsePost>(rc);
  kt2n::post::HoldPolicy hold_policy;
  if (q->hold_pinned) hold_policy.pinned = *q->hold_pinned;
  q->hold = std::make_unique<kt2n::post::StartupHold>(hold_policy);
  // A streamed non-DEMO response publishes through the hold (complete files and DEMO rows have no commits on it).
  q->awaiting_hold.store(q->output == KT2N_OUTPUT_STREAM && !q->demo, std::memory_order_relaxed);
#if KT2N_ENGINE_WITH_AV
  if (encoded_format(p->format)) {
    std::string why;
    q->enc = kt2n::post::StreamEncoder::create(format_name(p->format), &why);
    if (!q->enc) {
      set_err(err, KT2N_ERR_INTERNAL, "encoder: " + why, "response_format");
      return KT2N_ERR_INTERNAL;
    }
  }
#endif
  if (q->output == KT2N_OUTPUT_STREAM) {
    // Ceiling 0.999, lossy formats 0.891 (PeakNorm.for_request).
    q->peak.emplace(false, v->peak_gain_db, kt2n::post::is_lossy_format(format_name(p->format)) ? 0.891 : 0.999, 5.0);
    if (!(q->hold_pinned && *q->hold_pinned <= 0.0)) q->clock.hold();  // the startup hold withholds the first byte
  }
  if (!register_request(e, std::move(q), out)) {
    set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
    return KT2N_ERR_SERVICE_UNAVAILABLE;
  }
  return KT2N_OK;
}

KT2N_API int32_t kt2n_voice_build_cancel_queued(kt2n_engine* e, kt2n_job_id id, int32_t* cancelled) {
  if (!e || !cancelled) return KT2N_ERR_INVALID_ARGUMENT;
  *cancelled = 0;
  // Queue membership and voice_worker dequeue use the same mutex. No later
  // cancellation can interrupt an admitted stored create through this API.
  std::lock_guard<std::mutex> lk(e->build_mu);
  for (const auto& job : e->build_q) {
    if (job->id == id) {
      job->cancelled = true;
      *cancelled = 1;
      return KT2N_OK;
    }
  }
  return KT2N_OK;
}

KT2N_API int32_t kt2n_cancel(kt2n_engine* e, uint64_t id, int32_t reason) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  {
    std::lock_guard<std::mutex> blk(e->build_mu);
    auto bit = e->builds.find(id);
    if (bit != e->builds.end()) {
      bit->second->cancelled = true;
      e->build_cv.notify_all();
      return KT2N_OK;
    }
  }
  {
    std::lock_guard<std::mutex> slk(e->sess_mu);
    if (Session* s = session_of(e, id)) {
      if (!s->cancelled) {
        s->cancelled = true;
        s->cancel_status = reason == KT2N_CANCEL_DEADLINE ? KT2N_ERR_DEADLINE_EXCEEDED : KT2N_ERR_CANCELLED;
      }
      std::vector<uint64_t> phrases = s->phrase_ids;
      std::lock_guard<std::mutex> lk(e->mu);
      for (uint64_t pid : phrases) {
        auto it = e->requests.find(pid);
        if (it != e->requests.end() && !it->second->cancelled.load()) {
          it->second->cancel_status = s->cancel_status;
          it->second->cancelled = true;
        }
      }
      e->cv.notify_all();
      e->sess_cv.notify_all();
      return KT2N_OK;
    }
  }
  std::lock_guard<std::mutex> lk(e->mu);
  auto it = e->requests.find(id);
  if (it == e->requests.end()) return KT2N_ERR_INVALID_HANDLE;
  Request* q = it->second.get();
  if (!q->cancelled.load()) {
    q->cancel_status = reason == KT2N_CANCEL_DEADLINE ? KT2N_ERR_DEADLINE_EXCEEDED : KT2N_ERR_CANCELLED;
    q->cancelled = true;
  }
  e->cv.notify_all();
  return KT2N_OK;
}


KT2N_API int32_t kt2n_release(kt2n_engine* e, uint64_t id) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  {
    std::lock_guard<std::mutex> blk(e->build_mu);
    auto bit = e->builds.find(id);
    if (bit != e->builds.end()) {
      // The worker retains ownership through its final callback and loop accounting. Keep a local
      // owner too, so erasing the map cannot destroy the mutex while jl still holds it.
      const auto j = bit->second;
      std::lock_guard<std::mutex> jl(j->ev_lock);
      if (!j->terminal) return KT2N_ERR_INVALID_ARGUMENT;
      e->builds.erase(bit);
      return KT2N_OK;
    }
  }
  {
    std::lock_guard<std::mutex> slk(e->sess_mu);
    if (Session* s = session_of(e, id)) {
      if (!s->terminal) return KT2N_ERR_INVALID_ARGUMENT;
      // Phrases emit through the session object, so it is kept (small) until engine destroy; its phrase
      // requests are released for the conductor's sweep.
      std::lock_guard<std::mutex> lk(e->mu);
      for (uint64_t pid : s->phrase_ids) {
        auto pit = e->requests.find(pid);
        if (pit != e->requests.end()) pit->second->release_requested = true;
      }
      s->phrase_ids.clear();
      e->release_pending = true;
      return KT2N_OK;
    }
  }
  std::lock_guard<std::mutex> lk(e->mu);
  auto it = e->requests.find(id);
  if (it == e->requests.end()) return KT2N_ERR_INVALID_HANDLE;
  {
    std::lock_guard<std::mutex> qlk(it->second->mu);
    if (!it->second->terminal) return KT2N_ERR_INVALID_ARGUMENT;
  }
  // The conductor frees it once no row or post item points at it (its loop sweeps released requests).
  if (it->second->release_requested.exchange(true)) return KT2N_ERR_INVALID_HANDLE;
  e->release_pending = true;
  e->cv.notify_all();
  return KT2N_OK;
}

KT2N_API int32_t kt2n_session_open(kt2n_engine* e, const kt2n_session_params* p, kt2n_session_id* out,
                                   kt2n_error* err) {
  if (!e || !p || !out || p->struct_size < sizeof(kt2n_session_params)) {
    set_err(err, KT2N_ERR_INVALID_ARGUMENT, "session params are NULL or too small");
    return KT2N_ERR_INVALID_ARGUMENT;
  }
  if (p->format != KT2N_FORMAT_PCM && p->format != KT2N_FORMAT_F32) {
    set_err(err, KT2N_ERR_INVALID_RESPONSE_FORMAT, "sessions support PCM16 or float32 audio", "format");
    return KT2N_ERR_INVALID_RESPONSE_FORMAT;
  }
  Voice* v = voice_of(e, p->voice);
  if (!v || v->consent_only) {
    set_err(err, KT2N_ERR_INVALID_VOICE, "voice is not available", "voice");
    return KT2N_ERR_INVALID_VOICE;
  }
  if (v->kind != KT2N_VOICE_BUILTIN) {
    set_err(err, KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED, "Custom voices are not supported on realtime sessions.",
            "voice");
    return KT2N_ERR_CUSTOM_VOICE_REALTIME_NOT_SUPPORTED;
  }
  if (!(p->speed >= 0.25 && p->speed <= 4.0)) {
    set_err(err, KT2N_ERR_INVALID_SESSION_CONFIGURATION, "speed must be between 0.25 and 4.0", "speed");
    return KT2N_ERR_INVALID_SESSION_CONFIGURATION;
  }
  kt2n::lm::SamplerRequest sr;
  const kt2n_sampler& sm = p->sampler;
  if (sm.present & KT2N_SAMPLER_HAS_MODE) sr.mode = sm.mode == KT2N_SAMPLER_STABLE ? "stable" : "expressive";
  if (sm.present & KT2N_SAMPLER_HAS_TEMPERATURE) sr.temperature = sm.temperature;
  if (sm.present & KT2N_SAMPLER_HAS_TOP_P) sr.top_p = sm.top_p;
  if (sm.present & KT2N_SAMPLER_HAS_TOP_K) sr.top_k = double(sm.top_k);
  if (sm.present & KT2N_SAMPLER_HAS_MIN_P) sr.min_p = sm.min_p;
  if (sm.present & KT2N_SAMPLER_HAS_MAX_NEW_TOKENS) sr.max_new_tokens = double(sm.max_new_tokens);
  kt2n::lm::SamplerSettings settings;
  std::string param;
  const std::string msg = kt2n::lm::resolve_sampler(sr, kt2n::lm::SamplerMode::kExpressive, &settings, &param);
  if (!msg.empty()) {
    set_err(err, KT2N_ERR_INVALID_REQUEST, msg, param);
    return KT2N_ERR_INVALID_REQUEST;
  }
  auto s = std::make_unique<Session>();
  s->normalize = (p->flags & KT2N_REQ_NORMALIZE_OFF) == 0;
  s->user_data = p->user_data;
  s->flags = p->flags;
  s->voice = v;
  s->request_id = p->request_id ? p->request_id : "";
  s->seeded = (p->flags & KT2N_REQ_SEED) != 0;
  s->salt = s->seeded ? p->seed : policy::request_seed_salt(s->request_id);
  s->sampler = settings;
  s->speed = p->speed;
  s->format = p->format;
  s->deadline_ns = p->deadline_ns;
  s->opened_s = mono_s();
  s->peak.emplace(false, v->peak_gain_db, 0.999, 5.0);
  {
    std::lock_guard<std::mutex> slk(e->sess_mu);
    std::lock_guard<std::mutex> lk(e->mu);
    if (e->backend_failed.load()) {
      set_err(err, KT2N_ERR_SERVICE_UNAVAILABLE, "engine backend failed");
      return KT2N_ERR_SERVICE_UNAVAILABLE;
    }
    s->id = e->next_id++;
    *out = s->id;
    e->sessions.emplace(s->id, std::move(s));
  }
  e->sess_cv.notify_all();
  return KT2N_OK;
}

// A refused client event ends the session as production's socket handler does (api.py:6887-6955):
// input after done -> input_already_closed (invalid_request_error, close 1008); an oversized message or
// session -> ValueError -> speech_generation_failed (server_error, close 1011). sess_mu held.
static int32_t gate_error(kt2n_engine* e, Session* s, kt2n_error* err, kt2n::textplan::WsInputGate::Verdict v) {
  using V = kt2n::textplan::WsInputGate::Verdict;
  int32_t st = KT2N_OK;
  switch (v) {
    case V::kInputClosed:
      set_err(err, KT2N_ERR_INPUT_ALREADY_CLOSED, "Realtime TTS input is already closed");
      st = KT2N_ERR_INPUT_ALREADY_CLOSED;
      break;
    case V::kMessageTooLarge:
    case V::kInputTooLarge:
      set_err(err, KT2N_ERR_SPEECH_GENERATION_FAILED, "Speech generation failed");
      st = KT2N_ERR_SPEECH_GENERATION_FAILED;
      break;
    default:
      return KT2N_OK;
  }
  if (!s->cancelled) {
    s->cancelled = true;
    s->cancel_status = st;
    std::lock_guard<std::mutex> lk(e->mu);
    for (uint64_t pid : s->phrase_ids) {
      auto it = e->requests.find(pid);
      if (it != e->requests.end() && !it->second->cancelled.load()) {
        it->second->cancel_status = KT2N_ERR_CANCELLED;
        it->second->cancelled = true;
      }
    }
  }
  e->cv.notify_all();
  e->sess_cv.notify_all();
  return st;
}

KT2N_API int32_t kt2n_session_text_append(kt2n_engine* e, kt2n_session_id id, const char* text, size_t len,
                                          kt2n_error* err) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  const auto u32 = kt2n::textplan::from_utf8(std::string_view(text ? text : "", text ? len : 0));
  std::lock_guard<std::mutex> lk(e->sess_mu);
  Session* s = session_of(e, id);
  if (!s) return KT2N_ERR_INVALID_HANDLE;
  const auto v = s->gate.append(u32.size());
  if (v == kt2n::textplan::WsInputGate::Verdict::kSkip) return KT2N_OK;
  if (v != kt2n::textplan::WsInputGate::Verdict::kForward) return gate_error(e, s, err, v);
  s->planner.feed(kt2n::textplan::WsPlanner::EventType::kText, u32);
  s->dirty = true;
  e->sess_cv.notify_all();
  return KT2N_OK;
}

KT2N_API int32_t kt2n_session_commit(kt2n_engine* e, kt2n_session_id id, kt2n_error* err) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->sess_mu);
  Session* s = session_of(e, id);
  if (!s) return KT2N_ERR_INVALID_HANDLE;
  const auto v = s->gate.commit();
  if (v != kt2n::textplan::WsInputGate::Verdict::kForward) return gate_error(e, s, err, v);
  s->planner.feed(kt2n::textplan::WsPlanner::EventType::kCommit);
  s->dirty = true;
  e->sess_cv.notify_all();
  return KT2N_OK;
}

KT2N_API int32_t kt2n_session_done(kt2n_engine* e, kt2n_session_id id, kt2n_error* err) {
  if (!e) return KT2N_ERR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> lk(e->sess_mu);
  Session* s = session_of(e, id);
  if (!s) return KT2N_ERR_INVALID_HANDLE;
  const auto v = s->gate.done();
  if (v != kt2n::textplan::WsInputGate::Verdict::kForward) return gate_error(e, s, err, v);
  s->planner.feed(kt2n::textplan::WsPlanner::EventType::kDone);
  s->dirty = true;
  e->sess_cv.notify_all();
  return KT2N_OK;
}

KT2N_API int kt2n_event_fd(kt2n_engine* e) { return e ? e->efd : -1; }

KT2N_API int32_t kt2n_poll(kt2n_engine* e, kt2n_event* out, int32_t max, int32_t timeout_ms, int32_t* n) {
  if (!e || !out || !n || max <= 0) return KT2N_ERR_INVALID_ARGUMENT;
  std::unique_lock<std::mutex> lk(e->ev_mu);
  ++e->pollers;
  if (e->events.empty() && timeout_ms != 0) {
    auto pred = [&] { return !e->events.empty() || e->stop.load(); };
    if (timeout_ms < 0)
      e->ev_cv.wait(lk, pred);
    else
      cv_wait_for(e->ev_cv, lk, std::chrono::milliseconds(timeout_ms), pred);
  }
  // Shutdown returns no borrowed buffers: callers already waiting may finish safely while destroy
  // drains them. Callers must still stop new calls and release any earlier borrowed events first.
  if (e->stop.load()) {
    *n = 0;
    if (--e->pollers == 0) e->ev_drained.notify_all();
    return KT2N_OK;
  }
  int32_t k = 0;
  while (k < max && !e->events.empty()) {
    EventRec* r = e->events.front();
    e->events.pop_front();
    r->ev.data = r->data.empty() ? nullptr : reinterpret_cast<const uint8_t*>(r->data.data());
    r->ev.data_len = r->data.size();
    r->ev.name = r->name.empty() ? nullptr : r->name.c_str();
    r->ev.fields_json = r->fields.empty() ? nullptr : r->fields.c_str();
    out[k++] = r->ev;
    e->outstanding[{r->ev.id, r->ev.seq}].reset(r);
  }
  *n = k;
  if (e->events.empty() && e->efd >= 0) {
    uint64_t v;
    (void)!read(e->efd, &v, sizeof v);
  }
  if (--e->pollers == 0) e->ev_drained.notify_all();
  return KT2N_OK;
}

KT2N_API void kt2n_events_release(kt2n_engine* e, const kt2n_event* events, int32_t n) {
  if (!e || !events) return;
  std::lock_guard<std::mutex> lk(e->ev_mu);
  for (int32_t i = 0; i < n; ++i) e->outstanding.erase({events[i].id, events[i].seq});
}

KT2N_API int32_t kt2n_stats_get(kt2n_engine* e, kt2n_stats* out) {
  if (!e || !out) return KT2N_ERR_INVALID_ARGUMENT;
  kt2n_stats s{};
  s.struct_size = sizeof s;
  s.device_arena_bytes = e->device_reserved_bytes;
  s.device_arena_used = e->device_reserved_bytes;
  s.active_requests = e->active_requests.load();
  s.lm_rows = e->lm_rows.load();
  s.render_inflight = e->render_inflight.load();
  {
    std::lock_guard<std::mutex> lk(e->voices_mu);
    s.voices_loaded = int32_t(e->voices.size());
  }
  s.requests_done = e->done.load();
  s.requests_failed = e->failed.load();
  s.requests_cancelled = e->cancelled.load();
  s.lm_steps = e->lm_steps.load();
  s.renders = e->renders.load();
  s.render_rows = e->render_rows.load();
  std::memcpy(out, &s, std::min<size_t>(out->struct_size ? out->struct_size : sizeof s, sizeof s));
  return KT2N_OK;
}

KT2N_API int32_t kt2n_stats_json(kt2n_engine* e, char* buf, size_t cap, size_t* len) {
  if (!e || !len) return KT2N_ERR_INVALID_ARGUMENT;
  size_t held;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    held = e->requests.size();
  }
  std::ostringstream o;
  // "render":"stub" whenever the audio is placeholder (CPU stub or L7 stand-in stages): the adapter's stub
  // detection reads it; render_impl names the backend.
  const std::string rimpl = e->render->name();
  const bool placeholder = rimpl == "stub" || rimpl == "l7-standin";
  o << "{\"lm\":\"" << e->lm->name() << "\",\"render\":\"" << (placeholder ? "stub" : rimpl)
    << "\",\"render_impl\":\"" << rimpl
    << "\",\"normalizer\":\"" << (e->text->real_normalizer() ? "native" : "passthrough") << "\",\"tokenizer\":\""
    << (e->text->real_tokenizer() ? "native" : "codepoints") << "\",\"active_requests\":" << e->active_requests.load()
    << ",\"lm_rows\":" << e->lm_rows.load() << ",\"lm_steps\":" << e->lm_steps.load()
    << ",\"renders\":" << e->renders.load() << ",\"render_rows\":" << e->render_rows.load()
    << ",\"requests_done\":" << e->done.load() << ",\"requests_failed\":" << e->failed.load()
    << ",\"requests_cancelled\":" << e->cancelled.load() << ",\"levers_unknown\":" << e->levers_unknown.size() << ",\"levers_unknown_keys\":" << [&] {
         std::string a = "[";
         for (size_t k = 0; k < e->levers_unknown.size(); ++k) a += (k ? ",\"" : "\"") + e->levers_unknown[k] + "\"";
         return a + "]";
       }()
    << ",\"requests_held\":" << held << ",\"startup_sharing\":" << e->sharing.load()
    << ",\"steady_requests\":" << e->steady.load() << ",\"previews_due\":" << e->previews_due.load()
    << ",\"lm_backpressure\":" << e->lm_backpressure.load()
    << ",\"render_backpressure\":" << e->render_backpressure.load() << ",\"poll\":" << (e->poll ? "true" : "false")
    << ",\"poll_idle\":" << e->poll_idle.load() << ",\"render_state\":" << e->render->stats_json()
    // RC3: the yield keys appear only when KT2N_LM_RENDER_YIELD is on, so the flag-off stats text stays the
    // integration HEAD's byte for byte (engine_native.py reads stats into a fixed 4,096-byte buffer).
    << [&] {
         if (!e->lm_render_yield) return std::string();
         return ",\"lm_render_yield\":true,\"lm_render_yield_episodes\":" +
                std::to_string(e->lm_render_yield_episodes.load()) +
                ",\"lm_render_yield_switches\":" + std::to_string(e->lm_render_yield_switches.load());
       }()
    << ",\"render_voice_errors\":" << e->render_voice_errors << ",\"voice_store_full\":" << e->voice_store_full.load()
    << ",\"lm_voices_released\":" << e->lm_voices_released.load()
    << ",\"render_voices_released\":" << e->render_voices_released.load()
    << ",\"turns\":" << e->turns.load() << ",\"turn_us_mean\":"
    << (e->turns.load() ? double(e->turn_ns.load()) / 1000.0 / double(e->turns.load()) : 0.0)
    << ",\"turn_us_max\":" << double(e->turn_ns_max.load()) / 1000.0
    << ",\"turns_over_1ms\":" << e->turns_over_1ms.load() << ",\"turns_over_5ms\":" << e->turns_over_5ms.load()
    << ",\"turn_max_decide_us\":" << double(e->turn_max_decide_ns.load()) / 1000.0
    << ",\"turn_max_rows\":" << e->turn_max_rows.load()
    << ",\"calls_per_turn\":{\"commit_turns\":" << e->commit_turns.load() << ",\"commit_groups\":" << e->commit_groups.load()
    << ",\"commit_same_stream_extra\":" << e->commit_same_stream_extra.load()
    << ",\"commit_extra_rows\":" << e->commit_extra_rows.load() << ",\"commit_mixed_groups\":" << e->commit_mixed_groups.load()
    << ",\"commit_mixed_later_rows\":" << e->commit_mixed_later_rows.load() << ",\"final_turns\":" << e->final_turns.load()
    << ",\"final_calls\":" << e->final_calls.load() << ",\"final_extra_calls\":" << e->final_extra_calls.load() << "}"
    << ",\"split_urgent\":" << (e->split_urgent ? "true" : "false") << ",\"split_urgent_calls\":" << e->split_urgent_calls.load()
    << ",\"hold_urgent\":" << (e->hold_urgent > 0 ? "true" : "false") << ",\"hold_urgent_level\":" << e->hold_urgent
    << ",\"hold_urgent_calls\":" << e->hold_urgent_calls.load() << ",\"hold_urgent_finals\":" << e->hold_urgent_finals.load()
    << ",\"hold_urgent_rows\":" << e->hold_urgent_rows.load() << ",\"hold_urgent_splits\":" << e->hold_urgent_splits.load()
    << ",\"buffer_urgent_ms\":" << e->buffer_urgent_ms << ",\"buffer_urgent_rows\":" << e->buffer_urgent_rows.load()
    << ",\"buffer_urgent_finals\":" << e->buffer_urgent_finals.load()
    << ",\"buffer_role\":\"" << (e->buffer_to_aux ? "aux" : "urgent") << "\",\"buffer_mid_calls\":" << e->buffer_mid_calls.load()
    << ",\"lm_step_ms_by_rows\":{" << [&] {
         // "21+" (all steps above 20 rows, as before) plus its parts 21-32 / 33-48 / 49+ for the capacity runs.
         static const char* names[9] = {"1-4", "5-8", "9-12", "13-16", "17-20", "21+", "21-32", "33-48", "49+"};
         std::string by_rows;  // not "o": that name is the enclosing stats stream (-Werror=shadow, L14 c8)
         for (int b = 0; b < 9; ++b) {
           uint64_t n = 0, ns = 0;
           for (int k = (b < 5 ? b : b == 5 ? 5 : b - 1); k <= (b < 5 ? b : b == 5 ? 7 : b - 1); ++k) {
             n += e->lm_iv_steps[k].load();
             ns += e->lm_iv_ns[k].load();
           }
           if (!n) continue;
           if (!by_rows.empty()) by_rows += ",";
           by_rows += std::string("\"") + names[b] + "\":{\"steps\":" + std::to_string(n) + ",\"ms\":" +
                      jnum(double(ns) / 1e6 / double(n)) + "}";
         }
         return by_rows;
       }() << "}"
    << ",\"x_sched\":" << ((e->flags & KT2N_CFG_X_SCHED) ? "true" : "false")
    << ",\"dw_runway_ms\":" << e->lv.dw_runway_ms << ",\"max_streams\":" << e->max_streams
    << ",\"engine_rows\":" << e->row_cap << ",\"engine_rows_why\":\"" << e->row_cap_why << "\""
    << ",\"capacity_order\":" << (e->capacity_order && e->row_cap < e->max_streams ? "true" : "false") << "}";
  const std::string s = o.str();
  *len = s.size() + 1;
  if (!buf || cap < s.size() + 1) return KT2N_ERR_TOO_SMALL;
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return KT2N_OK;
}

}  // extern "C"
