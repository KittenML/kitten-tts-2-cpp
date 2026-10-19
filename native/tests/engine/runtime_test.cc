// L8: engine runtime tests on the CPU stubs (no GPU, no model files): the C ABI contract, the
// conductor's turn loop, admission back-pressure, cancel, stream and complete outputs.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "kt2n/engine.h"
#include "voice_blob.h"  // L11's writer: a synthetic catalogue + saved voice for the voice-release test

namespace {

struct Result {
  int status = -1;
  std::string bytes;
  int64_t samples = 0;
  int audio_events = 0;
  uint64_t last_seq = 0;
  bool seq_ok = true;
  bool final_seen = false;
  std::vector<int32_t> opening_skipped;  // kt2n_event.opening_skipped_tokens per AUDIO event
};

class Eng {
 public:
  explicit Eng(int max_streams = 10) {
    kt2n_engine_config cfg{};
    cfg.struct_size = sizeof cfg;
    cfg.abi_version = KT2N_ABI_VERSION;
    cfg.flags = KT2N_CFG_TIMELINE_ALLOWED;
    cfg.max_streams = max_streams;
    kt2n_error err{};
    const int status = kt2n_engine_create(&cfg, &e, &err);
    if (status != KT2N_OK) throw std::runtime_error(std::string("engine fixture initialization failed: ") + err.message);
  }
  ~Eng() { kt2n_engine_destroy(e); }
  uint64_t submit(const std::string& text, int output = KT2N_OUTPUT_STREAM, int format = KT2N_FORMAT_PCM) {
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = 1;
    p.request_id = "t";
    p.seed = 42;
    p.speed = 1.0;
    p.format = format;
    p.output = output;
    kt2n_request_id id = 0;
    kt2n_error err{};
    EXPECT_EQ(kt2n_submit(e, &p, &id, &err), KT2N_OK) << err.message;
    return id;
  }
  std::map<uint64_t, Result> run(size_t n, int timeout_s = 30) {
    std::map<uint64_t, Result> out;
    const auto t0 = std::chrono::steady_clock::now();
    size_t done = 0;
    kt2n_event ev[32];
    while (done < n && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(timeout_s)) {
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i) {
        Result& r = out[ev[i].id];
        if (ev[i].seq != 0 && ev[i].seq != r.last_seq + 1) r.seq_ok = false;
        r.last_seq = ev[i].seq;
        if (ev[i].kind == KT2N_EVENT_AUDIO) {
          r.bytes.append(reinterpret_cast<const char*>(ev[i].data), ev[i].data_len);
          r.samples += ev[i].samples;
          r.audio_events++;
          r.final_seen |= ev[i].is_final_chunk != 0;
          r.opening_skipped.push_back(ev[i].opening_skipped_tokens);
        } else if (ev[i].kind == KT2N_EVENT_DONE) {
          r.status = ev[i].status;
          ++done;
        }
      }
      kt2n_events_release(e, ev, k);
    }
    return out;
  }
  kt2n_engine* e = nullptr;
};

TEST(Runtime, ErrorTableMatchesProductionCodes) {
  EXPECT_STREQ(kt2n_error_lookup(KT2N_ERR_INVALID_INPUT)->code, "invalid_input");
  EXPECT_EQ(kt2n_error_lookup(KT2N_ERR_CUSTOM_VOICE_CREATION_BUSY)->http_status, 429);
  EXPECT_STREQ(kt2n_error_lookup(KT2N_ERR_DEADLINE_EXCEEDED)->code, "speech_generation_failed");
  EXPECT_STREQ(kt2n_error_lookup(9999)->code, "internal_error");
}

TEST(Runtime, ValidationIsSynchronous) {
  Eng g;
  kt2n_request_params p{};
  p.struct_size = sizeof p;
  p.text = "   ";
  p.text_len = 3;
  p.voice = 1;
  p.speed = 1.0;
  kt2n_request_id id;
  kt2n_error err{};
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_INPUT);
  EXPECT_STREQ(err.param, "input");
  p.text = "Hello.";
  p.text_len = 6;
  p.speed = 5.0;
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_REQUEST);
  EXPECT_STREQ(err.param, "speed");
  p.speed = 1.0;
  p.sampler.present = KT2N_SAMPLER_HAS_TEMPERATURE;
  p.sampler.temperature = 3.0;
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_REQUEST);
  EXPECT_STREQ(err.param, "temperature");
  EXPECT_STREQ(err.message, "temperature must be a number between 0.3 and 2.0.");
  p.sampler.present = 0;
  p.voice = 99;
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_INVALID_VOICE);
}

TEST(Runtime, RejectsInvalidCustomVoiceCapacity) {
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  for (int capacity : {-1, 4097}) {
    cfg.max_custom_voices = capacity;
    EXPECT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(e, nullptr);
    EXPECT_STREQ(err.param, "max_custom_voices");
  }
  // An older config ends before the appended capacity field. Its trailing
  // bytes must not be interpreted as a capacity override.
  cfg.struct_size = offsetof(kt2n_engine_config, max_custom_voices);
  EXPECT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
  kt2n_engine_destroy(e);
}

TEST(Runtime, StreamOneRequest) {
  Eng g;
  const uint64_t id = g.submit("Hello there. This is a small test of the native conductor.");
  auto r = g.run(1);
  ASSERT_EQ(r[id].status, KT2N_OK);
  EXPECT_GT(r[id].samples, 24000 / 2);
  EXPECT_EQ(int64_t(r[id].bytes.size()), r[id].samples * 2);
  EXPECT_TRUE(r[id].seq_ok);
  EXPECT_TRUE(r[id].final_seen);
  EXPECT_GT(r[id].audio_events, 1);  // streamed in pieces (hold, then commits)
  EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
}

// L8b 9e948c97 (the adapter's skip-aware WS hold): every AUDIO event carries production's
// control.n_opening_skipped_tokens, final at the FIRST audio event; build_info names the capability.
// The CPU stub LM opens every row with 3 sub-gate silence ids (backend.h opening_quiet_tokens), then
// non-set ids close the opening: 3 skipped (policy_test OpeningSkip* gives the same count). A complete
// file is a DEMO request, which never skips (row skip_armed), so its count stays production's default 0.
TEST(Runtime, AudioEventsCarryOpeningSkipCount) {
  EXPECT_NE(std::strstr(kt2n_build_info(), KT2N_CAP_AUDIO_OPENING_SKIP), nullptr) << kt2n_build_info();
  Eng g;
  const uint64_t a = g.submit("Hello there. This is a small test of the native conductor.");
  const uint64_t b = g.submit("A second request, so two rows share the conductor.", KT2N_OUTPUT_COMPLETE,
                              KT2N_FORMAT_WAV);
  auto r = g.run(2);
  for (uint64_t id : {a, b}) {
    ASSERT_EQ(r[id].status, KT2N_OK);
    ASSERT_FALSE(r[id].opening_skipped.empty());
    EXPECT_EQ(r[id].opening_skipped.front(), id == a ? 3 : 0) << (id == a ? "stream" : "complete (DEMO)");
    for (int32_t v : r[id].opening_skipped) EXPECT_EQ(v, r[id].opening_skipped.front());
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  EXPECT_GT(r[a].opening_skipped.size(), 1u);  // streamed: the count rides on every chunk
}

TEST(Runtime, CompleteFileWav) {
  Eng g;
  const uint64_t id = g.submit("A complete file, please. Two sentences long.", KT2N_OUTPUT_COMPLETE, KT2N_FORMAT_WAV);
  auto r = g.run(1);
  ASSERT_EQ(r[id].status, KT2N_OK);
  ASSERT_GT(r[id].bytes.size(), 44u);
  EXPECT_EQ(r[id].bytes.substr(0, 4), "RIFF");
  EXPECT_EQ(r[id].audio_events, 1);
  EXPECT_EQ(int64_t(r[id].bytes.size()) - 44, r[id].samples * 2);
}

TEST(Runtime, ConcurrencyAboveSeatsQueuesAndFinishesAll) {
  Eng g(4);
  std::vector<uint64_t> ids;
  for (int i = 0; i < 12; ++i) ids.push_back(g.submit("Request number " + std::to_string(i) + " is here, and it talks."));
  auto r = g.run(ids.size(), 60);
  for (uint64_t id : ids) {
    EXPECT_EQ(r[id].status, KT2N_OK) << id;
    EXPECT_GT(r[id].samples, 0) << id;
  }
  kt2n_stats s{};
  s.struct_size = sizeof s;
  kt2n_stats_get(g.e, &s);
  EXPECT_EQ(s.requests_done, 12u);
  EXPECT_GT(s.renders, 12u);
  for (uint64_t id : ids) EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  // The conductor frees released requests on its next loop.
  std::string js(2048, '\0');
  size_t len = 0;
  bool freed = false;
  for (int i = 0; i < 100 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    kt2n_stats_json(g.e, js.data(), js.size(), &len);
    freed = js.find("\"requests_held\":0,") != std::string::npos;
  }
  EXPECT_TRUE(freed) << js;
}

TEST(Runtime, LongTextManySegments) {
  Eng g;
  std::string text;
  for (int i = 0; i < 12; ++i) text += "This sentence number " + std::to_string(i) + " is part of a longer text that keeps going. ";
  const uint64_t id = g.submit(text);
  auto r = g.run(1, 60);
  ASSERT_EQ(r[id].status, KT2N_OK);
  EXPECT_GT(r[id].samples, 24000);
}

TEST(Runtime, CancelGivesTerminalEvent) {
  Eng g;
  const uint64_t id = g.submit("This one will be cancelled before it finishes speaking all of these words, sadly.");
  EXPECT_EQ(kt2n_cancel(g.e, id, KT2N_CANCEL_REQUEST), KT2N_OK);
  auto r = g.run(1);
  EXPECT_EQ(r[id].status, KT2N_ERR_CANCELLED);
  EXPECT_EQ(kt2n_cancel(g.e, 12345, KT2N_CANCEL_REQUEST), KT2N_ERR_INVALID_HANDLE);
}

}  // namespace

namespace {

TEST(Runtime, WsSessionTwoCommittedTurns) {
  Eng g;
  kt2n_session_params sp{};
  sp.struct_size = sizeof sp;
  sp.voice = 1;
  sp.request_id = "ws-1";
  sp.speed = 1.0;
  kt2n_session_id sid = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_session_open(g.e, &sp, &sid, &err), KT2N_OK) << err.message;
  const std::string a = "Hello from the socket. ";
  const std::string b = "And here is a second turn of text.";
  ASSERT_EQ(kt2n_session_text_append(g.e, sid, a.data(), a.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_commit(g.e, sid, &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_text_append(g.e, sid, b.data(), b.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_done(g.e, sid, &err), KT2N_OK);
  int phrases = 0, done = 0, status = -1;
  int64_t samples = 0;
  uint64_t last = 0;
  bool seq_ok = true;
  kt2n_event ev[32];
  const auto t0 = std::chrono::steady_clock::now();
  while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 32, 200, &k);
    for (int i = 0; i < k; ++i) {
      if (ev[i].id != sid) continue;
      if (ev[i].seq != 0 && ev[i].seq != last + 1) seq_ok = false;
      last = ev[i].seq;
      if (ev[i].kind == KT2N_EVENT_AUDIO) samples += ev[i].samples;
      if (ev[i].kind == KT2N_EVENT_PHRASE_DONE) ++phrases;
      if (ev[i].kind == KT2N_EVENT_DONE) {
        done = 1;
        status = ev[i].status;
      }
    }
    kt2n_events_release(g.e, ev, k);
  }
  EXPECT_EQ(status, KT2N_OK);
  EXPECT_GE(phrases, 2);
  EXPECT_GT(samples, 0);
  EXPECT_TRUE(seq_ok);
  EXPECT_EQ(kt2n_release(g.e, sid), KT2N_OK);
}

TEST(Runtime, SessionFloat32AndFormatValidation) {
  Eng g;
  kt2n_session_params p{};
  p.struct_size = sizeof p;
  p.voice = 1;
  p.speed = 1.0;
  p.format = KT2N_FORMAT_WAV;
  kt2n_session_id id = 0;
  kt2n_error err{};
  EXPECT_EQ(kt2n_session_open(g.e, &p, &id, &err), KT2N_ERR_INVALID_RESPONSE_FORMAT);
  p.format = KT2N_FORMAT_F32;
  p.struct_size = sizeof p - 1;
  EXPECT_EQ(kt2n_session_open(g.e, &p, &id, &err), KT2N_ERR_INVALID_ARGUMENT);
  p.struct_size = sizeof p;
  ASSERT_EQ(kt2n_session_open(g.e, &p, &id, &err), KT2N_OK) << err.message;
  const std::string text = "An incremental phrase in float32 format.";
  ASSERT_EQ(kt2n_session_text_append(g.e, id, text.data(), text.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_done(g.e, id, &err), KT2N_OK);
  auto result = g.run(1);
  ASSERT_EQ(result[id].status, KT2N_OK);
  ASSERT_GT(result[id].samples, 0);
  EXPECT_EQ(int64_t(result[id].bytes.size()), result[id].samples * 4);
}

}  // namespace

namespace {

int create_with_levers(const char* json, kt2n_error* err) {
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  cfg.levers_json = json;
  cfg.app = "us";
  kt2n_engine* e = nullptr;
  const int rc = kt2n_engine_create(&cfg, &e, err);
  kt2n_engine_destroy(e);
  return rc;
}

TEST(Runtime, LeversJsonFailsClosedOnAnotherValue) {
  kt2n_error err{};
  EXPECT_EQ(create_with_levers("{\"KITTENTTS_09_ACOUSTIC_LAZY_MIN_BATCH\":\"4\"}", &err), KT2N_ERR_UNSUPPORTED);
  EXPECT_STREQ(err.param, "KITTENTTS_09_ACOUSTIC_LAZY_MIN_BATCH");
  EXPECT_EQ(create_with_levers("{\"KITTENTTS_09_ACOUSTIC_LAZY_MIN_BATCH\":3,\"KITTENTTS_09_ACOUSTIC_LAZY\":\"true\","
                               "\"KITTENTTS_09_DW_RUNWAY_MS\":\"500\",\"UNRELATED_NAME\":\"x\"}",
                               &err),
            KT2N_OK)
      << err.message;
  EXPECT_EQ(create_with_levers("not json", &err), KT2N_ERR_UNSUPPORTED);
}

}  // namespace

namespace {

TEST(Runtime, WsOversizedMessageEndsSessionLikeProduction) {
  Eng g;
  kt2n_session_params sp{};
  sp.struct_size = sizeof sp;
  sp.voice = 1;
  sp.speed = 1.0;
  kt2n_session_id sid = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_session_open(g.e, &sp, &sid, &err), KT2N_OK);
  const std::string big(8193, 'a');
  EXPECT_EQ(kt2n_session_text_append(g.e, sid, big.data(), big.size(), &err), KT2N_ERR_SPEECH_GENERATION_FAILED);
  EXPECT_STREQ(err.code, "speech_generation_failed");
  int status = -1;
  kt2n_event ev[16];
  const auto t0 = std::chrono::steady_clock::now();
  while (status < 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 16, 100, &k);
    for (int i = 0; i < k; ++i)
      if (ev[i].id == sid && ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
    kt2n_events_release(g.e, ev, k);
  }
  EXPECT_EQ(status, KT2N_ERR_SPEECH_GENERATION_FAILED);
}

}  // namespace

namespace {

TEST(Runtime, WsInputAfterDoneEndsSessionInputAlreadyClosed) {
  Eng g;
  kt2n_session_params sp{};
  sp.struct_size = sizeof sp;
  sp.voice = 1;
  sp.speed = 1.0;
  kt2n_session_id sid = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_session_open(g.e, &sp, &sid, &err), KT2N_OK);
  const std::string a = "Some words.";
  ASSERT_EQ(kt2n_session_text_append(g.e, sid, a.data(), a.size(), &err), KT2N_OK);
  ASSERT_EQ(kt2n_session_done(g.e, sid, &err), KT2N_OK);
  EXPECT_EQ(kt2n_session_text_append(g.e, sid, a.data(), a.size(), &err), KT2N_ERR_INPUT_ALREADY_CLOSED);
  EXPECT_STREQ(err.code, "input_already_closed");
  int status = -1;
  kt2n_event ev[16];
  const auto t0 = std::chrono::steady_clock::now();
  while (status < 0 && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 16, 100, &k);
    for (int i = 0; i < k; ++i)
      if (ev[i].id == sid && ev[i].kind == KT2N_EVENT_DONE) status = ev[i].status;
    kt2n_events_release(g.e, ev, k);
  }
  EXPECT_EQ(status, KT2N_ERR_INPUT_ALREADY_CLOSED);
}

}  // namespace

namespace {

// Random mix: stream / complete / WAV, cancels at random times, releases at DONE; every request must end with
// exactly one terminal event and the engine must hold nothing afterwards.
TEST(Runtime, StressRandomCancelsAndOutputs) {
  Eng g(6);
  uint64_t rng = 12345;
  auto next = [&] {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(rng >> 33);
  };
  std::map<uint64_t, int> done;
  std::vector<uint64_t> ids;
  const char* texts[] = {"Short.", "A medium sentence that has a few words in it.",
                         "A longer text. It has two sentences, and it keeps going for a little while longer."};
  for (int i = 0; i < 60; ++i) {
    const int kind = int(next() % 3);
    ids.push_back(g.submit(texts[next() % 3], kind == 2 ? KT2N_OUTPUT_COMPLETE : KT2N_OUTPUT_STREAM,
                           kind == 1 ? KT2N_FORMAT_WAV : KT2N_FORMAT_PCM));
    if (next() % 4 == 0) kt2n_cancel(g.e, ids.back(), KT2N_CANCEL_REQUEST);
  }
  kt2n_event ev[64];
  const auto t0 = std::chrono::steady_clock::now();
  while (done.size() < ids.size() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60)) {
    int32_t k = 0;
    kt2n_poll(g.e, ev, 64, 100, &k);
    for (int i = 0; i < k; ++i) {
      if (ev[i].kind == KT2N_EVENT_DONE) {
        done[ev[i].id] += 1;
        EXPECT_TRUE(ev[i].status == KT2N_OK || ev[i].status == KT2N_ERR_CANCELLED) << ev[i].status;
      }
    }
    kt2n_events_release(g.e, ev, k);
    // Cancel a random live request now and then.
    if (next() % 8 == 0) kt2n_cancel(g.e, ids[next() % ids.size()], KT2N_CANCEL_REQUEST);
  }
  ASSERT_EQ(done.size(), ids.size());
  for (auto& [id, n] : done) EXPECT_EQ(n, 1) << id;
  for (uint64_t id : ids) EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  std::string js(4096, '\0');
  size_t len = 0;
  bool freed = false;
  for (int i = 0; i < 200 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    kt2n_stats_json(g.e, js.data(), js.size(), &len);
    freed = js.find("\"requests_held\":0,") != std::string::npos && js.find("\"active_requests\":0,") != std::string::npos;
  }
  EXPECT_TRUE(freed) << js;
}

}  // namespace

namespace {

// Render failures (stub: every 7th call fails): every request still ends with exactly one terminal event
// (OK or speech_generation_failed) and nothing stays held.
TEST(Runtime, RenderFailuresNeverStick) {
  setenv("KT2N_TEST_STUB_RENDER_FAIL_EVERY", "7", 1);
  Eng g(4);
  unsetenv("KT2N_TEST_STUB_RENDER_FAIL_EVERY");
  std::vector<uint64_t> ids;
  for (int i = 0; i < 20; ++i) ids.push_back(g.submit("Render failures must never leave a request hanging, ever."));
  auto r = g.run(ids.size(), 60);
  int failed = 0;
  for (uint64_t id : ids) {
    EXPECT_TRUE(r[id].status == KT2N_OK || r[id].status == KT2N_ERR_SPEECH_GENERATION_FAILED) << r[id].status;
    failed += r[id].status == KT2N_ERR_SPEECH_GENERATION_FAILED;
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  EXPECT_GT(failed, 0);
  std::string js(4096, '\0');
  size_t len = 0;
  bool freed = false;
  for (int i = 0; i < 300 && !freed; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    kt2n_stats_json(g.e, js.data(), js.size(), &len);
    freed = js.find("\"requests_held\":0,") != std::string::npos;
  }
  EXPECT_TRUE(freed) << js;
}

// Native admission back-pressure: a renderer that cannot open more rows (pool pages / slots) makes jobs wait in
// admission; every request still completes, none fails, and the renderer never held more rows than it allowed.
TEST(Runtime, RenderBackpressureWaitsNeverFails) {
  setenv("KT2N_TEST_STUB_RENDER_MAX_ROWS", "3", 1);
  Eng g(8);
  unsetenv("KT2N_TEST_STUB_RENDER_MAX_ROWS");
  std::vector<uint64_t> ids;
  for (int i = 0; i < 12; ++i) ids.push_back(g.submit("Back-pressure waits in admission, and it never fails a request."));
  auto r = g.run(ids.size(), 60);
  for (uint64_t id : ids) {
    EXPECT_EQ(r[id].status, KT2N_OK);
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  std::string js(8192, '\0');
  size_t len = 0;
  ASSERT_EQ(kt2n_stats_json(g.e, js.data(), js.size(), &len), KT2N_OK);
  js.resize(len ? len - 1 : 0);
  EXPECT_NE(js.find("\"open_rows_max\":3"), std::string::npos) << js;
  EXPECT_EQ(js.find("\"render_backpressure\":0,"), std::string::npos) << js;
}

// More seats than rows: admitted requests beyond the rows wait in admission
// (they are never opened and closed again at the renderer) and all of them finish.
TEST(Runtime, SeatsAboveRowsWaitInAdmission) {
  setenv("KT2N_ENGINE_ROWS", "3", 1);
  Eng g(64);
  unsetenv("KT2N_ENGINE_ROWS");
  std::vector<uint64_t> ids;
  for (int i = 0; i < 40; ++i) ids.push_back(g.submit("Seat " + std::to_string(i) + " waits for a row, then talks."));
  auto r = g.run(ids.size(), 60);
  for (uint64_t id : ids) {
    EXPECT_EQ(r[id].status, KT2N_OK) << id;
    EXPECT_GT(r[id].samples, 0) << id;
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  std::string js(8192, '\0');
  size_t len = 0;
  ASSERT_EQ(kt2n_stats_json(g.e, js.data(), js.size(), &len), KT2N_OK);
  js.resize(len ? len - 1 : 0);
  EXPECT_NE(js.find("\"max_streams\":64,\"engine_rows\":3,\"engine_rows_why\":\"KT2N_ENGINE_ROWS\""),
            std::string::npos) << js;
  // LM rows <= the cap. Render rows also hold rows whose LM finished until their final render (the real
  // renderer has cap + 4 slots for that), so they may exceed the cap by a few, never by the queue.
  auto num = [&js](const std::string& key) {
    const size_t at = js.find("\"" + key + "\":");
    return at == std::string::npos ? -1 : std::atoi(js.c_str() + at + key.size() + 3);
  };
  EXPECT_GE(num("turn_max_rows"), 1) << js;
  EXPECT_LE(num("turn_max_rows"), 3) << js;
  EXPECT_LE(num("open_rows_max"), 3 + 4) << js;
  EXPECT_NE(js.find("\"render_backpressure\":0,"), std::string::npos) << js;
  // Without the knob, CPU stubs (no device memory) keep the 60-row limit; the release's seats are unchanged.
  Eng big(64);
  Eng ten(10);
  for (Eng* x : {&big, &ten}) {
    std::string j2(8192, '\0');
    ASSERT_EQ(kt2n_stats_json(x->e, j2.data(), j2.size(), &len), KT2N_OK);
    j2.resize(len ? len - 1 : 0);
    EXPECT_NE(j2.find(x == &big ? "\"engine_rows\":60,\"engine_rows_why\":\"row limit\""
                                : "\"engine_rows\":10,\"engine_rows_why\":\"seats\""),
              std::string::npos) << j2;
  }
}

// Capacity mode admission order, end to end on the CPU stubs: closed loop as the capacity graph (every short
// request that finishes is replaced by a new one, 40 stay queued while one long request plays); everything
// finishes with the order off and on, and the stats say which ran.
TEST(Runtime, CapacityOrderAdmitsContinuingRequestsFirst) {
  std::string longtext;
  for (int i = 0; i < 6; ++i) longtext += "This sentence number " + std::to_string(i) + " is part of a longer text that keeps going. ";
  int before[2] = {-1, -1};  // short requests finished before the long one, order off / on
  for (int order = 0; order < 2; ++order) {
    setenv("KT2N_ENGINE_ROWS", "2", 1);
    setenv("KT2N_CAPACITY_ORDER", order ? "1" : "0", 1);
    Eng g(64);
    unsetenv("KT2N_ENGINE_ROWS");
    unsetenv("KT2N_CAPACITY_ORDER");
    const uint64_t lid = g.submit(longtext);
    int submitted = 0, outstanding = 0, shorts_done = 0;
    auto add_short = [&] {
      g.submit("Short request " + std::to_string(submitted++) + " here.");
      ++outstanding;
    };
    for (int i = 0; i < 40; ++i) add_short();
    bool long_done = false;
    kt2n_event ev[32];
    const auto t0 = std::chrono::steady_clock::now();
    while ((!long_done || outstanding > 0) && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(120)) {
      int32_t k = 0;
      kt2n_poll(g.e, ev, 32, 200, &k);
      std::vector<uint64_t> finished;
      for (int i = 0; i < k; ++i)
        if (ev[i].kind == KT2N_EVENT_DONE) {
          EXPECT_EQ(ev[i].status, KT2N_OK);
          finished.push_back(ev[i].id);
        }
      kt2n_events_release(g.e, ev, k);
      for (uint64_t id : finished) {
        EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
        if (id == lid) {
          long_done = true;
          before[order] = shorts_done;
          continue;
        }
        ++shorts_done;
        --outstanding;
        if (!long_done && submitted < 400) add_short();
      }
    }
    EXPECT_TRUE(long_done);
    EXPECT_EQ(outstanding, 0);
    std::string js(8192, '\0');
    size_t len = 0;
    ASSERT_EQ(kt2n_stats_json(g.e, js.data(), js.size(), &len), KT2N_OK);
    EXPECT_NE(js.find(order ? "\"capacity_order\":true" : "\"capacity_order\":false"), std::string::npos);
  }
  // Printed, not asserted: on the CPU stubs the long request's next segment is often not ready when a row frees
  // (production admits it only after its previous segment is acknowledged), so the order rarely gets to act and
  // the counts depend on box load (16 / 16 and 54 / 16 in runs on the box). The ordering itself is checked
  // exactly in policy_test (Admission.CapacityModeOffersContinuingRequestsFirst).
  std::printf("[capacity order] short requests finished before the long one: %d (production round-robin) vs %d "
              "(continuing first)\n", before[0], before[1]);
}

// RC1 release blocker (l0-quality 22:12, W.Q.qset:C02): a streamed segment of 466+ characters (tagged budget
// >= 2,047 tokens) was refused by L7's render-service open_row ("budget exceeds the noise table") after segment 0,
// on every voice. Needs a GPU (L7's RenderService with its stand-in stages: the real open_row, no weights, ~3 GB):
// run with KT2N_TEST_GPU=1; skipped otherwise (the CPU stub renderer has no noise table).
TEST(Runtime, GpuLongStreamedSegmentOpensItsRenderRow) {
  if (!std::getenv("KT2N_TEST_GPU")) GTEST_SKIP() << "set KT2N_TEST_GPU=1 on a GPU slot";
  const std::string text =
      "Good evening, and welcome to the last session of this year's community lecture series. Tonight we are talking "
      "about bridges: why they stand up, why a few of them fall down, and what engineers have learned from both. We "
      "will start with the oldest stone arches in the region, some of which have carried traffic for more than four "
      "hundred years without a single repair. Then we will look at the iron bridges of the railway age, which were "
      "lighter and cheaper, and occasionally far less forgiving. Finally, we will spend some time on the suspension "
      "bridge across the estuary, which many of you drive over every morning. I promise you that it is perfectly "
      "safe, and by the end of the evening I hope you will understand exactly why. Please keep your questions until "
      "the end; there will be plenty of time, and the coffee will still be warm.";
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  cfg.flags = KT2N_CFG_TIMELINE_ALLOWED;
  cfg.max_streams = 2;
  cfg.s3gen_file = "l7-standin";
  setenv("KT2N_L7_POOL_PAGES", "64", 1);
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  ASSERT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
  unsetenv("KT2N_L7_POOL_PAGES");
  std::map<int, int> status;
  for (int output : {int(KT2N_OUTPUT_STREAM), int(KT2N_OUTPUT_COMPLETE)}) {
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    p.text = text.data();
    p.text_len = text.size();
    p.voice = 1;
    p.request_id = "c02";
    p.seed = 42;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = output;
    kt2n_request_id id = 0;
    ASSERT_EQ(kt2n_submit(e, &p, &id, &err), KT2N_OK) << err.message;
    kt2n_event ev[32];
    bool done = false;
    const auto t0 = std::chrono::steady_clock::now();
    while (!done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60)) {
      int32_t k = 0;
      kt2n_poll(e, ev, 32, 200, &k);
      for (int i = 0; i < k; ++i)
        if (ev[i].kind == KT2N_EVENT_DONE && ev[i].id == id) {
          done = true;
          status[output] = ev[i].status;
          EXPECT_EQ(ev[i].status, KT2N_OK) << (output == KT2N_OUTPUT_STREAM ? "stream: " : "complete: ")
                                           << (ev[i].error ? ev[i].error->message : "");
        }
      kt2n_events_release(e, ev, k);
    }
    EXPECT_TRUE(done);
    kt2n_release(e, id);
  }
  kt2n_engine_destroy(e);
}

}  // namespace

namespace {

// The adapter's boundary (L8b): F32 frames (held, not levelled) and control_json policy fields.
TEST(Runtime, F32BoundaryAndControlJson) {
  Eng g;
  kt2n_request_params p{};
  p.struct_size = sizeof p;
  const std::string text = "A phrase for the adapter boundary.";
  p.text = text.data();
  p.text_len = text.size();
  p.voice = 1;
  p.speed = 1.0;
  p.format = KT2N_FORMAT_F32;
  p.control_json = "{\"startup_hold_seconds\":0,\"phrase\":true,\"emotion\":false,\"initial_audio_buffer_seconds\":0}";
  kt2n_request_id id = 0;
  kt2n_error err{};
  ASSERT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_OK) << err.message;
  auto r = g.run(1);
  ASSERT_EQ(r[id].status, KT2N_OK);
  EXPECT_EQ(int64_t(r[id].bytes.size()), r[id].samples * 4);
  p.control_json = "{\"delivery_override\":{\"x\":1}}";
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_UNSUPPORTED);
  EXPECT_STREQ(err.param, "delivery_override");
  p.control_json = "{\"demo_check\":true}";
  EXPECT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_ERR_UNSUPPORTED);
}

// Evicted custom voices give their LM store space and render table row back once their last request is freed
// (release_voices on the conductor; L3 release_voice, L7 remove_voice). CPU stubs: the stats count the releases.
TEST(Runtime, EvictedSavedVoicesAreReleased) {
  using namespace kt2n::voice;
  auto synth = [](const std::string& id, int tp, int p, unsigned seed, const char* kind) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd;
    std::vector<int32_t> prefix(static_cast<size_t>(p)), tok(static_cast<size_t>(tp));
    for (auto& x : prefix) x = int32_t(rng() % 158241);
    for (auto& x : tok) x = int32_t(rng() % 6561);
    std::vector<float> raw(512), proj(2048), feat(static_cast<size_t>(tp) * 160), emb(192);
    for (auto* v : {&raw, &proj, &feat, &emb})
      for (auto& x : *v) x = nd(rng);
    double nn = 0;
    for (float x : raw) nn += double(x) * x;
    for (float& x : raw) x = float(x / std::sqrt(nn));
    return make_clone_record(id, kind, prefix, raw, proj, tok, feat, emb, "sig-" + id);
  };
  auto blob_of = [](std::vector<VoiceRecord> v) {
    VoicesBlob b;
    b.top = default_top_meta();
    b.voices = std::move(v);
    return write_voices_blob(b);
  };
  std::vector<VoiceRecord> cat = {synth("v0", 7, 30, 10, "builtin"), synth("v1", 7, 30, 11, "builtin")};
  const auto cat_bytes = blob_of(cat);
  const std::string path = testing::TempDir() + "kt2n_release_test_voices.kt2n";
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(cat_bytes.data()), std::streamsize(cat_bytes.size()));
  }
  VoiceRecord r = synth("voice_s", 9, 40, 20, "saved");
  r.raw_speaker = cat[1].raw_speaker;
  r.raw_speaker[0] += 1e-3f;
  CloneIdentityInputs in{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'), "v1"};
  const auto saved = blob_of({make_clone_record("voice_s", "saved", r.lm_prefix, r.raw_speaker, r.projected_speaker,
                                                r.prompt_token, r.prompt_feat, r.embedding, "sig",
                                                production_identity(in))});
  kt2n_engine_config cfg{};
  cfg.struct_size = sizeof cfg;
  cfg.abi_version = KT2N_ABI_VERSION;
  cfg.max_streams = 4;
  cfg.voices_file = path.c_str();
  Eng g(0);  // the helper's engine is unused; this test needs a catalogue
  kt2n_engine* e = nullptr;
  kt2n_error err{};
  ASSERT_EQ(kt2n_engine_create(&cfg, &e, &err), KT2N_OK) << err.message;
  std::swap(g.e, e);
  kt2n_engine_destroy(e);
  const int cycles = 5;
  for (int c = 0; c < cycles; ++c) {
    kt2n_voice_id v = 0;
    ASSERT_EQ(kt2n_voice_load(g.e, saved.data(), saved.size(), &v, &err), KT2N_OK) << err.message << " " << err.code;
    kt2n_request_params p{};
    p.struct_size = sizeof p;
    p.flags = KT2N_REQ_SEED;
    const std::string text = "A short line on a saved voice.";
    p.text = text.data();
    p.text_len = text.size();
    p.voice = v;
    p.request_id = "t";
    p.seed = 42;
    p.speed = 1.0;
    p.format = KT2N_FORMAT_PCM;
    p.output = KT2N_OUTPUT_STREAM;
    kt2n_request_id id = 0;
    ASSERT_EQ(kt2n_submit(g.e, &p, &id, &err), KT2N_OK) << err.message;
    auto res = g.run(1);
    EXPECT_EQ(res[id].status, KT2N_OK);
    EXPECT_EQ(kt2n_voice_evict(g.e, v), KT2N_OK);
    EXPECT_EQ(kt2n_release(g.e, id), KT2N_OK);
  }
  std::string stats;
  for (int i = 0; i < 100; ++i) {  // the conductor frees released requests, then sweeps evicted voices
    std::vector<char> b(1 << 20);
    size_t len = 0;
    kt2n_stats_json(g.e, b.data(), b.size(), &len);
    stats.assign(b.data(), len);
    if (stats.find("\"render_voices_released\":" + std::to_string(cycles)) != std::string::npos) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_NE(stats.find("\"lm_voices_released\":" + std::to_string(cycles)), std::string::npos) << stats.substr(0, 400);
  EXPECT_NE(stats.find("\"render_voices_released\":" + std::to_string(cycles)), std::string::npos);
  EXPECT_NE(stats.find("\"voice_store_full\":0"), std::string::npos);
  std::remove(path.c_str());
}

}  // namespace
