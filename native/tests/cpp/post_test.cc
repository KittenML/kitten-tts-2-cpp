// Unit tests for the L10 post-processing ports (kt2n_post). Bit-level parity
// with the production Python is checked separately by tests/post/equiv_harness.py
// on recorded streams; these pin the properties and fixed values.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>

#include "kt2n/post/atempo.h"
#include "kt2n/post/post.h"

namespace post = kt2n::post;

namespace {

post::Pcm tone(int n, float amp, double hz = 180.0) {
  post::Pcm x(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = amp * static_cast<float>(std::sin(2.0 * M_PI * hz * i / 24000.0));
  return x;
}

post::Pcm noise(int n, float amp, unsigned seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  post::Pcm x(static_cast<size_t>(n));
  for (auto& v : x) v = amp * d(g);
  return x;
}

post::Pcm cat(std::initializer_list<post::Pcm> parts) {
  post::Pcm o;
  for (const auto& p : parts) o.insert(o.end(), p.begin(), p.end());
  return o;
}

template <class F>
post::Pcm run_chunked(const post::Pcm& x, const std::vector<int>& sizes, F&& feed) {
  post::Pcm out;
  size_t at = 0;
  for (int s : sizes) {
    auto r = feed(post::Samples(x.data() + at, static_cast<size_t>(s)));
    out.insert(out.end(), r.begin(), r.end());
    at += static_cast<size_t>(s);
  }
  return out;
}

}  // namespace

TEST(PostHelpers, PairwiseSumMatchesNaiveForSmall) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  EXPECT_EQ(post::pairwise_sum(a.data(), 5), 15.0);
  std::vector<double> b(240, 0.5);
  EXPECT_EQ(post::pairwise_sum(b.data(), 240), 120.0);
}

TEST(PostHelpers, LinspaceEndpoints) {
  auto up = post::linspace_f32(0.0, 1.0, 120);
  ASSERT_EQ(up.size(), 120u);
  EXPECT_EQ(up.front(), 0.0f);
  EXPECT_EQ(up.back(), 1.0f);
  auto one = post::linspace_f32(0.0, 1.0, 1);
  EXPECT_EQ(one[0], 0.0f);
  auto down1 = post::linspace_f32(1.0, 0.0, 1);
  EXPECT_EQ(down1[0], 1.0f);
}

TEST(PostHelpers, PyRoundIsHalfEven) {
  EXPECT_EQ(post::py_round(2.5), 2.0);
  EXPECT_EQ(post::py_round(3.5), 4.0);
  EXPECT_EQ(post::py_round(-0.5), -0.0);
}

TEST(Onset, LivePolicyGeometry) {
  post::LeadingSilenceTrimmer t;
  EXPECT_EQ(t.frame(), 240);
  EXPECT_EQ(t.keep_frames(), 0);
  EXPECT_EQ(t.lookback_frames(), 4);  // 45 ms -> 1080 samples // 240
  EXPECT_EQ(t.max_trim_samples(), 48000);
  EXPECT_EQ(t.fade_samples(), 120);
}

TEST(Onset, RejectsBadPolicy) {
  post::OnsetPolicy p;
  p.soft_gate_dbfs = -40.0;  // above the hard gate
  EXPECT_THROW(post::LeadingSilenceTrimmer{p}, std::invalid_argument);
  p = post::OnsetPolicy{};
  p.max_trim_ms = 5.0;  // less than a frame
  EXPECT_THROW(post::LeadingSilenceTrimmer{p}, std::invalid_argument);
}

TEST(Onset, TrimsSilenceAndIsChunkInvariant) {
  const post::Pcm x = cat({post::Pcm(9600, 0.0f), noise(2400, 1e-4f, 1), tone(12000, 0.3f)});
  post::LeadingSilenceTrimmer whole;
  post::Pcm a = whole.feed(x);
  auto f = whole.flush();
  a.insert(a.end(), f.begin(), f.end());
  post::LeadingSilenceTrimmer chunked;
  post::Pcm b = run_chunked(x, {1000, 3360, 960, 4800, 7000, 6880}, [&](post::Samples s) { return chunked.feed(s); });
  auto f2 = chunked.flush();
  b.insert(b.end(), f2.begin(), f2.end());
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * 4));
  EXPECT_GT(whole.dropped_samples(), 9000);
  EXPECT_LE(whole.dropped_samples(), 12000);
  EXPECT_FALSE(whole.armed());
  EXPECT_TRUE(whole.detected());
}

TEST(Onset, CapReleasesQuietOpening) {
  const post::Pcm x = post::Pcm(72000, 0.0f);  // 3 s of digital silence
  post::LeadingSilenceTrimmer t;
  post::Pcm out = t.feed(x);
  EXPECT_EQ(t.dropped_samples(), 48000);  // the 2 s cap
  EXPECT_EQ(out.size(), 24000u);
  EXPECT_FALSE(t.armed());
}

TEST(Onset, WithholderCutsKeptTokenAudio) {
  post::KeptTokenWithholder w{post::LeadingSilenceTrimmer{}};
  w.set_kept_tokens(1);
  const post::Pcm x = tone(4800, 0.3f);
  post::Pcm out = w.feed(x);
  EXPECT_EQ(w.withheld_samples(), 480);  // 960 - 20 ms floor
  EXPECT_EQ(out.size(), 4800u - 480u);
}

TEST(Seam, TrimsBothEdgesWithFades) {
  const post::Pcm x = cat({post::Pcm(4800, 0.0f), tone(4800, 0.3f), post::Pcm(4800, 0.0f)});
  post::SeamEdgeTrimmer s(true);
  post::Pcm out = s.trim(x, true);
  EXPECT_EQ(out.size(), 480u + 4800u + 480u);
  EXPECT_EQ(out.front(), 0.0f);
  EXPECT_EQ(s.head_trimmed_samples(), 4800 - 480);
  EXPECT_EQ(s.tail_trimmed_samples(), 4800 - 480);
}

TEST(Seam, NoLoudFrameReturnsUntouched) {
  const post::Pcm x = noise(5000, 1e-4f, 3);
  post::SeamEdgeTrimmer s(true);
  post::Pcm out = s.trim(x, true);
  ASSERT_EQ(out.size(), x.size());
  EXPECT_EQ(0, std::memcmp(out.data(), x.data(), x.size() * 4));
}

TEST(Seam, ScopeRules) {
  post::SeamContext c;
  c.scope = post::SeamScope::kTagged;
  c.demo_chunked = true;
  c.dc_mode = 1;
  c.lane_sealed = true;
  c.lane_jobs = 1;
  EXPECT_FALSE(post::seam_trim_applies(c, 0));  // a request's only chunk
  c.lane_jobs = 3;
  EXPECT_TRUE(post::seam_trim_applies(c, 0));
  EXPECT_FALSE(post::seam_trim_head(c, 0));
  EXPECT_TRUE(post::seam_trim_head(c, 1));
  c.demo_chunked = false;
  EXPECT_FALSE(post::seam_trim_applies(c, 1));
  c.demo_request = true;
  c.lane_jobs = 2;
  EXPECT_TRUE(post::seam_trim_head(c, 0));
}

TEST(Gap, LiveGapIs3840Samples) { EXPECT_EQ(post::chunk_gap_samples(), 3840); }

TEST(Hold, LiveAloneFirstChunk) {
  post::StartupHold h;
  ASSERT_TRUE(h.has_target_fn());
  post::HoldObservation o;
  o.listeners = 1;
  o.trimmed_ms_so_far = 20.0;
  EXPECT_EQ(h.target(o, 1), 1440);  // min(0.1, 0.06) s
  EXPECT_EQ(h.target(o, 2), 2400);  // alone 0.1 s
  o.trimmed_ms_so_far = 340.0;
  EXPECT_EQ(h.target(o, 1), 2400);  // trimmed > 100 ms: the 0.1 s floor
  o.listeners = 10;
  EXPECT_EQ(h.target(o, 1), static_cast<int64_t>(post::py_round(24000 * (0.06 + 0.006 * 9))));
  o.opening_skipped_tokens = 8;
  EXPECT_EQ(h.target(o, 1), static_cast<int64_t>(post::py_round(24000 * (0.06 + 0.0125 * 9))));
  o.steady_joiner = true;
  EXPECT_EQ(h.target(o, 1), 5760);  // D_JOIN_HOLD_MS 240
}

TEST(Hold, ReleasesOnceThenPassesThrough) {
  post::StartupHold h;
  post::HoldObservation o;
  o.listeners = 1;
  o.trimmed_ms_so_far = 0.0;
  EXPECT_EQ(h.feed(post::Pcm(1000, 0.1f), o).size(), 0u);
  EXPECT_EQ(h.feed(post::Pcm(1000, 0.1f), o).size(), 0u);  // target rose to 2400
  EXPECT_EQ(h.feed(post::Pcm(1000, 0.1f), o).size(), 3000u);
  EXPECT_EQ(h.feed(post::Pcm(7, 0.1f), o).size(), 7u);
  EXPECT_TRUE(h.finish().empty());
}

TEST(Hold, PinnedZeroIsPassthrough) {
  post::HoldPolicy p;
  p.pinned = 0.0;
  post::StartupHold h(p);
  EXPECT_FALSE(h.has_target_fn());
  post::HoldObservation o;
  EXPECT_EQ(h.feed(post::Pcm(5, 0.1f), o).size(), 5u);
}

TEST(Pcm, NormalizeAndPcm16) {
  post::Pcm out;
  const post::Pcm x = {0.0f, 1.0f, -1.0f, 1.5f, -2.0f, 0.5f};
  ASSERT_EQ(post::normalize_audio(x, out), post::NormalizeStatus::kOk);
  std::vector<int16_t> i16;
  post::pcm16_from_float(out, i16);
  EXPECT_EQ(i16, (std::vector<int16_t>{0, 32767, -32767, 32767, -32767, 16383}));
  post::Pcm bad = {0.0f, NAN};
  EXPECT_EQ(post::normalize_audio(bad, out), post::NormalizeStatus::kNonFinite);
  EXPECT_EQ(post::normalize_audio(post::Pcm{}, out), post::NormalizeStatus::kEmpty);
}

TEST(Wav, Headers) {
  const std::string s = post::streaming_wav_header();
  ASSERT_EQ(s.size(), 44u);
  EXPECT_EQ(s.substr(0, 4), "RIFF");
  EXPECT_EQ(static_cast<unsigned char>(s[4]), 0xFF);
  const std::string c = post::complete_wav_header(4800);
  uint32_t riff = 0, data = 0, rate = 0;
  std::memcpy(&riff, c.data() + 4, 4);
  std::memcpy(&rate, c.data() + 24, 4);
  std::memcpy(&data, c.data() + 40, 4);
  EXPECT_EQ(riff, 4800u + 36u);
  EXPECT_EQ(rate, 24000u);
  EXPECT_EQ(data, 4800u);
  EXPECT_EQ(post::finalized_wav_header(s, 44 + 4800), c);
  EXPECT_EQ(post::finalized_wav_header("RIFX", 100), "RIFX");
  EXPECT_THROW(post::complete_wav_header(0xFFFFFFFFull), std::invalid_argument);
}

TEST(Peak, DemoFormula) {
  const post::Pcm x = {0.5f, -0.25f, 0.125f};
  EXPECT_EQ(post::demo_peak(x), 0.5f);
  std::vector<int16_t> i16;
  post::demo_int16_with_peak(x, 0.5f, i16);
  EXPECT_EQ(i16, (std::vector<int16_t>{32767, -16383, 8191}));
  post::demo_int16_with_peak(x, 0.0f, i16);
  EXPECT_EQ(i16, (std::vector<int16_t>{0, 0, 0}));
}

TEST(Peak, GainOnlyFallsAndNeverPassesCeiling) {
  post::PeakNorm pn(false, 6.0, 0.999, 5.0);
  const post::Pcm quiet = tone(2400, 0.2f);
  const post::Pcm loud = tone(4800, 0.9f);
  post::Pcm a = pn.process(quiet);
  const double g0 = pn.gain();
  post::Pcm b = pn.process(loud);
  EXPECT_LE(pn.gain(), g0);
  for (float v : b) EXPECT_LE(std::fabs(v), 0.999f + 1e-7f);
  EXPECT_GE(pn.falls(), 1);
}

TEST(Peak, ParsesGainTable) {
  const char* js = R"({"schema":"kittentts-peak-norm-gains.v1","default_gain_db":0.087,
    "voices":{"a":{"gain_db":3.5},"b":{"gain_db":0.0}},"aliases":{"Alpha":"a"}})";
  auto t = post::PeakGainTable::parse(js);
  std::string src;
  EXPECT_EQ(post::voice_gain_db(t, std::string(" Alpha "), post::PeakStreamMode::kVoice, "median", &src), 3.5);
  EXPECT_EQ(src, "voice:a");
  EXPECT_EQ(post::voice_gain_db(t, std::nullopt), 0.087);
  EXPECT_EQ(post::voice_gain_db(t, std::string("a"), post::PeakStreamMode::kUniform), 0.087);
  auto pn = post::PeakNorm::for_request(t, std::string("a"), false, "mp3");
  ASSERT_TRUE(pn.has_value());
  EXPECT_EQ(pn->ceiling(), 0.891);
  EXPECT_FALSE(post::PeakNorm::for_request(t, std::nullopt, false, "pcm", post::PeakStreamMode::kOff).has_value());
}

TEST(ATempo, StageTempos) {
  EXPECT_EQ(post::stage_tempos(1.5).size(), 1u);
  EXPECT_EQ(post::stage_tempos(0.25).size(), 2u);
  EXPECT_EQ(post::stage_tempos(4.0).size(), 2u);
  EXPECT_DOUBLE_EQ(post::stage_tempos(4.0)[0], 2.0);
}

TEST(ATempo, ChunkInvariantAndExactLength) {
  const post::Pcm x = cat({tone(24000, 0.3f, 150.0), noise(12000, 0.05f, 9)});
  for (double speed : {0.25, 0.5, 0.75, 1.25, 2.0, 4.0}) {
    post::SpeedStretcher a(speed);
    post::Pcm whole = a.feed(x);
    auto t = a.flush();
    whole.insert(whole.end(), t.begin(), t.end());
    EXPECT_EQ(static_cast<int64_t>(whole.size()),
              static_cast<int64_t>(post::py_round(static_cast<double>(x.size()) / speed)));
    post::SpeedStretcher b(speed);
    post::Pcm ch = run_chunked(x, {3360, 960, 1, 4799, 24000, 2880}, [&](post::Samples s) { return b.feed(s); });
    auto t2 = b.flush();
    ch.insert(ch.end(), t2.begin(), t2.end());
    ASSERT_EQ(whole.size(), ch.size()) << speed;
    EXPECT_EQ(0, std::memcmp(whole.data(), ch.data(), whole.size() * 4)) << speed;
  }
}

#include "kt2n/post/response.h"

TEST(Response, TwoSegmentsTrimSeamsAndGap) {
  // Plain demo-chunked request (Q_PLAIN_DEMO_CHUNKS=1, DC_SEAM_TRIM=tagged), 2 segments.
  post::SeamContext ctx;
  ctx.demo_chunked = true;
  ctx.dc_mode = 1;
  ctx.lane_sealed = true;
  ctx.lane_jobs = 2;
  post::ResponsePost rp;
  rp.set_kept_opening_tokens(1);
  const post::Pcm seg0 = cat({post::Pcm(4800, 0.0f), tone(9600, 0.3f), post::Pcm(4800, 0.0f)});
  const post::Pcm seg1 = cat({post::Pcm(2400, 0.0f), tone(4800, 0.3f), post::Pcm(2400, 0.0f)});
  size_t total = 0;
  rp.begin_segment(0, ctx);
  total += run_chunked(seg0, {3360, 4800, 11040}, [&](post::Samples s) { return rp.push(s); }).size();
  for (const auto& p : rp.end_segment(true)) total += p.size();
  rp.begin_segment(1, ctx);
  total += run_chunked(seg1, {9600}, [&](post::Samples s) { return rp.push(s); }).size();
  for (const auto& p : rp.end_segment(false)) total += p.size();
  total += rp.finish().size();
  // seg0: onset trim removes the 4800 quiet head (+ 480 withheld) -> tone + 20 ms tail;
  // gap 3840; seg1: 20 ms head + tone + 20 ms tail.
  EXPECT_EQ(total, (9600u + 480u) + 3840u + (480u + 4800u + 480u));
  EXPECT_TRUE(rp.trimmed_ms().has_value());
}
