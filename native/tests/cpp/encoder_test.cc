// In-process encoders (kt2n_encode): container magic, a whole stream, and an
// abandoned stream (client hang-up: destroyed without finish). Byte identity
// with the production CLI is checked by tools/post/kt2n_encode_bench.cc.
#include <gtest/gtest.h>

#include <cmath>

#include "kt2n/post/encoder.h"

namespace post = kt2n::post;

namespace {
post::Pcm tone(int n) {
  post::Pcm x(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = 0.3f * static_cast<float>(std::sin(2.0 * M_PI * 220.0 * i / 24000.0));
  return x;
}
}  // namespace

TEST(Encoder, EachFormatStreamsAndFinishes) {
  const post::Pcm x = tone(48000);
  for (const char* fmt : {"mp3", "aac", "opus", "flac"}) {
    std::string err;
    auto enc = post::StreamEncoder::create(fmt, &err);
    ASSERT_TRUE(enc) << fmt << ": " << err;
    std::string out = enc->take_header();
    const std::string f(fmt);
    if (f == "mp3") {
      EXPECT_EQ(out.substr(0, 3), "ID3");
    } else if (f == "opus") {
      EXPECT_EQ(out.substr(0, 4), "OggS");
    } else if (f == "flac") {
      EXPECT_EQ(out.substr(0, 4), "fLaC");
    }
    for (size_t at = 0; at < x.size(); at += 3360) {
      const size_t n = std::min<size_t>(3360, x.size() - at);
      ASSERT_TRUE(enc->push(post::Samples(x.data() + at, n), out, &err)) << err;
    }
    ASSERT_TRUE(enc->finish(out, &err)) << err;
    EXPECT_GT(out.size(), 2000u) << fmt;  // 2 s of a tone: a real stream, not just a header
    if (std::string(fmt) == "aac") {
      ASSERT_GE(out.size(), 2u);
      EXPECT_EQ(static_cast<unsigned char>(out[0]), 0xFF);  // ADTS sync word
      EXPECT_EQ(static_cast<unsigned char>(out[1]) & 0xF0, 0xF0);
    }
  }
}

TEST(Encoder, AbandonedStreamIsSafe) {
  const post::Pcm x = tone(5000);
  for (const char* fmt : {"mp3", "aac", "opus", "flac"}) {
    std::string err, out;
    auto enc = post::StreamEncoder::create(fmt, &err);
    ASSERT_TRUE(enc) << err;
    ASSERT_TRUE(enc->push(x, out, &err));
    enc.reset();  // no finish(): freed without a trailer
  }
}

TEST(Encoder, CliCommandIsProductions) {
  const auto cmd = post::ffmpeg_stream_command("mp3", 32);
  std::string joined;
  for (const auto& a : cmd) joined += a + " ";
  EXPECT_EQ(joined,
            "ffmpeg -hide_banner -loglevel error -probesize 32 -f f32le -ar 24000 -ac 1 -i pipe:0 "
            "-codec:a libmp3lame -b:a 128k -write_xing 0 -flush_packets 1 -f mp3 pipe:1 ");
}
