// L11: in-process upload decode (see decode.h). Non-seekable memory input (like the CLI's pipe:0), first-best audio
// stream, one decoder thread, then one SwrContext with default options to mono float 24 kHz (what the CLI's
// auto-inserted aresample does for `-ac 1 -ar 24000 -f f32le`), cut at max_seconds like `-t`.
#include "decode.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace kt2n::voice {
namespace {

struct Mem {
  const uint8_t* p;
  size_t len;
  size_t pos;
};

int read_mem(void* opaque, uint8_t* buf, int size) {
  auto* m = static_cast<Mem*>(opaque);
  size_t left = m->len - m->pos;
  if (left == 0) return AVERROR_EOF;
  size_t n = std::min(left, static_cast<size_t>(size));
  std::memcpy(buf, m->p + m->pos, n);
  m->pos += n;
  return static_cast<int>(n);
}

std::string averr(int e) {
  char b[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(e, b, sizeof b);
  return b;
}

}  // namespace

bool decode_audio(const uint8_t* data, size_t len, double max_seconds, std::vector<float>& out, std::string* err,
                  DecodeStats* stats) {
  out.clear();
  av_log_set_level(AV_LOG_ERROR);  // the CLI runs with -loglevel error
  const int kRate = 24000;
  const size_t max_samples = static_cast<size_t>(max_seconds * kRate + 0.5);
  Mem mem{data, len, 0};
  const int io_size = 32768;
  auto* io_buf = static_cast<uint8_t*>(av_malloc(io_size));
  AVIOContext* io = avio_alloc_context(io_buf, io_size, 0, &mem, read_mem, nullptr, nullptr);
  AVFormatContext* fmt = avformat_alloc_context();
  AVCodecContext* dec = nullptr;
  SwrContext* swr = nullptr;
  AVPacket* pkt = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  bool ok = false;
  const bool trace = std::getenv("KT2N_DECODE_TRACE") != nullptr;
  auto tt0 = std::chrono::steady_clock::now();
  auto lap = [&](const char* what) {
    if (!trace) return;
    auto t = std::chrono::steady_clock::now();
    std::fprintf(stderr, "decode %s %.3f ms\n", what, std::chrono::duration<double, std::milli>(t - tt0).count());
    tt0 = t;
  };
  auto fail = [&](const std::string& why) {
    if (err) *err = why;
    return false;
  };
  auto cleanup = [&]() {
    av_frame_free(&frame);
    av_packet_free(&pkt);
    swr_free(&swr);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    if (io) {
      av_freep(&io->buffer);
      avio_context_free(&io);
    }
  };
  do {
    fmt->pb = io;
    fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
    int r = avformat_open_input(&fmt, "pipe:0", nullptr, nullptr);
    if (r < 0) {
      fail("open: " + averr(r));
      break;
    }
    lap("open");
    r = avformat_find_stream_info(fmt, nullptr);
    if (r < 0) {
      fail("stream info: " + averr(r));
      break;
    }
    lap("stream_info");
#if LIBAVFORMAT_VERSION_MAJOR >= 59
    const AVCodec* codec = nullptr;
#else
    AVCodec* codec = nullptr;
#endif
    int si = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (si < 0 || !codec) {
      fail("no audio stream");
      break;
    }
    dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, fmt->streams[si]->codecpar);
    dec->thread_count = 1;
    if ((r = avcodec_open2(dec, codec, nullptr)) < 0) {
      fail("codec open: " + averr(r));
      break;
    }
    if (stats) {
      stats->codec = codec->name;
      stats->format = fmt->iformat ? fmt->iformat->name : "";
    }
    std::vector<float> chunk;
    bool done = false;
    auto convert = [&](AVFrame* f) -> bool {
      if (!swr) {
#if LIBAVUTIL_VERSION_MAJOR >= 58
        AVChannelLayout in_layout;
        if (f->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC || f->ch_layout.nb_channels == 0)
          av_channel_layout_default(&in_layout, f->ch_layout.nb_channels > 0 ? f->ch_layout.nb_channels : 1);
        else
          av_channel_layout_copy(&in_layout, &f->ch_layout);
        AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
        if (swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_FLT, kRate, &in_layout,
                                static_cast<AVSampleFormat>(f->format), f->sample_rate, 0, nullptr) < 0)
          return false;
        if (stats) stats->in_channels = in_layout.nb_channels;
        av_channel_layout_uninit(&in_layout);
#else
        uint64_t in_layout = f->channel_layout ? f->channel_layout : av_get_default_channel_layout(f->channels);
        swr = swr_alloc_set_opts(nullptr, AV_CH_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, kRate, in_layout,
                                 static_cast<AVSampleFormat>(f->format), f->sample_rate, 0, nullptr);
        if (stats) stats->in_channels = f->channels;
#endif
        if (!swr || swr_init(swr) < 0) return false;
        if (stats) stats->in_rate = f->sample_rate;
      }
      int cap = static_cast<int>(swr_get_out_samples(swr, f ? f->nb_samples : 0)) + 64;
      chunk.resize(static_cast<size_t>(cap));
      uint8_t* outp = reinterpret_cast<uint8_t*>(chunk.data());
      int n = swr_convert(swr, &outp, cap, f ? const_cast<const uint8_t**>(f->extended_data) : nullptr,
                          f ? f->nb_samples : 0);
      if (n < 0) return false;
      out.insert(out.end(), chunk.begin(), chunk.begin() + n);
      if (out.size() >= max_samples) done = true;
      return true;
    };
    bool bad = false;
    auto drain = [&]() {
      while (!done && !bad) {
        int rr = avcodec_receive_frame(dec, frame);
        if (rr == AVERROR(EAGAIN) || rr == AVERROR_EOF) break;
        if (rr < 0) {
          bad = true;
          fail("decode: " + averr(rr));
          break;
        }
        if (!convert(frame)) {
          bad = true;
          fail("resample");
        }
        av_frame_unref(frame);
      }
    };
    while (!done && !bad && av_read_frame(fmt, pkt) >= 0) {
      if (pkt->stream_index == si) {
        r = avcodec_send_packet(dec, pkt);
        if (r < 0 && r != AVERROR(EAGAIN) && r != AVERROR_INVALIDDATA) {
          bad = true;
          fail("send: " + averr(r));
        }
        drain();
      }
      av_packet_unref(pkt);
    }
    lap("decode_loop");
    if (bad) break;
    if (!done) {
      avcodec_send_packet(dec, nullptr);
      drain();
      if (bad) break;
      if (swr && !done) convert(nullptr);  // flush the resampler's delay line
    }
    if (out.size() > max_samples) out.resize(max_samples);
    ok = !out.empty();
    if (!ok) fail("no decodable audio");
  } while (false);
  cleanup();
  return ok;
}

}  // namespace kt2n::voice
