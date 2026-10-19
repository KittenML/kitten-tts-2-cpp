// In-process encoders: libavcodec/libavformat configured as production's ffmpeg
// CLI command (see include/kt2n/post/encoder.h). Builds against FFmpeg 4.4
// (libavcodec 58) and 5.x/6.x (59/60); the channel-layout API differs.
#include "kt2n/post/encoder.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace kt2n::post {

std::vector<std::string> ffmpeg_stream_command(std::string_view format, int probesize) {
  std::string out_fmt, codec;
  std::vector<std::string> codec_args;
  if (format == "mp3") {
    out_fmt = "mp3";
    codec_args = {"-codec:a", "libmp3lame", "-b:a", "128k", "-write_xing", "0"};
  } else if (format == "aac") {
    out_fmt = "adts";
    codec_args = {"-codec:a", "aac", "-b:a", "128k"};
  } else if (format == "opus") {
    out_fmt = "ogg";
    codec_args = {"-codec:a", "libopus", "-b:a", "96k"};
  } else if (format == "flac") {
    out_fmt = "flac";
    codec_args = {"-codec:a", "flac"};
  } else {
    throw std::invalid_argument("not an encoded format");
  }
  std::vector<std::string> cmd = {"ffmpeg", "-hide_banner", "-loglevel", "error"};
  if (probesize > 0) {
    cmd.push_back("-probesize");
    cmd.push_back(std::to_string(probesize));
  }
  for (const char* a : {"-f", "f32le", "-ar", "24000", "-ac", "1", "-i", "pipe:0"}) cmd.emplace_back(a);
  cmd.insert(cmd.end(), codec_args.begin(), codec_args.end());
  for (const char* a : {"-flush_packets", "1", "-f"}) cmd.emplace_back(a);
  cmd.push_back(out_fmt);
  cmd.emplace_back("pipe:1");
  return cmd;
}

namespace {

std::string av_err(int code) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(code, buf, sizeof buf);
  return buf;
}

class AvEncoder final : public StreamEncoder {
 public:
  ~AvEncoder() override { release(); }

  bool open(std::string_view format, std::string* error) {
    const char* muxer = nullptr;
    const char* codec_name = nullptr;
    int64_t bit_rate = 0;
    AVSampleFormat sample_fmt = AV_SAMPLE_FMT_FLTP;
    if (format == "mp3") {
      muxer = "mp3";
      codec_name = "libmp3lame";
      bit_rate = 128000;
      sample_fmt = AV_SAMPLE_FMT_FLTP;  // CLI: flt -> fltp (auto_resampler)
    } else if (format == "aac") {
      muxer = "adts";
      codec_name = "aac";
      bit_rate = 128000;
      sample_fmt = AV_SAMPLE_FMT_FLTP;
    } else if (format == "opus") {
      muxer = "ogg";
      codec_name = "libopus";
      bit_rate = 96000;
      sample_fmt = AV_SAMPLE_FMT_FLT;  // CLI: no conversion
    } else if (format == "flac") {
      muxer = "flac";
      codec_name = "flac";
      sample_fmt = AV_SAMPLE_FMT_S32;  // CLI: flt -> s32 (swresample), 24-bit FLAC
      convert_s32_ = true;
    } else {
      return fail(error, "unknown format");
    }
    int rc = avformat_alloc_output_context2(&oc_, nullptr, muxer, nullptr);
    if (rc < 0 || oc_ == nullptr) return fail(error, "avformat_alloc_output_context2: " + av_err(rc));
    const AVCodec* codec = avcodec_find_encoder_by_name(codec_name);
    if (codec == nullptr) return fail(error, std::string("encoder not available: ") + codec_name);
    enc_ = avcodec_alloc_context3(codec);
    if (enc_ == nullptr) return fail(error, "avcodec_alloc_context3");
    enc_->sample_rate = kSampleRate;
    enc_->sample_fmt = sample_fmt;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)
    av_channel_layout_default(&enc_->ch_layout, 1);
#else
    enc_->channels = 1;
    enc_->channel_layout = AV_CH_LAYOUT_MONO;
#endif
    enc_->time_base = AVRational{1, kSampleRate};
    if (bit_rate > 0) enc_->bit_rate = bit_rate;
    if (convert_s32_) {
      // The CLI copies the decoder's bits_per_raw_sample (pcm_f32le: 32), capped at
      // the sample format's width; flac then encodes 24 bits per sample.
      enc_->bits_per_raw_sample = 32;
    }
    if ((oc_->oformat->flags & AVFMT_GLOBALHEADER) != 0) enc_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    rc = avcodec_open2(enc_, codec, nullptr);
    if (rc < 0) return fail(error, "avcodec_open2: " + av_err(rc));
    st_ = avformat_new_stream(oc_, nullptr);
    if (st_ == nullptr) return fail(error, "avformat_new_stream");
    rc = avcodec_parameters_from_context(st_->codecpar, enc_);
    if (rc < 0) return fail(error, "avcodec_parameters_from_context: " + av_err(rc));
    st_->time_base = enc_->time_base;
    // fftools set_encoder_id(): stream "encoder" = "Lavc<maj>.<min>.<mic> <codec name>".
    {
      const unsigned v = avcodec_version();
      char id[128];
      std::snprintf(id, sizeof id, "Lavc%u.%u.%u %s", v >> 16, (v >> 8) & 0xFF, v & 0xFF, codec->name);
      av_dict_set(&st_->metadata, "encoder", id, AV_DICT_DONT_OVERWRITE);
    }
    io_buf_ = static_cast<unsigned char*>(av_malloc(kIoBuffer));
    if (io_buf_ == nullptr) return fail(error, "av_malloc");
    // Non-seekable output, like pipe:1 (no seek callback -> seekable = 0).
    io_ = avio_alloc_context(io_buf_, kIoBuffer, 1, this, nullptr, &AvEncoder::write_cb, nullptr);
    if (io_ == nullptr) return fail(error, "avio_alloc_context");
    io_buf_ = nullptr;  // owned by io_ now
    oc_->pb = io_;
    oc_->flush_packets = 1;  // -flush_packets 1
    AVDictionary* opts = nullptr;
    if (format == "mp3") av_dict_set(&opts, "write_xing", "0", 0);
    sink_ = &header_;
    rc = avformat_write_header(oc_, &opts);
    av_dict_free(&opts);
    if (rc < 0) return fail(error, "avformat_write_header: " + av_err(rc));
    avio_flush(io_);
    sink_ = nullptr;
    frame_size_ = enc_->frame_size > 0 ? enc_->frame_size : 1024;
    frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();
    if (frame_ == nullptr || pkt_ == nullptr) return fail(error, "av_frame_alloc/av_packet_alloc");
    if (convert_s32_) {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)
      AVChannelLayout mono;
      av_channel_layout_default(&mono, 1);
      rc = swr_alloc_set_opts2(&swr_, &mono, AV_SAMPLE_FMT_S32, kSampleRate, &mono, AV_SAMPLE_FMT_FLT,
                               kSampleRate, 0, nullptr);
      if (rc < 0) return fail(error, "swr_alloc_set_opts2: " + av_err(rc));
#else
      swr_ = swr_alloc_set_opts(nullptr, AV_CH_LAYOUT_MONO, AV_SAMPLE_FMT_S32, kSampleRate,
                                AV_CH_LAYOUT_MONO, AV_SAMPLE_FMT_FLT, kSampleRate, 0, nullptr);
      if (swr_ == nullptr) return fail(error, "swr_alloc_set_opts");
#endif
      rc = swr_init(swr_);
      if (rc < 0) return fail(error, "swr_init: " + av_err(rc));
    }
    return true;
  }

  std::string take_header() override {
    std::string h;
    h.swap(header_);
    return h;
  }

  bool push(Samples pcm, std::string& out, std::string* error) override {
    if (finished_) return fail(error, "push after finish");
    pending_.insert(pending_.end(), pcm.begin(), pcm.end());
    size_t at = 0;
    sink_ = &out;
    while (pending_.size() - at >= static_cast<size_t>(frame_size_)) {
      if (!encode_frame(pending_.data() + at, frame_size_, error)) {
        sink_ = nullptr;
        return false;
      }
      at += static_cast<size_t>(frame_size_);
    }
    sink_ = nullptr;
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(at));
    return true;
  }

  bool finish(std::string& out, std::string* error) override {
    if (finished_) return true;
    finished_ = true;
    sink_ = &out;
    bool ok = true;
    if (!pending_.empty()) ok = encode_frame(pending_.data(), static_cast<int>(pending_.size()), error);
    pending_.clear();
    if (ok) ok = drain(nullptr, error);
    if (ok) {
      const int rc = av_write_trailer(oc_);
      if (rc < 0) ok = fail(error, "av_write_trailer: " + av_err(rc));
      avio_flush(io_);
    }
    sink_ = nullptr;
    return ok;
  }

  int frame_size() const override { return frame_size_; }
  int64_t packets() const override { return packets_; }

 private:
  static constexpr int kIoBuffer = 32768;  // libavformat IO_BUFFER_SIZE (pipe:1)

#if LIBAVFORMAT_VERSION_MAJOR >= 61
  static int write_cb(void* opaque, const uint8_t* buf, int size) {
#else
  static int write_cb(void* opaque, uint8_t* buf, int size) {  // FFmpeg 4.4 - 6.0 signature
#endif
    auto* self = static_cast<AvEncoder*>(opaque);
    std::string* sink = self->sink_ != nullptr ? self->sink_ : &self->header_;
    sink->append(reinterpret_cast<const char*>(buf), static_cast<size_t>(size));
    return size;
  }

  bool fail(std::string* error, const std::string& what) {
    if (error != nullptr) *error = what;
    return false;
  }

  bool encode_frame(const float* x, int n, std::string* error) {
    av_frame_unref(frame_);
    frame_->nb_samples = n;
    frame_->format = enc_->sample_fmt;
    frame_->sample_rate = kSampleRate;
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)
    av_channel_layout_copy(&frame_->ch_layout, &enc_->ch_layout);
#else
    frame_->channels = 1;
    frame_->channel_layout = AV_CH_LAYOUT_MONO;
#endif
    int rc = av_frame_get_buffer(frame_, 0);
    if (rc < 0) return fail(error, "av_frame_get_buffer: " + av_err(rc));
    if (convert_s32_) {
      const uint8_t* in[1] = {reinterpret_cast<const uint8_t*>(x)};
      rc = swr_convert(swr_, frame_->data, n, in, n);
      if (rc != n) return fail(error, "swr_convert: " + av_err(rc));
    } else {
      // mono: flt and fltp have the same layout
      std::memcpy(frame_->data[0], x, static_cast<size_t>(n) * sizeof(float));
    }
    frame_->pts = next_pts_;
    next_pts_ += n;
    return drain(frame_, error);
  }

  bool drain(AVFrame* frame, std::string* error) {
    int rc = avcodec_send_frame(enc_, frame);
    if (rc < 0) return fail(error, "avcodec_send_frame: " + av_err(rc));
    while (true) {
      rc = avcodec_receive_packet(enc_, pkt_);
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return true;
      if (rc < 0) return fail(error, "avcodec_receive_packet: " + av_err(rc));
      av_packet_rescale_ts(pkt_, enc_->time_base, st_->time_base);
      pkt_->stream_index = st_->index;
      ++packets_;
      rc = av_interleaved_write_frame(oc_, pkt_);
      if (rc < 0) return fail(error, "av_interleaved_write_frame: " + av_err(rc));
    }
  }

  void release() {
    if (io_ != nullptr) {
      av_freep(&io_->buffer);
      avio_context_free(&io_);
    }
    if (io_buf_ != nullptr) av_freep(&io_buf_);
    if (oc_ != nullptr) {
      oc_->pb = nullptr;
      avformat_free_context(oc_);
      oc_ = nullptr;
    }
    if (enc_ != nullptr) avcodec_free_context(&enc_);
    if (frame_ != nullptr) av_frame_free(&frame_);
    if (pkt_ != nullptr) av_packet_free(&pkt_);
    if (swr_ != nullptr) swr_free(&swr_);
  }

  AVFormatContext* oc_ = nullptr;
  AVCodecContext* enc_ = nullptr;
  AVStream* st_ = nullptr;
  AVIOContext* io_ = nullptr;
  unsigned char* io_buf_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVPacket* pkt_ = nullptr;
  SwrContext* swr_ = nullptr;
  bool convert_s32_ = false;
  bool finished_ = false;
  int frame_size_ = 0;
  int64_t next_pts_ = 0;
  int64_t packets_ = 0;
  std::vector<float> pending_;
  std::string header_;
  std::string* sink_ = nullptr;
};

}  // namespace

std::unique_ptr<StreamEncoder> StreamEncoder::create(std::string_view format, std::string* error) {
  auto enc = std::make_unique<AvEncoder>();
  if (!enc->open(format, error)) return nullptr;
  return enc;
}

}  // namespace kt2n::post
