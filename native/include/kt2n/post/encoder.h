// In-process streaming encoders for the API's encoded formats, built on the
// same libraries production's ffmpeg CLI uses (libavcodec / libavformat /
// libswresample, with libmp3lame and libopus), configured to reproduce
// production's exact command (kittenml-api@ebcdd3f kitten_tts/api.py:3141-3176):
//
//   ffmpeg -hide_banner -loglevel error [-probesize 32] -f f32le -ar 24000 -ac 1
//          -i pipe:0 <codec args> -flush_packets 1 -f <fmt> pipe:1
//   mp3 : -codec:a libmp3lame -b:a 128k -write_xing 0   -f mp3
//   aac : -codec:a aac -b:a 128k                         -f adts
//   opus: -codec:a libopus -b:a 96k                      -f ogg
//   flac: -codec:a flac                                  -f flac
//
// What the CLI negotiates (measured on the box, ffmpeg 4.4.2, -loglevel verbose):
// mp3/aac flt->fltp, opus flt (24 kHz native), flac flt->s32 via swresample and
// 24-bit FLAC. Frames go to the encoder in its own frame size (buffersink
// frame_size), the last one short; the stream carries the CLI's "encoder" tag
// ("Lavc<version> <codec>"); packets are flushed to the output one by one
// (-flush_packets 1) through a non-seekable output (a pipe).
//
// Thread model: one encoder per response, used by one thread at a time.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "kt2n/post/post.h"

namespace kt2n::post {

class StreamEncoder {
 public:
  virtual ~StreamEncoder() = default;
  // format: "mp3" | "aac" | "opus" | "flac". nullptr + *error on failure.
  static std::unique_ptr<StreamEncoder> create(std::string_view format, std::string* error);
  // Bytes produced by opening the container (the header), available right
  // after create(); taken once.
  virtual std::string take_header() = 0;
  // Feed float32 mono 24 kHz samples (already normalised / peak-normed); the
  // encoded bytes that are final now are appended to `out`.
  virtual bool push(Samples pcm, std::string& out, std::string* error) = 0;
  // End of stream: drain the encoder, write the trailer.
  virtual bool finish(std::string& out, std::string* error) = 0;
  // Encoder frame size in samples (mp3 1152, aac 1024, opus 480, flac as opened).
  virtual int frame_size() const = 0;
  virtual int64_t packets() const = 0;
};

// The production CLI command for `format` (argv form), for the comparison arm.
std::vector<std::string> ffmpeg_stream_command(std::string_view format, int probesize = 32);

}  // namespace kt2n::post
