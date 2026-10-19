// L11: in-process upload decode with the FFmpeg libraries, mirroring production's CLI call
// (kitten_tts_09/runtime.py:11084 _decode_custom_audio):
//   ffmpeg -nostdin -threads 1 -i pipe:0 -map_metadata -1 -vn -sn -dn -ac 1 -ar 24000 -t 30.100 -f f32le pipe:1
// Decoding MP3/AAC/Opus/Vorbis/FLAC/WebM/MP4 by hand is not sensible (codecs are handled by a library);
// the libraries are the same ones the CLI uses, called in process instead of through a subprocess.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kt2n::voice {

struct DecodeStats {
  int in_rate = 0;
  int in_channels = 0;
  std::string codec;
  std::string format;
};

// Returns false on any failure (production maps that to 422 invalid_audio). out = mono float32 at 24 kHz,
// at most max_seconds (30.1 s like the CLI's -t), resampled/downmixed by libswresample with default options.
bool decode_audio(const uint8_t* data, size_t len, double max_seconds, std::vector<float>& out, std::string* err,
                  DecodeStats* stats = nullptr);

}  // namespace kt2n::voice
