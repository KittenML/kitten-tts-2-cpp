// L11: reader for the clone-build weights inside s3gen.kt2n (magic KT2NS3G1, the container format in tools/kt2n_assets/format.py "Companion files").
//
// Four precision groups feed the clone build, all f32 `copy`:
//   clone_s3tokenizer  tokenizer.*        (frozen list, tools/kt2n_assets/s3gen_tensors.tsv)
//   clone_campplus     speaker_encoder.*  (frozen list)
//   clone_pyannote     clone_pyannote.*   (amendment: the speaker-embedding net, catalogue bank excluded)
//   clone_consts       clone_consts.*     (amendment: front-end constants: mel bases, windows, resampler taps)
// The builder sees the tokenizer / CAMPPlus names unchanged and the two amendment groups without their group
// prefix, i.e. exactly the names of the three-file layout (S3Gen safetensors + voicebuild_consts +
// voicebuild_pyannote) it read before, so both layouts give the builder byte-identical weights.
//
// Host only (no CUDA): the file is mapped, validated and released; VoiceBuilder copies the bytes it needs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kt2n::voice {

struct CloneTensor {
  std::string name;       // the builder's name (group prefix of the amendment groups removed)
  std::string file_name;  // the name in s3gen.kt2n
  std::string group;
  std::vector<int64_t> shape;
  uint64_t abs_off = 0;  // absolute file offset of the data segment
  uint64_t len = 0;      // bytes (= numel * 4)
  std::string sha256;    // the segment's recorded sha256
};

struct CloneWeightsOptions {
  bool verify_sha256 = false;  // hash every clone segment against its recorded sha256 (FORMAT: optional at load)
  bool check_required = true;  // every tensor of required_clone_tensors() present with its shape (off: unit tests)
};

class CloneWeightsFile {
 public:
  // Maps and validates. Throws std::runtime_error naming the first problem.
  static CloneWeightsFile open(const std::string& path, const CloneWeightsOptions& opt = {});
  CloneWeightsFile(CloneWeightsFile&& o) noexcept;
  CloneWeightsFile& operator=(CloneWeightsFile&&) = delete;
  CloneWeightsFile(const CloneWeightsFile&) = delete;
  ~CloneWeightsFile();

  const std::vector<CloneTensor>& tensors() const { return tensors_; }
  const uint8_t* base() const { return base_; }
  uint64_t size() const { return size_; }
  // [span_begin, span_end): absolute offsets covering every clone segment (one upload when contiguous).
  uint64_t span_begin() const { return span_begin_; }
  uint64_t span_end() const { return span_end_; }
  uint64_t tensor_bytes() const { return tensor_bytes_; }
  double verify_ms() const { return verify_ms_; }

  static const char* const kGroups[4];

 private:
  CloneWeightsFile() = default;
  const uint8_t* base_ = nullptr;
  uint64_t size_ = 0;
  std::vector<CloneTensor> tensors_;
  uint64_t span_begin_ = 0, span_end_ = 0, tensor_bytes_ = 0;
  double verify_ms_ = 0.0;
};

// The amendment groups' required tensors and shapes (what VoiceBuilder reads); tokenizer / CAMPPlus shapes are
// fixed by tools/kt2n_assets/s3gen_tensors.tsv and checked by the builder's warm-up (every kernel runs once).
struct RequiredTensor {
  const char* name;
  std::vector<int64_t> shape;
};
const std::vector<RequiredTensor>& required_clone_tensors();

}  // namespace kt2n::voice
