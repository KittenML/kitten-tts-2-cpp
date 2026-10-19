// KT2N engine file (format version 1): read-only, memory-mapped reader.
//
// Layout (little-endian; authoritative spec shared with the converter lane):
//   64-byte header: magic "KT2NENG1"; u32 version = 1; u32 flags = 0;
//   u64 meta_off; u64 meta_len; u64 data_off; u64 data_len; u64 file_len;
//   8 reserved zero bytes.
//   meta: UTF-8 JSON at meta_off: {"format":"kt2n","version":1,"source":{...},
//     "config":{...},"runtime_config":{...},"quant":{"group":128,...},"tensors":[...]}
//   each tensor: {"name","kind":"ternary2"|"dense","dtype","shape":[logical],
//     "src_sha256": sha256 of the original tensor bytes,
//     ternary2: "codes":{"off","len","shape":[N,K/16],"dtype":"i32","sha256"},
//               "scales":{"off","len","shape":[N,K/128],"dtype":"bf16","sha256"},
//               optional "zeros" (number of zero weights), "neg_zero" (how many
//               of them are -0.0 in the source), and "zero_signs" (u8
//               [ceil(zeros/8)], present only when neg_zero > 0: the sign bit of
//               every code-0 weight in row-major order, LSB first, padding 0)
//     dense:    "data":{"off","len","shape","dtype","sha256"}}
//   Segment offsets are relative to data_off; each starts at a multiple of 4096.
//   Unknown extra fields are ignored. The converter lane's the container format in tools/kt2n_assets/format.py is
//   the long-form description (canonical layout: meta_off = 64, data_off = first
//   4096 boundary after the metadata, data_len a multiple of 4096, zero padding).
//
// EngineFile::open() validates everything that does not need hashing (header,
// bounds, alignment, shapes vs lengths, overlaps). verify_all() hashes and
// also checks the canonical layout.
//
// zero_signs only matter for rebuilding the source bytes (-0.0 vs +0.0); the
// engine never needs them: both decode to 0 in every kernel.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kt2n {

inline constexpr char kEngineMagic[8] = {'K', 'T', '2', 'N', 'E', 'N', 'G', '1'};
inline constexpr uint32_t kEngineFormatVersion = 1;
inline constexpr uint64_t kEngineHeaderSize = 64;
inline constexpr uint64_t kSegmentAlign = 4096;
inline constexpr int64_t kTernaryGroup = 128;
inline constexpr int64_t kTernaryKMultiple = 256;  // pack_exact requires K % 256 == 0

// Malformed or unsupported engine file.
class FormatError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// The file could not be opened / mapped.
class IoError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t flags;
  uint64_t meta_off;
  uint64_t meta_len;
  uint64_t data_off;
  uint64_t data_len;
  uint64_t file_len;
  uint8_t reserved[8];
};
static_assert(sizeof(FileHeader) == kEngineHeaderSize, "header must be 64 bytes");

enum class DType : uint8_t { kBF16, kF16, kF32, kF64, kI8, kI16, kI32, kI64, kU8, kU16, kU32, kU64, kBool };

// Accepts the spec's lowercase names ("bf16", "f32", "i32", ...) and the
// safetensors spellings ("BF16", "F32", "I32", ...).
std::optional<DType> parse_dtype(std::string_view name) noexcept;
std::string_view dtype_name(DType dtype) noexcept;  // canonical lowercase name
size_t dtype_size(DType dtype) noexcept;

enum class TensorKind : uint8_t { kTernary2, kDense };
std::string_view kind_name(TensorKind kind) noexcept;

struct Segment {
  uint64_t off = 0;      // relative to data_off, as stored
  uint64_t abs_off = 0;  // data_off + off (absolute file offset)
  uint64_t len = 0;
  std::vector<int64_t> shape;
  DType dtype = DType::kU8;
  std::string sha256;            // 64 lowercase hex
  const uint8_t* ptr = nullptr;  // into the mapping; valid while the EngineFile lives
};

struct TensorInfo {
  std::string name;
  TensorKind kind = TensorKind::kDense;
  DType dtype = DType::kBF16;  // logical dtype of the original tensor
  std::vector<int64_t> shape;  // logical shape
  std::string src_sha256;      // sha256 of the original tensor bytes
  Segment codes;               // ternary2 only: int32 [N, K/16]
  Segment scales;              // ternary2 only: bf16 [N, K/128]
  Segment zero_signs;          // ternary2, optional (has_zero_signs): u8 [ceil(zeros/8)]
  bool has_zero_signs = false;
  int64_t zeros = -1;          // ternary2: recorded number of zero weights (-1 = not recorded)
  int64_t neg_zero = -1;       // ternary2: recorded number of -0.0 weights (-1 = not recorded)
  Segment data;                // dense only

  int64_t numel() const noexcept;
  // ternary2: N and K of the logical [N, K] matrix.
  int64_t rows() const noexcept { return shape.empty() ? 0 : shape[0]; }
  int64_t cols() const noexcept { return shape.size() < 2 ? 0 : shape[1]; }
  // ("codes", seg), ("scales", seg) [+ ("zero_signs", seg)] for ternary2;
  // ("data", seg) for dense.
  std::vector<std::pair<std::string_view, const Segment*>> segments() const;
};

class EngineFile {
 public:
  // Maps the file read-only and validates it. Throws IoError / FormatError.
  static EngineFile open(const std::string& path);

  EngineFile(EngineFile&&) noexcept;
  EngineFile& operator=(EngineFile&&) noexcept;
  EngineFile(const EngineFile&) = delete;
  EngineFile& operator=(const EngineFile&) = delete;
  ~EngineFile();

  const std::string& path() const noexcept;
  const FileHeader& header() const noexcept;
  uint64_t size() const noexcept;  // bytes on disk (== header().file_len)
  const uint8_t* base() const noexcept;
  std::string_view meta_json() const noexcept;
  const std::vector<TensorInfo>& tensors() const noexcept;
  const TensorInfo* find(std::string_view name) const noexcept;
  std::span<const uint8_t> bytes(const Segment& seg) const noexcept;
  // True when data_off is a multiple of 4096, i.e. every segment is page-aligned
  // in the file (and in the mapping). Segment offsets are always checked.
  bool data_page_aligned() const noexcept;

  // Metadata lookups by JSON pointer (RFC 6901), e.g. "/config/hidden_size" or
  // "/runtime_config/audio_token_start". nullopt if missing or of another type.
  bool meta_has(std::string_view pointer) const;
  std::optional<int64_t> meta_int(std::string_view pointer) const;
  std::optional<double> meta_number(std::string_view pointer) const;
  std::optional<std::string> meta_string(std::string_view pointer) const;
  std::optional<bool> meta_bool(std::string_view pointer) const;
  // Compact JSON text of the value at pointer (objects keep file order).
  std::optional<std::string> meta_dump(std::string_view pointer) const;

 private:
  struct Impl;
  explicit EngineFile(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Full verification (hashes every byte).
// ---------------------------------------------------------------------------
struct VerifyIssue {
  std::string tensor;  // tensor name ("" for file-level issues)
  std::string check;   // e.g. "seg_sha256:codes", "src_sha256", "invalid_code"
  std::string detail;
};

struct VerifyOptions {
  int threads = 0;         // 0 = hardware concurrency
  // true: rebuild every ternary matrix (all ternary2 checks below). false:
  // segment hashes, layout and dense src_sha256 only (no per-weight work).
  bool reconstruct = true;
};

struct VerifyReport {
  bool ok = false;
  uint64_t tensors = 0;
  uint64_t ternary = 0;
  uint64_t dense = 0;
  uint64_t segments = 0;
  uint64_t bytes_hashed = 0;
  uint64_t invalid_codes = 0;  // codes == 3 (never valid)
  double seconds = 0.0;
  int threads = 0;
  std::vector<VerifyIssue> errors;    // any error -> ok == false
  std::vector<VerifyIssue> warnings;  // informational, ok stays true
};

// Checks:
//   * layout: meta_off == 64, data_off == first 4096 boundary after the
//     metadata, data_len % 4096 == 0, file_len == data_off + data_len, and every
//     byte not covered by the metadata or a segment is zero;
//   * every segment's sha256 (codes / scales / zero_signs / data) over exactly
//     `len` bytes;
//   * ternary2 (only with options.reconstruct): rebuilds bf16 [N, K] row by
//     row from codes + scales (+ the recorded -0.0 signs) and checks
//     sha256 == src_sha256; no code == 3;
//     zero count == "zeros", -0.0 count == "neg_zero", zero_signs padding bits
//     are 0; canonical pack_exact form (each scale finite and >= +0; a group
//     with scale 0 has only zero codes; a group with a non-zero scale has at
//     least one non-zero code);
//   * dense (open() requires the logical dtype and shape): data sha256 == src_sha256.
VerifyReport verify_all(const EngineFile& file, const VerifyOptions& options = {});

}  // namespace kt2n
