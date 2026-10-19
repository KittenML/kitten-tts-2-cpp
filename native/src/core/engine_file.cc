#include "kt2n/engine_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <unordered_set>

#include "json.h"
#include "kt2n/sha256.h"

namespace kt2n {

// ---------------------------------------------------------------------------
// dtype / kind helpers
// ---------------------------------------------------------------------------
namespace {

struct DTypeName {
  DType dtype;
  std::string_view lower;
  std::string_view upper;  // safetensors spelling
  size_t size;
};

constexpr DTypeName kDTypes[] = {
    {DType::kBF16, "bf16", "BF16", 2}, {DType::kF16, "f16", "F16", 2},   {DType::kF32, "f32", "F32", 4},
    {DType::kF64, "f64", "F64", 8},    {DType::kI8, "i8", "I8", 1},      {DType::kI16, "i16", "I16", 2},
    {DType::kI32, "i32", "I32", 4},    {DType::kI64, "i64", "I64", 8},   {DType::kU8, "u8", "U8", 1},
    {DType::kU16, "u16", "U16", 2},    {DType::kU32, "u32", "U32", 4},   {DType::kU64, "u64", "U64", 8},
    {DType::kBool, "bool", "BOOL", 1},
};

}  // namespace

std::optional<DType> parse_dtype(std::string_view name) noexcept {
  for (const auto& d : kDTypes) {
    if (name == d.lower || name == d.upper) return d.dtype;
  }
  return std::nullopt;
}

std::string_view dtype_name(DType dtype) noexcept {
  for (const auto& d : kDTypes) {
    if (d.dtype == dtype) return d.lower;
  }
  return "?";
}

size_t dtype_size(DType dtype) noexcept {
  for (const auto& d : kDTypes) {
    if (d.dtype == dtype) return d.size;
  }
  return 0;
}

std::string_view kind_name(TensorKind kind) noexcept {
  return kind == TensorKind::kTernary2 ? "ternary2" : "dense";
}

int64_t TensorInfo::numel() const noexcept {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::vector<std::pair<std::string_view, const Segment*>> TensorInfo::segments() const {
  if (kind == TensorKind::kTernary2) {
    if (has_zero_signs) return {{"codes", &codes}, {"scales", &scales}, {"zero_signs", &zero_signs}};
    return {{"codes", &codes}, {"scales", &scales}};
  }
  return {{"data", &data}};
}

// ---------------------------------------------------------------------------
// EngineFile
// ---------------------------------------------------------------------------
struct EngineFile::Impl {
  std::string path;
  int fd = -1;
  const uint8_t* map = nullptr;
  uint64_t size = 0;
  FileHeader header{};
  std::string_view meta_text;
  json::Value meta;
  std::vector<TensorInfo> tensors;

  ~Impl() {
    if (map != nullptr) munmap(const_cast<uint8_t*>(map), size);
    if (fd >= 0) close(fd);
  }
};

namespace {

[[noreturn]] void bad(const std::string& path, const std::string& what) {
  throw FormatError(path + ": " + what);
}

bool range_ok(uint64_t off, uint64_t len, uint64_t limit) noexcept {
  return off <= limit && len <= limit - off;
}

// Checked product of non-negative dims; false on overflow / negative.
bool checked_numel(const std::vector<int64_t>& shape, uint64_t& out) noexcept {
  uint64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) return false;
    const auto ud = static_cast<uint64_t>(d);
    if (ud != 0 && n > UINT64_MAX / ud) return false;
    n *= ud;
  }
  out = n;
  return true;
}

std::string shape_str(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

const json::Value& require(const std::string& ctx, const json::Value& obj, std::string_view key) {
  const json::Value* v = obj.get(key);
  if (v == nullptr) throw FormatError(ctx + ": missing key '" + std::string(key) + "'");
  return *v;
}

uint64_t require_u64(const std::string& ctx, const json::Value& obj, std::string_view key) {
  const json::Value& v = require(ctx, obj, key);
  if (!v.is_int() || v.as_int() < 0) {
    throw FormatError(ctx + ": '" + std::string(key) + "' must be a non-negative integer");
  }
  return static_cast<uint64_t>(v.as_int());
}

std::string require_string(const std::string& ctx, const json::Value& obj, std::string_view key) {
  const json::Value& v = require(ctx, obj, key);
  if (!v.is_string()) throw FormatError(ctx + ": '" + std::string(key) + "' must be a string");
  return v.as_string();
}

std::vector<int64_t> require_shape(const std::string& ctx, const json::Value& obj, std::string_view key) {
  const json::Value& v = require(ctx, obj, key);
  if (!v.is_array()) throw FormatError(ctx + ": '" + std::string(key) + "' must be an array");
  std::vector<int64_t> shape;
  for (const auto& d : v.items()) {
    if (!d.is_int() || d.as_int() < 0) {
      throw FormatError(ctx + ": '" + std::string(key) + "' must hold non-negative integers");
    }
    shape.push_back(d.as_int());
  }
  return shape;
}

DType require_dtype(const std::string& ctx, const json::Value& obj, std::string_view key) {
  const std::string name = require_string(ctx, obj, key);
  const auto dt = parse_dtype(name);
  if (!dt) throw FormatError(ctx + ": unknown dtype '" + name + "'");
  return *dt;
}

std::string require_sha(const std::string& ctx, const json::Value& obj, std::string_view key) {
  std::string s = require_string(ctx, obj, key);
  if (!is_sha256_hex(s)) throw FormatError(ctx + ": '" + std::string(key) + "' is not 64 lowercase hex");
  return s;
}

Segment parse_segment(const std::string& ctx, const json::Value& v, const FileHeader& h,
                      const uint8_t* base) {
  if (!v.is_object()) throw FormatError(ctx + ": segment must be an object");
  Segment s;
  s.off = require_u64(ctx, v, "off");
  s.len = require_u64(ctx, v, "len");
  s.shape = require_shape(ctx, v, "shape");
  s.dtype = require_dtype(ctx, v, "dtype");
  s.sha256 = require_sha(ctx, v, "sha256");
  if (s.off % kSegmentAlign != 0) {
    throw FormatError(ctx + ": off " + std::to_string(s.off) + " is not a multiple of 4096");
  }
  if (!range_ok(s.off, s.len, h.data_len)) {
    throw FormatError(ctx + ": [off, off+len) = [" + std::to_string(s.off) + ", +" + std::to_string(s.len) +
                      ") is outside data_len " + std::to_string(h.data_len));
  }
  uint64_t numel = 0;
  if (!checked_numel(s.shape, numel) || numel > UINT64_MAX / dtype_size(s.dtype) ||
      numel * dtype_size(s.dtype) != s.len) {
    throw FormatError(ctx + ": len " + std::to_string(s.len) + " does not match shape " + shape_str(s.shape) +
                      " x " + std::string(dtype_name(s.dtype)));
  }
  s.abs_off = h.data_off + s.off;
  s.ptr = base + s.abs_off;
  return s;
}

TensorInfo parse_tensor(const std::string& path, size_t index, const json::Value& t, const FileHeader& h,
                        const uint8_t* base) {
  std::string ctx = path + ": tensors[" + std::to_string(index) + "]";
  if (!t.is_object()) throw FormatError(ctx + " must be an object");
  TensorInfo info;
  info.name = require_string(ctx, t, "name");
  if (info.name.empty()) throw FormatError(ctx + ": empty name");
  ctx += " '" + info.name + "'";
  const std::string kind = require_string(ctx, t, "kind");
  info.dtype = require_dtype(ctx, t, "dtype");
  info.shape = require_shape(ctx, t, "shape");
  info.src_sha256 = require_sha(ctx, t, "src_sha256");
  uint64_t numel = 0;
  if (!checked_numel(info.shape, numel)) throw FormatError(ctx + ": bad shape");

  if (kind == "ternary2") {
    info.kind = TensorKind::kTernary2;
    if (info.dtype != DType::kBF16) throw FormatError(ctx + ": ternary2 must have dtype bf16");
    if (info.shape.size() != 2) throw FormatError(ctx + ": ternary2 must be 2-D [N, K]");
    const int64_t N = info.shape[0];
    const int64_t K = info.shape[1];
    if (N <= 0 || K <= 0 || K % kTernaryKMultiple != 0) {
      throw FormatError(ctx + ": ternary2 needs N > 0 and K % 256 == 0, got " + shape_str(info.shape));
    }
    info.codes = parse_segment(ctx + ".codes", require(ctx, t, "codes"), h, base);
    info.scales = parse_segment(ctx + ".scales", require(ctx, t, "scales"), h, base);
    if (info.codes.dtype != DType::kI32 || info.codes.shape != std::vector<int64_t>{N, K / 16}) {
      throw FormatError(ctx + ".codes: must be i32 " + shape_str({N, K / 16}) + ", got " +
                        std::string(dtype_name(info.codes.dtype)) + " " + shape_str(info.codes.shape));
    }
    if (info.scales.dtype != DType::kBF16 || info.scales.shape != std::vector<int64_t>{N, K / kTernaryGroup}) {
      throw FormatError(ctx + ".scales: must be bf16 " + shape_str({N, K / kTernaryGroup}) + ", got " +
                        std::string(dtype_name(info.scales.dtype)) + " " + shape_str(info.scales.shape));
    }
    if (t.get("zeros") != nullptr) {
      info.zeros = static_cast<int64_t>(require_u64(ctx, t, "zeros"));
      if (static_cast<uint64_t>(info.zeros) > numel) throw FormatError(ctx + ": zeros > N*K");
    }
    if (t.get("neg_zero") != nullptr) {
      info.neg_zero = static_cast<int64_t>(require_u64(ctx, t, "neg_zero"));
      if (info.zeros >= 0 && info.neg_zero > info.zeros) throw FormatError(ctx + ": neg_zero > zeros");
    }
    if (const json::Value* zs = t.get("zero_signs")) {
      if (info.zeros < 0) throw FormatError(ctx + ": zero_signs present but 'zeros' is not recorded");
      info.zero_signs = parse_segment(ctx + ".zero_signs", *zs, h, base);
      info.has_zero_signs = true;
      const int64_t want = (info.zeros + 7) / 8;
      if (info.zero_signs.dtype != DType::kU8 || info.zero_signs.shape != std::vector<int64_t>{want}) {
        throw FormatError(ctx + ".zero_signs: must be u8 " + shape_str({want}) + ", got " +
                          std::string(dtype_name(info.zero_signs.dtype)) + " " + shape_str(info.zero_signs.shape));
      }
    }
  } else if (kind == "dense") {
    info.kind = TensorKind::kDense;
    info.data = parse_segment(ctx + ".data", require(ctx, t, "data"), h, base);
    // FORMAT.md: a dense segment holds the source bytes unchanged (same dtype, same shape).
    if (info.data.dtype != info.dtype || info.data.shape != info.shape) {
      throw FormatError(ctx + ".data: must be " + std::string(dtype_name(info.dtype)) + " " + shape_str(info.shape) +
                        " like the logical tensor, got " + std::string(dtype_name(info.data.dtype)) + " " +
                        shape_str(info.data.shape));
    }
  } else {
    throw FormatError(ctx + ": unknown kind '" + kind + "'");
  }
  return info;
}

}  // namespace

EngineFile::EngineFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
EngineFile::EngineFile(EngineFile&&) noexcept = default;
EngineFile& EngineFile::operator=(EngineFile&&) noexcept = default;
EngineFile::~EngineFile() = default;

EngineFile EngineFile::open(const std::string& path) {
  auto impl = std::make_unique<Impl>();
  impl->path = path;
  impl->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (impl->fd < 0) throw IoError(path + ": open failed: " + std::strerror(errno));
  struct stat st {};
  if (fstat(impl->fd, &st) != 0) throw IoError(path + ": fstat failed: " + std::strerror(errno));
  if (!S_ISREG(st.st_mode)) throw IoError(path + ": not a regular file");
  impl->size = static_cast<uint64_t>(st.st_size);
  if (impl->size < kEngineHeaderSize) {
    bad(path, "file is " + std::to_string(impl->size) + " bytes, smaller than the 64-byte header");
  }
  void* m = mmap(nullptr, impl->size, PROT_READ, MAP_PRIVATE, impl->fd, 0);
  if (m == MAP_FAILED) throw IoError(path + ": mmap failed: " + std::strerror(errno));
  impl->map = static_cast<const uint8_t*>(m);

  FileHeader& h = impl->header;
  std::memcpy(&h, impl->map, sizeof(FileHeader));
  if (std::memcmp(h.magic, kEngineMagic, sizeof(kEngineMagic)) != 0) bad(path, "bad magic (not a KT2N engine file)");
  if (h.version != kEngineFormatVersion) {
    bad(path, "unsupported format version " + std::to_string(h.version) + " (this reader supports 1)");
  }
  if (h.flags != 0) bad(path, "flags must be 0, got " + std::to_string(h.flags));
  for (uint8_t b : h.reserved) {
    if (b != 0) bad(path, "reserved header bytes must be zero");
  }
  if (h.file_len != impl->size) {
    bad(path, "header file_len " + std::to_string(h.file_len) + " != actual size " + std::to_string(impl->size) +
                  " (truncated or padded file)");
  }
  if (h.meta_off < kEngineHeaderSize || !range_ok(h.meta_off, h.meta_len, impl->size)) {
    bad(path, "meta region out of bounds");
  }
  if (!range_ok(h.data_off, h.data_len, impl->size)) bad(path, "data region out of bounds");
  if (h.data_len > 0) {
    if (h.data_off < kEngineHeaderSize) bad(path, "data region overlaps the header");
    if (h.meta_off < h.data_off + h.data_len && h.data_off < h.meta_off + h.meta_len) {
      bad(path, "meta region overlaps the data region");
    }
  }

  impl->meta_text = std::string_view(reinterpret_cast<const char*>(impl->map + h.meta_off), h.meta_len);
  try {
    impl->meta = json::parse(impl->meta_text);
  } catch (const json::ParseError& e) {
    bad(path, std::string("metadata: ") + e.what());
  }
  const json::Value& meta = impl->meta;
  if (!meta.is_object()) bad(path, "metadata must be a JSON object");
  const json::Value* fmt = meta.get("format");
  if (fmt == nullptr || !fmt->is_string() || fmt->as_string() != "kt2n") bad(path, "metadata format must be \"kt2n\"");
  const json::Value* ver = meta.get("version");
  if (ver == nullptr || !ver->is_int() || ver->as_int() != 1) bad(path, "metadata version must be 1");
  const json::Value* tensors = meta.get("tensors");
  if (tensors == nullptr || !tensors->is_array()) bad(path, "metadata needs a 'tensors' array");

  std::unordered_set<std::string> names;
  impl->tensors.reserve(tensors->items().size());
  for (size_t i = 0; i < tensors->items().size(); ++i) {
    TensorInfo t = parse_tensor(path, i, tensors->items()[i], h, impl->map);
    if (!names.insert(t.name).second) bad(path, "duplicate tensor name '" + t.name + "'");
    impl->tensors.push_back(std::move(t));
  }

  bool any_ternary = false;
  for (const auto& t : impl->tensors) any_ternary = any_ternary || t.kind == TensorKind::kTernary2;
  if (any_ternary) {
    const json::Value* group = meta.at_pointer("/quant/group");
    if (group == nullptr || !group->is_int() || group->as_int() != kTernaryGroup) {
      bad(path, "metadata quant.group must be 128 when ternary2 tensors are present");
    }
  }

  // No two segments may overlap.
  std::vector<std::pair<uint64_t, uint64_t>> spans;
  for (const auto& t : impl->tensors) {
    for (const auto& [name, seg] : t.segments()) {
      if (seg->len > 0) spans.emplace_back(seg->off, seg->len);
    }
  }
  std::sort(spans.begin(), spans.end());
  for (size_t i = 1; i < spans.size(); ++i) {
    if (spans[i - 1].first + spans[i - 1].second > spans[i].first) {
      bad(path, "segments overlap at data offset " + std::to_string(spans[i].first));
    }
  }

  return EngineFile(std::move(impl));
}

const std::string& EngineFile::path() const noexcept { return impl_->path; }
const FileHeader& EngineFile::header() const noexcept { return impl_->header; }
uint64_t EngineFile::size() const noexcept { return impl_->size; }
const uint8_t* EngineFile::base() const noexcept { return impl_->map; }
std::string_view EngineFile::meta_json() const noexcept { return impl_->meta_text; }
const std::vector<TensorInfo>& EngineFile::tensors() const noexcept { return impl_->tensors; }

const TensorInfo* EngineFile::find(std::string_view name) const noexcept {
  for (const auto& t : impl_->tensors) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

std::span<const uint8_t> EngineFile::bytes(const Segment& seg) const noexcept {
  return {impl_->map + seg.abs_off, static_cast<size_t>(seg.len)};
}

bool EngineFile::data_page_aligned() const noexcept { return impl_->header.data_off % kSegmentAlign == 0; }

bool EngineFile::meta_has(std::string_view pointer) const { return impl_->meta.at_pointer(pointer) != nullptr; }

std::optional<int64_t> EngineFile::meta_int(std::string_view pointer) const {
  const json::Value* v = impl_->meta.at_pointer(pointer);
  if (v == nullptr || !v->is_int()) return std::nullopt;
  return v->as_int();
}

std::optional<double> EngineFile::meta_number(std::string_view pointer) const {
  const json::Value* v = impl_->meta.at_pointer(pointer);
  if (v == nullptr || !v->is_number()) return std::nullopt;
  return v->as_double();
}

std::optional<std::string> EngineFile::meta_string(std::string_view pointer) const {
  const json::Value* v = impl_->meta.at_pointer(pointer);
  if (v == nullptr || !v->is_string()) return std::nullopt;
  return v->as_string();
}

std::optional<bool> EngineFile::meta_bool(std::string_view pointer) const {
  const json::Value* v = impl_->meta.at_pointer(pointer);
  if (v == nullptr || !v->is_bool()) return std::nullopt;
  return v->as_bool();
}

std::optional<std::string> EngineFile::meta_dump(std::string_view pointer) const {
  const json::Value* v = impl_->meta.at_pointer(pointer);
  if (v == nullptr) return std::nullopt;
  return v->dump();
}

}  // namespace kt2n
