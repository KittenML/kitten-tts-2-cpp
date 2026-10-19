// L11: s3gen.kt2n clone-group reader (see clone_weights.h).
#include "clone_weights.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

#include "../core/json.h"
#include "kt2n/sha256.h"

namespace kt2n::voice {

const char* const CloneWeightsFile::kGroups[4] = {"clone_s3tokenizer", "clone_campplus", "clone_pyannote",
                                                  "clone_consts"};

namespace {
constexpr char kMagic[8] = {'K', 'T', '2', 'N', 'S', '3', 'G', '1'};
constexpr uint64_t kHeader = 64, kAlign = 4096;

[[noreturn]] void fail(const std::string& path, const std::string& why) {
  throw std::runtime_error("s3gen.kt2n clone groups (" + path + "): " + why);
}

uint64_t rd64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
uint32_t rd32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

// group -> (file-name prefix, prefix removed for the builder's name)
struct GroupRule {
  const char* group;
  const char* prefix;
  bool strip;
};
constexpr GroupRule kRules[4] = {{"clone_s3tokenizer", "tokenizer.", false},
                                 {"clone_campplus", "speaker_encoder.", false},
                                 {"clone_pyannote", "clone_pyannote.", true},
                                 {"clone_consts", "clone_consts.", true}};

bool non_negative_int(const json::Value* v) { return v && v->is_int() && v->as_int() >= 0; }
}  // namespace

const std::vector<RequiredTensor>& required_clone_tensors() {
  static const std::vector<RequiredTensor> req = {
      // clone_consts
      {"kaldi_mel_banks", {80, 257}},
      {"kaldi_povey_400", {400}},
      {"mel24_basis", {80, 961}},
      {"mel24_window", {1920}},
      {"poly_2_3_taps", {61}},
      {"s3_mel_filters", {128, 201}},
      {"s3_window", {400}},
      {"ta_24_16_kernel_cpu", {2, 1, 23}},
      {"ta_24_16_kernel_dev", {2, 1, 23}},
      // clone_pyannote (what the builder reads; the SincNet filters come precomputed)
      {"sincnet.wav_norm1d.weight", {1}},
      {"sincnet.wav_norm1d.bias", {1}},
      {"sincnet.conv1d.0.filters", {80, 1, 251}},
      {"sincnet.norm1d.0.weight", {80}},
      {"sincnet.norm1d.0.bias", {80}},
      {"sincnet.conv1d.1.weight", {60, 80, 5}},
      {"sincnet.conv1d.1.bias", {60}},
      {"sincnet.norm1d.1.weight", {60}},
      {"sincnet.norm1d.1.bias", {60}},
      {"sincnet.conv1d.2.weight", {60, 60, 5}},
      {"sincnet.conv1d.2.bias", {60}},
      {"sincnet.norm1d.2.weight", {60}},
      {"sincnet.norm1d.2.bias", {60}},
      {"tdnns.0.weight", {512, 60, 5}},
      {"tdnns.0.bias", {512}},
      {"tdnns.2.weight", {512}},
      {"tdnns.2.bias", {512}},
      {"tdnns.2.running_mean", {512}},
      {"tdnns.2.running_var", {512}},
      {"tdnns.3.weight", {512, 512, 3}},
      {"tdnns.3.bias", {512}},
      {"tdnns.5.weight", {512}},
      {"tdnns.5.bias", {512}},
      {"tdnns.5.running_mean", {512}},
      {"tdnns.5.running_var", {512}},
      {"tdnns.6.weight", {512, 512, 3}},
      {"tdnns.6.bias", {512}},
      {"tdnns.8.weight", {512}},
      {"tdnns.8.bias", {512}},
      {"tdnns.8.running_mean", {512}},
      {"tdnns.8.running_var", {512}},
      {"tdnns.9.weight", {512, 512, 1}},
      {"tdnns.9.bias", {512}},
      {"tdnns.11.weight", {512}},
      {"tdnns.11.bias", {512}},
      {"tdnns.11.running_mean", {512}},
      {"tdnns.11.running_var", {512}},
      {"tdnns.12.weight", {1500, 512, 1}},
      {"tdnns.12.bias", {1500}},
      {"tdnns.14.weight", {1500}},
      {"tdnns.14.bias", {1500}},
      {"tdnns.14.running_mean", {1500}},
      {"tdnns.14.running_var", {1500}},
      {"embedding.weight", {512, 3000}},
      {"embedding.bias", {512}},
  };
  return req;
}

CloneWeightsFile::CloneWeightsFile(CloneWeightsFile&& o) noexcept
    : base_(o.base_),
      size_(o.size_),
      tensors_(std::move(o.tensors_)),
      span_begin_(o.span_begin_),
      span_end_(o.span_end_),
      tensor_bytes_(o.tensor_bytes_),
      verify_ms_(o.verify_ms_) {
  o.base_ = nullptr;
  o.size_ = 0;
}

CloneWeightsFile::~CloneWeightsFile() {
  if (base_) munmap(const_cast<uint8_t*>(base_), size_);
}

CloneWeightsFile CloneWeightsFile::open(const std::string& path, const CloneWeightsOptions& opt) {
  CloneWeightsFile f;
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) fail(path, "cannot open");
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size < (off_t)kHeader) {
    ::close(fd);
    fail(path, "shorter than the 64-byte header");
  }
  f.size_ = (uint64_t)st.st_size;
  void* map = mmap(nullptr, f.size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) fail(path, "mmap failed");
  f.base_ = static_cast<const uint8_t*>(map);
  const uint8_t* b = f.base_;
  // Header (FORMAT "Header", same container as the engine file; only the magic differs).
  if (std::memcmp(b, kMagic, 8) != 0) fail(path, "magic is not KT2NS3G1");
  if (rd32(b + 8) != 1) fail(path, "version != 1");
  if (rd32(b + 12) != 0) fail(path, "flags != 0");
  const uint64_t meta_off = rd64(b + 16), meta_len = rd64(b + 24), data_off = rd64(b + 32), data_len = rd64(b + 40),
                 file_len = rd64(b + 48);
  for (int i = 56; i < 64; ++i)
    if (b[i]) fail(path, "reserved header bytes not zero");
  if (meta_off != kHeader) fail(path, "meta_off != 64");
  if (meta_len > f.size_ - kHeader) fail(path, "metadata past the end of the file");
  if (data_off != ((kHeader + meta_len + kAlign - 1) / kAlign) * kAlign) fail(path, "data_off not the first 4096 boundary");
  if (file_len != f.size_ || data_off > file_len || data_len != file_len - data_off || data_len % kAlign)
    fail(path, "file_len / data_len do not match the file");
  json::Value meta;
  try {
    meta = json::parse(std::string_view(reinterpret_cast<const char*>(b + meta_off), meta_len));
  } catch (const std::exception& e) {
    fail(path, std::string("metadata: ") + e.what());
  }
  const json::Value* fmt = meta.get("format");
  const json::Value* ver = meta.get("version");
  if (!fmt || !fmt->is_string() || fmt->as_string() != "kt2n" || !ver || !ver->is_int() || ver->as_int() != 1)
    fail(path, "metadata format/version");
  const json::Value* tensors = meta.get("tensors");
  if (!tensors || !tensors->is_array()) fail(path, "metadata has no tensors list");

  std::map<std::string, size_t> seen;
  size_t per_group[4] = {0, 0, 0, 0};
  f.span_begin_ = std::numeric_limits<uint64_t>::max();
  for (const json::Value& t : tensors->items()) {
    const json::Value* g = t.get("group");
    if (!g || !g->is_string()) continue;
    int gi = -1;
    for (int i = 0; i < 4; ++i)
      if (g->as_string() == kRules[i].group) gi = i;
    if (gi < 0) continue;
    const json::Value* name = t.get("name");
    if (!name || !name->is_string()) fail(path, "clone tensor without a name");
    const std::string& fname = name->as_string();
    const std::string& prefix = kRules[gi].prefix;
    if (fname.compare(0, prefix.size(), prefix) != 0 || fname.size() == prefix.size())
      fail(path, fname + ": name does not start with " + prefix + " (group " + kRules[gi].group + ")");
    const json::Value* kind = t.get("kind");
    const json::Value* dtype = t.get("dtype");
    const json::Value* tr = t.get("transform");
    if (!kind || !kind->is_string() || kind->as_string() != "dense") fail(path, fname + ": kind != dense");
    if (!dtype || !dtype->is_string() || dtype->as_string() != "f32") fail(path, fname + ": dtype != f32");
    if (!tr || !tr->is_string() || tr->as_string() != "copy") fail(path, fname + ": transform != copy");
    const json::Value* shape = t.get("shape");
    const json::Value* data = t.get("data");
    if (!shape || !shape->is_array() || !data || !data->is_object()) fail(path, fname + ": shape/data missing");
    CloneTensor ct;
    ct.file_name = fname;
    ct.name = kRules[gi].strip ? fname.substr(prefix.size()) : fname;
    ct.group = kRules[gi].group;
    uint64_t numel = 1;
    for (const json::Value& d : shape->items()) {
      if (!non_negative_int(&d)) fail(path, fname + ": bad dim");
      const uint64_t dim = (uint64_t)d.as_int();
      if (dim && numel > (std::numeric_limits<uint64_t>::max() / 4) / dim) fail(path, fname + ": shape overflows");
      numel *= dim;
      ct.shape.push_back(d.as_int());
    }
    const json::Value* dsh = data->get("shape");
    const json::Value* ddt = data->get("dtype");
    const json::Value* off = data->get("off");
    const json::Value* len = data->get("len");
    const json::Value* sha = data->get("sha256");
    if (!dsh || !dsh->is_array() || dsh->items().size() != ct.shape.size()) fail(path, fname + ": data.shape");
    for (size_t i = 0; i < ct.shape.size(); ++i)
      if (!dsh->items()[i].is_int() || dsh->items()[i].as_int() != ct.shape[i]) fail(path, fname + ": data.shape");
    if (!ddt || !ddt->is_string() || ddt->as_string() != "f32") fail(path, fname + ": data.dtype");
    if (!non_negative_int(off) || !non_negative_int(len)) fail(path, fname + ": data.off/len");
    if (!sha || !sha->is_string() || !is_sha256_hex(sha->as_string())) fail(path, fname + ": data.sha256");
    const uint64_t o = (uint64_t)off->as_int(), l = (uint64_t)len->as_int();
    if (o % kAlign) fail(path, fname + ": segment not 4096-aligned");
    if (l != numel * 4) fail(path, fname + ": len != numel * 4");
    if (o > data_len || l > data_len - o) fail(path, fname + ": segment past the data area");
    if (tr->as_string() == "copy") {
      const json::Value* src = t.get("src_sha256");
      if (!src || !src->is_string() || src->as_string() != sha->as_string()) fail(path, fname + ": copy src_sha256");
    }
    ct.abs_off = data_off + o;
    ct.len = l;
    ct.sha256 = sha->as_string();
    if (!seen.emplace(ct.name, f.tensors_.size()).second) fail(path, "duplicate clone tensor " + ct.name);
    f.span_begin_ = std::min(f.span_begin_, ct.abs_off);
    f.span_end_ = std::max(f.span_end_, ct.abs_off + ct.len);
    f.tensor_bytes_ += ct.len;
    per_group[gi]++;
    f.tensors_.push_back(std::move(ct));
  }
  for (int i = 0; i < 4; ++i)
    if (!per_group[i]) fail(path, std::string("group ") + kRules[i].group + " is missing");
  for (const RequiredTensor& r : opt.check_required ? required_clone_tensors() : std::vector<RequiredTensor>{}) {
    auto it = seen.find(r.name);
    if (it == seen.end()) fail(path, std::string("missing ") + r.name);
    if (f.tensors_[it->second].shape != r.shape) fail(path, std::string(r.name) + ": unexpected shape");
  }
  if (opt.verify_sha256) {
    const auto t0 = std::chrono::steady_clock::now();
    for (const CloneTensor& t : f.tensors_)
      if (sha256_hex(b + t.abs_off, t.len) != t.sha256) fail(path, t.file_name + ": segment sha256 mismatch");
    f.verify_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  return f;
}

}  // namespace kt2n::voice
