// L11: s3gen.kt2n clone-group reader (src/voice/clone_weights.{h,cc}) on small synthetic files.
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "clone_weights.h"
#include "kt2n/sha256.h"

using kt2n::voice::CloneWeightsFile;
using kt2n::voice::CloneWeightsOptions;

namespace {
int failures = 0;
#define EXPECT(c)                                                     \
  do {                                                                \
    if (!(c)) {                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

struct Spec {
  std::string name, group;
  std::vector<int64_t> shape;
  float fill = 1.f;
  std::string dtype = "f32", transform = "copy";
  int64_t off_delta = 0;   // corrupt the recorded offset
  int64_t len_delta = 0;   // corrupt the recorded length
  bool bad_sha = false;    // record a wrong sha256
};

std::string shape_json(const std::vector<int64_t>& s) {
  std::string o = "[";
  for (size_t i = 0; i < s.size(); ++i) o += (i ? "," : "") + std::to_string(s[i]);
  return o + "]";
}

// Writes a KT2NS3G1 container: 64-byte header, JSON metadata, 4096-aligned segments.
std::string write_file(const std::vector<Spec>& specs, const std::function<void(std::vector<uint8_t>&)>& patch = {}) {
  std::vector<std::vector<uint8_t>> blobs;
  std::string tensors = "[";
  uint64_t off = 0;
  std::vector<uint64_t> offs;
  for (size_t i = 0; i < specs.size(); ++i) {
    const Spec& s = specs[i];
    size_t n = 1;
    for (auto d : s.shape) n *= (size_t)d;
    std::vector<float> v(n, s.fill);
    std::vector<uint8_t> raw(n * 4);
    std::memcpy(raw.data(), v.data(), raw.size());
    std::string sha = kt2n::sha256_hex(raw.data(), raw.size());
    if (s.bad_sha) sha[0] = sha[0] == '0' ? '1' : '0';
    offs.push_back(off);
    tensors += std::string(i ? "," : "") + "{\"name\":\"" + s.name + "\",\"kind\":\"dense\",\"dtype\":\"" + s.dtype +
               "\",\"shape\":" + shape_json(s.shape) + ",\"group\":\"" + s.group + "\",\"transform\":\"" +
               s.transform + "\",\"src_sha256\":\"" + sha + "\",\"sources\":[],\"data\":{\"off\":" +
               std::to_string((int64_t)off + s.off_delta) + ",\"len\":" +
               std::to_string((int64_t)raw.size() + s.len_delta) + ",\"shape\":" + shape_json(s.shape) +
               ",\"dtype\":\"" + s.dtype + "\",\"sha256\":\"" + sha + "\"}}";
    off += (raw.size() + 4095) / 4096 * 4096;
    blobs.push_back(std::move(raw));
  }
  tensors += "]";
  const std::string meta = "{\"format\":\"kt2n\",\"version\":1,\"config\":{},\"tensors\":" + tensors + "}";
  const uint64_t data_off = (64 + meta.size() + 4095) / 4096 * 4096;
  const uint64_t data_len = off == 0 ? 4096 : off;
  std::vector<uint8_t> file(data_off + data_len, 0);
  std::memcpy(file.data(), "KT2NS3G1", 8);
  uint32_t ver = 1, flags = 0;
  uint64_t meta_off = 64, meta_len = meta.size(), file_len = file.size();
  std::memcpy(&file[8], &ver, 4);
  std::memcpy(&file[12], &flags, 4);
  std::memcpy(&file[16], &meta_off, 8);
  std::memcpy(&file[24], &meta_len, 8);
  std::memcpy(&file[32], &data_off, 8);
  std::memcpy(&file[40], &data_len, 8);
  std::memcpy(&file[48], &file_len, 8);
  std::memcpy(&file[64], meta.data(), meta.size());
  for (size_t i = 0; i < blobs.size(); ++i) std::memcpy(&file[data_off + offs[i]], blobs[i].data(), blobs[i].size());
  if (patch) patch(file);
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() / ("kt2n_clone_weights_test_" + std::to_string(::getpid()) + "_" +
                                                 std::to_string(counter++) + ".kt2n"))
          .string();
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(file.data()), (std::streamsize)file.size());
  return path;
}

std::vector<Spec> good() {
  return {
      {"flow.encoder.x", "encoder", {3}},  // not a clone group: ignored
      {"speaker_encoder.b", "clone_campplus", {4}, 2.f},
      {"tokenizer.a", "clone_s3tokenizer", {2, 3}, 3.f},
      {"clone_pyannote.tdnns.0.bias", "clone_pyannote", {5}, 4.f},
      {"clone_consts.mel24_window", "clone_consts", {1}, 5.f},
  };
}

bool throws(const std::string& path, CloneWeightsOptions opt, const char* needle) {
  try {
    CloneWeightsFile::open(path, opt);
  } catch (const std::exception& e) {
    const bool ok = std::string(e.what()).find(needle) != std::string::npos;
    if (!ok) std::fprintf(stderr, "  message was: %s (expected '%s')\n", e.what(), needle);
    return ok;
  }
  std::fprintf(stderr, "  no exception (expected '%s')\n", needle);
  return false;
}
}  // namespace

int main() {
  CloneWeightsOptions loose;
  loose.check_required = false;
  loose.verify_sha256 = true;
  std::vector<std::string> paths;
  {
    const std::string p = write_file(good());
    paths.push_back(p);
    CloneWeightsFile f = CloneWeightsFile::open(p, loose);
    EXPECT(f.tensors().size() == 4);
    EXPECT(f.tensors()[0].name == "speaker_encoder.b");
    EXPECT(f.tensors()[1].name == "tokenizer.a");
    EXPECT(f.tensors()[2].name == "tdnns.0.bias");     // amendment groups lose their prefix
    EXPECT(f.tensors()[3].name == "mel24_window");
    EXPECT(f.tensors()[2].group == "clone_pyannote");
    EXPECT(f.tensor_bytes() == 4 * (4 + 6 + 5 + 1));
    EXPECT(f.span_begin() == f.tensors()[0].abs_off);
    EXPECT(f.span_end() == f.tensors()[3].abs_off + 4);
    float v = 0.f;
    std::memcpy(&v, f.base() + f.tensors()[2].abs_off, 4);
    EXPECT(v == 4.f);
    // moved-from object releases nothing; the moved-to one keeps the mapping
    CloneWeightsFile g(std::move(f));
    EXPECT(g.tensors().size() == 4 && g.base() != nullptr);
  }
  auto with = [&](const std::function<void(std::vector<Spec>&)>& edit) {
    auto s = good();
    edit(s);
    paths.push_back(write_file(s));
    return paths.back();
  };
  EXPECT(throws(paths.emplace_back(write_file(good(), [](auto& b) { b[3] = 'X'; })), loose, "magic"));
  EXPECT(throws(paths.emplace_back(write_file(good(), [](auto& b) { b[60] = 1; })), loose, "reserved"));
  EXPECT(throws(paths.emplace_back(write_file(good(), [](auto& b) { b.push_back(0); })), loose, "file_len"));
  EXPECT(throws(with([](auto& s) { s[2].off_delta = 8; }), loose, "4096-aligned"));
  EXPECT(throws(with([](auto& s) { s[2].len_delta = -4; }), loose, "numel * 4"));
  EXPECT(throws(with([](auto& s) { s[2].off_delta = 1 << 20; }), loose, "past the data area"));
  EXPECT(throws(with([](auto& s) { s[3].name = "pyannote.tdnns.0.bias"; }), loose, "does not start with"));
  EXPECT(throws(with([](auto& s) { s[4].name = "clone_consts."; }), loose, "does not start with"));
  EXPECT(throws(with([](auto& s) { s[2].dtype = "bf16"; }), loose, "dtype"));
  EXPECT(throws(with([](auto& s) { s[2].transform = "bf16_rne"; }), loose, "transform"));
  EXPECT(throws(with([](auto& s) { s.erase(s.begin() + 1); }), loose, "clone_campplus is missing"));
  EXPECT(throws(with([](auto& s) { s[4].name = "clone_consts.tdnns.0.bias"; }), loose, "duplicate"));
  {
    const std::string p = with([](auto& s) { s[3].bad_sha = true; });
    EXPECT(throws(p, loose, "sha256"));  // src_sha256 equals data.sha256 (copy rule) but the bytes do not
    CloneWeightsOptions no_hash = loose;
    no_hash.verify_sha256 = false;
    bool opened = true;
    try {
      CloneWeightsFile::open(p, no_hash);
    } catch (...) {
      opened = false;
    }
    EXPECT(opened);  // hashing is opt-in, like the engine file's
  }
  {
    CloneWeightsOptions strict;  // the real required list: the tiny file has none of it
    EXPECT(throws(paths.front(), strict, "missing"));
  }
  EXPECT(throws("/nonexistent/s3gen.kt2n", loose, "cannot open"));
  for (const auto& p : paths) std::remove(p.c_str());
  if (failures) {
    std::fprintf(stderr, "test_clone_weights: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("test_clone_weights: PASS\n");
  return 0;
}
