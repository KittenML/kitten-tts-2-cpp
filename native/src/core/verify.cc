// verify_all(): hash every segment, rebuild every ternary matrix (unless
// VerifyOptions::reconstruct is false) and compare with the converter's
// recorded hashes. Tensors are checked in parallel; each tensor is streamed
// row by row, so memory stays at a few rows per thread.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "kt2n/bf16.h"
#include "kt2n/engine_file.h"
#include "kt2n/ref_kernels.h"
#include "kt2n/sha256.h"

namespace kt2n {
namespace {

struct TensorResult {
  std::vector<VerifyIssue> errors;
  std::vector<VerifyIssue> warnings;
  uint64_t bytes = 0;
  uint64_t segments = 0;
  uint64_t invalid = 0;
};

// Counts one kind of problem and remembers where it first happened.
struct Tally {
  uint64_t count = 0;
  std::string first;
  void hit(int64_t row, int64_t group) {
    if (count++ == 0) first = "row " + std::to_string(row) + " group " + std::to_string(group);
  }
};

std::string hash_bytes(std::span<const uint8_t> b) {
  Sha256 h;
  constexpr size_t kChunk = size_t{1} << 20;
  for (size_t off = 0; off < b.size(); off += kChunk) {
    h.update(b.data() + off, std::min(kChunk, b.size() - off));
  }
  return h.hexdigest();
}

// Rebuilds the matrix row by row: canonical-form, zero bookkeeping and src_sha256 checks.
void verify_ternary(const TensorInfo& t, TensorResult& r) {
  const int64_t N = t.rows();
  const int64_t K = t.cols();
  const int64_t words = K / ref::kCodesPerWord;
  const int64_t groups = K / ref::kGroup;
  const int64_t words_per_group = ref::kGroup / ref::kCodesPerWord;
  std::vector<int32_t> crow(static_cast<size_t>(words));
  std::vector<uint16_t> srow(static_cast<size_t>(groups));
  std::vector<uint16_t> wrow(static_cast<size_t>(K));
  Sha256 h;
  Tally negative_scale, nonfinite_scale, code_without_scale, scale_without_code;
  uint64_t invalid = 0;
  uint64_t zero_count = 0;  // code-0 weights seen so far (row-major)
  uint64_t neg_count = 0;   // of those, sign bit set in zero_signs
  const uint8_t* zs = t.has_zero_signs ? t.zero_signs.ptr : nullptr;
  const uint64_t zs_bits = t.has_zero_signs ? t.zero_signs.len * 8 : 0;
  bool zs_overflow = false;

  for (int64_t n = 0; n < N; ++n) {
    // memcpy: segment pointers are only as aligned as data_off.
    std::memcpy(crow.data(), t.codes.ptr + static_cast<size_t>(n * words) * 4, static_cast<size_t>(words) * 4);
    std::memcpy(srow.data(), t.scales.ptr + static_cast<size_t>(n * groups) * 2, static_cast<size_t>(groups) * 2);
    invalid += ref::unpack_ternary_row(crow.data(), srow.data(), K, wrow.data());

    // Zeros: count them and restore recorded -0.0 signs (row-major, LSB first).
    for (int64_t j = 0; j < words; ++j) {
      const auto word = static_cast<uint32_t>(crow[static_cast<size_t>(j)]);
      if (word == 0xFFFFFFFFu) continue;  // all code 3 (already counted as invalid)
      for (int64_t o = 0; o < ref::kCodesPerWord; ++o) {
        if (((word >> (2 * o)) & 3u) != 0u) continue;
        if (zs != nullptr) {
          if (zero_count < zs_bits) {
            if ((zs[zero_count >> 3] >> (zero_count & 7u)) & 1u) {
              wrow[static_cast<size_t>(j * ref::kCodesPerWord + o)] = kBf16SignMask;
              ++neg_count;
            }
          } else {
            zs_overflow = true;
          }
        }
        ++zero_count;
      }
    }

    for (int64_t g = 0; g < groups; ++g) {
      const uint16_t s = srow[static_cast<size_t>(g)];
      bool any_code = false;
      for (int64_t j = 0; j < words_per_group; ++j) {
        any_code = any_code || crow[static_cast<size_t>(g * words_per_group + j)] != 0;
      }
      if (!bf16_is_finite(s)) {
        nonfinite_scale.hit(n, g);
      } else if ((s & kBf16SignMask) != 0) {
        negative_scale.hit(n, g);
      } else if (s == 0 && any_code) {
        code_without_scale.hit(n, g);
      } else if (s != 0 && !any_code) {
        scale_without_code.hit(n, g);
      }
    }
    h.update(wrow.data(), static_cast<size_t>(K) * 2);
  }

  r.invalid += invalid;
  if (invalid > 0) {
    r.errors.push_back({t.name, "invalid_code", std::to_string(invalid) + " codes == 3 (never produced by pack_exact)"});
  }
  const std::pair<const char*, const Tally*> tallies[] = {
      {"noncanonical:nonfinite_scale", &nonfinite_scale},
      {"noncanonical:negative_scale", &negative_scale},
      {"noncanonical:code_without_scale", &code_without_scale},
      {"noncanonical:scale_without_code", &scale_without_code},
  };
  for (const auto& [check, tally] : tallies) {
    if (tally->count > 0) {
      r.errors.push_back({t.name, check, std::to_string(tally->count) + " groups, first at " + tally->first});
    }
  }

  // Zero bookkeeping against the recorded counts.
  if (t.zeros >= 0 && zero_count != static_cast<uint64_t>(t.zeros)) {
    r.errors.push_back({t.name, "zeros", "codes hold " + std::to_string(zero_count) + " zero weights, metadata says " +
                                             std::to_string(t.zeros)});
  }
  if (t.has_zero_signs) {
    if (zs_overflow) {
      r.errors.push_back({t.name, "zero_signs", "fewer sign bits (" + std::to_string(zs_bits) + ") than zero weights (" +
                                                    std::to_string(zero_count) + ")"});
    } else {
      bool padding_clean = true;
      for (uint64_t b = zero_count; b < zs_bits; ++b) padding_clean = padding_clean && ((zs[b >> 3] >> (b & 7u)) & 1u) == 0;
      if (!padding_clean) r.errors.push_back({t.name, "zero_signs", "padding bits after the last zero are not 0"});
    }
    if (t.neg_zero >= 0 && neg_count != static_cast<uint64_t>(t.neg_zero)) {
      r.errors.push_back({t.name, "neg_zero", "zero_signs hold " + std::to_string(neg_count) + " -0.0 signs, metadata says " +
                                                  std::to_string(t.neg_zero)});
    }
  } else if (t.neg_zero > 0) {
    r.errors.push_back({t.name, "zero_signs", "missing although neg_zero = " + std::to_string(t.neg_zero)});
  }

  r.bytes += static_cast<uint64_t>(N) * static_cast<uint64_t>(K) * 2;
  const std::string got = h.hexdigest();
  if (got != t.src_sha256) {
    r.errors.push_back({t.name, "src_sha256", "reconstructed bf16 sha256 " + got + " != src_sha256 " + t.src_sha256});
  }
}

void verify_tensor(const EngineFile& f, const TensorInfo& t, bool reconstruct, TensorResult& r) {
  std::string data_hash;
  for (const auto& [name, seg] : t.segments()) {
    const std::string got = hash_bytes(f.bytes(*seg));
    r.bytes += seg->len;
    r.segments += 1;
    if (got != seg->sha256) {
      r.errors.push_back({t.name, "seg_sha256:" + std::string(name), "sha256 " + got + " != recorded " + seg->sha256});
    }
    if (name == "data") data_hash = got;
  }
  if (t.kind == TensorKind::kDense) {
    // EngineFile::open() guarantees the data segment is stored in the logical dtype and shape.
    if (data_hash != t.src_sha256) {
      r.errors.push_back({t.name, "src_sha256", "data sha256 " + data_hash + " != src_sha256 " + t.src_sha256});
    }
  } else if (reconstruct) {
    verify_ternary(t, r);
  }
}

uint64_t tensor_bytes(const TensorInfo& t) {
  uint64_t b = 0;
  for (const auto& [name, seg] : t.segments()) b += seg->len;
  if (t.kind == TensorKind::kTernary2) b += static_cast<uint64_t>(t.rows()) * static_cast<uint64_t>(t.cols()) * 2;
  return b;
}

uint64_t align_up(uint64_t x) { return (x + kSegmentAlign - 1) / kSegmentAlign * kSegmentAlign; }

bool all_zero(const uint8_t* p, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i) {
    if (p[i] != 0) return false;
  }
  return true;
}

// Canonical layout rules (the container format in tools/kt2n_assets/format.py): header fields and zero padding.
void verify_layout(const EngineFile& f, std::vector<VerifyIssue>& errors) {
  const FileHeader& h = f.header();
  if (h.meta_off != kEngineHeaderSize) {
    errors.push_back({"", "layout:meta_off", "meta_off " + std::to_string(h.meta_off) + " != 64"});
  }
  if (h.data_off != align_up(h.meta_off + h.meta_len)) {
    errors.push_back({"", "layout:data_off", "data_off " + std::to_string(h.data_off) +
                                                 " is not the first 4096 boundary after the metadata"});
  }
  if (h.data_len % kSegmentAlign != 0) {
    errors.push_back({"", "layout:data_len", "data_len " + std::to_string(h.data_len) + " is not a multiple of 4096"});
  }
  if (h.data_off + h.data_len != h.file_len) {
    errors.push_back({"", "layout:file_len", "data_off + data_len != file_len"});
  }
  // Every byte outside the header, the metadata and the segments must be zero.
  std::vector<std::pair<uint64_t, uint64_t>> used;  // absolute [begin, end)
  used.emplace_back(0, kEngineHeaderSize);
  used.emplace_back(h.meta_off, h.meta_off + h.meta_len);
  for (const auto& t : f.tensors()) {
    for (const auto& [name, seg] : t.segments()) used.emplace_back(seg->abs_off, seg->abs_off + seg->len);
  }
  std::sort(used.begin(), used.end());
  uint64_t cursor = 0;
  uint64_t dirty = 0;
  for (const auto& [b, e] : used) {
    if (b > cursor && !all_zero(f.base() + cursor, b - cursor)) ++dirty;
    cursor = std::max(cursor, e);
  }
  if (f.size() > cursor && !all_zero(f.base() + cursor, f.size() - cursor)) ++dirty;
  if (dirty > 0) {
    errors.push_back({"", "layout:padding", std::to_string(dirty) + " padding gaps contain non-zero bytes"});
  }
}

}  // namespace

VerifyReport verify_all(const EngineFile& file, const VerifyOptions& options) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto& tensors = file.tensors();
  VerifyReport rep;

  // Largest first, so the long tensors start early.
  std::vector<size_t> order(tensors.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return tensor_bytes(tensors[a]) > tensor_bytes(tensors[b]); });

  int threads = options.threads > 0 ? options.threads : static_cast<int>(std::thread::hardware_concurrency());
  threads = std::max(1, std::min<int>(threads, static_cast<int>(std::max<size_t>(1, tensors.size()))));
  rep.threads = threads;

  std::vector<TensorResult> results(tensors.size());
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    while (true) {
      const size_t k = next.fetch_add(1);
      if (k >= order.size()) return;
      const size_t i = order[k];
      verify_tensor(file, tensors[i], options.reconstruct, results[i]);
    }
  };
  if (threads == 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  }

  verify_layout(file, rep.errors);
  // Report in file order (deterministic output).
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    auto& r = results[i];
    rep.tensors += 1;
    rep.ternary += t.kind == TensorKind::kTernary2 ? 1 : 0;
    rep.dense += t.kind == TensorKind::kDense ? 1 : 0;
    rep.segments += r.segments;
    rep.bytes_hashed += r.bytes;
    rep.invalid_codes += r.invalid;
    for (auto& e : r.errors) rep.errors.push_back(std::move(e));
    for (auto& w : r.warnings) rep.warnings.push_back(std::move(w));
  }
  rep.ok = rep.errors.empty();
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return rep;
}

}  // namespace kt2n
