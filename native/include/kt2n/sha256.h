// SHA-256 (FIPS 180-4); used to check engine-file segments and reconstructed tensors against the converter's
// recorded hashes, and voices.kt2n segments. Portable C++, plus an x86 SHA-NI path picked at run time (11x faster
// on the box CPU; KT2N_SHA256_PORTABLE=1 forces the portable path).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace kt2n {

class Sha256 {
 public:
  using Digest = std::array<uint8_t, 32>;

  Sha256() noexcept { reset(); }
  void reset() noexcept;
  void update(const void* data, size_t len) noexcept;
  // Finalizes and returns the digest; the object is reset afterwards.
  Digest finish() noexcept;
  // finish() as 64 lowercase hex characters.
  std::string hexdigest();

 private:
  void compress(const uint8_t* block) noexcept;

  std::array<uint32_t, 8> h_{};
  std::array<uint8_t, 64> buf_{};
  size_t buf_len_ = 0;
  uint64_t total_len_ = 0;
};

// Lowercase hex of a byte range.
std::string to_hex(const uint8_t* data, size_t len);
// One-shot SHA-256 as 64 lowercase hex characters.
std::string sha256_hex(const void* data, size_t len);
// True if s is exactly 64 lowercase hex characters.
bool is_sha256_hex(std::string_view s) noexcept;

}  // namespace kt2n
