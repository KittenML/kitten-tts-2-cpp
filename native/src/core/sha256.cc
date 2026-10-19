#include "kt2n/sha256.h"

#include <cstdlib>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KT2N_SHA256_X86 1
#endif

namespace kt2n {
namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t rotr(uint32_t x, int n) noexcept { return (x >> n) | (x << (32 - n)); }

inline uint32_t load_be32(const uint8_t* p) noexcept {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

#if KT2N_SHA256_X86
// SHA-256 with the x86 SHA extensions (SHA-NI): the same compression function, 4 rounds per sha256rnds2 pair.
// Measured on the box CPU (10 Oct): 174 MB/s portable vs 1939 MB/s for OpenSSL 3.5 (also SHA-NI) on 1.44 MB.
// State layout follows Intel's reference: STATE0 = ABEF, STATE1 = CDGH.
__attribute__((target("sha,sse4.1"))) void compress_shani(uint32_t* state, const uint8_t* data, size_t blocks) {
  const __m128i kMask = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);  // big-endian words
  __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state));
  __m128i st1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state + 4));
  tmp = _mm_shuffle_epi32(tmp, 0xB1);
  st1 = _mm_shuffle_epi32(st1, 0x1B);
  __m128i st0 = _mm_alignr_epi8(tmp, st1, 8);
  st1 = _mm_blend_epi16(st1, tmp, 0xF0);
  for (; blocks; --blocks, data += 64) {
    const __m128i abef = st0, cdgh = st1;
    __m128i m[4];
    for (int i = 0; i < 4; ++i)
      m[i] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * i)), kMask);
    for (int g = 0; g < 16; ++g) {  // 16 groups of 4 rounds; m[g % 4] holds W[4g .. 4g+3]
      __m128i msg = _mm_add_epi32(m[g & 3], _mm_loadu_si128(reinterpret_cast<const __m128i*>(kK + 4 * g)));
      st1 = _mm_sha256rnds2_epu32(st1, st0, msg);
      st0 = _mm_sha256rnds2_epu32(st0, st1, _mm_shuffle_epi32(msg, 0x0E));
      if (g < 12) {  // W[4g+16 .. 4g+19] = msg2(msg1(W[4g..], W[4g+4..]) + W[4g+9..4g+12], W[4g+12..])
        const __m128i w0 = m[g & 3], w1 = m[(g + 1) & 3], w2 = m[(g + 2) & 3], w3 = m[(g + 3) & 3];
        __m128i t = _mm_add_epi32(_mm_sha256msg1_epu32(w0, w1), _mm_alignr_epi8(w3, w2, 4));
        m[g & 3] = _mm_sha256msg2_epu32(t, w3);
      }
    }
    st0 = _mm_add_epi32(st0, abef);
    st1 = _mm_add_epi32(st1, cdgh);
  }
  tmp = _mm_shuffle_epi32(st0, 0x1B);
  st1 = _mm_shuffle_epi32(st1, 0xB1);
  st0 = _mm_blend_epi16(tmp, st1, 0xF0);
  st1 = _mm_alignr_epi8(st1, tmp, 8);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(state), st0);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(state + 4), st1);
}

bool use_shani() noexcept {
  static const bool on = [] {
    const char* force = std::getenv("KT2N_SHA256_PORTABLE");  // "1": portable path (comparisons, debugging)
    if (force && force[0] == '1') return false;
    __builtin_cpu_init();
    return __builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1");
  }();
  return on;
}
#endif

}  // namespace

void Sha256::reset() noexcept {
  h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  buf_len_ = 0;
  total_len_ = 0;
}

void Sha256::compress(const uint8_t* block) noexcept {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) w[i] = load_be32(block + 4 * i);
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = h + s1 + ch + kK[i] + w[i];
    const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h_[0] += a;
  h_[1] += b;
  h_[2] += c;
  h_[3] += d;
  h_[4] += e;
  h_[5] += f;
  h_[6] += g;
  h_[7] += h;
}

void Sha256::update(const void* data, size_t len) noexcept {
  const auto* p = static_cast<const uint8_t*>(data);
  total_len_ += len;
  if (buf_len_ > 0) {
    const size_t take = len < 64 - buf_len_ ? len : 64 - buf_len_;
    std::memcpy(buf_.data() + buf_len_, p, take);
    buf_len_ += take;
    p += take;
    len -= take;
    if (buf_len_ == 64) {
#if KT2N_SHA256_X86
      if (use_shani())
        compress_shani(h_.data(), buf_.data(), 1);
      else
#endif
        compress(buf_.data());
      buf_len_ = 0;
    }
  }
#if KT2N_SHA256_X86
  if (len >= 64 && use_shani()) {
    const size_t blocks = len / 64;
    compress_shani(h_.data(), p, blocks);
    p += blocks * 64;
    len -= blocks * 64;
  }
#endif
  while (len >= 64) {
    compress(p);
    p += 64;
    len -= 64;
  }
  if (len > 0) {
    std::memcpy(buf_.data(), p, len);
    buf_len_ = len;
  }
}

Sha256::Digest Sha256::finish() noexcept {
  const uint64_t bit_len = total_len_ * 8u;
  const uint8_t pad_first = 0x80;
  update(&pad_first, 1);
  const uint8_t zero = 0;
  while (buf_len_ != 56) update(&zero, 1);
  uint8_t len_be[8];
  for (int i = 0; i < 8; ++i) len_be[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
  update(len_be, 8);
  Digest out{};
  for (int i = 0; i < 8; ++i) {
    out[4 * i + 0] = static_cast<uint8_t>(h_[i] >> 24);
    out[4 * i + 1] = static_cast<uint8_t>(h_[i] >> 16);
    out[4 * i + 2] = static_cast<uint8_t>(h_[i] >> 8);
    out[4 * i + 3] = static_cast<uint8_t>(h_[i]);
  }
  reset();
  return out;
}

std::string Sha256::hexdigest() {
  const Digest d = finish();
  return to_hex(d.data(), d.size());
}

std::string to_hex(const uint8_t* data, size_t len) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(len * 2, '0');
  for (size_t i = 0; i < len; ++i) {
    out[2 * i] = kHex[data[i] >> 4];
    out[2 * i + 1] = kHex[data[i] & 0x0F];
  }
  return out;
}

std::string sha256_hex(const void* data, size_t len) {
  Sha256 h;
  h.update(data, len);
  return h.hexdigest();
}

bool is_sha256_hex(std::string_view s) noexcept {
  if (s.size() != 64) return false;
  for (char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

}  // namespace kt2n
