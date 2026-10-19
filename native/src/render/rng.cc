// numpy-exact PCG64 + SeedSequence + float32 ziggurat (see include/kt2n/render/rng.h).
// Arithmetic notes (must match numpy's C build, x86-64 SSE, no FMA, -ffp-contract=off here):
//   * the wedge test compares a float expression against the DOUBLE exp(-0.5 * x * x)
//     (distributions.c:206-207 calls exp, not expf);
//   * the tail uses log1pf (npy_log1pf) on floats.
#include "kt2n/render/rng.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kt2n/sha256.h"

namespace kt2n::render {
namespace {

#include "ziggurat_f32.inc"

constexpr uint32_t kInitA = 0x43b0d7e5u;
constexpr uint32_t kMultA = 0x931e8875u;
constexpr uint32_t kInitB = 0x8b51f9ddu;
constexpr uint32_t kMultB = 0x58f38dedu;
constexpr uint32_t kMixMultL = 0xca01f9ddu;
constexpr uint32_t kMixMultR = 0x4973f715u;
constexpr uint32_t kXShift = 16;
constexpr int kPoolSize = 4;

inline uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
  value ^= hash_const;
  hash_const *= kMultA;
  value *= hash_const;
  value ^= value >> kXShift;
  return value;
}

inline uint32_t mix(uint32_t x, uint32_t y) {
  uint32_t result = kMixMultL * x - kMixMultR * y;
  result ^= result >> kXShift;
  return result;
}

constexpr u128 kPcgMult =
    (static_cast<u128>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;

inline uint64_t rotr64(uint64_t value, unsigned rot) {
  return (value >> rot) | (value << ((64u - rot) & 63u));
}

}  // namespace

void seed_sequence_generate_state(uint64_t entropy, uint32_t* out, int32_t n_words) {
  // _int_to_uint32_array: [0] for 0, else 32-bit words lowest first.
  uint32_t ent[2];
  int n_ent = 0;
  if (entropy == 0) {
    ent[n_ent++] = 0;
  } else {
    uint64_t n = entropy;
    while (n > 0) {
      ent[n_ent++] = static_cast<uint32_t>(n & 0xffffffffu);
      n >>= 32;
    }
  }
  // mix_entropy into a pool of 4.
  uint32_t pool[kPoolSize];
  uint32_t hash_const = kInitA;
  for (int i = 0; i < kPoolSize; ++i) pool[i] = hashmix(i < n_ent ? ent[i] : 0u, hash_const);
  for (int i_src = 0; i_src < kPoolSize; ++i_src)
    for (int i_dst = 0; i_dst < kPoolSize; ++i_dst)
      if (i_src != i_dst) pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], hash_const));
  for (int i_src = kPoolSize; i_src < n_ent; ++i_src)
    for (int i_dst = 0; i_dst < kPoolSize; ++i_dst) pool[i_dst] = mix(pool[i_dst], hashmix(ent[i_src], hash_const));
  // generate_state
  uint32_t hb = kInitB;
  for (int32_t i = 0; i < n_words; ++i) {
    uint32_t v = pool[i % kPoolSize];
    v ^= hb;
    hb *= kMultB;
    v *= hb;
    v ^= v >> kXShift;
    out[i] = v;
  }
}

Pcg64::Pcg64(uint64_t seed) {
  uint32_t w[8];
  seed_sequence_generate_state(seed, w, 8);
  // generate_state(4, np.uint64): little-endian pairs.
  uint64_t v[4];
  for (int i = 0; i < 4; ++i) v[i] = uint64_t(w[2 * i]) | (uint64_t(w[2 * i + 1]) << 32);
  const u128 initstate = (static_cast<u128>(v[0]) << 64) | v[1];
  const u128 initseq = (static_cast<u128>(v[2]) << 64) | v[3];
  // pcg_setseq_128_srandom_r
  state_ = 0;
  inc_ = (initseq << 1u) | 1u;
  state_ = state_ * kPcgMult + inc_;
  state_ += initstate;
  state_ = state_ * kPcgMult + inc_;
  has_uint32_ = 0;
  uinteger_ = 0;
}

uint64_t Pcg64::next64() {
  state_ = state_ * kPcgMult + inc_;
  const uint64_t hi = static_cast<uint64_t>(state_ >> 64);
  const uint64_t lo = static_cast<uint64_t>(state_);
  return rotr64(hi ^ lo, static_cast<unsigned>(state_ >> 122u));
}

uint32_t Pcg64::next32() {
  if (has_uint32_) {
    has_uint32_ = 0;
    return uinteger_;
  }
  const uint64_t next = next64();
  has_uint32_ = 1;
  uinteger_ = static_cast<uint32_t>(next >> 32);
  return static_cast<uint32_t>(next & 0xffffffffu);
}

float Pcg64::next_float() { return static_cast<float>(next32() >> 8) * (1.0f / 16777216.0f); }

float Pcg64::standard_normal_f() {
  for (;;) {
    const uint32_t r = next32();
    const int idx = static_cast<int>(r & 0xffu);
    const uint32_t sign = (r >> 8) & 0x1u;
    const uint32_t rabs = (r >> 9) & 0x0007fffffu;
    float x = static_cast<float>(rabs) * wi_float[idx];
    if (sign & 0x1u) x = -x;
    if (rabs < ki_float[idx]) return x;
    if (idx == 0) {
      for (;;) {
        const float xx = -ziggurat_nor_inv_r_f * std::log1p(-next_float());
        const float yy = -std::log1p(-next_float());
        if (yy + yy > xx * xx) return ((rabs >> 8) & 0x1u) ? -(ziggurat_nor_r_f + xx) : ziggurat_nor_r_f + xx;
      }
    } else {
      const float lhs = (fi_float[idx - 1] - fi_float[idx]) * next_float() + fi_float[idx];
      if (static_cast<double>(lhs) < std::exp(-0.5 * static_cast<double>(x) * static_cast<double>(x))) return x;
    }
  }
}

void Pcg64::fill_standard_normal_f(float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = standard_normal_f();
}

void Pcg64::skip_standard_normal_f(size_t n) {
  for (size_t i = 0; i < n; ++i) (void)standard_normal_f();
}

Pcg64State Pcg64::state() const {
  Pcg64State s{};
  s.state_hi = static_cast<uint64_t>(state_ >> 64);
  s.state_lo = static_cast<uint64_t>(state_);
  s.inc_hi = static_cast<uint64_t>(inc_ >> 64);
  s.inc_lo = static_cast<uint64_t>(inc_);
  s.has_uint32 = has_uint32_;
  s.uinteger = uinteger_;
  return s;
}

// ---- flow noise ------------------------------------------------------------------------------------
namespace {
void fill_rows(Pcg64& rng, int32_t width, float* out, int32_t stride, int32_t col0) {
  for (int32_t c = 0; c < kMel; ++c) rng.fill_standard_normal_f(out + size_t(c) * size_t(stride) + size_t(col0), size_t(width));
}
}  // namespace

void draw_voice_prompt_noise(int32_t prompt_frames, float* out, int32_t stride) {
  if (prompt_frames < 0 || stride < prompt_frames) throw std::invalid_argument("draw_voice_prompt_noise: bad stride");
  Pcg64 rng(kFlowNoiseSeedBase);
  fill_rows(rng, prompt_frames, out, stride, 0);
}

int32_t draw_row_target_noise(uint64_t seed, int32_t prompt_frames, int32_t budget, float* out, int32_t stride) {
  const int32_t w = target_noise_width(budget);
  if (prompt_frames < 0 || budget < 0 || w > stride)
    throw std::invalid_argument("draw_row_target_noise: W=" + std::to_string(w) + " > stride " +
                                std::to_string(stride));
  Pcg64 rng(kFlowNoiseSeedBase + seed);
  rng.skip_standard_normal_f(size_t(kMel) * size_t(prompt_frames));  // the discarded prompt draw
  fill_rows(rng, w, out, stride, 0);
  return w;
}

int32_t draw_row_noise_extension(uint64_t seed, int32_t budget, float* out, int32_t stride) {
  const int32_t have = target_noise_width(budget);
  const int32_t frames = noise_extension_frames(budget);
  if (frames <= have) return have;
  if (have > stride)
    throw std::invalid_argument("draw_row_noise_extension: W=" + std::to_string(have) + " > stride " +
                                std::to_string(stride));
  // Production draws (80, F - W) in one call: row-major, each channel's F - W values in turn. F = (2 x budget + 3)
  // x 2 passes the table (stride = 2 x kMaxRowTokens = 8,192 frames) for budgets of 2,047+ tokens, i.e. streamed
  // segments of 466+ characters (4.4 tokens per character, cap 2,341). A row can never read past the table: its
  // tokens are bounded by the LM context (4,096 incl. the prompt). So each channel keeps its first stride - W
  // values and the rest are drawn and discarded: every value kept is production's, at production's position.
  // (Before this, open_row refused such rows: RC1 quality take W.Q.qset:C02, l0-quality 22:12.)
  const int32_t keep = std::min(frames, stride) - have;
  Pcg64 rng(kFlowNoiseExtSeedBase + seed);
  for (int32_t c = 0; c < kMel; ++c) {
    rng.fill_standard_normal_f(out + size_t(c) * size_t(stride) + size_t(have), size_t(keep));
    rng.skip_standard_normal_f(size_t(frames - have - keep));
  }
  return have + keep;
}

uint64_t request_seed_salt(std::string_view request_id) {
  Sha256 h;
  h.update(request_id.data(), request_id.size());
  const Sha256::Digest d = h.finish();
  const uint32_t be = (uint32_t(d[0]) << 24) | (uint32_t(d[1]) << 16) | (uint32_t(d[2]) << 8) | uint32_t(d[3]);
  return uint64_t(be & 0x3FFFFFFFu) * kSeedSaltUnit;
}

}  // namespace kt2n::render
