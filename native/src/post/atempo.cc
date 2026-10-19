// Adapted from FFmpeg libavfilter/af_atempo.c through a prior Python port.
// Upstream copyright (c) 2012 Pavel Koshevoy; LGPL-2.1-or-later.
// See native/NOTICE and native/licenses/FFmpeg-LGPL-2.1.txt.
// KittenML adaptation, 2026: mono C++ streaming, float64 time-domain
// correlation, staged speeds and exact-end handling. "py:" marks ported lines.
#include "kt2n/post/atempo.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kt2n::post {

namespace {
constexpr double kPi = 3.141592653589793;  // math.pi

int64_t trunc_to_int(double v) { return static_cast<int64_t>(v); }  // C cast / int(): toward zero
}  // namespace

ATempo::ATempo(double tempo, int sample_rate) : tempo_(tempo) {
  if (!(tempo > 0.0 && tempo < 1000.0) || !std::isfinite(tempo))
    throw std::invalid_argument("atempo: bad tempo");
  int window = sample_rate / 24;
  int nlevels = 0;  // av_log2(window) == window.bit_length() - 1
  for (int w = window; w > 1; w >>= 1) ++nlevels;
  const int pot = 1 << nlevels;
  if (pot < window) window = pot * 2;
  window_ = window;
  half_ = window / 2;
  ring_ = window * 3;
  hann_.resize(static_cast<size_t>(window));
  const double two_pi = 2.0 * kPi;
  for (int i = 0; i < window; ++i) {
    // py: t = arange(window) / float(window - 1)
    //     (0.5 * (1.0 - np.cos(2.0 * math.pi * t))).astype(float32).astype(float64)
    const double t = static_cast<double>(i) / static_cast<double>(window - 1);
    const double h = 0.5 * (1.0 - std::cos(two_pi * t));
    hann_[static_cast<size_t>(i)] = static_cast<double>(static_cast<float>(h));
  }
  // yae_clear: the first fragment starts half a window early.
  frags_[0].pos0 = -half_;
  frags_[0].pos1 = -half_;
  frags_[1].pos0 = 0;
  frags_[1].pos1 = 0;
}

bool ATempo::load_frag(bool flushing) {
  Fragment& frag = curr();
  const int64_t window = window_;
  const int64_t stop_here = frag.pos0 + window;
  if (!flushing && stop_here > loaded_) {
    loaded_ = std::min(stop_here, avail_);
    if (loaded_ < stop_here) return false;
  }
  const int64_t missing = stop_here > loaded_ ? stop_here - loaded_ : 0;
  const int64_t nsamples = missing < window ? window - missing : 0;
  frag.nsamples = nsamples;
  frag.has_wdata = false;
  const int64_t start = loaded_ - std::min<int64_t>(loaded_, ring_);
  int64_t zeros = 0;
  if (frag.pos0 < start) zeros = std::min(start - frag.pos0, nsamples);
  const int64_t lo = frag.pos0 + zeros - base_;
  frag.data.assign(static_cast<size_t>(window), 0.0);
  if (zeros < nsamples) {
    const int64_t count = nsamples - zeros;
    if (lo < 0 || lo + count > static_cast<int64_t>(buf_.size()))
      throw std::logic_error("atempo: fragment outside the input history");
    std::copy(buf_.begin() + lo, buf_.begin() + lo + count, frag.data.begin() + zeros);
  }
  return true;
}

int64_t ATempo::align(const Fragment& frag, const Fragment& prv, int64_t drift) const {
  const int64_t window = window_;
  const int64_t half = half_;
  const int64_t delta_max = half;
  const int64_t i0 = std::min<int64_t>(std::max<int64_t>(half - delta_max - drift, 0), window);
  const int64_t i1 = std::max<int64_t>(std::min<int64_t>(half + delta_max - drift, window - window / 16), 0);
  if (i1 <= i0) return -drift;
  // full[i + window - 1] = sum_m prev[m + i] * frag[m], m in [0, window - i)
  const double* p = prv.data.data();
  const double* f = frag.data.data();
  double best = 0.0;
  int64_t best_i = i0;
  for (int64_t i = i0; i < i1; ++i) {
    const int64_t n = window - i;
    // Blocked like a 16-wide BLAS ddot (4 x 4 partial sums), then the tail.
    double acc[16] = {0.0};
    int64_t m = 0;
    const int64_t n16 = n & ~int64_t{15};
    for (; m < n16; m += 16) {
      for (int k = 0; k < 16; ++k) acc[k] += p[m + i + k] * f[m + k];
    }
    double s = 0.0;
    {
      double a0[4], a1[4];
      for (int k = 0; k < 4; ++k) {
        a0[k] = acc[k] + acc[4 + k];
        a1[k] = acc[8 + k] + acc[12 + k];
      }
      double a[4];
      for (int k = 0; k < 4; ++k) a[k] = a0[k] + a1[k];
      const double lo2 = a[0] + a[2];
      const double hi2 = a[1] + a[3];
      s = lo2 + hi2;
    }
    for (; m < n; ++m) s += p[m + i] * f[m];
    // py: weights = (drift + i) * (i - i0) * (i1 - i)
    const double di = static_cast<double>(i);
    const double w = ((static_cast<double>(drift) + di) * (di - static_cast<double>(i0))) *
                     (static_cast<double>(i1) - di);
    const double metric = s * w;
    if (i == i0 || metric > best) {  // first maximum (argmax)
      best = metric;
      best_i = i;
    }
  }
  return best_i - half;
}

int64_t ATempo::adjust_position() {
  Fragment& prv = prev();
  Fragment& frag = curr();
  // origin is (0, 0)
  const double prev_output_position = static_cast<double>(prv.pos1 + half_) * tempo_;
  const double ideal_output_position = static_cast<double>(prv.pos0 + half_);
  const int64_t drift = trunc_to_int(prev_output_position - ideal_output_position);
  const int64_t correction = align(frag, prv, drift);
  if (correction != 0) {
    frag.pos0 -= correction;
    frag.nsamples = 0;
  }
  return correction;
}

void ATempo::overlap_add(std::vector<double>& out) {
  Fragment& prv = prev();
  Fragment& frag = curr();
  const int64_t start_here = std::max(out_pos_, frag.pos1);
  const int64_t stop_here = std::min(prv.pos1 + prv.nsamples, frag.pos1 + frag.nsamples);
  const int64_t overlap = stop_here - start_here;
  if (overlap <= 0) return;
  const int64_t ia = start_here - prv.pos1;
  const int64_t ib = start_here - frag.pos1;
  auto weight = [this](Fragment& fr) {
    if (!fr.has_wdata) {
      fr.wdata.resize(fr.data.size());
      for (size_t k = 0; k < fr.data.size(); ++k) fr.wdata[k] = fr.data[k] * hann_[k];
      fr.has_wdata = true;
    }
  };
  weight(prv);
  weight(frag);
  const size_t base = out.size();
  out.resize(base + static_cast<size_t>(overlap));
  for (int64_t k = 0; k < overlap; ++k)
    out[base + static_cast<size_t>(k)] = prv.wdata[static_cast<size_t>(ia + k)] + frag.wdata[static_cast<size_t>(ib + k)];
  if (frag.pos0 < 0) {
    const int64_t raw = std::min(-frag.pos0, overlap);
    for (int64_t k = 0; k < raw; ++k) out[base + static_cast<size_t>(k)] = prv.data[static_cast<size_t>(ia + k)];
  }
  out_pos_ = stop_here;
}

void ATempo::advance() {
  const double fragment_step = tempo_ * static_cast<double>(half_);
  ++nfrag_;
  Fragment& prv = prev();
  Fragment& frag = curr();
  frag.pos0 = prv.pos0 + trunc_to_int(fragment_step);
  frag.pos1 = prv.pos1 + half_;
  frag.nsamples = 0;
}

void ATempo::apply(std::vector<double>& out) {
  while (true) {
    if (state_ == State::kLoad) {
      if (!load_frag(false)) break;
      if (nfrag_ == 0) {
        advance();
        continue;
      }
      state_ = State::kAdjust;
    }
    if (state_ == State::kAdjust) state_ = adjust_position() != 0 ? State::kReload : State::kOverlapAdd;
    if (state_ == State::kReload) {
      if (!load_frag(false)) break;
      state_ = State::kOverlapAdd;
    }
    if (state_ == State::kOverlapAdd) {
      overlap_add(out);
      advance();
      state_ = State::kLoad;
    }
  }
}

void ATempo::trim_history() {
  const int64_t keep_from = std::max(base_, loaded_ - ring_ - window_);
  if (keep_from > base_) {
    buf_.erase(buf_.begin(), buf_.begin() + (keep_from - base_));
    base_ = keep_from;
  }
}

Pcm ATempo::emit(const std::vector<double>& out) {
  Pcm r(out.size());
  for (size_t i = 0; i < out.size(); ++i) r[i] = static_cast<float>(out[i]);
  return r;
}

Pcm ATempo::feed(Samples chunk) {
  if (flushed_) throw std::logic_error("atempo: feed after flush");
  if (chunk.empty()) return {};
  for (float v : chunk) buf_.push_back(static_cast<double>(v));
  avail_ += static_cast<int64_t>(chunk.size());
  std::vector<double> out;
  apply(out);
  trim_history();
  return emit(out);
}

Pcm ATempo::flush() {
  if (flushed_) return {};
  flushed_ = true;
  state_ = State::kFlush;
  std::vector<double> out;
  if (nfrag_ == 0) {
    // Less than half a window in total: passes unchanged.
    out.assign(buf_.begin(), buf_.end());
    return emit(out);
  }
  const int64_t half = half_;
  for (int pass = 0; pass < 1000000; ++pass) {
    Fragment& frag = curr();
    if (loaded_ == frag.pos0 + frag.nsamples && out_pos_ == frag.pos1 + frag.nsamples) break;
    if (frag.pos0 + frag.nsamples < loaded_) {
      load_frag(true);
      if (adjust_position() != 0) load_frag(true);
    }
    Fragment& f2 = curr();
    const int64_t overlap_end = f2.pos1 + std::min(half, f2.nsamples);
    while (out_pos_ < overlap_end) {
      const size_t before = out.size();
      overlap_add(out);
      if (out.size() == before) break;
    }
    if (f2.pos0 + f2.nsamples < loaded_) {
      advance();
      continue;
    }
    const int64_t start_here = std::max(out_pos_, overlap_end);
    const int64_t stop_here = f2.pos1 + f2.nsamples;
    if (stop_here > start_here) {
      const int64_t offset = start_here - f2.pos1;
      out.insert(out.end(), f2.data.begin() + offset, f2.data.begin() + f2.nsamples);
      out_pos_ = stop_here;
    }
    break;
  }
  buf_.clear();
  return emit(out);
}

std::vector<double> stage_tempos(double speed) {
  if (!std::isfinite(speed) || speed <= 0.0) throw std::invalid_argument("speed must be positive");
  const int stages = std::max(1, static_cast<int>(std::ceil(std::fabs(std::log2(speed)) - 1e-12)));
  const double tempo = std::pow(speed, 1.0 / static_cast<double>(stages));
  return std::vector<double>(static_cast<size_t>(stages), tempo);
}

SpeedStretcher::SpeedStretcher(double speed, int sample_rate, bool exact_end)
    : speed_(speed), tempos_(stage_tempos(speed)), exact_end_(exact_end) {
  for (double t : tempos_) stages_.emplace_back(t, sample_rate);
}

Pcm SpeedStretcher::feed(Samples chunk) {
  if (chunk.empty()) return {};
  samples_in_ += static_cast<int64_t>(chunk.size());
  Pcm out(chunk.begin(), chunk.end());
  for (ATempo& stage : stages_) {
    out = stage.feed(out);
    if (out.empty()) break;
  }
  samples_out_ += static_cast<int64_t>(out.size());
  return out;
}

Pcm SpeedStretcher::flush() {
  if (flushed_) return {};
  flushed_ = true;
  Pcm out;
  if (exact_end_ && samples_in_ != 0) {
    const int window = stages_[0].window();
    const double max_tempo = *std::max_element(tempos_.begin(), tempos_.end());
    const int64_t pad_len = 3 * static_cast<int64_t>(window) * static_cast<int64_t>(stages_.size()) *
                            static_cast<int64_t>(std::ceil(max_tempo));
    Pcm piece(static_cast<size_t>(pad_len), 0.0f);
    for (ATempo& stage : stages_) {
      piece = stage.feed(piece);
      if (piece.empty()) break;
    }
    out.insert(out.end(), piece.begin(), piece.end());
  }
  Pcm carry;
  for (ATempo& stage : stages_) {
    Pcm head = carry.empty() ? Pcm{} : stage.feed(carry);
    Pcm tail = stage.flush();
    head.insert(head.end(), tail.begin(), tail.end());
    carry = std::move(head);
  }
  out.insert(out.end(), carry.begin(), carry.end());
  if (exact_end_) {
    // py: want = max(0, int(round(self.samples_in / self.speed)) - self.samples_out)
    const int64_t want = std::max<int64_t>(
        0, static_cast<int64_t>(py_round(static_cast<double>(samples_in_) / speed_)) - samples_out_);
    out.resize(static_cast<size_t>(want), 0.0f);
  }
  samples_out_ += static_cast<int64_t>(out.size());
  return out;
}

}  // namespace kt2n::post
