// C++ ports of the production host-audio post path (kittenml-api@ebcdd3f).
// See include/kt2n/post/post.h for the source map. Every comment "py:" quotes
// the line being ported.
#include "kt2n/post/post.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "json.h"

namespace kt2n::post {

// ============================================================ numpy-exact helpers

double pairwise_sum(const double* a, std::ptrdiff_t n) {
  // numpy/core/src/umath/loops_utils.h.src DOUBLE_pairwise_sum (PW_BLOCKSIZE 128).
  if (n < 8) {
    double res = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int k = 0; k < 8; ++k) r[k] = a[k];
    std::ptrdiff_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      r[0] += a[i + 0];
      r[1] += a[i + 1];
      r[2] += a[i + 2];
      r[3] += a[i + 3];
      r[4] += a[i + 4];
      r[5] += a[i + 5];
      r[6] += a[i + 6];
      r[7] += a[i + 7];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  std::ptrdiff_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

double frame_mean_square(const float* x, std::ptrdiff_t n) {
  // py: np.mean(np.square(block_f64), axis=1) == (0.0 + pairwise(sq)) / n
  // (verified on numpy 1.26.4: 0 mismatches in 958 frames, l10 lane log).
  thread_local std::vector<double> scratch;
  if (scratch.size() < static_cast<size_t>(n)) scratch.resize(static_cast<size_t>(n));
  double* buf = scratch.data();
  for (std::ptrdiff_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(x[i]);
    buf[i] = v * v;
  }
  return (0.0 + pairwise_sum(buf, n)) / static_cast<double>(n);
}

std::vector<float> linspace_f32(double start, double stop, std::ptrdiff_t num) {
  // numpy 1.26 function_base.linspace, then .astype(float32).
  std::vector<float> out;
  if (num <= 0) return out;
  out.resize(static_cast<size_t>(num));
  const std::ptrdiff_t div = num - 1;
  const double delta = stop - start;
  if (div > 0) {
    const double step = delta / static_cast<double>(div);
    if (step == 0.0) {
      for (std::ptrdiff_t i = 0; i < num; ++i) {
        double y = static_cast<double>(i) / static_cast<double>(div);
        y = y * delta;
        y += start;
        out[static_cast<size_t>(i)] = static_cast<float>(y);
      }
    } else {
      for (std::ptrdiff_t i = 0; i < num; ++i) {
        double y = static_cast<double>(i) * step;
        y += start;
        out[static_cast<size_t>(i)] = static_cast<float>(y);
      }
    }
    out[static_cast<size_t>(num - 1)] = static_cast<float>(stop);
  } else {
    double y = 0.0 * delta;
    y += start;
    out[0] = static_cast<float>(y);
  }
  return out;
}

double py_round(double v) { return std::nearbyint(v); }

double dbfs_to_amplitude(double dbfs) { return std::pow(10.0, dbfs / 20.0); }

namespace {

Pcm concat(const Pcm& a, Samples b) {
  Pcm out;
  out.reserve(a.size() + b.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return out;
}

void append(Pcm& out, Samples b) { out.insert(out.end(), b.begin(), b.end()); }

int64_t py_floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

}  // namespace

// ============================================================ LeadingSilenceTrimmer

LeadingSilenceTrimmer::LeadingSilenceTrimmer(const OnsetPolicy& p) {
  if (p.keep_ms < 0 || p.lookback_ms < 0 || p.fade_ms < 0)
    throw std::invalid_argument("onset trim windows cannot be negative");
  if (p.max_trim_ms <= 0)
    throw std::invalid_argument("a positive trim cap is required; disable it instead");
  if (p.soft_gate_dbfs > p.hard_gate_dbfs)
    throw std::invalid_argument("the soft gate must sit below the hard gate");
  const double sr = static_cast<double>(p.sample_rate);
  // py: self.frame = max(1, round(frame_ms * sample_rate / 1000.0))
  frame_ = std::max(1, static_cast<int>(py_round(p.frame_ms * sr / 1000.0)));
  hard_gate_ = dbfs_to_amplitude(p.hard_gate_dbfs);
  soft_gate_ = dbfs_to_amplitude(p.soft_gate_dbfs);
  keep_frames_ = static_cast<int>(
      py_floordiv(static_cast<int64_t>(py_round(p.keep_ms * sr / 1000.0)), frame_));
  lookback_frames_ = static_cast<int>(
      py_floordiv(static_cast<int64_t>(py_round(p.lookback_ms * sr / 1000.0)), frame_));
  max_trim_samples_ =
      py_floordiv(static_cast<int64_t>(py_round(p.max_trim_ms * sr / 1000.0)), frame_) * frame_;
  if (max_trim_samples_ <= 0) throw std::invalid_argument("the trim cap must be at least one frame");
  fade_samples_ = static_cast<int>(py_round(p.fade_ms * sr / 1000.0));
  sample_rate_ = p.sample_rate;
}

double LeadingSilenceTrimmer::trimmed_ms() const noexcept {
  // py: self.dropped_samples / self.sample_rate * 1000.0
  return static_cast<double>(dropped_) / static_cast<double>(sample_rate_) * 1000.0;
}

Pcm LeadingSilenceTrimmer::release(Pcm audio, int64_t fade_limit) {
  armed_ = false;
  pending_.clear();
  pending_.shrink_to_fit();
  held_ = 0;
  if (dropped_ != 0) {
    // _fade_in(audio, limit): span = min(fade_samples, max(0, limit), len(audio))
    const int64_t span = std::min<int64_t>(
        {static_cast<int64_t>(fade_samples_), std::max<int64_t>(0, fade_limit),
         static_cast<int64_t>(audio.size())});
    if (span > 0) {
      const std::vector<float> ramp = linspace_f32(0.0, 1.0, span);
      for (int64_t i = 0; i < span; ++i) audio[static_cast<size_t>(i)] *= ramp[static_cast<size_t>(i)];
    }
  }
  return audio;
}

Pcm LeadingSilenceTrimmer::feed(Samples chunk) {
  if (!armed_) return Pcm(chunk.begin(), chunk.end());
  if (chunk.empty() && pending_.empty()) return {};
  Pcm buffer = pending_.empty() ? Pcm(chunk.begin(), chunk.end()) : concat(pending_, chunk);
  const int64_t len = static_cast<int64_t>(buffer.size());
  const int64_t count = len / frame_;
  // levels = frame_rms(buffer, frame); loud = flatnonzero(levels >= hard_gate)
  std::vector<double> levels(static_cast<size_t>(count));
  int64_t loud = -1;
  for (int64_t f = 0; f < count; ++f) {
    levels[static_cast<size_t>(f)] = std::sqrt(frame_mean_square(buffer.data() + f * frame_, frame_));
    if (loud < 0 && levels[static_cast<size_t>(f)] >= hard_gate_) loud = f;
  }
  if (loud >= 0) {
    detected_ = true;
    int64_t index = loud;
    const int64_t floor_index = std::max<int64_t>(0, index - lookback_frames_);
    while (index > floor_index && levels[static_cast<size_t>(index - 1)] >= soft_gate_) --index;
    int64_t cut = std::max<int64_t>(0, (index - keep_frames_) * frame_);
    cut = std::min(cut, max_trim_samples_ - dropped_);
    cut = std::max<int64_t>(0, std::min(cut, len));
    dropped_ += cut;
    Pcm rest(buffer.begin() + cut, buffer.end());
    return release(std::move(rest), index * frame_ - cut);
  }
  // No onset yet: drop whole frames from the front, keep the retain window.
  const int64_t budget = max_trim_samples_ - dropped_;
  const int64_t retain = static_cast<int64_t>(keep_frames_ + lookback_frames_ + 1) * frame_;
  int64_t droppable = std::max<int64_t>(0, len - retain);
  droppable = std::min(droppable, budget);
  droppable = py_floordiv(droppable, frame_) * frame_;
  dropped_ += droppable;
  Pcm remainder(buffer.begin() + droppable, buffer.end());
  if (dropped_ >= max_trim_samples_) {
    const int64_t n = static_cast<int64_t>(remainder.size());
    return release(std::move(remainder), n);
  }
  held_ = static_cast<int64_t>(remainder.size());
  pending_ = std::move(remainder);
  return {};
}

Pcm LeadingSilenceTrimmer::flush() {
  if (!armed_) return {};
  if (pending_.empty() && dropped_ == 0) return {};
  Pcm buffer = std::move(pending_);
  const int64_t n = static_cast<int64_t>(buffer.size());
  return release(std::move(buffer), n);
}

// ============================================================ KeptTokenWithholder

KeptTokenWithholder::KeptTokenWithholder(LeadingSilenceTrimmer trimmer, double floor_ms,
                                         int sample_rate)
    : trimmer_(std::move(trimmer)), sample_rate_(sample_rate) {
  // py: int(round(max(0.0, floor_ms) * SAMPLE_RATE / 1000.0))
  floor_ = static_cast<int>(py_round(std::max(0.0, floor_ms) * sample_rate / 1000.0));
}

int64_t KeptTokenWithholder::budget() const noexcept {
  // py: max(0, kept * _N_SAMPLES_PER_TOKEN - self._floor) if kept > 0 else 0
  const int64_t per_token = sample_rate_ / 25;
  return kept_tokens_ > 0 ? std::max<int64_t>(0, kept_tokens_ * per_token - floor_) : 0;
}

double KeptTokenWithholder::trimmed_ms() const noexcept {
  // py: self.dropped_samples * 1000.0 / SAMPLE_RATE
  return static_cast<double>(dropped_samples()) * 1000.0 / static_cast<double>(sample_rate_);
}

Pcm KeptTokenWithholder::feed(Samples chunk) {
  if (!remaining_.has_value() && !chunk.empty()) remaining_ = budget();
  if (remaining_.has_value() && *remaining_ != 0) {
    const int64_t cut = std::min<int64_t>(*remaining_, static_cast<int64_t>(chunk.size()));
    *remaining_ -= cut;
    withheld_ += cut;
    chunk = chunk.subspan(static_cast<size_t>(cut));
  }
  return trimmer_.feed(chunk);
}

// ============================================================ SeamEdgeTrimmer

SeamEdgeTrimmer::SeamEdgeTrimmer(bool trim_head, const SeamPolicy& p) : trim_head_(trim_head) {
  const double sr = static_cast<double>(p.sample_rate);
  // py: max(1, int(sample_rate * frame_ms / 1000.0)) -- int() truncates
  frame_ = std::max(1, static_cast<int>(sr * p.frame_ms / 1000.0));
  gate_ = dbfs_to_amplitude(p.gate_dbfs);
  keep_ = static_cast<int>(sr * p.keep_ms / 1000.0);
  fade_ = std::max(1, static_cast<int>(sr * p.fade_ms / 1000.0));
}

Pcm SeamEdgeTrimmer::feed(Samples chunk) {
  if (finished_) throw std::logic_error("seam trimmer already finished");
  Pcm buffer = carry_.empty() ? Pcm(chunk.begin(), chunk.end()) : concat(carry_, chunk);
  const int64_t len = static_cast<int64_t>(buffer.size());
  const int64_t whole = (len / frame_) * frame_;
  carry_.assign(buffer.begin() + whole, buffer.end());
  if (whole == 0) return {};
  Pcm out;
  out.reserve(static_cast<size_t>(whole));
  const int64_t nframes = whole / frame_;
  for (int64_t f = 0; f < nframes; ++f) {
    const float* fp = buffer.data() + f * frame_;
    Samples frame(fp, static_cast<size_t>(frame_));
    // py: np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12) > gate
    const double level = std::sqrt(frame_mean_square(fp, frame_) + 1e-12);
    const bool loud = level > gate_;
    if (!loud_seen_) {
      if (!loud) {
        if (trim_head_) {
          held_.emplace_back(frame.begin(), frame.end());
        } else {
          append(out, frame);
        }
        continue;
      }
      loud_seen_ = true;
      if (trim_head_) {
        Pcm held;
        for (const Pcm& h : held_) append(held, h);
        const int64_t hl = static_cast<int64_t>(held.size());
        const int64_t lead_from = std::max<int64_t>(0, hl - keep_);
        head_trimmed_ += lead_from;  // len(held) - len(lead)
        held_.clear();
        Pcm joined(held.begin() + lead_from, held.end());
        append(joined, frame);
        // _fade_in: audio[:count] *= linspace(0, 1, fade)[:count]
        const int64_t cnt = std::min<int64_t>(fade_, static_cast<int64_t>(joined.size()));
        if (cnt > 0) {
          const std::vector<float> ramp = linspace_f32(0.0, 1.0, fade_);
          for (int64_t i = 0; i < cnt; ++i) joined[static_cast<size_t>(i)] *= ramp[static_cast<size_t>(i)];
        }
        append(out, joined);
        continue;
      }
      append(out, frame);
      continue;
    }
    if (loud) {
      for (const Pcm& h : held_) append(out, h);
      held_.clear();
      stretch_released_ = 0;
      append(out, frame);
    } else {
      held_.emplace_back(frame.begin(), frame.end());
    }
  }
  return out;
}

Pcm SeamEdgeTrimmer::finish(bool trim_tail) {
  finished_ = true;
  Pcm tail;
  for (const Pcm& h : held_) append(tail, h);
  append(tail, carry_);
  held_.clear();
  carry_.clear();
  if (!loud_seen_ || !trim_tail) return tail;
  Pcm kept(tail.begin(), tail.begin() + std::min<int64_t>(keep_, static_cast<int64_t>(tail.size())));
  tail_trimmed_ += static_cast<int64_t>(tail.size()) - static_cast<int64_t>(kept.size());
  const int64_t cnt = std::min<int64_t>(fade_, static_cast<int64_t>(kept.size()));
  if (cnt > 0) {
    // kept[len - count:] *= linspace(1, 0, fade)[fade - count:]
    const std::vector<float> ramp = linspace_f32(1.0, 0.0, fade_);
    const int64_t k0 = static_cast<int64_t>(kept.size()) - cnt;
    for (int64_t i = 0; i < cnt; ++i)
      kept[static_cast<size_t>(k0 + i)] *= ramp[static_cast<size_t>(fade_ - cnt + i)];
  }
  return kept;
}

Pcm SeamEdgeTrimmer::release(int64_t limit) {
  if (finished_ || !loud_seen_ || held_.empty()) return {};
  Pcm out;
  int64_t held = 0;
  for (const Pcm& h : held_) held += static_cast<int64_t>(h.size());
  while (!held_.empty() &&
         stretch_released_ + static_cast<int64_t>(held_.front().size()) <= limit &&
         held - static_cast<int64_t>(held_.front().size()) >= keep_) {
    const int64_t n = static_cast<int64_t>(held_.front().size());
    held -= n;
    stretch_released_ += n;
    released_ += n;
    append(out, held_.front());
    held_.erase(held_.begin());
  }
  return out;
}

Pcm SeamEdgeTrimmer::trim(Samples audio, bool trim_tail) {
  Pcm head = feed(audio);
  Pcm tail = finish(trim_tail);
  append(head, tail);
  return head;
}

bool seam_trim_applies(const SeamContext& c, int index) {
  const bool only_chunk = index == 0 && c.lane_known && c.lane_sealed && c.lane_jobs == 1;
  if (c.demo_request) return !only_chunk;
  if (c.scope == SeamScope::kAll) return true;
  if (!(c.scope == SeamScope::kTagged && c.demo_chunked)) return false;
  if (c.dc_mode == 0) return true;  // a tagged demo-chunked request: the release
  if (index < c.dc_release_chunks) return false;
  if (only_chunk) return false;
  return true;
}

bool seam_trim_head(const SeamContext& c, int index) {
  if (c.followup) return true;
  if (!c.demo_request) return index > 0;
  // q_demo_seam_head
  if (index > 0) return true;
  if (!c.lane_known) return false;
  return !c.lane_sealed || c.lane_jobs > 1;
}

Pcm trim_and_fade(Samples audio, int sample_rate) {
  Pcm values(audio.begin(), audio.end());
  const int64_t n = static_cast<int64_t>(values.size());
  if (n < static_cast<int64_t>(sample_rate * 0.05)) return values;
  const int hop = std::max(1, static_cast<int>(sample_rate * 0.01));
  const int64_t frames = n / hop;
  const double gate = dbfs_to_amplitude(-45.0);
  int64_t first = -1, last = -1;
  for (int64_t f = 0; f < frames; ++f) {
    const double rms = std::sqrt(frame_mean_square(values.data() + f * hop, hop) + 1e-12);
    if (rms > gate) {
      if (first < 0) first = f;
      last = f;
    }
  }
  if (first < 0) return values;
  const int64_t keep = static_cast<int64_t>(sample_rate * 0.02);
  const int64_t start = std::max<int64_t>(0, first * hop - keep);
  const int64_t end = std::min<int64_t>(n, (last + 1) * hop + keep);
  Pcm result(values.begin() + start, values.begin() + end);
  const int fade = std::max(1, static_cast<int>(sample_rate * 0.008));
  if (static_cast<int64_t>(result.size()) > static_cast<int64_t>(fade) * 2) {
    const std::vector<float> up = linspace_f32(0.0, 1.0, fade);
    const std::vector<float> down = linspace_f32(1.0, 0.0, fade);
    const size_t r0 = result.size() - static_cast<size_t>(fade);
    for (int i = 0; i < fade; ++i) result[static_cast<size_t>(i)] *= up[static_cast<size_t>(i)];
    for (int i = 0; i < fade; ++i) result[r0 + static_cast<size_t>(i)] *= down[static_cast<size_t>(i)];
  }
  return result;
}

int64_t chunk_gap_samples(double gap_seconds, int64_t credit, int sample_rate) {
  // py: max(0, int(SAMPLE_RATE * q_chunk_gap_seconds(control)) - q_credit)
  return std::max<int64_t>(0, static_cast<int64_t>(sample_rate * gap_seconds) - credit);
}

// ============================================================ startup hold

double adaptive_hold_seconds(int sharing, double base, double per_listener, double cap) {
  const double extra = per_listener * static_cast<double>(std::max(0, sharing - 1));
  return std::max(0.0, std::min(cap, base + extra));
}

StartupHold::StartupHold(const HoldPolicy& policy) : policy_(policy) {
  const bool pinned = policy_.pinned.has_value();
  const bool adaptive = pinned ? false : policy_.adaptive;
  const std::optional<double> alone = pinned ? std::nullopt : policy_.alone;
  // join_hold: 0 when pinned or no steady callable (decided per observation below)
  const double join_hold = pinned ? 0.0 : policy_.join_hold_ms / 1000.0;
  has_fn_ = adaptive || alone.has_value() || join_hold > 0.0;
  // startup_hold_samples: the fixed hold = the pinned value or the cap.
  const double seconds = pinned ? *policy_.pinned : policy_.cap;
  fixed_target_ = std::max<int64_t>(
      0, static_cast<int64_t>(py_round(policy_.sample_rate * std::max(0.0, seconds))));
  passthrough_ = !has_fn_ && fixed_target_ <= 0;
  target_ = has_fn_ ? 0 : fixed_target_;
}

int64_t StartupHold::target(const HoldObservation& obs, int call) const {
  const bool pinned = policy_.pinned.has_value();
  // startup_hold_policy(): a pinned hold is fixed with cap = max(0, pinned).
  const bool adaptive = pinned ? false : policy_.adaptive;
  const double cap = pinned ? std::max(0.0, *policy_.pinned) : policy_.cap;
  const double base = pinned ? 0.0 : policy_.base;
  const double policy_per_listener = pinned ? 0.0 : policy_.per_listener;
  const std::optional<double> skip_value = pinned ? std::nullopt : policy_.skip_per_listener;
  const std::optional<double> alone = pinned ? std::nullopt : policy_.alone;
  const double join_hold =
      (pinned || !obs.steady_available) ? 0.0 : policy_.join_hold_ms / 1000.0;
  const std::optional<double> alone_first = alone.has_value() ? policy_.alone_first : std::nullopt;
  const double first_max_trim_ms = alone_first.has_value() ? policy_.first_max_trim_ms : 0.0;

  const int listeners = std::max(1, obs.listeners);
  double seconds = 0.0;
  if (alone.has_value() && listeners == 1) {
    seconds = std::max(0.0, *alone);
    if (alone_first.has_value() && call == 1) {
      if (!obs.trimmed_ms_so_far.has_value() || *obs.trimmed_ms_so_far <= first_max_trim_ms)
        seconds = std::min(seconds, std::max(0.0, *alone_first));
    }
  } else if (adaptive) {
    double per_listener = policy_per_listener;
    // skip_hold_per_listener: value when the skip dropped >= skip_min_tokens tokens
    if (skip_value.has_value() && obs.opening_skipped_tokens >= policy_.skip_min_tokens)
      per_listener = std::max(per_listener, *skip_value);
    if (policy_.min_sharing != 0 && listeners < policy_.min_sharing) per_listener = 0.0;
    seconds = adaptive_hold_seconds(listeners, base, per_listener, cap);
  } else {
    seconds = std::max(0.0, cap);
  }
  if (join_hold > 0.0 && listeners > 1 && obs.steady_available) {
    if (obs.steady_joiner) seconds = std::max(seconds, join_hold);
  }
  return std::max<int64_t>(0, static_cast<int64_t>(py_round(policy_.sample_rate * seconds)));
}

Pcm StartupHold::feed(Samples chunk, const HoldObservation& obs) {
  if (finished_) throw std::logic_error("startup hold already finished");
  if (passthrough_ || released_) return Pcm(chunk.begin(), chunk.end());
  append(held_, chunk);
  if (has_fn_) {
    ++calls_;
    target_ = std::max(target_, target(obs, calls_));
    peak_sharing_ = std::max(peak_sharing_, std::max(1, obs.listeners));
  }
  if (static_cast<int64_t>(held_.size()) >= target_) {
    released_ = true;
    Pcm out = std::move(held_);
    held_.clear();
    return out;
  }
  return {};
}

Pcm StartupHold::finish() {
  finished_ = true;
  Pcm out = std::move(held_);
  held_.clear();
  return out;
}

// ============================================================ normalise / PCM / WAV

NormalizeStatus normalize_audio(Samples in, Pcm& out) {
  if (in.empty()) return NormalizeStatus::kEmpty;
  for (float v : in)
    if (!std::isfinite(v)) return NormalizeStatus::kNonFinite;
  out.resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    // np.clip(values, -1.0, 1.0) on float32
    float v = in[i];
    if (v < -1.0f) v = -1.0f;
    if (v > 1.0f) v = 1.0f;
    out[i] = v;
  }
  return NormalizeStatus::kOk;
}

namespace {
inline int16_t f32_to_i16_trunc(float v) {
  // numpy astype(int16) truncates toward zero; inputs are within [-32767, 32767]
  // after the clip. Out-of-range values (never produced) saturate here.
  if (!(v > -32768.0f)) return std::numeric_limits<int16_t>::min();
  if (!(v < 32767.0f)) return v >= 32767.0f ? int16_t{32767} : int16_t{0};
  return static_cast<int16_t>(v);
}
}  // namespace

void pcm16_from_float(Samples in, std::vector<int16_t>& out) {
  out.resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) out[i] = f32_to_i16_trunc(in[i] * 32767.0f);
}

void pcm16_bytes_from_float(Samples in, std::string& out) {
  const size_t base = out.size();
  out.resize(base + 2 * in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    const uint16_t v = static_cast<uint16_t>(f32_to_i16_trunc(in[i] * 32767.0f));
    out[base + 2 * i] = static_cast<char>(v & 0xFF);
    out[base + 2 * i + 1] = static_cast<char>(v >> 8);
  }
}

namespace {
void put_u32(std::string& s, uint32_t v) {
  for (int i = 0; i < 4; ++i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void put_u16(std::string& s, uint16_t v) {
  s.push_back(static_cast<char>(v & 0xFF));
  s.push_back(static_cast<char>(v >> 8));
}
void set_u32(std::string& s, size_t off, uint32_t v) {
  for (int i = 0; i < 4; ++i) s[off + static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
}
}  // namespace

std::string wav_header(uint32_t riff_size, uint32_t data_size, int sample_rate) {
  // py: struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", riff, b"WAVE", b"fmt ", 16, 1, 1,
  //                 SR, SR * 2, 2, 16, b"data", data)
  std::string h;
  h.reserve(kWavHeaderBytes);
  h.append("RIFF");
  put_u32(h, riff_size);
  h.append("WAVE");
  h.append("fmt ");
  put_u32(h, 16);
  put_u16(h, 1);
  put_u16(h, 1);
  put_u32(h, static_cast<uint32_t>(sample_rate));
  put_u32(h, static_cast<uint32_t>(sample_rate) * 2u);
  put_u16(h, 2);
  put_u16(h, 16);
  h.append("data");
  put_u32(h, data_size);
  return h;
}

std::string streaming_wav_header(int sample_rate) {
  return wav_header(kWavUnknownSize, kWavUnknownSize, sample_rate);
}

std::string complete_wav_header(uint64_t data_bytes, int sample_rate) {
  const uint64_t riff = data_bytes + kWavHeaderBytes - 8;
  if (riff > static_cast<uint64_t>(kWavUnknownSize) - 1)
    throw std::invalid_argument("WAV response exceeds the RIFF size limit");
  return wav_header(static_cast<uint32_t>(riff), static_cast<uint32_t>(data_bytes), sample_rate);
}

std::string finalized_wav_header(std::string_view header, uint64_t total_bytes) {
  if (header.size() != static_cast<size_t>(kWavHeaderBytes) ||
      total_bytes < static_cast<uint64_t>(kWavHeaderBytes) || header.substr(0, 4) != "RIFF" ||
      header.substr(8, 4) != "WAVE" || header.substr(36, 4) != "data")
    return std::string(header);
  if (total_bytes - 8 > 0xFFFFFFFFull) throw std::overflow_error("int too big to convert");
  std::string out(header);
  set_u32(out, 4, static_cast<uint32_t>(total_bytes - 8));
  set_u32(out, 40, static_cast<uint32_t>(total_bytes - kWavHeaderBytes));
  return out;
}

// ============================================================ peak normalisation

float demo_peak(Samples audio) {
  float peak = 0.0f;
  bool any = false;
  for (float v : audio) {
    const float a = std::fabs(v);
    if (!any || a > peak) peak = a;
    any = true;
  }
  return any ? peak : 0.0f;
}

void demo_int16_with_peak(Samples audio, float peak, std::vector<int16_t>& out) {
  out.assign(audio.size(), 0);
  if (!(peak > 0.0f)) return;
  for (size_t i = 0; i < audio.size(); ++i) {
    float v = audio[i] / peak;          // float32 / float32
    v = v * static_cast<float>(kDemoFullScale);  // float32 * 32767
    out[i] = f32_to_i16_trunc(v);
  }
}

bool is_lossy_format(std::string_view f) { return f == "mp3" || f == "aac" || f == "opus"; }

PeakGainTable PeakGainTable::parse(std::string_view text) {
  const json::Value root = json::parse(text);
  const json::Value* schema = root.get("schema");
  if (schema == nullptr || !schema->is_string() || schema->as_string() != "kittentts-peak-norm-gains.v1")
    throw std::invalid_argument("peak gains: schema != kittentts-peak-norm-gains.v1");
  const json::Value* voices = root.get("voices");
  if (voices == nullptr || !voices->is_object() || voices->members().empty())
    throw std::invalid_argument("peak gains: no voices");
  PeakGainTable t;
  for (const auto& [id, row] : voices->members()) {
    const json::Value* g = row.get("gain_db");
    if (g == nullptr || !g->is_number()) throw std::invalid_argument("peak gains: " + id + " has no gain_db");
    const double db = g->as_double();
    if (!(std::isfinite(db) && db >= 0.0 && db <= 60.0))
      throw std::invalid_argument("peak gains: " + id + " gain_db outside 0..60");
    t.voices.emplace(id, db);
  }
  const json::Value* d = root.get("default_gain_db");
  if (d == nullptr || !d->is_number()) throw std::invalid_argument("peak gains: no default_gain_db");
  t.default_gain_db = d->as_double();
  if (!(std::isfinite(t.default_gain_db) && t.default_gain_db >= 0.0 && t.default_gain_db <= 60.0))
    throw std::invalid_argument("peak gains: default_gain_db outside 0..60");
  const json::Value* aliases = root.get("aliases");
  if (aliases != nullptr && aliases->is_object()) {
    for (const auto& [alias, target] : aliases->members()) {
      if (!target.is_string() || t.voices.find(target.as_string()) == t.voices.end())
        throw std::invalid_argument("peak gains: alias " + alias + " names an unknown voice");
      t.aliases.emplace(alias, target.as_string());
    }
  }
  return t;
}

PeakGainTable PeakGainTable::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parse(text);
}

namespace {
std::string strip(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos) return {};
  const size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}
}  // namespace

double voice_gain_db(const PeakGainTable& table, const std::optional<std::string>& voice,
                     PeakStreamMode stream, std::string_view custom, std::string* source) {
  if (stream == PeakStreamMode::kVoice && voice.has_value()) {
    const std::string name = strip(*voice);
    std::string id;
    if (table.voices.count(name) != 0) {
      id = name;
    } else {
      const auto a = table.aliases.find(name);
      if (a != table.aliases.end()) id = a->second;
    }
    if (!id.empty()) {
      if (source != nullptr) *source = "voice:" + id;
      return table.voices.at(id);
    }
  }
  if (custom == "median") {
    if (source != nullptr) *source = "median";
    return table.default_gain_db;
  }
  if (custom == "3" || custom == "6" || custom == "12") {
    if (source != nullptr) *source = "custom:" + std::string(custom);
    return std::stod(std::string(custom));
  }
  throw std::invalid_argument("custom must be one of median, 3, 6, 12");
}

PeakNorm::PeakNorm(bool whole, double start_gain_db, double ceiling, double ms_per_db, int sample_rate)
    : whole_(whole),
      start_gain_db_(start_gain_db),
      ceiling_(ceiling),
      ms_per_db_(ms_per_db),
      sample_rate_(sample_rate) {
  if (!(ceiling > 0.0 && ceiling <= 1.0)) throw std::invalid_argument("ceiling outside (0, 1]");
  if (!(ms_per_db > 0.0)) throw std::invalid_argument("ms_per_db must be > 0");
  gain_ = std::pow(10.0, start_gain_db_ / 20.0);
  // py: math.log(10.0) / 20.0 / (self.ms_per_db * self.sample_rate / 1000.0)
  max_fall_per_sample_ =
      std::log(10.0) / 20.0 / (ms_per_db_ * static_cast<double>(sample_rate_) / 1000.0);
}

std::optional<PeakNorm> PeakNorm::for_request(const PeakGainTable& table,
                                              const std::optional<std::string>& voice, bool whole,
                                              std::string_view response_format, PeakStreamMode stream,
                                              std::string_view custom, double ms_per_db,
                                              int ceiling_permille, int lossy_ceiling_permille,
                                              int sample_rate) {
  if (!whole && stream == PeakStreamMode::kOff) return std::nullopt;
  const double gain_db = whole ? 0.0 : voice_gain_db(table, voice, stream, custom);
  const int permille = is_lossy_format(response_format) ? lossy_ceiling_permille : ceiling_permille;
  return PeakNorm(whole, gain_db, static_cast<double>(permille) / 1000.0, ms_per_db, sample_rate);
}

namespace {
double quiet_output() {
  // py: QUIET_OUTPUT = 10.0 ** (-50.0 / 20.0)
  static const double q = std::pow(10.0, -50.0 / 20.0);
  return q;
}

Pcm scaled_const(Samples x, double gain) {
  // py: y = (x.astype(np.float64) * gain).astype(np.float32); clip to [-1, 1]
  Pcm y(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    float v = static_cast<float>(static_cast<double>(x[i]) * gain);
    if (v < -1.0f) v = -1.0f;
    if (v > 1.0f) v = 1.0f;
    y[i] = v;
  }
  return y;
}
}  // namespace

std::vector<double> PeakNorm::fall_curve(const std::vector<double>& magnitude, double& slope) const {
  const size_t n = magnitude.size();
  const double log_gain = std::log(gain_);
  const double log_ceiling = std::log(ceiling_);
  std::vector<double> log_limit(n);
  for (size_t i = 0; i < n; ++i) {
    const double m = magnitude[i];
    double v = m > 0.0 ? log_ceiling - std::log(std::max(m, 1e-300))
                       : std::numeric_limits<double>::infinity();
    log_limit[i] = std::min(v, log_gain);  // np.minimum(log_limit, log_gain)
  }
  const double s = max_fall_per_sample_;
  // backward[i] = minimum.accumulate((log_limit + index * s)[::-1])[::-1] - index * s
  std::vector<double> backward(n);
  double run = std::numeric_limits<double>::infinity();
  for (size_t k = n; k-- > 0;) {
    const double t = log_limit[k] + static_cast<double>(k) * s;
    run = (k == n - 1) ? t : std::min(run, t);
    backward[k] = run;
  }
  for (size_t i = 0; i < n; ++i) backward[i] = backward[i] - static_cast<double>(i) * s;
  slope = s;
  if (backward[0] < log_gain - s) {
    double best = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i) {
      const double v = (log_gain - log_limit[i]) / (static_cast<double>(i) + 1.0);
      if (i == 0 || v > best) best = v;
    }
    slope = best;
    for (size_t i = 0; i < n; ++i) {
      const double line = log_gain - slope * (static_cast<double>(i) + 1.0);
      backward[i] = std::max(line, backward[i]);
    }
  }
  std::vector<double> gain(n);
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double c = std::min(backward[i], log_gain);
    acc = (i == 0) ? c : std::min(acc, c);
    gain[i] = std::exp(acc);
  }
  bool any_over = false;
  for (size_t i = 0; i < n; ++i)
    if (magnitude[i] * gain[i] > ceiling_) any_over = true;
  if (any_over) {
    double g = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double v = gain[i];
      if (magnitude[i] * gain[i] > ceiling_) v = ceiling_ / std::max(magnitude[i], 1e-300);
      g = (i == 0) ? v : std::min(g, v);
      gain[i] = g;
    }
  }
  return gain;
}

Pcm PeakNorm::process(Samples x) {
  ++chunks_;
  if (x.empty()) return {};
  std::vector<double> magnitude(x.size());
  double chunk_peak = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    magnitude[i] = static_cast<double>(std::fabs(x[i]));
    if (i == 0 || magnitude[i] > chunk_peak) chunk_peak = magnitude[i];
  }
  if (out_peak_ <= quiet_output()) {
    if (chunk_peak * gain_ > ceiling_) gain_ = ceiling_ / chunk_peak;
    peak_ = std::max(peak_, chunk_peak);
    out_peak_ = std::max(out_peak_, chunk_peak * gain_);
    return scaled_const(x, gain_);
  }
  peak_ = std::max(peak_, chunk_peak);
  if (chunk_peak * gain_ <= ceiling_) {
    out_peak_ = std::max(out_peak_, chunk_peak * gain_);
    return scaled_const(x, gain_);
  }
  double slope = 0.0;
  const std::vector<double> curve = fall_curve(magnitude, slope);
  const double slope_db_per_ms = slope * 20.0 / std::log(10.0) * sample_rate_ / 1000.0;
  if (slope > max_fall_per_sample_ * (1.0 + 1e-9)) {
    ++forced_;
    max_forced_db_per_ms_ = std::max(max_forced_db_per_ms_, slope_db_per_ms);
  }
  ++falls_;
  gain_ = curve.back();
  double m = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double v = magnitude[i] * curve[i];
    if (i == 0 || v > m) m = v;
  }
  out_peak_ = std::max(out_peak_, m);
  Pcm y(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    float v = static_cast<float>(static_cast<double>(x[i]) * curve[i]);
    if (v < -1.0f) v = -1.0f;
    if (v > 1.0f) v = 1.0f;
    y[i] = v;
  }
  return y;
}

NormalizeStatus level_publication(Samples publication, PeakNorm* peak, Pcm& out) {
  Pcm normalized;
  const NormalizeStatus st = normalize_audio(publication, normalized);
  if (st != NormalizeStatus::kOk) return st;
  out = (peak != nullptr && !peak->whole()) ? peak->process(normalized) : std::move(normalized);
  return NormalizeStatus::kOk;
}

void whole_file_lossy(Samples chunk, float peak, double ceiling, Pcm& out) {
  out.assign(chunk.size(), 0.0f);
  if (!(peak > 0.0f)) return;
  // py: (samples.astype(np.float64) * (float(ceiling) / float(peak))).astype(np.float32), clip
  const double factor = ceiling / static_cast<double>(peak);
  for (size_t i = 0; i < chunk.size(); ++i) {
    float v = static_cast<float>(static_cast<double>(chunk[i]) * factor);
    if (v < -1.0f) v = -1.0f;
    if (v > 1.0f) v = 1.0f;
    out[i] = v;
  }
}

void whole_file_flac(Samples chunk, float peak, Pcm& out) {
  std::vector<int16_t> i16;
  demo_int16_with_peak(chunk, peak, i16);
  out.resize(i16.size());
  for (size_t i = 0; i < i16.size(); ++i) out[i] = static_cast<float>(i16[i]) / 32768.0f;
}

}  // namespace kt2n::post
