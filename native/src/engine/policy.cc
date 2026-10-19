// Pure scheduling and playback decisions; see policy.h for the contracts.
#include "policy.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "kt2n/sha256.h"

namespace kt2n::engine::policy {

Levers live_levers(const std::string& app) {
  Levers lv;
  if (app == "fq") {
    lv.complete_slots = 6;
    lv.dw_runway_ms = 500.0;
    lv.dw_max_wait_ms = 3000.0;
  }
  return lv;
}

// ---------------------------------------------------------------- opening skip
namespace {
// Opening-silence candidates; IDs 1466 and 3653 are excluded by opening_quiet_id.
constexpr int kSubGate[] = {
    331,  729,  735,  736,  770,  979,  1458, 1463, 1465, 1466, 1490, 1701, 1704, 1788, 1789, 1809,
    1947, 1950, 2005, 2028, 2031, 2109, 2112, 2139, 2933, 2948, 2957, 2960, 3003, 3246, 3247, 3408,
    3653, 3662, 3676, 3677, 3680, 3685, 3704, 3713, 3729, 3732, 3888, 3891, 3894, 3945, 3969, 3972,
    3975, 3976, 4053, 4056, 4057, 4134, 4137, 4138, 4215, 4216, 4218, 4219, 4227, 4296, 4299, 4300,
    4308, 5109, 5190, 5352, 5891, 5900, 5919, 6078, 6081, 6082, 6159, 6162, 6163, 6168, 6243, 6252,
    6321, 6324, 6325, 6333, 6399, 6402, 6405, 6406, 6414, 6483, 6486, 6487, 6495};
static_assert(sizeof(kSubGate) / sizeof(kSubGate[0]) == 93, "SUB_GATE_AUDIO_IDS has 93 ids");
}  // namespace

bool opening_quiet_id(int v) {
  if (v == 1466 || v == 3653) return false;
  return std::binary_search(std::begin(kSubGate), std::end(kSubGate), v);
}

void opening_admit(OpeningSkip& s, bool armed, int32_t value, std::vector<int32_t>& audio_tokens,
                   const Levers& lv) {
  // the skip applies only while closed is unset, on segment 0, with no audio token yet.
  if (s.closed || !armed || !lv.skip_opening || !audio_tokens.empty()) {
    s.closed = true;
    audio_tokens.push_back(value);
    return;
  }
  const int held = int(s.skipped.size() + s.pending.size());
  const int cap = lv.skip_cap;
  if (opening_quiet_id(value) && held < cap) {
    s.skipped.insert(s.skipped.end(), s.pending.begin(), s.pending.end());
    s.pending.clear();
    s.skipped.push_back(value);
    return;
  }
  if (held < cap && int(s.pending.size()) + 1 < lv.skip_run) {  // no strict-after cutoff
    s.pending.push_back(value);
    return;
  }
  s.closed = true;
  s.skipped_tokens = int(s.skipped.size());
  if (lv.skip_keep > 0 && !s.skipped.empty()) {
    const int keep = std::min<int>(lv.skip_keep, int(s.skipped.size()));
    audio_tokens.insert(audio_tokens.end(), s.skipped.end() - keep, s.skipped.end());
    s.kept_tokens = keep;
  }
  audio_tokens.insert(audio_tokens.end(), s.pending.begin(), s.pending.end());
  s.pending.clear();
  audio_tokens.push_back(value);
}

void opening_restore(OpeningSkip& s, std::vector<int32_t>& audio_tokens) {
  if (!audio_tokens.empty() || (s.skipped.empty() && s.pending.empty())) return;
  audio_tokens.insert(audio_tokens.end(), s.skipped.begin(), s.skipped.end());
  audio_tokens.insert(audio_tokens.end(), s.pending.begin(), s.pending.end());
  s.skipped.clear();
  s.pending.clear();
}

// ---------------------------------------------------------------- commit sizing
int schedule_width(const std::vector<int>& schedule, int commit_index) {
  if (schedule.empty()) return 5;
  return schedule[size_t(std::min<int>(commit_index, int(schedule.size()) - 1))];
}

CommitState new_commit_state(const Levers& lv, bool demo_held) {
  CommitState c;
  c.held = demo_held;
  c.next = demo_held ? (1 << 30) : schedule_width(lv.helper_schedule, 0);  //
  return c;
}

bool commit_ready(const CommitState& c, int token_count) {
  if (c.inflight || c.held) return false;
  return token_count >= c.next + kLookahead;
}

bool d_prepare(CommitState& c, int token_count, int sharing, const Levers& lv) {
  if (c.d_decided || c.held || token_count < lv.d_decide_at_tokens) return false;
  c.d_decided = true;
  c.alone = sharing == 1;
  if (c.alone && c.commit_index == 0 && c.committed == 0) c.next = lv.alone_schedule.front();
  return true;
}

void after_commit(CommitState& c, const Levers& lv) {
  c.committed = c.next;  // ids[:next+3] rendered, look-ahead cropped: committed = next
  c.commit_index += 1;
  c.next = c.committed + schedule_width(lv.helper_schedule, c.commit_index);   //
  if (c.alone) c.next = c.committed + schedule_width(lv.alone_schedule, c.commit_index);  //
}

// ---------------------------------------------------------------- startup sizes
double adaptive_hold_seconds(int sharing, const Levers& lv) {
  const double s = lv.hold_base + lv.hold_per_listener * double(std::max(1, sharing) - 1);
  return std::max(0.0, std::min(lv.hold_cap, s));
}

double start_target_seconds(int sharing_rows, bool realtime_phrase, const Levers& lv) {
  const double hold = realtime_phrase ? 0.0 : adaptive_hold_seconds(sharing_rows, lv);
  const double reserve = 0.0;  // STREAM_INITIAL_BUFFER_SECONDS=0 live
  return std::max({0.0, hold, reserve});
}

// ---------------------------------------------------------------- T9
T9Result t9_acoustic_rows(const std::vector<T9Row>& rows, const Levers& lv) {
  T9Result r;
  if (!lv.acoustic_deferral) {
    for (int i = 0; i < int(rows.size()); ++i) r.selected.push_back(i);
    return r;
  }
  std::vector<bool> sel(rows.size(), false);
  std::vector<int> normal;
  for (int i = 0; i < int(rows.size()); ++i) {
    const T9Row& row = rows[size_t(i)];
    if (row.finished) {
      sel[size_t(i)] = true;
      r.trigger = true;
      continue;
    }
    if (row.held_short) continue;
    if (row.segment_index == 0 && (!row.preview_ready || row.published_s < row.start_target_s)) {
      r.starting = true;
      sel[size_t(i)] = true;
      if (lv.acoustic_lazy && row.row_ready) r.trigger = true;
      continue;
    }
    std::optional<double> est = row.reservoir_s;
    if (est && lv.t9_true_reservoir) *est -= row.dropped_s;
    if (!est || *est < lv.defer_margin_s) {
      sel[size_t(i)] = true;
      r.trigger = true;
    } else {
      normal.push_back(i);
    }
  }
  if (r.starting && lv.defer_starting) {
    r.deferred = int(normal.size());
    r.deferred_rows = normal;
  } else if (lv.acoustic_lazy) {
    int ready = 0;
    for (int i : normal) ready += rows[size_t(i)].row_ready ? 1 : 0;
    if (ready >= lv.lazy_min_batch) r.trigger = true;
    for (int i : normal) {
      if (r.trigger || rows[size_t(i)].backlog >= lv.lazy_backlog_tokens) {
        sel[size_t(i)] = true;
      } else {
        r.deferred += 1;
        r.deferred_rows.push_back(i);
      }
    }
  } else {
    for (int i : normal) sel[size_t(i)] = true;  // ACOUSTIC_TURN_ROWS=0: all
  }
  for (int i = 0; i < int(rows.size()); ++i)
    if (sel[size_t(i)]) r.selected.push_back(i);
  return r;
}

// ---------------------------------------------------------------- preview
bool preview_due(bool preview_ready, int token_count, int committed, bool held_short, int next_commit,
                 int commit_index, const Levers& lv) {
  if (!lv.preview_enabled || preview_ready || held_short || committed != 0) return false;
  if (token_count < lv.preview_tokens) return false;
  const bool first = commit_index == 0 && committed == 0;
  if (first && next_commit + kLookahead >= lv.preview_tokens) return false;  // q_first_no_preview
  return true;
}

// ---------------------------------------------------------------- groups
int64_t geometry_owner(int prompt_tokens, int prompt_frames, int embedding_width) {
  return (int64_t(prompt_tokens) << 40) | (int64_t(prompt_frames) << 16) | int64_t(embedding_width);
}

std::pair<int, double> d_buffer_key(const std::vector<std::optional<double>>& buffers_ms, const Levers& lv) {
  double lowest = INFINITY;
  bool first = false;
  for (const auto& b : buffers_ms) {
    if (!b)
      first = true;
    else
      lowest = std::min(lowest, *b);
  }
  if (lowest < lv.urgency_ms) return {0, lowest};
  if (first) return {1, 0.0};
  return {2, lowest};
}

std::vector<Group> plan_commit_groups(const std::vector<CommitRow>& ready, const std::vector<CommitRow>& partners,
                                      const Levers& lv, int* held_rows) {
  // 1. Group. Insertion order of a Python dict = first appearance.
  std::vector<Group> groups;
  auto key_of = [&](const CommitRow& r) {
    GroupKey k;
    if (lv.ragged_groups) {
      k.kind = r.first ? 0 : 1;
    } else {
      k.kind = 2;
      k.geometry = r.geometry;
    }
    return k;
  };
  for (const CommitRow& r : ready) {
    const GroupKey k = key_of(r);
    auto it = std::find_if(groups.begin(), groups.end(), [&](const Group& g) { return g.key == k; });
    if (it == groups.end()) {
      groups.push_back(Group{k, {}});
      it = groups.end() - 1;
    }
    it->rows.push_back(r.id);
  }
  // 2. First-commit grouping: a ragged first group above the chunk_first slots is cut.
  if (lv.ragged_groups && lv.chunk_first_slots > 0) {
    std::vector<Group> cut;
    for (Group& g : groups) {
      if (g.key.kind != 0 || int(g.rows.size()) <= lv.chunk_first_slots) {
        cut.push_back(std::move(g));
        continue;
      }
      for (size_t start = 0, part = 0; start < g.rows.size(); start += size_t(lv.chunk_first_slots), ++part) {
        Group p{g.key, {}};
        p.key.part = int(part);
        const size_t end = std::min(g.rows.size(), start + size_t(lv.chunk_first_slots));
        p.rows.assign(g.rows.begin() + long(start), g.rows.begin() + long(end));
        cut.push_back(std::move(p));
      }
    }
    groups = std::move(cut);
  }
  // (_bs_split_chunk_overflow is PyTorch chunk-cache capacity, not a decision: native renders
  //  any width from paged K/V; the rows of the turn are the same.)
  // 3. Coalescing. Partner keys use the geometry owner.
  int held = 0;
  if (lv.coalesce_tokens > 0 && !groups.empty()) {
    struct Soon {
      GroupKey key;
      int needed;
    };
    std::vector<Soon> soonest;
    for (const CommitRow& p : partners) {
      const int needed = p.next + kLookahead - p.token_count;
      if (!(needed > 0 && needed <= lv.coalesce_tokens)) continue;
      GroupKey k;
      k.kind = 2;
      k.geometry = p.geometry;
      auto it = std::find_if(soonest.begin(), soonest.end(), [&](const Soon& s) { return s.key == k; });
      if (it == soonest.end())
        soonest.push_back({k, needed});
      else
        it->needed = std::min(it->needed, needed);
    }
    std::vector<Group> kept;
    for (Group& g : groups) {
      std::optional<int> partner;
      for (const Soon& s : soonest)
        if (s.key == g.key) partner = s.needed;
      bool all_wait = true;
      for (int id : g.rows) {
        const CommitRow& r = *std::find_if(ready.begin(), ready.end(), [&](const CommitRow& x) { return x.id == id; });
        bool may = r.commit_index >= 1 && partner.has_value();
        if (may) {
          const int overdue = r.token_count - (r.next + kLookahead);
          may = overdue + *partner <= lv.coalesce_tokens && r.reservoir_s.has_value() &&
                *r.reservoir_s >= lv.coalesce_min_reservoir_s;
        }
        if (!may) {
          all_wait = false;
          break;
        }
      }
      if (all_wait) {
        held += int(g.rows.size());
        continue;
      }
      kept.push_back(std::move(g));
    }
    groups = std::move(kept);
  }
  if (held_rows) *held_rows = held;
  // 4. Order: sorted(groups.items()) then a stable urgency sort.
  std::stable_sort(groups.begin(), groups.end(), [](const Group& a, const Group& b) { return a.key < b.key; });
  if (groups.size() > 1 && lv.urgency_order) {
    std::vector<std::pair<int, double>> keys;
    std::vector<std::pair<std::pair<int, double>, Group>> tagged;
    for (Group& g : groups) {
      std::vector<std::optional<double>> b;
      for (int id : g.rows)
        b.push_back(std::find_if(ready.begin(), ready.end(), [&](const CommitRow& x) { return x.id == id; })->d_buffer_ms);
      tagged.push_back({d_buffer_key(b, lv), std::move(g)});
    }
    std::stable_sort(tagged.begin(), tagged.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    groups.clear();
    for (auto& t : tagged) groups.push_back(std::move(t.second));
  }
  return groups;
}

std::vector<Group> plan_commit_groups_xsched(const std::vector<CommitRow>& ready, const Levers& lv) {
  constexpr double kMarginMs = 60.0;
  auto deadline = [](const CommitRow& r) { return r.d_buffer_ms ? *r.d_buffer_ms - kMarginMs : -INFINITY; };
  std::vector<CommitRow> urgent, rest;
  for (const CommitRow& r : ready) (r.commit_index <= 1 ? urgent : rest).push_back(r);
  auto by_deadline = [&](const CommitRow& a, const CommitRow& b) { return deadline(a) < deadline(b); };
  std::stable_sort(urgent.begin(), urgent.end(), by_deadline);
  std::stable_sort(rest.begin(), rest.end(), by_deadline);
  std::vector<Group> out;
  auto emit = [&](const std::vector<CommitRow>& rows, int kind) {
    if (rows.empty()) return;
    // Batches of at most chunk_first_slots rows, in deadline order.
    const size_t cap = size_t(std::max(1, lv.chunk_first_slots));
    for (size_t start = 0, part = 0; start < rows.size(); start += cap, ++part) {
      Group g;
      g.key.kind = kind;
      g.key.part = int(part);
      for (size_t i = start; i < std::min(rows.size(), start + cap); ++i) g.rows.push_back(rows[i].id);
      out.push_back(std::move(g));
    }
  };
  emit(urgent, 0);
  emit(rest, 1);
  return out;
}

bool launch_async(int active_rows_this_turn, const Levers& lv) {
  // every row's commit_index >= BS_ASYNC_FROM_COMMIT (0): always true.
  return lv.async_steady && active_rows_this_turn >= lv.async_min_rows;
}

std::vector<std::vector<int>> final_flush_buckets(std::vector<std::pair<int, int>> rows, const Levers& lv) {
  std::stable_sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  std::vector<std::vector<int>> buckets;
  std::vector<int> cur;
  int smallest = 0;
  for (const auto& [id, width] : rows) {
    if (!cur.empty() && (int(cur.size()) >= lv.final_bucket_rows || double(width) > smallest * lv.final_bucket_ratio)) {
      buckets.push_back(std::move(cur));
      cur.clear();
    }
    if (cur.empty()) smallest = width;
    cur.push_back(id);
  }
  if (!cur.empty()) buckets.push_back(std::move(cur));
  return buckets;
}

// ---------------------------------------------------------------- DeliveryClock
void DeliveryClock::play(double now, int64_t samples) {
  const double seconds = double(std::max<int64_t>(0, samples)) / kSampleRate;
  if (!play_end_) {
    first_at_ = now;
    play_end_ = now + seconds;
    return;
  }
  if (*play_end_ < now) {
    underruns_ += 1;
    underrun_ms_ += (now - *play_end_) * 1000.0;
    play_end_ = now;
  }
  *play_end_ += seconds;
}
void DeliveryClock::publish(double now, int64_t samples) {
  published_ += std::max<int64_t>(0, samples);
  if (held_) {
    pending_ += std::max<int64_t>(0, samples);
    return;
  }
  play(now, samples);
}
void DeliveryClock::hold() {
  if (!play_end_) held_ = true;
}
void DeliveryClock::release(double now) {
  if (!held_) return;
  held_ = false;
  const int64_t p = pending_;
  pending_ = 0;
  if (p) play(now, p);
}
std::optional<double> DeliveryClock::buffer_ms(double now) const {
  if (!play_end_) return std::nullopt;
  return (*play_end_ - now) * 1000.0;
}

// ---------------------------------------------------------------- DW gate
DwReason dw_due(const DwInput& in, const Levers& lv) {
  if (!lv.dw_gate) return DwReason::kIdle;
  if (!in.oldest_queued_s) return DwReason::kNone;
  if (!in.any_realtime && !in.any_resting) return DwReason::kIdle;
  const double waited = in.now_s - *in.oldest_queued_s;
  if (waited >= lv.dw_hard_s) return DwReason::kHard;
  if (in.waiting) return DwReason::kNone;
  const double low = in.low_buffer_ms.value_or(0.0);
  const bool gap_ok = !in.last_whole_render_s || (in.now_s - *in.last_whole_render_s) * 1000.0 >= lv.dw_min_gap_ms;
  if (gap_ok && low >= lv.dw_runway_ms) return DwReason::kRunway;
  if (waited * 1000.0 >= lv.dw_max_wait_ms && low >= lv.dw_max_wait_low_ms) return DwReason::kMaxWait;
  return DwReason::kNone;
}
const char* dw_reason_name(DwReason r) {
  switch (r) {
    case DwReason::kIdle: return "idle";
    case DwReason::kHard: return "hard";
    case DwReason::kRunway: return "runway";
    case DwReason::kMaxWait: return "max_wait";
    default: return "none";
  }
}

// ---------------------------------------------------------------- clone interleave
bool cf_interleave_due(bool any_active, bool any_before_first_audio, double waited_ms,
                       std::optional<double> since_last_build_ms, const Levers& lv) {
  if (!lv.cf_interleave) return false;
  if (!any_active) return true;
  if (waited_ms >= lv.cf_max_wait_ms) {
    // CC_INTERLEAVE_FORCED_GAP 250 ms (live) == the min gap.
    return !(since_last_build_ms && *since_last_build_ms < lv.cf_min_gap_ms);
  }
  if (any_before_first_audio) return false;
  if (since_last_build_ms && *since_last_build_ms < lv.cf_min_gap_ms) return false;
  return true;
}

// ---------------------------------------------------------------- admission take
std::vector<int> admission_take(const std::vector<LaneView>& lanes_in, int capacity, size_t* cursor,
                                const Levers& lv) {
  std::vector<int> taken;
  if (capacity <= 0 || lanes_in.empty()) return taken;
  std::vector<LaneView> lanes = lanes_in;
  const size_t n = lanes.size();
  size_t c = *cursor % n;
  bool progress = true;
  while (int(taken.size()) < capacity && progress) {
    progress = false;
    for (size_t k = 0; k < n; ++k) {
      const size_t i = c;  // popleft + append: the lane at the front is examined, then rotated
      c = (c + 1) % n;
      if (lanes[i].has_job && lanes[i].inflight < lv.lane_inflight) {
        taken.push_back(int(i));
        lanes[i].inflight += 1;
        lanes[i].has_job = false;  // the caller re-offers the lane's next job on the next take
        progress = true;
        if (int(taken.size()) >= capacity) break;
      }
    }
  }
  *cursor = c;
  return taken;
}

std::vector<int> admission_take_capacity(const std::vector<LaneView>& lanes, const std::vector<bool>& continuing,
                                         int capacity, size_t* cursor_cont, size_t* cursor, const Levers& lv) {
  std::vector<LaneView> cont = lanes, fresh = lanes;
  for (size_t i = 0; i < lanes.size(); ++i) (i < continuing.size() && continuing[i] ? fresh : cont)[i].has_job = false;
  std::vector<int> taken = admission_take(cont, capacity, cursor_cont, lv);
  const auto more = admission_take(fresh, capacity - int(taken.size()), cursor, lv);
  taken.insert(taken.end(), more.begin(), more.end());
  return taken;
}

// ---------------------------------------------------------------- prefetch
bool segment_prefetch_due(bool last_segment, bool segment_done, int64_t received_samples, size_t segment_chars,
                          int inflight_segments, const Levers& lv) {
  if (!lv.prefetch || last_segment) return false;
  if (!segment_done) {
    const double expected = double(segment_chars) / std::max(lv.prefetch_chars_per_s, 1e-6) * kSampleRate;
    if (double(received_samples) < lv.prefetch_fraction * expected) return false;
  }
  return inflight_segments + 1 + 1 <= lv.request_batch_size;  // SEGMENT_PREFETCH_FREE_ROWS=1
}

// ---------------------------------------------------------------- seeds
uint64_t request_seed_salt(const std::string& request_id) {
  kt2n::Sha256 h;
  h.update(request_id.data(), request_id.size());
  const std::string hex = h.hexdigest();
  const uint32_t be = uint32_t(std::stoul(hex.substr(0, 8), nullptr, 16));
  return uint64_t(be & 0x3FFFFFFFu) * 1024ull;
}

}  // namespace kt2n::engine::policy
