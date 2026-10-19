// Ragged row descriptors for one render call (include/kt2n/render/call.h).
#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "kt2n/render/call.h"
#include "kt2n/render/rng.h"

namespace kt2n::render {

const char* build_status_name(BuildStatus s) {
  switch (s) {
    case BuildStatus::kOk: return "ok";
    case BuildStatus::kNoPages: return "no_pages";
    case BuildStatus::kTooLong: return "too_long";
    case BuildStatus::kBadInput: return "bad_input";
  }
  return "?";
}

int32_t suffix_bucket_for(Phase call_phase, int32_t max_suffix) {
  int32_t b = -1;
  if (call_phase == Phase::kFirstCommit) b = bucket_of(kFirstSuffixBuckets, max_suffix);
  else if (call_phase == Phase::kSteadyCommit || call_phase == Phase::kFinalFlush)
    b = bucket_of(kSteadySuffixBuckets, max_suffix);
  return b > 0 ? b : whole_bucket(max_suffix);
}

BuildStatus build_call(Phase call_phase, const RowRender* rows, int32_t n_rows, HostPagePool& pool, CallPlan& out) {
  out = CallPlan{};
  out.phase = call_phase;
  if (n_rows <= 0 || n_rows > kRowBuckets[sizeof(kRowBuckets) / sizeof(kRowBuckets[0]) - 1]) return BuildStatus::kBadInput;
  const bool scratch_call = call_phase == Phase::kWholeRender || call_phase == Phase::kPreview;
  out.n_rows = n_rows;
  out.rows_bucket = bucket_of(kRowBuckets, n_rows);
  out.flow_steps = call_phase == Phase::kPreview ? 1 : kFlowSteps;
  out.min_commit_index = 1 << 30;

  // Pass 1: geometry, validation, page needs.
  struct RowPlan {
    int32_t past, pt, tok_suffix, fr_keys, n_ids, n_pages;
  };
  std::vector<RowPlan> rp(static_cast<size_t>(n_rows));
  int32_t max_suffix = 0;
  int32_t scratch_need = 0;
  for (int32_t i = 0; i < n_rows; ++i) {
    const RowRender& r = rows[i];
    const RenderGeometry& g = r.geom;
    if (r.voice == nullptr || r.slot < 0) return BuildStatus::kBadInput;
    const bool row_phase_ok = g.phase == call_phase ||
                              (call_phase == Phase::kSteadyCommit && g.phase == Phase::kFirstCommit);
    if (!row_phase_ok) return BuildStatus::kBadInput;
    if (!scratch_call && r.pages == nullptr) return BuildStatus::kBadInput;
    RowPlan& p = rp[size_t(i)];
    p.pt = r.voice->prompt_tokens;
    if (int32_t(r.voice->prompt_ids.size()) != p.pt || r.voice->prompt_frames != kMelPerToken * p.pt)
      return BuildStatus::kBadInput;
    p.past = scratch_call ? 0 : r.pages->tok_past();
    p.n_ids = g.render_tokens - g.silence_tokens;
    if (p.n_ids < 0 || (p.n_ids > 0 && r.ids == nullptr)) return BuildStatus::kBadInput;
    if (p.past > p.pt + p.n_ids) return BuildStatus::kBadInput;
    if (p.pt + g.render_tokens > kMaxRowTokens) return BuildStatus::kTooLong;
    // A noise table shorter than the row is not an error: production pads the missing target
    // frames with zeros (incremental_s3gen.py _flow); the gather kernel does the same past noise_w.
    p.tok_suffix = p.pt + g.render_tokens - p.past;
    p.fr_keys = kMelPerToken * (p.past + p.tok_suffix);
    p.n_pages = pages_for_frames(p.fr_keys);
    max_suffix = std::max(max_suffix, p.tok_suffix);
    if (scratch_call) scratch_need += p.n_pages;
    out.min_commit_index = std::min(out.min_commit_index, g.commit_index);
  }
  out.suffix_bucket = suffix_bucket_for(call_phase, max_suffix);
  int32_t max_keys = 0;
  for (const RowPlan& p : rp) max_keys = std::max(max_keys, p.fr_keys);
  out.keys_bucket = keys_bucket_for(max_keys);

  // Pass 2: pages (all-or-nothing for scratch; streaming rows keep what they already grew).
  if (scratch_call) {
    out.scratch_pages.resize(size_t(scratch_need));
    if (!pool.alloc(scratch_need, out.scratch_pages.data())) {
      out.scratch_pages.clear();
      return BuildStatus::kNoPages;
    }
  } else {
    int32_t need = 0;
    for (int32_t i = 0; i < n_rows; ++i)
      need += std::max(0, rp[size_t(i)].n_pages - int32_t(rows[i].pages->pages().size()));
    if (need > pool.pages_free()) return BuildStatus::kNoPages;
    for (int32_t i = 0; i < n_rows; ++i)
      if (!rows[i].pages->ensure_frames(pool, rp[size_t(i)].fr_keys)) return BuildStatus::kNoPages;
  }

  // Pass 3: descriptors.
  out.rows.assign(size_t(out.rows_bucket), RowDesc{});
  for (int32_t i = n_rows; i < out.rows_bucket; ++i) out.rows[size_t(i)].slot = -1;
  out.new_tok_past.resize(size_t(n_rows));
  int32_t tok_off = 0, pcm_off = 0, page_off = 0, scratch_off = 0;
  for (int32_t i = 0; i < n_rows; ++i) {
    const RowRender& r = rows[i];
    const RenderGeometry& g = r.geom;
    const RowPlan& p = rp[size_t(i)];
    RowDesc& d = out.rows[size_t(i)];
    d.slot = r.slot;
    d.voice = r.voice->voice;
    d.phase = uint16_t(g.phase);
    uint16_t flags = r.extra_flags;
    if (g.finalize) flags |= kRowFinalize;
    if (call_phase == Phase::kWholeRender) flags |= kRowFullAttention;
    if (call_phase == Phase::kPreview) flags |= kRowNoPoolWrite;
    if (g.tail_published) flags |= kRowFirstCommitTail;
    if (g.crossfade_head) flags |= kRowSeamCrossfade;
    if (g.trim_fade_head) flags |= kRowTrimFade;
    d.flags = flags;
    d.lookahead = int16_t(g.lookahead);
    d.flow_steps = int16_t(out.flow_steps);

    d.tok_past = p.past;
    d.tok_suffix = p.tok_suffix;
    d.tok_keys = p.past + p.tok_suffix;
    d.tok_off = tok_off;
    d.fr_past = kMelPerToken * p.past;
    d.fr_suffix = kMelPerToken * p.tok_suffix;
    d.fr_keys = p.fr_keys;
    d.fr_off = kMelPerToken * tok_off;
    d.prompt_frames = r.voice->prompt_frames;

    d.out_fr_begin = kMelPerToken * (p.pt + g.committed_before);
    d.out_fr_count = g.new_frames;
    d.pcm_off = pcm_off;
    d.pcm_count = g.pcm_samples;
    d.publish_skip = g.publish_skip;

    d.page_off = page_off;
    d.n_pages = p.n_pages;
    d.n_shared_pages = scratch_call ? 0 : r.pages->n_shared();

    d.noise_w = r.noise_w;
    d.committed_tokens = g.committed_before;
    d.src_cache = g.src_cache_in;
    d.mel_cache = g.mel_cache_in;
    d.tail_fade = (g.tail_published || g.publish_skip > 0) ? kTailFadeSamples : 0;
    d.t_mid = 0.5f;  // Q_MF_TMID_PERMILLE=500
    d.seed = r.seed;
    d.philox_seed = call_phase == Phase::kPreview ? preview_seed(rows[0].seed) : hift_seed(r.seed, g.committed_before);
    d.philox_offset = r.philox_offset;
    d.reserved[0] = g.commit_index;  // the row's commit index at launch (HiFT tail rules key on commit 0)

    // Packed suffix tokens.
    if (p.past < p.pt) {
      out.tokens.insert(out.tokens.end(), r.voice->prompt_ids.begin() + p.past, r.voice->prompt_ids.end());
      out.tokens.insert(out.tokens.end(), r.ids, r.ids + p.n_ids);
    } else {
      out.tokens.insert(out.tokens.end(), r.ids + (p.past - p.pt), r.ids + p.n_ids);
    }
    for (int32_t s = 0; s < g.silence_tokens; ++s) out.tokens.push_back(kSilenceToken);

    // Page table.
    if (scratch_call) {
      out.page_table.insert(out.page_table.end(), out.scratch_pages.begin() + scratch_off,
                            out.scratch_pages.begin() + scratch_off + p.n_pages);
      scratch_off += p.n_pages;
    } else {
      const auto& pg = r.pages->pages();
      out.page_table.insert(out.page_table.end(), pg.begin(), pg.begin() + p.n_pages);
      if (r.pages->needs_prefix_copy()) out.prefix_copy_rows.push_back(i);
    }

    const int32_t end_abs = p.pt + g.committed_after;
    out.new_tok_past[size_t(i)] =
        (g.finalize || scratch_call) ? p.past : std::max(p.past, end_abs / kTokenChunk * kTokenChunk);

    tok_off += p.tok_suffix;
    pcm_off += g.pcm_samples;
    page_off += p.n_pages;
  }
  out.total_tokens = tok_off;
  out.total_frames = kMelPerToken * tok_off;
  // Estimator packing: 16-aligned per row (see CallPlan::est_rows).
  out.est_rows = out.rows;
  int32_t est_off = 0;
  for (int32_t i = 0; i < n_rows; ++i) {
    out.est_rows[size_t(i)].fr_off = est_off;
    est_off += (out.rows[size_t(i)].fr_suffix + kEstRowAlign - 1) / kEstRowAlign * kEstRowAlign;
  }
  out.est_total_frames = est_off;
  out.total_pcm = pcm_off;
  if (int32_t(out.tokens.size()) != tok_off) throw std::logic_error("build_call: token packing mismatch");
  return BuildStatus::kOk;
}

bool merge_plans(const CallPlan* const* parts, int32_t n_parts, CallPlan& out) {
  out = CallPlan{};
  if (n_parts <= 0) return false;
  Phase phase = parts[0]->phase;
  int32_t n = 0;
  for (int32_t k = 0; k < n_parts; ++k) {
    const Phase q = parts[k]->phase;
    if (q == Phase::kWholeRender || q == Phase::kPreview || !parts[k]->scratch_pages.empty()) return false;
    if (q != phase) {
      // Commit phases mix (a final is a commit whose rows carry kRowFinalize): the graph key's phase is canonical
      // (graph_phase: first / steady / final share one graph) and no kernel or stage reads the call's phase at run
      // time, only per-row flags, so a mixed call is a steady call (its suffix buckets).
      auto commit = [](Phase x) {
        return x == Phase::kFirstCommit || x == Phase::kSteadyCommit || x == Phase::kFinalFlush;
      };
      if (!commit(q) || !commit(phase)) return false;
      phase = Phase::kSteadyCommit;
    }
    if (parts[k]->flow_steps != parts[0]->flow_steps) return false;
    n += parts[k]->n_rows;
  }
  if (n > kRowBuckets[sizeof(kRowBuckets) / sizeof(kRowBuckets[0]) - 1]) return false;
  out.phase = phase;
  out.n_rows = n;
  out.rows_bucket = bucket_of(kRowBuckets, n);
  out.flow_steps = parts[0]->flow_steps;
  out.min_commit_index = 1 << 30;
  out.rows.assign(size_t(out.rows_bucket), RowDesc{});
  for (int32_t i = n; i < out.rows_bucket; ++i) out.rows[size_t(i)].slot = -1;
  int32_t row = 0, tok_off = 0, pcm_off = 0, page_off = 0, max_suffix = 0, max_keys = 0;
  for (int32_t k = 0; k < n_parts; ++k) {
    const CallPlan& p = *parts[k];
    out.min_commit_index = std::min(out.min_commit_index, p.min_commit_index);
    for (int32_t i = 0; i < p.n_rows; ++i) {
      RowDesc d = p.rows[size_t(i)];
      d.tok_off += tok_off;
      d.fr_off = kMelPerToken * d.tok_off;
      d.pcm_off += pcm_off;
      d.page_off += page_off;
      max_suffix = std::max(max_suffix, d.tok_suffix);
      max_keys = std::max(max_keys, d.fr_keys);
      out.rows[size_t(row + i)] = d;
    }
    for (int32_t i : p.prefix_copy_rows) out.prefix_copy_rows.push_back(row + i);
    out.tokens.insert(out.tokens.end(), p.tokens.begin(), p.tokens.end());
    out.page_table.insert(out.page_table.end(), p.page_table.begin(), p.page_table.end());
    out.new_tok_past.insert(out.new_tok_past.end(), p.new_tok_past.begin(), p.new_tok_past.end());
    row += p.n_rows;
    tok_off += p.total_tokens;
    pcm_off += p.total_pcm;
    page_off += int32_t(p.page_table.size());
  }
  out.suffix_bucket = suffix_bucket_for(phase, max_suffix);
  out.keys_bucket = keys_bucket_for(max_keys);
  out.total_tokens = tok_off;
  out.total_frames = kMelPerToken * tok_off;
  out.total_pcm = pcm_off;
  out.est_rows = out.rows;  // same 16-aligned estimator packing as build_call
  int32_t est_off = 0;
  for (int32_t i = 0; i < n; ++i) {
    out.est_rows[size_t(i)].fr_off = est_off;
    est_off += (out.rows[size_t(i)].fr_suffix + kEstRowAlign - 1) / kEstRowAlign * kEstRowAlign;
  }
  out.est_total_frames = est_off;
  if (int32_t(out.tokens.size()) != tok_off) return false;
  return true;
}

void apply_call_state(const CallPlan& plan, RowRender* rows, int32_t n_rows) {
  if (plan.phase == Phase::kWholeRender || plan.phase == Phase::kPreview) return;
  for (int32_t i = 0; i < n_rows && i < int32_t(plan.new_tok_past.size()); ++i) {
    if (rows[i].pages == nullptr) continue;
    rows[i].pages->set_tok_past(plan.new_tok_past[size_t(i)]);
  }
  for (int32_t i : plan.prefix_copy_rows) rows[i].pages->mark_prefix_copied();
}

}  // namespace kt2n::render

namespace kt2n::render {

namespace {
// One row's contribution to a call's graph key: its suffix tokens, key frames and commit index.
struct RowShape {
  int32_t s, k, ci;
  bool operator<(const RowShape& o) const { return std::tie(s, k, ci) < std::tie(o.s, o.k, o.ci); }
};
}  // namespace

std::set<GraphKey> enumerate_graph_keys(const VoicePrefix& voice, const CommitPlanConfig& commit, int32_t max_rows,
                                        int32_t row_tokens) {
  PoolView pv{};
  pv.n_pages = 1 << 16;
  HostPagePool pool(pv);
  VoicePrefix vp = voice;
  if (!vp.built && vp.prompt_tokens > 0) {
    // Not built yet (L8 builds a voice's prefix on its first use): the geometry the build will give
    // (service.cu build_voice_prefix), so a voice can be prewarmed before its prefix exists.
    vp.stable_tokens = stable_prefix_tokens(vp.prompt_tokens);
    const int32_t fr = kMelPerToken * vp.stable_tokens;
    vp.shared_pages.assign(size_t(fr / kPageFrames), 0);
    vp.partial_page = (fr % kPageFrames) != 0 ? 0 : -1;
    vp.built = vp.stable_tokens > 0;
  }
  if (vp.built) {  // stand-in page numbers on the bookkeeping pool
    std::vector<int32_t> fake(vp.shared_pages.size());
    if (!fake.empty()) pool.alloc(int32_t(fake.size()), fake.data());
    vp.shared_pages = fake;
    if (vp.partial_page >= 0) pool.alloc(1, &vp.partial_page);
  }
  // A row's final can come well after the commit it could have taken (coalescing holds, a commit still in
  // flight while the LM keeps decoding: ~27 tokens at N=20 during a 44 ms bulk render), so finals are
  // enumerated up to this many tokens past the state's own next boundary.
  constexpr int32_t kLateFinalTokens = 64;
  const int32_t max_ids = row_tokens + kLateFinalTokens + kLookahead + 8;
  std::vector<int32_t> ids(size_t(max_ids), 0);

  // 1. Every row shape one row can take, replayed through BOTH schedules (alone and shared: a row decided
  //    shared can still be the only row of its call, and an alone row can share a call), on the bookkeeping pool.
  std::set<RowShape> first_rows, steady_rows, final_rows, whole_rows;
  for (const bool alone : {true, false}) {
    CommitPlanConfig cpc = commit;
    RowCommitState st(&cpc);
    st.decide_alone(cpc.decide_at_tokens, alone ? 1 : 2);
    RowPages pages;
    pages.admit(pool, &vp);
    RowRender row;
    row.slot = 0;
    row.voice = &vp;
    row.ids = ids.data();
    row.pages = &pages;
    row.noise_w = target_noise_width(row_tokens);
    row.extra_flags = kRowLookaheadPadZero;
    CallPlan plan;
    for (;;) {
      const bool more = st.next_commit_tokens() + kLookahead <= row_tokens;
      const int32_t hi = std::min(row_tokens, st.next_commit_tokens() + kLookahead + kLateFinalTokens);
      for (int32_t n = std::max(1, st.committed_tokens() + 1); n <= hi; ++n) {
        row.geom = st.plan_final(n);
        if (build_call(Phase::kFinalFlush, &row, 1, pool, plan) == BuildStatus::kOk)
          final_rows.insert(RowShape{plan.rows[0].tok_suffix, plan.rows[0].fr_keys, st.commit_index()});
      }
      if (!more) break;
      row.geom = st.plan_commit();
      const Phase cp = st.commit_index() == 0 ? Phase::kFirstCommit : Phase::kSteadyCommit;
      if (build_call(cp, &row, 1, pool, plan) != BuildStatus::kOk) break;
      (cp == Phase::kFirstCommit ? first_rows : steady_rows)
          .insert(RowShape{plan.rows[0].tok_suffix, plan.rows[0].fr_keys, st.commit_index()});
      apply_call_state(plan, &row, 1);
      st.commit_launched(row.geom);
    }
    pages.release(pool);
  }
  for (int32_t n = 1; n <= row_tokens; ++n) {
    RowRender row;
    row.slot = 0;
    row.voice = &vp;
    row.ids = ids.data();
    row.noise_w = target_noise_width(row_tokens);
    row.geom = plan_whole_render(n);
    CallPlan plan;
    if (build_call(Phase::kWholeRender, &row, 1, pool, plan) != BuildStatus::kOk) continue;
    whole_rows.insert(RowShape{plan.rows[0].tok_suffix, plan.rows[0].fr_keys, 0});
    pool.release(plan.scratch_pages.data(), int32_t(plan.scratch_pages.size()));
  }

  // 2. Calls. A call's key takes the max suffix and the max keys over its rows (buckets are monotone, so the bucket
  //    of the max is the max of the buckets), and its stream from its smallest commit index (stream_for, as L8's
  //    conductor does). Two rows reach every (max suffix, max keys) pair a call of any size can have (one row holds
  //    the longest suffix, one the most keys; more rows can repeat a shape), so a rows bucket of B >= 2 takes every
  //    pair of shapes, with the stream decided by the most urgent row the call can carry.
  std::set<GraphKey> keys;
  auto add = [&](Phase cp, int32_t rows, int32_t s, int32_t k, int32_t min_ci) {
    keys.insert(GraphKey{int32_t(stream_for(cp, min_ci)), int32_t(graph_phase(cp)), bucket_of(kRowBuckets, rows),
                         suffix_bucket_for(cp, s), keys_bucket_for(k)});
  };
  // Bucketed shapes (dedupe before the pair loop): (suffix bucket, keys bucket, smallest commit index).
  auto bucketed = [](Phase cp, const std::set<RowShape>& rows) {
    std::set<RowShape> b;
    for (const RowShape& r : rows)  // commit index only matters as <= 1 (urgent) or not (stream_for)
      b.insert(RowShape{suffix_bucket_for(cp, r.s), keys_bucket_for(r.k), std::min(r.ci, 2)});
    return b;
  };
  auto calls = [&](Phase cp, const std::set<RowShape>& singles, const std::set<RowShape>& riders, int32_t max_b) {
    // singles: shapes a 1-row call can have; riders: every shape a row of a multi-row call can have.
    const std::set<RowShape> one = bucketed(cp, singles), many = bucketed(cp, riders);
    for (int32_t B : kRowBuckets) {
      if (B > max_b) break;
      if (B == 1) {
        for (const RowShape& r : one) add(cp, 1, r.s, r.k, r.ci);
        continue;
      }
      for (const RowShape& a : many)
        for (const RowShape& b : many) {
          const int32_t s = std::max(a.s, b.s), k = std::max(a.k, b.k);
          add(cp, B, s, k, std::min(a.ci, b.ci));
          if (B >= 3)  // a third row of a smaller shape may make the call urgent
            for (const RowShape& c : many)
              if (c.s <= s && c.k <= k && c.ci < std::min(a.ci, b.ci)) add(cp, B, s, k, c.ci);
        }
    }
  };
  // First-commit calls: every row is a first commit (L8: a call with any later commit is a steady call).
  calls(Phase::kFirstCommit, first_rows, first_rows, max_rows);
  // Steady calls: later commits, first commits may ride along.
  std::set<RowShape> commit_rows = steady_rows;
  commit_rows.insert(first_rows.begin(), first_rows.end());
  calls(Phase::kSteadyCommit, steady_rows, commit_rows, max_rows);
  calls(Phase::kFinalFlush, final_rows, final_rows, max_rows);
  // stream:false whole renders: one row per call (several DEMO rows ending on the same turn share a call in L8;
  // those keys are not enumerated: rows bucket x 64-token suffix buckets would be ~25 graphs per bucket).
  calls(Phase::kWholeRender, whole_rows, whole_rows, 1);
  return keys;
}

}  // namespace kt2n::render
