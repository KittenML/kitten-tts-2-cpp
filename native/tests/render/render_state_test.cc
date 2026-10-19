// L7 render state: commit geometry vs production's oracle tables, opening-silence skip, page pool,
// ragged descriptors.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "kt2n/render/call.h"
#include "kt2n/render/commit_plan.h"
#include "kt2n/render/rng.h"
#include "kt2n/render/state.h"

using namespace kt2n::render;

namespace {

struct Pub {
  int32_t before, end, samples;
  bool final;
};

// Drive one row through its commits and its final flush exactly as the conductor would (ready on the
// turn it holds next + 3 tokens) and collect (committed_before, end, published samples).
// `commit_tokens` bounds the commit loop (the oracle's batched rows commit in lockstep up to the
// shortest row, then each row's final flush covers its own tokens).
// The oracle recordings (production ebcdd3f, 8 Oct deploy) ran the helper schedule 5,10,15,25,50 for shared rows;
// the default shared schedule equals the single-request schedule.
CommitPlanConfig oracle_20261008_config() {
  CommitPlanConfig cfg;
  cfg.shared_schedule = {5, 10, 15, 25, 50};
  return cfg;
}

std::vector<Pub> drive(int32_t tokens, int32_t sharing, int32_t commit_tokens = -1,
                       const CommitPlanConfig& cfg = CommitPlanConfig{}) {
  if (commit_tokens < 0) commit_tokens = tokens;
  RowCommitState st(&cfg);
  std::vector<Pub> pubs;
  for (int32_t held = 1; held <= commit_tokens; ++held) {
    st.decide_alone(held, sharing);
    if (st.ready(held)) {
      const RenderGeometry g = st.plan_commit();
      pubs.push_back({g.committed_before, g.committed_after, g.pcm_samples, false});
      st.commit_launched(g);
    }
  }
  const RenderGeometry f = st.plan_final(tokens);
  pubs.push_back({f.committed_before, f.committed_after, f.pcm_samples, true});
  return pubs;
}

}  // namespace

// Oracle eager-r5 (production ebcdd3f) render/B1/plan.json: alone schedule, 152 tokens.
TEST(CommitPlan, MatchesOracleB1Alone) {
  const std::vector<Pub> want = {{0, 4, 3360, false},     {4, 9, 1440, false},     {9, 14, 4800, false},
                                 {14, 24, 9600, false},   {24, 39, 14400, false},  {39, 64, 24000, false},
                                 {64, 114, 48000, false}, {114, 155, 43200, true}};
  const auto got = drive(152, 1);
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < want.size(); ++i) {
    EXPECT_EQ(got[i].before, want[i].before) << i;
    EXPECT_EQ(got[i].end, want[i].end) << i;
    EXPECT_EQ(got[i].samples, want[i].samples) << i;
    EXPECT_EQ(got[i].final, want[i].final) << i;
  }
}

// Oracle render/B4 + B10 plan.json: helper cadence (shared, the 8 Oct schedule 5,10,15,25,50), rows of 152..183
// tokens. The geometry port is checked with the oracle's own schedule.
TEST(CommitPlan, MatchesOracleB10Shared) {
  const int32_t tokens[] = {152, 160, 160, 168, 168, 156, 160, 180, 156, 172};
  const int32_t final_samples[] = {51840, 59520, 59520, 67200, 67200, 55680, 59520, 78720, 55680, 71040};
  const std::vector<Pub> head = {{0, 5, 4320, false},    {5, 15, 6240, false},   {15, 30, 14400, false},
                                 {30, 55, 24000, false}, {55, 105, 48000, false}};
  const CommitPlanConfig oracle = oracle_20261008_config();
  for (int r = 0; r < 10; ++r) {
    const auto got = drive(tokens[r], 3, 152, oracle);
    ASSERT_EQ(got.size(), head.size() + 1) << r;
    for (size_t i = 0; i < head.size(); ++i) {
      EXPECT_EQ(got[i].before, head[i].before);
      EXPECT_EQ(got[i].end, head[i].end);
      EXPECT_EQ(got[i].samples, head[i].samples);
    }
    EXPECT_EQ(got.back().before, 105);
    EXPECT_EQ(got.back().end, tokens[r] + 3);
    EXPECT_EQ(got.back().samples, final_samples[r]) << r;
  }
}

// Default shared and single-request commit schedules match, so the B1 geometry holds for any sharing.
TEST(CommitPlan, SharedRowsTakeTheAloneScheduleSince20261012) {
  const CommitPlanConfig def;
  EXPECT_EQ(def.shared_schedule, def.alone_schedule);
  const auto alone = drive(152, 1);
  for (int32_t sharing : {2, 3, 16}) {
    const auto got = drive(152, sharing);
    ASSERT_EQ(got.size(), alone.size()) << sharing;
    for (size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i].before, alone[i].before) << sharing << " " << i;
      EXPECT_EQ(got[i].end, alone[i].end) << sharing << " " << i;
      EXPECT_EQ(got[i].samples, alone[i].samples) << sharing << " " << i;
    }
  }
}

TEST(CommitPlan, WholeRenderAndPreview) {
  // Oracle whole/B1: 152 tokens + 3 x 4299 -> 155 tokens, 148,800 samples.
  const RenderGeometry w = plan_whole_render(152);
  EXPECT_EQ(w.render_tokens, 155);
  EXPECT_EQ(w.pcm_samples, 148800);
  EXPECT_TRUE(w.trim_fade_head);
  CommitPlanConfig cfg;
  const RenderGeometry p = plan_preview(cfg);
  EXPECT_EQ(p.render_tokens, 8);
  EXPECT_EQ(p.pcm_samples, 4800);
  // LIVE_PATHS 1.1: a shared row (c0 = 5) never previews; an alone row (c0 = 4) can at 8 tokens.
  PreviewInputs in;
  in.tokens_held = 8;
  in.first_commit_tokens = 5;
  EXPECT_FALSE(preview_eligible(cfg, in));
  in.first_commit_tokens = 4;
  EXPECT_TRUE(preview_eligible(cfg, in));
  in.committed_tokens = 4;
  EXPECT_FALSE(preview_eligible(cfg, in));
}

TEST(CommitPlan, DecideAtFourTokensOnce) {
  // With the 8 Oct shared schedule (the decision itself is schedule independent).
  const CommitPlanConfig cfg = oracle_20261008_config();
  RowCommitState st(&cfg);
  st.decide_alone(3, 1);
  EXPECT_FALSE(st.decided());
  EXPECT_EQ(st.next_commit_tokens(), 5);
  st.decide_alone(4, 2);  // shared at the deciding turn
  EXPECT_TRUE(st.decided());
  EXPECT_FALSE(st.alone());
  st.decide_alone(5, 1);  // later sharing changes nothing
  EXPECT_FALSE(st.alone());
  EXPECT_EQ(st.next_commit_tokens(), 5);
  EXPECT_FALSE(st.ready(7));
  EXPECT_TRUE(st.ready(8));
  st.set_inflight(true);
  EXPECT_FALSE(st.ready(8));
  // The default (production since 12 Oct 2026): every row's commit 0 is 4 tokens, ready at 7.
  const CommitPlanConfig def;
  RowCommitState sd(&def);
  EXPECT_EQ(sd.next_commit_tokens(), 4);
  sd.decide_alone(4, 2);
  EXPECT_FALSE(sd.alone());
  EXPECT_EQ(sd.next_commit_tokens(), 4);
  EXPECT_FALSE(sd.ready(6));
  EXPECT_TRUE(sd.ready(7));
}

TEST(CommitPlan, Coalescing) {
  CommitPlanConfig cfg;
  CoalesceRow r{1, 18, 15, 0.5};  // ready, overdue 0
  EXPECT_TRUE(coalesce_group_waits(cfg, &r, 1, 12));
  EXPECT_FALSE(coalesce_group_waits(cfg, &r, 1, 13));  // partner too far
  EXPECT_FALSE(coalesce_group_waits(cfg, &r, 1, 0));   // no partner
  r.tokens_held = 20;                                  // overdue 2: 2 + 11 > 12
  EXPECT_FALSE(coalesce_group_waits(cfg, &r, 1, 11));
  EXPECT_TRUE(coalesce_group_waits(cfg, &r, 1, 10));
  r.reservoir_s = 0.29;
  EXPECT_FALSE(coalesce_group_waits(cfg, &r, 1, 10));
  r.reservoir_s = -1;  // no estimate
  EXPECT_FALSE(coalesce_group_waits(cfg, &r, 1, 10));
  CoalesceRow first{0, 8, 5, 5.0};  // a first commit never waits
  EXPECT_FALSE(coalesce_group_waits(cfg, &first, 1, 1));
}

TEST(OpeningSkip, WithholdsKeepsOneAndClosesOnTwo) {
  OpeningSkipConfig cfg;
  std::vector<int32_t> out;
  OpeningSkip sk(cfg, true);
  EXPECT_TRUE(opening_silence_id(4299));
  EXPECT_FALSE(opening_silence_id(1466));  // speech-level ids are not skipped (no_speech set)
  EXPECT_FALSE(opening_silence_id(3653));
  // set, set, stray(held), set (drops stray), non, non -> keep last skipped + the 2 non-set
  for (int32_t v : {4299, 4300, 100, 6495, 200, 201}) sk.admit(v, out);
  EXPECT_TRUE(sk.closed());
  EXPECT_EQ(out, (std::vector<int32_t>{6495, 200, 201}));
  EXPECT_EQ(sk.skipped(), 4);
  sk.admit(4299, out);  // after closing everything passes
  EXPECT_EQ(out.back(), 4299);
}

TEST(OpeningSkip, CapAndRestore) {
  OpeningSkipConfig cfg;
  std::vector<int32_t> out;
  OpeningSkip sk(cfg, true);
  for (int i = 0; i < 48; ++i) sk.admit(4299, out);
  EXPECT_TRUE(out.empty());
  sk.admit(4299, out);  // the 49th: cap reached -> closes, keeps 1 + this one
  EXPECT_EQ(out, (std::vector<int32_t>{4299, 4299}));
  OpeningSkip sk2(cfg, true);
  std::vector<int32_t> out2;
  for (int32_t v : {4299, 4300, 100}) sk2.admit(v, out2);
  sk2.restore(out2);
  EXPECT_EQ(out2, (std::vector<int32_t>{4299, 4300, 100}));
  OpeningSkip off(cfg, false);  // later segment / DEMO
  std::vector<int32_t> out3;
  off.admit(4299, out3);
  EXPECT_EQ(out3, (std::vector<int32_t>{4299}));
}

TEST(PagePool, AllocRetainRelease) {
  PoolView v{};
  v.n_pages = 8;
  HostPagePool pool(v);
  int32_t a[5];
  ASSERT_TRUE(pool.alloc(5, a));
  EXPECT_EQ(pool.pages_free(), 3);
  int32_t b[4];
  EXPECT_FALSE(pool.alloc(4, b));  // all-or-nothing
  EXPECT_EQ(pool.pages_free(), 3);
  pool.retain(a, 2);
  pool.release(a, 5);
  EXPECT_EQ(pool.pages_free(), 6);  // two shared pages still held once
  pool.release(a, 2);
  EXPECT_EQ(pool.pages_free(), 8);
  EXPECT_THROW(pool.release(a, 1), std::logic_error);
}

TEST(Geometry, PageBytesAndPrefix) {
  EXPECT_EQ(kPageBytes, 16515072u);
  EXPECT_NEAR(double(kPageBytes) / kPageTokens / 1e6, 0.516, 0.001);  // ~0.52 MB per token
  EXPECT_EQ(stable_prefix_tokens(250), 225);
  EXPECT_EQ(stable_prefix_tokens(237), 225);
  EXPECT_EQ(stable_prefix_tokens(27), 0);
  EXPECT_EQ(pages_for_frames(450), 8);
}

namespace {
VoicePrefix make_voice(int32_t pt, HostPagePool& pool, bool build_prefix) {
  VoicePrefix v;
  v.voice = 3;
  v.prompt_tokens = pt;
  v.prompt_frames = 2 * pt;
  v.prompt_ids.resize(size_t(pt));
  std::iota(v.prompt_ids.begin(), v.prompt_ids.end(), 1000);
  if (build_prefix) {
    v.stable_tokens = stable_prefix_tokens(pt);
    const int32_t full = 2 * v.stable_tokens / kPageFrames;
    v.shared_pages.resize(size_t(full));
    EXPECT_TRUE(pool.alloc(full, v.shared_pages.data()));
    if ((2 * v.stable_tokens) % kPageFrames) {
      EXPECT_TRUE(pool.alloc(1, &v.partial_page));
    }
    v.built = true;
  }
  return v;
}
}  // namespace

TEST(Descriptors, StreamingRowFromVoicePrefix) {
  PoolView pv{};
  pv.n_pages = 64;
  HostPagePool pool(pv);
  VoicePrefix voice = make_voice(250, pool, true);  // stable 225 -> 7 shared pages + a partial page
  EXPECT_EQ(voice.shared_pages.size(), 7u);
  EXPECT_GE(voice.partial_page, 0);
  const int32_t free0 = pool.pages_free();

  CommitPlanConfig cfg;
  RowCommitState st(&cfg);
  st.decide_alone(4, 1);
  RowPages pages;
  pages.admit(pool, &voice);
  EXPECT_EQ(pages.tok_past(), 225);
  EXPECT_TRUE(pages.needs_prefix_copy());
  std::vector<int32_t> ids(40);
  std::iota(ids.begin(), ids.end(), 1);

  RowRender row;
  row.slot = 5;
  row.voice = &voice;
  row.ids = ids.data();
  row.pages = &pages;
  row.seed = 279732224;
  row.noise_w = target_noise_width(196);
  row.geom = st.plan_commit();  // commit 0: ids[:7]
  CallPlan plan;
  ASSERT_EQ(build_call(Phase::kFirstCommit, &row, 1, pool, plan), BuildStatus::kOk);
  EXPECT_EQ(plan.rows_bucket, 1);
  const RowDesc& d = plan.rows[0];
  EXPECT_EQ(d.tok_past, 225);
  EXPECT_EQ(d.tok_suffix, 25 + 7);  // prompt[225:] + 7 ids
  EXPECT_EQ(plan.suffix_bucket, 32);
  EXPECT_EQ(d.fr_keys, 2 * 257);
  EXPECT_EQ(d.n_pages, 9);
  EXPECT_EQ(d.n_shared_pages, 7);
  EXPECT_EQ(d.out_fr_begin, 500);
  EXPECT_EQ(d.out_fr_count, 8);
  EXPECT_EQ(d.pcm_count, 3360);
  EXPECT_EQ(d.lookahead, 3);
  EXPECT_EQ(d.flags & kRowTrimFade, kRowTrimFade);
  EXPECT_EQ(d.flags & kRowFirstCommitTail, kRowFirstCommitTail);
  EXPECT_EQ(d.philox_seed, hift_seed(279732224, 0));
  EXPECT_EQ(d.philox_seed, 2797325240000ull);  // oracle B1 commit0 hift0/call.json seeds
  EXPECT_EQ(plan.tokens.front(), 1000 + 225);
  EXPECT_EQ(plan.tokens.back(), 7);
  EXPECT_EQ(plan.prefix_copy_rows, std::vector<int32_t>{0});
  EXPECT_EQ(plan.new_tok_past[0], 250);  // floor((250 + 4) / 25) * 25
  EXPECT_EQ(pool.pages_free(), free0 - 2);  // the row's private pages 7 and 8
  apply_call_state(plan, &row, 1);
  st.commit_launched(row.geom);
  EXPECT_FALSE(pages.needs_prefix_copy());

  // Commit 1 (ids[:12]): suffix = ids[0:12] from position 250.
  row.geom = st.plan_commit();
  ASSERT_EQ(build_call(Phase::kSteadyCommit, &row, 1, pool, plan), BuildStatus::kOk);
  EXPECT_EQ(plan.rows[0].tok_past, 250);
  EXPECT_EQ(plan.rows[0].tok_suffix, 12);
  EXPECT_EQ(plan.suffix_bucket, 16);
  EXPECT_EQ(plan.rows[0].pcm_count, 1440);
  EXPECT_EQ(plan.rows[0].publish_skip, 3360);
  EXPECT_EQ(plan.rows[0].src_cache, 3840);
  EXPECT_EQ(plan.rows[0].mel_cache, 8);
  EXPECT_EQ(plan.rows[0].philox_seed, hift_seed(279732224, 4));

  pages.release(pool);
  EXPECT_EQ(pool.pages_free(), free0);
}

TEST(Descriptors, RaggedBatchPackingAndBackPressure) {
  PoolView pv{};
  pv.n_pages = 40;
  HostPagePool pool(pv);
  VoicePrefix voice = make_voice(250, pool, true);
  // Three SHARED rows with 5-token first commits (the 8 Oct helper schedule; the packing arithmetic below is
  // schedule independent, the numbers are written for 5 + 3 tokens per row).
  const CommitPlanConfig cfg = oracle_20261008_config();
  std::vector<RowCommitState> st(3, RowCommitState(&cfg));
  std::vector<RowPages> pages(3);
  std::vector<int32_t> ids(300, 7);
  std::vector<RowRender> rows(3);
  for (int i = 0; i < 3; ++i) {
    st[size_t(i)].decide_alone(4, 3);
    pages[size_t(i)].admit(pool, &voice);
    rows[size_t(i)].slot = i;
    rows[size_t(i)].voice = &voice;
    rows[size_t(i)].ids = ids.data();
    rows[size_t(i)].pages = &pages[size_t(i)];
    rows[size_t(i)].seed = 100 + uint64_t(i);
    rows[size_t(i)].noise_w = target_noise_width(196);
    rows[size_t(i)].geom = st[size_t(i)].plan_commit();
  }
  CallPlan plan;
  ASSERT_EQ(build_call(Phase::kFirstCommit, rows.data(), 3, pool, plan), BuildStatus::kOk);
  EXPECT_EQ(plan.rows_bucket, 3);
  EXPECT_EQ(plan.rows[1].tok_off, plan.rows[0].tok_suffix);
  EXPECT_EQ(plan.rows[2].pcm_off, 2 * 4320);
  EXPECT_EQ(plan.total_pcm, 3 * 4320);
  EXPECT_EQ(int32_t(plan.page_table.size()), 3 * 9);
  // Estimator packing (L5 contract): 16-aligned fr_off, each row owns round_up(fr_suffix, 16) frames, everything
  // else identical to the encoder's table; total within the frames the bucket provides.
  ASSERT_EQ(plan.est_rows.size(), plan.rows.size());
  int32_t est_end = 0;
  for (int i = 0; i < 3; ++i) {
    const RowDesc& e = plan.est_rows[size_t(i)];
    const RowDesc& d = plan.rows[size_t(i)];
    EXPECT_EQ(e.fr_off % kEstRowAlign, 0);
    EXPECT_GE(e.fr_off, est_end);
    est_end = e.fr_off + (d.fr_suffix + 15) / 16 * 16;
    EXPECT_EQ(d.fr_off, 2 * d.tok_off);  // the encoder's contiguous packing
    EXPECT_EQ(e.fr_suffix, d.fr_suffix);
    EXPECT_EQ(e.tok_off, d.tok_off);
    EXPECT_EQ(e.page_off, d.page_off);
  }
  EXPECT_EQ(plan.rows[1].fr_off, 66);      // 33 tokens per row: contiguous packing is NOT 16-aligned
  EXPECT_EQ(plan.est_rows[1].fr_off, 80);  // round_up(66, 16)
  EXPECT_EQ(plan.est_total_frames, est_end);
  EXPECT_LE(plan.est_total_frames, 2 * plan.rows_bucket * plan.suffix_bucket);
  // A final flush of 300 tokens for each row needs 3 x (2 * 553 / 64 -> 18) pages: not enough left.
  for (int i = 0; i < 3; ++i) rows[size_t(i)].geom = st[size_t(i)].plan_final(300);
  const int32_t free_before = pool.pages_free();
  EXPECT_EQ(build_call(Phase::kFinalFlush, rows.data(), 3, pool, plan), BuildStatus::kNoPages);
  EXPECT_EQ(pool.pages_free(), free_before);
}

TEST(Descriptors, WholeRenderScratchPages) {
  PoolView pv{};
  pv.n_pages = 64;
  HostPagePool pool(pv);
  VoicePrefix voice = make_voice(250, pool, false);
  std::vector<int32_t> ids(152, 9);
  RowRender row;
  row.slot = 0;
  row.voice = &voice;
  row.ids = ids.data();
  row.geom = plan_whole_render(152);
  CallPlan plan;
  ASSERT_EQ(build_call(Phase::kWholeRender, &row, 1, pool, plan), BuildStatus::kOk);
  EXPECT_EQ(plan.rows[0].tok_past, 0);
  EXPECT_EQ(plan.rows[0].tok_suffix, 250 + 155);
  EXPECT_EQ(plan.suffix_bucket, 448);  // 64-token buckets
  EXPECT_EQ(plan.rows[0].flags & kRowFullAttention, kRowFullAttention);
  EXPECT_EQ(plan.rows[0].pcm_count, 148800);
  EXPECT_EQ(int32_t(plan.scratch_pages.size()), pages_for_frames(810));
  EXPECT_EQ(plan.tokens.back(), kSilenceToken);
  pool.release(plan.scratch_pages.data(), int32_t(plan.scratch_pages.size()));
  EXPECT_EQ(pool.pages_free(), 64);
}

// row_pages_needed is the exact maximum: a streaming row driven through every commit and its final on the
// voice prefix allocates exactly that many pages; a whole render takes exactly the demo count as scratch.
TEST(Descriptors, RowPagesNeededIsExact) {
  for (int32_t n : {4, 7, 31, 152, 196, 500, 1200}) {
    PoolView pv{};
    pv.n_pages = 400;
    HostPagePool pool(pv);
    VoicePrefix voice = make_voice(250, pool, true);
    const int32_t free0 = pool.pages_free();
    CommitPlanConfig cfg;
    RowCommitState st(&cfg);
    st.decide_alone(4, 1);
    RowPages pages;
    pages.admit(pool, &voice);
    std::vector<int32_t> ids(size_t(n), 5);
    RowRender row;
    row.slot = 0;
    row.voice = &voice;
    row.ids = ids.data();
    row.pages = &pages;
    row.noise_w = target_noise_width(n);
    CallPlan plan;
    while (st.next_commit_tokens() + kLookahead <= n) {
      row.geom = st.plan_commit();
      ASSERT_EQ(build_call(Phase::kSteadyCommit, &row, 1, pool, plan), BuildStatus::kOk);
      apply_call_state(plan, &row, 1);
      st.commit_launched(row.geom);
    }
    row.geom = st.plan_final(n);
    ASSERT_EQ(build_call(Phase::kFinalFlush, &row, 1, pool, plan), BuildStatus::kOk);
    EXPECT_EQ(free0 - pool.pages_free(), row_pages_needed(&voice, 250, n, false)) << "tokens " << n;
    pages.release(pool);

    RowRender whole = row;
    whole.pages = nullptr;
    whole.geom = plan_whole_render(n);
    ASSERT_EQ(build_call(Phase::kWholeRender, &whole, 1, pool, plan), BuildStatus::kOk);
    EXPECT_EQ(int32_t(plan.scratch_pages.size()), row_pages_needed(&voice, 250, n, true)) << "tokens " << n;
    pool.release(plan.scratch_pages.data(), int32_t(plan.scratch_pages.size()));
  }
  EXPECT_EQ(row_pages_needed(nullptr, 250, 196, false), pages_for_frames(2 * (250 + 199)));
}

// Graph prewarm covers what live rows produce: rows of several lengths (alone and in a 3-row shared batch) are
// driven through their commits and final on a real pool; every call's key must be in the enumerated set.
TEST(Descriptors, PrewarmKeysCoverLiveCalls) {
  PoolView pv{};
  pv.n_pages = 2000;
  HostPagePool pool(pv);
  VoicePrefix voice = make_voice(250, pool, true);
  CommitPlanConfig cfg;
  const std::set<GraphKey> keys = enumerate_graph_keys(voice, cfg, 20, 320);
  EXPECT_GT(keys.size(), 10u);
  auto key_of = [](const CallPlan& p, int32_t n_rows, int32_t ci) {
    return GraphKey{int32_t(stream_for(p.phase, ci)), int32_t(graph_phase(p.phase)), bucket_of(kRowBuckets, n_rows),
                    p.suffix_bucket, p.keys_bucket};
  };
  int32_t checked = 0;
  for (int32_t B : {1, 3}) {
    for (int32_t n : {9, 57, 152, 199, 318}) {
      std::vector<RowCommitState> st(static_cast<size_t>(B), RowCommitState(&cfg));
      std::vector<RowPages> pages(static_cast<size_t>(B));
      std::vector<RowRender> rows(static_cast<size_t>(B));
      std::vector<int32_t> ids(size_t(n), 3);
      for (int32_t i = 0; i < B; ++i) {
        st[size_t(i)].decide_alone(4, B);
        pages[size_t(i)].admit(pool, &voice);
        rows[size_t(i)].slot = i;
        rows[size_t(i)].voice = &voice;
        rows[size_t(i)].ids = ids.data();
        rows[size_t(i)].pages = &pages[size_t(i)];
        rows[size_t(i)].noise_w = target_noise_width(n);
      }
      CallPlan plan;
      while (st[0].next_commit_tokens() + kLookahead <= n) {
        const int32_t ci = st[0].commit_index();
        for (int32_t i = 0; i < B; ++i) rows[size_t(i)].geom = st[size_t(i)].plan_commit();
        const Phase cp = ci == 0 ? Phase::kFirstCommit : Phase::kSteadyCommit;
        ASSERT_EQ(build_call(cp, rows.data(), B, pool, plan), BuildStatus::kOk);
        EXPECT_EQ(keys.count(key_of(plan, B, ci)), 1u) << "B " << B << " n " << n << " commit " << ci;
        ++checked;
        apply_call_state(plan, rows.data(), B);
        for (int32_t i = 0; i < B; ++i) st[size_t(i)].commit_launched(rows[size_t(i)].geom);
      }
      for (int32_t i = 0; i < B; ++i) rows[size_t(i)].geom = st[size_t(i)].plan_final(n);
      ASSERT_EQ(build_call(Phase::kFinalFlush, rows.data(), B, pool, plan), BuildStatus::kOk);
      // Finals are enumerated at every 4th token; the suffix / keys buckets are 16 / 256 tokens wide, so a final at
      // any length shares its key with an enumerated one.
      EXPECT_EQ(keys.count(key_of(plan, B, st[0].commit_index())), 1u) << "final B " << B << " n " << n;
      ++checked;
      for (auto& pg : pages) pg.release(pool);
    }
  }
  EXPECT_GT(checked, 40);
}

// L8 (13:37) saw live graph captures at N=4 that the lockstep replay above never produces: a shared-schedule row
// alone in its call (1 row, 5-token first commit; 1 row, 15), rows at different positions in one call (2 rows,
// 15 + 30), and late finals. This drives a conductor-like mix (rows start at random turns, the schedule is decided
// at 4 tokens from how many rows are active, ready rows are held back at random (coalescing / a commit in flight),
// ready rows are split into random groups, finals are grouped at random) and checks every call's key.
TEST(Descriptors, PrewarmKeysCoverRandomConductorMixes) {
  PoolView pv{};
  pv.n_pages = 1 << 16;
  HostPagePool pool(pv);
  VoicePrefix voice = make_voice(250, pool, true);
  CommitPlanConfig cfg;
  constexpr int32_t kMaxRows = 20, kTokens = 320;
  const std::set<GraphKey> keys = enumerate_graph_keys(voice, cfg, kMaxRows, kTokens);
  // An unbuilt voice enumerates the geometry its prefix build will give: the same keys.
  HostPagePool pool2(pv);
  VoicePrefix unbuilt = make_voice(250, pool2, false);
  EXPECT_EQ(enumerate_graph_keys(unbuilt, cfg, kMaxRows, kTokens), keys);
  int32_t per_role[8] = {0};
  for (const GraphKey& k : keys) per_role[k.role & 7]++;
  std::printf("[ keys ] %zu (urgent %d, bulk %d)\n", keys.size(), per_role[int(StreamRole::kSUrgent)],
              per_role[int(StreamRole::kSBulk)]);

  std::mt19937 rng(20261010);
  auto uni = [&](int32_t lo, int32_t hi) { return std::uniform_int_distribution<int32_t>(lo, hi)(rng); };
  struct SimRow {
    int32_t len = 0, start = 0, held = 0;
    bool done = false;
    RowCommitState st;
    RowPages pages;
    std::vector<int32_t> ids;
    explicit SimRow(const CommitPlanConfig* c) : st(c) {}
  };
  int32_t checked = 0, missed = 0;
  auto check = [&](Phase cp, std::vector<SimRow*>& g) {
    std::vector<RowRender> rr(g.size());
    int32_t min_ci = 1 << 30;
    for (size_t i = 0; i < g.size(); ++i) {
      rr[i].slot = int32_t(i);
      rr[i].voice = &voice;
      rr[i].ids = g[i]->ids.data();
      rr[i].pages = &g[i]->pages;
      rr[i].noise_w = target_noise_width(g[i]->len);
      rr[i].extra_flags = kRowLookaheadPadZero;
      rr[i].geom = cp == Phase::kFinalFlush ? g[i]->st.plan_final(g[i]->held) : g[i]->st.plan_commit();
      min_ci = std::min(min_ci, g[i]->st.commit_index());
    }
    CallPlan plan;
    ASSERT_EQ(build_call(cp, rr.data(), int32_t(g.size()), pool, plan), BuildStatus::kOk);
    const GraphKey k{int32_t(stream_for(cp, min_ci)), int32_t(graph_phase(cp)), plan.rows_bucket, plan.suffix_bucket,
                     plan.keys_bucket};
    ++checked;
    if (keys.count(k) == 0 && ++missed <= 10)
      ADD_FAILURE() << "missed key: role " << k.role << " phase " << int(cp) << " rows " << g.size() << " suffix "
                    << k.suffix_bucket << " keys " << k.keys_bucket;
    if (cp != Phase::kFinalFlush) {
      apply_call_state(plan, rr.data(), int32_t(g.size()));
      for (size_t i = 0; i < g.size(); ++i) g[i]->st.commit_launched(rr[i].geom);
    }
  };
  auto groups_of = [&](std::vector<SimRow*> v) {
    std::shuffle(v.begin(), v.end(), rng);
    std::vector<std::vector<SimRow*>> out;
    for (size_t i = 0; i < v.size();) {
      const size_t n = size_t(uni(1, int32_t(std::min<size_t>(kMaxRows, v.size() - i))));
      out.emplace_back(v.begin() + long(i), v.begin() + long(i + n));
      i += n;
    }
    return out;
  };
  for (int32_t episode = 0; episode < 120; ++episode) {
    const int32_t n_rows = uni(1, 24);
    std::vector<std::unique_ptr<SimRow>> rows;
    for (int32_t i = 0; i < n_rows; ++i) {
      rows.push_back(std::make_unique<SimRow>(&cfg));
      SimRow& r = *rows.back();
      r.len = uni(1, kTokens);
      r.start = uni(0, 40);
      r.ids.assign(size_t(r.len + 8), 7);
      r.pages.admit(pool, &voice);
    }
    for (int32_t turn = 0; turn < 40 + kTokens + 200; ++turn) {
      int32_t active = 0;
      for (auto& r : rows) active += (!r->done && turn >= r->start) ? 1 : 0;
      if (active == 0 && turn > 40) break;
      std::vector<SimRow*> ready, ending;
      for (auto& r : rows) {
        if (r->done || turn < r->start) continue;
        if (r->held < r->len && uni(0, 9) < 8) ++r->held;  // an LM step most turns
        r->st.decide_alone(r->held, active);
        if (r->st.ready(r->held) && uni(0, 9) >= 3) ready.push_back(r.get());
      }
      for (auto& g : groups_of(ready)) {
        bool all_first = true;
        for (SimRow* r : g) all_first &= r->st.commit_index() == 0 && r->st.committed_tokens() == 0;
        check(all_first ? Phase::kFirstCommit : Phase::kSteadyCommit, g);
      }
      for (auto& r : rows)
        if (!r->done && turn >= r->start && r->held == r->len && uni(0, 9) < 7) ending.push_back(r.get());
      for (auto& g : groups_of(ending)) {
        check(Phase::kFinalFlush, g);
        for (SimRow* r : g) {
          r->done = true;
          r->pages.release(pool);
        }
      }
    }
    for (auto& r : rows)
      if (!r->done) r->pages.release(pool);
  }
  for (int32_t n = 1; n <= kTokens; ++n) {  // 1-row whole renders
    RowRender row;
    row.slot = 0;
    row.voice = &voice;
    std::vector<int32_t> ids(size_t(n + 8), 7);
    row.ids = ids.data();
    row.noise_w = target_noise_width(n);
    row.geom = plan_whole_render(n);
    CallPlan plan;
    ASSERT_EQ(build_call(Phase::kWholeRender, &row, 1, pool, plan), BuildStatus::kOk);
    const GraphKey k{int32_t(stream_for(Phase::kWholeRender, 0)), int32_t(graph_phase(Phase::kWholeRender)), 1,
                     plan.suffix_bucket, plan.keys_bucket};
    ++checked;
    if (keys.count(k) == 0 && ++missed <= 10) ADD_FAILURE() << "missed whole render key, tokens " << n;
    pool.release(plan.scratch_pages.data(), int32_t(plan.scratch_pages.size()));
  }
  std::printf("[ calls ] %d checked, %d missed\n", checked, missed);
  EXPECT_EQ(missed, 0);
  EXPECT_GT(checked, 3000);
}

// Call merging (RenderService, sprint 5): merge_plans() over separately built calls must equal ONE build_call over
// all their rows (same descriptors, packing, page table, buckets), so a merged GPU call renders exactly what one call
// of those rows would. Two identical "worlds" replay the same random rows (positions, schedules, finals); world A
// builds the calls one by one, world B builds them as one call.
TEST(Descriptors, MergedPlanEqualsOneBuild) {
  CommitPlanConfig cfg;
  std::mt19937 rng(20261010);
  auto uni = [&](int32_t lo, int32_t hi) { return std::uniform_int_distribution<int32_t>(lo, hi)(rng); };
  struct World {
    std::unique_ptr<HostPagePool> pool;
    VoicePrefix voice;
    std::vector<RowCommitState> st;
    std::vector<RowPages> pages;
  };
  int32_t merged_ok = 0;
  for (int32_t trial = 0; trial < 300; ++trial) {
    const int32_t n = uni(2, 16);
    std::vector<int32_t> len(static_cast<size_t>(n)), adv(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      len[size_t(i)] = uni(20, 300);
      adv[size_t(i)] = uni(0, len[size_t(i)] - 10);  // tokens the row has already committed through
    }
    const bool finals = uni(0, 3) == 0;
    std::vector<int32_t> ids(400);
    std::iota(ids.begin(), ids.end(), 1);
    World w[2];
    for (World& x : w) {
      PoolView pv{};
      pv.n_pages = 1 << 14;
      x.pool = std::make_unique<HostPagePool>(pv);
      x.voice = make_voice(250, *x.pool, true);
      x.st.assign(size_t(n), RowCommitState(&cfg));
      x.pages.assign(size_t(n), RowPages{});
      for (int32_t i = 0; i < n; ++i) {
        RowCommitState& st = x.st[size_t(i)];
        x.pages[size_t(i)].admit(*x.pool, &x.voice);
        // Advance the row through its commits up to adv tokens (same in both worlds: deterministic).
        for (int32_t held = 1; held <= adv[size_t(i)]; ++held) {
          st.decide_alone(held, i % 3 == 0 ? 1 : 4);
          if (!st.ready(held)) continue;
          RowRender rr;
          rr.slot = i;
          rr.voice = &x.voice;
          rr.ids = ids.data();
          rr.pages = &x.pages[size_t(i)];
          rr.noise_w = target_noise_width(len[size_t(i)]);
          rr.geom = st.plan_commit();
          CallPlan p;
          ASSERT_EQ(build_call(rr.geom.phase == Phase::kFirstCommit ? Phase::kFirstCommit : Phase::kSteadyCommit, &rr, 1,
                               *x.pool, p),
                    BuildStatus::kOk);
          apply_call_state(p, &rr, 1);
          st.commit_launched(rr.geom);
        }
      }
    }
    // The rows' next calls: their next commit (or their final), with each row's own geometry.
    auto make_rr = [&](World& x, int32_t i) {
      RowRender rr;
      rr.slot = i;
      rr.voice = &x.voice;
      rr.ids = ids.data();
      rr.pages = &x.pages[size_t(i)];
      rr.noise_w = target_noise_width(len[size_t(i)]);
      rr.seed = 77 + uint64_t(i);
      rr.extra_flags = kRowLookaheadPadZero;
      RowCommitState& st = x.st[size_t(i)];
      if (finals) {
        rr.geom = st.plan_final(len[size_t(i)]);
      } else {
        st.decide_alone(std::max(4, adv[size_t(i)]), i % 3 == 0 ? 1 : 4);
        rr.geom = st.plan_commit();
      }
      return rr;
    };
    std::vector<RowRender> ra(static_cast<size_t>(n)), rb(static_cast<size_t>(n));
    bool all_first = true;
    for (int32_t i = 0; i < n; ++i) {
      ra[size_t(i)] = make_rr(w[0], i);
      rb[size_t(i)] = make_rr(w[1], i);
      all_first &= ra[size_t(i)].geom.phase == Phase::kFirstCommit;
    }
    auto phase_of = [&](const RowRender* r, int32_t k) {
      if (finals) return Phase::kFinalFlush;
      for (int32_t i = 0; i < k; ++i)
        if (r[i].geom.phase != Phase::kFirstCommit) return Phase::kSteadyCommit;
      return Phase::kFirstCommit;
    };
    // World A: random contiguous groups, one build per group (rows in order), then merged.
    std::vector<CallPlan> parts;
    for (int32_t i = 0; i < n;) {
      const int32_t k = uni(1, std::min(4, n - i));
      parts.emplace_back();
      ASSERT_EQ(build_call(phase_of(&ra[size_t(i)], k), &ra[size_t(i)], k, *w[0].pool, parts.back()), BuildStatus::kOk);
      i += k;
    }
    std::vector<const CallPlan*> pp;
    for (const CallPlan& p : parts) pp.push_back(&p);
    CallPlan merged, one;
    ASSERT_TRUE(merge_plans(pp.data(), int32_t(pp.size()), merged));
    ASSERT_EQ(build_call(phase_of(rb.data(), n), rb.data(), n, *w[1].pool, one), BuildStatus::kOk);
    EXPECT_EQ(int(merged.phase), int(one.phase));
    EXPECT_EQ(merged.n_rows, one.n_rows);
    EXPECT_EQ(merged.rows_bucket, one.rows_bucket);
    EXPECT_EQ(merged.suffix_bucket, one.suffix_bucket);
    EXPECT_EQ(merged.keys_bucket, one.keys_bucket);
    EXPECT_EQ(merged.flow_steps, one.flow_steps);
    EXPECT_EQ(merged.total_tokens, one.total_tokens);
    EXPECT_EQ(merged.total_frames, one.total_frames);
    EXPECT_EQ(merged.total_pcm, one.total_pcm);
    EXPECT_EQ(merged.est_total_frames, one.est_total_frames);
    EXPECT_EQ(merged.min_commit_index, one.min_commit_index);
    EXPECT_EQ(merged.tokens, one.tokens);
    EXPECT_EQ(merged.page_table, one.page_table);
    EXPECT_EQ(merged.new_tok_past, one.new_tok_past);
    EXPECT_EQ(merged.prefix_copy_rows, one.prefix_copy_rows);
    ASSERT_EQ(merged.rows.size(), one.rows.size());
    ASSERT_EQ(merged.est_rows.size(), one.est_rows.size());
    for (size_t r = 0; r < one.rows.size(); ++r) {
      EXPECT_EQ(0, std::memcmp(&merged.rows[r], &one.rows[r], sizeof(RowDesc))) << "row " << r << " trial " << trial;
      EXPECT_EQ(0, std::memcmp(&merged.est_rows[r], &one.est_rows[r], sizeof(RowDesc))) << "est row " << r;
    }
    if (!::testing::Test::HasFailure()) ++merged_ok;
    (void)all_first;
  }
  std::printf("[ merged ] %d of 300 merged calls equal one build\n", merged_ok);
  // Not mergeable: whole renders, mixed final + commit.
  PoolView pv{};
  pv.n_pages = 4096;
  HostPagePool pool(pv);
  VoicePrefix voice = make_voice(250, pool, true);
  RowRender whole;
  whole.slot = 0;
  whole.voice = &voice;
  std::vector<int32_t> ids2(64, 7);
  whole.ids = ids2.data();
  whole.noise_w = target_noise_width(40);
  whole.geom = plan_whole_render(40);
  CallPlan wp, wq;
  ASSERT_EQ(build_call(Phase::kWholeRender, &whole, 1, pool, wp), BuildStatus::kOk);
  ASSERT_EQ(build_call(Phase::kWholeRender, &whole, 1, pool, wq), BuildStatus::kOk);
  const CallPlan* two[2] = {&wp, &wq};
  CallPlan bad;
  EXPECT_FALSE(merge_plans(two, 2, bad));
  // A final mixed with commits: a steady call whose rows are the parts' rows with shifted offsets (no single
  // build_call can make this call; the GPU work is per row, see merge_plans).
  CommitPlanConfig cfg2;
  RowCommitState sf(&cfg2), sc(&cfg2);
  RowPages pf, pc;
  pf.admit(pool, &voice);
  pc.admit(pool, &voice);
  std::vector<int32_t> ids3(200, 9);
  RowRender rf, rc;
  rf.slot = 1;
  rc.slot = 2;
  rf.voice = rc.voice = &voice;
  rf.ids = rc.ids = ids3.data();
  rf.pages = &pf;
  rc.pages = &pc;
  rf.noise_w = rc.noise_w = target_noise_width(120);
  sf.decide_alone(4, 4);
  sc.decide_alone(4, 4);
  rf.geom = sf.plan_final(60);
  rc.geom = sc.plan_commit();
  CallPlan pfin, pcom, mixed;
  ASSERT_EQ(build_call(Phase::kFinalFlush, &rf, 1, pool, pfin), BuildStatus::kOk);
  ASSERT_EQ(build_call(Phase::kFirstCommit, &rc, 1, pool, pcom), BuildStatus::kOk);
  const CallPlan* mix[2] = {&pfin, &pcom};
  ASSERT_TRUE(merge_plans(mix, 2, mixed));
  EXPECT_EQ(int(mixed.phase), int(Phase::kSteadyCommit));
  EXPECT_EQ(mixed.n_rows, 2);
  EXPECT_EQ(mixed.total_pcm, pfin.total_pcm + pcom.total_pcm);
  EXPECT_EQ(mixed.rows[0].flags, pfin.rows[0].flags);
  EXPECT_EQ(mixed.rows[1].flags, pcom.rows[0].flags);
  EXPECT_EQ(mixed.rows[1].tok_off, pfin.total_tokens);
  EXPECT_EQ(mixed.rows[1].pcm_off, pfin.total_pcm);
  EXPECT_EQ(mixed.rows[1].page_off, int32_t(pfin.page_table.size()));
  EXPECT_EQ(mixed.suffix_bucket, suffix_bucket_for(Phase::kSteadyCommit, std::max(pfin.rows[0].tok_suffix, pcom.rows[0].tok_suffix)));
  EXPECT_NE(mixed.rows[0].flags & kRowFinalize, 0);
}

// A budget whose extension passes the noise table (streamed segments of 466+ characters; RC1 quality take
// W.Q.qset:C02 failed at open_row): the frames kept are exactly the full draw's, at the same positions.
TEST(Rng, NoiseExtensionClampedToTheTable) {
  const int32_t budget = 2341, pf = 600;  // the tagged budget cap (4.4 x int(380 x 1.4) + 1)
  const int32_t w = target_noise_width(budget), f = noise_extension_frames(budget);
  ASSERT_GT(f, 8192);
  const int32_t wide = f + 16, narrow = 8192;
  std::vector<float> a(size_t(kMel) * wide, -7.f), b(size_t(kMel) * narrow, -7.f);
  ASSERT_EQ(draw_row_target_noise(77, pf, budget, a.data(), wide), w);
  ASSERT_EQ(draw_row_target_noise(77, pf, budget, b.data(), narrow), w);
  EXPECT_EQ(draw_row_noise_extension(77, budget, a.data(), wide), f);
  EXPECT_EQ(draw_row_noise_extension(77, budget, b.data(), narrow), narrow);
  for (int32_t c = 0; c < kMel; ++c)
    for (int32_t x = 0; x < narrow; ++x)
      ASSERT_EQ(a[size_t(c) * wide + size_t(x)], b[size_t(c) * narrow + size_t(x)]) << c << " " << x;
  // Below the cap nothing changes: F fits, F returned, the same values as before.
  std::vector<float> d(size_t(kMel) * narrow, 0.f);
  EXPECT_EQ(draw_row_noise_extension(5, 1614, d.data(), narrow), noise_extension_frames(1614));
}

TEST(Rng, SaltAndSeeds) {
  EXPECT_EQ(hift_seed(279732224, 0), 2797325240000ull);
  EXPECT_EQ(request_seed_salt("abc") % 1024, 0u);
  EXPECT_EQ(target_noise_width(196), 398);
  EXPECT_EQ(noise_extension_frames(196), 790);
}
