// L8: unit tests for production's timing decisions (src/engine/policy.h). Each case states the
// production rule it checks.
#include <gtest/gtest.h>

#include "policy.h"

using namespace kt2n::engine::policy;

namespace {
Levers lv() { return live_levers("us"); }
}  // namespace

TEST(OpeningSkip, QuietIdsWithheldLastKeptRunOfTwoCloses) {
  // 4299 x3 skipped, then two non-set ids close; keep the last skipped id as context.
  OpeningSkip s;
  std::vector<int32_t> audio;
  for (int v : {4299, 4299, 331}) opening_admit(s, true, v, audio, lv());
  EXPECT_TRUE(audio.empty());
  opening_admit(s, true, 100, audio, lv());  // non-set: pending (run 1 < 2)
  EXPECT_TRUE(audio.empty());
  opening_admit(s, true, 101, audio, lv());  // second non-set: the opening is over
  EXPECT_EQ(audio, (std::vector<int32_t>{331, 100, 101}));
  EXPECT_EQ(s.skipped_tokens, 3);
  EXPECT_EQ(s.kept_tokens, 1);
  opening_admit(s, true, 4299, audio, lv());  // after close: everything passes
  EXPECT_EQ(audio.back(), 4299);
}

TEST(OpeningSkip, StrayBetweenQuietIdsIsDropped) {
  OpeningSkip s;
  std::vector<int32_t> audio;
  for (int v : {4299, 100, 4299, 200, 201}) opening_admit(s, true, v, audio, lv());
  // 100 was held, then a quiet id followed: it joins the skipped run.
  EXPECT_EQ(audio, (std::vector<int32_t>{4299, 200, 201}));
  EXPECT_EQ(s.skipped_tokens, 3);
}

TEST(OpeningSkip, SpeechLevelIdsAreNotQuietAndCapIs48) {
  EXPECT_FALSE(opening_quiet_id(1466));
  EXPECT_FALSE(opening_quiet_id(3653));
  EXPECT_TRUE(opening_quiet_id(4299));
  OpeningSkip s;
  std::vector<int32_t> audio;
  for (int i = 0; i < 48; ++i) opening_admit(s, true, 4299, audio, lv());
  EXPECT_TRUE(audio.empty());
  opening_admit(s, true, 4299, audio, lv());  // held == cap: closes
  EXPECT_EQ(audio.size(), 2u);                // kept 1 + this one
}

TEST(OpeningSkip, NotArmedPassesThroughAndRestore) {
  OpeningSkip s;
  std::vector<int32_t> audio;
  opening_admit(s, false, 4299, audio, lv());
  EXPECT_EQ(audio.size(), 1u);
  OpeningSkip t;
  std::vector<int32_t> a2;
  for (int v : {4299, 4299}) opening_admit(t, true, v, a2, lv());
  opening_restore(t, a2);  // a generation of only silence keeps its audio
  EXPECT_EQ(a2.size(), 2u);
}

TEST(Commit, AloneScheduleIs4_5_5_10_15_25_50) {
  CommitState c = new_commit_state(lv(), false);
  EXPECT_EQ(c.next, 4);  // helper schedule head (4,5,5,10,15,25,50 for every row)
  EXPECT_FALSE(d_prepare(c, 3, 1, lv()));  // < 4 tokens: no decision
  EXPECT_TRUE(d_prepare(c, 4, 1, lv()));
  EXPECT_EQ(c.next, 4);
  EXPECT_FALSE(d_prepare(c, 9, 3, lv()));  // decided once
  std::vector<int> sizes;
  int prev = 0;
  for (int k = 0; k < 9; ++k) {
    sizes.push_back(c.next - prev);
    prev = c.next;
    after_commit(c, lv());
  }
  EXPECT_EQ(sizes, (std::vector<int>{4, 5, 5, 10, 15, 25, 50, 50, 50}));
}

TEST(Commit, SharedRowKeepsHelperSchedule) {
  // Both schedules use 4,5,5,10,15,25,50, so sharing at the deciding turn does not change commit widths.
  CommitState c = new_commit_state(lv(), false);
  d_prepare(c, 4, 2, lv());
  EXPECT_FALSE(c.alone);
  std::vector<int> sizes;
  int prev = 0;
  for (int k = 0; k < 9; ++k) {
    sizes.push_back(c.next - prev);
    prev = c.next;
    after_commit(c, lv());
  }
  EXPECT_EQ(sizes, (std::vector<int>{4, 5, 5, 10, 15, 25, 50, 50, 50}));
}

TEST(Commit, SharedAndAloneRowsCommitTheSameSizes) {
  for (int sharing : {1, 2, 16}) {
    CommitState c = new_commit_state(lv(), false);
    EXPECT_FALSE(commit_ready(c, 6)) << sharing;
    d_prepare(c, 4, sharing, lv());
    EXPECT_EQ(c.alone, sharing == 1);
    EXPECT_FALSE(commit_ready(c, 6)) << sharing;
    EXPECT_TRUE(commit_ready(c, 7)) << sharing;  // 4 + 3 look-ahead
    std::vector<int> ends;
    for (int k = 0; k < 7; ++k) {
      ends.push_back(c.next);
      after_commit(c, lv());
    }
    EXPECT_EQ(ends, (std::vector<int>{4, 9, 14, 24, 39, 64, 114})) << sharing;
  }
}

TEST(Commit, ReadyNeedsLookaheadAndNoInflightAndNotHeld) {
  CommitState c = new_commit_state(lv(), false);
  EXPECT_FALSE(commit_ready(c, 6));
  EXPECT_TRUE(commit_ready(c, 7));  // helper head 4 + 3 look-ahead (was 8 with the 8 Oct head 5)
  c.inflight = true;
  EXPECT_FALSE(commit_ready(c, 20));
  CommitState d = new_commit_state(lv(), true);
  EXPECT_FALSE(commit_ready(d, 5000));
}

TEST(Hold, AdaptiveFormula) {
  EXPECT_DOUBLE_EQ(adaptive_hold_seconds(1, lv()), 0.06);
  EXPECT_NEAR(adaptive_hold_seconds(11, lv()), 0.12, 1e-12);
  EXPECT_DOUBLE_EQ(adaptive_hold_seconds(10000, lv()), 2.0);
  EXPECT_DOUBLE_EQ(start_target_seconds(5, true, lv()), 0.0);  // realtime phrase: hold 0
}

TEST(T9, StartingRowDefersNormalRows) {
  std::vector<T9Row> rows(3);
  rows[0].segment_index = 0;  // starting (no publication)
  rows[1].segment_index = 0;
  rows[1].preview_ready = true;
  rows[1].published_s = 5.0;
  rows[1].reservoir_s = 2.0;
  rows[1].row_ready = true;
  rows[2].segment_index = 1;
  rows[2].preview_ready = true;
  rows[2].reservoir_s = 0.1;  // urgent
  for (auto& r : rows) r.start_target_s = 0.072;
  const T9Result r = t9_acoustic_rows(rows, lv());
  EXPECT_EQ(r.selected, (std::vector<int>{0, 2}));
  EXPECT_EQ(r.deferred, 1);
  EXPECT_TRUE(r.starting);
}

TEST(T9, LazyTriggersOnThreeReadyOrBacklog) {
  std::vector<T9Row> rows(4);
  for (auto& r : rows) {
    r.segment_index = 1;
    r.preview_ready = true;
    r.reservoir_s = 2.0;
  }
  rows[0].row_ready = rows[1].row_ready = true;
  rows[3].backlog = 50;
  T9Result r = t9_acoustic_rows(rows, lv());
  EXPECT_EQ(r.selected, (std::vector<int>{3}));  // 2 ready < 3: only the backlog row
  rows[2].row_ready = true;
  r = t9_acoustic_rows(rows, lv());
  EXPECT_EQ(r.selected, (std::vector<int>{0, 1, 2, 3}));
}

TEST(T9, TrueReservoirSubtractsDroppedAndFinishedAlwaysSelected) {
  std::vector<T9Row> rows(2);
  rows[0].segment_index = 1;
  rows[0].preview_ready = true;
  rows[0].reservoir_s = 0.5;
  rows[0].dropped_s = 0.25;  // 0.25 < 0.3: urgent
  rows[1].finished = true;
  const T9Result r = t9_acoustic_rows(rows, lv());
  EXPECT_EQ(r.selected, (std::vector<int>{0, 1}));
}

TEST(Preview, BlockedForSharedRowsFiresForAloneRowAtEight) {
  // The predicate itself: a c0 of 5 blocks it, a c0 of 4 does not.
  EXPECT_FALSE(preview_due(false, 8, 0, false, 5, 0, lv()));  // c0 = 5: 5 + 3 >= 8
  EXPECT_TRUE(preview_due(false, 8, 0, false, 4, 0, lv()));   // c0 = 4 (alone; every row since 12 Oct)
  EXPECT_FALSE(preview_due(false, 7, 0, false, 4, 0, lv()));
  EXPECT_FALSE(preview_due(true, 8, 0, false, 4, 0, lv()));
}

TEST(Preview, NeverDueWhenCommitZeroLaunchesOnItsSevenTokenTurn) {
  // Every row now has c0 = 4, so q_first_no_preview never blocks. The preview stays unreachable because the
  // conductor launches commit 0 on the turn the row holds 7 tokens (one token per turn, T9 always selects a row
  // with no publication, coalescing never holds a first commit) and the launch advances committed at once
  // (engine.cc after_commit at launch). Walk one row turn by turn, shared and alone.
  for (int sharing : {1, 4, 16}) {
    CommitState c = new_commit_state(lv(), false);
    bool launched = false;
    for (int n = 1; n <= 60; ++n) {
      EXPECT_FALSE(preview_due(false, n, c.committed, c.held, c.next, c.commit_index, lv()))
          << "sharing " << sharing << " tokens " << n;
      d_prepare(c, n, sharing, lv());
      if (commit_ready(c, n)) {
        if (!launched) EXPECT_EQ(n, 7) << sharing;
        launched = true;
        after_commit(c, lv());
      }
    }
    EXPECT_TRUE(launched);
  }
}

TEST(Groups, FirstCommitsBeforeSteadyThenUrgency) {
  std::vector<CommitRow> ready(3);
  ready[0] = {0, false, 2, 30, 24, 1, 2.0, 500.0};   // steady, buffer 500
  ready[1] = {1, true, 0, 8, 5, 2, std::nullopt, std::nullopt};  // first commit
  ready[2] = {2, false, 1, 20, 9, 3, 1.0, 60.0};     // steady, buffer 60 < 80: urgent
  const auto g = plan_commit_groups(ready, {}, lv());
  ASSERT_EQ(g.size(), 2u);
  // Steady group holds a row at 60 ms < 80: (0, 60) sorts before the first-commit group (1, 0).
  EXPECT_EQ(g[0].key.kind, 1);
  EXPECT_EQ(g[0].rows, (std::vector<int>{0, 2}));
  EXPECT_EQ(g[1].key.kind, 0);
}

TEST(Groups, CoalescingIsInertUnderRaggedGroups) {
  // The partner key (geometry owner) never equals a ragged group key: nothing is held live.
  std::vector<CommitRow> ready = {{0, false, 2, 33, 30, 7, 2.0, 2000.0}};
  std::vector<CommitRow> partners = {{1, false, 2, 25, 24, 7, 2.0, 2000.0}};
  int held = -1;
  const auto g = plan_commit_groups(ready, partners, lv(), &held);
  EXPECT_EQ(held, 0);
  EXPECT_EQ(g.size(), 1u);
}

TEST(Groups, CoalescingHoldsWhenKeysMatch) {
  Levers l = lv();
  l.ragged_groups = false;  // geometry keys (release behaviour without the chunk caches)
  std::vector<CommitRow> ready = {{0, false, 2, 33, 30, 7, 2.0, 2000.0}};
  std::vector<CommitRow> partners = {{1, false, 2, 25, 24, 7, 2.0, 2000.0}};  // needs 2 tokens
  int held = -1;
  auto g = plan_commit_groups(ready, partners, l, &held);
  EXPECT_EQ(held, 1);
  EXPECT_TRUE(g.empty());
  ready[0].commit_index = 0;  // a first commit never waits
  ready[0].first = false;
  g = plan_commit_groups(ready, partners, l, &held);
  EXPECT_EQ(held, 0);
  ready[0].commit_index = 2;
  ready[0].reservoir_s = 0.2;  // < 0.3 s: may not wait
  g = plan_commit_groups(ready, partners, l, &held);
  EXPECT_EQ(held, 0);
}

TEST(Groups, RaggedFirstCutAtTwelve) {
  std::vector<CommitRow> ready;
  for (int i = 0; i < 14; ++i) ready.push_back({i, true, 0, 8, 5, 1, std::nullopt, std::nullopt});
  const auto g = plan_commit_groups(ready, {}, lv());
  ASSERT_EQ(g.size(), 2u);
  EXPECT_EQ(g[0].rows.size(), 12u);
  EXPECT_EQ(g[1].rows.size(), 2u);
}

TEST(Async, TwoDecodedRows) {
  EXPECT_FALSE(launch_async(1, lv()));
  EXPECT_TRUE(launch_async(2, lv()));
}

TEST(Finals, BucketsAtEightRowsOrTwiceTheShortest) {
  std::vector<std::pair<int, int>> rows;
  for (int i = 0; i < 10; ++i) rows.push_back({i, 100 + i});
  rows.push_back({10, 250});
  const auto b = final_flush_buckets(rows, lv());
  ASSERT_EQ(b.size(), 3u);
  EXPECT_EQ(b[0].size(), 8u);
  EXPECT_EQ(b[1], (std::vector<int>{8, 9}));
  EXPECT_EQ(b[2], (std::vector<int>{10}));  // 250 > 2 x 108
}

TEST(Clock, HoldReleaseAndUnderrun) {
  DeliveryClock c;
  c.hold();
  c.publish(0.0, 2400);  // 100 ms pending
  EXPECT_FALSE(c.buffer_ms(0.0).has_value());
  c.release(1.0);
  EXPECT_NEAR(*c.buffer_ms(1.0), 100.0, 1e-9);
  c.publish(1.3, 4800);  // ran dry at 1.1: underrun 200 ms, then 200 ms more
  EXPECT_EQ(c.underruns(), 1);
  EXPECT_NEAR(c.underrun_ms(), 200.0, 1e-6);
  EXPECT_NEAR(*c.buffer_ms(1.3), 200.0, 1e-6);
}

TEST(Dw, Rules) {
  DwInput in;
  in.now_s = 100.0;
  in.oldest_queued_s = 99.0;
  EXPECT_EQ(dw_due(in, lv()), DwReason::kIdle);
  in.any_realtime = true;
  in.waiting = true;
  EXPECT_EQ(dw_due(in, lv()), DwReason::kNone);
  in.waiting = false;
  in.low_buffer_ms = 1400.0;
  EXPECT_EQ(dw_due(in, lv()), DwReason::kRunway);
  in.last_whole_render_s = 99.9;  // < 200 ms gap
  EXPECT_EQ(dw_due(in, lv()), DwReason::kNone);
  in.oldest_queued_s = 89.0;  // waited 11 s >= 10 s, low >= 250
  EXPECT_EQ(dw_due(in, lv()), DwReason::kMaxWait);
  in.oldest_queued_s = 87.0;
  in.waiting = true;
  EXPECT_EQ(dw_due(in, lv()), DwReason::kHard);
  // fq: runway 500 ms.
  DwInput f;
  f.now_s = 10.0;
  f.oldest_queued_s = 9.9;
  f.any_realtime = true;
  f.low_buffer_ms = 600.0;
  EXPECT_EQ(dw_due(f, live_levers("us")), DwReason::kNone);
  EXPECT_EQ(dw_due(f, live_levers("fq")), DwReason::kRunway);
}

TEST(Interleave, Gate) {
  EXPECT_TRUE(cf_interleave_due(false, true, 0, 0.0, lv()));
  EXPECT_FALSE(cf_interleave_due(true, true, 100, std::nullopt, lv()));
  EXPECT_TRUE(cf_interleave_due(true, false, 100, std::nullopt, lv()));
  EXPECT_FALSE(cf_interleave_due(true, false, 100, 100.0, lv()));
  EXPECT_TRUE(cf_interleave_due(true, true, 1600, 300.0, lv()));
  EXPECT_FALSE(cf_interleave_due(true, true, 1600, 100.0, lv()));
}

TEST(Admission, RoundRobinOneRowPerLane) {
  std::vector<LaneView> lanes = {{true, 0}, {true, 1}, {true, 0}, {false, 0}};
  size_t cursor = 0;
  auto t = admission_take(lanes, 10, &cursor, lv());
  EXPECT_EQ(t, (std::vector<int>{0, 2}));
  cursor = 2;
  t = admission_take(lanes, 1, &cursor, lv());
  EXPECT_EQ(t, (std::vector<int>{2}));
}

TEST(Admission, CapacityModeOffersContinuingRequestsFirst) {
  // Lanes 1 and 4 are playing requests waiting for their next segment; 0, 2, 3 are new; 5 has no job ready.
  std::vector<LaneView> lanes = {{true, 0}, {true, 0}, {true, 0}, {true, 0}, {true, 0}, {false, 0}};
  std::vector<bool> cont = {false, true, false, false, true, true};
  size_t cc = 0, c = 0;
  EXPECT_EQ(admission_take_capacity(lanes, cont, 2, &cc, &c, lv()), (std::vector<int>{1, 4}));
  EXPECT_EQ(c, 0u);  // the new requests' cursor did not move
  cc = 0;
  EXPECT_EQ(admission_take_capacity(lanes, cont, 3, &cc, &c, lv()), (std::vector<int>{1, 4, 0}));
  cc = 0;
  c = 3;  // production's round-robin among the new ones, from their own cursor
  EXPECT_EQ(admission_take_capacity(lanes, cont, 4, &cc, &c, lv()), (std::vector<int>{1, 4, 3, 0}));
  // Nobody continuing: identical to production's admission_take.
  std::vector<bool> none(6, false);
  size_t c1 = 2, c2 = 2, cc2 = 0;
  EXPECT_EQ(admission_take_capacity(lanes, none, 3, &cc2, &c1, lv()), admission_take(lanes, 3, &c2, lv()));
  EXPECT_EQ(c1, c2);
}

TEST(Prefetch, Rule) {
  // 100 chars -> 5 s expected; 0.8 x = 4 s = 96000 samples.
  EXPECT_FALSE(segment_prefetch_due(false, false, 95999, 100, 1, lv()));
  EXPECT_TRUE(segment_prefetch_due(false, false, 96000, 100, 1, lv()));
  EXPECT_TRUE(segment_prefetch_due(false, true, 0, 100, 8, lv()));
  EXPECT_FALSE(segment_prefetch_due(false, true, 0, 100, 9, lv()));  // 9 + 1 + 1 > 10
  EXPECT_FALSE(segment_prefetch_due(true, true, 0, 100, 0, lv()));   // last segment
}

TEST(Seed, SaltMatchesProductionFormula) {
  // (BE32(sha256(id)[0:4]) & 0x3FFFFFFF) * 1024, values from Python hashlib.
  EXPECT_EQ(request_seed_salt("abc"), 1004491439104ull);
  EXPECT_EQ(request_seed_salt("req-1"), 349418547200ull);
}

TEST(XSched, UrgentClassFirstThenDeadline) {
  std::vector<CommitRow> ready(4);
  ready[0] = {0, false, 5, 90, 80, 1, 2.0, 900.0};
  ready[1] = {1, false, 1, 20, 9, 1, 0.5, 400.0};   // commit 1: urgent class
  ready[2] = {2, false, 3, 60, 50, 1, 0.2, 150.0};  // lowest buffer of the rest
  ready[3] = {3, true, 0, 8, 5, 1, std::nullopt, std::nullopt};  // first commit: urgent, no buffer
  const auto g = plan_commit_groups_xsched(ready, live_levers("us"));
  ASSERT_EQ(g.size(), 2u);
  EXPECT_EQ(g[0].rows, (std::vector<int>{3, 1}));
  EXPECT_EQ(g[1].rows, (std::vector<int>{2, 0}));
}
