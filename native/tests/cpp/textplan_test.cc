// Unit tests for the native text tail (lane L9b).  Expected values are
// production outputs (kittenml-api@ebcdd3f, live settings) copied from the
// replay goldens (tests/textplan/golden-v3.tar.xz); the full replay is
// `kt2n_textplan replay` (tests/textplan/README.md).  The tokenizer test runs only when
// KT2N_TOKENIZER_JSON points at the live checkpoint's tokenizer.json.
#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "kt2n/textplan.h"
#include "kt2n/textplan_request.h"

namespace t = kt2n::textplan;
using t::U32;

namespace {

U32 u(const char* s) { return t::from_utf8(s); }

TEST(Text, Utf8RoundTrip) {
  const std::string s = "a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80z";
  const U32 cps = t::from_utf8(s);
  ASSERT_EQ(cps.size(), 5u);
  EXPECT_EQ(cps[3], U'\U0001F600');
  EXPECT_EQ(t::to_utf8(cps), s);
  EXPECT_EQ(t::from_utf8("\xFF" "a"), U"�" U"a");
}

TEST(Text, PythonPredicates) {
  EXPECT_TRUE(t::py_isspace(U'\x1c'));    // CPython counts U+001C..U+001F
  EXPECT_TRUE(t::py_isspace(U'　'));
  EXPECT_FALSE(t::py_isspace(U'​'));
  EXPECT_TRUE(t::py_isalnum(U'٣'));  // Arabic-Indic three
  EXPECT_EQ(t::py_casefold(U"Straße"), U"strasse");
}

TEST(Text, ControlSpansAndMask) {
  const U32 text = u("[ANGRY] I (((really))) mean it <Laugh> [unknown] <laug");
  const t::Emphasis emphasis = t::emphasis_spans(text);
  ASSERT_EQ(emphasis.size(), 1u);
  EXPECT_EQ(emphasis[0], U"(((really)))");
  const auto spans = t::control_spans(text, &emphasis);
  ASSERT_EQ(spans.size(), 3u);
  EXPECT_EQ(spans[0], (t::Span{0, 7}));
  EXPECT_EQ(spans[1], (t::Span{10, 22}));
  EXPECT_EQ(spans[2], (t::Span{31, 38}));
  EXPECT_TRUE(t::text_has_tag(text, nullptr));
  EXPECT_EQ(t::leading_expression_tag(text), std::optional<U32>(U"[ANGRY]"));
  const t::Masked masked = t::mask_control_spans(text, emphasis);
  EXPECT_EQ(masked.text.size(), text.size());
  EXPECT_EQ(masked.text[6], U']');
  EXPECT_EQ(masked.text[0], char32_t(0xF0000));
  EXPECT_EQ(t::unmask_control_spans(masked.text, masked.restore), text);
  EXPECT_FALSE(t::speakable_outside_spans(u("[sad] <sigh>"), nullptr));
  // U+017F (long s) matches "s" under Python's re.IGNORECASE.
  EXPECT_TRUE(t::text_has_tag(u("<\xC5\xBFigh>"), nullptr));
}

TEST(Text, DemoSplitLivePlan) {
  const U32 prepared = u(
      "Harriet had lived in the house by the harbor for almost forty years. Every morning she walked "
      "down to the water with a flask of tea and a notebook. She wrote down the names of the boats, the "
      "color of the sky, and the direction of the wind. Her neighbors thought it was a strange habit, but "
      "nobody ever asked her about it. One winter, a young fisherman noticed that she had not come down "
      "for three days. He knocked on her door, and when she answered, she looked tired but happy. She "
      "handed him a stack of notebooks tied together with string. Inside were decades of careful notes "
      "about the tides and the weather. The fisherman spent the whole evening reading them by the fire. "
      "By spring, half the harbor was using her notes to plan their trips. Harriet still walks down to "
      "the water every morning, but now she has company.");
  const auto abbreviations = t::Abbreviations::from_lines("");
  const auto segments = t::plan_segments(prepared, true, {}, abbreviations);
  ASSERT_EQ(segments.size(), 3u);
  EXPECT_EQ(segments[2], u("By spring, half the harbor was using her notes to plan their trips. Harriet still "
                           "walks down to the water every morning, but now she has company."));
  EXPECT_EQ(segments[0].size(), 321u);
  EXPECT_EQ(segments[1].size(), 352u);
}

TEST(Text, DemoSplitNeverCutsInsideATag) {
  const U32 prepared = u(
      "[tender] I found your old letters in the attic this morning, tucked inside the blue suitcase we "
      "took to the coast that summer. <pause> I sat on the floor and read every one of them. You wrote "
      "about the lighthouse, and the storm that kept us inside for two days, and the dog that followed "
      "us home from the harbour. <laugh> You were so sure he belonged to someone. He never did. <sigh> I "
      "had forgotten how carefully you chose your words, how you always ended with a question so that I "
      "would have to write back. So here is my answer, thirty years late. Yes. It was the best summer of "
      "my life, and I would go back tomorrow if the road were still there.");
  const t::RequestText request = t::decide_request(prepared);
  EXPECT_TRUE(request.emotion);
  EXPECT_EQ(request.dc_mode, 0);
  const auto segments = t::plan_segments(prepared, true, request.emphasis, t::Abbreviations::from_lines(""));
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[1].substr(0, 7), U"<sigh> ");
  EXPECT_EQ(segments[0].substr(segments[0].size() - 13), U"He never did.");
}

TEST(Text, Budgets) {
  EXPECT_EQ(t::tagged_segment_budget(10), 200);
  EXPECT_EQ(t::tagged_segment_budget(380), 1672);
  EXPECT_EQ(t::tagged_segment_budget(531), 2336);
  EXPECT_EQ(t::tagged_segment_budget(600), 2341);  // cap = int(4.4 * 532) + 1
  EXPECT_EQ(t::demo_budgets({120}, 1, 0), std::vector<int>({1000}));
  EXPECT_EQ(t::demo_budgets({380, 40}, 2, 0), std::vector<int>({855, 200}));
}

TEST(Text, RealtimePhraseCut) {
  const U32 text = u("The quarterly figures came in this morning. Revenue rose by twelve percent and we are");
  EXPECT_EQ(t::realtime_phrase_cut(text, false), 44u);       // sentence end of >= 24 chars
  EXPECT_EQ(t::realtime_phrase_cut(text, false, 380, 50), 0u);  // turn rule: >= 50 chars
  EXPECT_EQ(t::realtime_phrase_cut(text, true), text.size());
  EXPECT_EQ(t::realtime_phrase_cut(u("Dr. Smith is here and ready to see you now."), false), 0u);
  EXPECT_EQ(t::r_word_safe_cut(u("hello wor"), 9), 6u);
}

TEST(Text, WsPlannerCommittedTurnIsOnePhrase) {
  t::WsPlanner planner;
  planner.feed(t::WsPlanner::EventType::kText,
               u("The quarterly figures came in this morning. Revenue rose by twelve percent."));
  // No streamed cut yet: the first sentence is under the turn rule's 50
  // characters and the second has no whitespace after it.  Wait for the 0.3 s
  // phrase idle (or more input).
  auto action = planner.step(0, true);
  ASSERT_EQ(action.kind, t::WsPlanner::Action::Kind::kWait);
  EXPECT_DOUBLE_EQ(action.until_ms, 300);
  EXPECT_FALSE(action.needs_slot);
  planner.feed(t::WsPlanner::EventType::kCommit);
  action = planner.step(1, true);
  ASSERT_EQ(action.kind, t::WsPlanner::Action::Kind::kPhrase);
  EXPECT_EQ(action.phrase.text.size(), 75u);
  EXPECT_FALSE(action.phrase.emotion);
  EXPECT_FALSE(action.phrase.continues_turn);
  // done arrives while the phrase generates: production waits for the
  // generation slot before it reads the next event.
  planner.feed(t::WsPlanner::EventType::kDone);
  action = planner.step(2, false);
  ASSERT_EQ(action.kind, t::WsPlanner::Action::Kind::kWait);
  EXPECT_TRUE(action.needs_slot);
  action = planner.step(600, true);
  EXPECT_EQ(action.kind, t::WsPlanner::Action::Kind::kDone);
}

// The socket reader's input rules (api.py receive_input).  The three scripted
// timelines with one 18,065-19,144 character append are refused by the real
// production handler with "Input message too large" (tests/textplan/
// ws_real_handler.py); the gate gives the same verdict.
TEST(Text, WsInputGateMatchesTheSocketReader) {
  using V = t::WsInputGate::Verdict;
  t::WsInputGate gate;
  EXPECT_EQ(gate.append(0), V::kSkip);  // an empty append is a no-op
  EXPECT_EQ(gate.input_characters(), 0u);
  EXPECT_EQ(gate.append(8192), V::kForward);
  EXPECT_EQ(gate.append(8193), V::kMessageTooLarge);
  EXPECT_EQ(gate.append(18065), V::kMessageTooLarge);
  EXPECT_EQ(gate.commit(), V::kForward);
  EXPECT_EQ(gate.done(), V::kForward);
  EXPECT_EQ(gate.append(5), V::kInputClosed);
  EXPECT_EQ(gate.commit(), V::kInputClosed);
  EXPECT_EQ(gate.done(), V::kInputClosed);
  // The session total is counted before the check: the crossing message fails.
  t::WsInputGate total;
  for (int i = 0; i < 32; ++i) ASSERT_EQ(total.append(8192), V::kForward);  // 262,144 exactly
  EXPECT_EQ(total.append(1), V::kInputTooLarge);
}

// The conductor facade: production's order, budgets, and ordinals.
TEST(Text, RequestPlanFacade) {
  const U32 text = u(
      "Harriet had lived in the house by the harbor for almost forty years. Every morning she walked "
      "down to the water with a flask of tea and a notebook. She wrote down the names of the boats, the "
      "color of the sky, and the direction of the wind. Her neighbors thought it was a strange habit, but "
      "nobody ever asked her about it. One winter, a young fisherman noticed that she had not come down "
      "for three days. He knocked on her door, and when she answered, she looked tired but happy. She "
      "handed him a stack of notebooks tied together with string. Inside were decades of careful notes "
      "about the tides and the weather. The fisherman spent the whole evening reading them by the fire. "
      "By spring, half the harbor was using her notes to plan their trips. Harriet still walks down to "
      "the water every morning, but now she has company.");
  const auto abbreviations = t::Abbreviations::from_lines("");
  t::RequestPlan plan(text, t::passthrough_prepare(text), abbreviations, t::BudgetRule::kStream);
  EXPECT_FALSE(plan.emotion());
  EXPECT_EQ(plan.dc_mode(), 1);
  ASSERT_EQ(plan.segments().size(), 3u);
  EXPECT_EQ(plan.segments()[0].size(), 321u);
  EXPECT_EQ(plan.segments()[1].size(), 352u);
  EXPECT_EQ(plan.total_chunks(), -1);
  EXPECT_THROW(plan.add_normalized(1, plan.segments()[1], nullptr), std::logic_error);
  const auto c0 = plan.add_normalized(0, plan.segments()[0], nullptr);
  ASSERT_EQ(c0.size(), 1u);
  EXPECT_EQ(c0[0].budget, t::tagged_segment_budget(321));
  EXPECT_EQ(c0[0].ordinal, 0);
  EXPECT_TRUE(c0[0].suffix.empty());  // no tokenizer
  EXPECT_EQ(plan.total_chunks(), 3);  // len(raw_segments) - 1 + len(segment 0's chunks)
  // Empty grammar output creates no chunks.
  EXPECT_TRUE(plan.add_normalized(1, U"  ", nullptr).empty());
  const auto c2 = plan.add_normalized(2, plan.segments()[2], nullptr);
  ASSERT_EQ(c2.size(), 1u);
  EXPECT_EQ(c2[0].ordinal, 1);
  EXPECT_EQ(c2[0].segment, 2);
  EXPECT_TRUE(plan.sealed());
  EXPECT_THROW(plan.add_normalized(3, U"x", nullptr), std::logic_error);

  // stream:false: one chunk alone gets 1000, otherwise 2.25 per character.
  t::RequestPlan one(U"Hello there.", t::passthrough_prepare(U"Hello there."), abbreviations, t::BudgetRule::kDemo);
  EXPECT_EQ(one.add_normalized(0, U"Hello there.", nullptr)[0].budget, 1000);
  t::RequestPlan demo(text, t::passthrough_prepare(text), abbreviations, t::BudgetRule::kDemo);
  EXPECT_EQ(demo.add_normalized(0, demo.segments()[0], nullptr)[0].budget, int(321 / 20.0 * 25 * 1.8));

  // A WS turn's stamp wins over the phrase text; a tag decides otherwise.
  t::RequestPlan stamped(U"just words", t::passthrough_prepare(U"just words"), abbreviations,
                         t::BudgetRule::kStream, true);
  EXPECT_TRUE(stamped.emotion());
  EXPECT_EQ(stamped.dc_mode(), 0);
  t::RequestPlan tagged(U"[joyful] hi", t::passthrough_prepare(U"[joyful] hi"), abbreviations, t::BudgetRule::kStream);
  EXPECT_TRUE(tagged.emotion());
  // Only production's ten tags count: [happy] is plain text.
  t::RequestPlan untagged(U"[happy] hi", t::passthrough_prepare(U"[happy] hi"), abbreviations, t::BudgetRule::kStream);
  EXPECT_FALSE(untagged.emotion());
}

TEST(Text, TokenizerAndPrompt) {
  const char* path = std::getenv("KT2N_TOKENIZER_JSON");
  if (path == nullptr) GTEST_SKIP() << "KT2N_TOKENIZER_JSON not set";
  const auto tok = t::Tokenizer::load(path);
  EXPECT_EQ(tok.size(), 158237u);
  EXPECT_EQ(tok.encode(std::string_view("{emo: 1}")), std::vector<int32_t>({90, 6726, 25, 220, 16, 92}));
  const auto suffix = t::target_suffix(tok, U"Hi", true);
  EXPECT_EQ(suffix.front(), 90);
  EXPECT_EQ(suffix[6], t::kTextStart);
  EXPECT_EQ(suffix[suffix.size() - 2], t::kFinalSeg);
  EXPECT_EQ(suffix.back(), t::kSpeechStart);
  t::RequestPlan plan(U"Hi.", t::passthrough_prepare(U"Hi."), t::Abbreviations::from_lines(""), t::BudgetRule::kStream);
  EXPECT_EQ(plan.add_normalized(0, U"Hi.", &tok)[0].suffix, t::target_suffix(tok, U"Hi.", false));
  // Added tokens match inside user text, as in production.
  EXPECT_EQ(tok.encode(std::string_view("[START]<|a_5|>")), std::vector<int32_t>({t::kStart, t::kAudioIdBase + 5}));
}

}  // namespace
