// kt2n text facade for the conductor (lane L9b): see include/kt2n/textplan_request.h.
#include "kt2n/textplan_request.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace kt2n::textplan {

Prepared passthrough_prepare(std::u32string_view request_text) {
  Prepared p;
  p.text = U32(request_text);
  p.wfst = false;
  p.emphasis = emphasis_spans(request_text);
  return p;
}

RequestPlan::RequestPlan(std::u32string_view request_text, Prepared prepared,
                         const Abbreviations& abbreviations, BudgetRule rule,
                         std::optional<bool> emotion_preset, const Settings& settings)
    : settings_(settings), rule_(rule), prepared_(std::move(prepared)) {
  // Step 1 (runtime.py:20505-20517): the request-level {emo: 1} decision and
  // the demo-chunk mode, from the WHOLE request text (a WS turn's stamp wins).
  decision_ = decide_request(request_text, emotion_preset);
  // Steps 2-3 (plan_normalization with demo chunks): the planner works on the
  // PREPARED text with prepare()'s emphasis spans.
  segments_ = plan_segments(prepared_.text, prepared_.wfst, prepared_.emphasis, abbreviations, settings_);
}


std::vector<LmChunk> RequestPlan::add_normalized(std::size_t index, std::u32string_view normalized,
                                                 const Tokenizer* tokenizer) {
  if (index != next_ || next_ >= segments_.size())
    throw std::logic_error("RequestPlan::add_normalized: segment " + std::to_string(index) +
                           " out of order (next " + std::to_string(next_) + " of " +
                           std::to_string(segments_.size()) + ")");
  // normalize_planned_segment's re-split (split_text(demo=True)) with the
  // request's prepared emphasis spans.
  std::vector<U32> pieces = resplit_segment(normalized, prepared_.emphasis, settings_);
  // control.q_demo_chunks (runtime.py:20643): set once, from segment 0.
  if (index == 0) total_chunks_ = int(segments_.size()) - 1 + int(pieces.size());
  std::vector<int> budgets;
  if (rule_ == BudgetRule::kDemo) {
    std::vector<std::size_t> lens;
    lens.reserve(pieces.size());
    for (const U32& p : pieces) lens.push_back(p.size());
    // start_index = len(lane.jobs) before this batch (runtime.py:19806).
    budgets = demo_budgets(lens, total_chunks_, chunks_, settings_);
  }
  std::vector<LmChunk> out;
  out.reserve(pieces.size());
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    LmChunk c;
    c.budget = rule_ == BudgetRule::kDemo ? budgets[i] : tagged_segment_budget(pieces[i].size(), settings_);
    if (tokenizer != nullptr) c.suffix = target_suffix(*tokenizer, pieces[i], decision_.emotion);
    c.ordinal = chunks_ + int(i);
    c.segment = int(index);
    c.text = std::move(pieces[i]);
    out.push_back(std::move(c));
  }
  chunks_ += int(out.size());
  ++next_;
  return out;
}


}  // namespace kt2n::textplan
