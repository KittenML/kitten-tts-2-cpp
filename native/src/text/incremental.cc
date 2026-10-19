#include "incremental.h"

#include <algorithm>
#include <deque>
#include <functional>

#include "pyre.h"
#include "unicode.h"

namespace kt2n::text {
namespace {

const Regex& sentence_end() {
  static const Regex r(py_pattern(R"((?<=[.!?\x{3002}\x{ff01}\x{ff1f}\x{061f}\x{0964}])["\x{2019}\x{201d}')\]]*\s+)"));
  return r;
}
const Regex& mask_range() {
  static const Regex r(R"([\x{F0000}-\x{FFFFF}])");
  return r;
}

}  // namespace

std::vector<std::string> merge_abbreviation_boundaries(const SpeechTextNormalizer& n,
                                                       const std::vector<std::string>& segments,
                                                       const IncrementalPolicy& policy,
                                                       const std::vector<std::string>& emphasis) {
  const long cap = std::max<long>(1, static_cast<long>(policy.chunk_characters) * std::max(1, policy.merge_limit));
  std::deque<std::string> queue;
  for (const auto& s : segments) {
    if (!s.empty()) queue.push_back(s);
  }
  std::vector<std::pair<std::string, std::string>> restore;
  if (n.emo_enabled()) {
    bool any = false;
    for (const auto& s : queue) any |= !n.control_spans(s, &emphasis).empty();
    if (any) {
      for (auto& s : queue) {
        if (mask_range().search(s)) continue;
        const auto spans = n.control_spans(s, &emphasis);
        if (spans.empty() || restore.size() + spans.size() > 0xFFFF0 - 0xF0000) continue;
        std::string masked;
        size_t pos = 0;
        for (size_t i = 0; i < spans.size(); ++i) {
          const std::u32string span = to_u32(std::string_view(s).substr(spans[i].first, spans[i].second - spans[i].first));
          std::u32string mark(span.size() - 1, static_cast<char32_t>(0xF0000 + restore.size()));
          mark.push_back(span.back());
          masked += s.substr(pos, spans[i].first - pos);
          masked += to_utf8(mark);
          restore.emplace_back(to_utf8(mark), to_utf8(span));
          pos = spans[i].second;
        }
        masked += s.substr(pos);
        s = std::move(masked);
      }
    }
  }
  std::vector<std::string> merged;
  while (!queue.empty()) {
    std::string pending = queue.front();
    queue.pop_front();
    while (!queue.empty() && static_cast<long>(cp_len(pending)) < cap && n.ends_on_abbreviation(pending)) {
      std::vector<std::string> sentences;
      for (const auto& part : sentence_end().split(queue.front())) {
        std::string p = py_strip(part);
        if (!p.empty()) sentences.push_back(std::move(p));
      }
      if (sentences.size() <= 1) {
        pending = py_strip(pending + " " + queue.front());
        queue.pop_front();
        continue;
      }
      pending = py_strip(pending + " " + sentences[0]);
      queue.front() = join(std::vector<std::string>(sentences.begin() + 1, sentences.end()), " ");
    }
    merged.push_back(std::move(pending));
  }
  if (!restore.empty()) {
    for (auto& s : merged) {
      for (const auto& [m, span] : restore) s = py_replace(s, m, span);
    }
  }
  return merged;
}

SegmentResult normalize_planned_segment(SpeechTextNormalizer& n, const PreparedText& prepared,
                                        const std::vector<std::string>& raw_segments, size_t index,
                                        const IncrementalPolicy& policy, int idle_instances,
                                        int segment_zero_in_flight, const std::atomic<bool>* cancelled) {
  SegmentResult result;
  const std::vector<std::string>& emphasis = prepared.emphasis;
  const bool last = index + 1 == raw_segments.size();
  const int priority = index == 0 ? 0 : 1;  // grammar_priority
  auto norm = [&](const std::string& piece, bool capitalize, bool terminate) {
    NormalizeOptions o;
    o.wfst = prepared.wfst;
    o.voice_language = prepared.voice_language;
    o.normalize = prepared.normalize;
    o.voice_masked_values = prepared.voice_masked_values;
    o.capitalize = capitalize;
    o.terminate = terminate;
    o.emphasis = &emphasis;
    o.priority = priority;
    return n.normalize_prepared(piece, o);
  };
  auto join_nonempty = [](const std::vector<std::string>& parts) {
    std::vector<std::string> keep;
    for (const auto& p : parts) {
      if (!p.empty()) keep.push_back(p);
    }
    return join(keep, " ");
  };
  // N_SPLIT: segment 0's grammar call in parallel pieces (runtime.n_split_pieces).
  if (index == 0 && policy.n_split_characters > 0 && prepared.wfst) {
    int workers = policy.n_pool_workers;
    const int budget = segment_zero_in_flight > 1 ? 1 : std::max(1, idle_instances);
    workers = std::min(workers, budget);
    if (workers > 1) {
      const std::vector<std::string> sentences = n.grammar_pieces(raw_segments[0], policy.piece_characters, emphasis);
      std::vector<std::string> pieces;
      for (const auto& sentence : sentences) {
        for (auto& part : n.split_for_grammar(sentence, policy.n_split_characters, workers)) pieces.push_back(part);
      }
      if (pieces.size() > sentences.size()) {
        const size_t last_piece = pieces.size() - 1;
        const bool terminate = raw_segments.size() == 1;
        // Same per-piece call as production's normalize_pieces (so the same
        // output); only the scheduling differs: pieces the fast path answers
        // without the grammar run inline on this thread (microseconds), and
        // only grammar pieces fan out to the persistent helpers.
        std::vector<std::string> spoken(pieces.size());
        std::vector<size_t> light, heavy;
        for (size_t k = 0; k < pieces.size(); ++k) {
          NormalizeOptions probe;
          probe.wfst = true;
          probe.emphasis = &emphasis;
          (n.needs_grammar(pieces[k], probe) ? heavy : light).push_back(k);
        }
        auto piece = [&](size_t k) { spoken[k] = norm(pieces[k], k == 0, terminate && k == last_piece); };
        std::vector<std::function<void()>> tasks;
        tasks.emplace_back([&] {
          for (size_t k : light) piece(k);
          if (!heavy.empty()) piece(heavy[0]);
        });
        for (size_t h = 1; h < heavy.size(); ++h) tasks.emplace_back([&, h] { piece(heavy[h]); });
        n.run_parallel(tasks);
        result.text = join_nonempty(spoken);
        result.grammar_pieces = static_cast<int>(pieces.size());
        result.n_split = true;
        return result;
      }
    }
  }
  const std::vector<std::string> pieces =
      prepared.wfst ? n.grammar_pieces(raw_segments[index], policy.piece_characters, emphasis)
                    : std::vector<std::string>{};
  if (pieces.size() <= 1) {
    result.text = norm(raw_segments[index], index == 0, last);
    return result;
  }
  std::vector<std::string> spoken;
  for (size_t k = 0; k < pieces.size(); ++k) {
    if (k && cancelled && cancelled->load()) break;
    spoken.push_back(norm(pieces[k], index == 0 && k == 0, last && k + 1 == pieces.size()));
  }
  result.text = join_nonempty(spoken);
  result.grammar_pieces = static_cast<int>(pieces.size());
  return result;
}

SegmentResult normalize_planned_segment(SpeechTextNormalizer& n, const PreparedText& prepared,
                                        const std::vector<std::string>& raw_segments, size_t index,
                                        const IncrementalPolicy& policy, const std::atomic<bool>* cancelled) {
  if (index != 0) return normalize_planned_segment(n, prepared, raw_segments, index, policy, 1, 1, cancelled);
  SpeechTextNormalizer::SegmentZeroGuard guard(n);
  // split_budget() reads the in-flight count including this request.
  const int budget = n.split_budget();
  return normalize_planned_segment(n, prepared, raw_segments, 0, policy, budget, 1, cancelled);
}

}  // namespace kt2n::text
