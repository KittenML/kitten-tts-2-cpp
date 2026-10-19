// Realtime (WebSocket) phrase planner: kitten_tts/api.py @ ebcdd3f
// (realtime_phrase_cut, _r_turn_cut, _r_word_safe_cut, _RealtimeEmotionTurn,
// batch_input_sentences' live R_WS_TURN_BATCHER branch) and
// runtime.RealtimeEmotionPolicy.safe_cut.
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "textplan_internal.h"

namespace kt2n::textplan {
namespace {

bool realtime_ends_on_abbreviation(std::u32string_view text) {
  static const std::unordered_set<U32> table = {
      U"mr", U"mrs", U"ms", U"dr", U"st", U"jr", U"sr", U"prof", U"vs", U"etc", U"inc", U"ltd",
      U"co", U"corp", U"dept", U"est", U"approx", U"jan", U"feb", U"mar", U"apr", U"jun",
      U"jul", U"aug", U"sep", U"sept", U"oct", U"nov", U"dec", U"mon", U"tue", U"tues",
      U"wed", U"thu", U"thur", U"thurs", U"fri", U"sat", U"sun", U"no", U"fig", U"vol",
      U"e.g", U"i.e", U"u.s", U"u.k", U"a.m", U"p.m"};
  const U32 stripped = py_rstrip_chars(py_rstrip(text), U"\"'’”)]");
  if (stripped.empty() || stripped.back() != U'.') return false;
  std::size_t start = stripped.size();
  while (start > 0 && !py_isspace(stripped[start - 1])) --start;
  const U32 token = py_lstrip_chars(std::u32string_view(stripped).substr(start), U"\"'“‘([");
  if (token.empty()) return false;  // token[:-1] of "" is "" -> not an abbreviation
  const U32 word = py_casefold(std::u32string_view(token).substr(0, token.size() - 1));
  return (word.size() == 1 && py_isalpha(word[0])) || table.count(word) > 0;
}

std::size_t rstripped_len(std::u32string_view s, std::size_t end) {
  while (end > 0 && py_isspace(s[end - 1])) --end;
  return end;
}
std::size_t stripped_len(std::u32string_view s, std::size_t end) {
  const std::size_t b = rstripped_len(s, end);
  std::size_t a = 0;
  while (a < b && py_isspace(s[a])) ++a;
  return b - a;
}
bool blank(std::u32string_view s) {
  for (char32_t c : s)
    if (!py_isspace(c)) return false;
  return true;
}

constexpr std::u32string_view kRealtimeTagNames[] = {
    U"angry", U"contemplative", U"excited", U"gasp", U"giggle", U"growl", U"gulp",
    U"joyful", U"laugh", U"mundane", U"nervous", U"pause", U"sad", U"scoff",
    U"sigh", U"sob", U"stern", U"surprised", U"tender", U"um"};

bool at_dollar(std::u32string_view s, std::size_t p) {
  return p == s.size() || (p + 1 == s.size() && s[p] == U'\n');
}

// _realtime_open_span(head)
std::optional<std::size_t> realtime_open_span(std::u32string_view head) {
  const std::size_t n = head.size();
  // \(\(\([^()\n]{0,80}\){0,2}$
  for (std::size_t i = 0; i + 3 <= n; ++i) {
    if (head[i] != U'(' || head[i + 1] != U'(' || head[i + 2] != U'(') continue;
    std::size_t k = i + 3;
    while (k < n && head[k] != U'(' && head[k] != U')' && head[k] != U'\n') ++k;
    if (k - (i + 3) > 80) continue;
    std::size_t p = k;
    int closes = 0;
    while (p < n && head[p] == U')' && closes < 2) ++p, ++closes;
    if (at_dollar(head, p)) return i;
  }
  // [\[<]([A-Za-z]{0,13})$
  for (std::size_t i = 0; i < n; ++i) {
    if (head[i] != U'[' && head[i] != U'<') continue;
    std::size_t k = i + 1;
    while (k < n && ((head[k] >= 'a' && head[k] <= 'z') || (head[k] >= 'A' && head[k] <= 'Z'))) ++k;
    const std::size_t letters = k - i - 1;
    if (letters > 13 || !at_dollar(head, k)) continue;
    U32 prefix(head.substr(i + 1, letters));
    for (char32_t& c : prefix)
      if (c >= 'A' && c <= 'Z') c += 32;
    for (auto name : kRealtimeTagNames)
      if (name.substr(0, std::min(name.size(), prefix.size())) == prefix && name.size() >= prefix.size())
        return i;
    return std::nullopt;  // the only candidate position (no [ or < can follow it)
  }
  return std::nullopt;
}

}  // namespace

std::size_t realtime_phrase_cut(std::u32string_view text, bool final, int max_characters,
                                int min_sentence_characters) {
  if (blank(text)) return 0;
  const std::size_t max_c = std::size_t(max_characters);
  if (final && rstripped_len(text, text.size()) <= max_c) return text.size();
  std::size_t best = 0;
  for (const Match& m : sentence_ends(text)) {
    const std::size_t end = rstripped_len(text, m.end);
    if (end > max_c) break;
    if (stripped_len(text, end) < std::size_t(min_sentence_characters)) continue;
    if (realtime_ends_on_abbreviation(text.substr(0, end))) continue;
    best = m.end;
  }
  if (best) return best;
  if (!final && rstripped_len(text, text.size()) < std::size_t(std::max(1, max_characters / 2))) return 0;
  const std::u32string_view window = text.substr(0, std::min(text.size(), max_c + 1));
  const std::size_t floor = std::max<std::size_t>(1, rstripped_len(window, window.size()) / 3);
  std::size_t clause = 0;
  for (const Match& m : clause_breaks(window)) {
    const char32_t c = text[m.start];
    const std::size_t end = (c == U',' || c == U';' || c == U':') ? m.start + 1 : m.start;
    if (floor <= end && end <= max_c) clause = end;
  }
  if (clause) return clause;
  const U32 trimmed = py_rstrip(window);
  const long word = py_rfind(trimmed, U" ", max_c + 1);
  if (word >= long(floor)) return std::size_t(word);
  return std::min(text.size(), max_c);
}

std::pair<std::size_t, bool> r_turn_cut(std::u32string_view pending, std::size_t committed_upto,
                                        bool input_done, const Settings& settings) {
  if (committed_upto > 0) return {committed_upto, true};
  if (input_done) return {pending.size(), true};
  return {realtime_phrase_cut(pending, false, settings.ws_max_phrase_characters,
                              settings.ws_turn_min_sentence_characters),
          false};
}

std::size_t r_word_safe_cut(std::u32string_view text, std::size_t cut) {
  if (cut == 0 || cut > text.size() || !py_isalnum(text[cut - 1])) return cut;
  if (cut < text.size() && !py_isalnum(text[cut])) return cut;
  const long space = std::max(py_rfind(text, U" ", cut), py_rfind(text, U"\n", cut));
  return space > 0 ? std::size_t(space) + 1 : 0;
}

std::size_t emotion_safe_cut(std::u32string_view text, std::size_t cut, bool final,
                             std::optional<int> max_characters) {
  if (cut == 0) return cut;
  if (auto span = cut_inside_control_span(text, cut, nullptr)) {
    const std::u32string_view before = text.substr(0, std::min(span->start, text.size()));
    if (max_characters && span->end > std::size_t(*max_characters) && !blank(before) &&
        speakable_outside_spans(before, nullptr)) {
      cut = span->start;
    } else {
      cut = span->end;
    }
  }
  if (final) return cut;
  if (auto opener = realtime_open_span(text.substr(0, std::min(cut, text.size())))) cut = *opener;
  const std::u32string_view head = text.substr(0, std::min(cut, text.size()));
  if (!blank(head) && !speakable_outside_spans(head, nullptr)) return 0;
  return cut;
}

// ---- _RealtimeEmotionTurn (request level) ------------------------------------
void EmotionTurn::reset() {
  applied_ = false;
  lead_tag_.reset();
  phrases_ = 0;
}

std::size_t EmotionTurn::safe_cut(std::u32string_view text, std::size_t cut, bool final) const {
  return emotion_safe_cut(text, cut, final, max_characters_);
}

void EmotionTurn::commit(std::u32string_view pending) {
  if (pending.empty()) {
    turn_ends_.clear();
    reset();
  } else if (turn_ends_.empty() || turn_ends_.back() != pending.size()) {
    turn_ends_.push_back(pending.size());
  }
}

bool EmotionTurn::decision(std::u32string_view pending) const {
  const std::size_t end = turn_ends_.empty() ? pending.size() : turn_ends_.front();
  return applied_ || text_has_tag(pending.substr(0, std::min(end, pending.size())), nullptr);
}

std::size_t EmotionTurn::turn_cut(std::u32string_view pending, std::size_t cut) const {
  if (cut == 0) return cut;
  std::vector<std::size_t> inside;
  for (std::size_t end : turn_ends_)
    if (0 < end && end < cut) inside.push_back(end);
  if (inside.empty()) return cut;
  const bool current = decision(pending);
  std::vector<std::size_t> bounds = turn_ends_;
  bounds.push_back(pending.size());
  for (std::size_t number = 0; number < inside.size(); ++number) {
    const std::size_t end = inside[number];
    const std::size_t to = bounds[number + 1];
    const std::u32string_view following =
        end < to && end <= pending.size() ? pending.substr(end, std::min(to, pending.size()) - end)
                                          : std::u32string_view();
    if (!blank(following) && text_has_tag(following, nullptr) != current) return end;
  }
  return cut;
}

void EmotionTurn::consumed(std::size_t count) {
  if (count == 0) return;
  bool closed = false;
  std::vector<std::size_t> after;
  for (std::size_t end : turn_ends_) {
    if (end <= count) closed = true;
    if (end > count) after.push_back(end - count);
  }
  turn_ends_ = std::move(after);
  if (closed) reset();
}

EmotionTurn::Stamp EmotionTurn::stamp(std::u32string_view text_batch, std::u32string_view pending) {
  Stamp out;
  const std::size_t cut = text_batch.size();
  std::vector<std::size_t> closed, after;
  for (std::size_t end : turn_ends_) {
    if (end <= cut) closed.push_back(end);
    if (end > cut) after.push_back(end - cut);
  }
  const std::u32string_view head = closed.empty() ? text_batch : text_batch.substr(0, closed.front());
  if (phrases_ == 0) {
    lead_tag_ = leading_expression_tag(head);
  } else if (lead_tag_) {
    out.carry_tag = lead_tag_;
  }
  // request level
  std::u32string_view rest;
  if (closed.empty()) rest = after.empty() ? pending : pending.substr(0, std::min(after.front(), pending.size()));
  applied_ = applied_ || text_has_tag(head, nullptr) || text_has_tag(rest, nullptr);
  out.emotion = applied_;
  phrases_ += 1;
  turn_ends_ = after;
  if (!closed.empty()) {
    const std::u32string_view tail = text_batch.substr(closed.back());
    reset();
    if (!blank(tail)) {
      lead_tag_ = leading_expression_tag(tail);
      phrases_ = 1;
      const std::u32string_view rest2 =
          after.empty() ? pending : pending.substr(0, std::min(after.front(), pending.size()));
      applied_ = text_has_tag(tail, nullptr) || text_has_tag(rest2, nullptr);
    }
  }
  return out;
}

// ---- The sentence batcher ----------------------------------------------------
WsInputGate::Verdict WsInputGate::append(std::size_t code_points) {
  if (closed_) return Verdict::kInputClosed;
  if (code_points == 0) return Verdict::kSkip;
  if (code_points > settings_.ws_max_message_characters) return Verdict::kMessageTooLarge;
  input_characters_ += code_points;
  if (input_characters_ > settings_.ws_max_input_characters) return Verdict::kInputTooLarge;
  return Verdict::kForward;
}

WsInputGate::Verdict WsInputGate::done() {
  if (closed_) return Verdict::kInputClosed;
  closed_ = true;
  return Verdict::kForward;
}

WsPlanner::WsPlanner(const Settings& settings)
    : settings_(settings), emo_turn_(settings.ws_max_phrase_characters) {}

void WsPlanner::feed(EventType type, std::u32string_view value) {
  queue_.push_back({type, U32(value)});
}

bool WsPlanner::apply(const Queued& event) {
  switch (event.type) {
    case EventType::kText:
      pending_ += event.value;
      return false;
    case EventType::kCommit:
      committed_upto_ = pending_.size();
      emo_turn_.commit(pending_);
      return !pending_.empty();
    case EventType::kDone:
      input_done_ = true;
      return true;
  }
  return false;
}

WsPlanner::Action WsPlanner::step(double now_ms, bool slot_free) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  std::size_t queue_head = 0;
  auto pop = [&](Queued& out) {
    if (queue_head >= queue_.size()) return false;
    out = std::move(queue_[queue_head++]);
    return true;
  };
  auto finish = [&](Action action) {
    queue_.erase(queue_.begin(), queue_.begin() + long(queue_head));
    return action;
  };
  for (int guard = 0; guard < 1000000; ++guard) {
    std::size_t cut = 0;
    bool emo_final = false;
    bool flush = false;
    if (mode_ == Mode::kTop) {
      if (input_done_ && pending_.empty()) {
        done_ = true;
        Action action;
        action.kind = Action::Kind::kDone;
        return finish(action);
      }
      if (phrase_index_ > 0 && !slot_held_) {
        if (!slot_free) {
          Action action;
          action.kind = Action::Kind::kWait;
          action.until_ms = kInf;
          action.needs_slot = true;
          return finish(action);
        }
        slot_held_ = true;
      }
      flush = input_done_;
      Queued event;
      while (pop(event)) flush = apply(event) || flush;
      std::tie(cut, emo_final) = r_turn_cut(pending_, committed_upto_, input_done_, settings_);
      if (settings_.ws_cut_grace_ms > 0 && cut && !emo_final) {
        mode_ = Mode::kGrace;
        deadline_ms_ = now_ms + settings_.ws_cut_grace_ms;
        flush_ = flush;
      }
    }
    if (mode_ == Mode::kGrace) {
      flush = flush_;
      Queued event;
      bool waiting = false;
      // Python: while not flush: wait for the next event until the deadline
      // (an event at or before the deadline wins), apply it.  Every queued
      // event arrived at or before ``now_ms`` <= the deadline.
      while (!flush) {
        if (pop(event)) {
          flush = apply(event) || flush;
          continue;
        }
        if (now_ms < deadline_ms_) waiting = true;
        break;
      }
      if (waiting) {
        flush_ = flush;
        Action action;
        action.kind = Action::Kind::kWait;
        action.until_ms = deadline_ms_;
        return finish(action);
      }
      mode_ = Mode::kTop;
      std::tie(cut, emo_final) = r_turn_cut(pending_, committed_upto_, input_done_, settings_);
    }
    if (mode_ == Mode::kTop) {
      if (cut) cut = emo_turn_.safe_cut(pending_, cut, emo_final);
      if (!cut && !(flush && !pending_.empty() && blank(pending_))) {
        if (input_done_ && pending_.empty()) continue;
        mode_ = Mode::kIdle;
        deadline_ms_ = pending_.empty() ? kInf : now_ms + settings_.ws_phrase_idle_ms;
      }
    }
    if (mode_ == Mode::kIdle) {
      Queued event;
      if (pop(event)) {
        idle_timeouts_ = 0;
        apply(event);
        mode_ = Mode::kTop;
        continue;
      }
      if (now_ms < deadline_ms_) {
        Action action;
        action.kind = Action::Kind::kWait;
        action.until_ms = deadline_ms_;
        return finish(action);
      }
      mode_ = Mode::kTop;
      cut = realtime_phrase_cut(pending_, true, settings_.ws_max_phrase_characters,
                                settings_.ws_min_sentence_characters);
      idle_timeouts_ += 1;
      if (idle_timeouts_ < 2) cut = r_word_safe_cut(pending_, cut);
      if (!cut) continue;
      const std::size_t held = emo_turn_.safe_cut(pending_, cut, false);
      cut = held ? held : emo_turn_.safe_cut(pending_, cut, true);
    }
    if (cut) cut = emo_turn_.turn_cut(pending_, cut);
    if (!cut || blank(std::u32string_view(pending_).substr(0, std::min(cut, pending_.size())))) {
      const std::size_t dropped = pending_.size();
      if (cut) pending_ = pending_.substr(std::min(cut, pending_.size()));
      if (blank(pending_)) pending_.clear();
      emo_turn_.consumed(dropped - pending_.size());
      const std::size_t gone = dropped - pending_.size();
      committed_upto_ = committed_upto_ > gone ? committed_upto_ - gone : 0;
      continue;
    }
    const std::size_t take = std::min(cut, pending_.size());
    Phrase phrase;
    phrase.text = pending_.substr(0, take);
    pending_ = pending_.substr(take);
    const bool was_committed = committed_upto_ > 0;
    committed_upto_ = committed_upto_ > cut ? committed_upto_ - cut : 0;
    phrase.continues_turn = turn_phrases_ > 0;
    turn_phrases_ = (was_committed && committed_upto_ == 0) ? 0 : turn_phrases_ + 1;
    idle_timeouts_ = 0;
    phrase_index_ += 1;
    // The first phrase takes the (free) slot here; later ones took it at the top.
    slot_held_ = false;
    const EmotionTurn::Stamp stamp = emo_turn_.stamp(phrase.text, pending_);
    phrase.emotion = stamp.emotion;
    phrase.carry_tag = stamp.carry_tag;
    Action action;
    action.kind = Action::Kind::kPhrase;
    action.phrase = std::move(phrase);
    return finish(action);
  }
  Action action;
  action.kind = Action::Kind::kDone;
  return finish(action);
}

}  // namespace kt2n::textplan
