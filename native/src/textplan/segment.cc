// Segmentation, abbreviations and token budgets
// (kitten_tts_09/runtime.py + normalization.py @ ebcdd3f, live settings).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "textplan_internal.h"

namespace kt2n::textplan {

bool in_mask_range(char32_t c);

namespace {

constexpr std::u32string_view kTerminal = U".!?,;:。！？，；：؟،।";
constexpr std::u32string_view kClosers = U"\"’”')]";

// _merge_short_chunks(parts, max_characters, minimum)
std::vector<U32> merge_short_chunks(std::vector<U32> output, int max_characters, int minimum) {
  if (output.empty() || minimum <= 0) return output;
  const std::size_t limit = std::size_t(int(max_characters * 1.4));
  bool changed = true;
  while (changed && output.size() > 1) {
    changed = false;
    for (std::size_t index = 0; index < output.size(); ++index) {
      const U32 chunk = output[index];
      if (chunk.size() >= std::size_t(minimum)) continue;
      const bool previous_ok =
          index > 0 && output[index - 1].size() + chunk.size() + 1 <= limit;
      const bool next_ok =
          index + 1 < output.size() && chunk.size() + output[index + 1].size() + 1 <= limit;
      if (previous_ok && (!next_ok || output[index - 1].size() <= output[index + 1].size())) {
        output[index - 1] = py_rstrip(output[index - 1]) + U" " + py_lstrip(chunk);
        output.erase(output.begin() + long(index));
      } else if (next_ok) {
        output[index] = py_rstrip(chunk) + U" " + py_lstrip(output[index + 1]);
        output.erase(output.begin() + long(index) + 1);
      } else {
        continue;
      }
      changed = true;
      break;
    }
  }
  return output;
}

// _outside_masked_span(value, cut)
std::size_t outside_masked_span(std::u32string_view value, std::size_t cut) {
  if (!(0 < cut && cut < value.size())) return cut;
  const char32_t before = value[cut - 1];
  if (!in_mask_range(before)) return cut;
  std::size_t start = cut - 1;
  while (start > 0 && value[start - 1] == before) --start;
  if (start > 0) return start;
  std::size_t end = cut;
  while (end < value.size() && value[end] == before) ++end;
  return std::min(value.size(), end + 1);
}

// _pack_sentences(value, max_characters, minimum, span_safe)
std::vector<U32> pack_sentences(std::u32string_view value, int max_characters, int minimum,
                                bool span_safe) {
  if (!max_characters || value.size() <= std::size_t(max_characters)) return {U32(value)};
  const std::size_t max_c = std::size_t(max_characters);
  std::vector<U32> output;
  U32 pending;
  for (U32 sentence : sentences_of(value)) {
    while (sentence.size() > max_c) {
      const long clause_cut = std::max({py_rfind(sentence, U", ", max_c),
                                        py_rfind(sentence, U"; ", max_c),
                                        py_rfind(sentence, U" — ", max_c)});
      std::size_t cut;
      if (clause_cut > long(max_c / 3)) {
        cut = std::size_t(clause_cut) + 1;
      } else {
        const long word_cut = py_rfind(sentence, U" ", max_c + 1);
        cut = word_cut > long(max_c / 3) ? std::size_t(word_cut) : max_c;
        if (span_safe) cut = outside_masked_span(sentence, cut);
      }
      U32 head = py_strip(std::u32string_view(sentence).substr(0, cut));
      sentence = py_strip(std::u32string_view(sentence).substr(std::min(cut, sentence.size())));
      if (!pending.empty()) {
        output.push_back(pending);
        pending.clear();
      }
      output.push_back(std::move(head));
    }
    if (pending.empty()) {
      pending = sentence;
    } else if (pending.size() + sentence.size() + 1 <= max_c) {
      pending += U" ";
      pending += sentence;
    } else {
      output.push_back(pending);
      pending = sentence;
    }
  }
  if (!pending.empty()) output.push_back(pending);
  std::vector<U32> merged = merge_short_chunks(std::move(output), max_characters, minimum);
  for (U32& chunk : merged) chunk = ensure_terminal_punctuation(chunk);
  return merged;
}

// _merge_control_only_segments(segments, emphasis)
std::vector<U32> merge_control_only_segments(std::vector<U32> output, const Emphasis& emphasis) {
  std::size_t index = 0;
  while (index < output.size() && output.size() > 1) {
    const U32 segment = output[index];
    if (speakable_outside_spans(segment, &emphasis)) {
      ++index;
      continue;
    }
    // segment.strip().rstrip(",").rstrip()
    U32 lone = py_rstrip(py_rstrip_chars(py_strip(segment), U","));
    if (index + 1 < output.size()) {
      output[index + 1] = lone + U" " + py_lstrip(output[index + 1]);
      output.erase(output.begin() + long(index));
    } else {
      output[index - 1] = py_rstrip(output[index - 1]) + U" " + lone;
      output.erase(output.begin() + long(index));
      index = index > 0 ? index - 1 : 0;
    }
  }
  return output;
}

std::pair<int, int> demo_size(int chars, const Settings& s) {
  if (chars <= 0 || chars == s.demo_chunk_characters)
    return {s.demo_chunk_characters, s.demo_chunk_min_characters};
  // max(1, int(round(size * 130 / 380))): Python round() is half-to-even.
  const double scaled = double(chars) * s.demo_chunk_min_characters / s.demo_chunk_characters;
  return {chars, std::max(1, int(std::nearbyint(scaled)))};
}

}  // namespace

U32 ensure_terminal_punctuation(std::u32string_view chunk) {
  U32 value = py_rstrip(chunk);
  if (value.empty()) return value;
  const char32_t last = value.back();
  if (kTerminal.find(last) != std::u32string_view::npos ||
      kClosers.find(last) != std::u32string_view::npos)
    return value;
  value.push_back(U',');
  return value;
}

std::vector<U32> demo_split_text(std::u32string_view text, const Emphasis& emphasis, int chars,
                                 const Settings& settings) {
  const U32 value = py_strip(text);
  if (value.empty()) return {};
  const auto [size, minimum] = demo_size(chars, settings);
  const Masked masked = mask_control_spans(value, emphasis);
  std::vector<U32> parts = pack_sentences(masked.text, size, minimum, !masked.restore.empty());
  if (masked.restore.empty()) return parts;
  for (U32& part : parts) part = unmask_control_spans(std::move(part), masked.restore);
  return merge_control_only_segments(std::move(parts), emphasis);
}

std::vector<U32> emo_segments(const std::vector<U32>& segments, const Emphasis& emphasis) {
  bool any = false;
  for (const U32& segment : segments)
    if (!control_spans(segment, &emphasis).empty()) {
      any = true;
      break;
    }
  if (!any) return segments;
  return merge_control_only_segments(segments, emphasis);
}

// ---- Abbreviations -----------------------------------------------------------
struct Abbreviations::Impl {
  std::unordered_set<U32> whitelist;
};

namespace {
const std::unordered_set<U32>& builtin_abbreviations() {
  static const std::unordered_set<U32> table = {
      U"mr.", U"mrs.", U"ms.", U"dr.", U"prof.", U"rev.", U"fr.", U"sr.", U"jr.",
      U"st.", U"sgt.", U"lt.", U"capt.", U"col.", U"gen.", U"gov.", U"pres.",
      U"supt.", U"hon.", U"messrs.", U"mme.", U"mlle.",
      U"jan.", U"feb.", U"mar.", U"apr.", U"jun.", U"jul.", U"aug.", U"sep.",
      U"sept.", U"oct.", U"nov.", U"dec.",
      U"mon.", U"tue.", U"tues.", U"wed.", U"thu.", U"thur.", U"thurs.", U"fri.",
      U"sat.", U"sun.",
      U"ave.", U"blvd.", U"rd.", U"ln.", U"ct.", U"dr", U"sq.", U"mt.", U"ft.",
      U"pt.", U"apt.", U"bldg.", U"dept.", U"univ.",
      U"inc.", U"ltd.", U"co.", U"corp.", U"llc.", U"plc.", U"est.", U"no.",
      U"nos.", U"vol.", U"vols.", U"ed.", U"eds.", U"fig.", U"figs.", U"ch.",
      U"chap.", U"sec.", U"pp.", U"approx.", U"min.", U"max.", U"avg.", U"qty.",
      U"etc.", U"vs.", U"viz.", U"cf.", U"al.", U"ibid.", U"op.", U"seq."};
  return table;
}
bool ascii_letter(char32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
}  // namespace

Abbreviations Abbreviations::from_lines(std::string_view whitelist) {
  auto impl = std::make_shared<Impl>();
  std::size_t position = 0;
  while (position <= whitelist.size()) {
    std::size_t end = whitelist.find('\n', position);
    if (end == std::string_view::npos) end = whitelist.size();
    std::string_view line = whitelist.substr(position, end - position);
    if (!line.empty()) impl->whitelist.insert(from_utf8(line));
    position = end + 1;
  }
  Abbreviations out;
  out.impl_ = std::move(impl);
  return out;
}

Abbreviations Abbreviations::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read abbreviation whitelist: " + path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return from_lines(buffer.str());
}

std::size_t Abbreviations::whitelist_size() const { return impl_ ? impl_->whitelist.size() : 0; }

bool Abbreviations::ends_on_abbreviation(std::u32string_view text) const {
  const U32 stripped = py_rstrip_chars(py_rstrip(text), U"\"'’”)]");
  if (stripped.empty() || stripped.back() != U'.') return false;
  std::size_t start = stripped.size();
  while (start > 0 && !py_isspace(stripped[start - 1])) --start;
  const U32 token = py_lstrip_chars(std::u32string_view(stripped).substr(start), U"\"'“‘([");
  if (token.empty()) return false;
  for (std::size_t i = 0; i + 2 < token.size(); ++i)
    if (ascii_letter(token[i]) && token[i + 1] == U'.' && ascii_letter(token[i + 2])) return true;
  const U32 word = py_rstrip_chars(token, U".");
  if (word.size() == 1 && py_isalpha(word[0])) return true;
  if (!py_isalpha_str(word)) return false;
  const U32 token_folded = py_casefold(token);
  if (impl_ && !impl_->whitelist.empty()) {
    if (impl_->whitelist.count(token_folded) || impl_->whitelist.count(py_casefold(word))) return true;
  }
  return builtin_abbreviations().count(token_folded) > 0;
}

std::vector<U32> merge_abbreviation_boundaries(const std::vector<U32>& segments,
                                               const Abbreviations& abbreviations,
                                               int max_characters, const Emphasis& emphasis,
                                               int limit) {
  const std::size_t cap = std::size_t(std::max(1, max_characters * std::max(1, limit)));
  std::vector<U32> queue;
  for (const U32& segment : segments)
    if (!segment.empty()) queue.push_back(segment);
  std::vector<std::pair<U32, U32>> restore;
  bool any = false;
  for (const U32& segment : queue)
    if (!control_spans(segment, &emphasis).empty()) {
      any = true;
      break;
    }
  if (any) {
    for (U32& segment : queue) {
      Masked masked = mask_control_spans(segment, emphasis, restore.size());
      segment = std::move(masked.text);
      for (auto& entry : masked.restore) restore.push_back(std::move(entry));
    }
  }
  std::vector<U32> merged;
  std::size_t head = 0;
  while (head < queue.size()) {
    U32 pending = queue[head++];
    while (head < queue.size() && pending.size() < cap &&
           abbreviations.ends_on_abbreviation(pending)) {
      std::vector<U32> sentences = sentences_of(queue[head]);
      if (sentences.size() <= 1) {
        pending = py_strip(pending + U" " + queue[head]);
        ++head;
        continue;
      }
      pending = py_strip(pending + U" " + sentences[0]);
      sentences.erase(sentences.begin());
      queue[head] = join(sentences, U" ");
    }
    merged.push_back(std::move(pending));
  }
  if (!restore.empty())
    for (U32& segment : merged) segment = unmask_control_spans(std::move(segment), restore);
  return merged;
}

// ---- The live request plan ------------------------------------------------
RequestText decide_request(std::u32string_view request_text, std::optional<bool> preset) {
  RequestText out;
  out.emphasis = emphasis_spans(request_text);
  // decide_request_emotion: a decision already on the control wins.
  out.emotion = preset.has_value() ? *preset : text_has_tag(request_text, nullptr);
  out.leading_tag = leading_expression_tag(request_text);
  out.demo_chunks = true;          // Q_PLAIN_DEMO_CHUNKS=1 / L_TAGGED_CHUNKING=demo
  out.dc_mode = out.emotion ? 0 : 1;
  return out;
}

std::vector<U32> plan_segments(std::u32string_view prepared_text, bool wfst,
                               const Emphasis& emphasis, const Abbreviations& abbreviations,
                               const Settings& settings) {
  std::vector<U32> segments = demo_split_text(prepared_text, emphasis, 0, settings);
  if (wfst)
    segments = merge_abbreviation_boundaries(segments, abbreviations,
                                             settings.demo_chunk_characters, emphasis,
                                             settings.normalize_merge_limit);
  return emo_segments(segments, emphasis);
}

std::vector<U32> resplit_segment(std::u32string_view normalized, const Emphasis& emphasis,
                                 const Settings& settings) {
  return demo_split_text(normalized, emphasis, 0, settings);
}

// ---- Budgets ---------------------------------------------------------------
int tagged_segment_budget(std::size_t chars, const Settings& s) {
  const double per_char = s.tagged_budget_per_char;
  int cap;
  if (s.tagged_budget_max_seconds > 0) {
    cap = int(s.tagged_budget_max_seconds * 25);
  } else {
    cap = std::max(int(s.max_seconds_per_segment * 25),
                   int(per_char * int(s.demo_chunk_characters * 1.4)) + 1);
  }
  return std::max(200, std::min(cap, int(double(chars) * per_char)));
}

std::vector<int> demo_budgets(const std::vector<std::size_t>& batch_chars, int total_chunks,
                              int start_index, const Settings& s) {
  const bool single = total_chunks <= 1 && batch_chars.size() == 1;
  std::vector<int> out;
  for (std::size_t chars : batch_chars) {
    if (single && start_index == 0) {
      out.push_back(s.demo_max_new_tokens);
    } else {
      const int scaled = int(double(chars) / s.demo_chars_per_sec * 25 * 1.8);
      out.push_back(std::max(200, std::min(s.demo_max_new_tokens, scaled)));
    }
  }
  return out;
}

}  // namespace kt2n::textplan
