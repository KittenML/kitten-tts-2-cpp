// Handwritten Qwen2 byte-level BPE tokenizer (HF tokenizers 0.22 semantics)
// and the LM prompt builder.
//
// tokenizer.json of the live checkpoint: normalizer NFC; pre-tokenizer
// Sequence[Split(Qwen2 regex, Isolated), ByteLevel(no prefix space, no
// regex)]; model BPE (no dropout, no unk, ignore_merges false); 6594 added
// tokens (special ones, [START]/[STOP] and <tool_call>-style ones matched on
// the raw text, the audio tokens <|a_N|> and <final_seg> on the NFC text;
// leftmost-longest, as HF's Aho-Corasick split), all without
// lstrip / rstrip / single_word.  encode() = tokenizer.encode(text,
// add_special_tokens=False) (no post-processor ids are added).
#include <algorithm>
#include <array>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "json.h"
#include "textplan_internal.h"

namespace kt2n::textplan {
namespace {

struct Range {
  char32_t lo, hi;
};
struct Ccc {
  char32_t lo, hi;
  int ccc;
};
struct TokDecomp {
  char32_t cp;
  char32_t d[4];
};
struct Compose {
  char32_t a, b, c;
};
// \p{L}, \p{N} and \s as HF tokenizers' own regex engine (onig) sees them:
// measured from its pre-tokenizer by tools/l9b/gen_tok_classes.py (onig carries
// newer Unicode tables than CPython 3.11).
#include "tok_classes.inc"

template <std::size_t N>
bool in_table(const Range (&table)[N], char32_t c) {
  std::size_t lo = 0, hi = N;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (table[mid].hi < c) {
      lo = mid + 1;
    } else if (table[mid].lo > c) {
      hi = mid;
    } else {
      return true;
    }
  }
  return false;
}
bool onig_space(char32_t c) {
  if (c < 0x80) return c == ' ' || (c >= 0x09 && c <= 0x0D);
  return in_table(kTokSpace, c);
}
bool letter(char32_t c) {
  if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  return in_table(kTokLetter, c);
}
bool number(char32_t c) {
  if (c < 0x80) return c >= '0' && c <= '9';
  return in_table(kTokNumber, c);
}
// ---- NFC with HF's normalizer data (an older Unicode than CPython's) -------
int tok_ccc(char32_t c) {
  if (c < 0x300) return 0;
  std::size_t lo = 0, hi = std::size(kTokCcc);
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (kTokCcc[mid].hi < c) {
      lo = mid + 1;
    } else if (kTokCcc[mid].lo > c) {
      hi = mid;
    } else {
      return kTokCcc[mid].ccc;
    }
  }
  return 0;
}

struct NfcTables {
  std::unordered_map<char32_t, std::array<char32_t, 4>> decomp;
  std::unordered_map<uint64_t, char32_t> compose;
  NfcTables() {
    decomp.reserve(std::size(kTokDecomp));
    for (const auto& d : kTokDecomp) decomp.emplace(d.cp, std::array<char32_t, 4>{d.d[0], d.d[1], d.d[2], d.d[3]});
    compose.reserve(std::size(kTokCompose));
    for (const auto& c : kTokCompose) compose.emplace((uint64_t(c.a) << 21) | c.b, c.c);
  }
};
const NfcTables& nfc_tables() {
  static const NfcTables tables;
  return tables;
}

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr int kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount,
              kSCount = kLCount * kNCount;

U32 nfc(std::u32string_view s) {
  bool simple = true;
  for (char32_t c : s)
    if (c >= 0x300) {
      simple = false;
      break;
    }
  if (simple) return U32(s);  // nothing below U+0300 decomposes, combines or composes
  const auto& tables = nfc_tables();
  U32 d;
  d.reserve(s.size() + 8);
  for (char32_t c : s) {
    if (c >= kSBase && c < kSBase + char32_t(kSCount)) {
      const int x = int(c - kSBase);
      d.push_back(kLBase + char32_t(x / kNCount));
      d.push_back(kVBase + char32_t((x % kNCount) / kTCount));
      if (x % kTCount) d.push_back(kTBase + char32_t(x % kTCount));
      continue;
    }
    auto it = tables.decomp.find(c);
    if (it == tables.decomp.end()) {
      d.push_back(c);
    } else {
      for (char32_t x : it->second)
        if (x) d.push_back(x);
    }
  }
  // Canonical ordering (stable insertion sort of each run of non-starters).
  for (std::size_t i = 1; i < d.size(); ++i) {
    const int cc = tok_ccc(d[i]);
    if (cc == 0) continue;
    std::size_t j = i;
    while (j > 0) {
      const int prev = tok_ccc(d[j - 1]);
      if (prev == 0 || prev <= cc) break;
      std::swap(d[j - 1], d[j]);
      --j;
    }
  }
  // Canonical composition.
  U32 out;
  out.reserve(d.size());
  std::size_t starter = U32::npos;
  int last_cc = -1;  // ccc of the last character kept after the starter (-1: none)
  for (char32_t c : d) {
    const int cc = tok_ccc(c);
    if (starter != U32::npos) {
      const char32_t st = out[starter];
      const bool adjacent = last_cc == -1;
      const bool blocked = !adjacent && (last_cc == 0 || last_cc >= cc);
      if (adjacent && st >= kLBase && st < kLBase + char32_t(kLCount) && c >= kVBase &&
          c < kVBase + char32_t(kVCount)) {
        out[starter] = kSBase + ((st - kLBase) * char32_t(kVCount) + (c - kVBase)) * char32_t(kTCount);
        continue;
      }
      if (adjacent && st >= kSBase && st < kSBase + char32_t(kSCount) && (st - kSBase) % kTCount == 0 &&
          c > kTBase && c < kTBase + char32_t(kTCount)) {
        out[starter] = st + (c - kTBase);
        continue;
      }
      if (!blocked) {
        auto it = tables.compose.find((uint64_t(st) << 21) | c);
        if (it != tables.compose.end()) {
          out[starter] = it->second;
          continue;
        }
      }
    }
    if (cc == 0) {
      starter = out.size();
      last_cc = -1;
    } else {
      last_cc = cc;
    }
    out.push_back(c);
  }
  return out;
}

bool crlf(char32_t c) { return c == U'\r' || c == U'\n'; }
bool other(char32_t c) { return !onig_space(c) && !letter(c) && !number(c); }
bool icase(char32_t c, char lower) {
  if (c == char32_t(lower) || c == char32_t(lower - 32)) return true;
  return lower == 's' && c == 0x17F;  // U+017F LATIN SMALL LETTER LONG S folds to "s"
}

// One match of the Qwen2 pre-tokenizer regex at ``i`` (always non-empty):
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
// | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::size_t pretoken_end(std::u32string_view s, std::size_t i) {
  const std::size_t n = s.size();
  const char32_t c = s[i];
  // 1. contractions
  if (c == U'\'' && i + 1 < n) {
    const char32_t a = s[i + 1];
    const char32_t b = i + 2 < n ? s[i + 2] : 0;
    if (icase(a, 's') || icase(a, 't') || icase(a, 'm') || icase(a, 'd')) return i + 2;
    if ((icase(a, 'r') && icase(b, 'e')) || (icase(a, 'v') && icase(b, 'e')) ||
        (icase(a, 'l') && icase(b, 'l')))
      return i + 3;
  }
  // 2. [^\r\n\p{L}\p{N}]?\p{L}+
  {
    std::size_t j = i;
    if (!crlf(c) && !letter(c) && !number(c) && i + 1 < n && letter(s[i + 1])) j = i + 1;
    if (j < n && letter(s[j])) {
      while (j < n && letter(s[j])) ++j;
      return j;
    }
  }
  // 3. \p{N}
  if (number(c)) return i + 1;
  // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
  {
    std::size_t j = i;
    if (c == U' ' && i + 1 < n && other(s[i + 1])) j = i + 1;
    if (j < n && other(s[j])) {
      while (j < n && other(s[j])) ++j;
      while (j < n && crlf(s[j])) ++j;
      return j;
    }
  }
  // whitespace alternatives
  if (onig_space(c)) {
    std::size_t k = i;
    while (k < n && onig_space(s[k])) ++k;
    // 5. \s*[\r\n]+ : ends after the last \r / \n of the run
    for (std::size_t m = k; m > i; --m)
      if (crlf(s[m - 1])) return m;
    // 6. \s+(?!\S)
    if (k == n) return k;
    if (k - i >= 2) return k - 1;
    // 7. \s+
    return k;
  }
  return i + 1;  // unreachable: every code point is covered above
}

std::array<char32_t, 256> byte_to_unicode() {
  std::array<char32_t, 256> table{};
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  std::vector<bool> present(256, false);
  for (int b : bs) present[std::size_t(b)] = true, table[std::size_t(b)] = char32_t(b);
  int extra = 0;
  for (int b = 0; b < 256; ++b)
    if (!present[std::size_t(b)]) table[std::size_t(b)] = char32_t(256 + extra++);
  return table;
}

struct AddedToken {
  U32 content;
  int32_t id;
};

}  // namespace

struct Tokenizer::Impl {
  std::unordered_map<std::string, int32_t> vocab;  // raw bytes -> id
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges;  // (l, r) -> (rank, new id)
  std::array<int32_t, 256> byte_id{};
  // Added tokens by trie pass: [0] raw text (normalized=false), [1] NFC text.
  std::unordered_map<U32, int32_t> added[2];
  std::size_t added_max_len[2] = {0, 0};
  std::unordered_set<char32_t> first_chars[2], last_chars[2];
  std::size_t total = 0;

  void bpe(const std::string& bytes, std::vector<int32_t>& out) const;
  void pretokenize_bpe(std::u32string_view piece, std::vector<int32_t>& out) const;
  void split_added(std::u32string_view text, int pass, std::vector<int32_t>& out) const;
};

void Tokenizer::Impl::bpe(const std::string& bytes, std::vector<int32_t>& out) const {
  struct Symbol {
    int32_t c;
    int prev, next;
    int len;
  };
  struct Merge {
    int pos;
    int32_t rank, new_id;
  };
  // ignore_merges is false: the merges always run, even for a whole-word token.
  std::vector<Symbol> symbols;
  symbols.reserve(bytes.size());
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    symbols.push_back({byte_id[static_cast<unsigned char>(bytes[i])], int(i) - 1,
                       i + 1 < bytes.size() ? int(i + 1) : -1, 1});
  }
  auto cmp = [](const Merge& a, const Merge& b) {
    if (a.rank != b.rank) return a.rank > b.rank;  // min rank first
    return a.pos > b.pos;                          // then min position
  };
  std::priority_queue<Merge, std::vector<Merge>, decltype(cmp)> queue(cmp);
  auto lookup = [&](int32_t l, int32_t r) -> const std::pair<int32_t, int32_t>* {
    auto it = merges.find((uint64_t(uint32_t(l)) << 32) | uint32_t(r));
    return it == merges.end() ? nullptr : &it->second;
  };
  for (std::size_t i = 0; i + 1 < symbols.size(); ++i)
    if (auto m = lookup(symbols[i].c, symbols[i + 1].c)) queue.push({int(i), m->first, m->second});
  while (!queue.empty()) {
    const Merge top = queue.top();
    queue.pop();
    Symbol& cur = symbols[std::size_t(top.pos)];
    if (cur.len == 0 || cur.next == -1) continue;
    const int next_pos = cur.next;
    const Symbol right = symbols[std::size_t(next_pos)];
    const auto* target = lookup(cur.c, right.c);
    if (target == nullptr || target->second != top.new_id) continue;
    cur.c = top.new_id;
    cur.len += right.len;
    cur.next = right.next;
    symbols[std::size_t(next_pos)].len = 0;
    if (right.next > -1 && std::size_t(right.next) < symbols.size())
      symbols[std::size_t(right.next)].prev = top.pos;
    if (cur.prev >= 0) {
      if (auto m = lookup(symbols[std::size_t(cur.prev)].c, cur.c)) queue.push({cur.prev, m->first, m->second});
    }
    if (cur.next >= 0 && std::size_t(cur.next) < symbols.size()) {
      if (auto m = lookup(cur.c, symbols[std::size_t(cur.next)].c)) queue.push({top.pos, m->first, m->second});
    }
  }
  for (const Symbol& s : symbols)
    if (s.len) out.push_back(s.c);
}

void Tokenizer::Impl::pretokenize_bpe(std::u32string_view piece, std::vector<int32_t>& out) const {
  std::size_t i = 0;
  std::string bytes;
  while (i < piece.size()) {
    const std::size_t e = pretoken_end(piece, i);
    bytes.clear();
    for (std::size_t k = i; k < e; ++k) append_utf8(bytes, piece[k]);
    bpe(bytes, out);
    i = e;
  }
}

void Tokenizer::Impl::split_added(std::u32string_view text, int pass, std::vector<int32_t>& out) const {
  const auto& table = added[pass];
  const std::size_t max_len = added_max_len[pass];
  std::size_t start = 0;  // start of the pending non-added piece
  std::size_t i = 0;
  auto flush = [&](std::size_t end) {
    if (end <= start) return;
    const std::u32string_view piece = text.substr(start, end - start);
    if (pass == 0) {
      const U32 normalized = nfc(piece);
      split_added(normalized, 1, out);
    } else {
      pretokenize_bpe(piece, out);
    }
  };
  while (i < text.size()) {
    // Leftmost-longest: the longest added token starting at i.
    std::size_t best = 0;
    int32_t best_id = -1;
    if (!table.empty() && first_chars[pass].count(text[i])) {
      const std::size_t limit = std::min(max_len, text.size() - i);
      for (std::size_t len = limit; len >= 1; --len) {
        if (!last_chars[pass].count(text[i + len - 1])) continue;
        auto it = table.find(U32(text.substr(i, len)));
        if (it != table.end()) {
          best = len;
          best_id = it->second;
          break;
        }
      }
    }
    if (best) {
      flush(i);
      out.push_back(best_id);
      i += best;
      start = i;
    } else {
      ++i;
    }
  }
  flush(text.size());
}

Tokenizer Tokenizer::from_json(std::string_view tokenizer_json) {
  const json::Value root = json::parse(tokenizer_json);
  auto impl = std::make_shared<Impl>();
  const auto table = byte_to_unicode();
  std::unordered_map<char32_t, unsigned char> inverse;
  for (int b = 0; b < 256; ++b) inverse[table[std::size_t(b)]] = static_cast<unsigned char>(b);
  auto to_bytes = [&](const std::string& mapped) {
    std::string bytes;
    for (char32_t c : from_utf8(mapped)) {
      auto it = inverse.find(c);
      if (it == inverse.end()) throw std::runtime_error("tokenizer.json: vocab key outside the byte map");
      bytes.push_back(char(it->second));
    }
    return bytes;
  };
  const json::Value* model = root.get("model");
  if (!model || !model->get("type") || model->get("type")->as_string() != "BPE")
    throw std::runtime_error("tokenizer.json: expected a BPE model");
  const json::Value* vocab = model->get("vocab");
  const json::Value* merges = model->get("merges");
  if (!vocab || !vocab->is_object() || !merges || !merges->is_array())
    throw std::runtime_error("tokenizer.json: vocab / merges missing");
  std::unordered_map<std::string, int32_t> by_mapped;
  by_mapped.reserve(vocab->members().size());
  impl->vocab.reserve(vocab->members().size());
  int32_t max_id = -1;
  for (const auto& [key, value] : vocab->members()) {
    const int32_t id = int32_t(value.as_int());
    by_mapped.emplace(key, id);
    impl->vocab.emplace(to_bytes(key), id);
    max_id = std::max(max_id, id);
  }
  for (int b = 0; b < 256; ++b) {
    std::string key;
    append_utf8(key, table[std::size_t(b)]);
    auto it = by_mapped.find(key);
    if (it == by_mapped.end()) throw std::runtime_error("tokenizer.json: a byte has no token");
    impl->byte_id[std::size_t(b)] = it->second;
  }
  impl->merges.reserve(merges->items().size());
  int32_t rank = 0;
  for (const json::Value& merge : merges->items()) {
    std::string left, right;
    if (merge.is_array() && merge.items().size() == 2) {
      left = merge.items()[0].as_string();
      right = merge.items()[1].as_string();
    } else if (merge.is_string()) {
      const std::string& s = merge.as_string();
      const std::size_t space = s.find(' ');
      left = s.substr(0, space);
      right = s.substr(space + 1);
    } else {
      throw std::runtime_error("tokenizer.json: bad merge");
    }
    auto l = by_mapped.find(left), r = by_mapped.find(right), m = by_mapped.find(left + right);
    if (l == by_mapped.end() || r == by_mapped.end() || m == by_mapped.end())
      throw std::runtime_error("tokenizer.json: merge refers to a missing token");
    impl->merges.emplace((uint64_t(uint32_t(l->second)) << 32) | uint32_t(r->second),
                         std::make_pair(rank, m->second));
    ++rank;
  }
  const json::Value* added = root.get("added_tokens");
  if (added && added->is_array()) {
    for (const json::Value& token : added->items()) {
      const int32_t id = int32_t(token.get("id")->as_int());
      const U32 content = from_utf8(token.get("content")->as_string());
      const bool normalized = token.get("normalized") && token.get("normalized")->as_bool();
      if ((token.get("lstrip") && token.get("lstrip")->as_bool()) ||
          (token.get("rstrip") && token.get("rstrip")->as_bool()) ||
          (token.get("single_word") && token.get("single_word")->as_bool()))
        throw std::runtime_error("tokenizer.json: lstrip/rstrip/single_word added tokens are not supported");
      if (content.empty()) throw std::runtime_error("tokenizer.json: empty added token");
      const int pass = normalized ? 1 : 0;
      impl->first_chars[pass].insert(content.front());
      impl->last_chars[pass].insert(content.back());
      impl->added[pass].emplace(content, id);
      impl->added_max_len[pass] = std::max(impl->added_max_len[pass], content.size());
      max_id = std::max(max_id, id);
    }
  }
  impl->total = std::size_t(max_id + 1);
  Tokenizer out;
  out.impl_ = std::move(impl);
  return out;
}

Tokenizer Tokenizer::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read tokenizer: " + path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return from_json(buffer.str());
}

std::size_t Tokenizer::size() const { return impl_ ? impl_->total : 0; }

std::vector<int32_t> Tokenizer::encode(std::u32string_view text) const {
  std::vector<int32_t> out;
  out.reserve(text.size() / 3 + 4);
  impl_->split_added(text, 0, out);
  return out;
}

std::vector<int32_t> Tokenizer::encode(std::string_view utf8) const { return encode(from_utf8(utf8)); }

// ---- Prompt ------------------------------------------------------------------
std::vector<int32_t> reference_prompt(const std::vector<int32_t>& transcript_ids,
                                      const std::vector<int32_t>& s3_tokens) {
  std::vector<int32_t> out;
  out.reserve(transcript_ids.size() + s3_tokens.size() + 5);
  out.push_back(kStart);
  out.push_back(kRefTextStart);
  out.insert(out.end(), transcript_ids.begin(), transcript_ids.end());
  out.push_back(kRefTextEnd);
  out.push_back(kRefSpeechStart);
  for (int32_t token : s3_tokens) out.push_back(kAudioIdBase + token);
  out.push_back(kRefSpeechEnd);
  return out;
}

std::vector<int32_t> target_suffix(const Tokenizer& tokenizer, std::u32string_view text, bool emotion) {
  std::vector<int32_t> out;
  if (emotion) {
    const auto ids = tokenizer.encode(kEmotionControl);
    out.insert(out.end(), ids.begin(), ids.end());
  }
  out.push_back(kTextStart);
  const auto ids = tokenizer.encode(text);
  out.insert(out.end(), ids.begin(), ids.end());
  out.push_back(kFinalSeg);
  out.push_back(kSpeechStart);
  return out;
}

std::vector<int32_t> target_prompt(const Tokenizer& tokenizer, std::u32string_view text,
                                   const std::vector<int32_t>& reference_prefix, bool use_icl,
                                   bool emotion) {
  std::vector<int32_t> out;
  if (use_icl) {
    out = reference_prefix;
  } else {
    out.push_back(kStart);
  }
  const auto suffix = target_suffix(tokenizer, text, emotion);
  out.insert(out.end(), suffix.begin(), suffix.end());
  return out;
}

}  // namespace kt2n::textplan
