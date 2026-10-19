#include "nemo.h"

#include <fst/fstlib.h>
#include <fst/label-reachable.h>
#include <fst/lookahead-filter.h>
#include <fst/lookahead-matcher.h>
#include <fst/matcher-fst.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <stdexcept>

#include "fast_tagger.h"
#include "pyre.h"
#include "unicode.h"

namespace kt2n::text {
namespace {

using fst::StdArc;
using fst::StdVectorFst;

// ---- production's lookahead tagger (kittenml-api ebcdd3f native/kt_la_tagger.cc) ----
// Input-side label lookahead through epsilons; NO weight/prefix pushing, so the
// connected lattice has the same paths and weights as pynini's plain compose.
constexpr uint32_t kKtFlags = fst::kInputLookAheadMatcher | fst::kLookAheadEpsilons |
                              fst::kLookAheadNonEpsilons | fst::kLookAheadKeepRelabelData;
inline constexpr char kt_la_fst_type[] = "kt_ilabel_lookahead";
using KtMatcher = fst::LabelLookAheadMatcher<fst::SortedMatcher<fst::ConstFst<StdArc>>, kKtFlags,
                                             fst::FastLogAccumulator<StdArc>>;
using KtLaFst = fst::MatcherFst<fst::ConstFst<StdArc>, KtMatcher, kt_la_fst_type,
                                fst::LabelLookAheadRelabeler<StdArc>>;
using KtRelabeler = fst::LabelLookAheadRelabeler<StdArc>;

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// pynini.accep(string) in byte mode: a linear chain 0 -> 1 -> ... -> n.
StdVectorFst string_acceptor(std::string_view bytes) {
  StdVectorFst f;
  f.ReserveStates(static_cast<int>(bytes.size()) + 1);
  auto s = f.AddState();
  f.SetStart(s);
  for (char ch : bytes) {
    const auto label = static_cast<StdArc::Label>(static_cast<unsigned char>(ch));
    const auto next = f.AddState();
    f.AddArc(s, StdArc(label, label, StdArc::Weight::One(), next));
    s = next;
  }
  f.SetFinal(s, StdArc::Weight::One());
  return f;
}

// Fst.string() of a single-path FST: its output labels as bytes, epsilons skipped.
// false = not a string FST (pynini raises).
bool path_string(const StdVectorFst& path, std::string* out) {
  out->clear();
  auto current = path.Start();
  if (current == fst::kNoStateId) return false;
  for (size_t guard = 0; guard <= static_cast<size_t>(path.NumStates()); ++guard) {
    const size_t n = path.NumArcs(current);
    if (n == 0) break;
    if (n != 1) return false;
    fst::ArcIterator<StdVectorFst> arcs(path, current);
    const auto& arc = arcs.Value();
    if (arc.olabel != 0) {
      if (arc.olabel < 0 || arc.olabel > 255) return false;
      out->push_back(static_cast<char>(arc.olabel));
    }
    current = arc.nextstate;
  }
  if (path.Final(current) == StdArc::Weight::Zero()) return false;
  return true;
}

// ---- TokenParser (nemo_text_processing/text_normalization/token_parser.py) ----
struct TVal;
using TDict = std::vector<std::pair<std::string, TVal>>;  // OrderedDict
struct TVal {
  enum Kind { kStr, kDict, kBool, kNone } kind = kNone;
  std::string s;
  std::shared_ptr<TDict> d;
};

void dict_set(TDict& d, const std::string& key, TVal v) {
  for (auto& kv : d) {
    if (kv.first == key) {
      kv.second = std::move(v);  // OrderedDict keeps the first position
      return;
    }
  }
  d.emplace_back(key, std::move(v));
}

struct ParseError : std::runtime_error {
  explicit ParseError(const char* cls) : std::runtime_error(cls) {}
};

class TokenParser {
 public:
  explicit TokenParser(std::string_view text) : t_(text) {
    if (t_.empty()) throw ParseError("IndexError");  // text[0]
    index_ = 0;
    eos_ = false;
  }

  std::vector<TDict> parse() {
    std::vector<TDict> l;
    while (parse_ws()) {
      TDict token;
      if (!parse_token(&token)) break;
      l.push_back(std::move(token));
    }
    return l;
  }

 private:
  char ch() const { return t_[index_]; }
  bool is(char c) const { return !eos_ && ch() == c; }

  bool read() {
    if (index_ < t_.size() - 1) {
      ++index_;
      return true;
    }
    eos_ = true;
    return false;
  }

  bool parse_ws() {
    bool not_eos = !eos_;
    while (not_eos && ch() == ' ') not_eos = read();
    return not_eos;
  }

  void parse_char(char exp) {
    if (!is(exp)) throw ParseError("AssertionError");
    read();
  }

  // None when no key characters.
  bool parse_string_key(std::string* key) {
    // assert self.char not in string.whitespace and self.char != EOS
    if (eos_) throw ParseError("AssertionError");
    const char c0 = ch();
    if (c0 == ' ' || c0 == '\t' || c0 == '\n' || c0 == '\r' || c0 == '\x0b' || c0 == '\x0c')
      throw ParseError("AssertionError");
    key->clear();
    auto incl = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
    while (!eos_ && incl(ch())) {
      key->push_back(ch());
      if (!read()) throw ParseError("ValueError");
    }
    return !key->empty();
  }

  TVal parse_string_value() {
    if (eos_) throw ParseError("AssertionError");
    std::string l;
    while (true) {
      if (eos_) {
        // self.char == "<EOS>" != '"' -> appends "<EOS>"; read() fails -> ValueError
        throw ParseError("ValueError");
      }
      if (ch() == '"') {
        if (index_ + 1 >= t_.size()) throw ParseError("IndexError");
        if (t_[index_ + 1] == ' ') break;
      }
      l.push_back(ch());
      if (!read()) throw ParseError("ValueError");
    }
    TVal v;
    if (l.empty()) {
      v.kind = TVal::kNone;
    } else {
      v.kind = TVal::kStr;
      v.s = std::move(l);
    }
    return v;
  }

  TVal parse_token_value() {
    if (is(':')) {
      parse_char(':');
      parse_ws();
      parse_char('"');
      TVal v = parse_string_value();
      parse_char('"');
      return v;
    }
    if (is('{')) {
      auto d = std::make_shared<TDict>();
      parse_char('{');
      std::vector<TDict> list = parse();
      for (auto& tok : list) {
        for (auto& kv : tok) dict_set(*d, kv.first, kv.second);
      }
      parse_char('}');
      TVal v;
      v.kind = TVal::kDict;
      v.d = std::move(d);
      return v;
    }
    throw ParseError("ValueError");
  }

  bool parse_token(TDict* out) {
    std::string key;
    if (!parse_string_key(&key)) return false;
    parse_ws();
    TVal value;
    if (key == "preserve_order") {
      parse_char(':');
      parse_ws();
      for (char c : std::string_view("true")) parse_char(c);
      value.kind = TVal::kBool;
    } else {
      value = parse_token_value();
    }
    dict_set(*out, key, std::move(value));
    return true;
  }

  std::string_view t_;
  size_t index_ = 0;
  bool eos_ = false;
};

// ---- permutations (Normalizer._permute / generate_permutations) ----
double factorial(size_t n) {
  double f = 1;
  for (size_t i = 2; i <= n; ++i) f *= static_cast<double>(i);
  return f;
}

double estimate_perms(const TDict& group) {
  double n = 1;
  for (const auto& kv : group) {
    if (kv.second.kind == TVal::kDict) n *= estimate_perms(*kv.second.d);
  }
  return n * factorial(group.size());
}

struct PermuteError : std::runtime_error {
  PermuteError() : std::runtime_error("ValueError") {}
};

std::vector<std::string> permute(const TDict& d) {
  std::vector<std::string> l;
  bool preserve = false;
  for (const auto& kv : d) preserve |= kv.first == "preserve_order";
  std::vector<size_t> order(d.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  do {
    std::vector<std::string> subl{""};
    for (size_t idx : order) {
      const auto& [k, v] = d[idx];
      if (v.kind == TVal::kStr) {
        for (auto& x : subl) x += k + ": \"" + v.s + "\" ";
      } else if (v.kind == TVal::kDict) {
        std::vector<std::string> rec = permute(*v.d);
        std::vector<std::string> next;
        next.reserve(subl.size() * rec.size());
        for (const auto& x : subl) {
          for (const auto& r : rec) next.push_back(x + " " + k + " { " + r + " } ");
        }
        subl = std::move(next);
      } else if (v.kind == TVal::kBool) {
        for (auto& x : subl) x += k + ": true ";
      } else {
        throw PermuteError();
      }
    }
    l.insert(l.end(), subl.begin(), subl.end());
    if (preserve) break;
  } while (std::next_permutation(order.begin(), order.end()));
  return l;
}

}  // namespace

std::string pynini_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '[' || c == ']' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

// ---- sacremoses MosesDetokenizer, lang="en", unescape=False ----
std::string moses_detokenize_en(std::string_view input) {
  static const Regex kHyphen(" @-@ ");
  std::string text = " " + std::string(input) + " ";
  text = kHyphen.sub(text, "-");
  std::map<std::string, long> quote_counts{{"'", 0}, {"\"", 0}, {"``", 0}, {"`", 0}, {"''", 0}};
  std::string prepend_space = " ";
  std::string out;
  const std::vector<std::string> tokens = py_split_ws(text);
  auto all_of = [](const std::u32string& u, auto pred) {
    if (u.empty()) return false;
    for (char32_t c : u) {
      if (!pred(c)) return false;
    }
    return true;
  };
  for (size_t i = 0; i < tokens.size(); ++i) {
    const std::string& token = tokens[i];
    const std::u32string u = to_u32(token);
    const bool is_currency = all_of(u, [](char32_t c) {
      return moses_is_sc(c) || c == '(' || c == '[' || c == '{' || c == U'¿' || c == U'¡';
    });
    const bool is_punct = all_of(u, [](char32_t c) {
      return c == ',' || c == '.' || c == '?' || c == '!' || c == ':' || c == ';' || c == '\\' || c == '%' ||
             c == '}' || c == ']' || c == ')';
    });
    const bool is_open_quote = all_of(u, [](char32_t c) {
      return c == '\'' || c == '"' || c == U'„' || c == U'“' || c == '`';
    });
    if (moses_is_cjk(u[0])) {
      if (i > 0) {
        const std::u32string prev = to_u32(tokens[i - 1]);
        if (moses_is_cjk(prev.back())) {
          out += token;
          prepend_space = " ";
          continue;
        }
      }
      out += prepend_space + token;
      prepend_space = " ";
    } else if (is_currency) {
      out += prepend_space + token;
      prepend_space = "";
    } else if (is_punct) {
      out += token;
      prepend_space = " ";
    } else if (i > 0 && u.size() >= 2 && u[0] == '\'' && moses_is_alpha(u[1])) {
      out += token;
      prepend_space = " ";
    } else if (is_open_quote) {
      std::string normalized_quo = token;
      if (all_of(u, [](char32_t c) { return c == U'„' || c == U'“' || c == U'”'; }))
        normalized_quo = "\"";
      long& count = quote_counts[normalized_quo];  // .get(x, 0)
      if (count % 2 == 0) {
        if (token == "'" && i > 0 && !tokens[i - 1].empty() && tokens[i - 1].back() == 's') {
          out += token;
          prepend_space = " ";
        } else {
          out += prepend_space + token;
          prepend_space = "";
          count += 1;
        }
      } else {
        out += token;
        prepend_space = " ";
        count += 1;
      }
    } else {
      out += prepend_space + token;
      prepend_space = " ";
    }
  }
  static const Regex kOneSpace(" {2,}");
  out = kOneSpace.sub(out, " ");
  return py_strip(out);
}

// ---- data_loader_utils.post_process_punct(input, normalized_text) ----
std::string post_process_punct(std::string_view input_text, std::string_view normalized) {
  std::string input_s(input_text);
  if (input_s.find("``") != std::string::npos && normalized.find("``") == std::string_view::npos) {
    input_s = py_replace(input_s, "``", "\"");
  }
  auto explode = [](std::string_view s) {
    std::vector<std::string> v;
    for (size_t i = 0; i < s.size();) {
      const size_t a = i;
      decode(s, i);
      v.emplace_back(s.substr(a, i - a));
    }
    return v;
  };
  const std::vector<std::string> input = explode(input_s);
  std::vector<std::string> nt = explode(normalized);
  static const std::string kPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  for (char pc : kPunctuation) {
    const std::string punct(1, pc);
    if (std::find(input.begin(), input.end(), punct) == input.end()) continue;
    // try: ... except: skip the rest of this mark
    const auto count_in = std::count(input.begin(), input.end(), punct);
    const auto count_out = std::count(nt.begin(), nt.end(), punct);
    const bool equal = count_in == count_out;
    size_t idx_in = 0, idx_out = 0;
    while (std::find(input.begin() + std::min(idx_in, input.size()), input.end(), punct) != input.end()) {
      auto it_out = std::find(nt.begin() + std::min(idx_out, nt.size()), nt.end(), punct);
      if (it_out == nt.end()) break;  // ValueError -> except
      idx_out = static_cast<size_t>(it_out - nt.begin());
      idx_in = static_cast<size_t>(std::find(input.begin() + idx_in, input.end(), punct) - input.begin());
      auto is_valid = [&]() {
        return (idx_out > 0 && idx_in > 0 && nt[idx_out - 1] == input[idx_in - 1]) ||
               (idx_out + 1 < nt.size() && idx_in + 1 < input.size() && nt[idx_out + 1] == input[idx_in + 1]);
      };
      if (!equal && !is_valid()) {
        idx_in += 1;
        continue;
      }
      if (idx_in > 0 && idx_out > 0) {
        if (nt[idx_out - 1] == " " && input[idx_in - 1] != " ") {
          nt[idx_out - 1] = "";
        } else if (nt[idx_out - 1] != " " && input[idx_in - 1] == " ") {
          nt[idx_out - 1] += " ";
        }
      }
      if (idx_in + 1 < input.size() && idx_out + 1 < nt.size()) {
        if (nt[idx_out + 1] == " " && input[idx_in + 1] != " ") {
          nt[idx_out + 1] = "";
        } else if (nt[idx_out + 1] != " " && input[idx_in + 1] == " ") {
          nt[idx_out] = nt[idx_out] + " ";
        }
      }
      idx_out += 1;
      idx_in += 1;
    }
  }
  std::string joined;
  for (const auto& s : nt) joined += s;
  static const Regex kSpaces(" +");
  return kSpaces.sub(joined, " ");
}

// ---- NemoGrammar ----
struct NemoGrammar::Impl {
  std::unique_ptr<KtLaFst> tagger;
  // The tagger graph exactly as read (not relabelled, not re-sorted): pynini's
  // own `text @ tagger.fst`, production's fallback when the lookahead tagger
  // gives no answer. Read-only, shared by every copy.
  std::shared_ptr<const fst::ConstFst<StdArc>> plain_tagger;
  std::unique_ptr<FastTagger> fast;  // exact or declines (ties) -> OpenFst
  uint64_t fast_ok = 0, fast_declined = 0, plain_fallbacks = 0;
  std::unique_ptr<StdVectorFst> verbalizer;
  std::unique_ptr<StdVectorFst> post;

  // Production's worker (kittenml-api ebcdd3f normalization.py,
  // install_lookahead_tagger -> find_tags): the native lookahead tagger first;
  // when it returns non-zero (no path / error) or bytes that are not UTF-8,
  // `return original_find(escaped)`, i.e. pynini's plain composition, whose
  // answer (or exception) then stands. Example: a NUL byte is label 0 =
  // epsilon in the string acceptor; the lookahead composition finds no path
  // for it, the plain composition skips it ("Pay $5\0 now." -> "Pay five
  // dollars now.").
  // 0 ok, 1 no path / not a string, 2 error
  int tag(std::string_view raw, std::string* out) {
    const int code = tag_lookahead(raw, out);
    if (code == 0 && valid_utf8(*out)) return 0;
    ++plain_fallbacks;
    return tag_plain(raw, out);
  }

  // NeMo's own find_tags + select_tag: pynini.compose(accep(escape(text)),
  // tagger.fst) (connect, AUTO filter, no arc-sorting), then
  // shortestpath(nshortest=1, unique=True).string().
  int tag_plain(std::string_view raw, std::string* out) {
    out->clear();
    StdVectorFst input = string_acceptor(raw);
    StdVectorFst lattice;
    fst::Compose(input, *plain_tagger, &lattice, fst::ComposeOptions(/*connect=*/true, fst::AUTO_FILTER));
    if (lattice.Properties(fst::kError, false)) return 2;
    if (lattice.Start() == fst::kNoStateId) return 1;
    StdVectorFst path;
    fst::ShortestPath(lattice, &path, /*nshortest=*/1, /*unique=*/true);
    if (path.Properties(fst::kError, false)) return 2;
    return path_string(path, out) ? 0 : 1;
  }

  int tag_lookahead(std::string_view raw, std::string* out) {
    if (fast) {
      if (fast->tag(raw, out) == FastTagger::kOk) {
        ++fast_ok;
        return 0;
      }
      ++fast_declined;
    }
    StdVectorFst input = string_acceptor(raw);
    KtRelabeler::Relabel(&input, *tagger, false);
    fst::ArcSort(&input, fst::OLabelCompare<StdArc>());
    StdVectorFst lattice;
    fst::ComposeOptions options(/*connect=*/true, fst::AUTO_FILTER);
    fst::Compose(input, *tagger, &lattice, options);
    if (lattice.Properties(fst::kError, false)) return 2;
    if (lattice.Start() == fst::kNoStateId) return 1;
    StdVectorFst path;
    fst::ShortestPath(lattice, &path, /*nshortest=*/1, /*unique=*/true);
    if (path.Properties(fst::kError, false)) return 2;
    return path_string(path, out) ? 0 : 1;
  }

  // lattice = escape(tagged) @ verbalizer; false when the lattice is empty.
  bool verbalize_lattice(std::string_view tagged, StdVectorFst* lattice) {
    StdVectorFst input = string_acceptor(tagged);
    fst::Compose(input, *verbalizer, lattice, fst::ComposeOptions(true, fst::AUTO_FILTER));
    if (lattice->Properties(fst::kError, false)) throw std::runtime_error("FstOpError");
    return lattice->NumStates() != 0;
  }

  // pynini.lib.rewrite.top_rewrite(escape(text), post_processor.fst)
  int top_rewrite(std::string_view text, std::string* out) {
    StdVectorFst input = string_acceptor(text);
    StdVectorFst lattice;
    fst::Compose(input, *post, &lattice, fst::ComposeOptions(true, fst::ALT_SEQUENCE_FILTER));
    if (lattice.Properties(fst::kError, false)) return 2;
    if (lattice.Start() == fst::kNoStateId) return 1;  // rewrite.Error
    fst::Project(&lattice, fst::ProjectType::OUTPUT);
    fst::RmEpsilon(&lattice);
    StdVectorFst path;
    fst::ShortestPath(lattice, &path);
    if (path.Properties(fst::kError, false)) return 2;
    return path_string(path, out) ? 0 : 2;
  }
};

NemoGrammar::NemoGrammar() : impl_(std::make_unique<Impl>()) {}

std::string NemoGrammar::tagger_stats() const {
  return std::string("{\"fast\":") + (impl_->fast ? "true" : "false") + ",\"fast_ok\":" +
         std::to_string(impl_->fast_ok) + ",\"fast_declined\":" + std::to_string(impl_->fast_declined) +
         ",\"fast_ties\":" + std::to_string(impl_->fast ? impl_->fast->stats().ties : 0) +
         ",\"fast_nodes\":" + std::to_string(impl_->fast ? impl_->fast->stats().nodes : 0) +
         ",\"plain_fallbacks\":" + std::to_string(impl_->plain_fallbacks) + "}";
}
NemoGrammar::~NemoGrammar() = default;

std::unique_ptr<NemoGrammar> NemoGrammar::load(const std::string& dir, std::string* error) {
  std::unique_ptr<NemoGrammar> g(new NemoGrammar());
  try {
    std::unique_ptr<StdVectorFst> plain(StdVectorFst::Read(dir + "/tagger.fst"));
    if (!plain) {
      *error = "tagger.fst unreadable";
      return nullptr;
    }
    // Before any sorting: the fallback composes against the graph as pynini
    // holds it (same states, same arc order -> same tie-breaking).
    g->impl_->plain_tagger = std::make_shared<const fst::ConstFst<StdArc>>(*plain);
    if (g->impl_->plain_tagger->Properties(fst::kError, false)) {
      *error = "tagger.fst conversion failed";
      return nullptr;
    }
    if (!plain->Properties(fst::kILabelSorted, true)) fst::ArcSort(plain.get(), fst::ILabelCompare<StdArc>());
    // Experiment, OFF by default: the flat-array Viterbi declines on ~96% of
    // the corpus (exact ties with different tags; OpenFst's queue order picks),
    // so it only adds work. KT2N_TEXT_FAST_TAGGER=1 turns it on for study.
    const char* fast_env = std::getenv("KT2N_TEXT_FAST_TAGGER");
    if (fast_env != nullptr && std::string(fast_env) == "1") {
      std::string why;
      g->impl_->fast = FastTagger::build(*plain, &why);  // nullptr -> OpenFst only
    }
    g->impl_->tagger = std::make_unique<KtLaFst>(*plain);
    if (g->impl_->tagger->Properties(fst::kError, false)) {
      *error = "tagger lookahead build failed";
      return nullptr;
    }
    g->impl_->verbalizer.reset(StdVectorFst::Read(dir + "/verbalizer.fst"));
    g->impl_->post.reset(StdVectorFst::Read(dir + "/post_processor.fst"));
    if (!g->impl_->verbalizer || !g->impl_->post) {
      *error = "verbalizer.fst / post_processor.fst unreadable";
      return nullptr;
    }
    // One plain composition now, single-threaded, so the shared fallback
    // graph's property bits (arc-sortedness the matchers test) are computed
    // before other copies read it.
    std::string warm;
    if (g->impl_->tag_plain("Warm up 1 2 3.", &warm) != 0) {
      *error = "tagger.fst gives no tagging for the warm-up text";
      return nullptr;
    }
  } catch (const std::exception& e) {
    *error = e.what();
    return nullptr;
  }
  return g;
}

std::unique_ptr<NemoGrammar> NemoGrammar::load_sharing(const NemoGrammar& base, const std::string& dir,
                                                       std::string* error) {
  std::unique_ptr<NemoGrammar> g(new NemoGrammar());
  try {
    // safe=true: a copy for another thread; the ConstFst impl and the
    // LabelReachable data are shared (shared_ptr), nothing is rebuilt.
    g->impl_->tagger = std::make_unique<KtLaFst>(*base.impl_->tagger, /*safe=*/true);
    if (g->impl_->tagger->Properties(fst::kError, false)) {
      *error = "tagger copy failed";
      return nullptr;
    }
    g->impl_->plain_tagger = base.impl_->plain_tagger;
    g->impl_->verbalizer.reset(StdVectorFst::Read(dir + "/verbalizer.fst"));
    g->impl_->post.reset(StdVectorFst::Read(dir + "/post_processor.fst"));
    if (!g->impl_->verbalizer || !g->impl_->post) {
      *error = "verbalizer.fst / post_processor.fst unreadable";
      return nullptr;
    }
  } catch (const std::exception& e) {
    *error = e.what();
    return nullptr;
  }
  return g;
}

NemoResult NemoGrammar::tag(std::string_view raw) {
  NemoResult r;
  const int code = impl_->tag(raw, &r.text);
  if (code == 0 && valid_utf8(r.text)) {
    r.ok = true;
  } else {
    r.text.clear();
    r.error = code == 2 ? "FstOpError" : (code == 1 ? "FstOpError" : "UnicodeDecodeError");
  }
  return r;
}

NemoResult NemoGrammar::normalize(std::string_view original) {
  NemoResult r;
  stats_ = Stats{};
  const std::string text = py_strip(original);
  if (text.empty()) {
    r.ok = true;
    return r;
  }
  const std::string escaped = pynini_escape(text);
  double t0 = now_ms();
  NemoResult tagged = tag(text);  // the native tagger reads the unescaped bytes
  stats_.tag_ms = now_ms() - t0;
  if (!tagged.ok) return tagged;
  std::vector<TDict> tokens;
  try {
    TokenParser parser(tagged.text);
    tokens = parser.parse();
  } catch (const ParseError& e) {
    r.error = e.what();
    return r;
  }
  // _split_tokens_to_reduce_number_of_permutations (outside the try: raises)
  constexpr double kMaxPerms = 729;
  std::vector<std::pair<size_t, size_t>> splits;
  size_t prev = 0;
  double current = 1;
  for (size_t i = 0; i < tokens.size(); ++i) {
    const double n = estimate_perms(tokens[i]);
    if (n * current > kMaxPerms) {
      splits.emplace_back(prev, i);
      prev = i;
      current = 1;
    }
    if (n > kMaxPerms) {
      r.error = "ValueError";
      return r;
    }
    current *= n;
  }
  splits.emplace_back(prev, tokens.size());

  t0 = now_ms();
  std::string output;
  for (const auto& [a, b] : splits) {
    try {
      // generate_permutations: options of each token, in order, product-wise.
      std::vector<std::vector<std::string>> options;
      std::vector<size_t> pick(b - a, 0);
      bool found = false;
      StdVectorFst lattice;
      // Lazily: Python builds token idx's options when the generator reaches it.
      std::function<bool(size_t, const std::string&)> helper = [&](size_t idx, const std::string& prefix) -> bool {
        if (idx == b) {
          lattice = StdVectorFst();
          if (impl_->verbalize_lattice(prefix, &lattice)) return true;  // break
          return false;
        }
        const std::vector<std::string> opts = permute(tokens[idx]);
        for (const auto& o : opts) {
          if (helper(idx + 1, prefix + o)) return true;
        }
        return false;
      };
      // pynini.escape is applied to each serialization before composing; the
      // acceptor is built from the unescaped bytes, which is what escape+compile gives.
      found = helper(a, "");
      (void)found;
      std::string best;
      StdVectorFst path;
      fst::ShortestPath(lattice, &path, 1, true);
      if (path.Properties(fst::kError, false) || !path_string(path, &best) || !valid_utf8(best)) {
        // select_verbalizer raised inside the try: normalize returns the escaped text
        r.ok = true;
        r.text = escaped;
        stats_.verbalize_ms = now_ms() - t0;
        return r;
      }
      output += " " + best;
    } catch (const std::exception&) {
      r.ok = true;
      r.text = escaped;
      stats_.verbalize_ms = now_ms() - t0;
      return r;
    }
  }
  stats_.verbalize_ms = now_ms() - t0;
  static const Regex kSpaceDup(" {2,}");
  output = kSpaceDup.sub(std::string_view(output).substr(1), " ");

  // post_process (en): strip, escape, top_rewrite -- outside the try: raises.
  t0 = now_ms();
  {
    const std::string stripped = py_strip(output);
    if (stripped.empty()) {
      output = stripped;
    } else {
      std::string rewritten;
      const int code = impl_->top_rewrite(stripped, &rewritten);
      if (code != 0 || !valid_utf8(rewritten)) {
        r.error = code == 1 ? "Error" : "FstOpError";
        return r;
      }
      output = std::move(rewritten);
    }
  }
  stats_.post_ms = now_ms() - t0;
  t0 = now_ms();
  output = moses_detokenize_en(output);
  output = post_process_punct(original, output);
  stats_.moses_ms = now_ms() - t0;
  r.ok = true;
  r.text = std::move(output);
  return r;
}

NemoPool::NemoPool(const std::string& dir, int instances, std::string* error) {
  // KT2N_TEXT_SHARE_TAGGER=0: every copy loads and builds its own tagger
  // (sprint-1 behaviour, ~4x the tagger memory); default: copies 1.. share
  // copy 0's tagger graph, after one warm-up call on copy 0 so any lazily
  // computed lookahead data is complete before other threads read it.
  const char* share_env = std::getenv("KT2N_TEXT_SHARE_TAGGER");
  const bool share = share_env == nullptr || std::string(share_env) != "0";
  for (int i = 0; i < instances; ++i) {
    if (share && i == 1) grammars_[0]->normalize("Warm up 1 2 3.");
    auto g = share && i > 0 ? NemoGrammar::load_sharing(*grammars_[0], dir, error) : NemoGrammar::load(dir, error);
    if (!g) {
      grammars_.clear();
      idle_.clear();
      return;
    }
    idle_.push_back(g.get());
    grammars_.push_back(std::move(g));
  }
}

int NemoPool::idle_count() {
  std::lock_guard<std::mutex> lock(mu_);
  return std::max(0, static_cast<int>(idle_.size()) - static_cast<int>(waiting_.size()));
}

NemoResult NemoPool::normalize(std::string_view text, int priority) {
  NemoGrammar* g = nullptr;
  {
    std::unique_lock<std::mutex> lock(mu_);
    const std::pair<int, uint64_t> ticket{priority, next_ticket_++};
    waiting_.push_back(ticket);
    cv_.wait(lock, [&] {
      return !idle_.empty() && *std::min_element(waiting_.begin(), waiting_.end()) == ticket;
    });
    waiting_.erase(std::find(waiting_.begin(), waiting_.end(), ticket));
    g = idle_.back();
    idle_.pop_back();
  }
  cv_.notify_all();
  NemoResult r;
  try {
    r = g->normalize(text);
  } catch (const std::exception&) {
    r = NemoResult{false, "", "RuntimeError"};
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    idle_.push_back(g);
  }
  cv_.notify_all();
  return r;
}

}  // namespace kt2n::text
