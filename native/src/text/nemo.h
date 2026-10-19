// L9a text: NeMo text_processing 1.2.0 English normalisation, native.
//
// Ports `Normalizer(input_case="cased", lang="en").normalize(text,
// verbose=False, punct_post_process=True)` -- the exact call production's pool
// workers make -- on the grammar graphs exported from the reference install
// (tools/l9a/nemo_oracle.py: tagger.fst, verbalizer.fst, post_processor.fst):
//
//   strip -> tagger composition (OpenFst 1.8.3 + the input-label lookahead
//   matcher of production's kt_la_tagger.cc, N_TAGGER=lookahead) -> shortest
//   path -> TokenParser -> permutation split (729) -> verbalizer composition
//   per permutation until one is non-empty -> shortest path -> SPACE_DUP ->
//   post-processing rewrite (alt_sequence compose, project, rmepsilon,
//   shortest path) -> Moses detokenizer (sacremoses 0.2.0, en) ->
//   post_process_punct.
//
// One NemoGrammar is NOT thread-safe (OpenFst caches properties on shared
// graphs); NemoPool hands one instance to each caller.
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace kt2n::text {

// What Python would do: return a string, or raise (the class name is kept so
// the glue can report it like production's status field).
struct NemoResult {
  bool ok = false;
  std::string text;
  std::string error;  // Python exception class name when !ok
};

class NemoGrammar {
 public:
  // `dir` holds tagger.fst, verbalizer.fst, post_processor.fst.
  static std::unique_ptr<NemoGrammar> load(const std::string& dir, std::string* error);
  // Another copy for a parallel caller that SHARES `base`'s tagger graph and
  // lookahead tables (read-only after `base`'s first call; OpenFst's
  // MatcherFst copy shares the ConstFst and add-on data). Verbalizer and
  // post-processing graphs are read from `dir` per copy (mutable VectorFst).
  static std::unique_ptr<NemoGrammar> load_sharing(const NemoGrammar& base, const std::string& dir,
                                                   std::string* error);
  ~NemoGrammar();

  NemoResult normalize(std::string_view text);
  // select_tag(find_tags(escape(text))) on already-stripped text.
  NemoResult tag(std::string_view raw);

  struct Stats {
    double tag_ms = 0, verbalize_ms = 0, post_ms = 0, moses_ms = 0;
  };
  const Stats& last_stats() const { return stats_; }
  // {"fast":bool,"fast_ok":n,"fast_declined":n}: the flat-array tagger's use.
  std::string tagger_stats() const;

 private:
  NemoGrammar();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Stats stats_;
};

// Production's NormalizerPool hand-off: a caller takes a free grammar copy;
// waiting callers are served by priority (0 = text the first byte waits on,
// 1 = everything else), then arrival order.
class NemoPool {
 public:
  NemoPool(const std::string& dir, int instances, std::string* error);
  bool ok() const { return !grammars_.empty(); }
  NemoResult normalize(std::string_view text, int priority = 1);
  // pool.idle_count(): free copies not already promised to a waiting caller.
  int idle_count();
  int size() const { return static_cast<int>(grammars_.size()); }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::unique_ptr<NemoGrammar>> grammars_;
  std::vector<NemoGrammar*> idle_;
  std::vector<std::pair<int, uint64_t>> waiting_;  // (priority, ticket), unordered
  uint64_t next_ticket_ = 0;
};

// pynini.escape: "[", "]" and "\" get a backslash.
std::string pynini_escape(std::string_view s);
// sacremoses MosesDetokenizer(lang="en").detokenize([text], unescape=False).
std::string moses_detokenize_en(std::string_view text);
// nemo_text_processing ... data_loader_utils.post_process_punct(input, normalized).
std::string post_process_punct(std::string_view input, std::string_view normalized);

}  // namespace kt2n::text
