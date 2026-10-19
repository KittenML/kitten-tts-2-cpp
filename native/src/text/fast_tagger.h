// L9a text: a flat-array Viterbi for the NeMo tagger, exact or it declines.
//
// Production tags with OpenFst: compose(string acceptor, tagger) with an
// input-label lookahead matcher, connect, ShortestPath(n=1). Composing a
// STRING with the tagger is a left-to-right walk over (tagger state, input
// position) pairs, so the same shortest path can be found with plain arrays:
// per input position, relax the tagger's input-epsilon arcs in a fixed
// topological order of its epsilon graph, then the arcs that read the next
// byte. A per-state "which bytes can I read next through epsilons" bitset does
// the lookahead's pruning.
//
// Exactness: path weights are the same float sums, accumulated in the same
// order (start to end), and a node keeps a candidate only on strict
// improvement, as OpenFst's relaxation does. OpenFst's answer is some chain of
// node-minimal relaxations ending in the minimal final total; which one it
// keeps when candidates tie exactly depends on its queue order. So each node
// also tracks whether all its minimal chains write the SAME output (64-bit hash
// + length of the output so far). Equal-weight duplicates that write the same
// tags (common in NeMo's grammar) are harmless; when the winning node is
// reached by minimal chains with different outputs, tag() returns kTie and the
// caller runs the OpenFst tagger, so the answer is always production's.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <fst/fst-decl.h>

namespace kt2n::text {

class FastTagger {
 public:
  enum Result { kOk = 0, kNoPath = 1, kTie = 3 };

  // nullptr (with *why) when the graph is outside what this search handles
  // (input-epsilon cycle, non-byte input labels, negative weights on epsilons);
  // the caller then always uses OpenFst.
  static std::unique_ptr<FastTagger> build(const fst::Fst<fst::StdArc>& tagger, std::string* why);

  Result tag(std::string_view input, std::string* out);
  // Debug (study only): after a kTie, the winner's output and one rival
  // minimal chain's output, so the kind of tie can be inspected.
  bool last_tie_outputs(std::string* best, std::string* rival) const;

  struct Stats {
    uint64_t calls = 0, ties = 0, nodes = 0;
  };
  const Stats& stats() const { return stats_; }

 private:
  struct Arc {
    int32_t ilabel;
    int32_t olabel;
    float weight;
    int32_t next;
  };
  struct Node {
    int32_t state;
    int32_t parent;  // node index, -1 for the start
    int32_t olabel;  // output label of the arc into this node
    float dist;
    uint32_t out_len;  // output bytes written so far
    uint64_t out_hash;
    bool ambiguous;  // minimal chains into this node write different outputs
    int32_t alt_parent = -1;  // debug: first equal-weight rival (different output)
    int32_t alt_olabel = 0;
  };

  int32_t start_ = -1;
  std::vector<float> final_;          // +inf when not final
  std::vector<int32_t> arc_begin_;    // CSR, size n+1
  std::vector<int32_t> eps_end_;      // first non-epsilon arc of a state
  std::vector<Arc> arcs_;
  std::vector<int32_t> rank_;         // topological rank in the epsilon graph
  std::vector<uint64_t> first_;       // 4 words per state: bytes readable via epsilons
  std::vector<uint8_t> final_reach_;  // a final state reachable via epsilons

  // scratch (one tagger per thread)
  std::vector<int32_t> slot_[2];
  std::vector<uint32_t> stamp_[2];
  uint32_t epoch_ = 0;
  std::vector<Node> nodes_;
  int32_t last_best_ = -1;
  std::vector<std::pair<int32_t, int32_t>> heap_;  // (rank, node)
  Stats stats_;
};

}  // namespace kt2n::text
