#include "fast_tagger.h"

#include <fst/fstlib.h>

#include <algorithm>
#include <limits>
#include <queue>

namespace kt2n::text {

std::unique_ptr<FastTagger> FastTagger::build(const fst::Fst<fst::StdArc>& tagger, std::string* why) {
  using fst::StdArc;
  std::unique_ptr<FastTagger> t(new FastTagger());
  const auto start = tagger.Start();
  if (start == fst::kNoStateId) {
    *why = "no start state";
    return nullptr;
  }
  int32_t n = 0;
  for (fst::StateIterator<fst::Fst<StdArc>> si(tagger); !si.Done(); si.Next()) n = std::max<int32_t>(n, si.Value() + 1);
  t->start_ = static_cast<int32_t>(start);
  t->final_.assign(n, std::numeric_limits<float>::infinity());
  t->arc_begin_.assign(n + 1, 0);
  t->eps_end_.assign(n, 0);
  std::vector<std::vector<Arc>> per(n);
  for (int32_t q = 0; q < n; ++q) {
    const auto f = tagger.Final(q);
    if (f != StdArc::Weight::Zero()) t->final_[q] = f.Value();
    for (fst::ArcIterator<fst::Fst<StdArc>> ai(tagger, q); !ai.Done(); ai.Next()) {
      const auto& a = ai.Value();
      if (a.ilabel < 0 || a.ilabel > 255 || a.olabel < 0 || a.olabel > 255) {
        *why = "non-byte label";
        return nullptr;
      }
      per[q].push_back(Arc{static_cast<int32_t>(a.ilabel), static_cast<int32_t>(a.olabel), a.weight.Value(),
                           static_cast<int32_t>(a.nextstate)});
    }
    std::stable_sort(per[q].begin(), per[q].end(), [](const Arc& x, const Arc& y) { return x.ilabel < y.ilabel; });
  }
  for (int32_t q = 0; q < n; ++q) {
    t->arc_begin_[q + 1] = t->arc_begin_[q] + static_cast<int32_t>(per[q].size());
    int32_t e = 0;
    while (e < static_cast<int32_t>(per[q].size()) && per[q][e].ilabel == 0) ++e;
    t->eps_end_[q] = t->arc_begin_[q] + e;
    t->arcs_.insert(t->arcs_.end(), per[q].begin(), per[q].end());
  }
  // Topological order of the epsilon graph (Kahn); a cycle -> decline.
  std::vector<int32_t> indeg(n, 0);
  for (int32_t q = 0; q < n; ++q) {
    for (int32_t k = t->arc_begin_[q]; k < t->eps_end_[q]; ++k) ++indeg[t->arcs_[k].next];
  }
  std::vector<int32_t> order;
  order.reserve(n);
  for (int32_t q = 0; q < n; ++q) {
    if (indeg[q] == 0) order.push_back(q);
  }
  for (size_t h = 0; h < order.size(); ++h) {
    const int32_t q = order[h];
    for (int32_t k = t->arc_begin_[q]; k < t->eps_end_[q]; ++k) {
      if (--indeg[t->arcs_[k].next] == 0) order.push_back(t->arcs_[k].next);
    }
  }
  if (static_cast<int32_t>(order.size()) != n) {
    *why = "input-epsilon cycle";
    return nullptr;
  }
  t->rank_.assign(n, 0);
  for (int32_t r = 0; r < n; ++r) t->rank_[order[r]] = r;
  t->first_.assign(static_cast<size_t>(n) * 4, 0);
  t->final_reach_.assign(n, 0);
  for (int32_t r = n - 1; r >= 0; --r) {
    const int32_t q = order[r];
    uint64_t* fq = &t->first_[static_cast<size_t>(q) * 4];
    uint8_t reach = t->final_[q] != std::numeric_limits<float>::infinity();
    for (int32_t k = t->arc_begin_[q]; k < t->arc_begin_[q + 1]; ++k) {
      const Arc& a = t->arcs_[k];
      if (a.ilabel == 0) {
        const uint64_t* fr = &t->first_[static_cast<size_t>(a.next) * 4];
        for (int w = 0; w < 4; ++w) fq[w] |= fr[w];
        reach |= t->final_reach_[a.next];
      } else {
        fq[a.ilabel >> 6] |= uint64_t{1} << (a.ilabel & 63);
      }
    }
    t->final_reach_[q] = reach;
  }
  for (int l = 0; l < 2; ++l) {
    t->slot_[l].assign(n, -1);
    t->stamp_[l].assign(n, 0);
  }
  return t;
}

FastTagger::Result FastTagger::tag(std::string_view input, std::string* out) {
  ++stats_.calls;
  out->clear();
  const int32_t n = static_cast<int32_t>(input.size());
  // Fresh stamps per layer: layer i of this call uses epoch_ + i + 1.
  if (epoch_ > 0xF0000000u - static_cast<uint32_t>(n) - 4) {
    for (int l = 0; l < 2; ++l) std::fill(stamp_[l].begin(), stamp_[l].end(), 0);
    epoch_ = 0;
  }
  const uint32_t base = epoch_;
  epoch_ += static_cast<uint32_t>(n) + 2;
  nodes_.clear();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  using Item = std::pair<int32_t, int32_t>;  // (rank, node)
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
  std::vector<int32_t> next_layer;

  auto relax = [&](int layer, int32_t state, float d, int32_t parent, int32_t olabel, bool* created) -> int32_t {
    const int l = layer & 1;
    const uint32_t stamp = base + static_cast<uint32_t>(layer) + 1;
    uint64_t h = 0;
    uint32_t len = 0;
    bool amb = false;
    if (parent >= 0) {
      const Node& p = nodes_[parent];
      h = p.out_hash;
      len = p.out_len;
      amb = p.ambiguous;
    }
    if (olabel != 0) {
      h = (h + static_cast<uint64_t>(olabel) + 1) * 0x9E3779B97F4A7C15ull;
      h ^= h >> 29;
      ++len;
    }
    *created = false;
    if (stamp_[l][state] != stamp) {
      stamp_[l][state] = stamp;
      slot_[l][state] = static_cast<int32_t>(nodes_.size());
      nodes_.push_back(Node{state, parent, olabel, d, len, h, amb, -1, 0});
      *created = true;
      return slot_[l][state];
    }
    Node& x = nodes_[slot_[l][state]];
    if (d < x.dist) {
      x.dist = d;
      x.parent = parent;
      x.olabel = olabel;
      x.out_len = len;
      x.out_hash = h;
      x.ambiguous = amb;
      x.alt_parent = -1;
    } else if (d == x.dist) {
      if (amb || h != x.out_hash || len != x.out_len) {
        x.ambiguous = true;
        if (x.alt_parent < 0 && !amb) {
          x.alt_parent = parent;
          x.alt_olabel = olabel;
        }
      }
    }
    return slot_[l][state];
  };

  bool created = false;
  const int32_t s0 = relax(0, start_, 0.0f, -1, 0, &created);
  heap.push({rank_[start_], s0});
  int32_t best = -1;
  float best_total = kInf;
  bool best_tie = false;
  for (int32_t i = 0; i <= n; ++i) {
    const int b = i < n ? static_cast<unsigned char>(input[i]) : -1;
    next_layer.clear();
    std::vector<int32_t> layer_nodes;
    while (!heap.empty()) {
      const int32_t xi = heap.top().second;
      heap.pop();
      const int32_t q = nodes_[xi].state;
      if (i == n) layer_nodes.push_back(xi);
      if (b >= 0) {
        if (!((first_[static_cast<size_t>(q) * 4 + (b >> 6)] >> (b & 63)) & 1)) continue;
      } else if (!final_reach_[q]) {
        continue;
      }
      const float d = nodes_[xi].dist;
      for (int32_t k = arc_begin_[q]; k < eps_end_[q]; ++k) {
        const Arc& a = arcs_[k];
        const int32_t y = relax(i, a.next, d + a.weight, xi, a.olabel, &created);
        if (created) heap.push({rank_[a.next], y});
      }
      if (b >= 0) {
        // non-epsilon arcs are sorted by input label: binary search the byte
        const Arc* lo = &arcs_[eps_end_[q]];
        const Arc* hi = arcs_.data() + arc_begin_[q + 1];
        const Arc* it = std::lower_bound(lo, hi, b, [](const Arc& a, int v) { return a.ilabel < v; });
        for (; it != hi && it->ilabel == b; ++it) {
          const int32_t y = relax(i + 1, it->next, d + it->weight, xi, it->olabel, &created);
          if (created) next_layer.push_back(y);
        }
      }
    }
    if (i == n) {
      for (int32_t xi : layer_nodes) {
        const Node& x = nodes_[xi];
        const float f = final_[x.state];
        if (f == kInf) continue;
        const float total = x.dist + f;
        if (total < best_total) {
          best_total = total;
          best = xi;
          best_tie = false;
        } else if (total == best_total) {
          const Node& b = nodes_[best];
          if (x.ambiguous || x.out_hash != b.out_hash || x.out_len != b.out_len) best_tie = true;
        }
      }
      break;
    }
    if (next_layer.empty()) break;
    for (int32_t y : next_layer) heap.push({rank_[nodes_[y].state], y});
  }
  stats_.nodes += nodes_.size();
  if (best < 0) return kNoPath;
  // The winning node's flag already covers every node of its minimal chains.
  const bool tie = best_tie || nodes_[best].ambiguous;
  last_best_ = best;
  std::string rev;
  for (int32_t xi = best; xi >= 0; xi = nodes_[xi].parent) {
    if (nodes_[xi].olabel != 0) rev.push_back(static_cast<char>(nodes_[xi].olabel));
  }
  if (tie) {
    ++stats_.ties;
    return kTie;
  }
  out->assign(rev.rbegin(), rev.rend());
  return kOk;
}

bool FastTagger::last_tie_outputs(std::string* best, std::string* rival) const {
  if (last_best_ < 0) return false;
  auto chain_out = [&](int32_t xi) {
    std::string rev;
    for (; xi >= 0; xi = nodes_[xi].parent) {
      if (nodes_[xi].olabel) rev.push_back(static_cast<char>(nodes_[xi].olabel));
    }
    return std::string(rev.rbegin(), rev.rend());
  };
  *best = chain_out(last_best_);
  // the deepest node on the winning chain whose rival is recorded
  std::string suffix_rev;
  for (int32_t xi = last_best_; xi >= 0; xi = nodes_[xi].parent) {
    const Node& x = nodes_[xi];
    if (x.alt_parent >= 0) {
      std::string head = chain_out(x.alt_parent);
      if (x.alt_olabel) head.push_back(static_cast<char>(x.alt_olabel));
      *rival = head + std::string(suffix_rev.rbegin(), suffix_rev.rend());
      return true;
    }
    if (x.olabel) suffix_rev.push_back(static_cast<char>(x.olabel));
  }
  return false;
}

}  // namespace kt2n::text
