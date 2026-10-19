// Convert the public Kitten text-processing compact graph to OpenFst.
// State IDs, arc order and binary32 weights are preserved exactly.
#include <fst/fstlib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) try {
  if (argc != 3) throw std::runtime_error("usage: kt2n_prepare_text_data INPUT.fst OUTPUT.fst");
  std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open input graph");
  const auto length = in.tellg();
  if (length < 24 || length > std::streamoff(1ULL << 32)) throw std::runtime_error("invalid graph size");
  std::vector<unsigned char> data(static_cast<size_t>(length));
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(data.data()), length)) throw std::runtime_error("truncated graph");
  if (std::memcmp(data.data(), "KITTEN1\0", 8)) throw std::runtime_error("not a public Kitten compact graph");
  size_t cursor = 8;
  auto read = [&]() {
    const auto* p = data.data() + cursor;
    cursor += 4;
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
  };
  const uint32_t start = read(), states = read(), arcs = read();
  if (!states || start >= states || states > uint32_t(std::numeric_limits<int32_t>::max()) ||
      24ULL + 8ULL * states + 12ULL * arcs != data.size()) throw std::runtime_error("invalid graph dimensions");
  std::vector<uint32_t> offsets(size_t(states) + 1), labels(arcs), targets(arcs);
  std::vector<float> finals(states), weights(arcs);
  for (auto& v : offsets) v = read();
  auto read_float = [&]() { const auto bits = read(); float v; std::memcpy(&v, &bits, 4); return v; };
  for (auto& v : finals) v = read_float();
  for (auto& v : labels) v = read();
  for (auto& v : targets) v = read();
  for (auto& v : weights) v = read_float();
  if (offsets.front() || offsets.back() != arcs || !std::is_sorted(offsets.begin(), offsets.end()))
    throw std::runtime_error("invalid arc offsets");
  fst::StdVectorFst graph;
  graph.ReserveStates(states);
  for (uint32_t i = 0; i < states; ++i) graph.AddState();
  graph.SetStart(static_cast<int32_t>(start));
  for (uint32_t i = 0; i < states; ++i) {
    if (std::isnan(finals[i]) || finals[i] == -std::numeric_limits<float>::infinity())
      throw std::runtime_error("invalid final weight");
    graph.SetFinal(static_cast<int32_t>(i), fst::TropicalWeight(finals[i]));
    for (uint32_t j = offsets[i]; j < offsets[i + 1]; ++j) {
      if (targets[j] >= states || labels[j] > 65535 || !std::isfinite(weights[j]))
        throw std::runtime_error("invalid arc");
      graph.AddArc(static_cast<int32_t>(i), fst::StdArc(labels[j] & 255, labels[j] >> 8,
                   fst::TropicalWeight(weights[j]), static_cast<int32_t>(targets[j])));
    }
  }
  if (!graph.Write(argv[2])) throw std::runtime_error("cannot write OpenFst graph");
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
