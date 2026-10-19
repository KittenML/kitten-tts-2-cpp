// L6 HiFT as the render call's post-graph stage (include/kt2n/render/hift_stage.h).
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"
#include "kt2n/render/hift.h"
#include "kt2n/render/hift_stage.h"
#include "cu_driver.h"
#include "device_zero.h"

namespace kt2n::render {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

using drv::CtxScope;  // green context scope (driver entry points resolved at run time, cu_driver.h)

struct HostTensor {
  std::vector<int64_t> shape;
  const float* data = nullptr;
  size_t numel = 0;
};

// [cout][cin][K] -> [cin][K][cout] (L6's layout for stride-1 convs).
std::vector<float> to_cin_k_cout(const HostTensor& t) {
  const int64_t O = t.shape[0], C = t.shape[1], K = t.shape[2];
  std::vector<float> r(t.numel);
  for (int64_t o = 0; o < O; ++o)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t k = 0; k < K; ++k) r[size_t((c * K + k) * O + o)] = t.data[size_t((o * C + c) * K + k)];
  return r;
}

// Fill the input of one equal-width group: in[b][c][t] = t < mc ? mel cache : new frame t - mc.
struct PackRow {
  int32_t slot, mc, src_frame;  // src_frame: first new frame in mel_out (packed index)
};
__global__ void hift_pack_kernel(const PackRow* __restrict__ rows, int32_t T, const float* __restrict__ mel_out,
                                 const float* __restrict__ mel_cache, float* __restrict__ in) {
  const int32_t b = blockIdx.y;
  const PackRow r = rows[b];
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < kMel * T; i += gridDim.x * blockDim.x) {
    const int32_t c = i / T, t = i - c * T;
    float v;
    // L6 writes the cache in the [80][8] slot layout, first min(8, T) columns valid (hift.h, 10 Oct 09:4x).
    if (t < r.mc) v = mel_cache[(size_t(r.slot) * kMel + size_t(c)) * kMelCacheFrames + size_t(t)];
    else v = mel_out[(size_t(r.src_frame) + size_t(t - r.mc)) * kMel + size_t(c)];
    in[(size_t(b) * kMel + size_t(c)) * size_t(T) + size_t(t)] = v;
  }
}

}  // namespace

struct HiftStage::Impl {
  std::vector<void*> weights;  // device
  static constexpr int kR = kCallerRoles;  // urgent, bulk, aux (RenderCaller role index)
  void* handle[kR] = {};
  int32_t slots = 0, max_b = 0;
  int32_t max_t[kR] = {}, max_fr[kR] = {};  // per role: urgent, bulk (bulk covers whole renders), aux (0 = none)
  float* mel_cache = nullptr;  // [slots][80 * 8]
  float* src = nullptr;        // [slots][2][3840]
  float* held = nullptr;       // [slots][2][3840]
  std::vector<uint8_t> parity;
  float* d_in[kR] = {};          // [max_b][80][max_t]
  Kt2nHiftRow* d_rows[kR] = {};  // [max_b] per group (sequential groups: reused)
  PackRow* d_pack[kR] = {};
  Kt2nHiftRow* h_rows = nullptr;                // pinned [kR][64][max_b]
  PackRow* h_pack = nullptr;                    // pinned [kR][64][max_b]
  int32_t ring[kR] = {};
  // Pipelined post stage (RenderCallerConfig::pipeline_post): the pack kernel runs on the flow stream (so the
  // role's next flow call may overwrite its mel output right after), L6 on the post stream. The packed HiFT
  // input is a ring of kInRing buffers per role: pack g writes buffer j after the L6 run that last read j is
  // done (ev_free), L6 g starts after pack g (ev_packed).
  static constexpr int kInRing = 4;
  float* d_in_ring[kR][kInRing] = {};
  cudaEvent_t ev_packed[kR][kInRing] = {};
  cudaEvent_t ev_free[kR][kInRing] = {};
  bool free_recorded[kR][kInRing] = {};
  int32_t in_ring[kR] = {};
  int64_t mismatches = 0;
  // Graph mode (hift_stage.h): one exec per (role, B, T); first-use counts decide when to capture.
  bool graph_mode = false;  // measured (12:10): no GPU gain, +1.4 ms GPU at B10; KT2N_HIFT_GRAPH=1 turns it on
  std::map<std::tuple<int32_t, int32_t, int32_t>, cudaGraphExec_t> graphs;
  std::map<std::tuple<int32_t, int32_t, int32_t>, int32_t> seen;
  HiftStage::GraphStats stats;
  static constexpr size_t kMaxGraphsPerStage = 1024;

  // Group work for (role, B, T) reading the FIXED descriptor region of the role (d_rows / d_pack slot 0).
  void enqueue_group(int32_t role, int32_t B, int32_t T, const float* mel_out, cudaStream_t s) {
    hift_pack_kernel<<<dim3(unsigned((kMel * T + 255) / 256), unsigned(B)), 256, 0, s>>>(d_pack[role], T, mel_out,
                                                                                       mel_cache, d_in[role]);
    check(cudaGetLastError(), "hift pack kernel");
    const int rc = kt2n_hift_run_rows(handle[role], B, T, d_in[role], d_rows[role], nullptr, nullptr, s);
    if (rc != 0) throw std::runtime_error(std::string("kt2n_hift_run_rows: ") + kt2n_hift_last_error());
  }
  cudaGraphExec_t capture(int32_t role, int32_t B, int32_t T, const float* mel_out, cudaStream_t s) {
    cudaGraph_t g = nullptr;
    check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "hift begin capture");
    try {
      enqueue_group(role, B, T, mel_out, s);
    } catch (...) {
      cudaStreamEndCapture(s, &g);
      if (g) cudaGraphDestroy(g);
      throw;
    }
    check(cudaStreamEndCapture(s, &g), "hift end capture");
    cudaGraphExec_t e = nullptr;
    check(cudaGraphInstantiate(&e, g, 0), "hift graph instantiate");
    cudaGraphDestroy(g);
    check(cudaGraphUpload(e, s), "hift graph upload");
    ++stats.captures;
    return e;
  }
};

HiftStage::HiftStage() : impl_(new Impl) {
  const char* g = std::getenv("KT2N_HIFT_GRAPH");
  if (g != nullptr) impl_->graph_mode = g[0] == '1';
}
HiftStage::~HiftStage() {
  Impl& m = *impl_;
  for (auto& kv : m.graphs) cudaGraphExecDestroy(kv.second);
  for (void* h : m.handle)
    if (h) kt2n_hift_destroy(h);
  for (void* w : m.weights) cudaFree(w);
  cudaFree(m.mel_cache);
  cudaFree(m.src);
  cudaFree(m.held);
  for (int r = 0; r < Impl::kR; ++r) {
    cudaFree(m.d_in[r]);
    cudaFree(m.d_rows[r]);
    cudaFree(m.d_pack[r]);
    for (int j = 0; j < Impl::kInRing; ++j) {
      cudaFree(m.d_in_ring[r][j]);
      if (m.ev_packed[r][j]) cudaEventDestroy(m.ev_packed[r][j]);
      if (m.ev_free[r][j]) cudaEventDestroy(m.ev_free[r][j]);
    }
  }
  cudaFreeHost(m.h_rows);
  cudaFreeHost(m.h_pack);
}

std::unique_ptr<HiftStage> HiftStage::load(const std::string& path, int32_t slots, int32_t max_b, int32_t max_t,
                                           int32_t max_frame_rows, std::string* err) {
  return load(path, slots, max_b, HiftRoleSize{max_t, max_frame_rows}, HiftRoleSize{max_t, max_frame_rows}, err);
}

std::unique_ptr<HiftStage> HiftStage::load(const std::string& path, int32_t slots, int32_t max_b, HiftRoleSize urgent,
                                           HiftRoleSize bulk, std::string* err, CUctx_st* bulk_ctx, HiftRoleSize aux) {
  std::unique_ptr<HiftStage> st(new HiftStage());
  Impl& m = *st->impl_;
  try {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat sb {};
    fstat(fd, &sb);
    const size_t size = size_t(sb.st_size);
    void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) throw std::runtime_error("mmap failed");
    const auto* base = static_cast<const uint8_t*>(map);
    if (size < 64 || std::memcmp(base, "KT2NS3G1", 8) != 0) throw std::runtime_error("not an s3gen.kt2n file");
    uint64_t meta_off, meta_len, data_off;
    std::memcpy(&meta_off, base + 16, 8);
    std::memcpy(&meta_len, base + 24, 8);
    std::memcpy(&data_off, base + 32, 8);
    const json::Value meta = json::parse(std::string_view(reinterpret_cast<const char*>(base + meta_off), meta_len));
    std::map<std::string, HostTensor> t;
    for (const auto& e : meta.get("tensors")->items()) {
      if (e.get("group")->as_string() != "hift") continue;
      if (e.get("dtype")->as_string() != "f32") throw std::runtime_error("hift tensor not f32");
      const json::Value* d = e.get("data");
      HostTensor x;
      x.numel = 1;
      for (const auto& s : d->get("shape")->items()) {
        x.shape.push_back(s.as_int());
        x.numel *= size_t(s.as_int());
      }
      const uint64_t off = uint64_t(d->get("off")->as_int()), len = uint64_t(d->get("len")->as_int());
      if (len != x.numel * 4 || data_off + off + len > size) throw std::runtime_error("bad hift segment");
      x.data = reinterpret_cast<const float*>(base + data_off + off);
      t[e.get("name")->as_string()] = x;
    }
    auto get = [&](const std::string& n) -> const HostTensor& {
      auto it = t.find("mel2wav." + n);
      if (it == t.end()) throw std::runtime_error("missing hift tensor " + n);
      return it->second;
    };
    std::vector<std::vector<float>> host;
    auto add = [&](const HostTensor& x) { host.emplace_back(x.data, x.data + x.numel); };
    auto add_t = [&](const HostTensor& x) { host.push_back(to_cin_k_cout(x)); };
    auto add_snake = [&](const HostTensor& alpha) {
      add(alpha);
      std::vector<float> inv(alpha.numel);
      for (size_t i = 0; i < alpha.numel; ++i) inv[i] = 1.0f / (alpha.data[i] + 1e-9f);  // Snake no_div_by_zero
      host.push_back(std::move(inv));
    };
    for (int i : {0, 2, 4, 6, 8}) {
      add_t(get("f0_predictor.condnet." + std::to_string(i) + ".weight"));
      add(get("f0_predictor.condnet." + std::to_string(i) + ".bias"));
    }
    add(get("f0_predictor.classifier.weight"));
    add(get("f0_predictor.classifier.bias"));
    add(get("m_source.l_linear.weight"));
    add(get("m_source.l_linear.bias"));
    add_t(get("conv_pre.weight"));
    add(get("conv_pre.bias"));
    auto resblock = [&](const std::string& p) {
      for (int j = 0; j < 3; ++j) {
        const std::string J = std::to_string(j);
        add_snake(get(p + ".activations1." + J + ".alpha"));
        add_t(get(p + ".convs1." + J + ".weight"));
        add(get(p + ".convs1." + J + ".bias"));
        add_snake(get(p + ".activations2." + J + ".alpha"));
        add_t(get(p + ".convs2." + J + ".weight"));
        add(get(p + ".convs2." + J + ".bias"));
      }
    };
    for (int i = 0; i < 3; ++i) {
      const std::string I = std::to_string(i);
      add(get("ups." + I + ".weight"));
      add(get("ups." + I + ".bias"));
      add(get("source_downs." + I + ".weight"));
      add(get("source_downs." + I + ".bias"));
      resblock("source_resblocks." + I);
      for (int j = 0; j < 3; ++j) resblock("resblocks." + std::to_string(i * 3 + j));
    }
    add_t(get("conv_post.weight"));
    add(get("conv_post.bias"));
    // stft_window: scipy get_window("hann", 16, fftbins=True) (float64) -> float32.
    std::vector<float> win(kIstftNfft);
    for (int n = 0; n < kIstftNfft; ++n) win[size_t(n)] = float(0.5 - 0.5 * std::cos(2.0 * M_PI * double(n) / double(kIstftNfft)));
    host.push_back(win);
    munmap(map, size);
    if (host.size() != 319) throw std::runtime_error("hift weight list has " + std::to_string(host.size()) + " tensors");
    for (const auto& h : host) {
      void* d = nullptr;
      check(cudaMalloc(&d, h.size() * sizeof(float)), "hift weight");
      check(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "hift weight");
      m.weights.push_back(d);
    }
    for (int r = 0; r < Impl::kR; ++r) {
      const HiftRoleSize z = r == 0 ? urgent : r == 1 ? bulk : aux;
      if (r == 2 && z.max_t <= 0) continue;  // no aux role
      m.max_t[r] = z.max_t;
      m.max_fr[r] = std::max(z.max_frame_rows, z.max_t);
      CtxScope cs(r == 1 ? bulk_ctx : nullptr);  // the handle's side streams / events belong to the role's context
      m.handle[r] = kt2n_hift_create2(const_cast<const void* const*>(m.weights.data()), int(m.weights.size()), max_b,
                                      m.max_t[r], m.max_fr[r]);
      if (m.handle[r] == nullptr) throw std::runtime_error(std::string("kt2n_hift_create: ") + kt2n_hift_last_error());
      if (z.tc_min_k != -1 && kt2n_hift_set_policy(m.handle[r], z.tc_min_k, 0) != 0)
        throw std::runtime_error(std::string("kt2n_hift_set_policy: ") + kt2n_hift_last_error());
      check(cudaMalloc(reinterpret_cast<void**>(&m.d_in[r]), size_t(m.max_fr[r]) * kMel * sizeof(float)), "hift in");
      check(cudaMalloc(reinterpret_cast<void**>(&m.d_rows[r]), size_t(max_b) * 64 * sizeof(Kt2nHiftRow)), "hift rows");
      check(cudaMalloc(reinterpret_cast<void**>(&m.d_pack[r]), size_t(max_b) * 64 * sizeof(PackRow)), "hift pack");
      for (int j = 0; j < Impl::kInRing; ++j) {
        check(cudaMalloc(reinterpret_cast<void**>(&m.d_in_ring[r][j]), size_t(m.max_fr[r]) * kMel * sizeof(float)),
              "hift in ring");
        check(cudaEventCreateWithFlags(&m.ev_packed[r][j], cudaEventDisableTiming), "event");
        check(cudaEventCreateWithFlags(&m.ev_free[r][j], cudaEventDisableTiming), "event");
      }
    }
    check(cudaHostAlloc(reinterpret_cast<void**>(&m.h_rows), size_t(Impl::kR) * 64 * size_t(max_b) * sizeof(Kt2nHiftRow),
                        cudaHostAllocPortable), "pinned");
    check(cudaHostAlloc(reinterpret_cast<void**>(&m.h_pack), size_t(Impl::kR) * 64 * size_t(max_b) * sizeof(PackRow),
                        cudaHostAllocPortable), "pinned");
    m.slots = slots;
    m.max_b = max_b;
    check(cudaMalloc(reinterpret_cast<void**>(&m.mel_cache), size_t(slots) * kMel * kMelCacheFrames * sizeof(float)), "cache");
    check(cudaMalloc(reinterpret_cast<void**>(&m.src), size_t(slots) * 2 * kSourceCacheSamples * sizeof(float)), "cache");
    check(cudaMalloc(reinterpret_cast<void**>(&m.held), size_t(slots) * 2 * kSourceCacheSamples * sizeof(float)), "cache");
    zero_device(m.mel_cache, size_t(slots) * kMel * kMelCacheFrames * sizeof(float), "cache");
    m.parity.assign(size_t(slots), 0);
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return nullptr;
  }
  return st;
}

int64_t HiftStage::plan_mismatches() const { return impl_->mismatches; }

namespace {
CUstream_st* hift_post(void* ctx, const CallPlan& plan, const CallDesc& h, const CallDesc*, int32_t role, CUstream_st* s,
                       CUstream_st* ps) {
  auto& m = *static_cast<HiftStage::Impl*>(ctx);
  const bool split = ps != nullptr && ps != s && !m.graph_mode;  // graph mode keeps everything on `s`
  // Group real rows by T (vocoder input width).
  std::map<int32_t, std::vector<int32_t>> groups;
  for (int32_t i = 0; i < plan.n_rows; ++i) {
    const RowDesc& d = plan.rows[size_t(i)];
    if (d.slot < 0 || (d.flags & kRowCancelled) || d.out_fr_count <= 0) continue;
    groups[d.mel_cache + d.out_fr_count].push_back(i);
  }
  // Pinned staging is a ring of 64 groups per role (host writes must not race an in-flight copy).
  for (auto& [T, idx] : groups) {
    if (T > m.max_t[role] || T > m.max_fr[role])
      throw std::runtime_error("HiFT width " + std::to_string(T) + " exceeds the role's max_t " + std::to_string(m.max_t[role]));
    const size_t per = size_t(std::max(1, std::min(m.max_b, m.max_fr[role] / T)));
    for (size_t g0 = 0; g0 < idx.size(); g0 += per) {
      const int32_t B = int32_t(std::min(idx.size() - g0, per));
      const int32_t slot_ring = m.ring[role];
      m.ring[role] = (m.ring[role] + 1) % 64;
      Kt2nHiftRow* hr = m.h_rows + (size_t(role) * 64 + size_t(slot_ring)) * size_t(m.max_b);
      PackRow* hp = m.h_pack + (size_t(role) * 64 + size_t(slot_ring)) * size_t(m.max_b);
      for (int32_t b = 0; b < B; ++b) {
        const RowDesc& d = plan.rows[size_t(idx[g0 + size_t(b)])];
        const int32_t slot = d.slot;
        if (d.committed_tokens == 0 && d.src_cache == 0) m.parity[size_t(slot)] = 0;  // a new segment row
        const int32_t par = m.parity[size_t(slot)];
        Kt2nHiftPlanIn in{};
        in.frames = T;
        in.finalize = (d.flags & kRowFinalize) ? 1 : 0;
        in.first_commit_tail = 1;
        in.short_first = 1;
        in.row_fade = kTailFadeSamples;
        in.commit_index = d.reserved[0];
        in.held_len = d.src_cache;
        in.pending_tail = d.publish_skip;
        Kt2nHiftPlanOut po{};
        kt2n_hift_plan(&in, &po);
        if (po.out_len != d.pcm_count || po.error) ++m.mismatches;
        Kt2nHiftRow r{};
        std::memset(&r, 0, sizeof r);
        r.seed = d.philox_seed;
        float* src0 = m.src + (size_t(slot) * 2 + size_t(par)) * kSourceCacheSamples;
        float* src1 = m.src + (size_t(slot) * 2 + size_t(1 - par)) * kSourceCacheSamples;
        float* held0 = m.held + (size_t(slot) * 2 + size_t(par)) * kSourceCacheSamples;
        float* held1 = m.held + (size_t(slot) * 2 + size_t(1 - par)) * kSourceCacheSamples;
        r.src_in = d.src_cache > 0 ? src0 : nullptr;
        r.src_out = src1;
        r.mel_cache_out = m.mel_cache + size_t(slot) * kMel * kMelCacheFrames;
        r.held_in = d.src_cache > 0 ? held0 : nullptr;
        r.held_out = held1;
        r.out = h.pcm_f32 + d.pcm_off;
        r.src_len = d.src_cache;
        r.held_len = d.src_cache;
        r.flags = po.flags;
        r.emit_len = po.emit_len;
        r.overlap = po.overlap;
        r.skip = po.skip;
        r.splice_count = po.splice_count;
        r.splice_start = po.splice_start;
        r.out_len = po.out_len;
        r.tail_len = po.tail_len;
        kt2n_hift_phases(r.seed, r.phase);
        hr[b] = r;
        if (po.flags & KT2N_HIFT_HOLD) m.parity[size_t(slot)] ^= 1;
        // The mel comes from the estimator, which writes at ITS 16-aligned packing (CallPlan::est_rows).
        const int32_t mel_off = plan.est_rows.empty() ? d.fr_off : plan.est_rows[size_t(idx[g0 + size_t(b)])].fr_off;
        hp[b] = PackRow{slot, d.mel_cache, mel_off + (d.out_fr_begin - d.fr_past)};
      }
      if (split) {
        // Pack on the flow stream (reads this call's mel output, which the next flow call overwrites), L6 on
        // the post stream. d_pack is read only by pack kernels (all on `s`), d_rows only by L6 (all on `ps`), so
        // each fixed region is refreshed in the order of its readers.
        const int j = m.in_ring[role];
        m.in_ring[role] = (j + 1) % HiftStage::Impl::kInRing;
        float* in = m.d_in_ring[role][j];
        if (m.free_recorded[role][j]) check(cudaStreamWaitEvent(s, m.ev_free[role][j], 0), "hift in free");
        check(cudaMemcpyAsync(m.d_pack[role], hp, size_t(B) * sizeof(PackRow), cudaMemcpyHostToDevice, s), "hift pack");
        hift_pack_kernel<<<dim3(unsigned((kMel * T + 255) / 256), unsigned(B)), 256, 0, s>>>(m.d_pack[role], T, h.mel_out,
                                                                                           m.mel_cache, in);
        check(cudaGetLastError(), "hift pack kernel");
        check(cudaEventRecord(m.ev_packed[role][j], s), "hift packed");
        check(cudaStreamWaitEvent(ps, m.ev_packed[role][j], 0), "hift wait packed");
        check(cudaMemcpyAsync(m.d_rows[role], hr, size_t(B) * sizeof(Kt2nHiftRow), cudaMemcpyHostToDevice, ps), "hift rows");
        const int rc = kt2n_hift_run_rows(m.handle[role], B, T, in, m.d_rows[role], nullptr, nullptr, ps);
        if (rc != 0) throw std::runtime_error(std::string("kt2n_hift_run_rows: ") + kt2n_hift_last_error());
        check(cudaEventRecord(m.ev_free[role][j], ps), "hift in free");
        m.free_recorded[role][j] = true;
        ++m.stats.eager_launches;
        continue;
      }
      // Device descriptors at the role's FIXED region (graphs bake these addresses); the copies are ordered
      // on `s` after the previous group's / call's HiFT work, which is what makes reusing the region safe.
      // The pinned host side stays a ring: an async copy reads it when it executes, not when enqueued.
      check(cudaMemcpyAsync(m.d_rows[role], hr, size_t(B) * sizeof(Kt2nHiftRow), cudaMemcpyHostToDevice, s), "hift rows");
      check(cudaMemcpyAsync(m.d_pack[role], hp, size_t(B) * sizeof(PackRow), cudaMemcpyHostToDevice, s), "hift pack");
      cudaGraphExec_t exec = nullptr;
      if (m.graph_mode) {
        const auto key = std::make_tuple(int32_t(role), B, T);
        auto it = m.graphs.find(key);
        if (it != m.graphs.end()) {
          exec = it->second;
        } else if (m.seen[key]++ >= 1 && m.graphs.size() < HiftStage::Impl::kMaxGraphsPerStage) {
          exec = m.capture(role, B, T, h.mel_out, s);
          m.graphs.emplace(key, exec);
        }
      }
      if (exec != nullptr) {
        check(cudaGraphLaunch(exec, s), "hift graph launch");
        ++m.stats.graph_launches;
      } else {
        m.enqueue_group(role, B, T, h.mel_out, s);
        ++m.stats.eager_launches;
      }
    }
  }
  return split ? ps : s;
}
}  // namespace

void HiftStage::set_graph_mode(bool on) { impl_->graph_mode = on; }
int32_t HiftStage::max_width(int32_t role) const { return role >= 0 && role < Impl::kR ? impl_->max_t[role] : 0; }

void HiftStage::prewarm(const RenderCaller& caller, int32_t max_shared_rows) {
  Impl& m = *impl_;
  if (!m.graph_mode) return;
  auto add = [&](StreamRole r, int32_t B, int32_t T) {
    const int32_t role = caller_role(r);
    if (role < 0 || m.handle[role] == nullptr || B > m.max_b || T > m.max_t[role] || B * T > m.max_fr[role]) return;
    const auto key = std::make_tuple(role, B, T);
    if (m.graphs.count(key) != 0) return;
    m.graphs.emplace(key, m.capture(role, B, T, caller.device_mel_out(r), caller.stream(r)));
  };
  add(StreamRole::kSUrgent, 1, 8);    // alone commit 0 (4 tokens, no mel cache)
  add(StreamRole::kSUrgent, 1, 18);   // alone commit 1 (5 tokens + 8 cached frames)
  for (int32_t b = 1; b <= max_shared_rows; ++b) add(StreamRole::kSUrgent, b, 10);  // shared commit 0 (5 tokens)
  for (int32_t T : {18, 28, 38, 58, 108}) add(StreamRole::kSBulk, 1, T);  // alone commits 2-6
  add(StreamRole::kSAux, 1, 8);  // aux role (when on): a first commit while S_urgent is busy
  for (int32_t b = 1; b <= max_shared_rows; ++b) add(StreamRole::kSAux, b, 10);
}

void HiftStage::warm(const RenderCaller& caller) {
  Impl& m = *impl_;
  // Run L6 once per shape class on scratch rows so CUDA's lazy module loading happens here, not inside the
  // first live call (measured 12:13: the first commit 0 of a process spent 6.3 ms in launch with eager HiFT,
  // 1.1 ms with CUDA_MODULE_LOADING=EAGER). Small and large widths take different conv kernels (SIMT split vs
  // tensor-core), so both are run. Nothing of the slots' state is touched.
  struct Shape {
    int32_t role, B, T;
  };
  std::vector<Shape> shapes = {{0, 1, 8}, {0, std::min(m.max_b, 10), 10}, {1, 1, 108}, {1, std::min(m.max_b, 10), 108},
                               {1, 1, std::min(m.max_t[1], 1024)}, {2, 1, 8}, {2, std::min(m.max_b, 4), 10}};
  size_t max_out = 0, max_in = 0;
  for (const Shape& z : shapes) {
    max_out = std::max(max_out, size_t(z.B) * size_t(z.T) * 480);
    max_in = std::max(max_in, size_t(z.B) * size_t(z.T) * kMel);
  }
  float *out = nullptr, *scratch = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&out), max_out * sizeof(float)), "hift warm");
  check(cudaMalloc(reinterpret_cast<void**>(&scratch), size_t(kMel) * kMelCacheFrames * sizeof(float) + 4 * kSourceCacheSamples * sizeof(float)),
        "hift warm");
  std::vector<Kt2nHiftRow> rows;
  for (const Shape& z : shapes) {
    if (m.handle[z.role] == nullptr || z.T > m.max_t[z.role] || z.B * z.T > m.max_fr[z.role]) continue;
    const StreamRole sr = z.role == 0 ? StreamRole::kSUrgent : z.role == 1 ? StreamRole::kSBulk : StreamRole::kSAux;
    CtxScope cs(caller.role_context(sr));
    CUstream_st* s = caller.stream(sr);
    rows.assign(size_t(z.B), Kt2nHiftRow{});
    for (int32_t b = 0; b < z.B; ++b) {
      Kt2nHiftPlanIn in{};
      in.frames = z.T;
      in.finalize = 1;
      in.first_commit_tail = 1;
      in.short_first = 1;
      in.row_fade = kTailFadeSamples;
      Kt2nHiftPlanOut po{};
      kt2n_hift_plan(&in, &po);
      Kt2nHiftRow& r = rows[size_t(b)];
      std::memset(&r, 0, sizeof r);
      r.seed = 3000000ull + uint64_t(b);
      r.src_out = scratch + size_t(kMel) * kMelCacheFrames;
      r.held_out = r.src_out + kSourceCacheSamples;
      r.mel_cache_out = scratch;
      r.out = out + size_t(b) * size_t(z.T) * 480;
      r.flags = po.flags & ~KT2N_HIFT_HOLD;
      r.emit_len = po.emit_len;
      r.overlap = po.overlap;
      r.out_len = std::min<int32_t>(po.out_len, z.T * 480);
      r.tail_len = po.tail_len;
      kt2n_hift_phases(r.seed, r.phase);
    }
    check(cudaMemcpyAsync(m.d_rows[z.role], rows.data(), rows.size() * sizeof(Kt2nHiftRow), cudaMemcpyHostToDevice, s),
          "hift warm rows");
    check(cudaMemsetAsync(m.d_in[z.role], 0, size_t(z.B) * size_t(z.T) * kMel * sizeof(float), s), "hift warm in");
    if (kt2n_hift_run_rows(m.handle[z.role], z.B, z.T, m.d_in[z.role], m.d_rows[z.role], nullptr, nullptr, s) != 0)
      throw std::runtime_error(std::string("hift warm: ") + kt2n_hift_last_error());
    check(cudaStreamSynchronize(s), "hift warm");
  }
  cudaFree(out);
  cudaFree(scratch);
  (void)max_in;
}

HiftStage::GraphStats HiftStage::graph_stats() const {
  GraphStats g = impl_->stats;
  g.cached = int32_t(impl_->graphs.size());
  return g;
}

RenderPostStage HiftStage::post_stage() {
  RenderPostStage p;
  p.name = "l6.hift";
  p.ctx = impl_.get();
  p.run = &hift_post;
  return p;
}

}  // namespace kt2n::render
