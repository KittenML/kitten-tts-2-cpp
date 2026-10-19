// L7 render program (include/kt2n/render/program.h).
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "kt2n/render/call.h"
#include "kt2n/render/program.h"

namespace kt2n::render {
namespace {

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Estimator inputs for every real row, at the ESTIMATOR's 16-aligned packing (est_rows = CallDesc::rows +
// rows_bucket, CallPlan::est_rows): x0 (numpy noise), cond (prompt mel), and mu copied from the encoder's
// contiguous packing (fr_off = 2 x tok_off). The round_up(fr_suffix, 16) - fr_suffix padding frames of a row are
// zero-filled (the estimator's conv tiles read them; causal convs and the attention mask keep them out of the
// row's valid frames). spks per row (padding rows: 0).
__global__ void est_inputs_kernel(const CallDesc* __restrict__ call, float* __restrict__ x0, float* __restrict__ cond,
                                  float* __restrict__ spks, const float* __restrict__ mu_enc, float* __restrict__ mu_est) {
  const int32_t r = blockIdx.x;
  const RowDesc& d = call->rows[r];                       // encoder packing (mu source)
  const RowDesc& e = call->rows[call->rows_bucket + r];   // estimator packing
  const VoiceView& vv = call->voices;
  const bool real = r < call->n_rows && d.slot >= 0 && !(d.flags & kRowCancelled);
  if (blockIdx.y == 0)
    for (int32_t c = threadIdx.x; c < kMel; c += blockDim.x)
      spks[size_t(r) * kMel + size_t(c)] = real ? vv.spk_cond[size_t(d.voice) * kMel + size_t(c)] : 0.0f;
  if (!real) return;
  const StateView& st = call->state;
  const int32_t span = (d.fr_suffix + kEstRowAlign - 1) / kEstRowAlign * kEstRowAlign;
  const int32_t n = span * kMel;
  for (int32_t idx = blockIdx.y * blockDim.x + threadIdx.x; idx < n; idx += gridDim.y * blockDim.x) {
    const int32_t i = idx / kMel;
    const int32_t c = idx - i * kMel;
    const size_t o = (size_t(e.fr_off) + size_t(i)) * kMel + size_t(c);
    if (i >= d.fr_suffix) {
      x0[o] = 0.0f;
      cond[o] = 0.0f;
      mu_est[o] = 0.0f;
      continue;
    }
    const int32_t f = d.fr_past + i;
    float v, cv = 0.0f;
    if (f < d.prompt_frames) {
      v = vv.prompt_noise[(size_t(d.voice) * kMel + size_t(c)) * size_t(vv.max_prompt_frames) + size_t(f)];
      cv = vv.prompt_feat[(size_t(d.voice) * size_t(vv.max_prompt_frames) + size_t(f)) * kMel + size_t(c)];
    } else {
      const int32_t t = f - d.prompt_frames;
      v = t < st.noise_cap ? st.target_noise[(size_t(d.slot) * kMel + size_t(c)) * size_t(st.noise_cap) + size_t(t)] : 0.0f;
    }
    x0[o] = v;
    cond[o] = cv;
    mu_est[o] = mu_enc[(size_t(d.fr_off) + size_t(i)) * kMel + size_t(c)];
  }
}

}  // namespace

struct FlowProgram::Impl {
  const enc::Weights* enc_w;
  const est::Estimator* est;
  FlowProgramConfig cfg;
  std::mutex mu;
  // Op lists per (workspace base, rows bucket, suffix bucket): OpDesc pointers are baked at build time.
  std::map<std::tuple<const void*, int32_t, int32_t>, std::vector<OpDesc>> ops;

  struct Layout {
    size_t enc, x0, cond, mu, spks, est, total;
    int32_t frames_cap;
  };
  // The estimator works on frames_cap (a multiple of 64) packed frames; the encoder runs at a suffix
  // bucket whose 2 x rows x suffix frames equal frames_cap when possible, so every frame the estimator
  // reads has an encoder row behind it.
  static int32_t enc_suffix_bucket(int32_t rb, int32_t sb) {
    const int32_t cap = int32_t(align_up(size_t(2 * rb * sb), 64));
    return (cap % (2 * rb) == 0) ? cap / (2 * rb) : sb;
  }
  Layout layout(Phase, int32_t rb, int32_t sb, int32_t keys) const {
    Layout L{};
    const int32_t frames = 2 * rb * sb;
    L.frames_cap = int32_t(align_up(size_t(frames), 64));
    size_t off = 0;
    L.enc = off;
    off = align_up(off + enc::workspace_layout(rb, enc_suffix_bucket(rb, sb)).bytes() + size_t(64) * kMel * sizeof(float),
                   256);
    L.x0 = off;
    off = align_up(off + size_t(L.frames_cap) * kMel * sizeof(float), 256);
    L.cond = off;
    off = align_up(off + size_t(L.frames_cap) * kMel * sizeof(float), 256);
    L.mu = off;  // mu at the estimator's 16-aligned packing
    off = align_up(off + size_t(L.frames_cap) * kMel * sizeof(float), 256);
    L.spks = off;
    off = align_up(off + size_t(rb) * kMel * sizeof(float), 256);
    L.est = off;
    off = align_up(off + est->workspace_bytes(rb, L.frames_cap, std::max(keys, 64)), 256);
    L.total = off;
    return L;
  }
};

namespace {
uint64_t flow_ws_bytes(void* ctx, const CallDesc& h) {
  auto* m = static_cast<FlowProgram::Impl*>(ctx);
  return m->layout(Phase(h.phase), h.rows_bucket, h.suffix_bucket, h.reserved[0]).total;
}

void flow_enqueue(void* ctx, const CallDesc& h, const CallDesc* d, uint64_t ws_off, CUstream_st* s) {
  auto* m = static_cast<FlowProgram::Impl*>(ctx);
  const Phase phase = Phase(h.phase);
  const int32_t rb = h.rows_bucket, keys = h.reserved[0];
  const auto L = m->layout(phase, rb, h.suffix_bucket, keys);
  const int32_t sb = FlowProgram::Impl::enc_suffix_bucket(rb, h.suffix_bucket);
  char* base = static_cast<char*>(h.workspace) + ws_off;
  // 1. encoder
  const std::vector<OpDesc>* ops;
  {
    std::lock_guard<std::mutex> g(m->mu);
    auto key = std::make_tuple(static_cast<const void*>(base + L.enc), rb, sb);
    auto it = m->ops.find(key);
    if (it == m->ops.end())
      it = m->ops.emplace(key, enc::build_ops(*m->enc_w, base + L.enc, rb, sb, m->cfg.enc_conv_math)).first;
    ops = &it->second;
  }
  const int e = enc::launch_ops(*ops, d, s, m->cfg.pdl);
  if (e != 0) throw std::runtime_error(std::string("encoder launch: ") + cudaGetErrorString(cudaError_t(e)));
  if (std::getenv("KT2N_FLOW_ENC_ONLY") != nullptr) {  // debug: publish the encoder's mu as the "mel"
    const enc::WorkspaceLayout EL0 = enc::workspace_layout(rb, sb);
    cudaMemcpyAsync(h.mel_out, reinterpret_cast<const float*>(base + L.enc) + EL0.mu,
                    size_t(2 * rb * sb) * kMel * sizeof(float), cudaMemcpyDeviceToDevice, s);
    return;
  }
  // 2. estimator inputs
  float* x0 = reinterpret_cast<float*>(base + L.x0);
  float* cond = reinterpret_cast<float*>(base + L.cond);
  float* spks = reinterpret_cast<float*>(base + L.spks);
  const enc::WorkspaceLayout EL = enc::workspace_layout(rb, sb);
  float* mu_est = reinterpret_cast<float*>(base + L.mu);
  est_inputs_kernel<<<dim3(unsigned(rb), 8), 256, 0, s>>>(d, x0, cond, spks,
                                                         reinterpret_cast<const float*>(base + L.enc) + EL.mu, mu_est);
  // 3. estimator (rows = the estimator's 16-aligned table, right after the encoder's)
  est::EstCall c{};
  c.n_rows = rb;
  c.rows_cap = rb;
  c.frames_cap = L.frames_cap;
  c.max_keys = std::max(keys, 64);
  c.flow_steps = phase == Phase::kPreview ? 1 : kFlowSteps;
  c.rows = h.rows + rb;
  c.page_table = h.page_table;
  c.pool = h.pool;
  c.state = h.state;
  c.x0 = x0;
  c.mu = mu_est;
  c.cond = cond;
  c.spks = spks;
  c.mel_out = h.mel_out;
  c.workspace = base + L.est;
  // History (10:00-10:45): with L5's attention v1, a call after a call of another bucket on the same stream read
  // the previous call's data as split-combine counters (NaN in steady commits), fixed here by clearing the
  // estimator's region per call. L5's attention v2 (82a594f, on-chip merge) no longer needs it: the same oracle
  // sequence without the clear is identical on every call. The clear stays as a
  // switch (KT2N_FLOW_EST_MEMSET=1); the region is 2.6-22 MB at B1.
  static const bool est_memset = std::getenv("KT2N_FLOW_EST_MEMSET") != nullptr;
  if (est_memset) cudaMemsetAsync(base + L.est, 0, m->est->workspace_bytes(rb, L.frames_cap, std::max(keys, 64)), s);
  m->est->run(c, s, m->cfg.pdl);
}
}  // namespace

FlowProgram::FlowProgram(const enc::Weights& enc_w, const est::Estimator& est, const FlowProgramConfig& cfg)
    : impl_(new Impl) {
  impl_->enc_w = &enc_w;
  impl_->est = &est;
  impl_->cfg = cfg;
}

FlowProgram::~FlowProgram() = default;

RenderStage FlowProgram::stage() {
  RenderStage s;
  s.name = "l7.flow(l4 encoder + l5 estimator)";
  s.ctx = impl_.get();
  s.workspace_bytes = &flow_ws_bytes;
  s.enqueue = &flow_enqueue;
  return s;
}

}  // namespace kt2n::render
