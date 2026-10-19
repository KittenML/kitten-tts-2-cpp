// Native MeanFlow estimator: loader, step-time precompute and the per-call launch sequence.
// See include/kt2n/render/estimator.h.
#include "kt2n/render/estimator.h"
#include "kt2n/cuda_caps.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "est_kernels.cuh"
#include "json.h"
#include "kt2n/bf16.h"

namespace kt2n::render::est {
namespace {

#define EST_CUDA(x)                                                                                   \
  do {                                                                                                \
    cudaError_t err__ = (x);                                                                          \
    if (err__ != cudaSuccess)                                                                         \
      throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(err__) + " at " #x); \
  } while (0)

constexpr int kConvCin[kNumConvs] = {320, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                                     256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 256, 256, 256};

int conv_offset(int conv) {
  int off = 0;
  for (int i = 0; i < conv; ++i) off += 2 * kConvCin[i];
  return off;
}
constexpr int kTailWidth = 16512;

struct Tensor {
  std::vector<int64_t> shape;
  const uint16_t* data;  // bf16 bits in the mapped file
  size_t numel;
};

struct ResnetW {
  bf16 *c1w, *c1b, *ln1g, *ln1b, *c2w, *c2b, *ln2g, *ln2b, *rw, *rb;
  int cin;
};
struct TfW {
  bf16 *qkv, *ow, *ob, *n1g, *n1b, *n3g, *n3b, *f1w, *f1b, *f2w, *f2b;
};

float bf(uint16_t b) { return bf16_to_f32(b); }
float rb(float x) { return round_through_bf16(x); }

}  // namespace

struct Estimator::Impl {
  bool cluster_conv = false;
  bool supports_pdl = false;
  void* arena = nullptr;
  size_t arena_bytes = 0;
  ResnetW res[kNumResnets];
  TfW tf[kNumAttn];
  bf16 *downw, *downb, *upw, *upb, *fw, *fb, *flg, *flb, *pw, *pb;
  bf16* tproj = nullptr;  // [kStepTimes][14][256]
  std::vector<float> temb_host[kStepTimes];
};

Estimator::Estimator() : impl_(new Impl) {}
Estimator::~Estimator() {
  if (impl_ && impl_->arena) cudaFree(impl_->arena);
}
int32_t Estimator::tail_width() { return kTailWidth; }

bool Estimator::debug_trace(void* buf, uint32_t cap) {
#if defined(KT2N_EST_TRACE)
  EstTraceRec* p = static_cast<EstTraceRec*>(buf);
  const unsigned zero = 0, c = buf ? cap : 0;
  EST_CUDA(cudaMemcpyToSymbol(g_est_trace, &p, sizeof(p)));
  EST_CUDA(cudaMemcpyToSymbol(g_est_trace_cap, &c, sizeof(c)));
  EST_CUDA(cudaMemcpyToSymbol(g_est_trace_n, &zero, sizeof(zero)));
  return true;
#else
  (void)buf;
  (void)cap;
  return false;
#endif
}
uint32_t Estimator::debug_trace_count() {
#if defined(KT2N_EST_TRACE)
  unsigned n = 0;
  EST_CUDA(cudaMemcpyFromSymbol(&n, g_est_trace_n, sizeof(n)));
  return n;
#else
  return 0;
#endif
}
size_t Estimator::weight_bytes() const { return impl_->arena_bytes; }

namespace {
// LN-prologue GEMMs only (QKV, FF1): tile shapes that exist for the large-bucket wave choice (r18).
template <int BM, int BN, int WM, int BK, int ST>
void set_ln_attrs(int big) {
  EST_CUDA(cudaFuncSetAttribute(est_gemm_kernel<BM, BN, WM, BK, ST, GM_QKV, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_gemm_kernel<BM, BN, WM, BK, ST, GM_FF1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
}
template <int BM, int BN, int WM, int BK, int ST>
void set_gemm_attrs(int big) {
  EST_CUDA(cudaFuncSetAttribute(est_gemm_kernel<BM, BN, WM, BK, ST, GM_QKV, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_gemm_kernel<BM, BN, WM, BK, ST, GM_OUT, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_gemm_kernel<BM, BN, WM, BK, ST, GM_FF1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_gemm_kernel<BM, BN, WM, BK, ST, GM_FF2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_gemm_kernel<BM, BN, WM, BK, ST, GM_FINAL, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
}
}  // namespace

std::unique_ptr<Estimator> Estimator::load(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  fstat(fd, &st);
  const size_t size = size_t(st.st_size);
  void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) throw std::runtime_error("mmap failed");
  const auto* base = static_cast<const uint8_t*>(map);
  if (size < 64 || std::memcmp(base, "KT2NS3G1", 8) != 0) throw std::runtime_error("not an s3gen.kt2n file");
  uint64_t meta_off, meta_len, data_off, file_len;
  std::memcpy(&meta_off, base + 16, 8);
  std::memcpy(&meta_len, base + 24, 8);
  std::memcpy(&data_off, base + 32, 8);
  std::memcpy(&file_len, base + 48, 8);
  if (file_len != size || meta_off != 64) throw std::runtime_error("s3gen.kt2n header mismatch");
  const json::Value meta = json::parse(std::string_view(reinterpret_cast<const char*>(base + meta_off), meta_len));
  const json::Value* cfg = meta.at_pointer("/config/estimator");
  if (!cfg || cfg->get("num_mid_blocks")->as_int() != kEstMidBlocks || cfg->get("num_heads")->as_int() != kEstHeads ||
      cfg->get("attention_head_dim")->as_int() != kEstHeadDim || cfg->get("n_blocks")->as_int() != kEstTfPerBlock)
    throw std::runtime_error("s3gen.kt2n estimator geometry differs from render/ops.h");
  std::map<std::string, Tensor> tensors;
  for (const auto& t : meta.get("tensors")->items()) {
    if (t.get("group")->as_string() != "estimator") continue;
    if (t.get("dtype")->as_string() != "bf16") throw std::runtime_error("estimator tensor not bf16");
    const json::Value* d = t.get("data");
    Tensor x;
    x.numel = 1;
    for (const auto& s : d->get("shape")->items()) { x.shape.push_back(s.as_int()); x.numel *= size_t(s.as_int()); }
    const uint64_t off = uint64_t(d->get("off")->as_int()), len = uint64_t(d->get("len")->as_int());
    if (len != x.numel * 2 || data_off + off + len > size) throw std::runtime_error("bad estimator segment");
    x.data = reinterpret_cast<const uint16_t*>(base + data_off + off);
    tensors[t.get("name")->as_string()] = x;
  }
  auto get = [&](const std::string& name, std::vector<int64_t> shape) -> const Tensor& {
    auto it = tensors.find("flow.decoder.estimator." + name);
    if (it == tensors.end()) throw std::runtime_error("missing estimator tensor " + name);
    if (it->second.shape != shape) throw std::runtime_error("shape mismatch for " + name);
    return it->second;
  };

  std::unique_ptr<Estimator> est(new Estimator());
  Impl& I = *est->impl_;
  const auto caps = kt2n::cuda::current_caps();
  I.cluster_conv = caps.clusters;
  I.supports_pdl = caps.pdl();
  std::vector<uint16_t> host;  // staged arena
  host.reserve(80u << 20);
  std::vector<std::pair<size_t, bf16**>> fixups;
  auto put = [&](const uint16_t* src, size_t n, bf16** dst) {
    while (host.size() % 64) host.push_back(0);  // 128-byte alignment
    fixups.push_back({host.size(), dst});
    host.insert(host.end(), src, src + n);
  };
  auto put_vec = [&](const std::vector<uint16_t>& v, bf16** dst) { put(v.data(), v.size(), dst); };
  // conv weight [256][cin][3] -> [256][3][cin]
  auto conv_w = [&](const Tensor& t, int cin) {
    std::vector<uint16_t> o(size_t(256) * 3 * cin);
    for (int n = 0; n < 256; ++n)
      for (int c = 0; c < cin; ++c)
        for (int j = 0; j < 3; ++j) o[(size_t(n) * 3 + j) * cin + c] = t.data[(size_t(n) * cin + c) * 3 + j];
    return o;
  };
  auto res_prefix = [](int r) -> std::string {
    if (r == 0) return "down_blocks.0.0.";
    if (r == kNumResnets - 1) return "up_blocks.0.0.";
    return "mid_blocks." + std::to_string(r - 1) + ".0.";
  };
  auto tf_prefix = [](int l) -> std::string {
    if (l < 4) return "down_blocks.0.1." + std::to_string(l) + ".";
    if (l >= 52) return "up_blocks.0.1." + std::to_string(l - 52) + ".";
    return "mid_blocks." + std::to_string((l - 4) / 4) + ".1." + std::to_string((l - 4) % 4) + ".";
  };
  for (int r = 0; r < kNumResnets; ++r) {
    const int cin = (r == 0) ? 320 : (r == kNumResnets - 1 ? 512 : 256);
    const std::string p = res_prefix(r);
    ResnetW& R = I.res[r];
    R.cin = cin;
    put_vec(conv_w(get(p + "block1.block.0.weight", {256, cin, 3}), cin), &R.c1w);
    put(get(p + "block1.block.0.bias", {256}).data, 256, &R.c1b);
    put(get(p + "block1.block.2.weight", {256}).data, 256, &R.ln1g);
    put(get(p + "block1.block.2.bias", {256}).data, 256, &R.ln1b);
    put_vec(conv_w(get(p + "block2.block.0.weight", {256, 256, 3}), 256), &R.c2w);
    put(get(p + "block2.block.0.bias", {256}).data, 256, &R.c2b);
    put(get(p + "block2.block.2.weight", {256}).data, 256, &R.ln2g);
    put(get(p + "block2.block.2.bias", {256}).data, 256, &R.ln2b);
    put(get(p + "res_conv.weight", {256, cin, 1}).data, size_t(256) * cin, &R.rw);
    put(get(p + "res_conv.bias", {256}).data, 256, &R.rb);
  }
  for (int l = 0; l < kNumAttn; ++l) {
    const std::string p = tf_prefix(l);
    TfW& T = I.tf[l];
    std::vector<uint16_t> qkv;
    for (const char* part : {"attn1.to_k.weight", "attn1.to_v.weight", "attn1.to_q.weight"}) {
      const Tensor& t = get(p + part, {512, 256});
      qkv.insert(qkv.end(), t.data, t.data + t.numel);
    }
    put_vec(qkv, &T.qkv);
    put(get(p + "attn1.to_out.0.weight", {256, 512}).data, 256 * 512, &T.ow);
    put(get(p + "attn1.to_out.0.bias", {256}).data, 256, &T.ob);
    put(get(p + "norm1.weight", {256}).data, 256, &T.n1g);
    put(get(p + "norm1.bias", {256}).data, 256, &T.n1b);
    put(get(p + "norm3.weight", {256}).data, 256, &T.n3g);
    put(get(p + "norm3.bias", {256}).data, 256, &T.n3b);
    put(get(p + "ff.net.0.proj.weight", {1024, 256}).data, 1024 * 256, &T.f1w);
    put(get(p + "ff.net.0.proj.bias", {1024}).data, 1024, &T.f1b);
    put(get(p + "ff.net.2.weight", {256, 1024}).data, 256 * 1024, &T.f2w);
    put(get(p + "ff.net.2.bias", {256}).data, 256, &T.f2b);
  }
  put_vec(conv_w(get("down_blocks.0.2.weight", {256, 256, 3}), 256), &I.downw);
  put(get("down_blocks.0.2.bias", {256}).data, 256, &I.downb);
  put_vec(conv_w(get("up_blocks.0.2.weight", {256, 256, 3}), 256), &I.upw);
  put(get("up_blocks.0.2.bias", {256}).data, 256, &I.upb);
  put_vec(conv_w(get("final_block.block.0.weight", {256, 256, 3}), 256), &I.fw);
  put(get("final_block.block.0.bias", {256}).data, 256, &I.fb);
  put(get("final_block.block.2.weight", {256}).data, 256, &I.flg);
  put(get("final_block.block.2.bias", {256}).data, 256, &I.flb);
  {
    // final_proj padded to 128 output rows (zeros) for the tensor-core GEMM (GM_FINAL uses n < 80).
    std::vector<uint16_t> pw(size_t(128) * 256, 0), pb(128, 0);
    const Tensor& w = get("final_proj.weight", {80, 256, 1});
    const Tensor& b = get("final_proj.bias", {80});
    std::copy(w.data, w.data + 80 * 256, pw.begin());
    std::copy(b.data, b.data + 80, pb.begin());
    put_vec(pw, &I.pw);
    put_vec(pb, &I.pb);
  }

  // Step-time embeddings (host, fp32 math with bf16 rounding at production's module outputs):
  // SinusoidalPosEmb(320) of t and r -> time_mlp (linear_1, SiLU, linear_2) -> time_embed_mixer(cat(t, r))
  // -> per resnet mlp = Linear(Mish(.)). Constant per step: computed once.
  const Tensor& l1w = get("time_mlp.linear_1.weight", {1024, 320});
  const Tensor& l1b = get("time_mlp.linear_1.bias", {1024});
  const Tensor& l2w = get("time_mlp.linear_2.weight", {1024, 1024});
  const Tensor& l2b = get("time_mlp.linear_2.bias", {1024});
  const Tensor& mxw = get("time_embed_mixer.weight", {1024, 2048});
  auto linear = [](const Tensor& w, const Tensor* b, const std::vector<float>& x, int n, int k) {
    std::vector<float> y(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      double s = 0.0;  // accumulation order differs from cuBLAS; bf16-rounded output
      for (int j = 0; j < k; ++j) s += double(bf(w.data[size_t(i) * k + j])) * double(x[size_t(j)]);
      if (b) s += double(bf(b->data[i]));
      y[size_t(i)] = rb(float(s));
    }
    return y;
  };
  auto mlp_t = [&](float tval) {
    const int half = 160;
    const float step = float(std::log(10000.0) / double(half - 1));
    std::vector<float> e(320);
    const float ts = rb(1000.0f * rb(tval));
    for (int i = 0; i < half; ++i) {
      const float fr = std::exp(float(i) * -step);
      const float arg = ts * fr;
      e[size_t(i)] = rb(std::sin(arg));
      e[size_t(i + half)] = rb(std::cos(arg));
    }
    std::vector<float> h = linear(l1w, &l1b, e, 1024, 320);
    for (auto& v : h) v = rb(v / (1.0f + std::exp(-v)));
    return linear(l2w, &l2b, h, 1024, 1024);
  };
  const float times[kStepTimes][2] = {{0.0f, 0.5f}, {0.5f, 1.0f}, {0.0f, 1.0f}};
  std::vector<uint16_t> tproj(size_t(kStepTimes) * kNumResnets * 256);
  for (int s = 0; s < kStepTimes; ++s) {
    std::vector<float> tt = mlp_t(times[s][0]), rr = mlp_t(times[s][1]);
    std::vector<float> cat(tt);
    cat.insert(cat.end(), rr.begin(), rr.end());
    std::vector<float> mix = linear(mxw, nullptr, cat, 1024, 2048);
    I.temb_host[s] = mix;
    std::vector<float> mm(mix.size());
    for (size_t i = 0; i < mix.size(); ++i) mm[i] = rb(mix[i] * std::tanh(std::log1p(std::exp(mix[i]))));
    for (int r = 0; r < kNumResnets; ++r) {
      const std::string p = res_prefix(r);
      std::vector<float> pr = linear(get(p + "mlp.1.weight", {256, 1024}), &get(p + "mlp.1.bias", {256}), mm, 256, 1024);
      for (int n = 0; n < 256; ++n) tproj[(size_t(s) * kNumResnets + r) * 256 + n] = f32_to_bf16_rne(pr[size_t(n)]);
    }
  }
  put_vec(tproj, &I.tproj);
  munmap(map, size);

  I.arena_bytes = host.size() * 2;
  EST_CUDA(cudaMalloc(&I.arena, I.arena_bytes));
  EST_CUDA(cudaMemcpy(I.arena, host.data(), I.arena_bytes, cudaMemcpyHostToDevice));
  for (auto& f : fixups) *f.second = reinterpret_cast<bf16*>(static_cast<uint16_t*>(I.arena) + f.first);

  // Opt in to > 48 KB of dynamic shared memory.
  const int big = 99 * 1024;
  for (auto* k : {est_conv_kernel<0, 2, true>, est_conv_kernel<1, 2, true>,
                  est_conv_kernel<2, 2, true>, est_conv_kernel<3, 2, true>})
    EST_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_conv_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_conv_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_conv_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  EST_CUDA(cudaFuncSetAttribute(est_conv_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  for (auto* k : {est_conv_kernel<0, 3>, est_conv_kernel<1, 3>, est_conv_kernel<2, 3>, est_conv_kernel<3, 3>,
                  est_conv_kernel<0, 2>, est_conv_kernel<1, 2>, est_conv_kernel<2, 2>, est_conv_kernel<3, 2>,
                  est_conv_kernel<0, 6>, est_conv_kernel<1, 6>, est_conv_kernel<2, 6>, est_conv_kernel<3, 6>,
                  est_conv_kernel<0, 8>, est_conv_kernel<1, 8>, est_conv_kernel<2, 8>, est_conv_kernel<3, 8>})
    EST_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  set_gemm_attrs<64, 32, 4, 64, 4>(big);
  set_gemm_attrs<16, 128, 1, 64, 4>(big);
  for (auto* k : {est_gemm_kernel<16, 32, 1, 64, 8, GM_OUT, 1, 2>, est_gemm_kernel<16, 32, 1, 64, 8, GM_FF2, 1, 2>,
                  est_gemm_kernel<16, 16, 1, 64, 8, GM_OUT, 1, 1>, est_gemm_kernel<16, 16, 1, 64, 8, GM_FF2, 1, 1>,
                  est_gemm_kernel<32, 32, 2, 64, 6, GM_OUT, 1, 4>, est_gemm_kernel<32, 32, 2, 64, 6, GM_FF2, 1, 4>})
    EST_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
  set_gemm_attrs<32, 64, 2, 64, 4>(big);
  set_gemm_attrs<64, 64, 2, 64, 3>(big);
  set_gemm_attrs<64, 128, 2, 32, 3>(big);
  set_gemm_attrs<128, 128, 2, 32, 3>(big);
  set_ln_attrs<64, 256, 2, 32, 3>(big);
  set_ln_attrs<32, 128, 2, 32, 3>(big);
  set_ln_attrs<32, 256, 2, 32, 3>(big);
  set_ln_attrs<64, 192, 2, 32, 3>(big);
  set_ln_attrs<128, 192, 2, 32, 2>(big);
  EST_CUDA(cudaFuncSetAttribute(est_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(attn_smem_bytes())));
  EST_CUDA(cudaFuncSetAttribute(est_attn2_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(attn2_smem_bytes<1>())));
  EST_CUDA(cudaFuncSetAttribute(est_attn2_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(attn2_smem_bytes<2>())));
  EST_CUDA(cudaFuncSetAttribute(est_attn2_kernel<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(attn2_smem_bytes<1, 2>())));
  EST_CUDA(cudaFuncSetAttribute(est_attn2_kernel<2, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(attn2_smem_bytes<2, 1, true>())));
  EST_CUDA(cudaFuncSetAttribute(est_attn2_kernel<3, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(attn2_smem_bytes<3, 1, true>())));
  return est;
}

// ---------------------------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------------------------
namespace {
// Keys per attention split: fixed per process (KT2N_EST_ATTN_SPLIT, multiple of 64, default kAttnSplit) so
// rows stay batch invariant; the knob exists for the split-size measurement only.
int attn_split() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_ATTN_SPLIT");
    int x = e ? std::atoi(e) : kAttnSplit;
    if (x < 64 || x % 64) x = kAttnSplit;
    return x;
  }();
  return v;
}
// r15: PDL trigger right after the dependency wait ("wait -> trigger -> work", the probe's fastest pattern) so
// the next kernel's pre-wait prologue overlaps this kernel's work. KT2N_EST_TRIG=late restores the r12 placement
// (after the main loop) for the A/B. Scheduling only: the arithmetic is the same.
int trig_early() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_TRIG");
    return (e && std::string(e) == "late") ? 0 : 1;
  }();
  return v;
}
// KT2N_EST_ATTN=v1 selects the split-K pool attention with global partials (kept for the A/B).
bool attn_v1() {
  static const bool v = [] {
    const char* e = std::getenv("KT2N_EST_ATTN");
    return e && std::string(e) == "v1";
  }();
  return v;
}
// Queries per attention CTA above 512 frames: 32 (default; r9: B20 attention 20.7 -> 16.8 us, B32 31.0 -> 27.2 us,
// final mel bit-identical) or 16 (KT2N_EST_ATTN_QG=16, for the A/B).
int attn_qg() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_ATTN_QG");
    return (e && std::atoi(e) == 16) ? 16 : 32;
  }();
  return v;
}
// Attention at frames_cap <= 512 (KT2N_EST_ATTN_SMALL for the A/B): "2" = 2 stages per warp, padded rows (r6-r16),
// "2s" = 2 stages, swizzled rows (r17 default), "3s" = 3 stages, swizzled rows (two K/V tiles in flight per warp).
// r17, all bit-identical, wall p50 2 / 2s / 3s: B1 1.99 / 1.95 / 1.98 ms, B2 2.28 / 2.05 / 2.06, B4 (80 fr)
// 3.48-3.53 / 3.17-3.37 / 3.36. The third stage cut the traced loop only 4.16 -> 3.74 us and lost on the wall
// clock: the per-tile time is the warp's own mma / softmax dependency chain (one warp per scheduler), not loads.
int attn_small() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_ATTN_SMALL");
    if (e && std::string(e) == "2") return 2;
    if (e && std::string(e) == "3s") return 3;
    return 1;
  }();
  return v;
}
struct Ws {
  EstRow* erows;
  int32_t* frame_row;
  AttnItem* items;
  int32_t* n_items;
  bf16 *xin, *b0, *b1, *b2, *q, *ao, *ff;
  float *part_o, *part_ml;
  int32_t* counters;
  bf16* stage;
  size_t bytes;
  int items_cap, nsplit;
};
Ws carve(void* base, int rows_cap, int frames_cap, int max_keys) {
  Ws w{};
  size_t off = 0;
  auto take = [&](size_t bytes) {
    off = (off + 255) / 256 * 256;
    void* p = static_cast<char*>(base) + off;
    off += bytes;
    return p;
  };
  w.items_cap = frames_cap / 64 + rows_cap;
  w.nsplit = std::max(1, (max_keys + attn_split() - 1) / attn_split());
  w.erows = static_cast<EstRow*>(take(sizeof(EstRow) * size_t(rows_cap)));
  w.frame_row = static_cast<int32_t*>(take(4 * size_t(frames_cap)));
  w.items = static_cast<AttnItem*>(take(sizeof(AttnItem) * size_t(w.items_cap)));
  w.n_items = static_cast<int32_t*>(take(4));
  w.xin = static_cast<bf16*>(take(2 * size_t(frames_cap) * kIn));
  w.b0 = static_cast<bf16*>(take(2 * size_t(frames_cap) * kHid));
  w.b1 = static_cast<bf16*>(take(2 * size_t(frames_cap) * kHid));
  w.b2 = static_cast<bf16*>(take(2 * size_t(frames_cap) * kHid));
  w.q = static_cast<bf16*>(take(2 * size_t(frames_cap) * kInner));
  w.ao = static_cast<bf16*>(take(2 * size_t(frames_cap) * kInner));
  w.ff = static_cast<bf16*>(take(2 * size_t(frames_cap) * kFfn));
  w.part_o = static_cast<float*>(take(4 * size_t(w.items_cap) * 8 * w.nsplit * 4096));
  w.part_ml = static_cast<float*>(take(4 * size_t(w.items_cap) * 8 * w.nsplit * 128));
  w.counters = static_cast<int32_t*>(take(4 * size_t(w.items_cap) * 8));
  w.stage = static_cast<bf16*>(take(2 * size_t(kFlowSteps) * rows_cap * kTailWidth));
  w.bytes = off;
  return w;
}

template <typename... KArgs, typename... Args>
void launch_c(bool pdl, int cluster_y, void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st,
              Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  unsigned n = 0;
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (cluster_y > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = 1;
    attr[n].val.clusterDim.y = unsigned(cluster_y);
    attr[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = n ? attr : nullptr;
  cfg.numAttrs = n;
  EST_CUDA(cudaLaunchKernelEx(&cfg, kern, args...));
}
template <typename... KArgs, typename... Args>
void launch(bool pdl, void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args... args) {
  launch_c(pdl, 1, kern, grid, block, smem, st, args...);
}
// Cluster of `cz` CTAs along z (split-K GEMMs).
template <typename KArg>
void launch_k(bool pdl, void (*kern)(KArg), dim3 grid, dim3 block, size_t smem, cudaStream_t st, const KArg& arg, int cz) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  unsigned n = 0;
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  attr[n].id = cudaLaunchAttributeClusterDimension;
  attr[n].val.clusterDim.x = 1;
  attr[n].val.clusterDim.y = 1;
  attr[n].val.clusterDim.z = unsigned(cz);
  ++n;
  cfg.attrs = attr;
  cfg.numAttrs = n;
  EST_CUDA(cudaLaunchKernelEx(&cfg, kern, arg));
}
}  // namespace

size_t Estimator::workspace_bytes(int32_t rows_cap, int32_t frames_cap, int32_t max_keys) const {
  return carve(nullptr, rows_cap, frames_cap, max_keys).bytes + 256;
}

int32_t Estimator::kernels_per_call(int32_t flow_steps) const {
  // prep + assemble + per step (14 resnets x 2 + 56 x 5 + down + up + final block + final proj) + tail commit
  const int epilogues = impl_->cluster_conv ? 0 : kNumResnets * 2 + 1;
  return 2 + flow_steps * (kNumResnets * 2 + kNumAttn * 5 + 4 + epilogues) + 1;
}

namespace {
template <int BM, int BN, int WM, int BK, int ST>
void gemm_ln(bool pdl, cudaStream_t st, int mode, const GemmArgs& g, int frames_cap) {
  const dim3 grid(unsigned(g.n / BN), unsigned(frames_cap / BM));
  if (mode == GM_QKV)
    launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_QKV, 1>, grid, dim3(256), gemm_smem_bytes<BM, BN, BK, ST, GM_QKV>(), st, g);
  else
    launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_FF1, 1>, grid, dim3(256), gemm_smem_bytes<BM, BN, BK, ST, GM_FF1>(), st, g);
}
template <int BM, int BN, int WM, int BK, int ST>
void gemm_all(bool pdl, cudaStream_t st, int mode, const GemmArgs& g, int frames_cap) {
  const dim3 grid(unsigned(g.n / BN), unsigned(frames_cap / BM));
  const dim3 blk(256);
  switch (mode) {
    case GM_QKV: launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_QKV, 1>, grid, blk, gemm_smem_bytes<BM, BN, BK, ST, GM_QKV>(), st, g); break;
    // Split-K over a (1, 1, KS) cluster (template KS) measured slower at every bucket (DSMEM partial exchange):
    // B1 FF2 21.7 us vs 6.2 us, B20 131 us. Kept in the kernel for the record; KS = 1 here.
    case GM_OUT: launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_OUT, 1>, grid, blk, gemm_smem_bytes<BM, BN, BK, ST, GM_OUT>(), st, g); break;
    case GM_FF1: launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_FF1, 1>, grid, blk, gemm_smem_bytes<BM, BN, BK, ST, GM_FF1>(), st, g); break;
    case GM_FF2: launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_FF2, 1>, grid, blk, gemm_smem_bytes<BM, BN, BK, ST, GM_FF2>(), st, g); break;
    default: launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_FINAL, 1>, grid, blk, gemm_smem_bytes<BM, BN, BK, ST, GM_FINAL>(), st, g); break;
  }
}
// Tile shape by bucket: small buckets spread the weights over more CTAs (narrow N tiles, whole M);
// the per-element k order is the same in every shape (batch invariance).
// N = 256 GEMMs with a streamed A (to_out K = 512, FF2 K = 1024) at small buckets: 64 x 32 tiles give only
// 8 CTAs at B1, each streaming 192 KB (FF2; r15 trace: 3.8 us main loop). r16 variants with 16-row tiles and
// 1-2 warps per CTA (each warp still one 16 x 16 tile, k ascending, so the bits do not change), deeper
// pipelines because few warps cannot hide latency: KT2N_EST_OUTTILE=64x32 | 16x32 | 16x16 | 32x32.
int out_tile() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_OUTTILE");
    if (e && std::string(e) == "64x32") return 0;
    if (e && std::string(e) == "16x16") return 2;
    if (e && std::string(e) == "32x32") return 3;
    return 1;  // 16x32
  }();
  return v;
}
template <int BM, int BN, int WM, int BK, int ST, int NWP>
void gemm_res(bool pdl, cudaStream_t st, int mode, const GemmArgs& g, int frames_cap) {
  const dim3 grid(unsigned(g.n / BN), unsigned(frames_cap / BM));
  const dim3 blk(32 * NWP);
  if (mode == GM_OUT)
    launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_OUT, 1, NWP>, grid, blk, gemm_smem_bytes<BM, BN, BK, ST, GM_OUT>(), st, g);
  else
    launch(pdl, est_gemm_kernel<BM, BN, WM, BK, ST, GM_FF2, 1, NWP>, grid, blk, gemm_smem_bytes<BM, BN, BK, ST, GM_FF2>(), st, g);
}
// LN-prologue GEMMs (QKV, FF1) at small buckets: every CTA normalises all BM rows of its M tile, and the
// r13 trace showed that LayerNorm (issue-bound, 64 rows per CTA) is most of their time. A 16-row M tile
// cuts it 4x; each warp keeps the same 16 x 16 output tile and k order, so the bits do not change.
// KT2N_EST_LNTILE=64x32 | 32x64 | 16x128 (default 16x128 since r13) for the A/B.
int ln_tile() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_LNTILE");
    if (e && std::string(e) == "64x32") return 64;
    if (e && std::string(e) == "32x64") return 32;
    return 16;
  }();
  return v;
}
// Frame bucket up to which the small-bucket tiles above are used (KT2N_EST_SMALL_FR for the A/B). r16b: 320
// (B2 at 192 frames 3.45 -> 2.75 ms, B4 3.47 / 3.79 -> 3.10 / 3.54 ms, bit-identical); 640 was slower at B10
// (7.45 vs 5.27 ms: 16-row tiles give too many small CTAs once there are enough rows to fill the GPU).
int small_fr() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_SMALL_FR");
    return e ? std::atoi(e) : 320;
  }();
  return v;
}
// LN-prologue GEMMs above the small buckets (r18/r19): with ~61-98 KB of shared memory per CTA only 1 (or 2)
// CTAs fit per SM, so the CTA count against the SM count decides the waves (r17b B32: QKV on 128 x 128 is
// 192-240 CTAs = 2 waves, 19.8 us, vs FF1 9.7 us for 2/3 of the work). KT2N_EST_LNTILE_L forces one shape
// for the A/B: 64x64 | 64x128 | 128x128 | 64x256 | 32x128 | 32x256 | 64x192 | 128x192 (unset = the wave
// cost below; a forced shape that does not divide the problem falls back to the r17 rule). Every shape keeps
// each output element's k order, so the bits never change (r18 / r19 cmp).
// Conv weight pipeline stages above the small buckets (KT2N_EST_CONV_ST_L = 4 | 3 | 2 | 32 for the A/B, where
// 32 = 3 stages for the res convs (block2) and 2 for the rest). Fewer stages = less shared memory = more CTAs
// per SM. r19 (nsys min, bit-identical): 3 stages let block2 fit 2 CTAs per SM, B20 22.6 -> 14.2 us, B32
// 33.0 -> 26.0 us, the other convs equal or slightly faster; 2 stages slowed block1 at B10 (7.6 -> 9.2 us).
int conv_st_large() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_CONV_ST_L");
    return e ? std::atoi(e) : 3;
  }();
  return v;
}
// Conv stages at small buckets (KT2N_EST_CONV_ST_S = 4 | 6 | 8 | 0 for the A/B; 0 = the r22 default: 6 stages
// for block1 / down-up / final convs, 4 for block2). r22 nsys min at B1, 4 -> 6 stages (bit-identical): block1
// 6.66 -> 6.18 us, down/up 5.44 -> 4.74, final 6.78 -> 6.30, block2 11.13 -> 11.26 (no gain); 8 stages slower
// everywhere. So the B1 conv is not weight-latency bound either: each warp's short mma chain is.
int conv_st_small() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_CONV_ST_S");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}
int ln_tile_large() {
  static const int v = [] {
    const char* e = std::getenv("KT2N_EST_LNTILE_L");
    if (!e) return -1;
    const std::string t(e);
    const char* names[] = {"64x64", "64x128", "128x128", "64x256", "32x128", "32x256", "64x192", "128x192"};
    for (int i = 0; i < 8; ++i)
      if (t == names[i]) return i;
    return -1;
  }();
  return v;
}
// r19 default: pick the shape by a wave cost, waves x per-CTA time, with every candidate at 1 CTA per SM. The
// per-CTA time was fitted to the r19 nsys minimums: 3.2 + 0.43 x (BM x BN / 1024) us (64x64 5.1, 64x128 7.0,
// 64x192 8.5-9.3, 64x256 / 128x128 9.4-10.7, 128x192 13.9 us measured). Picks, e.g.: QKV 64x128 at 640-832
// frames, 64x192 at 1024-1280, 64x256 at 1600, 128x192 at 2048-2560; FF1 64x64 at 640, 64x128 at 832-1280,
// 64x256 at 1600, 128x128 at 2048+.
int num_sms() {
  static const int v = [] {
    int dev = 0, n = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
      return 188;
    return n > 0 ? n : 188;
  }();
  return v;
}
int ln_tile_auto(int F, int n) {
  struct C { int id, bm, bn; };
  // ties go to the earlier entry (128 x 128 before 64 x 256: FF1 at B32 9.73 vs 9.98 us)
  static const C cands[] = {{0, 64, 64}, {1, 64, 128}, {6, 64, 192}, {2, 128, 128}, {3, 64, 256}, {7, 128, 192}};
  int best = 0;
  double best_cost = 1e30;
  for (const C& c : cands) {
    if (F % c.bm || n % c.bn) continue;
    const int ctas = (F / c.bm) * (n / c.bn);
    const int waves = (ctas + num_sms() - 1) / num_sms();
    const double cost = waves * (3.2 + 0.43 * double(c.bm) * c.bn / 1024.0);
    if (cost < best_cost - 1e-9) { best_cost = cost; best = c.id; }
  }
  return best;
}
bool gemm_ln_large(bool pdl, cudaStream_t st, int mode, const GemmArgs& g, int F) {
  int t = ln_tile_large();
  if (t < 0) t = ln_tile_auto(F, g.n);
  auto fits = [&](int bm, int bn) { return F % bm == 0 && g.n % bn == 0; };
  switch (t) {
    case 0: if (fits(64, 64)) { gemm_ln<64, 64, 2, 64, 3>(pdl, st, mode, g, F); return true; } break;
    case 1: if (fits(64, 128)) { gemm_ln<64, 128, 2, 32, 3>(pdl, st, mode, g, F); return true; } break;
    case 2: if (fits(128, 128)) { gemm_ln<128, 128, 2, 32, 3>(pdl, st, mode, g, F); return true; } break;
    case 3: if (fits(64, 256)) { gemm_ln<64, 256, 2, 32, 3>(pdl, st, mode, g, F); return true; } break;
    case 4: if (fits(32, 128)) { gemm_ln<32, 128, 2, 32, 3>(pdl, st, mode, g, F); return true; } break;
    case 5: if (fits(32, 256)) { gemm_ln<32, 256, 2, 32, 3>(pdl, st, mode, g, F); return true; } break;
    case 6: if (fits(64, 192)) { gemm_ln<64, 192, 2, 32, 3>(pdl, st, mode, g, F); return true; } break;
    case 7: if (fits(128, 192)) { gemm_ln<128, 192, 2, 32, 2>(pdl, st, mode, g, F); return true; } break;
    default: break;
  }
  return false;
}
void gemm(bool pdl, cudaStream_t st, int mode, const GemmArgs& g, int frames_cap) {
  // r10: 6 stages instead of 4 measured no gain at B1/B2 (FF2 5.89 vs 5.95 us, bits identical): the time is
  // fixed per-kernel cost (launch, prologue, epilogue), not weight tiles in flight.
  const bool ln = (mode == GM_QKV || mode == GM_FF1);
  const int SF = small_fr();
  if (frames_cap <= SF && ln && ln_tile() == 16) gemm_all<16, 128, 1, 64, 4>(pdl, st, mode, g, frames_cap);
  else if (frames_cap <= SF && ln && ln_tile() == 32) gemm_all<32, 64, 2, 64, 4>(pdl, st, mode, g, frames_cap);
  else if (frames_cap <= SF && (mode == GM_OUT || mode == GM_FF2) && out_tile() == 1) gemm_res<16, 32, 1, 64, 8, 2>(pdl, st, mode, g, frames_cap);
  else if (frames_cap <= SF && (mode == GM_OUT || mode == GM_FF2) && out_tile() == 2) gemm_res<16, 16, 1, 64, 8, 1>(pdl, st, mode, g, frames_cap);
  else if (frames_cap <= SF && (mode == GM_OUT || mode == GM_FF2) && out_tile() == 3) gemm_res<32, 32, 2, 64, 6, 4>(pdl, st, mode, g, frames_cap);
  else if (frames_cap <= 128) gemm_all<64, 32, 4, 64, 4>(pdl, st, mode, g, frames_cap);
  else if (ln && gemm_ln_large(pdl, st, mode, g, frames_cap)) return;
  // N = 256 outputs (to_out, FF2) or 128 (final): 128-wide tiles leave too few CTAs at large M (B20 FF2
  // 25.8 us with 20 CTAs); 64 x 64 keeps 4x more CTAs in flight.
  else if (g.n <= 256 && !std::getenv("KT2N_EST_GEMM_NO64N")) gemm_all<64, 64, 2, 64, 3>(pdl, st, mode, g, frames_cap);
  else if (frames_cap <= 1024) gemm_all<64, 64, 2, 64, 3>(pdl, st, mode, g, frames_cap);
  else if (frames_cap % 128 != 0 || std::getenv("KT2N_EST_GEMM_NO128")) gemm_all<64, 128, 2, 32, 3>(pdl, st, mode, g, frames_cap);
  else gemm_all<128, 128, 2, 32, 3>(pdl, st, mode, g, frames_cap);
}
}  // namespace

void Estimator::run(const EstCall& c, CUstream_st* stream_raw, bool pdl) const {
  const Impl& I = *impl_;
  pdl = pdl && I.supports_pdl;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_raw);
  if (c.frames_cap % 64) throw std::runtime_error("frames_cap must be a multiple of 64");
  Ws w = carve(c.workspace, c.rows_cap, c.frames_cap, c.max_keys);
  const int F = c.frames_cap;
  {
    const int threads = 256;
    const int blocks = std::max((F + threads - 1) / threads, (c.rows_cap + threads - 1) / threads);
    launch(pdl, est_prep_kernel, dim3(unsigned(blocks)), dim3(threads), 0, st, c.rows, c.rows_cap, F, w.erows,
           w.frame_row, w.items, w.n_items, w.items_cap);
  }
  launch(pdl, est_assemble_kernel, dim3(unsigned((F * kIn + 255) / 256)), dim3(256), 0, st,
         static_cast<const EstRow*>(w.erows), static_cast<const int32_t*>(w.frame_row), c.x0, c.mu, c.cond, c.spks,
         w.xin, F);
  const dim3 conv_grid(unsigned(F / kConvBM), unsigned(kConvCluster));
  const bool save_tails = c.state.est_tail != nullptr;
  for (int s = 0; s < c.flow_steps; ++s) {
    const int sv = (c.flow_steps == 1) ? int(StepTime::kPreview) : s;
    const int ps = (c.flow_steps == 1) ? 0 : s;  // pool / tail step slot
    bf16* X = w.b0;
    bf16* T = w.b1;
    bf16* spare = w.b2;
    bf16* skip = nullptr;
    int conv_i = 0;
    auto conv = [&](int mode, const bf16* in0, int c0, const bf16* in1, int c1, const bf16* wt, const bf16* bias,
                    const bf16* g, const bf16* b, const bf16* temb, const bf16* r0, int rc0, const bf16* r1, int rc1,
                    const bf16* rw, const bf16* rbias, bf16* out) {
      ConvArgs a{};
      a.in0 = in0; a.c0 = c0; a.ld0 = c0;
      a.in1 = in1; a.c1 = c1; a.ld1 = c1;
      a.w = wt; a.bias = bias; a.ln_g = g; a.ln_b = b; a.temb = temb;
      a.res_in0 = r0; a.res_c0 = rc0; a.res_ld0 = rc0; a.res_in1 = r1; a.res_c1 = rc1; a.res_ld1 = rc1;
      a.res_w = rw; a.res_b = rbias;
      a.out = out; a.ld_out = kHid;
      a.erows = w.erows; a.frame_row = w.frame_row;
      const int off = conv_offset(conv_i);
      a.tail_state = c.state.est_tail ? reinterpret_cast<const bf16*>(c.state.est_tail) +
                                            size_t(ps) * size_t(c.state.slots) * kTailWidth + off
                                      : nullptr;
      a.tail_slot_stride = kTailWidth;
      a.tail_stage = save_tails ? w.stage + size_t(ps) * c.rows_cap * kTailWidth + off : nullptr;
      a.stage_row_stride = kTailWidth;
      a.trig_early = trig_early();
      if (!I.cluster_conv) {
        a.portable_values = w.ff;
        a.portable_residual = w.ff + size_t(F) * kHid;
        const size_t bytes = conv_smem_bytes(c0 + c1, rc0 + rc1, 2);
        switch (mode) {
          case 0:
            launch(false, est_conv_kernel<0, 2, true>, conv_grid, dim3(kConvThreads), bytes, st, a);
            launch(false, est_conv_portable_epilogue<0>, dim3(F / kConvBM), dim3(512), 0, st, a);
            break;
          case 1:
            launch(false, est_conv_kernel<1, 2, true>, conv_grid, dim3(kConvThreads), bytes, st, a);
            launch(false, est_conv_portable_epilogue<1>, dim3(F / kConvBM), dim3(512), 0, st, a);
            break;
          case 2:
            launch(false, est_conv_kernel<2, 2, true>, conv_grid, dim3(kConvThreads), bytes, st, a);
            break;
          default:
            launch(false, est_conv_kernel<3, 2, true>, conv_grid, dim3(kConvThreads), bytes, st, a);
            launch(false, est_conv_portable_epilogue<3>, dim3(F / kConvBM), dim3(512), 0, st, a);
        }
        ++conv_i;
        return;
      }
      int cst = 4;
      if (F > small_fr()) {
        const int v = conv_st_large();
        cst = (v == 32) ? (mode == 1 ? 3 : 2) : (v == 3 || v == 2) ? v : 4;
      } else {
        const int v = conv_st_small();
        cst = (v == 6 || v == 8) ? v : (v == 0 && mode != 1) ? 6 : 4;
      }
      const size_t smem = conv_smem_bytes(c0 + c1, rc0 + rc1, cst);
      if (cst == 4) {
        switch (mode) {
          case 0: launch_c(pdl, kConvCluster, est_conv_kernel<0>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 1: launch_c(pdl, kConvCluster, est_conv_kernel<1>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 2: launch_c(pdl, kConvCluster, est_conv_kernel<2>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          default: launch_c(pdl, kConvCluster, est_conv_kernel<3>, conv_grid, dim3(kConvThreads), smem, st, a); break;
        }
      } else if (cst == 3) {
        switch (mode) {
          case 0: launch_c(pdl, kConvCluster, est_conv_kernel<0, 3>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 1: launch_c(pdl, kConvCluster, est_conv_kernel<1, 3>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 2: launch_c(pdl, kConvCluster, est_conv_kernel<2, 3>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          default: launch_c(pdl, kConvCluster, est_conv_kernel<3, 3>, conv_grid, dim3(kConvThreads), smem, st, a); break;
        }
      } else if (cst == 6) {
        switch (mode) {
          case 0: launch_c(pdl, kConvCluster, est_conv_kernel<0, 6>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 1: launch_c(pdl, kConvCluster, est_conv_kernel<1, 6>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 2: launch_c(pdl, kConvCluster, est_conv_kernel<2, 6>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          default: launch_c(pdl, kConvCluster, est_conv_kernel<3, 6>, conv_grid, dim3(kConvThreads), smem, st, a); break;
        }
      } else if (cst == 8) {
        switch (mode) {
          case 0: launch_c(pdl, kConvCluster, est_conv_kernel<0, 8>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 1: launch_c(pdl, kConvCluster, est_conv_kernel<1, 8>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 2: launch_c(pdl, kConvCluster, est_conv_kernel<2, 8>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          default: launch_c(pdl, kConvCluster, est_conv_kernel<3, 8>, conv_grid, dim3(kConvThreads), smem, st, a); break;
        }
      } else {
        switch (mode) {
          case 0: launch_c(pdl, kConvCluster, est_conv_kernel<0, 2>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 1: launch_c(pdl, kConvCluster, est_conv_kernel<1, 2>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          case 2: launch_c(pdl, kConvCluster, est_conv_kernel<2, 2>, conv_grid, dim3(kConvThreads), smem, st, a); break;
          default: launch_c(pdl, kConvCluster, est_conv_kernel<3, 2>, conv_grid, dim3(kConvThreads), smem, st, a); break;
        }
      }
      ++conv_i;
    };
    auto tproj = [&](int r) { return I.tproj + (size_t(sv) * kNumResnets + r) * 256; };
    auto resnet = [&](int r, const bf16* in0, int c0, const bf16* in1, int c1, bf16* out) {
      const ResnetW& R = I.res[r];
      conv(0, in0, c0, in1, c1, R.c1w, R.c1b, R.ln1g, R.ln1b, tproj(r), nullptr, 0, nullptr, 0, nullptr, nullptr, T);
      conv(1, T, 256, nullptr, 0, R.c2w, R.c2b, R.ln2g, R.ln2b, nullptr, in0, c0, in1, c1, R.rw, R.rb, out);
    };
    auto transformers = [&](int first_layer) {
      for (int l = first_layer; l < first_layer + 4; ++l) {
        const TfW& Tw = I.tf[l];
        GemmArgs g{};
        g.erows = w.erows; g.frame_row = w.frame_row; g.page_table = c.page_table; g.frames_cap = F;
        g.trig_early = trig_early();
        g.pool_layer = reinterpret_cast<bf16*>(c.pool.est) +
                       size_t(ps * kEstAttnLayers + l) * size_t(c.pool.n_pages) * kPageFrames * (2 * kInner);
        // QKV (LN1 prologue)
        g.a = X; g.lda = kHid; g.k = kHid; g.w = Tw.qkv; g.bias = nullptr; g.ln_g = Tw.n1g; g.ln_b = Tw.n1b;
        g.n = 3 * kInner; g.out = w.q; g.ld_out = kInner;
        gemm(pdl, st, GM_QKV, g, F);
        // attention
        AttnArgs at{};
        at.q = w.q; at.out = w.ao; at.pool_layer = g.pool_layer; at.page_table = c.page_table; at.erows = w.erows;
        at.items = w.items; at.n_items = w.n_items; at.part_o = w.part_o; at.part_ml = w.part_ml;
        at.counters = w.counters; at.nsplit_cap = w.nsplit; at.split = attn_split(); at.scale = 0.125f;
        at.trig_early = trig_early();
        if (attn_v1())
          launch(pdl, est_attn_kernel, dim3(unsigned(w.items_cap), 8u, unsigned(w.nsplit)), dim3(128),
                 attn_smem_bytes(), st, at);
        else if (F <= 512 && attn_small() == 3)  // r6/r7: 2+ stages faster up to B4 (320 fr: 6.3 vs 11.2 us), 1 stage from B10
          launch(pdl, est_attn2_kernel<3, 1, true>, dim3(unsigned(w.items_cap * 4), 8u), dim3(128), attn2_smem_bytes<3, 1, true>(), st, at);
        else if (F <= 512 && attn_small() == 1)
          launch(pdl, est_attn2_kernel<2, 1, true>, dim3(unsigned(w.items_cap * 4), 8u), dim3(128), attn2_smem_bytes<2, 1, true>(), st, at);
        else if (F <= 512)
          launch(pdl, est_attn2_kernel<2>, dim3(unsigned(w.items_cap * 4), 8u), dim3(128), attn2_smem_bytes<2>(), st, at);
        else if (attn_qg() == 32)
          launch(pdl, est_attn2_kernel<1, 2>, dim3(unsigned(w.items_cap * 2), 8u), dim3(128), attn2_smem_bytes<1, 2>(), st, at);
        else
          launch(pdl, est_attn2_kernel<1>, dim3(unsigned(w.items_cap * 4), 8u), dim3(128), attn2_smem_bytes<1>(), st, at);
        // to_out + residual
        g.a = w.ao; g.lda = kInner; g.k = kInner; g.w = Tw.ow; g.bias = Tw.ob; g.ln_g = g.ln_b = nullptr;
        g.n = kHid; g.out = X; g.ld_out = kHid;
        gemm(pdl, st, GM_OUT, g, F);
        // FF1 (LN3 prologue, GELU)
        g.a = X; g.lda = kHid; g.k = kHid; g.w = Tw.f1w; g.bias = Tw.f1b; g.ln_g = Tw.n3g; g.ln_b = Tw.n3b;
        g.n = kFfn; g.out = w.ff; g.ld_out = kFfn;
        gemm(pdl, st, GM_FF1, g, F);
        // FF2 + residual
        g.a = w.ff; g.lda = kFfn; g.k = kFfn; g.w = Tw.f2w; g.bias = Tw.f2b; g.ln_g = g.ln_b = nullptr;
        g.n = kHid; g.out = X; g.ld_out = kHid;
        gemm(pdl, st, GM_FF2, g, F);
      }
    };
    // down: resnet(xin) -> X, transformers, skip = X, down conv -> T
    resnet(0, w.xin, kIn, nullptr, 0, X);
    transformers(0);
    skip = X;
    conv(2, X, 256, nullptr, 0, I.downw, I.downb, nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, T);
    X = T;
    T = spare;
    for (int m = 0; m < kEstMidBlocks; ++m) {
      resnet(1 + m, X, 256, nullptr, 0, X);
      transformers(4 + 4 * m);
    }
    resnet(kNumResnets - 1, X, 256, skip, 256, X);
    transformers(52);
    conv(2, X, 256, nullptr, 0, I.upw, I.upb, nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, T);
    conv(3, T, 256, nullptr, 0, I.fw, I.fb, I.flg, I.flb, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, X);
    {
      GemmArgs g{};
      g.erows = w.erows; g.frame_row = w.frame_row; g.page_table = c.page_table; g.frames_cap = F;
      g.trig_early = trig_early();
      g.a = X; g.lda = kHid; g.k = kHid; g.w = I.pw; g.bias = I.pb; g.n = 128;
      g.xin = w.xin;
      g.dt = (c.flow_steps == 1) ? 1.0f : 0.5f;
      g.deriv_out = c.deriv_out[s];
      g.mel_out = (s + 1 == c.flow_steps) ? c.mel_out : nullptr;
      gemm(pdl, st, GM_FINAL, g, F);
    }
  }
  if (save_tails && c.flow_steps == kFlowSteps) {
    launch(pdl, est_tail_commit_kernel, dim3(unsigned(c.rows_cap), unsigned(c.flow_steps), unsigned((kTailWidth / 8 + 255) / 256)), dim3(256), 0, st,
           static_cast<const EstRow*>(w.erows), static_cast<const bf16*>(w.stage),
           reinterpret_cast<bf16*>(c.state.est_tail), c.rows_cap, c.state.slots, kTailWidth, c.flow_steps);
  }
}

}  // namespace kt2n::render::est
