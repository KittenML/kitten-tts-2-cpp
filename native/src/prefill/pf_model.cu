// kt2n p2-prefill: minimal weight holder.  Loads a .kt2n engine file (kt2n_core's mmap reader), copies the
// dense tensors to the GPU and repacks every projection family into k1's kernel layout.  One device arena;
// the result is a plain kt2n_pf_weights struct of device pointers (no global state).
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "kt2n/engine_file.h"
#include "kt2n/kernels/tern_proj.h"
#include "kt2n/prefill.h"
#include "pf_internal.h"

struct kt2n_pf_model {
  kt2n_pf_weights w{};
  void* arena = nullptr;
  size_t bytes = 0;
  int device = 0;
  std::string source_sha256;
  std::string path;
};

namespace kt2n_pf {

namespace {

constexpr size_t kAlign = 256;
size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

struct Plan {
  // offsets into the arena
  size_t embed = 0, final_norm = 0;
  struct L {
    size_t qkv_c, qkv_s, o_c, o_s, gu_c, gu_s, d_c, d_s, in_n, post_n, q_n, k_n;
  };
  std::vector<L> layer;
  size_t total = 0;
};

const kt2n::TensorInfo* need(const kt2n::EngineFile& f, const std::string& name, bool ternary,
                             std::vector<int64_t> shape) {
  const kt2n::TensorInfo* t = f.find(name);
  if (!t) throw PfError(KT2N_PF_EFILE, "engine file has no tensor " + name);
  if ((t->kind == kt2n::TensorKind::kTernary2) != ternary)
    throw PfError(KT2N_PF_EFILE, name + ": unexpected tensor kind");
  if (t->shape != shape) throw PfError(KT2N_PF_EFILE, name + ": unexpected shape");
  if (!ternary && t->data.dtype != kt2n::DType::kBF16) throw PfError(KT2N_PF_EFILE, name + ": dense tensor not bf16");
  return t;
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw PfError(KT2N_PF_ECUDA, std::string(what) + ": " + cudaGetErrorString(e));
}

// bad[0] |= any byte of a[0..n) != b[0..n)  (n % 16 == 0, 16-byte aligned)
__global__ void cmp16_kernel(const uint4* __restrict__ a, const uint4* __restrict__ b, size_t n16,
                             unsigned* __restrict__ bad) {
  unsigned diff = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 x = a[i], y = b[i];
    diff |= (x.x ^ y.x) | (x.y ^ y.y) | (x.z ^ y.z) | (x.w ^ y.w);
  }
  if (diff) atomicOr(bad, 1u);
}

void tk(int rc, const char* what) {
  if (rc != KT2N_TERN_OK) throw PfError(KT2N_PF_EKERNEL, std::string(what) + " failed (rc " + std::to_string(rc) + ")");
}

}  // namespace

kt2n_pf_model* load_model(const char* path, int device, const char* expect_sha) {
  if (!path || !*path) throw PfError(KT2N_PF_EINVAL, "engine file path is empty");
  kt2n::EngineFile f = [&]() {
    try {
      return kt2n::EngineFile::open(path);
    } catch (const std::exception& e) {
      throw PfError(KT2N_PF_EFILE, std::string("cannot open engine file: ") + e.what());
    }
  }();
  auto geti = [&](const char* ptr) -> int64_t {
    auto v = f.meta_int(ptr);
    if (!v) throw PfError(KT2N_PF_EFILE, std::string("engine file metadata lacks ") + ptr);
    return *v;
  };
  const int64_t layers = geti("/config/num_hidden_layers");
  const int64_t hidden = geti("/config/hidden_size");
  const int64_t inter = geti("/config/intermediate_size");
  const int64_t qh = geti("/config/num_attention_heads");
  const int64_t kvh = geti("/config/num_key_value_heads");
  const int64_t hd = f.meta_int("/config/head_dim").value_or(hidden / qh);
  const int64_t vocab = geti("/config/vocab_size");
  const double eps = f.meta_number("/config/rms_norm_eps").value_or(-1.0);
  const double theta = f.meta_number("/config/rope_theta").value_or(-1.0);
  // The kernel lanes are specialised to this geometry.
  if (layers < 1 || layers > KT2N_PF_MAX_LAYERS || hidden != 2048 || inter != 6144 || qh != 16 || kvh != 8 ||
      hd != 128 || vocab < KT2N_PF_AUDIO_LOW + KT2N_PF_AUDIO_N || eps <= 0.0)
    throw PfError(KT2N_PF_EFILE, "engine file geometry is not the Kitten TTS 2 LM this build supports");
  if (theta != 1e6) throw PfError(KT2N_PF_EFILE, "rope_theta != 1e6 (k2's built-in inv_freq table)");
  if (f.meta_bool("/config/tie_word_embeddings").value_or(true) != true)
    throw PfError(KT2N_PF_EFILE, "untied head not supported");
  std::string sha = f.meta_string("/source/safetensors/sha256").value_or("");
  if (expect_sha && *expect_sha && sha != expect_sha)
    throw PfError(KT2N_PF_EFILE, "engine file source sha256 " + sha + " != expected " + expect_sha);

  const int H = (int)hidden, I = (int)inter;
  const int NQ = (int)(qh * hd), NKV = (int)(kvh * hd), QKVN = NQ + 2 * NKV;
  const std::string pre = "model.layers.";
  // ----- plan the arena
  Plan plan;
  size_t off = 0;
  auto take = [&](size_t bytes) {
    size_t o = off;
    off = align_up(off + bytes);
    return o;
  };
  plan.embed = take((size_t)vocab * H * 2);
  plan.final_norm = take((size_t)H * 2);
  plan.layer.resize((size_t)layers);
  for (auto& L : plan.layer) {
    L.qkv_c = take(kt2n_tern_codes_bytes(QKVN, H));
    L.qkv_s = take(kt2n_tern_scales_bytes(QKVN, H));
    L.o_c = take(kt2n_tern_codes_bytes(H, NQ));
    L.o_s = take(kt2n_tern_scales_bytes(H, NQ));
    L.gu_c = take(kt2n_tern_codes_bytes(2 * I, H));
    L.gu_s = take(kt2n_tern_scales_bytes(2 * I, H));
    L.d_c = take(kt2n_tern_codes_bytes(H, I));
    L.d_s = take(kt2n_tern_scales_bytes(H, I));
    L.in_n = take((size_t)H * 2);
    L.post_n = take((size_t)H * 2);
    L.q_n = take((size_t)hd * 2);
    L.k_n = take((size_t)hd * 2);
  }
  plan.total = off;

  ck(cudaSetDevice(device), "cudaSetDevice");
  auto model = std::make_unique<kt2n_pf_model>();
  model->device = device;
  model->source_sha256 = sha;
  model->path = path;
  ck(cudaMalloc(&model->arena, plan.total), "cudaMalloc(weights)");
  model->bytes = plan.total;
  struct Guard {
    kt2n_pf_model* m;
    bool ok = false;
    ~Guard() {
      if (!ok && m->arena) cudaFree(m->arena);
    }
  } guard{model.get()};
  char* base = static_cast<char*>(model->arena);

  // staging for canonical codes/scales (largest family: gate + up)
  const size_t stage_bytes = align_up((size_t)2 * I * (H / 16) * 4) + align_up((size_t)2 * I * (H / 128) * 2) +
                             align_up((size_t)H * (I / 16) * 4) + align_up((size_t)H * (I / 128) * 2) + 4 * kAlign;
  void* stage = nullptr;
  // [0, 2*stage_bytes): canonical uploads; then one matrix's unrepacked copy (load verification) and a flag
  const size_t check_bytes = align_up((size_t)I * (H / 16) * 4) + align_up((size_t)I * (H / 128) * 2) + kAlign;
  ck(cudaMalloc(&stage, stage_bytes * 2 + check_bytes + kAlign), "cudaMalloc(stage)");
  char* check_codes = static_cast<char*>(stage) + stage_bytes * 2;
  char* check_scales = check_codes + align_up((size_t)I * (H / 16) * 4);
  auto* bad_flag = reinterpret_cast<unsigned*>(check_codes + check_bytes);
  struct StageGuard {
    void* p;
    ~StageGuard() { cudaFree(p); }
  } sguard{stage};
  cudaStream_t stream = nullptr;
  // EVERY upload and every repack runs on this one stream, in order.  (A plain cudaMemcpy from pageable memory
  // may return before its DMA has landed, and a non-blocking stream does not wait for the legacy stream: a
  // repack on another stream could read a half-written staging buffer.  Fixed 6 Oct after one parity run showed
  // every voice diverging from layer 13 on while the GPU was loaded by another process.)
  ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  struct StreamGuard {
    cudaStream_t s;
    ~StreamGuard() { cudaStreamDestroy(s); }
  } stguard{stream};

  auto copy_dense = [&](const kt2n::TensorInfo* t, size_t dst_off) {
    auto b = f.bytes(t->data);
    ck(cudaMemcpyAsync(base + dst_off, b.data(), b.size(), cudaMemcpyHostToDevice, stream), "copy dense");
  };
  // Upload canonical codes/scales of several ternary tensors into the staging buffer; returns device pointers.
  struct Can {
    const int32_t* codes;
    const void* scales;
  };
  auto upload = [&](std::vector<const kt2n::TensorInfo*> ts) {
    std::vector<Can> out;
    char* s = static_cast<char*>(stage);
    size_t o = 0;
    for (auto* t : ts) {
      auto c = f.bytes(t->codes);
      auto sc = f.bytes(t->scales);
      if (t->codes.dtype != kt2n::DType::kI32 || t->scales.dtype != kt2n::DType::kBF16)
        throw PfError(KT2N_PF_EFILE, t->name + ": unexpected ternary segment dtypes");
      if (o + align_up(c.size()) + align_up(sc.size()) > stage_bytes * 2)
        throw PfError(KT2N_PF_EFILE, "staging buffer too small");
      ck(cudaMemcpyAsync(s + o, c.data(), c.size(), cudaMemcpyHostToDevice, stream), "copy codes");
      Can can{reinterpret_cast<const int32_t*>(s + o), nullptr};
      o = align_up(o + c.size());
      ck(cudaMemcpyAsync(s + o, sc.data(), sc.size(), cudaMemcpyHostToDevice, stream), "copy scales");
      can.scales = s + o;
      o = align_up(o + sc.size());
      out.push_back(can);
    }
    return out;
  };

  // Load self-check: unrepack every matrix from the kernel layout (k1's exact inverse) and compare it on the
  // device with the canonical bytes just uploaded; any difference fails the load.
  ck(cudaMemsetAsync(bad_flag, 0, sizeof(unsigned), stream), "memset flag");
  auto verify = [&](const Can& can, size_t rows, size_t k, const void* codes, const void* scales, int n_total,
                    int row_offset, int row_step) {
    tk(kt2n_tern_unrepack(codes, scales, n_total, (int)k, row_offset, row_step, (int)rows,
                          reinterpret_cast<int32_t*>(check_codes), check_scales, stream),
       "unrepack (load check)");
    const size_t cb = rows * (k / 16) * 4, sb = rows * (k / 128) * 2;
    if (cb % 16 || sb % 16) throw PfError(KT2N_PF_EFILE, "load check: unaligned matrix sizes");
    cmp16_kernel<<<256, 256, 0, stream>>>(reinterpret_cast<const uint4*>(can.codes),
                                          reinterpret_cast<const uint4*>(check_codes), cb / 16, bad_flag);
    cmp16_kernel<<<64, 256, 0, stream>>>(reinterpret_cast<const uint4*>(can.scales),
                                         reinterpret_cast<const uint4*>(check_scales), sb / 16, bad_flag);
    ck(cudaGetLastError(), "load check");
  };

  copy_dense(need(f, "model.embed_tokens.weight", false, {vocab, hidden}), plan.embed);
  copy_dense(need(f, "model.norm.weight", false, {hidden}), plan.final_norm);
  kt2n_pf_weights& w = model->w;
  w.layers = (int32_t)layers;
  w.hidden = H;
  w.inter = I;
  w.q_heads = (int32_t)qh;
  w.kv_heads = (int32_t)kvh;
  w.head_dim = (int32_t)hd;
  w.vocab = vocab;
  w.rms_eps = (float)eps;
  w.embed = base + plan.embed;
  w.final_norm = base + plan.final_norm;
  w.inv_freq = nullptr;  // k2's built-in table == HF's buffer for rope_theta 1e6, head_dim 128 (checked above)
  for (int l = 0; l < (int)layers; ++l) {
    const auto& L = plan.layer[(size_t)l];
    const std::string p = pre + std::to_string(l) + ".";
    copy_dense(need(f, p + "input_layernorm.weight", false, {hidden}), L.in_n);
    copy_dense(need(f, p + "post_attention_layernorm.weight", false, {hidden}), L.post_n);
    copy_dense(need(f, p + "self_attn.q_norm.weight", false, {hd}), L.q_n);
    copy_dense(need(f, p + "self_attn.k_norm.weight", false, {hd}), L.k_n);
    {
      auto c = upload({need(f, p + "self_attn.q_proj.weight", true, {NQ, H}),
                       need(f, p + "self_attn.k_proj.weight", true, {NKV, H}),
                       need(f, p + "self_attn.v_proj.weight", true, {NKV, H})});
      tk(kt2n_tern_repack_qkv(c[0].codes, c[0].scales, c[1].codes, c[1].scales, c[2].codes, c[2].scales, NQ, NKV,
                              H, base + L.qkv_c, base + L.qkv_s, stream),
         "repack qkv");
      verify(c[0], NQ, H, base + L.qkv_c, base + L.qkv_s, QKVN, 0, 1);
      verify(c[1], NKV, H, base + L.qkv_c, base + L.qkv_s, QKVN, NQ, 1);
      verify(c[2], NKV, H, base + L.qkv_c, base + L.qkv_s, QKVN, NQ + NKV, 1);
      ck(cudaStreamSynchronize(stream), "repack qkv");
    }
    {
      auto c = upload({need(f, p + "self_attn.o_proj.weight", true, {H, NQ})});
      tk(kt2n_tern_repack(c[0].codes, c[0].scales, H, NQ, base + L.o_c, base + L.o_s, H, 0, 1, stream),
         "repack o");
      verify(c[0], H, NQ, base + L.o_c, base + L.o_s, H, 0, 1);
      ck(cudaStreamSynchronize(stream), "repack o");
    }
    {
      auto c = upload({need(f, p + "mlp.gate_proj.weight", true, {I, H}), need(f, p + "mlp.up_proj.weight", true, {I, H})});
      tk(kt2n_tern_repack_gate_up(c[0].codes, c[0].scales, c[1].codes, c[1].scales, I, H, base + L.gu_c,
                                  base + L.gu_s, stream),
         "repack gate_up");
      verify(c[0], I, H, base + L.gu_c, base + L.gu_s, 2 * I, 0, 2);
      verify(c[1], I, H, base + L.gu_c, base + L.gu_s, 2 * I, 1, 2);
      ck(cudaStreamSynchronize(stream), "repack gate_up");
    }
    {
      auto c = upload({need(f, p + "mlp.down_proj.weight", true, {H, I})});
      tk(kt2n_tern_repack(c[0].codes, c[0].scales, H, I, base + L.d_c, base + L.d_s, H, 0, 1, stream),
         "repack down");
      verify(c[0], H, I, base + L.d_c, base + L.d_s, H, 0, 1);
      ck(cudaStreamSynchronize(stream), "repack down");
    }
    kt2n_pf_layer& lw = w.layer[l];
    lw.qkv_codes = base + L.qkv_c;
    lw.qkv_scales = base + L.qkv_s;
    lw.o_codes = base + L.o_c;
    lw.o_scales = base + L.o_s;
    lw.gu_codes = base + L.gu_c;
    lw.gu_scales = base + L.gu_s;
    lw.down_codes = base + L.d_c;
    lw.down_scales = base + L.d_s;
    lw.input_norm = base + L.in_n;
    lw.post_norm = base + L.post_n;
    lw.q_norm = base + L.q_n;
    lw.k_norm = base + L.k_n;
  }
  unsigned bad = 0;
  ck(cudaMemcpyAsync(&bad, bad_flag, sizeof(bad), cudaMemcpyDeviceToHost, stream), "load check flag");
  ck(cudaStreamSynchronize(stream), "weights upload");
  ck(cudaDeviceSynchronize(), "weights upload");
  if (bad) throw PfError(KT2N_PF_EFILE, "load check failed: a repacked projection does not unpack to the file's codes");
  guard.ok = true;
  return model.release();
}

}  // namespace kt2n_pf

extern "C" int kt2n_pf_model_load(const char* path, int device, const char* expect_source_sha256,
                                  kt2n_pf_model** out) {
  return kt2n_pf::guarded([&] {
    if (!out) throw kt2n_pf::PfError(KT2N_PF_EINVAL, "out is NULL");
    *out = nullptr;
    *out = kt2n_pf::load_model(path, device, expect_source_sha256);
  });
}

extern "C" const kt2n_pf_weights* kt2n_pf_model_weights(const kt2n_pf_model* m) { return m ? &m->w : nullptr; }
extern "C" size_t kt2n_pf_model_bytes(const kt2n_pf_model* m) { return m ? m->bytes : 0; }
extern "C" const char* kt2n_pf_model_source_sha256(const kt2n_pf_model* m) {
  return m ? m->source_sha256.c_str() : "";
}
extern "C" void kt2n_pf_model_free(kt2n_pf_model* m) {
  if (!m) return;
  if (m->arena) {
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(m->device);
    cudaFree(m->arena);
    cudaSetDevice(prev);
  }
  delete m;
}
