// Render call execution: device arena, L7 kernels and the RenderCaller (include/kt2n/render/call.h,
// include/kt2n/render/arena.h).
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

#include "kt2n/render/arena.h"
#include "kt2n/render/call.h"
#include "cu_driver.h"
#include "device_zero.h"

namespace kt2n::render {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

using drv::CtxScope;
void cu_check(CUresult r, const char* what) { drv::check(r, what); }

// ---- kStateNoiseGather ---------------------------------------------------------------------------------
// grid = (rows_bucket, frame tiles); each thread handles (frame, channel) pairs of its row.
__global__ void noise_gather_kernel(const CallDesc* __restrict__ call) {
  const int32_t r = blockIdx.x;
  if (r >= call->n_rows || call->mel_out == nullptr) return;
  const RowDesc& d = call->rows[r];
  if (d.slot < 0 || (d.flags & kRowCancelled)) return;
  const StateView& st = call->state;
  const VoiceView& vv = call->voices;
  const int32_t n = d.fr_suffix * kMel;
  for (int32_t idx = blockIdx.y * blockDim.x + threadIdx.x; idx < n; idx += gridDim.y * blockDim.x) {
    const int32_t i = idx / kMel;
    const int32_t c = idx - i * kMel;
    const int32_t f = d.fr_past + i;  // absolute frame
    float v;
    if (f < d.prompt_frames) {
      v = vv.prompt_noise[(size_t(d.voice) * kMel + size_t(c)) * size_t(vv.max_prompt_frames) + size_t(f)];
    } else {
      const int32_t t = f - d.prompt_frames;
      v = t < st.noise_cap
              ? st.target_noise[(size_t(d.slot) * kMel + size_t(c)) * size_t(st.noise_cap) + size_t(t)]
              : 0.0f;
    }
    call->mel_out[(size_t(d.fr_off) + size_t(i)) * kMel + size_t(c)] = v;
  }
}

void noise_gather_enqueue(void*, const CallDesc& h, const CallDesc* d, uint64_t, CUstream_st* s) {
  dim3 grid(unsigned(h.rows_bucket), 8);
  noise_gather_kernel<<<grid, 256, 0, s>>>(d);
}

// ---- stub PCM ------------------------------------------------------------------------------------------
__global__ void stub_pcm_kernel(const CallDesc* __restrict__ call) {
  const int32_t r = blockIdx.x;
  if (r >= call->n_rows) return;
  const RowDesc& d = call->rows[r];
  if (d.slot < 0) return;
  for (int32_t j = blockIdx.y * blockDim.x + threadIdx.x; j < d.pcm_count; j += gridDim.y * blockDim.x)
    call->pcm_f32[d.pcm_off + j] = float((d.slot << 16) | (j & 0xFFFF));  // exact integers (< 2^24)
}

void stub_pcm_enqueue(void*, const CallDesc& h, const CallDesc* d, uint64_t, CUstream_st* s) {
  dim3 grid(unsigned(h.rows_bucket), 16);
  stub_pcm_kernel<<<grid, 256, 0, s>>>(d);
}

// ---- kStatePrefixCopy ----------------------------------------------------------------------------------
// One block per (domain chunk): enc1 12 chunks of 32 x 512 fp32, enc2 8 of 64 x 512 fp32,
// est 112 of 64 x 1024 bf16 -- each chunk is one page of one (layer, kv / step) plane.
__global__ void prefix_page_copy_kernel(PoolView p, int32_t src, int32_t dst) {
  const int32_t b = blockIdx.x;
  const uint4* s4;
  uint4* d4;
  size_t n16;
  if (b < kEnc1Layers * 2) {
    const size_t so = enc1_offset(p, b / 2, b % 2, src, 0), dof = enc1_offset(p, b / 2, b % 2, dst, 0);
    s4 = reinterpret_cast<const uint4*>(p.enc1 + so);
    d4 = reinterpret_cast<uint4*>(p.enc1 + dof);
    n16 = size_t(kPageTokens) * kEncDim * sizeof(float) / 16;
  } else if (b < kEnc1Layers * 2 + kEnc2Layers * 2) {
    const int32_t e = b - kEnc1Layers * 2;
    const size_t so = enc2_offset(p, e / 2, e % 2, src, 0), dof = enc2_offset(p, e / 2, e % 2, dst, 0);
    s4 = reinterpret_cast<const uint4*>(p.enc2 + so);
    d4 = reinterpret_cast<uint4*>(p.enc2 + dof);
    n16 = size_t(kPageFrames) * kEncDim * sizeof(float) / 16;
  } else {
    const int32_t e = b - kEnc1Layers * 2 - kEnc2Layers * 2;
    const int32_t step = e / kEstAttnLayers, layer = e % kEstAttnLayers;
    const size_t so = est_offset(p, step, layer, src, 0), dof = est_offset(p, step, layer, dst, 0);
    s4 = reinterpret_cast<const uint4*>(p.est + so);
    d4 = reinterpret_cast<uint4*>(p.est + dof);
    n16 = size_t(kPageFrames) * 2 * kEstInner * sizeof(uint16_t) / 16;
  }
  for (size_t i = threadIdx.x; i < n16; i += blockDim.x) d4[i] = s4[i];
}

}  // namespace

RenderStage noise_gather_stage() {
  RenderStage s;
  s.name = "l7.noise_gather";
  s.enqueue = &noise_gather_enqueue;
  return s;
}

RenderStage stub_pcm_stage() {
  RenderStage s;
  s.name = "l7.stub_pcm";
  s.enqueue = &stub_pcm_enqueue;
  return s;
}

// ---- RenderArena ---------------------------------------------------------------------------------------
RenderArena::RenderArena(const ArenaConfig& cfg) : cfg_(cfg) {
  check(cudaSetDevice(cfg.device), "cudaSetDevice");
  const size_t P = size_t(cfg.pool_pages), S = size_t(cfg.slots), V = size_t(cfg.voices);
  // Layout (each piece 256-byte aligned).
  size_t off = 0;
  auto take = [&](size_t bytes) {
    const size_t o = align_up(off, 256);
    off = o + bytes;
    return o;
  };
  const size_t o_enc1 = take(P * kEnc1PageBytes);
  const size_t o_enc2 = take(P * kEnc2PageBytes);
  const size_t o_est = take(P * kEstPageBytes);
  const size_t o_noise = take(S * kMel * size_t(cfg.noise_cap) * sizeof(float));
  const size_t o_pre = take(S * 2 * kEncDim * sizeof(float));
  const size_t o_up = take(S * 2 * kEncDim * sizeof(float));
  const size_t o_est_tail = take(size_t(kFlowSteps) * S * size_t(cfg.est_tail_width) * sizeof(uint16_t) + 16);
  const size_t o_mel = take(S * kMel * kMelCacheFrames * sizeof(float));
  const size_t o_src = take(S * kSourceCacheSamples * sizeof(float));
  const size_t o_held = take(S * kSourceCacheSamples * sizeof(float));
  const size_t o_phase = take(S * kHarmonics * sizeof(float));
  const size_t o_vfeat = take(V * size_t(cfg.max_prompt_frames) * kMel * sizeof(float));
  const size_t o_vnoise = take(V * kMel * size_t(cfg.max_prompt_frames) * sizeof(float));
  const size_t o_vspk = take(V * kMel * sizeof(float));
  const size_t o_vtok = take(V * size_t(cfg.max_prompt_frames / 2) * sizeof(int32_t));
  const size_t o_vframes = take(V * sizeof(int32_t));
  const size_t o_vpages = take(V * size_t(cfg.max_prefix_pages) * sizeof(int32_t));
  const size_t o_vptok = take(V * sizeof(int32_t));
  const size_t o_venc = take(V * 2 * 2 * kEncDim * sizeof(float));
  const size_t o_vest = take(V * size_t(kFlowSteps) * size_t(cfg.est_tail_width) * sizeof(uint16_t) + 16);
  bytes_ = align_up(off, 256);
  check(cudaMalloc(&base_, bytes_), "cudaMalloc(render arena)");
  zero_device(base_, bytes_, "zero render arena");
  char* b = static_cast<char*>(base_);
  pool_.enc1 = reinterpret_cast<float*>(b + o_enc1);
  pool_.enc2 = reinterpret_cast<float*>(b + o_enc2);
  pool_.est = reinterpret_cast<uint16_t*>(b + o_est);
  pool_.n_pages = cfg.pool_pages;
  state_.target_noise = reinterpret_cast<float*>(b + o_noise);
  state_.enc_pre_tail = reinterpret_cast<float*>(b + o_pre);
  state_.enc_up_tail = reinterpret_cast<float*>(b + o_up);
  state_.est_tail = reinterpret_cast<uint16_t*>(b + o_est_tail);
  state_.hift_mel_cache = reinterpret_cast<float*>(b + o_mel);
  state_.hift_src_cache = reinterpret_cast<float*>(b + o_src);
  state_.hift_held_tail = reinterpret_cast<float*>(b + o_held);
  state_.hift_phase = reinterpret_cast<float*>(b + o_phase);
  state_.slots = cfg.slots;
  state_.noise_cap = cfg.noise_cap;
  state_.est_tail_width = cfg.est_tail_width;
  voices_.prompt_feat = reinterpret_cast<const float*>(b + o_vfeat);
  voices_.prompt_noise = reinterpret_cast<const float*>(b + o_vnoise);
  voices_.spk_cond = reinterpret_cast<const float*>(b + o_vspk);
  voices_.prompt_tokens = reinterpret_cast<const int32_t*>(b + o_vtok);
  voices_.prompt_frames = reinterpret_cast<const int32_t*>(b + o_vframes);
  voices_.prefix_pages = reinterpret_cast<const int32_t*>(b + o_vpages);
  voices_.prefix_tokens = reinterpret_cast<const int32_t*>(b + o_vptok);
  voices_.prefix_enc_tails = reinterpret_cast<const float*>(b + o_venc);
  voices_.prefix_est_tails = reinterpret_cast<const uint16_t*>(b + o_vest);
  voices_.n_voices = cfg.voices;
  voices_.max_prompt_frames = cfg.max_prompt_frames;
  voices_.max_prefix_pages = cfg.max_prefix_pages;
}

RenderArena::~RenderArena() {
  if (base_ != nullptr) cudaFree(base_);
}

void RenderArena::upload_row_noise(int32_t slot, const float* host, int32_t w, CUstream_st* stream) {
  if (slot < 0 || slot >= cfg_.slots || w < 0 || w > cfg_.noise_cap) throw std::invalid_argument("upload_row_noise");
  float* dst = state_.target_noise + size_t(slot) * kMel * size_t(cfg_.noise_cap);
  // Zero the slot (frames past w read as production's zero padding), then the drawn frames.
  check(cudaMemsetAsync(dst, 0, size_t(kMel) * size_t(cfg_.noise_cap) * sizeof(float), stream), "noise memset");
  if (w > 0)
    check(cudaMemcpy2DAsync(dst, size_t(cfg_.noise_cap) * sizeof(float), host, size_t(w) * sizeof(float),
                            size_t(w) * sizeof(float), kMel, cudaMemcpyHostToDevice, stream),
          "noise upload");
}

void RenderArena::upload_row_noise_range(int32_t slot, const float* host, int32_t w0, int32_t w1, CUstream_st* stream) {
  if (slot < 0 || slot >= cfg_.slots || w0 < 0 || w1 < w0 || w1 > cfg_.noise_cap)
    throw std::invalid_argument("upload_row_noise_range");
  if (w1 == w0) return;
  float* dst = state_.target_noise + size_t(slot) * kMel * size_t(cfg_.noise_cap) + size_t(w0);
  const float* src = host + w0;  // host is [80][noise_cap]-strided like the device table
  check(cudaMemcpy2DAsync(dst, size_t(cfg_.noise_cap) * sizeof(float), src, size_t(cfg_.noise_cap) * sizeof(float),
                          size_t(w1 - w0) * sizeof(float), kMel, cudaMemcpyHostToDevice, stream),
        "noise extension upload");
}

void RenderArena::set_voice(int32_t voice, int32_t prompt_frames, const float* prompt_noise,
                            const int32_t* prompt_tokens, const float* prompt_feat, const float* spk_cond,
                            CUstream_st* stream) {
  if (voice < 0 || voice >= cfg_.voices || prompt_frames < 0 || prompt_frames > cfg_.max_prompt_frames ||
      (prompt_frames % kMelPerToken) != 0)
    throw std::invalid_argument("set_voice");
  const size_t mpf = size_t(cfg_.max_prompt_frames);
  // On a private non-blocking stream: a voice registered while serving (clone, saved voice) must not wait
  // for the render or LM streams' queued work, which the legacy default stream's synchronous copies do.
  auto copy = [stream](void* dst, const void* src, size_t bytes, const char* what) {
    if (stream != nullptr) check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), what);
    else check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), what);
  };
  if (prompt_noise != nullptr) {
    const size_t pitch = size_t(prompt_frames) * sizeof(float);
    if (stream != nullptr)
      check(cudaMemcpy2DAsync(const_cast<float*>(voices_.prompt_noise) + size_t(voice) * kMel * mpf, mpf * sizeof(float),
                              prompt_noise, pitch, pitch, kMel, cudaMemcpyHostToDevice, stream),
            "voice prompt noise");
    else
      check(cudaMemcpy2D(const_cast<float*>(voices_.prompt_noise) + size_t(voice) * kMel * mpf, mpf * sizeof(float),
                         prompt_noise, pitch, pitch, kMel, cudaMemcpyHostToDevice),
            "voice prompt noise");
  }
  if (prompt_tokens != nullptr)
    copy(const_cast<int32_t*>(voices_.prompt_tokens) + size_t(voice) * (mpf / 2), prompt_tokens,
         size_t(prompt_frames / 2) * sizeof(int32_t), "voice prompt tokens");
  if (prompt_feat != nullptr)
    copy(const_cast<float*>(voices_.prompt_feat) + size_t(voice) * mpf * kMel, prompt_feat,
         size_t(prompt_frames) * kMel * sizeof(float), "voice prompt feat");
  if (spk_cond != nullptr)
    copy(const_cast<float*>(voices_.spk_cond) + size_t(voice) * kMel, spk_cond, kMel * sizeof(float), "voice spk");
  copy(const_cast<int32_t*>(voices_.prompt_frames) + voice, &prompt_frames, sizeof(int32_t), "voice frames");
  // The sources are pageable host memory owned by the caller: done before returning.
  if (stream != nullptr) check(cudaStreamSynchronize(stream), "voice upload");
}

void RenderArena::copy_prefix_page(int32_t src_page, int32_t dst_page, CUstream_st* stream) {
  if (src_page < 0 || dst_page < 0 || src_page >= pool_.n_pages || dst_page >= pool_.n_pages)
    throw std::invalid_argument("copy_prefix_page");
  const int32_t blocks = kEnc1Layers * 2 + kEnc2Layers * 2 + kFlowSteps * kEstAttnLayers;
  prefix_page_copy_kernel<<<blocks, 256, 0, stream>>>(pool_, src_page, dst_page);
  check(cudaGetLastError(), "prefix_page_copy_kernel");
}

void RenderArena::save_prefix_tails(int32_t slot, int32_t voice, CUstream_st* stream) {
  const size_t enc = 2 * size_t(kEncDim) * sizeof(float);
  float* venc = const_cast<float*>(voices_.prefix_enc_tails) + size_t(voice) * 2 * 2 * kEncDim;
  check(cudaMemcpyAsync(venc, state_.enc_pre_tail + size_t(slot) * 2 * kEncDim, enc, cudaMemcpyDeviceToDevice, stream), "tails");
  check(cudaMemcpyAsync(venc + 2 * kEncDim, state_.enc_up_tail + size_t(slot) * 2 * kEncDim, enc, cudaMemcpyDeviceToDevice, stream),
        "tails");
  const size_t w = size_t(cfg_.est_tail_width);
  for (int32_t st = 0; st < kFlowSteps && w > 0; ++st)
    check(cudaMemcpyAsync(const_cast<uint16_t*>(voices_.prefix_est_tails) + (size_t(voice) * kFlowSteps + size_t(st)) * w,
                          state_.est_tail + (size_t(st) * size_t(cfg_.slots) + size_t(slot)) * w, w * sizeof(uint16_t),
                          cudaMemcpyDeviceToDevice, stream),
          "est tails");
}

void RenderArena::load_prefix_tails(int32_t voice, int32_t slot, CUstream_st* stream) {
  const size_t enc = 2 * size_t(kEncDim) * sizeof(float);
  const float* venc = voices_.prefix_enc_tails + size_t(voice) * 2 * 2 * kEncDim;
  check(cudaMemcpyAsync(state_.enc_pre_tail + size_t(slot) * 2 * kEncDim, venc, enc, cudaMemcpyDeviceToDevice, stream), "tails");
  check(cudaMemcpyAsync(state_.enc_up_tail + size_t(slot) * 2 * kEncDim, venc + 2 * kEncDim, enc, cudaMemcpyDeviceToDevice, stream),
        "tails");
  const size_t w = size_t(cfg_.est_tail_width);
  for (int32_t st = 0; st < kFlowSteps && w > 0; ++st)
    check(cudaMemcpyAsync(state_.est_tail + (size_t(st) * size_t(cfg_.slots) + size_t(slot)) * w,
                          voices_.prefix_est_tails + (size_t(voice) * kFlowSteps + size_t(st)) * w, w * sizeof(uint16_t),
                          cudaMemcpyDeviceToDevice, stream),
          "est tails");
}

void RenderArena::clear_slot(int32_t slot, CUstream_st* stream) {
  check(cudaMemsetAsync(state_.enc_pre_tail + size_t(slot) * 2 * kEncDim, 0, 2 * kEncDim * sizeof(float), stream), "clear");
  check(cudaMemsetAsync(state_.enc_up_tail + size_t(slot) * 2 * kEncDim, 0, 2 * kEncDim * sizeof(float), stream), "clear");
  const size_t w = size_t(cfg_.est_tail_width);
  for (int32_t st = 0; st < kFlowSteps && w > 0; ++st)
    check(cudaMemsetAsync(state_.est_tail + (size_t(st) * size_t(cfg_.slots) + size_t(slot)) * w, 0, w * sizeof(uint16_t), stream),
          "clear");
  check(cudaMemsetAsync(state_.hift_mel_cache + size_t(slot) * kMel * kMelCacheFrames, 0, kMel * kMelCacheFrames * sizeof(float),
                        stream),
        "clear");
  check(cudaMemsetAsync(state_.hift_src_cache + size_t(slot) * kSourceCacheSamples, 0, kSourceCacheSamples * sizeof(float), stream),
        "clear");
  check(cudaMemsetAsync(state_.hift_held_tail + size_t(slot) * kSourceCacheSamples, 0, kSourceCacheSamples * sizeof(float), stream),
        "clear");
}

// ---- RenderCaller --------------------------------------------------------------------------------------
namespace {
constexpr int kRoles = kCallerRoles;  // urgent, bulk, aux (aux only when RenderCallerConfig::aux_max_rows > 0)
int role_index(StreamRole r) {
  const int i = caller_role(r);
  if (i < 0) throw std::invalid_argument("render calls run on kSUrgent, kSBulk or kSAux");
  return i;
}

struct CallSlot {
  void* staging = nullptr;  // pinned: CallDesc | RowDesc[] | tokens | page table
  float* pcm = nullptr;     // pinned PCM out
  cudaEvent_t done = nullptr;
  cudaEvent_t t0 = nullptr, t1 = nullptr;
  bool busy = false;
  bool timed = false;
  uint64_t serial = 0;
  int32_t total_pcm = 0;
};

struct Role {
  CUcontext ctx = nullptr;             // bulk_sms: the role's green context (its SM partition); nullptr = primary
  cudaStream_t stream = nullptr;
  cudaStream_t post_stream = nullptr;  // pipeline_post: the post stage's (HiFT) stream, same priority
  void* d_desc = nullptr;
  size_t desc_bytes = 0;
  float* d_pcm = nullptr;
  int16_t* d_pcm16 = nullptr;
  float* d_mel = nullptr;
  void* d_ws = nullptr;
  size_t ws_bytes = 0;
  // [0, kMaxQueuedBatchesPerStream): the conductor's calls (1 executing + 1 queued). The extra last slot is for
  // internal work only (voice-prefix builds), so a build never waits for a conductor call to finish.
  CallSlot slots[kMaxQueuedBatchesPerStream + 1];
  int next = 0;
};
}  // namespace

struct RenderCaller::Impl {
  RenderCallerConfig cfg;
  PoolView pool;
  StateView state;
  VoiceView voices;
  std::vector<RenderStage> program;
  Role roles[kRoles];
  std::map<std::tuple<int, int, int, int, int>, cudaGraphExec_t> graphs;
  std::map<std::tuple<int, int, int, int, int>, int32_t> graph_nodes;
  uint64_t serial = 0;
  size_t staging_bytes = 0;
  RenderPostStage post{};
  CUgreenCtx green = nullptr;  // bulk_sms > 0: S_bulk's SM partition
  int32_t green_sms = 0;

  CallDesc host_call(int role, Phase phase, int32_t rows_bucket, int32_t suffix_bucket) const {
    CallDesc c{};
    c.phase = uint16_t(phase);
    c.rows_bucket = rows_bucket;
    c.suffix_bucket = suffix_bucket;
    c.flow_step = -1;
    c.pool = pool;
    c.state = state;
    c.voices = voices;
    c.workspace = roles[role].d_ws;
    c.workspace_bytes = roles[role].ws_bytes;
    c.mel_out = roles[role].d_mel;
    c.pcm_f32 = roles[role].d_pcm;
    c.pcm16 = roles[role].d_pcm16;
    char* dd = static_cast<char*>(roles[role].d_desc);
    const size_t o_rows = align_up(sizeof(CallDesc), 16);
    const size_t o_pages = align_up(o_rows + 2 * size_t(cfg.max_rows) * sizeof(RowDesc), 16);  // enc + est tables
    const size_t o_tok = align_up(o_pages + size_t(cfg.max_call_pages) * sizeof(int32_t), 16);
    c.rows = reinterpret_cast<const RowDesc*>(dd + o_rows);
    c.page_table = reinterpret_cast<const int32_t*>(dd + o_pages);
    c.tokens = reinterpret_cast<const int32_t*>(dd + o_tok);
    c.reserved[0] = 0;
    return c;
  }

  using GraphKey = std::tuple<int, int, int, int, int>;  // role, phase, rows bucket, suffix bucket, keys bucket
  // Capture the program for `key` on the role's stream (records only; nothing runs) and instantiate it.
  std::map<GraphKey, cudaGraphExec_t>::iterator capture(int role, const GraphKey& key) {
    Role& R = roles[role];
    const CallDesc* d_call = reinterpret_cast<const CallDesc*>(R.d_desc);
    CallDesc h = host_call(role, Phase(std::get<1>(key)), std::get<2>(key), std::get<3>(key));
    h.reserved[0] = std::get<4>(key);
    cudaGraph_t graph = nullptr;
    check(cudaStreamBeginCapture(R.stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    try {
      enqueue_program(role, h, d_call);
    } catch (...) {
      // Leave the stream usable: a failed capture must not poison every later call on this role.
      cudaGraph_t partial = nullptr;
      cudaStreamEndCapture(R.stream, &partial);
      if (partial != nullptr) cudaGraphDestroy(partial);
      (void)cudaGetLastError();
      throw;
    }
    check(cudaStreamEndCapture(R.stream, &graph), "end capture");
    size_t n_nodes = 0;
    check(cudaGraphGetNodes(graph, nullptr, &n_nodes), "graph nodes");
    graph_nodes[key] = int32_t(n_nodes);
    cudaGraphExec_t exec = nullptr;
    check(cudaGraphInstantiate(&exec, graph, 0), "graph instantiate");
    cudaGraphDestroy(graph);
    // Do the first-launch setup now (stream-ordered), not inside the first live call.
    check(cudaGraphUpload(exec, R.stream), "graph upload");
    return graphs.emplace(key, exec).first;
  }

  void enqueue_program(int role, const CallDesc& h, const CallDesc* d) {
    uint64_t off = 0;
    for (const RenderStage& s : program) {
      const uint64_t need =
          s.workspace_bytes ? s.workspace_bytes(s.ctx, h) : 0;
      if (off + need > roles[role].ws_bytes)
        throw std::runtime_error(std::string("render workspace too small for ") + s.name + ": need " +
                                 std::to_string(off + need) + " bytes");
      s.enqueue(s.ctx, h, d, off, roles[role].stream);
      off = align_up(off + need, 256);
    }
  }
};

RenderCaller::RenderCaller(const RenderCallerConfig& cfg, const PoolView& pool, const StateView& state,
                           const VoiceView& voices, std::vector<RenderStage> program)
    : impl_(new Impl) {
  Impl& m = *impl_;
  m.cfg = cfg;
  m.pool = pool;
  m.state = state;
  m.voices = voices;
  m.program = std::move(program);
  check(cudaSetDevice(cfg.device), "cudaSetDevice");
  int lo = 0, hi = 0;
  check(cudaDeviceGetStreamPriorityRange(&lo, &hi), "stream priorities");
  m.staging_bytes = align_up(sizeof(CallDesc), 16) + 2 * size_t(cfg.max_rows) * sizeof(RowDesc) +
                    size_t(cfg.max_call_tokens) * sizeof(int32_t) + size_t(cfg.max_call_pages) * sizeof(int32_t) + 128;
  if (cfg.bulk_sms > 0) {
    // S_bulk in its own SM partition (green context, sm_120 probe: 8-SM steps, full isolation). S_urgent stays in
    // the primary context, which may use EVERY SM, so the SMs outside the bulk partition are always free for
    // first commits (and the LM), and they can still borrow idle bulk SMs.
    check(cudaFree(nullptr), "primary context");
    const drv::Api& D = drv::api();  // resolved here only: the default path never touches the driver API
    CUdevice dev;
    cu_check(D.DeviceGet(&dev, cfg.device), "cuDeviceGet");
    CUdevResource all, grp[1], rem;
    cu_check(D.DeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
    unsigned nb = 1;
    cu_check(D.DevSmResourceSplitByCount(grp, &nb, &all, &rem, 0, unsigned(cfg.bulk_sms)), "cuDevSmResourceSplitByCount");
    CUdevResourceDesc d;
    cu_check(D.DevResourceGenerateDesc(&d, &grp[0], 1), "cuDevResourceGenerateDesc");
    cu_check(D.GreenCtxCreate(&m.green, d, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
    cu_check(D.CtxFromGreenCtx(&m.roles[1].ctx, m.green), "cuCtxFromGreenCtx");
    m.green_sms = int32_t(grp[0].sm.smCount);
  }
  for (int r = 0; r < kRoles; ++r) {
    if (r == 2 && cfg.aux_max_rows <= 0) continue;  // no aux role: nothing allocated
    Role& R = m.roles[r];
    // S_urgent at the top priority, S_bulk at the lowest, S_aux at the top or aux_priority_level levels below it
    // (lower number = higher priority: hi <= lo).
    const int prio = r == 1 ? lo : r == 2 ? std::min(lo, hi + std::max(0, cfg.aux_priority_level)) : hi;
    CtxScope cs(R.ctx);  // the role's streams and events belong to its context
    if (R.ctx != nullptr) {
      CUstream st = nullptr, pst = nullptr;
      cu_check(drv::api().GreenCtxStreamCreate(&st, m.green, CU_STREAM_NON_BLOCKING, prio), "green stream");
      cu_check(drv::api().GreenCtxStreamCreate(&pst, m.green, CU_STREAM_NON_BLOCKING, prio), "green stream");
      R.stream = reinterpret_cast<cudaStream_t>(st);
      R.post_stream = reinterpret_cast<cudaStream_t>(pst);
    } else {
      check(cudaStreamCreateWithPriority(&R.stream, cudaStreamNonBlocking, prio), "stream");
      // Created always (so set_pipeline_post can switch at run time); used only while pipeline_post is on.
      check(cudaStreamCreateWithPriority(&R.post_stream, cudaStreamNonBlocking, prio), "post stream");
    }
    R.desc_bytes = m.staging_bytes;
    check(cudaMalloc(&R.d_desc, R.desc_bytes), "desc arena");
    check(cudaMalloc(reinterpret_cast<void**>(&R.d_pcm), size_t(cfg.max_call_pcm) * sizeof(float)), "pcm");
    check(cudaMalloc(reinterpret_cast<void**>(&R.d_pcm16), size_t(cfg.max_call_pcm) * sizeof(int16_t)), "pcm16");
    check(cudaMalloc(reinterpret_cast<void**>(&R.d_mel), size_t(cfg.max_call_tokens) * 2 * kMel * sizeof(float)), "mel");
    R.ws_bytes = r == 2 ? cfg.aux_workspace_bytes : cfg.workspace_bytes;
    check(cudaMalloc(&R.d_ws, R.ws_bytes), "workspace");
    zero_device(R.d_ws, R.ws_bytes, "workspace zero");  // padding frames read zeros, never stale NaN
    zero_device(R.d_mel, size_t(cfg.max_call_tokens) * 2 * kMel * sizeof(float), "mel zero");
    for (CallSlot& s : R.slots) {
      // Portable: pinned for every context (a green context's copies from them stay asynchronous).
      check(cudaHostAlloc(&s.staging, m.staging_bytes, cudaHostAllocPortable), "pinned staging");
      check(cudaHostAlloc(reinterpret_cast<void**>(&s.pcm), size_t(cfg.max_call_pcm) * sizeof(float), cudaHostAllocPortable),
            "pinned pcm");
      check(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "event");
      check(cudaEventCreate(&s.t0), "event");
      check(cudaEventCreate(&s.t1), "event");
    }
  }
}

RenderCaller::~RenderCaller() {
  Impl& m = *impl_;
  for (auto& kv : m.graphs) cudaGraphExecDestroy(kv.second);
  for (Role& R : m.roles) {
    if (R.stream == nullptr) continue;  // the aux role when it is off
    CtxScope cs(R.ctx);
    if (R.stream) cudaStreamSynchronize(R.stream);
    if (R.post_stream) {
      cudaStreamSynchronize(R.post_stream);
      cudaStreamDestroy(R.post_stream);
    }
    for (CallSlot& s : R.slots) {
      cudaFreeHost(s.staging);
      cudaFreeHost(s.pcm);
      cudaEventDestroy(s.done);
      cudaEventDestroy(s.t0);
      cudaEventDestroy(s.t1);
    }
    cudaFree(R.d_desc);
    cudaFree(R.d_pcm);
    cudaFree(R.d_pcm16);
    cudaFree(R.d_mel);
    cudaFree(R.d_ws);
    cudaStreamDestroy(R.stream);
  }
  if (m.green != nullptr) drv::api().GreenCtxDestroy(m.green);
  delete impl_;
}

CUctx_st* RenderCaller::role_context(StreamRole role) const { return impl_->roles[role_index(role)].ctx; }
int32_t RenderCaller::bulk_partition_sms() const { return impl_->green_sms; }

bool RenderCaller::pipelined() const { return impl_->cfg.pipeline_post && impl_->post.run != nullptr; }

void RenderCaller::set_pipeline_post(bool on) {
  Impl& m = *impl_;
  if (on == m.cfg.pipeline_post) return;
  // Only between calls: everything queued so far completes before the mode changes (A/B benches).
  for (const Role& R : m.roles) {
    if (R.stream == nullptr) continue;
    CtxScope cs(R.ctx);
    check(cudaStreamSynchronize(R.stream), "pipeline switch");
    check(cudaStreamSynchronize(R.post_stream), "pipeline switch");
  }
  m.cfg.pipeline_post = on;
}

bool RenderCaller::can_launch(StreamRole role_kind, bool internal_slot) const {
  const Role& R = impl_->roles[role_index(role_kind)];
  if (R.stream == nullptr) return false;  // the aux role when it is off
  if (internal_slot) return !R.slots[kMaxQueuedBatchesPerStream].busy;
  for (int k = 0; k < kMaxQueuedBatchesPerStream; ++k)
    if (!R.slots[k].busy) return true;
  return false;
}

CUevent_st* RenderCaller::done_event(const RenderTicket& t) const { return impl_->roles[t.role].slots[t.call_slot].done; }

bool RenderCaller::launch(const CallPlan& plan, StreamRole role_kind, RenderTicket* ticket, RenderTiming* timing,
                          CUevent_st* const* waits, int32_t n_waits, bool internal_slot) {
  Impl& m = *impl_;
  const auto t_start = std::chrono::steady_clock::now();
  const int role = role_index(role_kind);
  Role& R = m.roles[role];
  CtxScope cs(R.ctx);
  if (R.stream == nullptr) throw std::invalid_argument("render call on the aux role, which is off");
  if (role == 2 && plan.rows_bucket > m.cfg.aux_max_rows)
    throw std::invalid_argument("render call has more rows than the aux role serves");
  if (plan.rows_bucket > m.cfg.max_rows || plan.total_tokens > m.cfg.max_call_tokens ||
      int32_t(plan.page_table.size()) > m.cfg.max_call_pages || plan.total_pcm > m.cfg.max_call_pcm)
    throw std::invalid_argument("render call exceeds the caller's staging capacity");
  // A free call slot (1 executing + 1 queued per stream).
  CallSlot* slot = nullptr;
  int slot_index = -1;
  if (internal_slot) {
    if (R.slots[kMaxQueuedBatchesPerStream].busy) return false;
    slot_index = kMaxQueuedBatchesPerStream;
    slot = &R.slots[slot_index];
  } else {
    for (int k = 0; k < kMaxQueuedBatchesPerStream; ++k) {
      const int idx = (R.next + k) % kMaxQueuedBatchesPerStream;
      CallSlot& s = R.slots[idx];
      if (s.busy) continue;
      slot = &s;
      slot_index = idx;
      break;
    }
    if (slot == nullptr) return false;
    R.next = (slot_index + 1) % kMaxQueuedBatchesPerStream;
  }

  // Staging and device layout at FIXED offsets: CallDesc | RowDesc[max_rows] | page table[max_call_pages] |
  // tokens[max_call_tokens]. Fixed offsets matter: lanes may bake the rows / page-table / token pointers
  // into a captured graph (the estimator's EstCall does), so they must not move between calls.
  char* st = static_cast<char*>(slot->staging);
  const size_t o_rows = align_up(sizeof(CallDesc), 16);
  const size_t o_pages = align_up(o_rows + 2 * size_t(m.cfg.max_rows) * sizeof(RowDesc), 16);  // enc + est tables
  const size_t o_tok = align_up(o_pages + size_t(m.cfg.max_call_pages) * sizeof(int32_t), 16);
  CallDesc c = m.host_call(role, plan.phase, plan.rows_bucket, plan.suffix_bucket);
  c.n_rows = plan.n_rows;
  c.total_tokens = plan.total_tokens;
  c.total_frames = plan.total_frames;
  c.total_pcm = plan.total_pcm;
  c.reserved[0] = plan.keys_bucket;
  std::memcpy(st, &c, sizeof(CallDesc));
  std::memcpy(st + o_rows, plan.rows.data(), size_t(plan.rows_bucket) * sizeof(RowDesc));
  // The estimator's table (16-aligned fr_off) right after the encoder's: CallDesc::rows + rows_bucket.
  if (int32_t(plan.est_rows.size()) == plan.rows_bucket)
    std::memcpy(st + o_rows + size_t(plan.rows_bucket) * sizeof(RowDesc), plan.est_rows.data(),
                size_t(plan.rows_bucket) * sizeof(RowDesc));
  else
    std::memcpy(st + o_rows + size_t(plan.rows_bucket) * sizeof(RowDesc), plan.rows.data(),
                size_t(plan.rows_bucket) * sizeof(RowDesc));
  if (!plan.tokens.empty()) std::memcpy(st + o_tok, plan.tokens.data(), plan.tokens.size() * sizeof(int32_t));
  if (!plan.page_table.empty())
    std::memcpy(st + o_pages, plan.page_table.data(), plan.page_table.size() * sizeof(int32_t));

  for (int32_t i = 0; i < n_waits; ++i)
    if (waits[i] != nullptr) check(cudaStreamWaitEvent(R.stream, waits[i], 0), "stream wait");
  const bool timed = timing_;
  if (timed) check(cudaEventRecord(slot->t0, R.stream), "t0");
  char* dd = static_cast<char*>(R.d_desc);
  check(cudaMemcpyAsync(dd, st, o_rows + 2 * size_t(plan.rows_bucket) * sizeof(RowDesc), cudaMemcpyHostToDevice, R.stream),
        "desc H2D");
  if (!plan.page_table.empty())
    check(cudaMemcpyAsync(dd + o_pages, st + o_pages, plan.page_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                          R.stream),
          "pages H2D");
  if (!plan.tokens.empty())
    check(cudaMemcpyAsync(dd + o_tok, st + o_tok, plan.tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice, R.stream),
          "tokens H2D");

  bool captured = false;
  const CallDesc* d_call = reinterpret_cast<const CallDesc*>(R.d_desc);
  if (m.cfg.use_graphs) {
    const auto key =
        std::make_tuple(role, int(graph_phase(plan.phase)), plan.rows_bucket, plan.suffix_bucket, plan.keys_bucket);
    auto it = m.graphs.find(key);
    if (it == m.graphs.end()) {
      it = m.capture(role, key);
      captured = true;
    }
    check(cudaGraphLaunch(it->second, R.stream), "graph launch");
    if (timing != nullptr) timing->graph_nodes = m.graph_nodes[key];
  } else {
    CallDesc h = m.host_call(role, plan.phase, plan.rows_bucket, plan.suffix_bucket);
    h.reserved[0] = plan.keys_bucket;
    m.enqueue_program(role, h, d_call);
  }
  cudaStream_t out = R.stream;  // where the call's PCM is complete
  if (m.post.run != nullptr) {
    CallDesc h = m.host_call(role, plan.phase, plan.rows_bucket, plan.suffix_bucket);
    h.reserved[0] = plan.keys_bucket;
    out = m.post.run(m.post.ctx, plan, h, d_call, role, R.stream, m.cfg.pipeline_post ? R.post_stream : R.stream);
  }
  if (plan.total_pcm > 0)
    check(cudaMemcpyAsync(slot->pcm, R.d_pcm, size_t(plan.total_pcm) * sizeof(float), cudaMemcpyDeviceToHost, out),
          "pcm D2H");
  if (timed) check(cudaEventRecord(slot->t1, out), "t1");
  check(cudaEventRecord(slot->done, out), "done");
  slot->busy = true;
  slot->timed = timed;
  slot->serial = ++m.serial;
  slot->total_pcm = plan.total_pcm;
  if (ticket != nullptr) *ticket = RenderTicket{role, slot_index, slot->serial};
  if (timing != nullptr) {
    timing->host_launch_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
    timing->graph_captured = captured;
  }
  return true;
}

bool RenderCaller::done(const RenderTicket& t) const {
  const CallSlot& s = impl_->roles[t.role].slots[t.call_slot];
  CtxScope cs(impl_->roles[t.role].ctx);
  if (s.serial != t.serial) return true;
  const cudaError_t e = cudaEventQuery(s.done);
  if (e == cudaErrorNotReady) return false;
  check(e, "render event");
  return true;
}

void RenderCaller::wait(const RenderTicket& t) const {
  const CallSlot& s = impl_->roles[t.role].slots[t.call_slot];
  CtxScope cs(impl_->roles[t.role].ctx);
  if (s.serial == t.serial) check(cudaEventSynchronize(s.done), "render wait");
}

const float* RenderCaller::pcm(const RenderTicket& t) const { return impl_->roles[t.role].slots[t.call_slot].pcm; }

float RenderCaller::gpu_ms(const RenderTicket& t) const {
  const CallSlot& s = impl_->roles[t.role].slots[t.call_slot];
  CtxScope cs(impl_->roles[t.role].ctx);
  if (!s.timed) return -1.f;
  float ms = -1.f;
  check(cudaEventElapsedTime(&ms, s.t0, s.t1), "elapsed");
  return ms;
}

void RenderCaller::release(const RenderTicket& t) {
  CallSlot& s = impl_->roles[t.role].slots[t.call_slot];
  if (s.serial == t.serial) s.busy = false;
}

CUstream_st* RenderCaller::stream(StreamRole role) const { return impl_->roles[role_index(role)].stream; }
void RenderCaller::set_post_stage(const RenderPostStage& post) { impl_->post = post; }
const float* RenderCaller::device_mel_out(StreamRole role) const { return impl_->roles[role_index(role)].d_mel; }
size_t RenderCaller::graphs_cached() const { return impl_->graphs.size(); }

uint64_t RenderCaller::workspace_need(Phase phase, int32_t rows_bucket, int32_t suffix_bucket, int32_t keys_bucket) const {
  const Impl& m = *impl_;
  CallDesc h = m.host_call(0, phase, rows_bucket, suffix_bucket);
  h.reserved[0] = keys_bucket;
  uint64_t off = 0;
  for (const RenderStage& st : m.program) {
    const uint64_t need = st.workspace_bytes ? st.workspace_bytes(st.ctx, h) : 0;
    off = align_up(off + need, 256);
  }
  return off;
}

uint64_t RenderCaller::workspace_bytes() const { return impl_->cfg.workspace_bytes; }
uint64_t RenderCaller::workspace_bytes(StreamRole role) const { return impl_->roles[role_index(role)].ws_bytes; }
int32_t RenderCaller::aux_max_rows() const { return impl_->cfg.aux_max_rows; }

int32_t RenderCaller::max_rows() const { return impl_->cfg.max_rows; }

bool RenderCaller::fits_staging(const CallPlan& plan, StreamRole role_kind) const {
  const Impl& m = *impl_;
  const int role = caller_role(role_kind);
  if (role < 0 || (role == 2 && plan.rows_bucket > m.cfg.aux_max_rows)) return false;
  return plan.rows_bucket <= m.cfg.max_rows && plan.total_tokens <= m.cfg.max_call_tokens &&
         int32_t(plan.page_table.size()) <= m.cfg.max_call_pages && plan.total_pcm <= m.cfg.max_call_pcm;
}

bool RenderCaller::has_graph(StreamRole role_kind, Phase phase, int32_t rows_bucket, int32_t suffix_bucket,
                             int32_t keys_bucket) const {
  const Impl& m = *impl_;
  if (!m.cfg.use_graphs) return true;  // eager mode: nothing to capture
  const int role = caller_role(role_kind);
  if (role < 0) return false;
  return m.graphs.count(std::make_tuple(role, int(graph_phase(phase)), rows_bucket, suffix_bucket, keys_bucket)) != 0;
}

bool RenderCaller::prewarm(StreamRole role_kind, Phase phase, int32_t rows_bucket, int32_t suffix_bucket,
                           int32_t keys_bucket) {
  Impl& m = *impl_;
  if (!m.cfg.use_graphs || rows_bucket > m.cfg.max_rows) return false;
  const int role = role_index(role_kind);
  if (m.roles[role].stream == nullptr || (role == 2 && rows_bucket > m.cfg.aux_max_rows)) return false;
  const auto key = std::make_tuple(role, int(graph_phase(phase)), rows_bucket, suffix_bucket, keys_bucket);
  if (m.graphs.count(key) != 0) return false;
  CtxScope cs(m.roles[role].ctx);
  m.capture(role, key);
  return true;
}

}  // namespace kt2n::render
