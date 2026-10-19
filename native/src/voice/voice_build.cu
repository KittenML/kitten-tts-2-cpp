// L11 native voice build: speaker embedding (pyannote/embedding x-vector SincNet), speech tokenizer
// (S3 v2 25 Hz, FSQ 6561 codes), CAMPPlus x-vector, prompt mel, resamplers, production checks.
// Production reference: kittenml-api ebcdd3f kitten_tts_09/runtime.py.
#include "voice_build.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "../core/json.h"
#include "clone_policy.h"
#include "clone_weights.h"
#include "vb_kernels.cuh"

namespace kt2n::voice {

// ------------------------------------------------------------------------------------------------
// Host-side helpers (production semantics)
// ------------------------------------------------------------------------------------------------
static constexpr int kSR = 24000;

AudioVerdict check_decoded(const float* x, size_t n) {
  if (n == 0) return AudioVerdict::kInvalidAudio;
  double dur = double(n) / kSR;
  if (!(dur >= 2.0 && dur <= 30.0)) return AudioVerdict::kInvalidDuration;
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(x[i])) return AudioVerdict::kInvalidAudio;
  float peak = 0.f;
  double ss = 0.0;
  size_t clipped = 0;
  for (size_t i = 0; i < n; ++i) {
    float a = std::fabs(x[i]);
    peak = std::max(peak, a);
    ss += double(x[i]) * double(x[i]);
    clipped += a >= 0.999f;
  }
  double rms = std::sqrt(ss / double(n));
  double clip_frac = double(clipped) / double(n);
  if (peak < 0.005f || rms < 0.001 || clip_frac > 0.05) return AudioVerdict::kInvalidSignal;
  return AudioVerdict::kOk;
}

const char* verdict_code(AudioVerdict v) {
  switch (v) {
    case AudioVerdict::kOk: return "ok";
    case AudioVerdict::kInvalidAudio: return "invalid_audio";
    case AudioVerdict::kInvalidDuration: return "invalid_audio_duration";
    case AudioVerdict::kInvalidSignal: return "invalid_audio_signal";
  }
  return "invalid_audio";
}

void canonical_pcm16(const float* x, size_t n, int16_t* pcm, float* wav24) {
  for (size_t i = 0; i < n; ++i) {
    float c = std::min(1.0f, std::max(-1.0f, x[i]));
    // numpy: np.rint(np.clip(x) * 32767.0) on float32 -> float32 multiply, round half to even
    float v = std::nearbyint(c * 32767.0f);
    pcm[i] = static_cast<int16_t>(v);
    wav24[i] = static_cast<float>(pcm[i]) / 32767.0f;
  }
}

std::vector<float> decoder_audio(const float* wav24, size_t n) {
  std::vector<float> out;
  const size_t minimum = 6 * kSR;
  size_t reps = n < minimum ? (minimum + n - 1) / n : 1;
  size_t total = std::min<size_t>(n * reps, 10 * kSR);
  out.resize(total);
  for (size_t i = 0; i < total; ++i) out[i] = wav24[i % n];
  return out;
}

// ------------------------------------------------------------------------------------------------
// safetensors -> device
// ------------------------------------------------------------------------------------------------
class DeviceTensors {
 public:
  struct T {
    float* d = nullptr;
    bool owned = true;  // false: points into a block of blocks_
    std::vector<int64_t> shape;
    std::vector<float> host;  // kept only for small tensors we read on the host
  };
  void load(const std::string& path, const std::string& prefix_filter, bool keep_host_small = true) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    std::string header(hlen, '\0');
    f.read(header.data(), (std::streamsize)hlen);
    auto meta = json::parse(header);
    const uint64_t base = 8 + hlen;
    for (const auto& [name, v] : meta.members()) {
      if (name == "__metadata__") {
        for (const auto& [k, mv] : v.members()) metadata_[k] = mv.as_string();
        continue;
      }
      if (!prefix_filter.empty() && name.rfind(prefix_filter, 0) != 0) continue;
      const std::string& dt = v.get("dtype")->as_string();
      if (dt != "F32") continue;  // ints (num_batches_tracked) are not needed
      T t;
      int64_t numel = 1;
      for (const auto& s : v.get("shape")->items()) {
        t.shape.push_back(s.as_int());
        numel *= s.as_int();
      }
      const auto& off = v.get("data_offsets")->items();
      uint64_t b0 = (uint64_t)off[0].as_int(), b1 = (uint64_t)off[1].as_int();
      if (b1 - b0 != (uint64_t)numel * 4) throw std::runtime_error("bad tensor size " + name);
      std::vector<float> host((size_t)numel);
      f.seekg((std::streamoff)(base + b0));
      f.read(reinterpret_cast<char*>(host.data()), (std::streamsize)(b1 - b0));
      if (!f) throw std::runtime_error("truncated tensor " + name);
      check(cudaMalloc(&t.d, (size_t)numel * 4), "cudaMalloc weights");
      // Own the allocation before the upload, so initialization failures free it.
      map_[name] = std::move(t);
      auto& stored = map_.at(name);
      check(cudaMemcpy(stored.d, host.data(), (size_t)numel * 4, cudaMemcpyHostToDevice), "upload");
      if (keep_host_small && numel <= 65536) stored.host = std::move(host);
      bytes_ += (size_t)numel * 4;
    }
  }
  // s3gen.kt2n clone groups (clone_weights.h): one device allocation and one copy when the segments are
  // contiguous (they are: the four clone groups close the file), else one copy per tensor into that allocation.
  void load_kt2n(const CloneWeightsFile& f, bool keep_host_small = true) {
    const uint64_t span = f.span_end() - f.span_begin();
    const bool one_copy = span <= f.tensor_bytes() + f.tensor_bytes() / 4;
    const size_t alloc = one_copy ? (size_t)span : (size_t)f.tensor_bytes() + 256 * f.tensors().size();
    char* block = nullptr;
    check(cudaMalloc(&block, alloc), "cudaMalloc clone weights");
    blocks_.push_back(block);
    if (one_copy)
      check(cudaMemcpy(block, f.base() + f.span_begin(), (size_t)span, cudaMemcpyHostToDevice), "upload clone weights");
    size_t cursor = 0;
    for (const CloneTensor& ct : f.tensors()) {
      T t;
      t.shape = ct.shape;
      if (one_copy) {
        t.d = reinterpret_cast<float*>(block + (ct.abs_off - f.span_begin()));
      } else {
        t.d = reinterpret_cast<float*>(block + cursor);
        check(cudaMemcpy(t.d, f.base() + ct.abs_off, (size_t)ct.len, cudaMemcpyHostToDevice), "upload");
        cursor += ((size_t)ct.len + 255) & ~size_t(255);
      }
      t.owned = false;
      const size_t numel = (size_t)ct.len / 4;
      if (keep_host_small && numel <= 65536) {
        t.host.resize(numel);
        std::memcpy(t.host.data(), f.base() + ct.abs_off, (size_t)ct.len);
      }
      bytes_ += (size_t)ct.len;
      map_[ct.name] = std::move(t);
    }
  }
  const T& at(const std::string& name) const {
    auto it = map_.find(name);
    if (it == map_.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
  }
  const float* d(const std::string& name) const { return at(name).d; }
  bool has(const std::string& name) const { return map_.count(name) != 0; }
  const std::map<std::string, T>& all() const { return map_; }
  const std::map<std::string, std::string>& metadata() const { return metadata_; }
  size_t bytes() const { return bytes_; }
  ~DeviceTensors() {
    for (auto& [k, t] : map_)
      if (t.owned) cudaFree(t.d);
    for (void* b : blocks_) cudaFree(b);
  }

 private:
  std::map<std::string, T> map_;
  std::vector<void*> blocks_;  // kt2n layout: the one allocation every clone tensor points into
  std::map<std::string, std::string> metadata_;
  size_t bytes_ = 0;
};

// ------------------------------------------------------------------------------------------------
float* VoiceBuilder::Arena::f(size_t n) {
  size_t bytes = ((n * 4) + 255) & ~size_t(255);
  if (used + bytes > cap) throw std::runtime_error("voice build arena exhausted");
  float* p = reinterpret_cast<float*>(base + used);
  used += bytes;
  return p;
}
int32_t* VoiceBuilder::Arena::i(size_t n) { return reinterpret_cast<int32_t*>(f(n)); }

VoiceBuilder::VoiceBuilder(const std::string& s3gen, const std::string& consts, const std::string& pyannote,
                           size_t arena_bytes) {
  const auto t0 = std::chrono::steady_clock::now();
  try {
    w_ = new DeviceTensors();
    w_->load(s3gen, "tokenizer.");
    w_->load(s3gen, "speaker_encoder.");
    w_->load(consts, "");
    w_->load(pyannote, "");
    weights_load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    init_device(arena_bytes);
  } catch (...) {
    release_device();
    throw;
  }
}

VoiceBuilder::VoiceBuilder(const S3genKt2n& src, size_t arena_bytes) {
  const auto t0 = std::chrono::steady_clock::now();
  try {
    w_ = new DeviceTensors();
    {
      CloneWeightsOptions opt;
      opt.verify_sha256 = src.verify_sha256;
      CloneWeightsFile f = CloneWeightsFile::open(src.path, opt);
      w_->load_kt2n(f);
      weights_verify_ms_ = f.verify_ms();
    }  // unmapped here: the builder keeps only device copies (+ host copies of the small tensors)
    weights_load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    init_device(arena_bytes);
  } catch (...) {
    release_device();
    throw;
  }
}

size_t VoiceBuilder::weight_bytes() const { return w_ ? w_->bytes() : 0; }

std::vector<VoiceBuilder::WeightView> VoiceBuilder::weights() const {
  std::vector<WeightView> v;
  if (!w_) return v;
  for (const auto& [name, t] : w_->all()) v.push_back({name, t.d, t.shape});
  return v;
}

void VoiceBuilder::init_device(size_t arena_bytes) {
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
  check(cudaEventCreate(&ev0_), "event");
  check(cudaEventCreate(&ev1_), "event");
  check(cudaMalloc(&arena_.base, arena_bytes), "arena");
  arena_.cap = arena_bytes;
  int dev = 0, sms = 0;
  check(cudaGetDevice(&dev), "device");
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "sm count");
  const size_t ws_floats = size_t(16) << 20;  // 64 MB split-K workspace per stream
  check(cudaMalloc(&d_splitk_, ws_floats * sizeof(float)), "splitk ws");
  sgemm_set_workspace(d_splitk_, ws_floats, sms);
  sgemm_set_stream_workspace(stream_, d_splitk_, ws_floats);
  // side stream + its own arena and workspace: the decoder reference's tokenizer branch runs there,
  // concurrently with fbank + CAMPPlus (both only read the 16 kHz audio)
  check(cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking), "side stream");
  check(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming), "event");
  check(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming), "event");
  check(cudaMalloc(&d_splitk2_, ws_floats * sizeof(float)), "splitk ws2");
  sgemm_set_stream_workspace(side_, d_splitk2_, ws_floats);
  arena2_.cap = size_t(256) << 20;
  check(cudaMalloc(&arena2_.base, arena2_.cap), "arena2");
  check(cudaStreamCreateWithFlags(&lm_, cudaStreamNonBlocking), "lm stream");
  check(cudaEventCreateWithFlags(&lm_done_, cudaEventDisableTiming), "event");
  check(cudaMalloc(&d_splitk3_, ws_floats * sizeof(float)), "splitk ws3");
  sgemm_set_stream_workspace(lm_, d_splitk3_, ws_floats);
  arena3_.cap = size_t(256) << 20;
  check(cudaMalloc(&arena3_.base, arena3_.cap), "arena3");
  check(cudaMallocHost(&h_lm_codes_, 4096 * sizeof(int32_t)), "pinned codes");
  check(cudaMalloc(&d_basis24_, sizeof(float) * 1920 * 1922), "basis");
  check(cudaMalloc(&d_basis400_, sizeof(float) * 400 * 402), "basis");
  check(cudaMalloc(&d_basis512_, sizeof(float) * 512 * 514), "basis");
  make_dft_basis(stream_, 1920, 961, d_basis24_);
  make_dft_basis(stream_, 400, 201, d_basis400_);
  make_dft_basis(stream_, 512, 257, d_basis512_);
  // rotary frequencies, fp32 as torch computes them: 1 / 10000 ** (2d/64)
  float fr[32];
  for (int d = 0; d < 32; ++d) {
    float e = (float)(2 * d) / 64.0f;
    fr[d] = 1.0f / std::pow(10000.0f, e);
  }
  check(cudaMalloc(&d_freqs_, sizeof(fr)), "freqs");
  check(cudaMemcpy(d_freqs_, fr, sizeof(fr), cudaMemcpyHostToDevice), "freqs");
  // The three-file layout carries a copy of the catalogue bank (live 48-voice npz) with the pyannote export;
  // s3gen.kt2n does not (voice data lives in voices.kt2n): set_bank() / VoiceService::set_catalogue() fill it.
  if (w_->has("bank.embeddings")) {
    const auto& b = w_->at("bank.embeddings");
    bank_ = b.host.empty() ? std::vector<float>() : b.host;
    if (bank_.empty()) {
      bank_.resize((size_t)b.shape[0] * b.shape[1]);
      check(cudaMemcpy(bank_.data(), b.d, bank_.size() * 4, cudaMemcpyDeviceToHost), "bank");
    }
    auto it = w_->metadata().find("voice_ids");
    if (it != w_->metadata().end()) {
      auto ids = json::parse(it->second);
      for (const auto& v : ids.items()) bank_ids_.push_back(v.as_string());
    }
  }
  check(cudaStreamSynchronize(stream_), "init sync");
}

VoiceBuilder::~VoiceBuilder() { release_device(); }

void VoiceBuilder::release_device() noexcept {
  for (cudaStream_t st : {stream_, side_, lm_})
    if (st) sgemm_clear_stream_workspace(st);
  for (auto& [k, g] : dec_graphs_) cudaGraphExecDestroy(g.exec);
  for (auto& [k, p] : derived_) cudaFree(p);
  cudaFree(d_basis24_);
  cudaFree(d_basis400_);
  cudaFree(d_basis512_);
  cudaFree(d_freqs_);
  cudaFree(arena_.base);
  cudaFree(d_splitk_);
  cudaFree(d_splitk2_);
  cudaFree(d_splitk3_);
  cudaFree(arena3_.base);
  if (h_lm_codes_) cudaFreeHost(h_lm_codes_);
  if (lm_done_) cudaEventDestroy(lm_done_);
  if (lm_) cudaStreamDestroy(lm_);
  cudaFree(arena2_.base);
  if (fork_) cudaEventDestroy(fork_);
  if (join_) cudaEventDestroy(join_);
  if (side_) cudaStreamDestroy(side_);
  if (ev0_) cudaEventDestroy(ev0_);
  if (ev1_) cudaEventDestroy(ev1_);
  if (stream_) cudaStreamDestroy(stream_);
  delete w_;
}

void VoiceBuilder::ev_start() { check(cudaEventRecord(ev0_, stream_), "ev"); }
void VoiceBuilder::ev_stop(const char* stage) {
  check(cudaEventRecord(ev1_, stream_), "ev");
  check(cudaEventSynchronize(ev1_), "ev sync");
  float ms = 0.f;
  check(cudaEventElapsedTime(&ms, ev0_, ev1_), "ev time");
  times.ms[stage] = ms;
}

void VoiceBuilder::tap(const char* name, const float* d, size_t n) {
  if (!keep_taps) return;
  std::vector<float> h(n);
  check(cudaMemcpyAsync(h.data(), d, n * 4, cudaMemcpyDeviceToHost, stream_), "tap");
  check(cudaStreamSynchronize(stream_), "tap sync");
  taps[name] = std::move(h);
}
void VoiceBuilder::tap_i(const char* name, const int32_t* d, size_t n) {
  if (!keep_taps) return;
  std::vector<int32_t> hi(n);
  check(cudaMemcpyAsync(hi.data(), d, n * 4, cudaMemcpyDeviceToHost, stream_), "tap");
  check(cudaStreamSynchronize(stream_), "tap sync");
  taps[name] = std::vector<float>(hi.begin(), hi.end());
}

// BatchNorm (eval) as a per-channel affine: scale = w/sqrt(var+eps), shift = b - mean*scale (fp64 on host).
void VoiceBuilder::bn(const std::string& p, std::vector<float>& scale, std::vector<float>& shift, bool affine) {
  const auto& mean = w_->at(p + ".running_mean");
  const auto& var = w_->at(p + ".running_var");
  size_t C = (size_t)mean.shape[0];
  std::vector<float> m(C), v(C), g(C, 1.f), b(C, 0.f);
  check(cudaMemcpy(m.data(), mean.d, C * 4, cudaMemcpyDeviceToHost), "bn");
  check(cudaMemcpy(v.data(), var.d, C * 4, cudaMemcpyDeviceToHost), "bn");
  if (affine) {
    check(cudaMemcpy(g.data(), w_->d(p + ".weight"), C * 4, cudaMemcpyDeviceToHost), "bn");
    check(cudaMemcpy(b.data(), w_->d(p + ".bias"), C * 4, cudaMemcpyDeviceToHost), "bn");
  }
  scale.resize(C);
  shift.resize(C);
  for (size_t c = 0; c < C; ++c) {
    double s = double(g[c]) / std::sqrt(double(v[c]) + 1e-5);
    scale[c] = (float)s;
    shift[c] = (float)(double(b[c]) - double(m[c]) * s);
  }
}
const float* VoiceBuilder::bn_scale(const std::string& p, bool affine) {
  auto key = p + "#scale";
  auto it = derived_.find(key);
  if (it != derived_.end()) return it->second;
  std::vector<float> sc, sh;
  bn(p, sc, sh, affine);
  float *ds, *dh;
  check(cudaMalloc(&ds, sc.size() * 4), "bn");
  check(cudaMalloc(&dh, sh.size() * 4), "bn");
  check(cudaMemcpy(ds, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "bn");
  check(cudaMemcpy(dh, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice), "bn");
  derived_[key] = ds;
  derived_[p + "#shift"] = dh;
  return ds;
}
const float* VoiceBuilder::bn_shift(const std::string& p, bool affine) {
  bn_scale(p, affine);
  return derived_.at(p + "#shift");
}

void VoiceBuilder::conv1d(const float* x, int cin, int T, const std::string& w, int cout, int K, int stride, int dil,
                          int pad, float* y, int t_out, const Epilogue& ep) {
  const float* W = w_->d(w);
  if (K == 1 && stride == 1 && pad == 0) {
    sgemm(stream_, cout, t_out, cin, W, cin, x, T, false, y, t_out, ep);
    return;
  }
  size_t mark = arena_.used;
  float* col = arena_.f((size_t)cin * K * t_out);
  im2col1d(stream_, x, cin, T, K, stride, dil, pad, t_out, col);
  sgemm(stream_, cout, t_out, cin * K, W, cin * K, col, t_out, false, y, t_out, ep);
  arena_.used = mark;
}

void VoiceBuilder::conv2d(const float* x, int cin, int H, int W, const std::string& w, int cout, int sh, int ph,
                          float* y, int Ho, const Epilogue& ep) {
  // 3x3 (or 1x1 when ph == 0 and the weight is 1x1), stride (sh, 1), padding (ph, ph)
  const auto& wt = w_->at(w);
  int kh = (int)wt.shape[2], kw = (int)wt.shape[3];
  int pw = ph;
  int Wo = W;
  size_t mark = arena_.used;
  float* col = arena_.f((size_t)cin * kh * kw * Ho * Wo);
  im2col2d(stream_, x, cin, H, W, kh, kw, sh, 1, ph, pw, Ho, Wo, col);
  sgemm(stream_, cout, Ho * Wo, cin * kh * kw, wt.d, cin * kh * kw, col, Ho * Wo, false, y, Ho * Wo, ep);
  arena_.used = mark;
}

// ------------------------------------------------------------------------------------------------
// Front ends
// ------------------------------------------------------------------------------------------------
std::vector<float> VoiceBuilder::resample_poly16(const float* wav24, size_t n) {
  arena_.reset();
  int n_out = (int)((2 * n + 2) / 3);
  float* dx = arena_.f(n);
  float* dy = arena_.f(n_out);
  check(cudaMemcpyAsync(dx, wav24, n * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  ev_start();
  resample_poly_2_3(stream_, dx, (int)n, w_->d("poly_2_3_taps"), dy, n_out);
  ev_stop("resample_poly");
  std::vector<float> out(n_out);
  check(cudaMemcpyAsync(out.data(), dy, (size_t)n_out * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  check(cudaStreamSynchronize(stream_), "sync");
  return out;
}

float* VoiceBuilder::prompt_mel(const float* d_wav24, int n, int* frames) {
  int F = (n + 1440 - 1920) / 480 + 1;
  float* fr = arena_.f((size_t)F * 1920);
  make_frames(stream_, d_wav24, n, F, 480, 720, true, 1920, w_->d("mel24_window"), 1920, false, fr);
  float* spec = arena_.f((size_t)F * 1922);
  sgemm(stream_, F, 1922, 1920, fr, 1920, d_basis24_, 1922, false, spec, 1922);
  float* mag = arena_.f((size_t)F * 961);
  spec_mag(stream_, spec, F, 961, 0, mag);
  float* mel = arena_.f((size_t)F * 80);
  sgemm(stream_, F, 80, 961, mag, 961, w_->d("mel24_basis"), 961, true, mel, 80);
  log_floor(stream_, mel, (long long)F * 80, 0);
  *frames = F;
  return mel;
}

float* VoiceBuilder::logmel_s3(const float* d_wav16, int n, int* frames) {
  int F = n / 160;  // torch.stft(center=True) gives n/160 + 1 frames; the last is dropped
  float* fr = arena_.f((size_t)F * 400);
  make_frames(stream_, d_wav16, n, F, 160, 200, true, 400, w_->d("s3_window"), 400, false, fr);
  float* spec = arena_.f((size_t)F * 402);
  sgemm(stream_, F, 402, 400, fr, 400, d_basis400_, 402, false, spec, 402);
  float* pw = arena_.f((size_t)F * 201);
  spec_mag(stream_, spec, F, 201, 1, pw);
  float* mel_t = arena_.f((size_t)F * 128);
  sgemm(stream_, F, 128, 201, pw, 201, w_->d("s3_mel_filters"), 201, true, mel_t, 128);
  log_floor(stream_, mel_t, (long long)F * 128, 2);
  float* scratch = arena_.f(128);
  s3_logmel_finish(stream_, mel_t, (long long)F * 128, scratch);
  float* mel = arena_.f((size_t)F * 128);
  transpose(stream_, mel_t, mel, F, 128);  // -> [128][F]
  *frames = F;
  return mel;
}

float* VoiceBuilder::kaldi_fbank(const float* d_wav16, int n, int* frames) {
  int F = 1 + (n - 400) / 160;
  float* fr = arena_.f((size_t)F * 512);
  make_frames(stream_, d_wav16, n, F, 160, 0, false, 400, w_->d("kaldi_povey_400"), 512, true, fr);
  float* spec = arena_.f((size_t)F * 514);
  sgemm(stream_, F, 514, 512, fr, 512, d_basis512_, 514, false, spec, 514);
  float* pw = arena_.f((size_t)F * 257);
  spec_mag(stream_, spec, F, 257, 1, pw);
  float* fb = arena_.f((size_t)F * 80);
  sgemm(stream_, F, 80, 257, pw, 257, w_->d("kaldi_mel_banks"), 257, true, fb, 80);
  log_floor(stream_, fb, (long long)F * 80, 1);
  tap("fbank_raw", fb, (size_t)F * 80);
  sub_col_mean(stream_, fb, F, 80);
  *frames = F;
  return fb;
}

// ------------------------------------------------------------------------------------------------
// Speech tokenizer (S3 v2): conv stem + 6 FSMN attention blocks + FSQ
// ------------------------------------------------------------------------------------------------
int VoiceBuilder::tokenizer(const float* d_logmel, int F, int32_t* d_codes) {
  const std::string P = "tokenizer.encoder.";
  const int D = 1280, H = 20;
  int T1 = (F + 2 - 3) / 2 + 1;
  int T2 = (T1 + 2 - 3) / 2 + 1;
  float* y1 = arena_.f((size_t)D * T1);
  Epilogue e1;
  e1.bias = w_->d(P + "conv1.bias");
  e1.per_row = true;
  e1.act = Act::kGelu;
  conv1d(d_logmel, 128, F, P + "conv1.weight", D, 3, 2, 1, 1, y1, T1, e1);
  float* y2 = arena_.f((size_t)D * T2);
  Epilogue e2 = e1;
  e2.bias = w_->d(P + "conv2.bias");
  conv1d(y1, D, T1, P + "conv2.weight", D, 3, 2, 1, 1, y2, T2, e2);
  float* x = arena_.f((size_t)T2 * D);
  transpose(stream_, y2, x, D, T2);  // [T2][D]
  float* h = arena_.f((size_t)T2 * D);
  float* q = arena_.f((size_t)T2 * D);
  float* k = arena_.f((size_t)T2 * D);
  float* v = arena_.f((size_t)T2 * D);
  float* fs = arena_.f((size_t)T2 * D);
  float* att = arena_.f((size_t)T2 * D);
  float* mlp = arena_.f((size_t)T2 * 4 * D);
  float* scores = arena_.f((size_t)H * T2 * T2);
  const float scale = 0.35355339059327373f;  // 64 ** -0.25
  for (int b = 0; b < 6; ++b) {
    std::string B = P + "blocks." + std::to_string(b) + ".";
    layer_norm(stream_, x, h, T2, D, w_->d(B + "attn_ln.weight"), w_->d(B + "attn_ln.bias"), 1e-5f);
    Epilogue eq;
    eq.bias = w_->d(B + "attn.query.bias");
    sgemm(stream_, T2, D, D, h, D, w_->d(B + "attn.query.weight"), D, true, q, D, eq);
    sgemm(stream_, T2, D, D, h, D, w_->d(B + "attn.key.weight"), D, true, k, D);
    Epilogue ev;
    ev.bias = w_->d(B + "attn.value.bias");
    sgemm(stream_, T2, D, D, h, D, w_->d(B + "attn.value.weight"), D, true, v, D, ev);
    fsmn(stream_, v, w_->d(B + "attn.fsmn_block.weight"), fs, T2, D, 31);
    rotary(stream_, q, T2, H, d_freqs_);
    rotary(stream_, k, T2, H, d_freqs_);
    scale_inplace(stream_, q, (long long)T2 * D, scale);
    scale_inplace(stream_, k, (long long)T2 * D, scale);
    attention_gemm(stream_, q, k, v, att, T2, H, scores);
    eltwise_add(stream_, x, fs, x, (long long)T2 * D);
    Epilogue eo;
    eo.bias = w_->d(B + "attn.out.bias");
    eo.residual = x;
    sgemm(stream_, T2, D, D, att, D, w_->d(B + "attn.out.weight"), D, true, x, D, eo);
    layer_norm(stream_, x, h, T2, D, w_->d(B + "mlp_ln.weight"), w_->d(B + "mlp_ln.bias"), 1e-5f);
    Epilogue e3;
    e3.bias = w_->d(B + "mlp.0.bias");
    e3.act = Act::kGelu;
    sgemm(stream_, T2, 4 * D, D, h, D, w_->d(B + "mlp.0.weight"), D, true, mlp, 4 * D, e3);
    Epilogue e4;
    e4.bias = w_->d(B + "mlp.2.bias");
    e4.residual = x;
    sgemm(stream_, T2, D, 4 * D, mlp, 4 * D, w_->d(B + "mlp.2.weight"), 4 * D, true, x, D, e4);
  }
  tap("tok_hidden", x, (size_t)T2 * D);
  float* proj = arena_.f((size_t)T2 * 8);
  Epilogue ep;
  ep.bias = w_->d("tokenizer.quantizer._codebook.project_down.bias");
  sgemm(stream_, T2, 8, D, x, D, w_->d("tokenizer.quantizer._codebook.project_down.weight"), D, true, proj, 8, ep);
  tap("tok_proj", proj, (size_t)T2 * 8);
  fsq_codes(stream_, proj, T2, d_codes);
  return T2;
}

std::vector<int32_t> VoiceBuilder::s3_tokens(const float* wav16, size_t n) {
  arena_.reset();
  float* dx = arena_.f(n);
  check(cudaMemcpyAsync(dx, wav16, n * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  ev_start();
  int F = 0;
  float* mel = logmel_s3(dx, (int)n, &F);
  tap("lm_logmel", mel, (size_t)128 * F);
  int32_t* codes = arena_.i(F / 4 + 4);
  int T = tokenizer(mel, F, codes);
  std::vector<int32_t> out(T);
  check(cudaMemcpyAsync(out.data(), codes, (size_t)T * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  ev_stop("s3_tokens");
  return out;
}

// ------------------------------------------------------------------------------------------------
// CAMPPlus x-vector
// ------------------------------------------------------------------------------------------------
void VoiceBuilder::campplus(const float* d_fbank, int T, float* d_out192) {
  const std::string P = "speaker_encoder.";
  // fbank [T][80] -> x [80][T] (H = 80 mel bins, W = T frames)
  float* x = arena_.f((size_t)80 * T);
  transpose(stream_, d_fbank, x, T, 80);
  auto bn_ep = [&](const std::string& p, Act act) {
    Epilogue e;
    e.scale = bn_scale(p);
    e.shift = bn_shift(p);
    e.per_row = true;
    e.act = act;
    return e;
  };
  // FCM head
  int Hc = 80;
  float* a = arena_.f((size_t)32 * Hc * T);
  conv2d(x, 1, Hc, T, P + "head.conv1.weight", 32, 1, 1, a, Hc, bn_ep(P + "head.bn1", Act::kRelu));
  auto res_block = [&](const std::string& p, float* in, int H, int stride) -> std::pair<float*, int> {
    int Ho = (H + 2 - 3) / stride + 1;
    float* t1 = arena_.f((size_t)32 * Ho * T);
    conv2d(in, 32, H, T, p + ".conv1.weight", 32, stride, 1, t1, Ho, bn_ep(p + ".bn1", Act::kRelu));
    float* sc = in;
    if (stride != 1) {
      sc = arena_.f((size_t)32 * Ho * T);
      conv2d(in, 32, H, T, p + ".shortcut.0.weight", 32, stride, 0, sc, Ho, bn_ep(p + ".shortcut.1", Act::kNone));
    }
    float* out = arena_.f((size_t)32 * Ho * T);
    Epilogue e = bn_ep(p + ".bn2", Act::kNone);
    e.residual = sc;
    conv2d(t1, 32, Ho, T, p + ".conv2.weight", 32, 1, 1, out, Ho, e);
    eltwise_act(stream_, out, out, (long long)32 * Ho * T, Act::kRelu);
    return {out, Ho};
  };
  auto r = res_block(P + "head.layer1.0", a, Hc, 2);
  r = res_block(P + "head.layer1.1", r.first, r.second, 1);
  r = res_block(P + "head.layer2.0", r.first, r.second, 2);
  r = res_block(P + "head.layer2.1", r.first, r.second, 1);
  int H3 = (r.second + 2 - 3) / 2 + 1;  // 10
  float* head = arena_.f((size_t)32 * H3 * T);
  conv2d(r.first, 32, r.second, T, P + "head.conv2.weight", 32, 2, 1, head, H3, bn_ep(P + "head.bn2", Act::kRelu));
  tap("cam_head", head, (size_t)32 * H3 * T);
  // TDNN 320 -> 128, k5 s2 p2
  int C = 32 * H3;
  int T2 = (T + 4 - 5) / 2 + 1;
  const int maxC = 1024;
  float* buf = arena_.f((size_t)maxC * T2);
  conv1d(head, C, T, P + "xvector.tdnn.linear.weight", 128, 5, 2, 1, 2, buf, T2,
         bn_ep(P + "xvector.tdnn.nonlinear.batchnorm", Act::kRelu));
  C = 128;
  float* pre = arena_.f((size_t)maxC * T2);
  float* hb = arena_.f((size_t)128 * T2);
  float* y = arena_.f((size_t)32 * T2);
  float* ctx = arena_.f((size_t)128 * T2);
  float* m = arena_.f((size_t)32 * T2);
  const int nseg = (T2 + 99) / 100;
  const int layers[3] = {12, 24, 16};
  const int dils[3] = {1, 2, 2};
  for (int bi = 0; bi < 3; ++bi) {
    for (int li = 0; li < layers[bi]; ++li) {
      std::string L = P + "xvector.block" + std::to_string(bi + 1) + ".tdnnd" + std::to_string(li + 1) + ".";
      std::string n1 = L + "nonlinear1.batchnorm";
      affine_act(stream_, buf, pre, C, T2, T2, T2, bn_scale(n1), bn_shift(n1), Act::kRelu);
      conv1d(pre, C, T2, L + "linear1.weight", 128, 1, 1, 1, 0, hb, T2, bn_ep(L + "nonlinear2.batchnorm", Act::kRelu));
      int d = dils[bi];
      conv1d(hb, 128, T2, L + "cam_layer.linear_local.weight", 32, 3, 1, d, d, y, T2, Epilogue{});
      // The CAM context (global mean + 100-frame segment mean) is constant inside each segment, so the
      // two 1x1 convs and the sigmoid run once per segment (nseg columns) instead of once per frame.
      cam_context_seg(stream_, hb, 128, T2, 100, ctx);
      cam_mlp(stream_, ctx, nseg, w_->d(L + "cam_layer.linear1.weight"), w_->d(L + "cam_layer.linear1.bias"),
              w_->d(L + "cam_layer.linear2.weight"), w_->d(L + "cam_layer.linear2.bias"), m);
      mul_seg(stream_, y, m, buf + (size_t)C * T2, 32, T2, 100);
      C += 32;
    }
    std::string Tn = P + "xvector.transit" + std::to_string(bi + 1) + ".";
    affine_act(stream_, buf, pre, C, T2, T2, T2, bn_scale(Tn + "nonlinear.batchnorm"),
               bn_shift(Tn + "nonlinear.batchnorm"), Act::kRelu);
    conv1d(pre, C, T2, Tn + "linear.weight", C / 2, 1, 1, 1, 0, buf, T2, Epilogue{});
    C /= 2;
  }
  std::string O = P + "xvector.out_nonlinear.batchnorm";
  affine_act(stream_, buf, pre, C, T2, T2, T2, bn_scale(O), bn_shift(O), Act::kRelu);
  float* st = arena_.f((size_t)2 * C);
  stats_pool(stream_, pre, C, T2, T2, st);
  Epilogue ed;
  ed.scale = bn_scale(P + "xvector.dense.nonlinear.batchnorm", false);
  ed.shift = bn_shift(P + "xvector.dense.nonlinear.batchnorm", false);
  ed.per_row = true;
  sgemm(stream_, 192, 1, 2 * C, w_->d(P + "xvector.dense.linear.weight"), 2 * C, st, 1, false, d_out192, 1, ed);
}

// ------------------------------------------------------------------------------------------------
// Decoder reference: tile/crop, prompt mel, sinc resample to 16 kHz, fbank + CAMPPlus, tokens, 2:1 align
// ------------------------------------------------------------------------------------------------
void VoiceBuilder::decoder_ref_enqueue(float* d24, int nd, DecOut& o) {
  o.mel = prompt_mel(d24, nd, &o.F);
  int n16 = (2 * nd + 2) / 3;
  float* d16 = arena_.f(n16);
  resample_sinc_3_2(stream_, d24, nd, w_->d("ta_24_16_kernel_dev"), d16, n16);
  tap("wav16_ta", d16, n16);
  const bool overlap = overlap_branches && !keep_taps;
  if (overlap) {
    // fork: the tokenizer branch on side_ with arena2_ (swap so the helpers use them), joined below
    check(cudaEventRecord(fork_, stream_), "fork");
    check(cudaStreamWaitEvent(side_, fork_, 0), "fork wait");
    std::swap(stream_, side_);
    std::swap(arena_, arena2_);
    arena_.reset();
  }
  int Fs = 0;
  float* lm = logmel_s3(d16, n16, &Fs);
  o.codes = arena_.i(Fs / 4 + 4);
  o.Tt = tokenizer(lm, Fs, o.codes);
  if (overlap) {
    check(cudaEventRecord(join_, stream_), "join");
    std::swap(stream_, side_);
    std::swap(arena_, arena2_);
  }
  int Tf = 0;
  float* fb = kaldi_fbank(d16, n16, &Tf);
  tap("fbank", fb, (size_t)Tf * 80);
  o.xv = arena_.f(192);
  campplus(fb, Tf, o.xv);
  if (overlap) check(cudaStreamWaitEvent(stream_, join_, 0), "join wait");
}

DecoderRef VoiceBuilder::decoder_ref(const float* wav24, size_t n) {
  arena_.reset();
  std::vector<float> dec = decoder_audio(wav24, n);
  int nd = (int)dec.size();
  float* d24 = arena_.f(nd);
  check(cudaMemcpyAsync(d24, dec.data(), (size_t)nd * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  DecoderRef out;
  ev_start();
  DecOut o;
  // CUDA graph for the common length: every upload >= 10 s is cropped to exactly 10 s (240000 samples).
  // The first call at a length runs eagerly (fills the BN caches); the next one captures, later ones replay.
  const bool graphable = use_graphs && !keep_taps && nd == 10 * kSR;
  auto it = dec_graphs_.find(nd);
  if (graphable && it != dec_graphs_.end()) {
    o = it->second.out;
    check(cudaGraphLaunch(it->second.exec, stream_), "graph launch");
  } else if (graphable && dec_warm_.count(nd)) {
    cudaGraph_t g = nullptr;
    check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture");
    decoder_ref_enqueue(d24, nd, o);
    check(cudaStreamEndCapture(stream_, &g), "end capture");
    cudaGraphExec_t ex = nullptr;
    check(cudaGraphInstantiate(&ex, g, 0), "instantiate");
    cudaGraphDestroy(g);
    dec_graphs_[nd] = DecGraphEntry{ex, o};
    check(cudaGraphLaunch(ex, stream_), "graph launch");
  } else {
    decoder_ref_enqueue(d24, nd, o);
    dec_warm_.insert(nd);
  }
  int F = o.F, Tt = o.Tt;
  std::vector<float> mel_h((size_t)F * 80);
  std::vector<int32_t> tok_h(Tt);
  out.xvector.resize(192);
  check(cudaMemcpyAsync(mel_h.data(), o.mel, mel_h.size() * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  check(cudaMemcpyAsync(tok_h.data(), o.codes, (size_t)Tt * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  check(cudaMemcpyAsync(out.xvector.data(), o.xv, 192 * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  ev_stop("decoder_ref");
  out.raw_tokens = Tt;
  out.raw_frames = F;
  if (keep_taps) taps["prompt_feat_raw"] = mel_h;
  // embed_ref truncation, then production's exact 2:1 alignment (runtime.py:8348)
  if (F != 2 * Tt) tok_h.resize(std::min<size_t>(tok_h.size(), (size_t)F / 2));
  int paired = std::min<int>((int)tok_h.size(), F / 2);
  out.dropped_tokens = (int)tok_h.size() - paired;
  out.dropped_frames = F - 2 * paired;
  out.aligned_ok = paired >= 1 && out.dropped_tokens <= 1 && out.dropped_frames <= 2;
  out.prompt_token.assign(tok_h.begin(), tok_h.begin() + paired);
  out.prompt_feat.assign(mel_h.begin(), mel_h.begin() + (size_t)2 * paired * 80);
  return out;
}

// ------------------------------------------------------------------------------------------------
// Speaker embedding (pyannote/embedding, SincNet + TDNN x-vector), fed 24 kHz like production
// ------------------------------------------------------------------------------------------------
void VoiceBuilder::pyannote(const float* d_wav16, int n, float* d_out512) {
  float* x = arena_.f(n);
  double* scratch = reinterpret_cast<double*>(arena_.f(2 * 64 * 2));  // 2*64 doubles for C = 1 (256-B aligned)
  instance_norm(stream_, d_wav16, x, 1, n, w_->d("sincnet.wav_norm1d.weight"), w_->d("sincnet.wav_norm1d.bias"),
                1e-5f, Act::kNone, 0.f, scratch);
  int T1 = (n - 251) / 10 + 1;
  float* s1 = arena_.f((size_t)80 * T1);
  {
    size_t mark = arena_.used;
    float* col = arena_.f((size_t)251 * T1);
    im2col1d(stream_, x, 1, n, 251, 10, 1, 0, T1, col);
    sgemm(stream_, 80, T1, 251, w_->d("sincnet.conv1d.0.filters"), 251, col, T1, false, s1, T1);
    arena_.used = mark;
  }
  abs_inplace(stream_, s1, (long long)80 * T1);
  int P1 = T1 / 3;
  float* p1 = arena_.f((size_t)80 * P1);
  maxpool1d(stream_, s1, p1, 80, T1, 3, P1);
  instance_norm(stream_, p1, p1, 80, P1, w_->d("sincnet.norm1d.0.weight"), w_->d("sincnet.norm1d.0.bias"), 1e-5f,
                Act::kLeaky, 0.01f);
  int C1 = P1 - 4;
  float* c1 = arena_.f((size_t)60 * C1);
  Epilogue eb;
  eb.bias = w_->d("sincnet.conv1d.1.bias");
  eb.per_row = true;
  conv1d(p1, 80, P1, "sincnet.conv1d.1.weight", 60, 5, 1, 1, 0, c1, C1, eb);
  int P2 = C1 / 3;
  float* p2 = arena_.f((size_t)60 * P2);
  maxpool1d(stream_, c1, p2, 60, C1, 3, P2);
  instance_norm(stream_, p2, p2, 60, P2, w_->d("sincnet.norm1d.1.weight"), w_->d("sincnet.norm1d.1.bias"), 1e-5f,
                Act::kLeaky, 0.01f);
  int C2 = P2 - 4;
  float* c2 = arena_.f((size_t)60 * C2);
  eb.bias = w_->d("sincnet.conv1d.2.bias");
  conv1d(p2, 60, P2, "sincnet.conv1d.2.weight", 60, 5, 1, 1, 0, c2, C2, eb);
  int P3 = C2 / 3;
  float* p3 = arena_.f((size_t)60 * P3);
  maxpool1d(stream_, c2, p3, 60, C2, 3, P3);
  instance_norm(stream_, p3, p3, 60, P3, w_->d("sincnet.norm1d.2.weight"), w_->d("sincnet.norm1d.2.bias"), 1e-5f,
                Act::kLeaky, 0.01f);
  tap("pya_sincnet", p3, (size_t)60 * P3);
  // TDNNs: conv -> LeakyReLU -> BatchNorm
  const int outs[5] = {512, 512, 512, 512, 1500};
  const int ks[5] = {5, 3, 3, 1, 1};
  const int ds[5] = {1, 2, 3, 1, 1};
  float* cur = p3;
  int C = 60, T = P3;
  for (int i = 0; i < 5; ++i) {
    int To = T - ds[i] * (ks[i] - 1);
    float* o = arena_.f((size_t)outs[i] * To);
    std::string cw = "tdnns." + std::to_string(3 * i);
    std::string bnn = "tdnns." + std::to_string(3 * i + 2);
    Epilogue e;
    e.bias = w_->d(cw + ".bias");
    e.per_row = true;
    e.act = Act::kLeaky;
    e.leaky_slope = 0.01f;
    e.act_first = true;
    e.scale = bn_scale(bnn);
    e.shift = bn_shift(bnn);
    conv1d(cur, C, T, cw + ".weight", outs[i], ks[i], 1, ds[i], 0, o, To, e);
    cur = o;
    C = outs[i];
    T = To;
  }
  tap("pya_tdnn", cur, (size_t)C * T);
  float* st = arena_.f((size_t)2 * C);
  stats_pool(stream_, cur, C, T, T, st);
  tap("pya_stats", st, (size_t)2 * C);
  Epilogue ee;
  ee.bias = w_->d("embedding.bias");
  ee.per_row = true;
  sgemm(stream_, 512, 1, 2 * C, w_->d("embedding.weight"), 2 * C, st, 1, false, d_out512, 1, ee);
}

std::vector<float> VoiceBuilder::speaker_embedding(const float* wav24, size_t n) {
  arena_.reset();
  float* d24 = arena_.f(n);
  check(cudaMemcpyAsync(d24, wav24, n * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  ev_start();
  int n16 = (int)((2 * n + 2) / 3);
  float* d16 = arena_.f(n16);
  resample_sinc_3_2(stream_, d24, (int)n, w_->d("ta_24_16_kernel_cpu"), d16, n16);
  tap("pya_w16", d16, n16);
  float* e = arena_.f(512);
  pyannote(d16, n16, e);
  std::vector<float> out(512);
  check(cudaMemcpyAsync(out.data(), e, 512 * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  ev_stop("speaker_embedding");
  if (keep_taps) taps["pya_raw"] = out;
  double ss = 0.0;
  for (float v : out) ss += double(v) * v;
  float inv = (float)(1.0 / std::sqrt(ss));
  for (float& v : out) v *= inv;
  if (speaker_fault_for_test) speaker_fault_for_test(out);
  return out;
}

std::vector<int32_t> VoiceBuilder::tokens_from_logmel(const float* logmel, int F) {
  arena_.reset();
  float* d = arena_.f((size_t)128 * F);
  check(cudaMemcpyAsync(d, logmel, (size_t)128 * F * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  ev_start();
  int32_t* codes = arena_.i(F / 4 + 4);
  int T = tokenizer(d, F, codes);
  std::vector<int32_t> out(T);
  check(cudaMemcpyAsync(out.data(), codes, (size_t)T * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  ev_stop("net_tokenizer");
  return out;
}

std::vector<float> VoiceBuilder::xvector_from_fbank(const float* fbank, int T) {
  arena_.reset();
  float* d = arena_.f((size_t)T * 80);
  check(cudaMemcpyAsync(d, fbank, (size_t)T * 80 * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  ev_start();
  float* xv = arena_.f(192);
  campplus(d, T, xv);
  std::vector<float> out(192);
  check(cudaMemcpyAsync(out.data(), xv, 192 * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  ev_stop("net_campplus");
  return out;
}

std::vector<float> VoiceBuilder::raw_embedding_from_w16(const float* w16, size_t n) {
  arena_.reset();
  float* d = arena_.f(n);
  check(cudaMemcpyAsync(d, w16, n * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  ev_start();
  float* e = arena_.f(512);
  pyannote(d, (int)n, e);
  std::vector<float> out(512);
  check(cudaMemcpyAsync(out.data(), e, 512 * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  ev_stop("net_pyannote");
  return out;
}

// LM-prefix tokens enqueued on lm_ (resample_poly -> log-mel -> tokenizer -> pinned D2H); no host wait.
void VoiceBuilder::lm_tokens_begin(const float* wav24, size_t n) {
  std::swap(stream_, lm_);
  std::swap(arena_, arena3_);
  arena_.reset();
  float* d24 = arena_.f(n);
  check(cudaMemcpyAsync(d24, wav24, n * 4, cudaMemcpyHostToDevice, stream_), "h2d");
  int n16 = (int)((2 * n + 2) / 3);
  float* d16 = arena_.f(n16);
  resample_poly_2_3(stream_, d24, (int)n, w_->d("poly_2_3_taps"), d16, n16);
  int F = 0;
  float* mel = logmel_s3(d16, n16, &F);
  int32_t* codes = arena_.i(F / 4 + 4);
  lm_T_ = tokenizer(mel, F, codes);
  if (lm_T_ > 4096) throw std::runtime_error("too many LM tokens");
  check(cudaMemcpyAsync(h_lm_codes_, codes, (size_t)lm_T_ * 4, cudaMemcpyDeviceToHost, stream_), "d2h");
  check(cudaEventRecord(lm_done_, stream_), "lm event");
  std::swap(stream_, lm_);
  std::swap(arena_, arena3_);
}

std::vector<int32_t> VoiceBuilder::lm_tokens_end() {
  check(cudaEventSynchronize(lm_done_), "lm sync");
  return std::vector<int32_t>(h_lm_codes_, h_lm_codes_ + lm_T_);
}

double VoiceBuilder::warmup() {
  auto t0 = std::chrono::steady_clock::now();
  bool taps_were = keep_taps;
  keep_taps = false;
  // production's synthetic warm-up tone (runtime.py cf_build_warmup): 6 s, 180 Hz with 3 Hz tremolo
  std::vector<float> tone(6 * kSR), noise(10 * kSR);
  for (size_t i = 0; i < tone.size(); ++i) {
    double t = double(i) / kSR;
    double v = 0.1 * std::sin(2.0 * M_PI * 180.0 * t) * (0.6 + 0.4 * std::sin(2.0 * M_PI * 3.0 * t));
    tone[i] = (float)(std::nearbyint(v * 32767.0) / 32767.0);
  }
  uint32_t st = 12345u;
  for (auto& v : noise) {
    st = st * 1664525u + 1013904223u;
    v = ((float)(st >> 8) / 16777216.0f - 0.5f) * 0.2f;
  }
  (void)speaker_embedding(tone.data(), tone.size());
  auto w16 = resample_poly16(tone.data(), tone.size());
  (void)s3_tokens(w16.data(), w16.size());
  (void)decoder_ref(tone.data(), tone.size());
  (void)decoder_ref(noise.data(), noise.size());  // eager at 10 s (fills caches)
  (void)decoder_ref(noise.data(), noise.size());  // captures the 10 s graph
  keep_taps = taps_were;
  check(cudaStreamSynchronize(stream_), "warmup sync");
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

CloneResult VoiceBuilder::build_clone(const float* decoded24, size_t n, const int32_t* text_ids, size_t n_text,
                                      const float* consent_speaker, const PrefixIds& ids) {
  CloneResult r;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  auto t0 = now();
  AudioVerdict v = check_decoded(decoded24, n);
  if (v != AudioVerdict::kOk) {
    r.status = 200 + (int)v - 1;  // 200 invalid_audio, 201 duration, 202 signal
    r.code = verdict_code(v);
    return r;
  }
  r.pcm16.resize(n);
  std::vector<float> wav24(n);
  canonical_pcm16(decoded24, n, r.pcm16.data(), wav24.data());
  auto t1 = now();
  r.raw_speaker = speaker_embedding(wav24.data(), n);
  if (!speaker_embedding_ok(r.raw_speaker)) {  // production _pyannote_embedding's check (clone_policy.h)
    r.status = 213;
    r.code = "custom_voice_service_error";
    return r;
  }
  auto t2 = now();
  if (consent_speaker) {
    float sim = 0.f;  // np.dot of two float32 vectors
    for (int d = 0; d < 512; ++d) sim += r.raw_speaker[d] * consent_speaker[d];
    r.consent_similarity = sim;
    // Production: `similarity < 0.70` -> 422. Written as !(sim >= 0.70) so a non-finite similarity (a corrupt
    // stored consent embedding) fails closed instead of passing; finite values behave exactly as production.
    if (!(sim >= 0.70f)) {
      r.status = 206;
      r.code = "speaker_verification_failed";
      return r;
    }
  }
  r.nearest_index = nearest_voice(r.raw_speaker, &r.nearest_similarity);
  if (r.nearest_index < 0) {  // no catalogue bank (set_bank / set_catalogue never called): a service fault
    r.status = 213;
    r.code = "custom_voice_service_error";
    return r;
  }
  auto t3 = now();
  if (overlap_lm_tokens && !keep_taps) {
    // LM-prefix tokens (from resample_poly) and the decoder reference (from the sinc resampler) share no
    // inputs: run them concurrently on separate streams, arenas and split-K workspaces.
    lm_tokens_begin(wav24.data(), n);
    r.decoder = decoder_ref(wav24.data(), n);
    r.speech_tokens = lm_tokens_end();
  } else {
    auto w16 = resample_poly16(wav24.data(), n);
    r.speech_tokens = s3_tokens(w16.data(), w16.size());
    r.decoder = decoder_ref(wav24.data(), n);
  }
  auto t4 = now();
  // production order: the decoder reference's alignment error is raised before the artifact's prefix check
  if (!r.decoder.aligned_ok) {
    r.status = 204;
    r.code = "invalid_reference_audio";
    return r;
  }
  r.lm_prefix.reserve(n_text + r.speech_tokens.size() + 5);
  r.lm_prefix.push_back(ids.start);
  r.lm_prefix.push_back(ids.ref_text_start);
  r.lm_prefix.insert(r.lm_prefix.end(), text_ids, text_ids + n_text);
  r.lm_prefix.push_back(ids.ref_text_end);
  r.lm_prefix.push_back(ids.ref_speech_start);
  for (int32_t t : r.speech_tokens) r.lm_prefix.push_back(ids.audio_id_base + t);
  r.lm_prefix.push_back(ids.ref_speech_end);
  if ((int)r.lm_prefix.size() > ids.max_prefix) {
    r.status = 203;
    r.code = "invalid_reference_audio";
    return r;
  }
  r.wall_ms = {{"checks_pcm16", ms(t0, t1)}, {"speaker_embedding", ms(t1, t2)}, {"consent_nearest", ms(t2, t3)},
               {"tokens_and_decoder_ref", ms(t3, t4)}, {"total", ms(t0, now())}};
  return r;
}

int VoiceBuilder::nearest_voice(const std::vector<float>& emb, float* best_sim) const {
  size_t nv = bank_.size() / 512;
  int best = -1;
  float best_s = -1.f;
  for (size_t i = 0; i < nv; ++i) {
    const float* c = bank_.data() + i * 512;
    double nn = 0.0;
    for (int d = 0; d < 512; ++d) nn += double(c[d]) * c[d];
    float norm = (float)std::sqrt(nn);
    if (norm <= 0.f) continue;
    float s = 0.f;
    for (int d = 0; d < 512; ++d) s += emb[d] * (c[d] / norm);
    if (s > best_s) {
      best_s = s;
      best = (int)i;
    }
  }
  if (best_sim) *best_sim = best_s;
  return best;
}

}  // namespace kt2n::voice
