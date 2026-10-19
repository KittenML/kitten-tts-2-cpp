// L11 native voice build (clone prepare + build), production kittenml-api ebcdd3f semantics.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "vb_kernels.cuh"

namespace kt2n::voice {

// Production's checks after the decode (runtime.py:11084-11186).
enum class AudioVerdict : int { kOk = 0, kInvalidAudio = 1, kInvalidDuration = 2, kInvalidSignal = 3 };
AudioVerdict check_decoded(const float* x, size_t n);
const char* verdict_code(AudioVerdict v);  // the API error code string
// pcm = rint(clip(x,-1,1)*32767); wav24 = pcm/32767.
void canonical_pcm16(const float* x, size_t n, int16_t* pcm, float* wav24);
// Decoder reference audio: tile to >= 6 s, then crop to the first 10 s (24 kHz).
std::vector<float> decoder_audio(const float* wav24, size_t n);

struct DecoderRef {
  std::vector<int32_t> prompt_token;  // aligned 2:1
  std::vector<float> prompt_feat;     // [2*tokens][80]
  std::vector<float> xvector;         // 192
  int raw_tokens = 0, raw_frames = 0, dropped_tokens = 0, dropped_frames = 0;
  bool aligned_ok = false;
};

// Special ids of the RL-330 LM (runtime_config.json; production runtime.py:90-99).
struct PrefixIds {
  int32_t start = 151673;
  int32_t ref_text_start = 158239;
  int32_t ref_text_end = 158240;
  int32_t ref_speech_start = 158237;
  int32_t ref_speech_end = 158238;
  int32_t audio_id_base = 151675;
  int32_t max_prefix = 8192;  // custom_voices.py MAX_REFERENCE_PREFIX_TOKENS
};

// One whole clone build (production's prepare + _build_custom_voice_artifact), on decoded 24 kHz float audio.
// status: 0 ok, else the engine.h kt2n_status of production's error:
//   200 invalid_audio, 201 invalid_audio_duration, 202 invalid_audio_signal, 203 invalid_reference_audio (prefix
//   too long), 204 reference alignment (422), 206 speaker_verification_failed (422), 213 custom_voice_service_error.
struct CloneResult {
  int32_t status = 0;
  std::string code;                   // production error.code ("" when ok)
  std::vector<int16_t> pcm16;         // canonical PCM16 (what the store keeps)
  std::vector<float> raw_speaker;     // 512, L2-normalised (the upload's own embedding)
  int32_t nearest_index = -1;         // catalogue row whose projected speaker conditions the LM
  float nearest_similarity = 0.f;
  float consent_similarity = 0.f;     // when a consent embedding was given
  std::vector<int32_t> lm_prefix;     // _reference_prompt
  std::vector<int32_t> speech_tokens; // the S3 tokens inside the prefix
  DecoderRef decoder;                 // prompt_token / prompt_feat / x-vector
  std::map<std::string, double> wall_ms;
};

struct StageTimes {  // device milliseconds (cudaEvent) per stage of the last call
  std::map<std::string, float> ms;
};

class DeviceTensors;  // safetensors / s3gen.kt2n -> device

// The engine's weight file: s3gen.kt2n with the four clone groups (clone_weights.h).
struct S3genKt2n {
  std::string path;
  bool verify_sha256 = false;  // hash every clone segment at load (FORMAT: optional; tests turn it on)
};

class VoiceBuilder {
 public:
  // Three-file layout (lab tools): s3gen = the S3Gen safetensors (tokenizer.* and speaker_encoder.* are used);
  // consts: voicebuild_consts.safetensors; pyannote: voicebuild_pyannote.safetensors (weights + 48-voice bank).
  VoiceBuilder(const std::string& s3gen, const std::string& consts, const std::string& pyannote,
               size_t arena_bytes = size_t(768) << 20);
  // Engine layout: every clone weight from s3gen.kt2n. No bank inside: call set_bank() (VoiceService::
  // set_catalogue does) before any build, else build_clone answers 213 custom_voice_service_error.
  explicit VoiceBuilder(const S3genKt2n& s3gen_kt2n, size_t arena_bytes = size_t(768) << 20);
  double weights_load_ms() const { return weights_load_ms_; }      // read + upload (+ verify) at construction
  double weights_verify_ms() const { return weights_verify_ms_; }  // sha256 of the clone segments (kt2n, opt-in)
  size_t weight_bytes() const;
  struct WeightView {  // device weights by the builder's name (layout checks)
    std::string name;
    const float* d;
    std::vector<int64_t> shape;
  };
  std::vector<WeightView> weights() const;
  ~VoiceBuilder();
  VoiceBuilder(const VoiceBuilder&) = delete;
  VoiceBuilder& operator=(const VoiceBuilder&) = delete;

  // --- step 1 (prepare) ---
  std::vector<float> resample_poly16(const float* wav24, size_t n);           // scipy resample_poly(2,3)
  std::vector<float> speaker_embedding(const float* wav24, size_t n);         // 512, L2-normalised
  // Test only: edits every embedding speaker_embedding() returns (fault injection for the 213 check on every
  // path, tools/voice/service_check.cu). Empty in production.
  std::function<void(std::vector<float>&)> speaker_fault_for_test;
  int nearest_voice(const std::vector<float>& emb, float* best_sim = nullptr) const;  // first max wins
  const std::vector<std::string>& bank_voice_ids() const { return bank_ids_; }
  // Replace the nearest-flagship bank with the catalogue's raw rows ([n][512], voices.kt2n order). Call before any
  // build (not thread-safe against a running build).
  void set_bank(std::vector<float> rows512, std::vector<std::string> ids) {
    bank_ = std::move(rows512);
    bank_ids_ = std::move(ids);
  }

  // --- step 2 (build) ---
  std::vector<int32_t> s3_tokens(const float* wav16, size_t n);               // LM prefix speech tokens
  DecoderRef decoder_ref(const float* wav24, size_t n);                        // tile/crop + embed_ref + align

  // The whole build. consent_speaker: 512 floats (P3/P5 paths) or nullptr. Thread: one build at a time per builder.
  CloneResult build_clone(const float* decoded24, size_t n, const int32_t* text_ids, size_t n_text,
                          const float* consent_speaker = nullptr, const PrefixIds& ids = PrefixIds{});
  // Production's activation warm-up (CF_BUILD_WARMUP): runs every kernel once so later builds never allocate,
  // and captures the 10 s decoder-reference graph. Returns milliseconds.
  double warmup();

  // Network-only entry points (same inputs as the ONNX Runtime / PyTorch comparison arms).
  std::vector<int32_t> tokens_from_logmel(const float* logmel_128xF, int F);
  std::vector<float> xvector_from_fbank(const float* fbank_Tx80, int T);
  std::vector<float> raw_embedding_from_w16(const float* w16, size_t n);

  // Debug taps for the oracle checks (host copies of the last call's intermediates).
  std::map<std::string, std::vector<float>> taps;
  bool keep_taps = false;
  bool use_graphs = true;  // CUDA graph for the 10 s decoder reference (most uploads)
  bool overlap_branches = true;  // decoder ref: tokenizer branch on a second stream, concurrent with the x-vector
  bool overlap_lm_tokens = true; // build_clone: LM-prefix tokens on a third stream, concurrent with the decoder ref
  StageTimes times;
  cudaStream_t stream() const { return stream_; }

 private:
  struct Arena {
    char* base = nullptr;
    size_t cap = 0, used = 0;
    float* f(size_t n);
    int32_t* i(size_t n);
    void reset() { used = 0; }
  };
  float* logmel_s3(const float* d_wav16, int n, int* frames);                 // [128][F]
  int tokenizer(const float* d_logmel, int F, int32_t* d_codes);              // returns tokens
  float* prompt_mel(const float* d_wav24, int n, int* frames);                // [F][80]
  float* kaldi_fbank(const float* d_wav16, int n, int* frames);               // [T][80], mean removed
  void campplus(const float* d_fbank, int T, float* d_out192);
  void pyannote(const float* d_wav16, int n, float* d_out512);
  void tap(const char* name, const float* d, size_t n);
  void tap_i(const char* name, const int32_t* d, size_t n);
  void bn(const std::string& prefix, std::vector<float>& scale, std::vector<float>& shift, bool affine = true);
  const float* bn_scale(const std::string& prefix, bool affine = true);
  const float* bn_shift(const std::string& prefix, bool affine = true);
  void conv1d(const float* x, int cin, int T, const std::string& w, int cout, int K, int stride, int dil, int pad,
              float* y, int t_out, const Epilogue& ep);
  void conv2d(const float* x, int cin, int H, int W, const std::string& w, int cout, int sh, int ph, float* y,
              int Ho, const Epilogue& ep);
  struct DecOut {
    float* mel = nullptr;
    int F = 0;
    int32_t* codes = nullptr;
    int Tt = 0;
    float* xv = nullptr;
  };
  struct DecGraphEntry {
    cudaGraphExec_t exec;
    DecOut out;
  };
  void decoder_ref_enqueue(float* d24, int nd, DecOut& o);
  std::map<int, DecGraphEntry> dec_graphs_;
  std::set<int> dec_warm_;
  void ev_start();
  void ev_stop(const char* stage);

  void init_device(size_t arena_bytes);
  void release_device() noexcept;
  double weights_load_ms_ = 0.0, weights_verify_ms_ = 0.0;
  DeviceTensors* w_ = nullptr;
  Arena arena_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev0_ = nullptr, ev1_ = nullptr;
  float* d_basis24_ = nullptr;   // [1920][1922]
  float* d_basis400_ = nullptr;  // [400][402]
  float* d_basis512_ = nullptr;  // [512][514]
  float* d_freqs_ = nullptr;     // 32 rotary frequencies
  float* d_splitk_ = nullptr;    // deterministic split-K partials
  float* d_splitk2_ = nullptr;   // the side stream's own partials
  cudaStream_t side_ = nullptr;
  cudaEvent_t fork_ = nullptr, join_ = nullptr;
  Arena arena2_;
  // LM-prefix tokens on their own stream / arena / split-K workspace (build_clone overlap)
  cudaStream_t lm_ = nullptr;
  cudaEvent_t lm_done_ = nullptr;
  Arena arena3_;
  float* d_splitk3_ = nullptr;
  int32_t* h_lm_codes_ = nullptr;  // pinned
  int lm_T_ = 0;
  void lm_tokens_begin(const float* wav24, size_t n);
  std::vector<int32_t> lm_tokens_end();
  std::map<std::string, float*> derived_;  // BN scale/shift
  std::vector<float> bank_;  // [48][512]
  std::vector<std::string> bank_ids_;
};

}  // namespace kt2n::voice
