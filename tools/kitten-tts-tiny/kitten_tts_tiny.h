// Kitten TTS 2 mini / nano / pico (config type KITTEN2_TINY) on ggml: C API.
//
// One model = one GGUF written by convert_kitten_tiny_to_gguf.py (weights, config, vocab and voices).
// Input is phonemes (or token ids); text to phonemes is the Python front end (kitten_tts_cpp.KittenTTS2Tiny).
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(KT_TINY_BUILD)
#    define KT_TINY_API __declspec(dllexport)
#  else
#    define KT_TINY_API __declspec(dllimport)
#  endif
#else
#  define KT_TINY_API __attribute__((visibility("default")))
#endif

typedef struct kt_tiny_model kt_tiny_model;

typedef struct kt_tiny_params {
    const char * device;     // "auto" (first GPU, else CPU), "cpu", "cuda", "metal", "vulkan", or a ggml device name such as "Vulkan0"
    int          n_threads;  // CPU threads; 0 = the physical cores this process may use (affinity mask and, on Linux,
                             // the cgroup CPU quota; on Apple, the performance cores), at most 8
    int          dequantize; // 1: convert f16 / q8_0 weights to f32 when loading (small file, f32 arithmetic); 0: compute in
                             // the file's type (less memory; ggml then also rounds activations to that type)
    int          allow_tf32; // 0 (default): f32 matrix products in full f32 on GPUs. 1: let CUDA use TF32 tensor cores
                             // and Vulkan fp16 products (faster on some GPUs, lossy: ~2-3 dB log-mel on pico)
} kt_tiny_params;

enum kt_tiny_noise_mode {
    KT_TINY_NOISE_RANDOM   = 0, // Gaussian noise in the harmonic source, as the reference model does
    KT_TINY_NOISE_NONE     = 1, // no noise (deterministic)
    KT_TINY_NOISE_EXTERNAL = 2, // caller-supplied standard-normal values, [n_samples][9]
};

typedef struct kt_tiny_synth_options {
    float           speed;       // 0.25 to 4; 1 is normal, > 1 speaks faster
    uint64_t        seed;        // KT_TINY_NOISE_RANDOM
    int             noise_mode;  // enum kt_tiny_noise_mode
    const float   * noise;       // KT_TINY_NOISE_EXTERNAL: n_samples * 9 values (n_samples = 600 * frames)
    size_t          noise_len;
    const int32_t * durations;   // optional: frames per token (overrides the predicted durations)
} kt_tiny_synth_options;

KT_TINY_API kt_tiny_params        kt_tiny_default_params(void);
KT_TINY_API kt_tiny_synth_options kt_tiny_default_options(void);

// Returns NULL on failure; kt_tiny_last_error() says why (per thread).
KT_TINY_API kt_tiny_model * kt_tiny_load(const char * gguf_path, const kt_tiny_params * params);
KT_TINY_API void            kt_tiny_free(kt_tiny_model * model);
KT_TINY_API const char *    kt_tiny_last_error(void);

// JSON: model name, languages, voices, sample rate, device, weight type, config.json and vocab.
KT_TINY_API const char * kt_tiny_info_json(kt_tiny_model * model);

// The 256-value style row a chunk of n_tokens tokens uses (voice = key or display name, any case). 0 = ok.
KT_TINY_API int kt_tiny_voice_style(kt_tiny_model * model, const char * language, const char * voice,
                                    int n_tokens, float * style256);

// The raw model: token ids and a style row -> waveform (24 kHz) and frames per token. 0 = ok, -1 = error
// (bad arguments are reported, not asserted: token ids must be in [0, vocabulary size), n_tokens in [2, 512]).
// One model must not be used by two threads at once; separate models are independent.
// *audio and *durations are allocated by the library; release them with kt_tiny_free_buffer.
KT_TINY_API int kt_tiny_synthesize(kt_tiny_model * model, const int32_t * tokens, int n_tokens,
                                   const float * style256, const kt_tiny_synth_options * options,
                                   float ** audio, size_t * n_audio, int32_t ** durations);

// One chunk of phonemes as kittenml does it: tokens [0] + ids + [10, 0], the voice's style row for that
// length, synthesis, the end token's frames cut (and the start pause unless keep_start), 5 ms fades. speed: 0.25 to 4
// (1.0 is normal). 0 = ok, -1 = error (kt_tiny_last_error).
KT_TINY_API int kt_tiny_generate_chunk(kt_tiny_model * model, const char * phonemes_utf8, const char * language,
                                       const char * voice, float speed, int keep_start, uint64_t seed,
                                       float ** audio, size_t * n_audio);

// JSON timings (ms) and buffer sizes of the last call. The string belongs to the model and changes with its next call:
// read it on the thread that made the call, before the model is used again.
KT_TINY_API const char * kt_tiny_last_report(kt_tiny_model * model);

KT_TINY_API void kt_tiny_free_buffer(void * buffer);

#ifdef __cplusplus
}
#endif
