// CUDA helpers of kitten-tts-tiny (built when ggml is built with CUDA). Internal to the library.
// Pointers named gx / out / f0 / p300 / har are device pointers (the data of ggml tensors in a CUDA buffer).
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { KT_CUDA_NOISE_RANDOM = 0, KT_CUDA_NOISE_NONE = 1, KT_CUDA_NOISE_EXTERNAL = 2 };

typedef struct kt_cuda_ctx kt_cuda_ctx;

kt_cuda_ctx * kt_cuda_init(int device);
void          kt_cuda_free(kt_cuda_ctx * ctx);
const char *  kt_cuda_last_error(void);

// Register the recurrent weights of one BiLSTM: whh is host memory laid out as ggml stores it, [2][4H][H].
int kt_cuda_add_lstm(kt_cuda_ctx * ctx, const char * key, const float * whh, int H);
int kt_cuda_has_lstm(kt_cuda_ctx * ctx, const char * key);

// Both directions of the recurrence. gx: input projections incl. biases, [2][T][4H] with strides (in floats)
// gx_s1 per step and gx_s2 per direction; direction 1 is already time-reversed. out: [T][2H] (row stride out_s1),
// written in time order.
int kt_cuda_lstm(kt_cuda_ctx * ctx, const char * key, const float * gx, long long gx_s1, long long gx_s2,
                 float * out, long long out_s1, int H, int T);

// The same on `stream` (a cudaStream_t) without waiting: for a graph node that ggml's CUDA backend runs.
int kt_cuda_lstm_async(kt_cuda_ctx * ctx, void * stream, const char * key, const float * gx, long long gx_s1, long long gx_s2,
                       float * out, long long out_s1, int H, int T);

// Harmonic source + forward STFT. f0: [L0]; p300: [L0][9] cumulative phases (computed on the host);
// har: [frames][n_fft + 2] magnitudes then angles. noise_host: [L0 * upp][9] for KT_CUDA_NOISE_EXTERNAL.
int kt_cuda_harmonic(kt_cuda_ctx * ctx, const float * f0, const float * p300, int L0, int upp, const float * lin_w,
                     float lin_b, int noise_mode, unsigned long long seed, const float * noise_host, size_t noise_len,
                     int n_fft, int hop, float * har, long long frames);

// Fused AdaIN + activation + zero padding + convolution window, as graph nodes (no host wait).
// kt_cuda_adain_reserve: scratch for the largest op of a graph (C channels, pad_floats = padded length * Cp), before it runs.
// kt_cuda_adain_window: x: [T][C] (row stride xs1 floats); h: the style projection [gamma C, beta C]; nw / nb: optional affine
// norm weight / bias [C]; act 0 none, 1 leaky relu (slope), 2 snake (alpha, 1/alpha [C]). out: [Tout][K * Cp] windows of
// the padded signal (step t reads rows t + k d - pl, k < K; channels >= C are zero). stream: a cudaStream_t.
int kt_cuda_adain_reserve(kt_cuda_ctx * ctx, int C, long long pad_floats);
int kt_cuda_adain_window(kt_cuda_ctx * ctx, void * stream, const float * x, long long xs1, int C, int T, const float * h, const float * nw,
                         const float * nb, const float * alpha, const float * ialpha, int act, float slope, float eps,
                         int Cp, int K, int d, int pl, int Tout, float * out);

// The same AdaIN + activation + padding, then the convolution itself (implicit GEMM on the padded signal, f32 products and
// sums, no window copy). w: [Cout][Cp * K] f32; bias [Cout] or NULL; resid [Tout][Cout] added to the result, or NULL;
// out: [Tout][Cout]. Needs Cp % 16 == 0 and Cout % 4 == 0.
int kt_cuda_adain_conv(kt_cuda_ctx * ctx, void * stream, const float * x, long long xs1, int C, int T, const float * h, const float * nw,
                       const float * nb, const float * alpha, const float * ialpha, int act, float slope, float eps,
                       int Cp, int K, int d, int pl, int Tout, const float * w, const float * bias, const float * resid, int Cout, float * out);

#ifdef __cplusplus
}
#endif
