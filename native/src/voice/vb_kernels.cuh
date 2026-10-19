// L11 native voice build: the small CUDA op library behind the clone networks
// (speaker embedding, speech tokenizer, x-vector) and the audio front ends.
//
// Math mode, every op: fp32 inputs, fp32 FFMA accumulation, no TF32, no fast-math.
// (Production runs the matmuls in strict fp32 and lets cuDNN use TF32 for convs; plain
// fp32 everywhere is the more precise side of that.)
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace kt2n::voice {

enum class Act : int { kNone = 0, kRelu = 1, kLeaky = 2, kGelu = 3, kSigmoid = 4 };

// Epilogue of the GEMM / conv kernels, applied per output element (row m, column n):
//   v = acc + bias[m or n]
//   act_first ? v = act(v) * scale[ch] + shift[ch] : v = act(v * scale[ch] + shift[ch])
//   v += residual[m*ldc+n]
// ch = m when per_row, else n. Null pointers skip that step.
struct Epilogue {
  const float* bias = nullptr;
  const float* scale = nullptr;
  const float* shift = nullptr;
  const float* residual = nullptr;
  bool per_row = false;  // bias/scale/shift indexed by row m (conv: output channel) instead of column n
  bool act_first = false;
  Act act = Act::kNone;
  float leaky_slope = 0.01f;
};

// C[M,N] = A[M,K] * op(B) with op(B)[k,n] = trans_b ? B[n*ldb+k] : B[k*ldb+n]. Row-major A and C.
void sgemm(cudaStream_t s, int M, int N, int K, const float* A, int lda, const float* B, int ldb,
           bool trans_b, float* C, int ldc, const Epilogue& ep = Epilogue{});

// Split-K workspace for the large-tile path (floats) and the SM count used to pick the split.
void sgemm_set_workspace(float* ws, size_t floats, int sm_count);
// A separate split-K workspace for a stream that runs GEMMs concurrently with others (falls back to the global one).
void sgemm_set_stream_workspace(cudaStream_t stream, float* ws, size_t floats);
// Forget a stream (its owner is being destroyed; a later stream may reuse the handle value).
void sgemm_clear_stream_workspace(cudaStream_t stream);
// Large-tile kernel choice (0: BK=8 base, 1: BK=16 float4, default) and split-K policy (0: fill SMs, 1: off).
void sgemm_set_variant(int variant, int split_policy);

// Batched strided version (batch over blockIdx.z): A += z*sa, B += z*sb, C += z*sc.
void sgemm_batched(cudaStream_t s, int batch, int M, int N, int K, const float* A, int lda, long long sa,
                   const float* B, int ldb, long long sb, bool trans_b, float* C, int ldc, long long sc,
                   const Epilogue& ep = Epilogue{});

// im2col for conv1d: x[Cin][T] -> col[(ci*K+k)][t_out], zero padding.
void im2col1d(cudaStream_t s, const float* x, int cin, int T, int K, int stride, int dil, int pad, int t_out,
              float* col);
// im2col for conv2d (NCHW, N=1): x[Cin][H][W] -> col[(ci*kh*kw + i*kw + j)][(ho*Wo + wo)].
void im2col2d(cudaStream_t s, const float* x, int cin, int H, int W, int kh, int kw, int sh, int sw, int ph,
              int pw, int Ho, int Wo, float* col);

// y[c][t] = act(x[c][t]*scale[c]+shift[c]) (in place allowed). act_first swaps the order.
void affine_act(cudaStream_t s, const float* x, float* y, int C, int T, int ldx, int ldy, const float* scale,
                const float* shift, Act act, bool act_first = false, float leaky = 0.01f);
void eltwise_add(cudaStream_t s, const float* a, const float* b, float* y, long long n);
void eltwise_mul(cudaStream_t s, const float* a, const float* b, float* y, long long n);
void eltwise_act(cudaStream_t s, const float* x, float* y, long long n, Act act, float leaky = 0.01f);
void abs_inplace(cudaStream_t s, float* x, long long n);

// MaxPool1d(k, stride=k, no padding) per channel.
void maxpool1d(cudaStream_t s, const float* x, float* y, int C, int T, int k, int t_out);
// Per-channel normalisation over time (biased variance, fp64 statistics): InstanceNorm1d.
// scratch: >= 2*64*C doubles enables the multi-block path for few channels over long time (C < 32, T >= 8192).
void instance_norm(cudaStream_t s, const float* x, float* y, int C, int T, const float* w, const float* b,
                   float eps, Act act, float leaky, double* scratch = nullptr);
// Per-channel mean and std (unbiased, n-1) over time: out[0:C] = mean, out[C:2C] = std. fp64 statistics.
void stats_pool(cudaStream_t s, const float* x, int C, int T, int ldx, float* out);
// Per-channel mean over time and ceil-mode segment averages broadcast back: ctx[c][t] = mean_c + seg_c(t).
void cam_context(cudaStream_t s, const float* x, int C, int T, int seg, float* ctx);
// Per-segment form: ctx_seg[c][g] = mean_c + avg_c(segment g), g < ceil(T/seg) (the context is constant per segment).
void cam_context_seg(cudaStream_t s, const float* x, int C, int T, int seg, float* ctx_seg);
// CAM gate per segment: m_seg[32][nseg] = sigmoid(W2[32][64] relu(W1[64][128] ctx_seg + b1) + b2).
void cam_mlp(cudaStream_t s, const float* ctx_seg, int nseg, const float* W1, const float* b1, const float* W2,
             const float* b2, float* m_seg);
// out[c][t] = y[c][t] * m[c][t / seg]
void mul_seg(cudaStream_t s, const float* y, const float* m, float* out, int C, int T, int seg);
// Subtract the per-column mean over rows (x[T][F], F columns): CAMPPlus feature normalisation.
void sub_col_mean(cudaStream_t s, float* x, int T, int F);

// LayerNorm over the last dim (rows of length D), fp32 statistics (two pass).
void layer_norm(cudaStream_t s, const float* x, float* y, int rows, int D, const float* w, const float* b,
                float eps);

// --- audio front ends ---
// resample_poly(x, 2, 3) exactly as scipy's upfirdn (float32 products and sums, ascending input index).
// taps: 61 float32 coefficients already multiplied by up (=2). n_out = ceil(2n/3).
void resample_poly_2_3(cudaStream_t s, const float* x, int n, const float* taps61, float* y, int n_out);
// torchaudio sinc resample 24k->16k: kernel[2][23], width 10, pad (10, 13), stride 3; n_out = ceil(2n/3).
void resample_sinc_3_2(cudaStream_t s, const float* x, int n, const float* kernel2x23, float* y, int n_out);
// Frames for a DFT: frame f = x[f*hop - pad_left + i] (reflect padding when reflect, else zeros), times win[i],
// written to out[f][0:n_fft] (zero beyond win_len). Optional Kaldi steps (dc removal + pre-emphasis 0.97).
void make_frames(cudaStream_t s, const float* x, int n, int n_frames, int hop, int pad_left, bool reflect,
                 int win_len, const float* win, int n_fft, bool kaldi, float* out);
// Real DFT basis [n_fft][2*bins]: cols [0,bins) = cos, [bins,2bins) = -sin (exact angles from fp64).
void make_dft_basis(cudaStream_t s, int n_fft, int bins, float* basis);
// mag[f][k] from re/im columns of spec[f][2*bins]: mode 0: sqrt(re^2+im^2+1e-9); 1: re^2+im^2; 2: |z|^2 via hypot
void spec_mag(cudaStream_t s, const float* spec, int frames, int bins, int mode, float* mag);
// Element log ops: mode 0: log(max(x,1e-5)); 1: log(max(x, FLT_EPSILON)); 2: log10(max(x,1e-10)).
void log_floor(cudaStream_t s, float* x, long long n, int mode);
// S3 log-mel finish: x = max(x, max(x)-8); x = (x+4)/4 (global max over the whole tensor).
void s3_logmel_finish(cudaStream_t s, float* x, long long n, float* scratch);

// --- tokenizer pieces ---
// Rotary embedding in place on q/k rows [T][H*64] (half-rotation layout), angles t*freq in fp32.
void rotary(cudaStream_t s, float* x, int T, int H, const float* freqs32);
// Depthwise conv k=31 (pad 15/15) over time on v[T][C] plus residual: out = conv(v) + v.
void fsmn(cudaStream_t s, const float* v, const float* w, float* out, int T, int C, int K);
// Softmax attention, no mask: q,k,v [T][H*64] (already scaled), out [T][H*64].
void attention(cudaStream_t s, const float* q, const float* k, const float* v, float* out, int T, int H);
// Same via batched GEMMs (scores [H][T][T] workspace) + fp32 row softmax: PyTorch's own order of operations.
void attention_gemm(cudaStream_t s, const float* q, const float* k, const float* v, float* out, int T, int H,
                    float* scores);
// FSQ: h[T][8] -> tanh*0.999, round-half-even, +1, sum 3^i -> code.
void fsq_codes(cudaStream_t s, const float* h, int T, int32_t* codes);
void scale_inplace(cudaStream_t s, float* x, long long n, float a);
// x[T][C] (time-major) -> y[C][T] and back.
void transpose(cudaStream_t s, const float* x, float* y, int R, int C);
// y[r][c] = x[r][c] for a copy into a wider buffer (concat along rows of a channel-major tensor).
void copy2d(cudaStream_t s, const float* x, int ldx, float* y, int ldy, int rows, int cols);
// Tile/crop helper: y[i] = x[i % n] for i < n_out.
void tile(cudaStream_t s, const float* x, int n, float* y, int n_out);

void check(cudaError_t e, const char* what);

}  // namespace kt2n::voice
