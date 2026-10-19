// RMSNorm, fused residual-add + RMSNorm, embedding gather (+ speaker
// injection) and embedding + RMSNorm (+ the step's RoPE table). See
// include/kt2n/kernels/ops.h.
//
// One CTA of 128 threads per row; thread t owns columns c*1024 + 8t + j
// (production `_rms_rows` / `_residual_rms` Triton layout: sizePerThread 8,
// 4 warps, checked in the TTGIR), so the fp32 sum of squares is accumulated
// in exactly production's order.
#include "kt2n/kernels/ops.h"
#include "ops_common.cuh"

namespace {

using namespace kt2n_ops_detail;

constexpr int kThreads = 128;

template <int R>
__device__ __forceinline__ void load_row(const __nv_bfloat16* p, float (&f)[R][8]) {
#pragma unroll
  for (int c = 0; c < R; ++c) {
    const uint4 u = *reinterpret_cast<const uint4*>(p + c * 1024 + 8 * threadIdx.x);
    unpack8(u, f[c]);
  }
}

template <int R>
__device__ __forceinline__ void store_row(__nv_bfloat16* p, const float (&f)[R][8]) {
#pragma unroll
  for (int c = 0; c < R; ++c)
    *reinterpret_cast<uint4*>(p + c * 1024 + 8 * threadIdx.x) = pack8(f[c]);
}

template <int R, bool PDL>
__global__ void __launch_bounds__(kThreads)
rmsnorm_kernel(const __nv_bfloat16* __restrict__ x, int64_t x_stride,
               const __nv_bfloat16* __restrict__ w, float eps,
               __nv_bfloat16* out, int64_t out_stride) {
  __shared__ float red[4];
  const int64_t row = blockIdx.x;
  float wf[R][8];
  load_row<R>(w, wf);
  pdl_wait<PDL>();
  float xf[R][8];
  load_row<R>(x + row * x_stride, xf);
  pdl_trigger<PDL>();
  float y[R][8];
  rmsnorm_row_128<R>(xf, wf, eps, red, y);
  store_row<R>(out + row * out_stride, y);
}

template <int R, bool PDL>
__global__ void __launch_bounds__(kThreads)
residual_rmsnorm_kernel(const __nv_bfloat16* residual, int64_t residual_stride,
                        const __nv_bfloat16* __restrict__ delta, int64_t delta_stride,
                        const __nv_bfloat16* __restrict__ w, float eps,
                        __nv_bfloat16* h_out, int64_t h_out_stride,
                        __nv_bfloat16* out, int64_t out_stride) {
  __shared__ float red[4];
  const int64_t row = blockIdx.x;
  float wf[R][8];
  load_row<R>(w, wf);
  pdl_wait<PDL>();
  float hf[R][8], df[R][8];
  load_row<R>(residual + row * residual_stride, hf);
  load_row<R>(delta + row * delta_stride, df);
  pdl_trigger<PDL>();
#pragma unroll
  for (int c = 0; c < R; ++c) {
#pragma unroll
    for (int j = 0; j < 8; ++j) hf[c][j] = rbf(__fadd_rn(hf[c][j], df[c][j]));
  }
  float y[R][8];
  rmsnorm_row_128<R>(hf, wf, eps, red, y);
  if (h_out != nullptr) store_row<R>(h_out + row * h_out_stride, hf);
  store_row<R>(out + row * out_stride, y);
}

// Source row for an embedding: the speaker vector when this row is injected,
// else the table row, else NULL (id out of range -> zeros).
__device__ __forceinline__ const __nv_bfloat16* embed_source(
    const int64_t* ids, int64_t row, const __nv_bfloat16* table, int64_t vocab, int hidden,
    const __nv_bfloat16* speakers, int64_t speaker_stride, const int32_t* speaker_index,
    int speaker_row) {
  if (speakers != nullptr) {
    if (speaker_index != nullptr) {
      const int32_t s = speaker_index[row];
      if (s >= 0) return speakers + static_cast<int64_t>(s) * speaker_stride;
    } else if (row == speaker_row) {
      return speakers;
    }
  }
  const int64_t id = ids[row];
  if (id < 0 || id >= vocab) return nullptr;
  return table + id * static_cast<int64_t>(hidden);
}

template <bool PDL>
__global__ void __launch_bounds__(kThreads)
embed_kernel(const int64_t* ids, const __nv_bfloat16* __restrict__ table, int64_t vocab,
             int hidden, const __nv_bfloat16* speakers, int64_t speaker_stride,
             const int32_t* speaker_index, int speaker_row, __nv_bfloat16* out,
             int64_t out_stride) {
  const int64_t row = blockIdx.x;
  pdl_wait<PDL>();
  const __nv_bfloat16* src = embed_source(ids, row, table, vocab, hidden, speakers,
                                          speaker_stride, speaker_index, speaker_row);
  pdl_trigger<PDL>();
  __nv_bfloat16* dst = out + row * out_stride;
  for (int col = 8 * threadIdx.x; col < hidden; col += 8 * kThreads) {
    const uint4 u = src ? *reinterpret_cast<const uint4*>(src + col) : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(dst + col) = u;
  }
}

template <int R, bool PDL>
__global__ void __launch_bounds__(kThreads)
embed_rmsnorm_kernel(const int64_t* ids, const __nv_bfloat16* __restrict__ table, int64_t vocab,
                     int hidden, const __nv_bfloat16* speakers, int64_t speaker_stride,
                     const int32_t* speaker_index, int speaker_row,
                     const __nv_bfloat16* __restrict__ w, float eps,
                     __nv_bfloat16* hidden_out, int64_t hidden_stride,
                     __nv_bfloat16* normed_out, int64_t normed_stride,
                     const int64_t* positions, const float* inv_freq, __nv_bfloat16* cos_out,
                     __nv_bfloat16* sin_out, int64_t cs_stride) {
  __shared__ float red[4];
  const int64_t row = blockIdx.x;
  float wf[R][8];
  load_row<R>(w, wf);
  const bool rope = cos_out != nullptr && threadIdx.x < 64;
  const float freq = rope ? inv_freq_at(inv_freq, threadIdx.x) : 0.0f;
  pdl_wait<PDL>();
  const int64_t pos = rope ? positions[row] : 0;
  const __nv_bfloat16* src = embed_source(ids, row, table, vocab, hidden, speakers,
                                          speaker_stride, speaker_index, speaker_row);
  float xf[R][8];
  if (src != nullptr) {
    load_row<R>(src, xf);
  } else {
#pragma unroll
    for (int c = 0; c < R; ++c) {
#pragma unroll
      for (int j = 0; j < 8; ++j) xf[c][j] = 0.0f;
    }
  }
  pdl_trigger<PDL>();
  if (rope) {
    // This step's RoPE table for the row (kt2n_ops_rope_cos_sin's values).
    float c, sn;
    rope_cs(freq, pos, c, sn);
    const int i = threadIdx.x;
    cos_out[row * cs_stride + i] = f2bf(c);
    sin_out[row * cs_stride + i] = f2bf(sn);
    if (cs_stride >= 128) {
      cos_out[row * cs_stride + 64 + i] = f2bf(c);
      sin_out[row * cs_stride + 64 + i] = f2bf(sn);
    }
  }
  store_row<R>(hidden_out + row * hidden_stride, xf);
  float y[R][8];
  rmsnorm_row_128<R>(xf, wf, eps, red, y);
  store_row<R>(normed_out + row * normed_stride, y);
}

// n in {1024, 2048, 4096, 8192}: production's BLOCK = next_power_of_2(n) == n,
// so the Triton layout (and our reduction order) is the full-tile one.
bool norm_width_ok(int n) {
  const int r = n / 1024;
  return (n % 1024) == 0 && (r == 1 || r == 2 || r == 4 || r == 8);
}

bool rows16(const void* p, int64_t stride) { return aligned_to(p, 16) && (stride % 8) == 0; }

// Dispatch on (n / 1024, pdl): calls f.run<R, PDL>().
template <typename F>
cudaError_t dispatch_norm(int n, int pdl, F& f) {
  switch (n / 1024) {
    case 1: return pdl ? f.template run<1, true>() : f.template run<1, false>();
    case 2: return pdl ? f.template run<2, true>() : f.template run<2, false>();
    case 4: return pdl ? f.template run<4, true>() : f.template run<4, false>();
    case 8: return pdl ? f.template run<8, true>() : f.template run<8, false>();
    default: return cudaErrorInvalidValue;
  }
}

struct RmsLaunch {
  const __nv_bfloat16* x;
  int64_t xs;
  const __nv_bfloat16* w;
  float eps;
  __nv_bfloat16* out;
  int64_t os;
  int rows;
  int pdl;
  cudaStream_t s;
  template <int R, bool P>
  cudaError_t run() {
    return launch_k(rmsnorm_kernel<R, P>, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0, s,
                    pdl, x, xs, w, eps, out, os);
  }
};

struct ResLaunch {
  const __nv_bfloat16* r;
  int64_t rs;
  const __nv_bfloat16* d;
  int64_t ds;
  const __nv_bfloat16* w;
  float eps;
  __nv_bfloat16* h;
  int64_t hs;
  __nv_bfloat16* out;
  int64_t os;
  int rows;
  int pdl;
  cudaStream_t s;
  template <int R, bool P>
  cudaError_t run() {
    return launch_k(residual_rmsnorm_kernel<R, P>, dim3(static_cast<unsigned>(rows)),
                    dim3(kThreads), 0, s, pdl, r, rs, d, ds, w, eps, h, hs, out, os);
  }
};

struct EmbLaunch {
  const int64_t* ids;
  const __nv_bfloat16* table;
  int64_t vocab;
  int hidden;
  const __nv_bfloat16* spk;
  int64_t spk_stride;
  const int32_t* spk_index;
  int spk_row;
  const __nv_bfloat16* w;
  float eps;
  __nv_bfloat16* h;
  int64_t hs;
  __nv_bfloat16* nrm;
  int64_t ns;
  const int64_t* pos;
  const float* inv;
  __nv_bfloat16* c;
  __nv_bfloat16* sn;
  int64_t css;
  int rows;
  int pdl;
  cudaStream_t s;
  template <int R, bool P>
  cudaError_t run() {
    return launch_k(embed_rmsnorm_kernel<R, P>, dim3(static_cast<unsigned>(rows)), dim3(kThreads),
                    0, s, pdl, ids, table, vocab, hidden, spk, spk_stride, spk_index, spk_row, w,
                    eps, h, hs, nrm, ns, pos, inv, c, sn, css);
  }
};

}  // namespace

extern "C" cudaError_t kt2n_ops_rmsnorm(const __nv_bfloat16* x, int64_t x_stride,
                                        const __nv_bfloat16* w, float eps,
                                        __nv_bfloat16* out, int64_t out_stride,
                                        int rows, int n, int pdl, cudaStream_t stream) {
  if (rows < 0 || !norm_width_ok(n)) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;  // before the buffer checks: an empty call may pass null buffers
  if (!x || !w || !out) return cudaErrorInvalidValue;
  if (!rows16(x, x_stride) || !rows16(out, out_stride) || !aligned_to(w, 16)) return cudaErrorInvalidValue;
  RmsLaunch l{x, x_stride, w, eps, out, out_stride, rows, pdl, stream};
  return dispatch_norm(n, pdl, l);
}

extern "C" cudaError_t kt2n_ops_residual_rmsnorm(const __nv_bfloat16* residual, int64_t residual_stride,
                                                 const __nv_bfloat16* delta, int64_t delta_stride,
                                                 const __nv_bfloat16* w, float eps,
                                                 __nv_bfloat16* h_out, int64_t h_out_stride,
                                                 __nv_bfloat16* out, int64_t out_stride,
                                                 int rows, int n, int pdl, cudaStream_t stream) {
  if (rows < 0 || !norm_width_ok(n)) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!residual || !delta || !w || !out) return cudaErrorInvalidValue;
  if (!rows16(residual, residual_stride) || !rows16(delta, delta_stride) || !rows16(out, out_stride) ||
      !aligned_to(w, 16))
    return cudaErrorInvalidValue;
  if (h_out && !rows16(h_out, h_out_stride)) return cudaErrorInvalidValue;
  ResLaunch l{residual, residual_stride, delta, delta_stride, w, eps, h_out, h_out_stride, out,
              out_stride, rows, pdl, stream};
  return dispatch_norm(n, pdl, l);
}

extern "C" cudaError_t kt2n_ops_embed(const int64_t* ids, int rows,
                                      const __nv_bfloat16* table, int64_t vocab, int hidden,
                                      const __nv_bfloat16* speakers, int64_t speaker_stride,
                                      const int32_t* speaker_index, int speaker_row,
                                      __nv_bfloat16* out, int64_t out_stride,
                                      int pdl, cudaStream_t stream) {
  if (rows < 0 || hidden <= 0 || (hidden % 8) != 0 || vocab <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!ids || !table || !out) return cudaErrorInvalidValue;
  if (!aligned_to(table, 16) || !rows16(out, out_stride)) return cudaErrorInvalidValue;
  if (speakers && !rows16(speakers, speaker_index ? speaker_stride : 0)) return cudaErrorInvalidValue;
  return launch_k(pdl ? embed_kernel<true> : embed_kernel<false>, dim3(static_cast<unsigned>(rows)),
                  dim3(kThreads), 0, stream, pdl, ids, table, vocab, hidden, speakers,
                  speaker_stride, speaker_index, speaker_row, out, out_stride);
}

extern "C" cudaError_t kt2n_ops_embed_rmsnorm(const int64_t* ids, int rows,
                                              const __nv_bfloat16* table, int64_t vocab, int hidden,
                                              const __nv_bfloat16* speakers, int64_t speaker_stride,
                                              const int32_t* speaker_index, int speaker_row,
                                              const __nv_bfloat16* w, float eps,
                                              __nv_bfloat16* hidden_out, int64_t hidden_stride,
                                              __nv_bfloat16* normed_out, int64_t normed_stride,
                                              const int64_t* positions, const float* inv_freq,
                                              __nv_bfloat16* cos_out, __nv_bfloat16* sin_out,
                                              int64_t cs_stride, int pdl, cudaStream_t stream) {
  if (rows < 0 || !norm_width_ok(hidden) || vocab <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!ids || !table || !w || !hidden_out || !normed_out) return cudaErrorInvalidValue;
  if (!aligned_to(table, 16) || !aligned_to(w, 16) || !rows16(hidden_out, hidden_stride) ||
      !rows16(normed_out, normed_stride))
    return cudaErrorInvalidValue;
  if (speakers && !rows16(speakers, speaker_index ? speaker_stride : 0)) return cudaErrorInvalidValue;
  if ((cos_out != nullptr) != (sin_out != nullptr)) return cudaErrorInvalidValue;
  if (cos_out && (!positions || cs_stride < 64)) return cudaErrorInvalidValue;
  EmbLaunch l{ids, table, vocab, hidden, speakers, speaker_stride, speaker_index, speaker_row, w, eps,
              hidden_out, hidden_stride, normed_out, normed_stride, positions, inv_freq, cos_out,
              sin_out, cs_stride, rows, pdl, stream};
  return dispatch_norm(hidden, pdl, l);
}
