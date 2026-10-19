// Speaker projection: Linear(in -> out, bias) + LayerNorm(out) in bf16, the
// module runtime.py builds as `spk_proj`. See include/kt2n/kernels/ops.h.
//
// LayerNorm replicates torch 2.7.1 aten/src/ATen/native/cuda/layer_norm_kernel.cu
// `vectorized_layer_norm_kernel` (bf16 in/out, fp32 math): block (32, 4),
// vec_size 4, per-thread Welford online sums over its vectors, warp
// shuffle-down combine, then the inter-warp tree through shared memory.
// torch is built with nvcc's default -fmad=true, so `a + b * c` in that
// source compiles to fma(b, c, a); the explicit __fmaf_rn calls below mirror
// that contraction (verified bitwise against torch.native_layer_norm,
// including its mean / rstd outputs).
#include "kt2n/kernels/ops.h"
#include "ops_common.cuh"

namespace {

using namespace kt2n_ops_detail;

// ------------------------------------------------------------------ Linear

constexpr int kLinWarps = 8;  // output features per CTA

__device__ __forceinline__ void load8_any(const void* x, int x_is_f32, int64_t idx, float (&f)[8]) {
  if (x_is_f32) {
    const float4* p = reinterpret_cast<const float4*>(static_cast<const float*>(x) + idx);
    const float4 a = p[0], b = p[1];
    f[0] = rbf(a.x); f[1] = rbf(a.y); f[2] = rbf(a.z); f[3] = rbf(a.w);
    f[4] = rbf(b.x); f[5] = rbf(b.y); f[6] = rbf(b.z); f[7] = rbf(b.w);
  } else {
    unpack8(*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(x) + idx), f);
  }
}

// One warp per output feature n: lane l accumulates k = 256*i + 8l + j in
// register order (i, j), fp32 (bf16 x bf16 products are exact in fp32, so
// fma == mul-then-add here), then the xor butterfly, then + bias, -> bf16.
__global__ void __launch_bounds__(kLinWarps * 32)
linear_bias_kernel(const void* x, int x_is_f32, int64_t x_stride,
                   const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ b,
                   __nv_bfloat16* out, int64_t out_stride, int in_dim, int out_dim) {
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.y * kLinWarps + (threadIdx.x >> 5);
  if (n >= out_dim) return;
  const __nv_bfloat16* wr = w + static_cast<int64_t>(n) * in_dim;
  float acc = 0.0f;
  for (int k0 = 0; k0 < in_dim; k0 += 256) {
    const int k = k0 + 8 * lane;
    float xf[8], wf[8];
    load8_any(x, x_is_f32, row * x_stride + k, xf);
    unpack8(*reinterpret_cast<const uint4*>(wr + k), wf);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = __fmaf_rn(xf[j], wf[j], acc);
  }
  acc = warp_sum_bfly(acc);
  if (lane == 0) {
    const float bias = b ? bf2f(b[n]) : 0.0f;
    out[row * out_stride + n] = f2bf(__fadd_rn(acc, bias));
  }
}

// --------------------------------------------------------------- LayerNorm

struct Welford {
  float mean, sigma2, count;
};

// cuWelfordOnlineSum: new_mean = mean + delta * (1/new_count) -> fma;
// sigma2 + delta * (val - new_mean) -> fma.
__device__ __forceinline__ Welford welford_online(float val, Welford c) {
  const float delta = __fsub_rn(val, c.mean);
  const float new_count = __fadd_rn(c.count, 1.0f);
  const float new_mean = __fmaf_rn(delta, __frcp_rn(new_count), c.mean);
  return {new_mean, __fmaf_rn(delta, __fsub_rn(val, new_mean), c.sigma2), new_count};
}

// cuWelfordCombine(dataB, dataA).
__device__ __forceinline__ Welford welford_combine(Welford B, Welford A) {
  const float delta = __fsub_rn(B.mean, A.mean);
  const float count = __fadd_rn(A.count, B.count);
  if (count > 0.0f) {
    const float coef = __frcp_rn(count);
    const float nA = __fmul_rn(A.count, coef);
    const float nB = __fmul_rn(B.count, coef);
    const float mean = __fmaf_rn(nA, A.mean, __fmul_rn(nB, B.mean));
    const float m2 = __fmul_rn(__fmul_rn(delta, delta), A.count);
    const float sigma2 = __fmaf_rn(m2, nB, __fadd_rn(A.sigma2, B.sigma2));
    return {mean, sigma2, count};
  }
  return {0.0f, 0.0f, count};
}

__global__ void __launch_bounds__(128)
layernorm_kernel(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* __restrict__ gamma,
                 const __nv_bfloat16* __restrict__ beta, float eps, __nv_bfloat16* y,
                 int64_t y_stride, float* mean_out, float* rstd_out, int n) {
  __shared__ float buf[6];  // torch: threads.y * 3 / 2 floats
  const int64_t row = blockIdx.x;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int thrx = tx + ty * 32;
  const int numx = 128;
  const int nvec = n / 4;
  const __nv_bfloat16* xr = x + row * x_stride;

  Welford wd{0.0f, 0.0f, 0.0f};
  for (int i = thrx; i < nvec; i += numx) {
    const uint2 u = *reinterpret_cast<const uint2*>(xr + 4 * i);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
    const float2 a = __bfloat1622float2(p[0]), b2 = __bfloat1622float2(p[1]);
    wd = welford_online(a.x, wd);
    wd = welford_online(a.y, wd);
    wd = welford_online(b2.x, wd);
    wd = welford_online(b2.y, wd);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const Welford wdB{__shfl_down_sync(0xffffffffu, wd.mean, offset),
                      __shfl_down_sync(0xffffffffu, wd.sigma2, offset),
                      __shfl_down_sync(0xffffffffu, wd.count, offset)};
    wd = welford_combine(wd, wdB);
  }
  float* meansigmabuf = buf;
  float* countbuf = buf + 4;
  for (int offset = 2; offset > 0; offset /= 2) {
    if (tx == 0 && ty >= offset && ty < 2 * offset) {
      const int wrt_y = ty - offset;
      meansigmabuf[2 * wrt_y] = wd.mean;
      meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
      countbuf[wrt_y] = wd.count;
    }
    __syncthreads();
    if (tx == 0 && ty < offset) {
      const Welford wdB{meansigmabuf[2 * ty], meansigmabuf[2 * ty + 1], countbuf[ty]};
      wd = welford_combine(wd, wdB);
    }
    __syncthreads();
  }
  if (tx == 0 && ty == 0) {
    meansigmabuf[0] = wd.mean;
    meansigmabuf[1] = __fdiv_rn(wd.sigma2, static_cast<float>(n));
  }
  __syncthreads();
  const float mean = meansigmabuf[0];
  const float rstd = rsqrtf(__fadd_rn(meansigmabuf[1], eps));

  __nv_bfloat16* yr = y + row * y_stride;
  for (int i = thrx; i < nvec; i += numx) {
    const uint2 u = *reinterpret_cast<const uint2*>(xr + 4 * i);
    const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(&u);
    __nv_bfloat16 o[4];
#pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      const float t = __fmul_rn(rstd, __fsub_rn(bf2f(xv[ii]), mean));
      float r;
      if (gamma != nullptr && beta != nullptr) {
        r = __fmaf_rn(bf2f(gamma[4 * i + ii]), t, bf2f(beta[4 * i + ii]));
      } else if (gamma != nullptr) {
        r = __fmul_rn(bf2f(gamma[4 * i + ii]), t);
      } else if (beta != nullptr) {
        r = __fmaf_rn(rstd, __fsub_rn(bf2f(xv[ii]), mean), bf2f(beta[4 * i + ii]));
      } else {
        r = t;
      }
      o[ii] = f2bf(r);
    }
    *reinterpret_cast<uint2*>(yr + 4 * i) = *reinterpret_cast<const uint2*>(o);
  }
  if (thrx == 0) {
    if (mean_out) mean_out[row] = mean;
    if (rstd_out) rstd_out[row] = rstd;
  }
}

}  // namespace

extern "C" cudaError_t kt2n_ops_linear_bias(const void* x, int x_is_f32, int64_t x_stride,
                                            const __nv_bfloat16* w, const __nv_bfloat16* b,
                                            __nv_bfloat16* out, int64_t out_stride, int rows,
                                            int in_dim, int out_dim, cudaStream_t stream) {
  if (rows < 0 || in_dim <= 0 || (in_dim % 256) != 0 || out_dim <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;  // before the buffer checks: an empty call may pass null buffers
  if (!x || !w || !out) return cudaErrorInvalidValue;
  if (!aligned_to(x, 16) || (x_stride % 8) != 0 || !aligned_to(w, 16)) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>((out_dim + kLinWarps - 1) / kLinWarps));
  return launch_k(linear_bias_kernel, grid, dim3(kLinWarps * 32), 0, stream, 0, x, x_is_f32,
                  x_stride, w, b, out, out_stride, in_dim, out_dim);
}

extern "C" cudaError_t kt2n_ops_layernorm(const __nv_bfloat16* x, int64_t x_stride,
                                          const __nv_bfloat16* w, const __nv_bfloat16* b,
                                          float eps, __nv_bfloat16* out, int64_t out_stride,
                                          float* mean_out, float* rstd_out, int rows, int n,
                                          cudaStream_t stream) {
  if (rows < 0 || n <= 0 || (n % 4) != 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (!x || !out) return cudaErrorInvalidValue;
  if (!aligned_to(x, 8) || (x_stride % 4) != 0 || !aligned_to(out, 8) || (out_stride % 4) != 0)
    return cudaErrorInvalidValue;
  return launch_k(layernorm_kernel, dim3(static_cast<unsigned>(rows)), dim3(32, 4), 0, stream, 0,
                  x, x_stride, w, b, eps, out, out_stride, mean_out, rstd_out, n);
}

extern "C" cudaError_t kt2n_ops_spk_proj(const void* x, int x_is_f32, int64_t x_stride,
                                         const __nv_bfloat16* w, const __nv_bfloat16* b,
                                         const __nv_bfloat16* ln_w, const __nv_bfloat16* ln_b,
                                         float ln_eps, __nv_bfloat16* out, int64_t out_stride,
                                         int rows, int in_dim, int out_dim, cudaStream_t stream) {
  if ((out_dim % 8) != 0) return cudaErrorInvalidValue;
  // The LayerNorm runs in place on `out`: check its constraints on `out` before the Linear writes anything,
  // so a bad argument launches nothing (the Linear alone accepts any `out`).
  if (rows > 0 && (!out || !aligned_to(out, 8) || (out_stride % 4) != 0)) return cudaErrorInvalidValue;
  cudaError_t e = kt2n_ops_linear_bias(x, x_is_f32, x_stride, w, b, out, out_stride, rows, in_dim,
                                       out_dim, stream);
  if (e != cudaSuccess) return e;
  return kt2n_ops_layernorm(out, out_stride, ln_w, ln_b, ln_eps, out, out_stride, nullptr, nullptr,
                            rows, out_dim, stream);
}
