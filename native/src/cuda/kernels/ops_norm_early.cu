// kt2n prof-step: residual add + RMSNorm with the PDL trigger at kernel start (include/kt2n/kernels/ops_norm_early.h).
//
// residual_rmsnorm_early_kernel is k2's residual_rmsnorm_kernel (src/cuda/kernels/ops_norm.cu) copied verbatim,
// same helpers (ops_common.cuh), same loads, same per-element operations, same reduction order, same stores.
// The only change is WHERE griddepcontrol.launch_dependents executes: first instruction (PDL build) instead of
// after the post-wait loads.  That lets the next projection launch (and pull its weights) while the projection
// before this norm is still running.  It cannot change a value: launch_dependents only lets the dependent grid be
// scheduled; the dependent's griddepcontrol.wait still waits for this grid's completion.
#include "kt2n/kernels/ops_norm_early.h"

#include "ops_common.cuh"

namespace {

using namespace kt2n_ops_detail;

constexpr int kThreads = 128;

// ops_norm.cu's row helpers (production `_rms_rows` layout: thread t owns columns c*1024 + 8t + j).
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
residual_rmsnorm_early_kernel(const __nv_bfloat16* residual, int64_t residual_stride,
                              const __nv_bfloat16* __restrict__ delta, int64_t delta_stride,
                              const __nv_bfloat16* __restrict__ w, float eps,
                              __nv_bfloat16* h_out, int64_t h_out_stride,
                              __nv_bfloat16* out, int64_t out_stride) {
  pdl_trigger<PDL>();  // <- the only difference from k2's kernel (there: after the two load_row calls below)
  __shared__ float red[4];
  const int64_t row = blockIdx.x;
  float wf[R][8];
  load_row<R>(w, wf);
  pdl_wait<PDL>();
  float hf[R][8], df[R][8];
  load_row<R>(residual + row * residual_stride, hf);
  load_row<R>(delta + row * delta_stride, df);
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

bool norm_width_ok(int n) {
  const int r = n / 1024;
  return (n % 1024) == 0 && (r == 1 || r == 2 || r == 4 || r == 8);
}

bool rows16(const void* p, int64_t stride) { return aligned_to(p, 16) && (stride % 8) == 0; }

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
    return launch_k(residual_rmsnorm_early_kernel<R, P>, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0, s,
                    pdl, r, rs, d, ds, w, eps, h, hs, out, os);
  }
};

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

}  // namespace

// Argument checks identical to kt2n_ops_residual_rmsnorm.
extern "C" cudaError_t kt2n_ops_residual_rmsnorm_early(const __nv_bfloat16* residual, int64_t residual_stride,
                                                       const __nv_bfloat16* delta, int64_t delta_stride,
                                                       const __nv_bfloat16* w, float eps,
                                                       __nv_bfloat16* h_out, int64_t h_out_stride,
                                                       __nv_bfloat16* out, int64_t out_stride,
                                                       int rows, int n, int pdl, cudaStream_t stream) {
  if (rows < 0 || !norm_width_ok(n)) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;  // before the buffer checks: an empty call may pass null buffers
  if (!residual || !delta || !w || !out) return cudaErrorInvalidValue;
  if (!rows16(residual, residual_stride) || !rows16(delta, delta_stride) || !rows16(out, out_stride) ||
      !aligned_to(w, 16))
    return cudaErrorInvalidValue;
  if (h_out && !rows16(h_out, h_out_stride)) return cudaErrorInvalidValue;
  ResLaunch l{residual, residual_stride, delta, delta_stride, w, eps, h_out, h_out_stride, out,
              out_stride, rows, pdl, stream};
  return dispatch_norm(n, pdl, l);
}
