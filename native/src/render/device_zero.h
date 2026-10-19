// Zeroing device buffers at create, ordered with every later stream.
//
// A plain cudaMemset of device memory runs on the legacy default stream and returns before it finishes; the render,
// voice and LM streams are non-blocking, so they do NOT wait for it, and a prefix build or render launched right
// after create could run while the memset is still zeroing (the render arena is ~20 GB: ~10 ms of memset). Here the memset runs on a private stream and the host waits for exactly
// that work (never for queued LM or render work), so whatever is launched afterwards sees zeros.
#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kt2n::render {

inline void zero_device(void* p, size_t bytes, const char* what) {
  if (p == nullptr || bytes == 0) return;
  cudaStream_t s = nullptr;
  cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  if (e == cudaSuccess) {
    e = cudaMemsetAsync(p, 0, bytes, s);
    const cudaError_t e2 = cudaStreamSynchronize(s);
    if (e == cudaSuccess) e = e2;
    cudaStreamDestroy(s);
  }
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace kt2n::render
