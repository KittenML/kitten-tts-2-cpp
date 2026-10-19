// CUDA smoke API (kt2n_cuda): device discovery plus a reference-faithful GPU
// unpack of a pack_exact ternary matrix, used to prove the toolchain, the
// static runtime and the arch list work end to end. Plain C ABI so the Python
// module and tests need no CUDA headers. Every function returns a cudaError_t
// value (0 = success) unless noted.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Number of visible CUDA devices; 0 when there is no driver / no device
// (never fails).
int kt2n_cuda_device_count(void);

// cudaRuntimeGetVersion of the statically linked runtime (e.g. 12080), or 0.
int kt2n_cuda_runtime_version(void);

// cudaDriverGetVersion (e.g. 13000 for driver 580), or 0 without a driver.
int kt2n_cuda_driver_version(void);

// Device name and compute capability of device `dev`. name gets at most
// name_len bytes including the terminating NUL.
int kt2n_cuda_device_info(int dev, int* major, int* minor, char* name, int name_len);

// Unpack codes int32 [n, k/16] + scales bf16 [n, k/128] into bf16 [n, k], all
// DEVICE pointers, on `stream` (a cudaStream_t; NULL = legacy default stream).
// Same decoding as kt2n::ref::unpack_ternary (code 3 -> 0). k % 128 == 0.
int kt2n_cuda_unpack_ternary_launch(const int32_t* d_codes, const uint16_t* d_scales, int64_t n, int64_t k,
                                    uint16_t* d_out, void* stream);

// Host-buffer convenience wrapper: copies in, launches, copies out, frees.
// Synchronous. Used by the GPU smoke test.
int kt2n_cuda_smoke_unpack(const int32_t* codes, const uint16_t* scales, int64_t n, int64_t k, uint16_t* out);

// Human-readable message for an error code returned above.
const char* kt2n_cuda_error_string(int err);

#ifdef __cplusplus
}  // extern "C"
#endif
