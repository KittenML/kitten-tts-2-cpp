// CUDA driver-API entry points resolved at run time through the runtime (cudaGetDriverEntryPoint), so
// libkt2n_engine.so carries no NEEDED libcuda.so.1 and a GPU-less container (the Modal image build's RTLD_NOW
// dlopen check) can load it. The runtime is linked statically and opens the driver itself when CUDA first
// initialises, exactly as before the green-context experiment linked CUDA::cuda_driver (L7 6fb3806e, 13:54).
// Only the S_bulk SM-partition experiment (KT2N_RENDER_BULK_SMS > 0) and the context scopes around its
// streams call these; the default path never resolves anything.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace kt2n::render::drv {

struct Api {
  decltype(&::cuGetErrorString) GetErrorString = nullptr;
  decltype(&::cuCtxGetCurrent) CtxGetCurrent = nullptr;
  decltype(&::cuCtxSetCurrent) CtxSetCurrent = nullptr;
  decltype(&::cuDeviceGet) DeviceGet = nullptr;
  decltype(&::cuDeviceGetDevResource) DeviceGetDevResource = nullptr;
  decltype(&::cuDevSmResourceSplitByCount) DevSmResourceSplitByCount = nullptr;
  decltype(&::cuDevResourceGenerateDesc) DevResourceGenerateDesc = nullptr;
  decltype(&::cuGreenCtxCreate) GreenCtxCreate = nullptr;
  decltype(&::cuGreenCtxDestroy) GreenCtxDestroy = nullptr;
  decltype(&::cuCtxFromGreenCtx) CtxFromGreenCtx = nullptr;
  decltype(&::cuGreenCtxStreamCreate) GreenCtxStreamCreate = nullptr;
};

namespace detail {
template <class F>
void resolve(const char* name, F& fn) {
  void* p = nullptr;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
  const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
  if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr)
    throw std::runtime_error(std::string("CUDA driver entry point ") + name + " unavailable (" +
                             (e != cudaSuccess ? cudaGetErrorString(e) : "symbol not found or driver too old") + ")");
  fn = reinterpret_cast<F>(p);
}
}  // namespace detail

// Resolved once per process, on first use (throws if the driver lacks a symbol, e.g. green contexts on an old
// driver; callers turn that into a refused create, never a crash).
inline const Api& api() {
  static Api a;
  static std::once_flag once;
  static std::string err;
  std::call_once(once, [] {
    try {
      detail::resolve("cuGetErrorString", a.GetErrorString);
      detail::resolve("cuCtxGetCurrent", a.CtxGetCurrent);
      detail::resolve("cuCtxSetCurrent", a.CtxSetCurrent);
      detail::resolve("cuDeviceGet", a.DeviceGet);
      detail::resolve("cuDeviceGetDevResource", a.DeviceGetDevResource);
      detail::resolve("cuDevSmResourceSplitByCount", a.DevSmResourceSplitByCount);
      detail::resolve("cuDevResourceGenerateDesc", a.DevResourceGenerateDesc);
      detail::resolve("cuGreenCtxCreate", a.GreenCtxCreate);
      detail::resolve("cuGreenCtxDestroy", a.GreenCtxDestroy);
      detail::resolve("cuCtxFromGreenCtx", a.CtxFromGreenCtx);
      detail::resolve("cuGreenCtxStreamCreate", a.GreenCtxStreamCreate);
    } catch (const std::exception& e) {
      err = e.what();
    }
  });
  if (!err.empty()) throw std::runtime_error(err);
  return a;
}

inline void check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* msg = nullptr;
  if (api().GetErrorString != nullptr) api().GetErrorString(r, &msg);
  throw std::runtime_error(std::string(what) + ": " + (msg ? msg : "CUDA driver error"));
}

// Makes a green context current for a scope (nullptr = nothing to do: the primary context; resolves nothing).
struct CtxScope {
  CUcontext prev = nullptr;
  bool on = false;
  explicit CtxScope(CUcontext c) {
    if (c == nullptr) return;
    const Api& d = api();
    d.CtxGetCurrent(&prev);
    if (prev == c) return;
    check(d.CtxSetCurrent(c), "cuCtxSetCurrent");
    on = true;
  }
  ~CtxScope() {
    if (on) api().CtxSetCurrent(prev);
  }
  CtxScope(const CtxScope&) = delete;
  CtxScope& operator=(const CtxScope&) = delete;
};

}  // namespace kt2n::render::drv
