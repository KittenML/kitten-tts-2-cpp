#pragma once

#include <cuda_runtime.h>

#include <array>

namespace kt2n::cuda {

struct DeviceCaps {
  int major = 0;
  int minor = 0;
  int shared_bytes = 0;
  bool clusters = false;
  bool pdl() const { return major >= 9; }
};

// Device properties are immutable. Cache them per host thread without a lock in the launch path.
inline DeviceCaps device_caps(int device) {
  static thread_local std::array<DeviceCaps, 64> cache{};
  if (device >= 0 && device < int(cache.size()) && cache[device].major) return cache[device];
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return {};
  DeviceCaps caps{prop.major, prop.minor, int(prop.sharedMemPerBlockOptin), false};
  if (prop.major >= 9) {
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, device) == cudaSuccess)
      caps.clusters = supported != 0;
  }
  if (device >= 0 && device < int(cache.size())) cache[device] = caps;
  return caps;
}

inline DeviceCaps current_caps() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return {};
  return device_caps(device);
}

inline bool pdl_enabled(bool requested) { return requested && current_caps().pdl(); }

}  // namespace kt2n::cuda
