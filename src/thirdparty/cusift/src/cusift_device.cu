// COLMAP-added shim (not upstream cusift) -- see ../README.md.

#include "cusift_device.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "cudautils.h"

// One pool per device, created on first use and kept for the process
// lifetime (see cusiftMemPool in cudautils.h).
cudaMemPool_t cusiftMemPool()
{
  int dev = 0;
  safeCall(cudaGetDevice(&dev));
  static std::mutex mutex;
  static std::vector<cudaMemPool_t> pools;
  std::lock_guard<std::mutex> lock(mutex);
  if (dev >= (int)pools.size()) pools.resize(dev + 1, nullptr);
  if (pools[dev] == nullptr) {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = dev;
    cudaMemPool_t pool = nullptr;
    if (cudaMemPoolCreate(&pool, &props) == cudaSuccess) {
      uint64_t threshold = UINT64_MAX;   // keep memory for reuse across images
      safeCall(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    } else {
      cudaGetLastError();   // clear the error; use the default pool as is
      safeCall(cudaDeviceGetDefaultMemPool(&pool, dev));
    }
    pools[dev] = pool;
  }
  return pools[dev];
}

namespace cusift {
namespace {

// Mirrors CompareCudaDevice in colmap/util/cuda.cc: prefer higher compute
// capability, then more multiprocessors. Kept in sync deliberately so that
// gpu_index = -1 selects the same device the rest of COLMAP would select.
bool IsBetterDevice(const cudaDeviceProp& d1, const cudaDeviceProp& d2) {
  return (d1.major > d2.major) ||
         ((d1.major == d2.major) && (d1.minor > d2.minor)) ||
         ((d1.major == d2.major) && (d1.minor == d2.minor) &&
          (d1.multiProcessorCount > d2.multiProcessorCount));
}

int FindBestDevice(int num_devices) {
  std::vector<cudaDeviceProp> props(num_devices);
  std::vector<int> indices(num_devices);
  for (int id = 0; id < num_devices; ++id) {
    indices[id] = id;
    if (cudaGetDeviceProperties(&props[id], id) != cudaSuccess) {
      return 0;
    }
  }
  std::sort(indices.begin(), indices.end(), [&props](int a, int b) {
    return IsBetterDevice(props[a], props[b]);
  });
  return indices[0];
}

}  // namespace

bool InitDevice(int gpu_index) {
  int num_devices = 0;
  if (cudaGetDeviceCount(&num_devices) != cudaSuccess || num_devices == 0) {
    return false;
  }
  const int selected =
      (gpu_index >= 0) ? gpu_index : FindBestDevice(num_devices);
  if (selected >= num_devices) {
    return false;
  }
  if (cudaSetDevice(selected) != cudaSuccess) {
    return false;
  }
  cusiftInitMemPool();
  return true;
}

void ReleaseUnusedMemory() {
  // Lets the stream-ordered frees already queued complete, so their memory
  // counts as unused. Errors are ignored: this runs from destructors.
  if (cudaStreamSynchronize(0) != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  cudaMemPoolTrimTo(cusiftMemPool(), 0);
}

MemoryPoolUsage GetMemoryPoolUsage() {
  cudaMemPool_t pool = cusiftMemPool();
  uint64_t used = 0;
  uint64_t reserved = 0;
  safeCall(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
  safeCall(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
  return {static_cast<std::size_t>(used), static_cast<std::size_t>(reserved)};
}

}  // namespace cusift
