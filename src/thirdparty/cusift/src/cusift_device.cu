// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yonatan Simson (see ../LICENSE)

// COLMAP-added shim (not upstream cusift) -- see ../README.md.

#include "cusift_device.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "cudautils.h"

namespace {

struct DevicePool {
  cudaMemPool_t pool = nullptr;
  // False when the device could not create one and the default pool is used.
  bool owned = false;
};

std::atomic<bool> failPoolCreationForTesting{false};

DevicePool CreateDevicePool(int dev)
{
  cudaMemPoolProps props = {};
  props.allocType = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = dev;
  DevicePool result;
  if (!failPoolCreationForTesting) {
    result.owned = cudaMemPoolCreate(&result.pool, &props) == cudaSuccess;
    if (!result.owned) cudaGetLastError();   // clear the error
  }
  if (result.owned) {
    uint64_t threshold = UINT64_MAX;   // keep memory for reuse across images
    safeCall(cudaMemPoolSetAttribute(result.pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  } else {
    // Use the default pool as is.
    safeCall(cudaDeviceGetDefaultMemPool(&result.pool, dev));
  }
  return result;
}

// One pool per device, created on first use and kept for the process
// lifetime (see cusiftMemPool in cudautils.h).
DevicePool CurrentDevicePool()
{
  int dev = 0;
  safeCall(cudaGetDevice(&dev));
  static std::mutex mutex;
  static std::vector<DevicePool> pools;
  std::lock_guard<std::mutex> lock(mutex);
  if (dev >= (int)pools.size()) pools.resize(dev + 1);
  if (pools[dev].pool == nullptr) pools[dev] = CreateDevicePool(dev);
  return pools[dev];
}

}  // namespace

cudaMemPool_t cusiftMemPool() { return CurrentDevicePool().pool; }

namespace cusift {

int CurrentDevice() { return cusiftCurrentDevice(); }

void InitMemoryPool() { cusiftInitMemPool(); }

void FailPoolCreationForTesting() { failPoolCreationForTesting = true; }

void ReleaseUnusedMemory() {
  const DevicePool devicePool = CurrentDevicePool();
  // The default pool is shared with the rest of the process: leave it alone.
  if (!devicePool.owned) return;
  // Lets the stream-ordered frees already queued complete, so their memory
  // counts as unused. Errors are ignored: this runs from destructors.
  if (cudaStreamSynchronize(0) != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  cudaMemPoolTrimTo(devicePool.pool, 0);
}

MemoryPoolUsage GetMemoryPoolUsage() {
  const DevicePool devicePool = CurrentDevicePool();
  if (!devicePool.owned) return {};
  cudaMemPool_t pool = devicePool.pool;
  uint64_t used = 0;
  uint64_t reserved = 0;
  safeCall(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
  safeCall(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
  return {static_cast<std::size_t>(used), static_cast<std::size_t>(reserved)};
}

}  // namespace cusift
