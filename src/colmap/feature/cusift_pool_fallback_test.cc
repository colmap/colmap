// SPDX-License-Identifier: BSD-3-Clause

// cusift's fallback to the device's default memory pool, on a device that
// cannot create its own. Its own executable: the fallback is chosen when a
// device's pool is first used, so the failure must be injected before any
// other cusift work in the process.

#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"

#include <cstddef>
#include <cstdint>

#include "cusift_device.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr size_t kMiB = 1024 * 1024;

uint64_t ReservedBytes(cudaMemPool_t pool) {
  uint64_t value = 0;
  THROW_CHECK_EQ(
      cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &value),
      cudaSuccess);
  return value;
}

void SetReleaseThreshold(cudaMemPool_t pool, uint64_t threshold) {
  THROW_CHECK_EQ(cudaMemPoolSetAttribute(
                     pool, cudaMemPoolAttrReleaseThreshold, &threshold),
                 cudaSuccess);
}

// Leaves `bytes` of freed memory in `pool`, held for reuse.
void HoldUnusedMemory(cudaMemPool_t pool, size_t bytes) {
  SetReleaseThreshold(pool, UINT64_MAX);
  void* buffer = nullptr;
  THROW_CHECK_EQ(cudaMallocFromPoolAsync(&buffer, bytes, pool, 0), cudaSuccess);
  THROW_CHECK_EQ(cudaFreeAsync(buffer, 0), cudaSuccess);
  THROW_CHECK_EQ(cudaStreamSynchronize(0), cudaSuccess);
}

TEST(CusiftPoolFallback, LeavesTheSharedDefaultPoolAlone) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA device available";
  cusift::FailPoolCreationForTesting();
  int device = 0;
  THROW_CHECK_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaMemPool_t pool = nullptr;
  THROW_CHECK_EQ(cudaDeviceGetDefaultMemPool(&pool, device), cudaSuccess);
  // Another user of the default pool keeps 64 MiB in it for reuse.
  uint64_t threshold_before = 0;
  THROW_CHECK_EQ(cudaMemPoolGetAttribute(
                     pool, cudaMemPoolAttrReleaseThreshold, &threshold_before),
                 cudaSuccess);
  HoldUnusedMemory(pool, 64 * kMiB);
  const uint64_t reserved_before = ReservedBytes(pool);
  ASSERT_GE(reserved_before, 64 * kMiB);

  const cusift::MemoryPoolUsage usage = cusift::GetMemoryPoolUsage();
  cusift::ReleaseUnusedMemory();

  // cusift reports only its own pool's memory, and has none...
  EXPECT_EQ(usage.used, 0);
  EXPECT_EQ(usage.reserved, 0);
  // ...and does not hand back what other users keep in the default pool.
  EXPECT_EQ(ReservedBytes(pool), reserved_before);
  SetReleaseThreshold(pool, threshold_before);
  THROW_CHECK_EQ(cudaMemPoolTrimTo(pool, 0), cudaSuccess);
}

}  // namespace
}  // namespace colmap
