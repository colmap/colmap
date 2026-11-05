// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CudaDeviceHasComputeCapability, InvalidDevice) {
  const int num_devices = GetNumCudaDevices();
  EXPECT_FALSE(CudaDeviceHasComputeCapability(-1, /*major=*/0));
  EXPECT_FALSE(CudaDeviceHasComputeCapability(num_devices, /*major=*/0));
  if (num_devices == 0) {
    // Without a device, the initialization error is sticky.
    GTEST_SKIP() << "No CUDA device available";
  }
  // The failed queries must not leave an error for later CUDA_CHECK() calls.
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaDeviceHasComputeCapability, ValidDevice) {
  const int num_devices = GetNumCudaDevices();
  if (num_devices == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  EXPECT_TRUE(CudaDeviceHasComputeCapability(0, /*major=*/0));
  EXPECT_FALSE(CudaDeviceHasComputeCapability(0, /*major=*/100));
  // A device with compute capability 8.x also has 7.x.
  if (CudaDeviceHasComputeCapability(0, /*major=*/8)) {
    EXPECT_TRUE(CudaDeviceHasComputeCapability(0, /*major=*/7));
  }
}

TEST(GetAvailableCudaMemoryBytes, InvalidAndValidDevice) {
  const int num_devices = GetNumCudaDevices();
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  EXPECT_FALSE(GetAvailableCudaMemoryBytes(-1, &free_bytes, &total_bytes));
  EXPECT_FALSE(
      GetAvailableCudaMemoryBytes(num_devices, &free_bytes, &total_bytes));
  if (num_devices == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_TRUE(GetAvailableCudaMemoryBytes(0, &free_bytes, &total_bytes));
  EXPECT_GT(total_bytes, 0);
  EXPECT_LE(free_bytes, total_bytes);
  int active_device = -1;
  ASSERT_EQ(cudaGetDevice(&active_device), cudaSuccess);
  EXPECT_EQ(active_device, 0);
}

}  // namespace
}  // namespace colmap
