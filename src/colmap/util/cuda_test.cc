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

}  // namespace
}  // namespace colmap
