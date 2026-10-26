// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CudaDeviceSupportsBf16, InvalidDevice) {
  const int num_devices = GetNumCudaDevices();
  EXPECT_FALSE(CudaDeviceSupportsBf16(-1));
  EXPECT_FALSE(CudaDeviceSupportsBf16(num_devices));
  if (num_devices == 0) {
    // Without a device, the initialization error is sticky.
    GTEST_SKIP() << "No CUDA device available";
  }
  // The failed queries must not leave an error for later CUDA_CHECK() calls.
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaDeviceHasTensorCores, InvalidDevice) {
  const int num_devices = GetNumCudaDevices();
  EXPECT_FALSE(CudaDeviceHasTensorCores(-1));
  EXPECT_FALSE(CudaDeviceHasTensorCores(num_devices));
  if (num_devices == 0) {
    // Without a device, the initialization error is sticky.
    GTEST_SKIP() << "No CUDA device available";
  }
  // The failed queries must not leave an error for later CUDA_CHECK() calls.
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  // A device with bf16 support (8.0 or newer) also has tensor cores.
  for (int device = 0; device < num_devices; ++device) {
    if (CudaDeviceSupportsBf16(device)) {
      EXPECT_TRUE(CudaDeviceHasTensorCores(device));
    }
  }
}

}  // namespace
}  // namespace colmap
