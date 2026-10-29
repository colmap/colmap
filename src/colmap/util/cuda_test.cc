// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CudaDeviceSupportsBf16, InvalidDevice) {
  EXPECT_FALSE(CudaDeviceSupportsBf16(-1));
  EXPECT_FALSE(CudaDeviceSupportsBf16(GetNumCudaDevices()));
  if (GetNumCudaDevices() == 0) {
    // Without a device, the initialization error is sticky.
    GTEST_SKIP() << "No CUDA device available";
  }
  // The failed queries must not leave an error for later CUDA_CHECK() calls.
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace
}  // namespace colmap
