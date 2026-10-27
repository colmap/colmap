// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cuda.h"

#include "colmap/util/cudacc.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CudaDeviceSupportsBf16Test, RejectsInvalidGpuIndex) {
  // No GPU needed: invalid input fails closed without querying a device.
  EXPECT_FALSE(CudaDeviceSupportsBf16(""));
  EXPECT_FALSE(CudaDeviceSupportsBf16("abc"));
  EXPECT_FALSE(CudaDeviceSupportsBf16("0,1"));
  EXPECT_FALSE(CudaDeviceSupportsBf16("0;1"));
}

TEST(CudaDeviceSupportsBf16Test, MatchesDeviceProperties) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "CUDA device required";
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  EXPECT_EQ(CudaDeviceSupportsBf16("0"), properties.major >= 8);
  // A negative index selects device 0, matching the ONNX provider behavior.
  EXPECT_EQ(CudaDeviceSupportsBf16("-1"), properties.major >= 8);
}

}  // namespace
}  // namespace colmap
