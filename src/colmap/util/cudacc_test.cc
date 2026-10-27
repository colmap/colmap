// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CudaSafeCall, Success) {
  EXPECT_NO_THROW(CudaSafeCall(cudaSuccess, "probe.cc", 42));
}

TEST(CudaSafeCall, InsufficientDriver) {
  try {
    CudaSafeCall(cudaErrorInsufficientDriver, "probe.cc", 42);
    FAIL() << "Expected a runtime error";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("probe.cc:42"), std::string::npos);
    EXPECT_NE(message.find("error " + std::to_string(static_cast<int>(
                                          cudaErrorInsufficientDriver))),
              std::string::npos);
    EXPECT_NE(message.find("driver"), std::string::npos);
    EXPECT_NE(message.find("COLMAP build"), std::string::npos);
    EXPECT_EQ(message.find("(null)"), std::string::npos);
  }
}

TEST(CudaSafeCall, UnknownErrorPreservesCode) {
  try {
    CudaSafeCall(static_cast<cudaError_t>(123456), "probe.cc", 7);
    FAIL() << "Expected a runtime error";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("123456"), std::string::npos);
    EXPECT_NE(message.find("probe.cc:7"), std::string::npos);
    EXPECT_EQ(message.find("(null)"), std::string::npos);
    EXPECT_EQ(message.find("Update the NVIDIA driver"), std::string::npos);
  }
}

}  // namespace
}  // namespace colmap
