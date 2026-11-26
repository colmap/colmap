// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <stdexcept>
#include <string>

#include <gmock/gmock.h>
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
    EXPECT_THAT(message, testing::HasSubstr("probe.cc:42"));
    EXPECT_THAT(
        message,
        testing::HasSubstr("error " + std::to_string(static_cast<int>(
                                          cudaErrorInsufficientDriver))));
    EXPECT_THAT(message, testing::HasSubstr("driver"));
    EXPECT_THAT(message, testing::HasSubstr("COLMAP build"));
    EXPECT_THAT(message, testing::Not(testing::HasSubstr("(null)")));
  }
}

TEST(CudaSafeCall, UnknownErrorPreservesCode) {
  try {
    CudaSafeCall(static_cast<cudaError_t>(123456), "probe.cc", 7);
    FAIL() << "Expected a runtime error";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_THAT(message, testing::HasSubstr("123456"));
    EXPECT_THAT(message, testing::HasSubstr("probe.cc:7"));
    EXPECT_THAT(message, testing::Not(testing::HasSubstr("(null)")));
    EXPECT_THAT(message,
                testing::Not(testing::HasSubstr("Update the NVIDIA driver")));
  }
}

}  // namespace
}  // namespace colmap
