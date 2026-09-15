// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/logging.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::string PrintingFn(const std::string& message) {
  if (message.empty()) {
    LOG(FATAL_THROW) << "Error in PrintingFn";
  }
  return message;
}

void ThrowCheck(const bool cond) { THROW_CHECK(cond) << "Error!"; }

void ThrowCheckEqual(const int val) { THROW_CHECK_EQ(val, 1) << "Error!"; }

bool CheckInRange(const double val) {
  CHECK_OPTION_IN_RANGE(val, 0, 1);
  return true;
}

TEST(ExceptionLogging, Nominal) {
  EXPECT_NO_THROW(ThrowCheck(true));
  EXPECT_THROW(ThrowCheck(false), std::invalid_argument);
  EXPECT_NO_THROW(ThrowCheckEqual(1));
  EXPECT_THROW(ThrowCheckEqual(0), std::invalid_argument);
  EXPECT_THROW(THROW_CHECK_NOTNULL(nullptr), std::invalid_argument);
  EXPECT_THROW({ LOG(FATAL_THROW) << "Error!"; }, std::invalid_argument);
  EXPECT_THROW(
      { LOG_FATAL_THROW(std::logic_error) << "Error!"; }, std::logic_error);
}

// Ensure that the condition is evaluated exactly once
// for both the positive and negative cases.
TEST(ExceptionLogging, NumConditionEvals) {
  int num_calls = 0;
  auto func = [&num_calls]() {
    ++num_calls;
    return true;
  };
  THROW_CHECK(func());
  EXPECT_EQ(num_calls, 1);
  try {
    THROW_CHECK(!func());
  } catch (...) {
    LOG(INFO) << "Caught exception";
  }
  EXPECT_EQ(num_calls, 2);
}

TEST(CheckOptionInRange, Nominal) {
  EXPECT_TRUE(CheckInRange(0));
  EXPECT_TRUE(CheckInRange(0.5));
  EXPECT_TRUE(CheckInRange(1));
  EXPECT_FALSE(CheckInRange(-0.1));
  EXPECT_FALSE(CheckInRange(1.1));
}

TEST(ExceptionLogging, Nested) {
  EXPECT_NO_THROW(PrintingFn("message"));
  EXPECT_THROW(PrintingFn(""), std::invalid_argument);
  EXPECT_THROW(
      { LOG(FATAL_THROW) << "Error: " << PrintingFn("message"); },
      std::invalid_argument);
  EXPECT_THROW(
      { LOG(FATAL_THROW) << "Error: " << PrintingFn(""); },
      std::invalid_argument);
}

}  // namespace
}  // namespace colmap
