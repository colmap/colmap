// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/logging.h"

#include <gmock/gmock.h>
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

bool CheckIn(const double val) {
  CHECK_OPTION_IN(val, 0, 1);
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

TEST(CheckOptionIn, Nominal) {
  EXPECT_TRUE(CheckIn(0));
  EXPECT_TRUE(CheckIn(0.5));
  EXPECT_TRUE(CheckIn(1));
  EXPECT_FALSE(CheckIn(-0.1));
  EXPECT_FALSE(CheckIn(1.1));
}

TEST(ExceptionLogging, MessageContent) {
  try {
    THROW_CHECK(1 == 2);
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string(e.what()),
                testing::HasSubstr("Check failed: 1 == 2"));
  }

  try {
    THROW_CHECK_EQ(1, 2) << "custom suffix";
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string(e.what()),
                testing::HasSubstr("Check failed: 1 == 2 (1 vs. 2)"));
    EXPECT_THAT(std::string(e.what()), testing::HasSubstr("custom suffix"));
  }

  try {
    LOG(FATAL_THROW) << "fatal message";
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_THAT(std::string(e.what()), testing::HasSubstr("fatal message"));
  }
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
