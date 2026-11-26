// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cudacc.h"

#include <array>
#include <cstdlib>
#include <exception>
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

#if defined(COLMAP_CUDA_TIMER_WRAP)
namespace {

struct EventCalls {
  bool armed = false;
  int fail_operation = 0;
  bool fail_cleanup = false;
  int creates = 0;
  int records = 0;
  int destroys = 0;
  cudaEvent_t recorded = nullptr;
  cudaStream_t stream = nullptr;
  std::array<cudaEvent_t, 2> destroyed{};
};

EventCalls calls;
std::array<int, 2> event_tokens;

cudaEvent_t TestEvent(const size_t index) {
  return reinterpret_cast<cudaEvent_t>(&event_tokens[index]);
}

}  // namespace

extern "C" cudaError_t __real_cudaEventCreate(cudaEvent_t*);
extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t __real_cudaEventDestroy(cudaEvent_t);

extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t* event) {
  if (!calls.armed) return __real_cudaEventCreate(event);
  ++calls.creates;
  *event = TestEvent(calls.creates - 1);
  // A failed call does not establish ownership, even if it wrote the output.
  if (calls.fail_operation == calls.creates) return cudaErrorMemoryAllocation;
  return cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t event,
                                              cudaStream_t stream) {
  if (!calls.armed) return __real_cudaEventRecord(event, stream);
  ++calls.records;
  calls.recorded = event;
  calls.stream = stream;
  return calls.fail_operation == 3 ? cudaErrorInvalidResourceHandle
                                   : cudaSuccess;
}

extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
  if (!calls.armed) return __real_cudaEventDestroy(event);
  if (calls.destroys < static_cast<int>(calls.destroyed.size())) {
    calls.destroyed[calls.destroys] = event;
  }
  ++calls.destroys;
  return calls.fail_cleanup ? cudaErrorLaunchFailure : cudaSuccess;
}
#endif  // COLMAP_CUDA_TIMER_WRAP

#if defined(COLMAP_CUDA_TIMER_WRAP) && !defined(cudaEventCreate) && \
    !defined(cudaEventRecord) && !defined(cudaEventDestroy)
namespace colmap {
namespace {

class CudaTimerTest : public testing::Test {
 protected:
  void SetUp() override {
    calls = EventCalls{};
    calls.armed = true;
  }
  void TearDown() override { calls = EventCalls{}; }
};

TEST_F(CudaTimerTest, SuccessfulTimerReleasesBothEvents) {
  {
    CudaTimer timer;
    EXPECT_EQ(calls.creates, 2);
    EXPECT_EQ(calls.records, 1);
    EXPECT_EQ(calls.recorded, TestEvent(0));
    EXPECT_EQ(calls.stream, nullptr);
    EXPECT_EQ(calls.destroys, 0);
  }
  EXPECT_EQ(calls.destroys, 2);
  EXPECT_EQ(calls.destroyed[0], TestEvent(0));
  EXPECT_EQ(calls.destroyed[1], TestEvent(1));
}

TEST_F(CudaTimerTest, ConstructionFailureCleansUpAndPreservesOriginalError) {
  for (const int operation : {1, 2, 3}) {
    for (const bool cleanup_failure : {false, true}) {
      SCOPED_TRACE(testing::Message() << operation << ":" << cleanup_failure);
      calls = EventCalls{};
      calls.armed = true;
      calls.fail_operation = operation;
      calls.fail_cleanup = cleanup_failure;
      const auto code = operation == 3 ? cudaErrorInvalidResourceHandle
                                       : cudaErrorMemoryAllocation;
      try {
        CudaTimer timer;
        FAIL() << "Expected timer construction to throw";
      } catch (const std::invalid_argument& error) {
        EXPECT_NE(
            std::string(error.what())
                .find("(error " + std::to_string(static_cast<int>(code)) + ")"),
            std::string::npos)
            << error.what();
      }
      EXPECT_EQ(calls.creates, operation == 1 ? 1 : 2);
      EXPECT_EQ(calls.records, operation == 3 ? 1 : 0);
      const int expected_destroys = operation - 1;
      EXPECT_EQ(calls.destroys, expected_destroys);
      for (int i = 0; i < expected_destroys; ++i) {
        EXPECT_EQ(calls.destroyed[i], TestEvent(i));
      }
    }
  }
}

#if GTEST_HAS_DEATH_TEST
// Isolate a noexcept-destructor termination from the rest of the suite.
// Returning status zero requires both cleanup attempts and the original error.
[[noreturn]] void CheckCleanupFailure(const bool unwind) {
  std::set_terminate([] { std::_Exit(86); });
  calls = EventCalls{};
  calls.armed = true;
  calls.fail_cleanup = true;
  bool caught_original = false;
  try {
    CudaTimer timer;
    if (unwind) throw std::runtime_error("original failure");
  } catch (const std::runtime_error& error) {
    caught_original = std::string(error.what()) == "original failure";
  }
  const bool expected = caught_original == unwind && calls.destroys == 2 &&
                        calls.destroyed[0] == TestEvent(0) &&
                        calls.destroyed[1] == TestEvent(1);
  std::_Exit(expected ? 0 : 1);
}

TEST_F(CudaTimerTest, DestructionFailureAttemptsBothEvents) {
  EXPECT_EXIT(CheckCleanupFailure(false), testing::ExitedWithCode(0), "");
}

TEST_F(CudaTimerTest, DestructionFailurePreservesUnwindingException) {
  EXPECT_EXIT(CheckCleanupFailure(true), testing::ExitedWithCode(0), "");
}
#else
TEST_F(CudaTimerTest, DestructionFailureControlsUnavailable) {
  GTEST_SKIP() << "Isolated termination controls require GTest death tests";
}
#endif  // GTEST_HAS_DEATH_TEST

}  // namespace
}  // namespace colmap
#else
TEST(CudaTimerTest, OwnershipFailureControlUnavailable) {
  GTEST_SKIP() << "Timer failure controls require a static CUDA owner and a "
                  "supported non-Windows/non-Apple wrap linker without event "
                  "API aliases; HIP and shared library interception are not "
                  "implemented";
}
#endif  // COLMAP_CUDA_TIMER_WRAP and unaliased event APIs
