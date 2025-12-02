// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/util/base_controller.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Concrete implementation of BaseController for testing
class TestController : public BaseController {
 public:
  TestController() {
    // Register callbacks during construction
    RegisterCallback(1);
    RegisterCallback(2);
    RegisterCallback(100);
  }

  void Run() override {
    // Simple run implementation that triggers callbacks
    Callback(1);
    if (!CheckIfStopped()) {
      Callback(2);
    }
  }

  // Expose RegisterCallback for testing
  void TestRegisterCallback(int id) { RegisterCallback(id); }
};

TEST(BaseController, AddCallbackToRegistered) {
  TestController controller;
  int counter = 0;
  controller.AddCallback(1, [&counter]() { counter += 1; });
  EXPECT_EQ(counter, 0);
  controller.Callback(1);
  EXPECT_EQ(counter, 1);
}

TEST(BaseController, AddMultipleCallbacks) {
  TestController controller;
  int counter = 0;
  controller.AddCallback(1, [&counter]() { counter += 1; });
  controller.AddCallback(1, [&counter]() { counter += 10; });
  controller.AddCallback(1, [&counter]() { counter += 100; });
  controller.Callback(1);
  EXPECT_EQ(counter, 111);
}

TEST(BaseController, AddCallbackToDifferentIDs) {
  TestController controller;
  int counter1 = 0;
  int counter2 = 0;
  controller.AddCallback(1, [&counter1]() { counter1 += 1; });
  controller.AddCallback(2, [&counter2]() { counter2 += 2; });
  controller.Callback(1);
  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(counter2, 0);
  controller.Callback(2);
  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(counter2, 2);
}

TEST(BaseController, CallbackWithNoCallbacksAdded) {
  TestController controller;
  // Calling a registered callback with no functions added should not crash
  EXPECT_NO_THROW(controller.Callback(1));
}

TEST(BaseController, CallbackExecutionOrder) {
  TestController controller;
  std::vector<int> execution_order;
  controller.AddCallback(
      1, [&execution_order]() { execution_order.push_back(1); });
  controller.AddCallback(
      1, [&execution_order]() { execution_order.push_back(2); });
  controller.AddCallback(
      1, [&execution_order]() { execution_order.push_back(3); });
  controller.Callback(1);
  ASSERT_EQ(execution_order.size(), 3);
  EXPECT_EQ(execution_order[0], 1);
  EXPECT_EQ(execution_order[1], 2);
  EXPECT_EQ(execution_order[2], 3);
}

TEST(BaseController, SetCheckIfStoppedFunc) {
  TestController controller;
  bool should_stop = false;

  // Set the check function
  controller.SetCheckIfStoppedFunc([&should_stop]() { return should_stop; });

  // Initially not stopped
  EXPECT_FALSE(controller.CheckIfStopped());

  // Change the flag
  should_stop = true;
  EXPECT_TRUE(controller.CheckIfStopped());
}

TEST(BaseController, CheckIfStoppedWithoutSetting) {
  TestController controller;
  // Without setting a check function, should return false
  EXPECT_FALSE(controller.CheckIfStopped());
}

TEST(BaseController, CheckIfStoppedMultipleCalls) {
  TestController controller;
  int call_count = 0;

  controller.SetCheckIfStoppedFunc([&call_count]() {
    ++call_count;
    return call_count >= 3;
  });

  EXPECT_FALSE(controller.CheckIfStopped());  // call_count = 1
  EXPECT_FALSE(controller.CheckIfStopped());  // call_count = 2
  EXPECT_TRUE(controller.CheckIfStopped());   // call_count = 3
  EXPECT_TRUE(controller.CheckIfStopped());   // call_count = 4
}

TEST(BaseController, RunMethod) {
  TestController controller;
  int counter1 = 0;
  int counter2 = 0;

  controller.AddCallback(1, [&counter1]() { ++counter1; });
  controller.AddCallback(2, [&counter2]() { ++counter2; });

  // Run should trigger callbacks
  controller.Run();

  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(counter2, 1);
}

TEST(BaseController, RunWithStopCheck) {
  TestController controller;
  int counter1 = 0;
  int counter2 = 0;

  controller.AddCallback(1, [&counter1]() { ++counter1; });
  controller.AddCallback(2, [&counter2]() { ++counter2; });

  // Set stop function to return true
  controller.SetCheckIfStoppedFunc([]() { return true; });

  // Run should trigger callback 1 but not callback 2 (due to stop check)
  controller.Run();

  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(counter2, 0);  // Should not execute due to stop
}

TEST(BaseController, ReplaceCheckIfStoppedFunc) {
  TestController controller;
  controller.SetCheckIfStoppedFunc([]() { return true; });
  EXPECT_TRUE(controller.CheckIfStopped());
  controller.SetCheckIfStoppedFunc([]() { return false; });
  EXPECT_FALSE(controller.CheckIfStopped());
}

TEST(BaseController, CallbackWithException) {
  TestController controller;
  int counter_before = 0;
  int counter_after = 0;

  controller.AddCallback(1, [&counter_before]() { ++counter_before; });
  controller.AddCallback(1,
                         []() { throw std::runtime_error("test exception"); });
  controller.AddCallback(1, [&counter_after]() { ++counter_after; });
  EXPECT_THROW(controller.Callback(1), std::runtime_error);
  EXPECT_EQ(counter_before, 1);
  EXPECT_EQ(counter_after, 0);
}

TEST(BaseController, EmptyCallbackList) {
  TestController controller;
  // Registered but no callbacks added - should not crash
  EXPECT_NO_THROW(controller.Callback(100));
}

}  // namespace
}  // namespace colmap
