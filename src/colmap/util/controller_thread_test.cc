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

#include "colmap/util/controller_thread.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Custom barrier implementation for deterministic testing
class Barrier {
 public:
  Barrier() : Barrier(2) {}

  explicit Barrier(const size_t count)
      : threshold_(count), count_(count), generation_(0) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto current_generation = generation_;
    if (!--count_) {
      ++generation_;
      count_ = threshold_;
      condition_.notify_all();
    } else {
      condition_.wait(lock, [this, current_generation] {
        return current_generation != generation_;
      });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  const size_t threshold_;
  size_t count_;
  size_t generation_;
};

// Simple test controller for basic tests
class SimpleController : public BaseController {
 public:
  enum { WORK_CALLBACK = 1 };

  SimpleController() { RegisterCallback(WORK_CALLBACK); }

  void Run() override {
    run_called_ = true;
    Callback(WORK_CALLBACK);
  }

  bool RunCalled() const { return run_called_; }

 private:
  bool run_called_ = false;
};

// Controller with barrier support for synchronization
class BarrierController : public BaseController {
 public:
  Barrier start_barrier;
  Barrier stop_barrier;
  Barrier end_barrier;

  void Run() override {
    start_barrier.Wait();
    stop_barrier.Wait();
    if (!CheckIfStopped()) {
      end_barrier.Wait();
    }
  }
};

// Controller with pause support
class PauseController : public BaseController {
 public:
  Barrier start_barrier;
  Barrier pause_barrier;
  Barrier paused_barrier;
  Barrier resumed_barrier;
  Barrier end_barrier;

  void Run() override {
    start_barrier.Wait();
    pause_barrier.Wait();
    paused_barrier.Wait();
    // This will block if paused
    CheckIfStopped();
    resumed_barrier.Wait();
    end_barrier.Wait();
  }
};

TEST(ControllerThread, GetController) {
  auto controller = std::make_shared<SimpleController>();
  ControllerThread<SimpleController> thread(controller);
  EXPECT_EQ(thread.GetController(), controller);
}

TEST(ControllerThread, RunExecutesControllerRun) {
  auto controller = std::make_shared<SimpleController>();
  ControllerThread<SimpleController> thread(controller);
  EXPECT_FALSE(controller->RunCalled());
  thread.Start();
  thread.Wait();
  EXPECT_TRUE(controller->RunCalled());
}

TEST(ControllerThread, ControllerCallbacksExecute) {
  auto controller = std::make_shared<SimpleController>();
  ControllerThread<SimpleController> thread(controller);
  int callback_count = 0;
  controller->AddCallback(SimpleController::WORK_CALLBACK,
                          [&callback_count]() { ++callback_count; });
  thread.Start();
  thread.Wait();
  EXPECT_EQ(callback_count, 1);
}

TEST(ControllerThread, StopDuringExecution) {
  auto controller = std::make_shared<BarrierController>();
  ControllerThread<BarrierController> thread(controller);

  thread.Start();
  controller->start_barrier.Wait();

  // Thread is now running
  EXPECT_TRUE(thread.IsRunning());

  // Stop the thread
  thread.Stop();
  controller->stop_barrier.Wait();

  thread.Wait();

  // Thread should be stopped
  EXPECT_TRUE(thread.IsStopped());
  EXPECT_TRUE(thread.IsFinished());
}

TEST(ControllerThread, IsStoppedIncludesPauseCheck) {
  auto controller = std::make_shared<PauseController>();
  ControllerThread<PauseController> thread(controller);

  thread.Start();
  controller->start_barrier.Wait();

  // Pause the thread
  thread.Pause();
  controller->pause_barrier.Wait();
  controller->paused_barrier.Wait();

  // Wait for thread to be paused
  while (!thread.IsPaused()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  EXPECT_TRUE(thread.IsPaused());

  // Resume the thread
  thread.Resume();
  controller->resumed_barrier.Wait();

  EXPECT_FALSE(thread.IsPaused());

  controller->end_barrier.Wait();
  thread.Wait();
}

TEST(ControllerThread, ThreadStateProgression) {
  auto controller = std::make_shared<BarrierController>();
  ControllerThread<BarrierController> thread(controller);

  // Initial state
  EXPECT_FALSE(thread.IsStarted());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  thread.Start();
  controller->start_barrier.Wait();

  // After start
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_TRUE(thread.IsRunning());
  EXPECT_FALSE(thread.IsFinished());

  controller->stop_barrier.Wait();
  controller->end_barrier.Wait();
  thread.Wait();

  // After finish
  EXPECT_TRUE(thread.IsStarted());
  EXPECT_FALSE(thread.IsRunning());
  EXPECT_TRUE(thread.IsFinished());
}

}  // namespace
}  // namespace colmap
