// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#pragma once

#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <atomic>
#include <climits>
#include <functional>
#include <future>
#include <list>
#include <queue>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace colmap {

// Reimplementation of threading with thread-related functions outside
// controller Following util/threading.h
// BaseController that supports templating in ControllerThread
class BaseController {
 public:
  BaseController();
  virtual ~BaseController() = default;

  enum {
    STARTED_CALLBACK = INT_MIN,
    FINISHED_CALLBACK,

    // thread-related callbacks from here, only useful in the Thread class,
    // empty as default
    LOCK_MUTEX_CALLBACK,
    BLOCK_IF_PAUSED_CALLBACK,
    SIGNAL_SETUP_CALLBACK,
    CHECK_VALID_SETUP_CALLBACK,
  };

  // Set callbacks that can be triggered within the main run function.
  void AddCallback(int id, const std::function<void()>& func);

  // This is the main run function to be implemented by the child class. If you
  // are looping over data and want to support the pause operation, call
  // `BlockIfPaused` at appropriate places in the loop. To support the stop
  // operation, check the `IsStopped` state and early return from this method.
  virtual void Run() = 0;

  // wrapped function for threading
  void RunFunc();

  // check if the thread is stopped
  void SetCheckIfStoppedFunc(const std::function<bool()>& func);
  bool CheckIfStopped();

  // To be called from outside. This blocks the caller until the thread is
  // setup, i.e. it signaled that its setup was valid or not. If it never gives
  // this signal, this call will block the caller infinitely. Check whether
  // setup is valid. Note that the result is only meaningful if the thread gives
  // a setup signal.
  bool CheckValidSetup();

  // test if setup is called
  bool SetupCalled() const { return setup_; }

 protected:
  // Register a new callback. Note that only registered callbacks can be
  // set/reset and called from within the thread. Hence, this method should be
  // called from the derived thread constructor.
  void RegisterCallback(int id);

  // Call back to the function with the specified name, if it exists.
  void Callback(int id) const;

  // Signal that the thread is setup. Only call this function once.
  void SignalValidSetup();
  void SignalInvalidSetup();

 private:
  bool setup_ = false;
  bool setup_valid_ = false;
  // list of callbacks
  std::unordered_map<int, std::list<std::function<void()>>> callbacks_;
  // check_if_stop function
  std::function<bool()> check_if_stopped_fn;
};

// Helper class to create single threads with simple controls
// Similar usage as ``Thread`` class in util/threading.h except for
// initialization. e.g.,
//
// std::shared_ptr<Controller> controller = std::make_shared<Controller>(args);
// std::unique_ptr<ControllerThread<Controller>> thread =
// std::make_unique<ControllerThread<Controller>>(controller);
//
//
template <class Controller>
class ControllerThread {
  // check if the Controller class is inherited from BaseController
  static_assert(std::is_base_of<BaseController, Controller>::value,
                "The controller needs to be inherited from BaseController");

 public:
  struct ThreadStatus {
    // Check the status
    bool IsStarted() const { return started; }
    bool IsStopped() const { return stopped; }
    bool IsPaused() const { return paused; }
    bool IsRunning() const { return started && !pausing && !finished; }
    bool IsFinished() const { return finished; }

    // Update the status
    void Start() {
      started = true;
      stopped = false;
      paused = false;
      pausing = false;
      finished = false;
    }

    bool started = false;
    bool stopped = false;
    bool paused = false;
    bool pausing = false;
    bool finished = false;
  };

  // Check the state of the thread.
  bool IsStarted() { return status_.IsStarted(); }
  bool IsStopped() {
    BlockIfPaused();
    return status_.IsStopped();
  }
  bool IsPaused() { return status_.IsPaused(); }
  bool IsRunning() { return status_.IsRunning(); }
  bool IsFinished() { return status_.IsFinished(); }

  explicit ControllerThread(std::shared_ptr<Controller> controller) {
    controller_ = controller;
    controller_->AddCallback(BaseController::FINISHED_CALLBACK,
                             [&]() { status_.finished = true; });
    controller_->AddCallback(BaseController::LOCK_MUTEX_CALLBACK, [&]() {
      std::unique_lock<std::mutex> lock(mutex_);
    });
    controller_->AddCallback(BaseController::SIGNAL_SETUP_CALLBACK,
                             [&]() { setup_condition_.notify_all(); });
    controller_->AddCallback(BaseController::CHECK_VALID_SETUP_CALLBACK, [&]() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (controller_->SetupCalled()) {
        setup_condition_.wait(lock);
      }
    });
    controller_->SetCheckIfStoppedFunc([&]() { return IsStopped(); });
  }
  ~ControllerThread() = default;

  void Start() {
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK(!status_.started || status_.finished);
    Wait();
    thread_ = std::thread(&ControllerThread::RunFunc, this);
    status_.Start();
  }

  void Stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    status_.stopped = true;
  }

  void Pause() {
    std::unique_lock<std::mutex> lock(mutex_);
    status_.paused = true;
    Resume();
  }

  void Resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (status_.paused) {
      status_.paused = false;
      pause_condition_.notify_all();
    }
  }

  void Wait() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void BlockIfPaused() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (status_.paused) {
      status_.pausing = true;
      pause_condition_.wait(lock);
      status_.pausing = false;
    }
  }

  void AddCallback(int id, const std::function<void()>& func) {
    controller_->AddCallback(id, func);
  }

 protected:
  // Get the unique identifier of the current thread.
  std::thread::id GetThreadId() const { return std::this_thread::get_id(); }

 private:
  // Wrapper around the main run function of the controller to set the finished
  // flag.
  void RunFunc() { controller_->RunFunc(); }

  ThreadStatus status_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable pause_condition_;
  std::condition_variable setup_condition_;

  std::shared_ptr<Controller> controller_;
};

}  // namespace colmap
