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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif

#ifdef __clang__
#pragma clang diagnostic pop  // -Wkeyword-macro
#endif

// Reimplementation of threading with thread-related functions outside
// controller Following util/threading.h

// Core methods of a controller, wrapped by BaseController
class CoreController {
 public:
  CoreController(){};
  virtual ~CoreController() = default;

  // Set callbacks that can be triggered within the main run function.
  void AddCallback(int id, const std::function<void()>& func);

  // Call back to the function with the specified name, if it exists.
  void Callback(int id) const;

  // This is the main run function to be implemented by the child class. If you
  // are looping over data and want to support the pause operation, call
  // `BlockIfPaused` at appropriate places in the loop. To support the stop
  // operation, check the `IsStopped` state and early return from this method.
  virtual void Run() = 0;

 protected:
  // Register a new callback. Note that only registered callbacks can be
  // set/reset and called from within the thread. Hence, this method should be
  // called from the derived thread constructor.
  void RegisterCallback(int id);

 private:
  std::unordered_map<int, std::list<std::function<void()>>> callbacks_;
};

// BaseController that supports templating in ControllerThread
class BaseController : public CoreController {
 public:
  struct ThreadStatus {
    // Check the status
    bool IsStarted() { return started; }
    bool IsStopped() { return stopped; }
    bool IsPaused() { return paused; }
    bool IsRunning() { return started && !pausing && !finished; }
    bool IsFinished() { return finished; }

    // Update the status
    void Start() {
      started = true;
      stopped = false;
      paused = false;
      pausing = false;
      finished = false;
      setup = false;
      setup_valid = false;
    }

    bool started = false;
    bool stopped = false;
    bool paused = false;
    bool pausing = false;
    bool finished = false;
    bool setup = false;
    bool setup_valid = false;
  };

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

  BaseController();
  virtual ~BaseController() = default;

  ///////////////////////////////////////////////////
  // Thread-related functions
  ///////////////////////////////////////////////////
  ThreadStatus status_;
  ThreadStatus* GetThreadStatus() { return &status_; }

  // Check the state of the thread.
  bool IsStarted();
  bool IsStopped();
  bool IsPaused();
  bool IsRunning();
  bool IsFinished();

  // To be called from inside the main run function. This blocks the main
  // caller, if the thread is paused, until the thread is resumed.
  void BlockIfPaused();

  // To be called from outside. This blocks the caller until the thread is
  // setup, i.e. it signaled that its setup was valid or not. If it never gives
  // this signal, this call will block the caller infinitely. Check whether
  // setup is valid. Note that the result is only meaningful if the thread gives
  // a setup signal.
  bool CheckValidSetup();

 protected:
  // Signal that the thread is setup. Only call this function once.
  void SignalValidSetup();
  void SignalInvalidSetup();
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
  static_assert(
      std::is_base_of<BaseController, Controller>::
          value);  // check if the Controller is inherited from BaseController

 public:
  ControllerThread(std::shared_ptr<Controller> controller) {
    controller_ = controller;
    controller_->AddCallback(BaseController::LOCK_MUTEX_CALLBACK, [&]() {
      std::unique_lock<std::mutex> lock(mutex_);
    });
    controller_->AddCallback(BaseController::BLOCK_IF_PAUSED_CALLBACK, [&]() {
      std::unique_lock<std::mutex> lock(mutex_);
      auto* status = controller_->GetThreadStatus();
      if (status->paused) {
        status->pausing = true;
        pause_condition_.wait(lock);
        status->pausing = false;
      }
    });
    controller_->AddCallback(BaseController::SIGNAL_SETUP_CALLBACK,
                             [&]() { setup_condition_.notify_all(); });
    controller_->AddCallback(BaseController::CHECK_VALID_SETUP_CALLBACK, [&]() {
      std::unique_lock<std::mutex> lock(mutex_);
      auto* status = controller_->GetThreadStatus();
      if (status->setup) {
        setup_condition_.wait(lock);
      }
    });
  }
  ~ControllerThread() = default;

  void Start() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto* status = controller_->GetThreadStatus();
    CHECK(!status->started || status->finished);
    Wait();
    thread_ = std::thread(&ControllerThread::RunFunc, this);
    status->Start();
  }

  void Stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    controller_->GetThreadStatus()->stopped = true;
  }

  void Pause() {
    std::unique_lock<std::mutex> lock(mutex_);
    controller_->GetThreadStatus()->paused = true;
    Resume();
  }

  void Resume() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto* status = controller_->GetThreadStatus();
    if (status->paused) {
      status->paused = false;
      pause_condition_.notify_all();
    }
  }

  void Wait() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  bool IsStarted() { return controller_->IsStarted(); }
  bool IsStopped() { return controller_->IsStopped(); }
  bool IsPaused() { return controller_->IsPaused(); }
  bool IsRunning() { return controller_->IsRunning(); }
  bool IsFinished() { return controller_->IsFinished(); }

  void AddCallback(int id, const std::function<void()>& func) {
    controller_->AddCallback(id, func);
  }

 protected:
  // Get the unique identifier of the current thread.
  std::thread::id GetThreadId() const { return std::this_thread::get_id(); }

 private:
  // Wrapper around the main run function of the controller to set the finished
  // flag.
  void RunFunc() {
    controller_->Callback(BaseController::STARTED_CALLBACK);
    controller_->Run();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      controller_->GetThreadStatus()->finished = true;
    }
    controller_->Callback(BaseController::FINISHED_CALLBACK);
  }

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable pause_condition_;
  std::condition_variable setup_condition_;

  std::shared_ptr<Controller> controller_;
};

}  // namespace colmap
