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
#include "colmap/util/threading.h"
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

  // Set callbacks that can be triggered within the main run function.
  void AddCallback(int id, const std::function<void()>& func);

  // This is the main run function to be implemented by the child class. If you
  // are looping over data and want to support the pause operation, call
  // `BlockIfPaused` at appropriate places in the loop. To support the stop
  // operation, check the `IsStopped` state and early return from this method.
  virtual void Run() = 0;

  // check if the thread is stopped
  void SetCheckIfStoppedFunc(const std::function<bool()>& func);
  bool CheckIfStopped();

 protected:
  // Register a new callback. Note that only registered callbacks can be
  // set/reset and called from within the thread. Hence, this method should be
  // called from the derived thread constructor.
  void RegisterCallback(int id);

  // Call back to the function with the specified name, if it exists.
  void Callback(int id) const;

 private:
  // list of callbacks
  std::unordered_map<int, std::list<std::function<void()>>> callbacks_;
  // check_if_stop function
  std::function<bool()> check_if_stopped_fn_;
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
class ControllerThread : public Thread {
  // check if the Controller class is inherited from BaseController
  static_assert(std::is_base_of<BaseController, Controller>::value,
                "The controller needs to be inherited from BaseController");

 public:
  explicit ControllerThread(std::shared_ptr<Controller> controller)
      : controller_(std::move(controller)) {
    controller_->SetCheckIfStoppedFunc([&]() { return IsStopped(); });
  }
  ~ControllerThread() = default;

  // get the handle to the controller in ControllerThread
  const std::shared_ptr<Controller> GetController() { return controller_; }

  // do BlockIfPaused() every time before checking IsStopped()
  bool IsStopped() {
    BlockIfPaused();
    return Thread::IsStopped();
  }

 private:
  void Run() override { controller_->Run(); }
  std::shared_ptr<Controller> controller_;
};

}  // namespace colmap
