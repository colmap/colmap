// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/base_controller.h"
#include "colmap/util/threading.h"

namespace colmap {

// Helper class to create single threads with simple controls
// Similar usage as ``Thread`` class in util/threading.h
//
// std::shared_ptr<Controller> controller = std::make_shared<Controller>(args);
// std::unique_ptr<ControllerThread<Controller>> thread =
//    std::make_unique<ControllerThread<Controller>>(controller);
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

  // get the handle to the controller in ControllerThread
  std::shared_ptr<Controller> GetController() { return controller_; }

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
