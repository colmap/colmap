// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/hash_containers.h"

#include <functional>
#include <list>

namespace colmap {

// Base class for controllers that separates the run logic from thread
// management. Follows a similar callback pattern as Thread in
// util/threading.h. Supports templating in ControllerThread at
// util/controller_thread.h.
class BaseController {
 public:
  BaseController();
  virtual ~BaseController() = default;

  // Set callbacks that can be triggered within the main run function.
  void AddCallback(int id, std::function<void()> func);

  // Call back to the function with the specified name, if it exists.
  void Callback(int id) const;

  // This is the main run function to be implemented by the child class.
  virtual void Run() = 0;

  // Set the function used to check if the thread is stopped.
  void SetCheckIfStoppedFunc(std::function<bool()> func);
  // Check if the thread is stopped.
  bool CheckIfStopped();

 protected:
  // Register a new callback. Note that only registered callbacks can be
  // set/reset and called from within the controller. Hence, this method should
  // be called from the derived controller constructor.
  void RegisterCallback(int id);

 private:
  // list of callbacks
  NodeHashMap<int, std::list<std::function<void()>>> callbacks_;
  // check_if_stop function
  std::function<bool()> check_if_stopped_fn_;
};

}  // namespace colmap
