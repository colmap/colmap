// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/base_controller.h"

#include "colmap/util/cancellation.h"
#include "colmap/util/logging.h"

namespace colmap {

BaseController::BaseController() {}

void BaseController::AddCallback(const int id, std::function<void()> func) {
  CHECK(func);
  CHECK_GT(callbacks_.count(id), 0) << "Callback not registered";
  callbacks_.at(id).push_back(std::move(func));
}

void BaseController::RegisterCallback(const int id) {
  callbacks_.emplace(id, std::list<std::function<void()>>());
}

void BaseController::Callback(const int id) const {
  CHECK_GT(callbacks_.count(id), 0) << "Callback not registered";
  for (const auto& callback : callbacks_.at(id)) {
    callback();
  }
}

void BaseController::SetCheckIfStoppedFunc(std::function<bool()> func) {
  check_if_stopped_fn_ = std::move(func);
}

bool BaseController::CheckIfStopped() {
  return ScopedSignalHandler::IsInterruptRequested() ||
         (check_if_stopped_fn_ && check_if_stopped_fn_());
}

}  // namespace colmap
