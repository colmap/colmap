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

#include "colmap/controllers/base_controller.h"

#include "colmap/util/logging.h"

namespace colmap {

BaseController::BaseController() {}

void BaseController::AddCallback(const int id,
                                 const std::function<void()>& func) {
  CHECK(func);
  CHECK_GT(callbacks_.count(id), 0) << "Callback not registered";
  callbacks_.at(id).push_back(func);
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

void BaseController::SetCheckIfStoppedFunc(const std::function<bool()>& func) {
  check_if_stopped_fn_ = func;
}

bool BaseController::CheckIfStopped() {
  if (check_if_stopped_fn_)
    return check_if_stopped_fn_();
  else
    return false;
}

}  // namespace colmap
