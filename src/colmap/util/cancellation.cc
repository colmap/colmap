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

#include "colmap/util/cancellation.h"

#include <atomic>
#include <cstdlib>

namespace colmap {
namespace {

std::atomic<int> received_signal{0};
static_assert(std::atomic<int>::is_always_lock_free);

void HandleSignal(const int signal) {
  int expected = 0;
  if (!received_signal.compare_exchange_strong(
          expected, signal, std::memory_order_relaxed)) {
    std::_Exit(128 + signal);
  }
}

}  // namespace

void CancellationToken::Cancel() { is_cancelled_.store(true); }

bool CancellationToken::IsCancelled() const { return is_cancelled_.load(); }

ScopedSignalHandler::ScopedSignalHandler() {
  received_signal.store(0, std::memory_order_relaxed);
  previous_sigint_handler_ = std::signal(SIGINT, HandleSignal);
  previous_sigterm_handler_ = std::signal(SIGTERM, HandleSignal);
}

ScopedSignalHandler::~ScopedSignalHandler() {
  std::signal(SIGINT, previous_sigint_handler_);
  std::signal(SIGTERM, previous_sigterm_handler_);
  received_signal.store(0, std::memory_order_relaxed);
}

int ScopedSignalHandler::ReceivedSignal() const {
  return received_signal.load(std::memory_order_relaxed);
}

bool ScopedSignalHandler::IsInterruptRequested() {
  return received_signal.load(std::memory_order_relaxed) != 0;
}

}  // namespace colmap
