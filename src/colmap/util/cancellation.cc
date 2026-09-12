// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cancellation.h"

#include "colmap/util/logging.h"

#include <atomic>
#include <cstdlib>

namespace colmap {
namespace {

std::atomic<int> received_signal{0};
std::atomic<bool> signal_handler_active{false};
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
  bool expected = false;
  THROW_CHECK(signal_handler_active.compare_exchange_strong(expected, true))
      << "ScopedSignalHandler instances cannot be nested";
  received_signal.store(0, std::memory_order_relaxed);
  previous_sigint_handler_ = std::signal(SIGINT, HandleSignal);
  previous_sigterm_handler_ = std::signal(SIGTERM, HandleSignal);
}

ScopedSignalHandler::~ScopedSignalHandler() {
  std::signal(SIGINT, previous_sigint_handler_);
  std::signal(SIGTERM, previous_sigterm_handler_);
  received_signal.store(0, std::memory_order_relaxed);
  signal_handler_active.store(false);
}

int ScopedSignalHandler::ReceivedSignal() const {
  return received_signal.load(std::memory_order_relaxed);
}

int ScopedSignalHandler::GetExitCode() const {
  const int signal = ReceivedSignal();
  return signal == 0 ? EXIT_SUCCESS : 128 + signal;
}

bool ScopedSignalHandler::IsInterruptRequested() {
  return received_signal.load(std::memory_order_relaxed) != 0;
}

}  // namespace colmap
