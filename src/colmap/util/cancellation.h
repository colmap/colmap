// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <atomic>
#include <csignal>

namespace colmap {

// Thread-safe cancellation state for cooperative cancellation of long-running
// operations. Cancellation tokens are single-use and remain cancelled once a
// cancellation request has been made.
class CancellationToken {
 public:
  void Cancel();
  bool IsCancelled() const;

 private:
  std::atomic<bool> is_cancelled_{false};
};

// Scoped handler for process interruption signals. The handler only records
// the first signal so that normal code can perform cleanup at a safe point. A
// second signal terminates the process immediately.
class ScopedSignalHandler {
 public:
  ScopedSignalHandler();
  ~ScopedSignalHandler();

  ScopedSignalHandler(const ScopedSignalHandler&) = delete;
  ScopedSignalHandler& operator=(const ScopedSignalHandler&) = delete;

  int ReceivedSignal() const;
  int GetExitCode() const;

  static bool IsInterruptRequested();

 private:
  using SignalHandler = void (*)(int);

  SignalHandler previous_sigint_handler_ = SIG_DFL;
  SignalHandler previous_sigterm_handler_ = SIG_DFL;
};

}  // namespace colmap
