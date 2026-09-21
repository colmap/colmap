// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/cancellation.h"

#include "colmap/util/threading.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

namespace colmap {
namespace {

volatile std::sig_atomic_t test_signal = 0;

void RecordTestSignal(const int signal) { test_signal = signal; }

class StoppableThread : public Thread {
 public:
  bool ObservedStop() const { return observed_stop_.load(); }

 private:
  void Run() override {
    while (!IsStopped()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    observed_stop_.store(true);
  }

  std::atomic<bool> observed_stop_{false};
};

TEST(CancellationToken, Cancel) {
  CancellationToken token;
  EXPECT_FALSE(token.IsCancelled());
  token.Cancel();
  EXPECT_TRUE(token.IsCancelled());
}

TEST(ScopedSignalHandler, RecordsFirstSignal) {
  ScopedSignalHandler signal_handler;
  EXPECT_FALSE(ScopedSignalHandler::IsInterruptRequested());
  EXPECT_EQ(signal_handler.GetExitCode(), EXIT_SUCCESS);
  std::raise(SIGINT);
  EXPECT_TRUE(ScopedSignalHandler::IsInterruptRequested());
  EXPECT_EQ(signal_handler.ReceivedSignal(), SIGINT);
  EXPECT_EQ(signal_handler.GetExitCode(), 128 + SIGINT);
}

TEST(ScopedSignalHandler, SecondSignalTerminatesImmediately) {
#if defined(_WIN32)
  EXPECT_DEATH(
      {
        ScopedSignalHandler signal_handler;
        std::raise(SIGINT);
        std::raise(SIGINT);
      },
      "");
#else
  EXPECT_EXIT(
      {
        ScopedSignalHandler signal_handler;
        std::raise(SIGINT);
        std::raise(SIGINT);
      },
      testing::ExitedWithCode(128 + SIGINT),
      "");
#endif
}

TEST(ScopedSignalHandler, RestoresPreviousHandlerAndClearsState) {
  test_signal = 0;
  const auto previous_handler = std::signal(SIGINT, RecordTestSignal);
  {
    ScopedSignalHandler signal_handler;
    std::raise(SIGINT);
    EXPECT_TRUE(ScopedSignalHandler::IsInterruptRequested());
  }

  EXPECT_FALSE(ScopedSignalHandler::IsInterruptRequested());
  std::raise(SIGINT);
  EXPECT_EQ(test_signal, SIGINT);
  std::signal(SIGINT, previous_handler);
}

TEST(ScopedSignalHandler, RejectsNestedInstances) {
  ScopedSignalHandler signal_handler;
  EXPECT_THROW(
      { const ScopedSignalHandler nested_signal_handler; },
      std::invalid_argument);

  std::raise(SIGINT);
  EXPECT_EQ(signal_handler.ReceivedSignal(), SIGINT);
}

TEST(ScopedSignalHandler, ThreadHandlesSignal) {
  ScopedSignalHandler signal_handler;
  StoppableThread thread;
  thread.Start();
  std::raise(SIGINT);
  thread.Wait();

  EXPECT_TRUE(thread.ObservedStop());
  EXPECT_EQ(signal_handler.ReceivedSignal(), SIGINT);
}

}  // namespace
}  // namespace colmap
