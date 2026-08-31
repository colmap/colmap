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
