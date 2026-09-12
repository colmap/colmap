// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/opengl_utils.h"

#include "colmap/util/cancellation.h"

#include <QApplication>
#include <atomic>
#include <csignal>
#include <thread>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(OpenGLContextManager, Nominal) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  OpenGLContextManager manager;

  std::thread thread([&manager]() {
    EXPECT_TRUE(manager.MakeCurrent());
    EXPECT_TRUE(manager.MakeCurrent());
    qApp->exit();
  });

  app.exec();
  thread.join();
}

TEST(RunThreadWithOpenGLContext, Nominal) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   private:
    void Run() { EXPECT_TRUE(opengl_context_.MakeCurrent()); }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}

TEST(RunThreadWithOpenGLContext, PreservesSignalHandler) {
  ScopedSignalHandler signal_handler;

  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   public:
    bool ObservedStop() const { return observed_stop_.load(); }

   private:
    void Run() override {
      EXPECT_TRUE(opengl_context_.MakeCurrent());
      std::raise(SIGINT);
      observed_stop_.store(IsStopped());
    }

    OpenGLContextManager opengl_context_;
    std::atomic<bool> observed_stop_{false};
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);

  EXPECT_TRUE(thread.ObservedStop());
  EXPECT_EQ(signal_handler.ReceivedSignal(), SIGINT);
}

}  // namespace
}  // namespace colmap
