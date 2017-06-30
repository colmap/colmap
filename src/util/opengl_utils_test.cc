// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "util/opengl_utils"
#include "util/testing.h"

#include <thread>

#include <QApplication>

#include "util/opengl_utils.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestOpenGLContextManager) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasMachineDisplay()) {
    return;
  }

  OpenGLContextManager manager;

  std::thread thread([&manager]() {
    manager.MakeCurrent();
    manager.MakeCurrent();
    qApp->exit();
  });

  app.exec();
  thread.join();
}

BOOST_AUTO_TEST_CASE(TestRunThreadWithOpenGLContext) {
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  if (!OpenGLContextManager::HasMachineDisplay()) {
    return;
  }

  class TestThread : public Thread {
   private:
    void Run() { opengl_context_.MakeCurrent(); }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  RunThreadWithOpenGLContext(&thread);
}
