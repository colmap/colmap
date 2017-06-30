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

#include "util/opengl_utils.h"

#include <QApplication>

#include "util/logging.h"

namespace colmap {

OpenGLContextManager::OpenGLContextManager()
    : parent_thread_(QThread::currentThread()),
      current_thread_(nullptr),
      make_current_action_(new QAction(this)) {
  CHECK_NOTNULL(QCoreApplication::instance());
  CHECK_EQ(QCoreApplication::instance()->thread(), QThread::currentThread());

  surface_.create();
  CHECK(context_.create());
  context_.makeCurrent(&surface_);
  CHECK(context_.isValid()) << "Could not create valid OpenGL context";

  connect(make_current_action_, &QAction::triggered, this,
          [this]() {
            CHECK_NOTNULL(current_thread_);
            context_.doneCurrent();
            context_.moveToThread(current_thread_);
          },
          Qt::BlockingQueuedConnection);
}

void OpenGLContextManager::MakeCurrent() {
  current_thread_ = QThread::currentThread();
  make_current_action_->trigger();
  context_.makeCurrent(&surface_);
  CHECK(context_.isValid()) << "Could not make current valid OpenGL context";
}

bool OpenGLContextManager::HasMachineDisplay() {
  QOffscreenSurface surface;
  QOpenGLContext context;
  surface.create();
  return context.create();
}

void RunThreadWithOpenGLContext(Thread* thread) {
  std::thread opengl_thread([thread]() {
    thread->Start();
    thread->Wait();
    CHECK_NOTNULL(QCoreApplication::instance())->exit();
  });
  CHECK_NOTNULL(QCoreApplication::instance())->exec();
  opengl_thread.join();
  // Make sure that all triggered OpenGLContextManager events are processed in
  // case the application exits before the contexts were made current.
  QCoreApplication::processEvents();
}

void GLError(const char* file, const int line) {
  GLenum error_code(glGetError());
  while (error_code != GL_NO_ERROR) {
    std::string error_name;
    switch (error_code) {
      case GL_INVALID_OPERATION:
        error_name = "INVALID_OPERATION";
        break;
      case GL_INVALID_ENUM:
        error_name = "INVALID_ENUM";
        break;
      case GL_INVALID_VALUE:
        error_name = "INVALID_VALUE";
        break;
      case GL_OUT_OF_MEMORY:
        error_name = "OUT_OF_MEMORY";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        error_name = "INVALID_FRAMEBUFFER_OPERATION";
        break;
      default:
        error_name = "UNKNOWN_ERROR";
        break;
    }
    fprintf(stderr, "OpenGL error [%s, line %i]: GL_%s", file, line,
            error_name.c_str());
    error_code = glGetError();
  }
}

}  // namespace colmap
