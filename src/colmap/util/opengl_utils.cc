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

#include "colmap/util/opengl_utils.h"

#include "colmap/util/logging.h"

namespace colmap {

#if defined(COLMAP_GUI_ENABLED)
OpenGLContextManager::OpenGLContextManager(int opengl_major_version,
                                           int opengl_minor_version)
    : parent_thread_(QThread::currentThread()),
      current_thread_(nullptr),
      make_current_action_(new QAction(this)) {
  CHECK_NOTNULL(QCoreApplication::instance());
  CHECK_EQ(QCoreApplication::instance()->thread(), QThread::currentThread());

  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setMajorVersion(opengl_major_version);
  format.setMinorVersion(opengl_minor_version);
  format.setSamples(4);
  format.setProfile(QSurfaceFormat::CompatibilityProfile);
  context_.setFormat(format);

  surface_.create();
  CHECK(context_.create());
  context_.makeCurrent(&surface_);
  CHECK(context_.isValid()) << "Could not create valid OpenGL context";

  connect(
      make_current_action_,
      &QAction::triggered,
      this,
      [this]() {
        CHECK_NOTNULL(current_thread_);
        context_.doneCurrent();
        context_.moveToThread(current_thread_);
      },
      Qt::BlockingQueuedConnection);
}

bool OpenGLContextManager::MakeCurrent() {
  current_thread_ = QThread::currentThread();
  make_current_action_->trigger();
  context_.makeCurrent(&surface_);
  return context_.isValid();
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
    fprintf(stderr,
            "OpenGL error [%s, line %i]: GL_%s",
            file,
            line,
            error_name.c_str());
    error_code = glGetError();
  }
}

#endif

}  // namespace colmap
