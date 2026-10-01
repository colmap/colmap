// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/util/opengl_utils.h"

#include "colmap/util/logging.h"

namespace colmap {

#if defined(COLMAP_GUI_ENABLED)

OpenGLContextManager::OpenGLContextManager(int opengl_major_version,
                                           int opengl_minor_version)
    : parent_thread_(QThread::currentThread()),
      current_thread_(nullptr),
      make_current_action_(new QAction(this)) {
  THROW_CHECK_NOTNULL(QCoreApplication::instance());
  THROW_CHECK_EQ(QCoreApplication::instance()->thread(),
                 QThread::currentThread());

  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  format.setMajorVersion(opengl_major_version);
  format.setMinorVersion(opengl_minor_version);
  format.setSamples(4);
  format.setProfile(QSurfaceFormat::CompatibilityProfile);
  context_.setFormat(format);

  surface_.create();
  THROW_CHECK(context_.create());
  context_.makeCurrent(&surface_);
  THROW_CHECK(context_.isValid()) << "Could not create valid OpenGL context";

  connect(
      make_current_action_,
      &QAction::triggered,
      this,
      [this]() {
        THROW_CHECK_NOTNULL(current_thread_);
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
    THROW_CHECK_NOTNULL(QCoreApplication::instance())->exit();
  });
  THROW_CHECK_NOTNULL(QCoreApplication::instance())->exec();
  opengl_thread.join();
  // Make sure that all triggered OpenGLContextManager events are processed in
  // case the application exits before the contexts were made current.
  QCoreApplication::processEvents();
}

#endif

}  // namespace colmap
