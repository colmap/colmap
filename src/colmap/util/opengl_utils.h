// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if defined(COLMAP_GUI_ENABLED)
#include <QAction>
#include <QApplication>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QThread>
#include <QWaitCondition>
#endif

#include "colmap/util/threading.h"

namespace colmap {

#ifdef DEBUG
#define glDebugLog() glError(__FILE__, __LINE__)
#else
#define glDebugLog()
#endif

#if defined(COLMAP_GUI_ENABLED)

// This class manages a thread-safe OpenGL context. Note that this class must be
// instantiated in the main Qt thread, since an OpenGL context must be created
// in it. The context can then be made current in any other thread.
class OpenGLContextManager : public QObject {
 public:
  explicit OpenGLContextManager(int opengl_major_version = 2,
                                int opengl_minor_version = 1);

  // Make the OpenGL context available by moving it from the thread where it was
  // created to the current thread and making it current.
  bool MakeCurrent();

 private:
  QOffscreenSurface surface_;
  QOpenGLContext context_;
  QThread* parent_thread_;
  QThread* current_thread_;
  QAction* make_current_action_;
};

// Run and wait for the thread, that uses the OpenGLContextManager, e.g.:
//
//    class TestThread : public Thread {
//     private:
//      void Run() { opengl_context_.MakeCurrent(); }
//      OpenGLContextManager opengl_context_;
//    };
//    QApplication app(argc, argv);
//    TestThread thread;
//    RunThreadWithOpenGLContext(&thread);
//
void RunThreadWithOpenGLContext(Thread* thread);

#else

// Dummy implementation when GUI support is disabled
class OpenGLContextManager {
 public:
  explicit OpenGLContextManager(int opengl_major_version = 2,
                                int opengl_minor_version = 1) {}
  inline bool MakeCurrent() { return false; }
};

inline void RunThreadWithOpenGLContext(Thread* thread) {}

#endif

}  // namespace colmap
