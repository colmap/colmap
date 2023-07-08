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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#ifdef GUI_ENABLED
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

#ifdef GUI_ENABLED

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

// Get the OpenGL errors and print them to stderr.
void GLError(const char* file, int line);

#else

// Dummy implementation when GUI support is disabled
class OpenGLContextManager {
 public:
  explicit OpenGLContextManager(int opengl_major_version = 2,
                                int opengl_minor_version = 1) {}
  inline bool MakeCurrent() { return false; }
};

inline void RunThreadWithOpenGLContext(Thread* thread) {}

inline void GLError(const char* file, const int line) {}

#endif

}  // namespace colmap
