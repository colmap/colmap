// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_OPENGL_UTILS_H_
#define COLMAP_SRC_OPENGL_UTILS_H_

#include <QAction>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QThread>
#include <QWaitCondition>

namespace colmap {

#ifdef DEBUG
#define glDebugLog() glError(__FILE__, __LINE__)
#else
#define glDebugLog()
#endif

// This class manages a thread-safe OpenGL context. Note that this class must be
// instantiated in the main Qt thread, since an OpenGL context must be created
// in it. The context can then be made current in any other thread.
class OpenGLContextManager : public QObject {
 public:
  OpenGLContextManager();

  void MakeCurrent();

 private:
  QOffscreenSurface surface_;
  QOpenGLContext context_;
  QThread* parent_thread_;
  QThread* current_thread_;
  QAction* make_current_action_;
  QMutex make_current_mutex_;
  QWaitCondition make_current_done_;
};

void GLError(const char* file, const int line);

}  // namespace colmap

#endif  // COLMAP_SRC_OPENGL_UTILS_H_
