// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#ifndef COLMAP_SRC_UI_LINE_PAINTER_H_
#define COLMAP_SRC_UI_LINE_PAINTER_H_

#include <QtCore>
#include <QtOpenGL>

#include "ui/point_painter.h"

namespace colmap {

class LinePainter {
 public:
  LinePainter();
  ~LinePainter();

  struct Data {
    Data() {}
    Data(const PointPainter::Data& p1, const PointPainter::Data& p2)
        : point1(p1), point2(p2) {}

    PointPainter::Data point1;
    PointPainter::Data point2;
  };

  void Setup();
  void Upload(const std::vector<LinePainter::Data>& data);
  void Render(const QMatrix4x4& pmv_matrix, const int width, const int height,
              const float line_width);

 private:
  QOpenGLShaderProgram shader_program_;
  QOpenGLVertexArrayObject vao_;
  QOpenGLBuffer vbo_;

  size_t num_geoms_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_LINE_PAINTER_H_
