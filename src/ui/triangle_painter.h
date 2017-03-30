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

#ifndef COLMAP_SRC_UI_TRIANGLE_PAINTER_H_
#define COLMAP_SRC_UI_TRIANGLE_PAINTER_H_

#include <QtCore>
#include <QtOpenGL>

#include "ui/point_painter.h"

namespace colmap {

class TrianglePainter {
 public:
  TrianglePainter();
  ~TrianglePainter();

  struct Data {
    Data() {}
    Data(const PointPainter::Data& p1, const PointPainter::Data& p2,
         const PointPainter::Data& p3)
        : point1(p1), point2(p2), point3(p3) {}

    PointPainter::Data point1;
    PointPainter::Data point2;
    PointPainter::Data point3;
  };

  void Setup();
  void Upload(const std::vector<TrianglePainter::Data>& data);
  void Render(const QMatrix4x4& pmv_matrix);

 private:
  QOpenGLShaderProgram shader_program_;
  QOpenGLVertexArrayObject vao_;
  QOpenGLBuffer vbo_;

  size_t num_geoms_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_TRIANGLE_PAINTER_H_
