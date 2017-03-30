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

#ifndef COLMAP_SRC_UI_POINT_PAINTER_H_
#define COLMAP_SRC_UI_POINT_PAINTER_H_

#include <QtCore>
#include <QtOpenGL>

namespace colmap {

class PointPainter {
 public:
  PointPainter();
  ~PointPainter();

  struct Data {
    Data() : x(0), y(0), z(0), r(0), g(0), b(0), a(0) {}
    Data(const float x_, const float y_, const float z_, const float r_,
         const float g_, const float b_, const float a_)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_), a(a_) {}

    float x, y, z;
    float r, g, b, a;
  };

  void Setup();
  void Upload(const std::vector<PointPainter::Data>& data);
  void Render(const QMatrix4x4& pmv_matrix, const float point_size);

 private:
  QOpenGLShaderProgram shader_program_;
  QOpenGLVertexArrayObject vao_;
  QOpenGLBuffer vbo_;

  size_t num_geoms_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_POINT_PAINTER_H_
