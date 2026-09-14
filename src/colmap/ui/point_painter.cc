// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/point_painter.h"

#include <cstddef>

namespace colmap {

// PainterBase uploads vertices as 3 floats + 4 bytes with the color at byte
// offset 12; line and triangle painters additionally rely on Data being a
// contiguous array of vertices.
static_assert(sizeof(PointPainter::Data) ==
              3 * sizeof(float) + 4 * sizeof(uint8_t));
static_assert(offsetof(PointPainter::Data, r) == 3 * sizeof(float));

void PointPainter::Setup() {
  SetupShaders({{QOpenGLShader::Vertex, ":/shaders/points.v.glsl"},
                {QOpenGLShader::Fragment, ":/shaders/points.f.glsl"}});
}

void PointPainter::Upload(const std::vector<PointPainter::Data>& data) {
  UploadGeoms(data, "a_position", sizeof(PointPainter::Data));
}

void PointPainter::Render(const QMatrix4x4& pmv_matrix,
                          const float point_size) {
  if (!BeginRender()) {
    return;
  }

  shader_program_.setUniformValue("u_pmv_matrix", pmv_matrix);
  shader_program_.setUniformValue("u_point_size", point_size);

  GLFunctions()->glDrawArrays(GL_POINTS, 0, (GLsizei)num_geoms_);

  EndRender();
}

}  // namespace colmap
