// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/line_painter.h"

namespace colmap {

void LinePainter::Setup() {
  SetupShaders({{QOpenGLShader::Vertex, ":/shaders/lines.v.glsl"},
                {QOpenGLShader::Geometry, ":/shaders/lines.g.glsl"},
                {QOpenGLShader::Fragment, ":/shaders/lines.f.glsl"}});
}

void LinePainter::Upload(const std::vector<LinePainter::Data>& data) {
  UploadGeoms(data, "a_pos", sizeof(PointPainter::Data));
}

void LinePainter::Render(const QMatrix4x4& pmv_matrix,
                         const int width,
                         const int height,
                         const float line_width) {
  if (!BeginRender()) {
    return;
  }

  shader_program_.setUniformValue("u_pmv_matrix", pmv_matrix);
  shader_program_.setUniformValue("u_inv_viewport",
                                  QVector2D(1.0f / width, 1.0f / height));
  shader_program_.setUniformValue("u_line_width", line_width);

  GLFunctions()->glDrawArrays(GL_LINES, 0, (GLsizei)(2 * num_geoms_));

  EndRender();
}

}  // namespace colmap
