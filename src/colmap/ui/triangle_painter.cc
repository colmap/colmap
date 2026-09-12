// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/ui/triangle_painter.h"

namespace colmap {

void TrianglePainter::Setup() {
  SetupShaders({{QOpenGLShader::Vertex, ":/shaders/triangles.v.glsl"},
                {QOpenGLShader::Fragment, ":/shaders/triangles.f.glsl"}});
}

void TrianglePainter::Upload(const std::vector<TrianglePainter::Data>& data) {
  UploadGeoms(data, "a_position", sizeof(PointPainter::Data));
}

void TrianglePainter::Render(const QMatrix4x4& pmv_matrix) {
  if (!BeginRender()) {
    return;
  }

  shader_program_.setUniformValue("u_pmv_matrix", pmv_matrix);

  GLFunctions()->glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(3 * num_geoms_));

  EndRender();
}

}  // namespace colmap
