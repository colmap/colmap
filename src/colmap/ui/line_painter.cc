// Copyright (c), ETH Zurich and UNC Chapel Hill.
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
