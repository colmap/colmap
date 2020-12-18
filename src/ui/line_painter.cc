// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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

#include "ui/line_painter.h"

#include "util/opengl_utils.h"

namespace colmap {

LinePainter::LinePainter() : num_geoms_(0) {}

LinePainter::~LinePainter() {
  vao_.destroy();
  vbo_.destroy();
}

void LinePainter::Setup() {
  vao_.destroy();
  vbo_.destroy();
  if (shader_program_.isLinked()) {
    shader_program_.release();
    shader_program_.removeAllShaders();
  }

  shader_program_.addShaderFromSourceFile(QOpenGLShader::Vertex,
                                          ":/shaders/lines.v.glsl");
  shader_program_.addShaderFromSourceFile(QOpenGLShader::Geometry,
                                          ":/shaders/lines.g.glsl");
  shader_program_.addShaderFromSourceFile(QOpenGLShader::Fragment,
                                          ":/shaders/lines.f.glsl");
  shader_program_.link();
  shader_program_.bind();

  vao_.create();
  vbo_.create();

#if DEBUG
  glDebugLog();
#endif
}

void LinePainter::Upload(const std::vector<LinePainter::Data>& data) {
  num_geoms_ = data.size();
  if (num_geoms_ == 0) {
    return;
  }

  vao_.bind();
  vbo_.bind();

  // Upload data array to GPU
  vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
  vbo_.allocate(data.data(),
                static_cast<int>(data.size() * sizeof(LinePainter::Data)));

  // in_position
  shader_program_.enableAttributeArray("a_pos");
  shader_program_.setAttributeBuffer("a_pos", GL_FLOAT, 0, 3,
                                     sizeof(PointPainter::Data));

  // in_color
  shader_program_.enableAttributeArray("a_color");
  shader_program_.setAttributeBuffer("a_color", GL_FLOAT, 3 * sizeof(GLfloat),
                                     4, sizeof(PointPainter::Data));

  // Make sure they are not changed from the outside
  vbo_.release();
  vao_.release();

#if DEBUG
  glDebugLog();
#endif
}

void LinePainter::Render(const QMatrix4x4& pmv_matrix, const int width,
                         const int height, const float line_width) {
  if (num_geoms_ == 0) {
    return;
  }

  shader_program_.bind();
  vao_.bind();

  shader_program_.setUniformValue("u_pmv_matrix", pmv_matrix);
  shader_program_.setUniformValue("u_inv_viewport",
                                  QVector2D(1.0f / width, 1.0f / height));
  shader_program_.setUniformValue("u_line_width", line_width);

  QOpenGLFunctions* gl_funcs = QOpenGLContext::currentContext()->functions();
  gl_funcs->glDrawArrays(GL_LINES, 0, (GLsizei)(2 * num_geoms_));

  // Make sure the VAO is not changed from the outside
  vao_.release();

#if DEBUG
  glDebugLog();
#endif
}

}  // namespace colmap
