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

#include "colmap/ui/mesh_painter.h"

#include "colmap/util/opengl_utils.h"

#include <algorithm>

namespace colmap {

MeshPainter::MeshPainter() : num_vertices_(0) {}

MeshPainter::~MeshPainter() {
  vao_.destroy();
  vbo_.destroy();
  if (texture_id_ != 0) {
    QOpenGLFunctions* gl_funcs = QOpenGLContext::currentContext()->functions();
    gl_funcs->glDeleteTextures(1, &texture_id_);
  }
}

void MeshPainter::Setup() {
  vao_.destroy();
  vbo_.destroy();
  if (texture_id_ != 0) {
    QOpenGLFunctions* gl_funcs = QOpenGLContext::currentContext()->functions();
    gl_funcs->glDeleteTextures(1, &texture_id_);
    texture_id_ = 0;
    has_texture_ = false;
  }
  if (shader_program_.isLinked()) {
    shader_program_.release();
    shader_program_.removeAllShaders();
  }

  shader_program_.addShaderFromSourceFile(QOpenGLShader::Vertex,
                                          ":/shaders/mesh.v.glsl");
  shader_program_.addShaderFromSourceFile(QOpenGLShader::Geometry,
                                          ":/shaders/mesh.g.glsl");
  shader_program_.addShaderFromSourceFile(QOpenGLShader::Fragment,
                                          ":/shaders/mesh.f.glsl");
  shader_program_.link();
  shader_program_.bind();

  vao_.create();
  vbo_.create();

#if DEBUG
  glDebugLog();
#endif
}

void MeshPainter::Upload(const std::vector<MeshPainter::Data>& data) {
  num_vertices_ = data.size();
  if (num_vertices_ == 0) {
    return;
  }

  vao_.bind();
  vbo_.bind();

  // Use glBufferData directly to support sizes exceeding INT_MAX.
  QOpenGLFunctions* gl_funcs = QOpenGLContext::currentContext()->functions();
  gl_funcs->glBufferData(GL_ARRAY_BUFFER,
                         data.size() * sizeof(MeshPainter::Data),
                         data.data(),
                         GL_DYNAMIC_DRAW);

  // a_position: 3 floats at offset 0
  shader_program_.enableAttributeArray("a_position");
  shader_program_.setAttributeBuffer(
      "a_position", GL_FLOAT, 0, 3, sizeof(MeshPainter::Data));

  // a_normal: 3 floats at offset 3*sizeof(float)
  shader_program_.enableAttributeArray("a_normal");
  shader_program_.setAttributeBuffer(
      "a_normal", GL_FLOAT, 3 * sizeof(GLfloat), 3, sizeof(MeshPainter::Data));

  // a_uv: 2 floats at offset 6*sizeof(float)
  shader_program_.enableAttributeArray("a_uv");
  shader_program_.setAttributeBuffer(
      "a_uv", GL_FLOAT, 6 * sizeof(GLfloat), 2, sizeof(MeshPainter::Data));

  // a_color: use glVertexAttribPointer directly because Qt's
  // setAttributeBuffer does not support the normalized parameter,
  // which is needed to map uint8 [0,255] to float [0.0,1.0] in the shader.
  shader_program_.enableAttributeArray("a_color");
  gl_funcs->glVertexAttribPointer(
      shader_program_.attributeLocation("a_color"),
      3,
      GL_UNSIGNED_BYTE,
      GL_TRUE,
      sizeof(MeshPainter::Data),
      reinterpret_cast<const void*>(  // NOLINT(performance-no-int-to-ptr)
          8 * sizeof(GLfloat)));

  vbo_.release();
  vao_.release();

#if DEBUG
  glDebugLog();
#endif
}

void MeshPainter::UploadTexture(std::vector<uint8_t> data,
                                const int width,
                                const int height,
                                const int channels) {
  QOpenGLFunctions* gl_funcs = QOpenGLContext::currentContext()->functions();

  if (texture_id_ != 0) {
    gl_funcs->glDeleteTextures(1, &texture_id_);
    texture_id_ = 0;
  }

  if (data.empty() || width <= 0 || height <= 0) {
    has_texture_ = false;
    return;
  }

  // Flip rows vertically in-place: the input image data is in top-to-bottom
  // order (standard image convention), but OpenGL's glTexImage2D interprets
  // the first row as the bottom of the texture. The PLY UV coordinates use
  // OpenGL convention (V=0 at bottom), so we must flip the image data to
  // match.
  const size_t row_bytes = static_cast<size_t>(width) * channels;
  for (int y = 0; y < height / 2; ++y) {
    std::swap_ranges(data.begin() + y * row_bytes,
                     data.begin() + y * row_bytes + row_bytes,
                     data.begin() + (height - 1 - y) * row_bytes);
  }

  gl_funcs->glGenTextures(1, &texture_id_);
  gl_funcs->glBindTexture(GL_TEXTURE_2D, texture_id_);

  gl_funcs->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  const GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
  gl_funcs->glTexImage2D(GL_TEXTURE_2D,
                         0,
                         format,
                         width,
                         height,
                         0,
                         format,
                         GL_UNSIGNED_BYTE,
                         data.data());

  gl_funcs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl_funcs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl_funcs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  gl_funcs->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  gl_funcs->glBindTexture(GL_TEXTURE_2D, 0);

  has_texture_ = true;
}

void MeshPainter::Render(const QMatrix4x4& pmv_matrix,
                         const QMatrix4x4& model_view_matrix,
                         const bool wireframe,
                         const bool color) {
  if (num_vertices_ == 0) {
    return;
  }

  shader_program_.bind();
  vao_.bind();

  shader_program_.setUniformValue("u_pmv_matrix", pmv_matrix);
  shader_program_.setUniformValue("u_model_view_matrix", model_view_matrix);
  shader_program_.setUniformValue("u_normal_matrix",
                                  model_view_matrix.normalMatrix());
  shader_program_.setUniformValue("u_wireframe", wireframe);
  shader_program_.setUniformValue("u_has_texture", has_texture_ && color);
  shader_program_.setUniformValue("u_color", color);

  QOpenGLFunctions* gl_funcs = QOpenGLContext::currentContext()->functions();

  if (has_texture_) {
    gl_funcs->glActiveTexture(GL_TEXTURE0);
    gl_funcs->glBindTexture(GL_TEXTURE_2D, texture_id_);
    shader_program_.setUniformValue("u_texture", 0);
  }

  gl_funcs->glEnable(GL_DEPTH_TEST);
  gl_funcs->glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(num_vertices_));

  if (has_texture_) {
    gl_funcs->glBindTexture(GL_TEXTURE_2D, 0);
  }

  vao_.release();

#if DEBUG
  glDebugLog();
#endif
}

}  // namespace colmap
