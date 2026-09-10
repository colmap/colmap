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

#pragma once

#include "colmap/util/opengl_utils.h"

#include <QtCore>
#include <QtOpenGL>
#include <cstddef>
#include <utility>
#include <vector>

namespace colmap {

// Shared OpenGL setup/upload/render logic for the point, line, and triangle
// painters. All of them store vertices as a 3-float position followed by a
// 4-byte RGBA color, with multi-vertex primitives (lines, triangles) stored as
// contiguous arrays of such vertices. Subclasses only provide the shader
// files, the position attribute name, and their Render uniforms.
class PainterBase {
 public:
  PainterBase();
  ~PainterBase();

  PainterBase(const PainterBase&) = delete;
  PainterBase& operator=(const PainterBase&) = delete;

 protected:
  void DestroyGL();
  void SetupShaders(
      std::initializer_list<std::pair<QOpenGLShader::ShaderType, const char*>>
          shaders);

  // Upload per-primitive data; vertex_stride is the size of a single vertex
  // (e.g. sizeof(PointPainter::Data)) and may be smaller than
  // sizeof(GeomData) for multi-vertex primitives.
  template <typename GeomData>
  void UploadGeoms(const std::vector<GeomData>& data,
                   const char* position_attr,
                   size_t vertex_stride) {
    num_geoms_ = data.size();
    if (num_geoms_ == 0) {
      return;
    }

    vao_.bind();
    vbo_.bind();

    // Upload data array to GPU
    vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    vbo_.allocate(data.data(),
                  static_cast<int>(data.size() * sizeof(GeomData)));

    // in_position
    shader_program_.enableAttributeArray(position_attr);
    shader_program_.setAttributeBuffer(
        position_attr, GL_FLOAT, 0, 3, vertex_stride);

    // in_color: use glVertexAttribPointer directly because Qt's
    // setAttributeBuffer does not support the normalized parameter,
    // which is needed to map uint8 [0,255] to float [0.0,1.0] in the shader.
    shader_program_.enableAttributeArray("a_color");
    QOpenGLFunctions* gl_funcs = QOpenGLContext::currentContext()->functions();
    gl_funcs->glVertexAttribPointer(
        shader_program_.attributeLocation("a_color"),
        4,
        GL_UNSIGNED_BYTE,
        GL_TRUE,
        vertex_stride,
        reinterpret_cast<const void*>(  // NOLINT(performance-no-int-to-ptr)
            3 * sizeof(GLfloat)));

    // Make sure they are not changed from the outside
    vbo_.release();
    vao_.release();

    glDebugLog();
  }

  // Bind shader program and VAO for rendering; returns false when empty.
  bool BeginRender();
  void EndRender();
  QOpenGLFunctions* GLFunctions();

  QOpenGLShaderProgram shader_program_;
  QOpenGLVertexArrayObject vao_;
  QOpenGLBuffer vbo_;

  size_t num_geoms_;
};

}  // namespace colmap
