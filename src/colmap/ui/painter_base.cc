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

#include "colmap/ui/painter_base.h"

namespace colmap {

PainterBase::PainterBase() : num_geoms_(0) {}

PainterBase::~PainterBase() { DestroyGL(); }

void PainterBase::DestroyGL() {
  vao_.destroy();
  vbo_.destroy();
}

void PainterBase::SetupShaders(
    std::initializer_list<std::pair<QOpenGLShader::ShaderType, const char*>>
        shaders) {
  DestroyGL();
  if (shader_program_.isLinked()) {
    shader_program_.release();
    shader_program_.removeAllShaders();
  }

  for (const auto& [type, path] : shaders) {
    shader_program_.addShaderFromSourceFile(type, path);
  }
  shader_program_.link();
  shader_program_.bind();

  vao_.create();
  vbo_.create();

  glDebugLog();
}

bool PainterBase::BeginRender() {
  if (num_geoms_ == 0) {
    return false;
  }

  shader_program_.bind();
  vao_.bind();
  return true;
}

void PainterBase::EndRender() {
  // Make sure the VAO is not changed from the outside
  vao_.release();

  glDebugLog();
}

QOpenGLFunctions* PainterBase::GLFunctions() {
  return QOpenGLContext::currentContext()->functions();
}

}  // namespace colmap
