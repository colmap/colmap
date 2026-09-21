// SPDX-License-Identifier: BSD-3-Clause

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
