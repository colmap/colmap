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

#include <QtCore>
#include <QtOpenGL>
#include <cstdint>

namespace colmap {

class MeshPainter {
 public:
  MeshPainter();
  ~MeshPainter();

  struct Data {
    Data()
        : px(0),
          py(0),
          pz(0),
          nx(0),
          ny(0),
          nz(0),
          u(0),
          v(0),
          r(0),
          g(0),
          b(0) {}
    Data(float px,
         float py,
         float pz,
         float nx,
         float ny,
         float nz,
         uint8_t r,
         uint8_t g,
         uint8_t b)
        : px(px),
          py(py),
          pz(pz),
          nx(nx),
          ny(ny),
          nz(nz),
          u(0),
          v(0),
          r(r),
          g(g),
          b(b) {}
    Data(float px,
         float py,
         float pz,
         float nx,
         float ny,
         float nz,
         float u,
         float v,
         uint8_t r,
         uint8_t g,
         uint8_t b)
        : px(px),
          py(py),
          pz(pz),
          nx(nx),
          ny(ny),
          nz(nz),
          u(u),
          v(v),
          r(r),
          g(g),
          b(b) {}

    float px, py, pz;
    float nx, ny, nz;
    float u, v;
    uint8_t r, g, b;
  };

  void Setup();
  void Upload(const std::vector<MeshPainter::Data>& data);
  void UploadTexture(std::vector<uint8_t> data,
                     int width,
                     int height,
                     int channels);
  void Render(const QMatrix4x4& pmv_matrix,
              const QMatrix4x4& model_view_matrix,
              bool wireframe,
              bool color);

 private:
  QOpenGLShaderProgram shader_program_;
  QOpenGLVertexArrayObject vao_;
  QOpenGLBuffer vbo_;

  size_t num_vertices_;
  GLuint texture_id_ = 0;
  bool has_texture_ = false;
};

}  // namespace colmap
