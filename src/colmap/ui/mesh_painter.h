// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/ui/painter_base.h"

#include <QtCore>
#include <QtOpenGL>
#include <cstdint>

namespace colmap {

class MeshPainter : public PainterBase {
 public:
  MeshPainter() = default;
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
  GLuint texture_id_ = 0;
  bool has_texture_ = false;
};

}  // namespace colmap
