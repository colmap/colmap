// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/ui/painter_base.h"

#include <QtCore>
#include <QtOpenGL>
#include <cstdint>

namespace colmap {

class PointPainter : public PainterBase {
 public:
  PointPainter() = default;
  ~PointPainter() = default;

  struct Data {
    Data() : x(0), y(0), z(0), r(0), g(0), b(0), a(0) {}
    Data(float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
        : x(x), y(y), z(z), r(r), g(g), b(b), a(a) {}

    float x, y, z;
    uint8_t r, g, b, a;
  };

  void Setup();
  void Upload(const std::vector<PointPainter::Data>& data);
  void Render(const QMatrix4x4& pmv_matrix, float point_size);
};

}  // namespace colmap
