// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/ui/point_painter.h"

#include <QtCore>
#include <QtOpenGL>

namespace colmap {

class LinePainter : public PainterBase {
 public:
  LinePainter() = default;
  ~LinePainter() = default;

  struct Data {
    Data() {}
    Data(const PointPainter::Data& p1, const PointPainter::Data& p2)
        : point1(p1), point2(p2) {}

    PointPainter::Data point1;
    PointPainter::Data point2;
  };

  void Setup();
  void Upload(const std::vector<LinePainter::Data>& data);
  void Render(const QMatrix4x4& pmv_matrix,
              int width,
              int height,
              float line_width);
};

}  // namespace colmap
