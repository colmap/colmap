// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/ui/point_painter.h"

#include <QtCore>
#include <QtOpenGL>

namespace colmap {

class TrianglePainter : public PainterBase {
 public:
  TrianglePainter() = default;
  ~TrianglePainter() = default;

  struct Data {
    Data() {}
    Data(const PointPainter::Data& p1,
         const PointPainter::Data& p2,
         const PointPainter::Data& p3)
        : point1(p1), point2(p2), point3(p3) {}

    PointPainter::Data point1;
    PointPainter::Data point2;
    PointPainter::Data point3;
  };

  void Setup();
  void Upload(const std::vector<TrianglePainter::Data>& data);
  void Render(const QMatrix4x4& pmv_matrix);
};

}  // namespace colmap
