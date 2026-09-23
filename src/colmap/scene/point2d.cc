// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/point2d.h"

namespace colmap {

std::ostream& operator<<(std::ostream& stream, const Point2D& point2D) {
  stream << "Point2D(xy=[" << point2D.xy(0) << ", " << point2D.xy(1)
         << "], point3D_id="
         << (point2D.HasPoint3D() ? std::to_string(point2D.point3D_id) : "-1")
         << ")";
  return stream;
}

}  // namespace colmap
