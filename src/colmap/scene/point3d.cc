// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/point3d.h"

namespace colmap {

std::ostream& operator<<(std::ostream& stream, const Point3D& point3D) {
  stream << "Point3D(xyz=[" << point3D.xyz(0) << ", " << point3D.xyz(1) << ", "
         << point3D.xyz(2) << "], track_len=" << point3D.track.Length() << ")";
  return stream;
}

}  // namespace colmap
