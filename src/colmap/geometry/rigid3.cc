// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/rigid3.h"

namespace colmap {

Eigen::Matrix3d CrossProductMatrix(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1),
      vector(0), 0;
  return matrix;
}

std::ostream& operator<<(std::ostream& stream, const Rigid3d& tform) {
  const static Eigen::IOFormat kVecFmt(
      Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ");
  stream << "Rigid3d(rotation_xyzw=["
         << tform.rotation().coeffs().format(kVecFmt) << "], translation=["
         << tform.translation().format(kVecFmt) << "])";
  return stream;
}

}  // namespace colmap
