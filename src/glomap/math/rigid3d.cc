#include "glomap/math/rigid3d.h"

namespace glomap {

Eigen::Vector3d RotationToAngleAxis(const Eigen::Matrix3d& rot) {
  const Eigen::AngleAxisd aa(rot);
  return aa.angle() * aa.axis();
}

Eigen::Matrix3d AngleAxisToRotation(const Eigen::Vector3d& aa_vec) {
  double aa_norm = aa_vec.norm();
  if (aa_norm > 1e-12) {
    return Eigen::AngleAxis<double>(aa_norm, aa_vec / aa_norm)
        .toRotationMatrix();
  } else {
    // Small angle approximation.
    Eigen::Matrix3d R;
    R(0, 0) = 1;
    R(1, 0) = aa_vec[2];
    R(2, 0) = -aa_vec[1];
    R(0, 1) = -aa_vec[2];
    R(1, 1) = 1;
    R(2, 1) = aa_vec[0];
    R(0, 2) = aa_vec[1];
    R(1, 2) = -aa_vec[0];
    R(2, 2) = 1;
    return R;
  }
}

}  // namespace glomap
