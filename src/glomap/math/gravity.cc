#include "gravity.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"

#include <Eigen/QR>

namespace glomap {

Eigen::Matrix3d RotationFromGravity(const Eigen::Vector3d& gravity) {
  THROW_CHECK_LT(std::abs(gravity.norm() - 1.0), 1e-6)
      << "Gravity vector must be normalized";

  Eigen::Matrix3d R;
  R.col(1) = gravity;

  Eigen::Matrix3d Q = gravity.householderQr().householderQ();
  Eigen::Matrix<double, 3, 2> N = Q.rightCols(2);
  R.col(0) = N.col(0);
  R.col(2) = N.col(1);
  if (R.determinant() < 0) {
    R.col(2) = -R.col(2);
  }
  return R;
}

double YAxisAngleFromRotation(const Eigen::Matrix3d& rotation) {
  return colmap::RotationMatrixToAngleAxis(rotation)[1];
}

Eigen::Matrix3d RotationFromYAxisAngle(double angle) {
  return colmap::AngleAxisToRotationMatrix(Eigen::Vector3d(0, angle, 0));
}

Eigen::Vector3d AverageGravityDirection(
    const std::vector<Eigen::Vector3d>& gravities) {
  if (gravities.empty()) {
    LOG(ERROR) << "Cannot average empty set of gravity directions";
    return Eigen::Vector3d::Zero();
  }

  // Build outer product sum matrix for principal component analysis.
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  for (const auto& g : gravities) {
    THROW_CHECK_LT(std::abs(g.norm() - 1.0), 1e-6)
        << "Gravity vectors must be normalized";
    A += g * g.transpose();
  }
  A /= gravities.size();

  // The first singular vector corresponds to the principal direction.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU);
  Eigen::Vector3d average = svd.matrixU().col(0);

  // Ensure consistent sign by aligning with majority of input vectors.
  int negative_count = 0;
  for (const auto& g : gravities) {
    if (g.dot(average) < 0) {
      negative_count++;
    }
  }
  if (negative_count > static_cast<int>(gravities.size()) / 2) {
    average = -average;
  }

  return average;
}

}  // namespace glomap
