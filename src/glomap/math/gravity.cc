#include "gravity.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"

#include <Eigen/QR>

namespace glomap {

Eigen::Matrix3d GravityAlignedRotation(const Eigen::Vector3d& gravity) {
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

Eigen::Vector3d AverageDirections(
    const std::vector<Eigen::Vector3d>& directions) {
  if (directions.empty()) {
    LOG(ERROR) << "Cannot average empty set of directions";
    return Eigen::Vector3d::Zero();
  }

  // Build outer product sum matrix for principal component analysis.
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  for (const auto& d : directions) {
    THROW_CHECK_LT(std::abs(d.norm() - 1.0), 1e-6)
        << "Direction vectors must be normalized";
    A += d * d.transpose();
  }
  A /= directions.size();

  // The first singular vector corresponds to the principal direction.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU);
  Eigen::Vector3d average = svd.matrixU().col(0);

  // Ensure consistent sign by aligning with majority of input vectors.
  int negative_count = 0;
  for (const auto& d : directions) {
    if (d.dot(average) < 0) {
      negative_count++;
    }
  }
  if (negative_count > static_cast<int>(directions.size()) / 2) {
    average = -average;
  }

  return average;
}

}  // namespace glomap
