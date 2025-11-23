#include "gravity.h"

#include "glomap/math/rigid3d.h"
#include "glomap/scene/types_sfm.h"

#include <Eigen/QR>

namespace glomap {

// The second col of R_align is gravity direction
Eigen::Matrix3d GetAlignRot(const Eigen::Vector3d& gravity) {
  Eigen::Matrix3d R;
  Eigen::Vector3d v = gravity.normalized();
  R.col(1) = v;

  Eigen::Matrix3d Q = v.householderQr().householderQ();
  Eigen::Matrix<double, 3, 2> N = Q.rightCols(2);
  R.col(0) = N.col(0);
  R.col(2) = N.col(1);
  if (R.determinant() < 0) {
    R.col(2) = -R.col(2);
  }
  return R;
}

double RotUpToAngle(const Eigen::Matrix3d& R_up) {
  return RotationToAngleAxis(R_up)[1];
}

Eigen::Matrix3d AngleToRotUp(double angle) {
  Eigen::Vector3d aa(0, angle, 0);
  return AngleAxisToRotation(aa);
}

// Code adapted from
// https://gist.github.com/PeteBlackerThe3rd/f73e9d569e29f23e8bd828d7886636a0
Eigen::Vector3d AverageGravity(const std::vector<Eigen::Vector3d>& gravities) {
  if (gravities.size() == 0) {
    std::cerr
        << "Error trying to calculate the average gravities of an empty set!\n";
    return Eigen::Vector3d::Zero();
  }

  // first build a 3x3 matrix which is the elementwise sum of the product of
  // each quaternion with itself
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();

  for (int g = 0; g < gravities.size(); ++g)
    A += gravities[g] * gravities[g].transpose();

  // normalise with the number of gravities
  A /= gravities.size();

  // Compute the SVD of this 3x3 matrix
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);

  Eigen::VectorXd singular_values = svd.singularValues();
  Eigen::MatrixXd U = svd.matrixU();

  // find the eigen vector corresponding to the largest eigen value
  int largest_eigen_value_index = -1;
  float largest_eigen_value;
  bool first = true;

  for (int i = 0; i < singular_values.rows(); ++i) {
    if (first) {
      largest_eigen_value = singular_values(i);
      largest_eigen_value_index = i;
      first = false;
    } else if (singular_values(i) > largest_eigen_value) {
      largest_eigen_value = singular_values(i);
      largest_eigen_value_index = i;
    }
  }

  Eigen::Vector3d average;
  average(0) = U(0, largest_eigen_value_index);
  average(1) = U(1, largest_eigen_value_index);
  average(2) = U(2, largest_eigen_value_index);

  int negative_counter = 0;
  for (int g = 0; g < gravities.size(); ++g) {
    if (gravities[g].dot(average) < 0) negative_counter++;
  }
  if (negative_counter > gravities.size() / 2) {
    average = -average;
  }

  return average;
}

double CalcAngle(const Eigen::Vector3d& gravity1,
                 const Eigen::Vector3d& gravity2) {
  double cos_r = gravity1.dot(gravity2) / (gravity1.norm() * gravity2.norm());
  cos_r = std::min(std::max(cos_r, -1.), 1.);

  return std::acos(cos_r) * 180 / EIGEN_PI;
}
}  // namespace glomap
