// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "estimators/affine_transform.h"

#include <Eigen/SVD>

#include "util/logging.h"

namespace colmap {

std::vector<AffineTransformEstimator::M_t> AffineTransformEstimator::Estimate(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2) {
  CHECK_EQ(points1.size(), points2.size());
  CHECK_GE(points1.size(), 3);

  // Sets up the linear system that we solve to obtain a least squared solution
  // for the affine transformation.
  Eigen::MatrixXd C(2 * points1.size(), 6);
  C.setZero();
  Eigen::VectorXd b(2 * points1.size(), 1);

  for (size_t i = 0; i < points1.size(); ++i) {
    const Eigen::Vector2d& x1 = points1[i];
    const Eigen::Vector2d& x2 = points2[i];

    C(2 * i, 0) = x1(0);
    C(2 * i, 1) = x1(1);
    C(2 * i, 2) = 1.0f;
    b(2 * i) = x2(0);

    C(2 * i + 1, 3) = x1(0);
    C(2 * i + 1, 4) = x1(1);
    C(2 * i + 1, 5) = 1.0f;
    b(2 * i + 1) = x2(1);
  }

  const Eigen::VectorXd nullspace =
      C.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

  Eigen::Map<const Eigen::Matrix<double, 3, 2>> A_t(nullspace.data());

  const std::vector<M_t> models = {A_t.transpose()};
  return models;
}

void AffineTransformEstimator::Residuals(const std::vector<X_t>& points1,
                                         const std::vector<Y_t>& points2,
                                         const M_t& A,
                                         std::vector<double>* residuals) {
  CHECK_EQ(points1.size(), points2.size());

  residuals->resize(points1.size());

  // Note that this code might not be as nice as Eigen expressions,
  // but it is significantly faster in various tests.

  const double A_00 = A(0, 0);
  const double A_01 = A(0, 1);
  const double A_02 = A(0, 2);
  const double A_10 = A(1, 0);
  const double A_11 = A(1, 1);
  const double A_12 = A(1, 2);

  for (size_t i = 0; i < points1.size(); ++i) {
    const double s_0 = points1[i](0);
    const double s_1 = points1[i](1);
    const double d_0 = points2[i](0);
    const double d_1 = points2[i](1);

    const double pd_0 = A_00 * s_0 + A_01 * s_1 + A_02;
    const double pd_1 = A_10 * s_0 + A_11 * s_1 + A_12;

    const double dd_0 = d_0 - pd_0;
    const double dd_1 = d_1 - pd_1;

    (*residuals)[i] = dd_0 * dd_0 + dd_1 * dd_1;
  }
}

}  // namespace colmap
