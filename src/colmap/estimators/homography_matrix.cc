// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/homography_matrix.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace colmap {

void HomographyMatrixEstimator::Estimate(const std::vector<X_t>& points1,
                                         const std::vector<Y_t>& points2,
                                         std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(), 4);
  THROW_CHECK(models != nullptr);

  models->clear();

  const size_t num_points = points1.size();

  // Setup constraint matrix.
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(2 * num_points, 9);
  for (size_t i = 0; i < num_points; ++i) {
    A.block<1, 3>(2 * i, 0) = points1[i].transpose().homogeneous();
    A.block<1, 3>(2 * i, 3).setZero();
    A.block<1, 3>(2 * i, 6) =
        -points2[i].x() * points1[i].transpose().homogeneous();
    A.block<1, 3>(2 * i + 1, 0).setZero();
    A.block<1, 3>(2 * i + 1, 3) = points1[i].transpose().homogeneous();
    A.block<1, 3>(2 * i + 1, 6) =
        -points2[i].y() * points1[i].transpose().homogeneous();
  }

  Eigen::Matrix3d H;
  if (num_points == 4) {
    const Eigen::Matrix<double, 9, 1> h = A.block<8, 8>(0, 0)
                                              .partialPivLu()
                                              .solve(-A.block<8, 1>(0, 8))
                                              .homogeneous();
    if (h.hasNaN()) {
      return;
    }
    H = Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose();
  } else {
    // Solve for the nullspace of the constraint matrix.
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    if (svd.rank() < 8) {
      return;
    }
    const Eigen::VectorXd nullspace = svd.matrixV().col(8);
    H = Eigen::Map<const Eigen::Matrix3d>(nullspace.data()).transpose();
  }

  if (std::abs(H.determinant()) < 1e-8) {
    return;
  }

  models->resize(1);
  (*models)[0] = H;
}

void HomographyMatrixEstimator::Residuals(const std::vector<X_t>& points1,
                                          const std::vector<Y_t>& points2,
                                          const M_t& H,
                                          std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());

  residuals->resize(points1.size());

  // Note that this code might not be as nice as Eigen expressions,
  // but it is significantly faster in various tests.

  const double H_00 = H(0, 0);
  const double H_01 = H(0, 1);
  const double H_02 = H(0, 2);
  const double H_10 = H(1, 0);
  const double H_11 = H(1, 1);
  const double H_12 = H(1, 2);
  const double H_20 = H(2, 0);
  const double H_21 = H(2, 1);
  const double H_22 = H(2, 2);

  for (size_t i = 0; i < points1.size(); ++i) {
    const double s_0 = points1[i](0);
    const double s_1 = points1[i](1);
    const double d_0 = points2[i](0);
    const double d_1 = points2[i](1);

    const double pd_0 = H_00 * s_0 + H_01 * s_1 + H_02;
    const double pd_1 = H_10 * s_0 + H_11 * s_1 + H_12;
    const double pd_2 = H_20 * s_0 + H_21 * s_1 + H_22;

    const double inv_pd_2 = 1.0 / pd_2;
    const double dd_0 = d_0 - pd_0 * inv_pd_2;
    const double dd_1 = d_1 - pd_1 * inv_pd_2;

    (*residuals)[i] = dd_0 * dd_0 + dd_1 * dd_1;
  }
}

}  // namespace colmap
