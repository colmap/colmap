// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/estimators/affine_transform.h"

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/SVD>

namespace colmap {

void AffineTransformEstimator::Estimate(const std::vector<X_t>& points1,
                                        const std::vector<Y_t>& points2,
                                        std::vector<M_t>* models) {
  THROW_CHECK_EQ(points1.size(), points2.size());
  THROW_CHECK_GE(points1.size(), 3);
  THROW_CHECK(models != nullptr);

  models->clear();

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

  models->resize(1);
  (*models)[0] = A_t.transpose();
}

void AffineTransformEstimator::Residuals(const std::vector<X_t>& points1,
                                         const std::vector<Y_t>& points2,
                                         const M_t& A,
                                         std::vector<double>* residuals) {
  THROW_CHECK_EQ(points1.size(), points2.size());

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
