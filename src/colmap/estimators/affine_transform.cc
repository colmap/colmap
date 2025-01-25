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

#include "colmap/estimators/affine_transform.h"

#include "colmap/optim/loransac.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace colmap {

void AffineTransformEstimator::Estimate(const std::vector<X_t>& src,
                                        const std::vector<Y_t>& tgt,
                                        std::vector<M_t>* tgt_from_src) {
  const size_t num_points = src.size();
  THROW_CHECK_EQ(num_points, tgt.size());
  THROW_CHECK_GE(num_points, 3);
  THROW_CHECK(tgt_from_src != nullptr);

  tgt_from_src->clear();

  // Sets up the linear system that we solve to obtain a least squared solution
  // for the affine transformation.
  Eigen::Matrix<double, Eigen::Dynamic, 6> A(2 * num_points, 6);
  Eigen::VectorXd b(2 * num_points, 1);
  for (size_t i = 0; i < num_points; ++i) {
    A.block<1, 3>(2 * i, 0) = src[i].transpose().homogeneous();
    A.block<1, 3>(2 * i, 3).setZero();
    b(2 * i) = tgt[i].x();
    A.block<1, 3>(2 * i + 1, 0).setZero();
    A.block<1, 3>(2 * i + 1, 3) = src[i].transpose().homogeneous();
    b(2 * i + 1) = tgt[i].y();
  }

  Eigen::Vector6d sol;
  if (num_points == 3) {
    sol = A.partialPivLu().solve(b);
    if (sol.hasNaN()) {
      return;
    }
  } else {
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 6>> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.rank() < 6) {
      return;
    }
    sol = svd.solve(b);
  }

  tgt_from_src->resize(1);
  (*tgt_from_src)[0] =
      Eigen::Map<const Eigen::Matrix<double, 3, 2>>(sol.data()).transpose();
}

void AffineTransformEstimator::Residuals(const std::vector<X_t>& src,
                                         const std::vector<Y_t>& tgt,
                                         const M_t& tgt_from_src,
                                         std::vector<double>* residuals) {
  const size_t num_points = src.size();
  THROW_CHECK_EQ(num_points, tgt.size());
  residuals->resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    (*residuals)[i] =
        (tgt[i] - tgt_from_src * src[i].homogeneous()).squaredNorm();
  }
}

bool EstimateAffine2d(const std::vector<Eigen::Vector2d>& src,
                      const std::vector<Eigen::Vector2d>& tgt,
                      Eigen::Matrix2x3d& tgt_from_src) {
  std::vector<Eigen::Matrix2x3d> models;
  AffineTransformEstimator::Estimate(src, tgt, &models);
  if (models.empty()) {
    return false;
  }
  THROW_CHECK_EQ(models.size(), 1);
  tgt_from_src = models[0];
  return true;
}

typename RANSAC<AffineTransformEstimator>::Report EstimateAffine2dRobust(
    const std::vector<Eigen::Vector2d>& src,
    const std::vector<Eigen::Vector2d>& tgt,
    const RANSACOptions& options,
    Eigen::Matrix2x3d& tgt_from_src) {
  LORANSAC<AffineTransformEstimator, AffineTransformEstimator> ransac(options);
  auto report = ransac.Estimate(src, tgt);
  if (report.success) {
    tgt_from_src = report.model;
  }
  return report;
}

}  // namespace colmap
