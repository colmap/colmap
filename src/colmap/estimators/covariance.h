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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/reconstruction.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// Update covariance (n = 6) for rigid3d.inverse()
Eigen::Matrix6d GetCovarianceForPoseInverse(const Eigen::Matrix6d& covar,
                                            const Rigid3d& rigid3);

// Covariance estimation for bundle adjustment (or extended) problem.
// The interface is applicable to all ceres problem extended on top of bundle
// adjustment. The Schur complement is computed explicitly to eliminate the
// hessian block for all the 3D points, which is essential to avoid Jacobian
// rank deficiency for large-scale reconstruction
class BundleAdjustmentCovarianceEstimator {
 public:
  static std::map<image_t, Eigen::MatrixXd> EstimatePoseCovarianceCeresBackend(
      ceres::Problem* problem, Reconstruction* reconstruction);

  // The parameter lambda is the dumping factor used to avoid Jacobian rank
  // deficiency for poorly conditioned 3D points
  static std::map<image_t, Eigen::MatrixXd> EstimatePoseCovariance(
      ceres::Problem* problem,
      Reconstruction* reconstruction,
      double lambda = 1e-6);
};

}  // namespace colmap
