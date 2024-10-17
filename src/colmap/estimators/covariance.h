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

#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/ceres.h>

namespace colmap {

struct BACovariance {
  // Indicates whether the covariances were estimated successfully.
  bool success = false;

  // Tangent space covariance in the order [rotation, translation]. If some
  // parameters are kept constant, the respective rows/columns are omitted.
  // The full pose covariance matrix has dimension 6x6.
  std::unordered_map<image_t, Eigen::MatrixXd> pose_covs;

  // Tangent space covariance for 3D points.
  std::unordered_map<point3D_t, Eigen::Matrix3d> point_covs;
};

enum class BACovarianceType {
  kOnlyPoses,
  kOnlyPoints,
  kPosesAndPoints,
};

BACovariance EstimateCeresBACovariance(
    const Reconstruction& reconstruction,
    ceres::Problem* problem,
    BACovarianceType type = BACovarianceType::kOnlyPoses);

BACovariance EstimateSchurBACovariance(
    const Reconstruction& reconstruction,
    ceres::Problem* problem,
    BACovarianceType type = BACovarianceType::kOnlyPoses,
    double damping = 1e-8);

}  // namespace colmap
