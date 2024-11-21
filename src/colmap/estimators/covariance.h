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

#include "colmap/estimators/bundle_adjustment.h"
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

struct BACovarianceOptions {
  enum class Params {
    kOnlyPoses,
    kOnlyPoints,
    kPosesAndPoints,
    kAll,  // + Others
  };

  // For which variables to compute the covariance.
  Params params = Params::kAll;

  // Damping factor for the Hessian in the Schur complement solver.
  // Enables to robustly deal with poorly conditioned points.
  double damping = 1e-8;
};

BACovariance EstimateBACovariance(const BACovarianceOptions& options,
                                  const Reconstruction& reconstruction,
                                  BundleAdjuster& bundle_adjuster);

namespace detail {

struct PoseParam {
  image_t image_id = kInvalidImageId;
  const double* qvec = nullptr;
  const double* tvec = nullptr;
};

std::vector<PoseParam> GetPoseParams(const Reconstruction& reconstruction,
                                     const ceres::Problem& problem);

struct PointParam {
  point3D_t point3D_id = kInvalidPoint3DId;
  const double* xyz = nullptr;
};

std::vector<PointParam> GetPointParams(const Reconstruction& reconstruction,
                                       const ceres::Problem& problem);

std::vector<const double*> GetOtherParams(
    const ceres::Problem& problem,
    const std::vector<PoseParam>& poses,
    const std::vector<PointParam>& points);

}  // namespace detail
}  // namespace colmap
