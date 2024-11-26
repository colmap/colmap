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

#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/problem.h>

namespace colmap {
namespace internal {
struct PoseParam;
}

struct BACovariance {
  explicit BACovariance(
      std::unordered_map<point3D_t, Eigen::MatrixXd> point_covs,
      std::unordered_map<image_t, std::pair<int, int>> pose_L_start_size,
      std::unordered_map<const double*, std::pair<int, int>> other_L_start_size,
      Eigen::MatrixXd L_inv);

  // Covariance for 3D points, conditioned on all other variables set constant.
  // If some dimensions are kept constant, the respective rows/columns are
  // omitted. Returns null if 3D point not a variable in the problem.
  std::optional<Eigen::MatrixXd> GetPointCov(point3D_t point3D_id) const;

  // Tangent space covariance in the order [rotation, translation]. If some
  // dimensions are kept constant, the respective rows/columns are omitted.
  // Returns null if image not a variable in the problem.
  std::optional<Eigen::MatrixXd> GetCamFromWorldCov(image_t image_id) const;
  std::optional<Eigen::MatrixXd> GetCam1FromCam2Cov(image_t image_id1,
                                                    image_t image_id2) const;

  // Tangent space covariance for any other variable parameter block in the
  // problem. If some dimensions are kept constant, the respective rows/columns
  // are omitted. Returns null if parameter block not a variable in the problem.
  std::optional<Eigen::MatrixXd> GetOtherParamsCov(const double* params) const;

 private:
  const std::unordered_map<point3D_t, Eigen::MatrixXd> point_covs_;
  const std::unordered_map<image_t, std::pair<int, int>> pose_L_start_size_;
  const std::unordered_map<const double*, std::pair<int, int>>
      other_L_start_size_;
  const Eigen::MatrixXd L_inv_;
};

struct BACovarianceOptions {
  enum class Params {
    POSES,
    POINTS,
    POSES_AND_POINTS,
    ALL,  // + Others
  };

  // For which parameters to compute the covariance.
  Params params = Params::ALL;

  // Damping factor for the Hessian in the Schur complement solver.
  // Enables to robustly deal with poorly conditioned parameters.
  double damping = 1e-8;

  // WARNING: This option will be removed in a future release, use at your own
  // risk. For custom bundle adjustment problems, this enables to specify a
  // custom set of pose parameter blocks to consider. Note that these pose
  // blocks must not necessarily be part of the reconstruction but they must
  // follow the standard requirement for applying the Schur complement trick.
  // TODO: This is a temporary option to enable extraction of pose covariances
  // for custom rig bundle adjustment problems. To be removed when proper rig
  // support is enabled in colmap natively.
  std::vector<internal::PoseParam> experimental_custom_poses;
};

// Computes covariances for the parameters in a bundle adjustment problem. It is
// important that the problem has a structure suitable for solving using the
// Schur complement trick. This is the case for the standard configuration of
// bundle adjustment problems, but be careful if you modify the underlying
// problem with custom residuals.
// Returns null if the estimation was not successful.
std::optional<BACovariance> EstimateBACovariance(
    const BACovarianceOptions& options,
    const Reconstruction& reconstruction,
    BundleAdjuster& bundle_adjuster);

namespace internal {

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

}  // namespace internal
}  // namespace colmap
