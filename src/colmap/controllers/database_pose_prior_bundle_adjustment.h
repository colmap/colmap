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

#pragma once

#include "colmap/estimators/bundle_adjustment.h"

#include <filesystem>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Absolute camera pose prior read from the pose_priors table. The quaternion is
// the COLMAP world-to-camera rotation. Rotation covariance is a 3x3 covariance
// of the SO(3) tangent-space error in radians squared.
struct DatabasePosePrior {
  image_t image_id = kInvalidImageId;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Identity();
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d rotation_covariance = Eigen::Matrix3d::Identity();
};

struct DatabasePosePriorBundleAdjustmentOptions {
  double prior_position_fallback_stddev = 1.0;
  double prior_rotation_fallback_stddev_rad = 0.08726646259971647;  // 5 deg.
  double prior_position_loss_scale = 2.7955321496988725;
  double prior_rotation_loss_scale = 2.7955321496988725;
};

// Reads position, quaternion rotation, and their covariances from the existing
// pose_priors table. The reader accepts rotation quaternion columns named
// rotation, rotation_quaternion, rotation_prior, prior_qvec, or qvec, and
// covariance columns named rotation_covariance,
// rotation_prior_covariance, prior_qvec_covariance, or qvec_covariance.
std::vector<DatabasePosePrior> ReadDatabasePosePriors(
    const std::filesystem::path& database_path,
    const DatabasePosePriorBundleAdjustmentOptions& options);

// Creates a Ceres bundle adjuster with reprojection, absolute camera-center,
// and absolute quaternion rotation residuals. Returns nullptr when the database
// does not contain enough valid priors to establish a metric frame.
std::unique_ptr<BundleAdjuster> CreateDatabasePosePriorBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const DatabasePosePriorBundleAdjustmentOptions& prior_options,
    BundleAdjustmentConfig config,
    const std::filesystem::path& database_path,
    Reconstruction& reconstruction);

}  // namespace colmap
