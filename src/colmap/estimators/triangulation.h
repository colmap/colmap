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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/base/camera.h"
#include "colmap/math/math.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Triangulation estimator to estimate 3D point from multiple observations.
// The triangulation must satisfy the following constraints:
//    - Sufficient triangulation angle between observation pairs.
//    - All observations must satisfy cheirality constraint.
//
// An observation is composed of an image measurement and the corresponding
// camera pose and calibration.
class TriangulationEstimator {
 public:
  enum class ResidualType {
    ANGULAR_ERROR,
    REPROJECTION_ERROR,
  };

  struct PointData {
    PointData() {}
    PointData(const Eigen::Vector2d& point_, const Eigen::Vector2d& point_N_)
        : point(point_), point_normalized(point_N_) {}
    // Image observation in pixels. Only needs to be set for REPROJECTION_ERROR.
    Eigen::Vector2d point;
    // Normalized image observation. Must always be set.
    Eigen::Vector2d point_normalized;
  };

  struct PoseData {
    PoseData() : camera(nullptr) {}
    PoseData(const Eigen::Matrix3x4d& proj_matrix_,
             const Eigen::Vector3d& pose_,
             const Camera* camera_)
        : proj_matrix(proj_matrix_), proj_center(pose_), camera(camera_) {}
    // The projection matrix for the image of the observation.
    Eigen::Matrix3x4d proj_matrix;
    // The projection center for the image of the observation.
    Eigen::Vector3d proj_center;
    // The camera for the image of the observation.
    const Camera* camera;
  };

  typedef PointData X_t;
  typedef PoseData Y_t;
  typedef Eigen::Vector3d M_t;

  // Specify settings for triangulation estimator.
  void SetMinTriAngle(double min_tri_angle);
  void SetResidualType(ResidualType residual_type);

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 2;

  // Estimate a 3D point from a two-view observation.
  //
  // @param point_data        Image measurement.
  // @param point_data        Camera poses.
  //
  // @return                  Triangulated point if successful, otherwise none.
  std::vector<M_t> Estimate(const std::vector<X_t>& point_data,
                            const std::vector<Y_t>& pose_data) const;

  // Calculate residuals in terms of squared reprojection or angular error.
  //
  // @param point_data        Image measurements.
  // @param point_data        Camera poses.
  // @param xyz               3D point.
  //
  // @return                  Residual for each observation.
  void Residuals(const std::vector<X_t>& point_data,
                 const std::vector<Y_t>& pose_data,
                 const M_t& xyz,
                 std::vector<double>* residuals) const;

 private:
  ResidualType residual_type_ = ResidualType::REPROJECTION_ERROR;
  double min_tri_angle_ = 0.0;
};

struct EstimateTriangulationOptions {
  // Minimum triangulation angle in radians.
  double min_tri_angle = 0.0;

  // The employed residual type.
  TriangulationEstimator::ResidualType residual_type =
      TriangulationEstimator::ResidualType::ANGULAR_ERROR;

  // RANSAC options for TriangulationEstimator.
  RANSACOptions ransac_options;

  void Check() const {
    CHECK_GE(min_tri_angle, 0.0);
    ransac_options.Check();
  }
};

// Robustly estimate 3D point from observations in multiple views using RANSAC
// and a subsequent non-linear refinement using all inliers. Returns true
// if the estimated number of inliers has more than two views.
bool EstimateTriangulation(
    const EstimateTriangulationOptions& options,
    const std::vector<TriangulationEstimator::PointData>& point_data,
    const std::vector<TriangulationEstimator::PoseData>& pose_data,
    std::vector<char>* inlier_mask,
    Eigen::Vector3d* xyz);

}  // namespace colmap
