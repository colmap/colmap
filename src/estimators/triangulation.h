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

#ifndef COLMAP_SRC_ESTIMATORS_TRIANGULATION_H_
#define COLMAP_SRC_ESTIMATORS_TRIANGULATION_H_

#include "base/camera.h"

#include <vector>

#include <Eigen/Core>

#include "optim/ransac.h"
#include "util/alignment.h"
#include "util/math.h"
#include "util/types.h"

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
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointData() {}
    PointData(const Eigen::Vector2d& point_, const Eigen::Vector2d& point_N_)
        : point(point_), point_normalized(point_N_) {}
    // Image observation in pixels. Only needs to be set for REPROJECTION_ERROR.
    Eigen::Vector2d point;
    // Normalized image observation. Must always be set.
    Eigen::Vector2d point_normalized;
  };

  struct PoseData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PoseData() : camera(nullptr) {}
    PoseData(const Eigen::Matrix3x4d& proj_matrix_,
             const Eigen::Vector3d& pose_, const Camera* camera_)
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
  void SetMinTriAngle(const double min_tri_angle);
  void SetResidualType(const ResidualType residual_type);

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
                 const std::vector<Y_t>& pose_data, const M_t& xyz,
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
    std::vector<char>* inlier_mask, Eigen::Vector3d* xyz);

}  // namespace colmap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(
    colmap::TriangulationEstimator::PointData)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(
    colmap::TriangulationEstimator::PoseData)

#endif  // COLMAP_SRC_ESTIMATORS_TRIANGULATION_H_
