// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// Cost function for aligning one 3D point with a reference 3D point with
// covariance. The Residual is computed in frame b. Coordinate transformation
// convention is equivalent to Sim3d.
struct Point3DAlignmentCostFunctor
    : public AutoDiffCostFunctor<Point3DAlignmentCostFunctor, 3, 3, 8> {
 public:
  explicit Point3DAlignmentCostFunctor(const Eigen::Vector3d& point_in_b_prior,
                                       bool use_log_scale = true)
      : point_in_b_prior_(point_in_b_prior), use_log_scale_(use_log_scale) {}

  template <typename T>
  bool operator()(const T* const point_in_a,
                  const T* const b_from_a,
                  T* residuals_ptr) const {
    // Select whether to exponentiate
    const T b_from_a_scale =
        use_log_scale_ ? ceres::exp(b_from_a[7]) : b_from_a[7];

    const Eigen::Matrix<T, 3, 1> point_in_b =
        EigenQuaternionMap<T>(b_from_a) * EigenVector3Map<T>(point_in_a) *
            b_from_a_scale +
        EigenVector3Map<T>(b_from_a + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = point_in_b - point_in_b_prior_.cast<T>();
    return true;
  }

 private:
  const Eigen::Vector3d point_in_b_prior_;
  const bool use_log_scale_;
};

}  // namespace colmap
