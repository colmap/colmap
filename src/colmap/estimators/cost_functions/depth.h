// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace colmap {

enum class DepthResidualType { kLinear, kLog };

// Axial-depth error using a world-to-camera pose and a log depth scale.
struct CamFromWorldDepthCostFunctor
    : public AutoDiffCostFunctor<CamFromWorldDepthCostFunctor, 1, 3, 7, 1> {
  CamFromWorldDepthCostFunctor(double depth_prior,
                               double stddev,
                               DepthResidualType residual_type)
      : depth_prior(depth_prior), stddev(stddev), residual_type(residual_type) {}

  const double depth_prior, stddev;
  const DepthResidualType residual_type;

  template <typename T>
  bool operator()(const T* point3D_in_world,
                  const T* cam_from_world,
                  const T* log_depth_scale,
                  T* result) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_world) *
            EigenVector3Map<T>(point3D_in_world) +
        EigenVector3Map<T>(cam_from_world + 4);
    const T depth = point3D_in_cam.z();
    if (depth <= T(0)) {
      result[0] = T(0);
      return true;
    }
    if (residual_type == DepthResidualType::kLog) {
      result[0] = (ceres::log(depth / T(depth_prior)) - *log_depth_scale) /
                  T(stddev / depth_prior);
    } else {
      result[0] =
          (depth * ceres::exp(-*log_depth_scale) - T(depth_prior)) / T(stddev);
    }
    return true;
  }
};

struct RigCamFromWorldDepthCostFunctor
    : public AutoDiffCostFunctor<RigCamFromWorldDepthCostFunctor,
                                 1,
                                 3,
                                 7,
                                 7,
                                 1> {
  explicit RigCamFromWorldDepthCostFunctor(
      const CamFromWorldDepthCostFunctor& depth_cost)
      : depth_cost(depth_cost) {}

  const CamFromWorldDepthCostFunctor depth_cost;

  template <typename T>
  bool operator()(const T* point3D_in_world,
                  const T* cam_from_rig,
                  const T* rig_from_world,
                  const T* log_depth_scale,
                  T* result) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_rig =
        EigenQuaternionMap<T>(rig_from_world) *
            EigenVector3Map<T>(point3D_in_world) +
        EigenVector3Map<T>(rig_from_world + 4);
    return depth_cost(
        point3D_in_rig.data(), cam_from_rig, log_depth_scale, result);
  }
};

}  // namespace colmap
