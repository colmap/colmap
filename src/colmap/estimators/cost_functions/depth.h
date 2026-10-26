// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace colmap {

enum class DepthResidualType { kLinear, kLog, kLogLinear };

// Axial-depth error using a world-to-camera pose and a log depth scale.
struct CamFromWorldDepthCostFunctor
    : public AutoDiffCostFunctor<CamFromWorldDepthCostFunctor, 1, 3, 7, 1> {
  CamFromWorldDepthCostFunctor(double depth_prior,
                               double stddev,
                               DepthResidualType residual_type)
      : depth_prior(depth_prior), stddev(stddev), residual_type(residual_type) {
    THROW_CHECK(residual_type != DepthResidualType::kLogLinear)
        << "BA depth constraints support only linear and log residuals";
  }

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

// Axial-depth error using a camera center and a fixed camera rotation.
struct WorldFromCamCenterDepthCostFunctor
    : public AutoDiffCostFunctor<WorldFromCamCenterDepthCostFunctor,
                                 1,
                                 3,
                                 3,
                                 1> {
  WorldFromCamCenterDepthCostFunctor(
      double depth_prior,
      double stddev,
      DepthResidualType residual_type,
      double log_linear_threshold,
      const Eigen::RowVector3d& cam_axis_in_world)
      : depth_prior(depth_prior),
        stddev(stddev),
        residual_type(residual_type),
        log_linear_threshold(log_linear_threshold),
        cam_axis_in_world(cam_axis_in_world) {}

  const double depth_prior, stddev;
  const DepthResidualType residual_type;
  const double log_linear_threshold;
  const Eigen::RowVector3d cam_axis_in_world;

  template <typename T>
  bool operator()(const T* point3D_in_world,
                  const T* cam_in_world,
                  const T* log_depth_scale,
                  T* result) const {
    return ComputeResidual(
        ComputeDepth(point3D_in_world, cam_in_world), *log_depth_scale, result);
  }

  template <typename T>
  T ComputeDepth(const T* point3D_in_world, const T* cam_in_world) const {
    return cam_axis_in_world.cast<T>().dot(
        EigenVector3Map<T>(point3D_in_world) -
        EigenVector3Map<T>(cam_in_world));
  }

  template <typename T>
  bool ComputeResidual(const T& depth,
                       const T& log_depth_scale,
                       T* result) const {
    if (residual_type == DepthResidualType::kLog) {
      result[0] = depth > T(0)
                      ? (ceres::log(depth / T(depth_prior)) - log_depth_scale) /
                            T(stddev / depth_prior)
                      : T(0);
    } else if (residual_type == DepthResidualType::kLogLinear) {
      const T ratio = depth / T(depth_prior) * ceres::exp(-log_depth_scale);
      const T threshold = T(log_linear_threshold);
      result[0] =
          (ratio >= threshold
               ? ceres::log(depth / T(depth_prior)) - log_depth_scale
               : ceres::log(threshold) + (ratio - threshold) / threshold) /
          T(stddev / depth_prior);
    } else {
      result[0] =
          (depth * ceres::exp(-log_depth_scale) - T(depth_prior)) / T(stddev);
    }
    return true;
  }
};

struct RigWorldFromCamCenterDepthConstantRigCostFunctor
    : public AutoDiffCostFunctor<
          RigWorldFromCamCenterDepthConstantRigCostFunctor,
          1,
          3,
          3,
          1> {
  RigWorldFromCamCenterDepthConstantRigCostFunctor(
      const WorldFromCamCenterDepthCostFunctor& depth_cost,
      double cam_from_rig_translation_z)
      : depth_cost(depth_cost),
        cam_from_rig_translation_z(cam_from_rig_translation_z) {}

  const WorldFromCamCenterDepthCostFunctor depth_cost;
  const double cam_from_rig_translation_z;

  template <typename T>
  bool operator()(const T* point3D_in_world,
                  const T* rig_in_world,
                  const T* log_depth_scale,
                  T* result) const {
    const T depth = depth_cost.ComputeDepth(point3D_in_world, rig_in_world) +
                    T(cam_from_rig_translation_z);
    return depth_cost.ComputeResidual(depth, *log_depth_scale, result);
  }
};

struct RigWorldFromCamCenterDepthCostFunctor
    : public AutoDiffCostFunctor<RigWorldFromCamCenterDepthCostFunctor,
                                 1,
                                 3,
                                 3,
                                 3,
                                 1> {
  RigWorldFromCamCenterDepthCostFunctor(
      const WorldFromCamCenterDepthCostFunctor& depth_cost,
      const Eigen::RowVector3d& cam_axis_in_rig)
      : depth_cost(depth_cost), cam_axis_in_rig(cam_axis_in_rig) {}

  const WorldFromCamCenterDepthCostFunctor depth_cost;
  const Eigen::RowVector3d cam_axis_in_rig;

  template <typename T>
  bool operator()(const T* point3D_in_world,
                  const T* rig_in_world,
                  const T* cam_in_rig,
                  const T* log_depth_scale,
                  T* result) const {
    const T depth =
        depth_cost.ComputeDepth(point3D_in_world, rig_in_world) -
        cam_axis_in_rig.cast<T>().dot(EigenVector3Map<T>(cam_in_rig));
    return depth_cost.ComputeResidual(depth, *log_depth_scale, result);
  }
};

}  // namespace colmap
