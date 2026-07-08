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

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Periodic (azimuthal) camera models such as EQUIRECTANGULAR wrap the x image
// coordinate at the ±π seam, so a raw pixel residual can jump by ~width across
// the seam (e.g. an observation at x ≈ 0 whose 3D point reprojects to
// x ≈ width). Wrap the x-residual into [-width/2, width/2) so the
// bundle-adjustment cost stays continuous across the seam. The offset is
// locally constant, so it does not perturb the residual's derivatives. No-op
// for non-periodic camera models. (Elevation has no wrap, so y is untouched.)
template <typename CameraModel, typename T>
inline void WrapEquirectangularHorizontalSeam(const T* camera_params,
                                              T* residuals) {
  if constexpr (CameraModel::model_id == CameraModelId::kEquirectangular) {
    const T width = camera_params[0];
    residuals[0] -= width * ceres::floor(residuals[0] / width + T(0.5));
  }
}

// Full reprojection error cost function with analytical Jacobians.
// Requires camera model to implement ImgFromCamWithJac().
template <typename CameraModel>
class AnalyticalReprojErrorCostFunction
    : public ceres::SizedCostFunction<2, 3, 7, CameraModel::num_params> {
 public:
  explicit AnalyticalReprojErrorCostFunction(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const double* point3D_in_world = parameters[0];
    const double* cam_from_world = parameters[1];
    const double* camera_params = parameters[2];

    double* J_point = jacobians ? jacobians[0] : nullptr;
    double* J_pose = jacobians ? jacobians[1] : nullptr;
    double* J_params = jacobians ? jacobians[2] : nullptr;

    Eigen::Map<Eigen::Vector2d> residuals_vec(residuals);

    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point_mat(
        J_point);
    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_mat(J_pose);
    Eigen::Map<
        Eigen::Matrix<double, 2, CameraModel::num_params, Eigen::RowMajor>>
        J_params_mat(J_params);
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_Rp_quat_mat;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_uvw_mat;

    const Eigen::Vector3d point3D_in_cam =
        QuaternionRotatePointWithJac(cam_from_world,
                                     point3D_in_world,
                                     J_pose ? J_Rp_quat_mat.data() : nullptr) +
        Eigen::Map<const Eigen::Vector3d>(cam_from_world + 4);

    if (!CameraModel::ImgFromCamWithJac(
            camera_params,
            point3D_in_cam[0],
            point3D_in_cam[1],
            point3D_in_cam[2],
            &residuals[0],
            &residuals[1],
            J_params,
            (J_point || J_pose) ? J_uvw_mat.data() : nullptr)) {
      residuals_vec.setZero();
      if (J_pose) {
        J_pose_mat.setZero();
      }
      if (J_point) {
        J_point_mat.setZero();
      }
      if (J_params) {
        J_params_mat.setZero();
      }
      return true;
    }

    residuals_vec -= point2D_;
    // No-op for non-periodic models. The offset is locally constant, so the
    // analytic Jacobians below are unaffected.
    WrapEquirectangularHorizontalSeam<CameraModel>(camera_params, residuals);

    if (J_point) {
      J_point_mat =
          J_uvw_mat *
          EigenQuaternionMap<double>(cam_from_world).toRotationMatrix();
    }
    if (J_pose) {
      J_pose_mat.leftCols<4>() = J_uvw_mat * J_Rp_quat_mat;
      J_pose_mat.rightCols<3>() = J_uvw_mat;
    }

    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

// Reprojection error cost function with analytical Jacobians for a fixed camera
// pose (variable point and camera calibration). Analytical counterpart of
// ReprojErrorConstantPoseCostFunctor. Requires camera model to implement
// ImgFromCamWithJac(). As in that functor, the fixed pose is stored as a
// precomputed rotation matrix and translation; besides the faster matrix-vector
// transform, the rotation matrix is reused directly for the point Jacobian,
// avoiding a quaternion-to-matrix conversion on every evaluation.
template <typename CameraModel>
class AnalyticalReprojErrorConstantPoseCostFunction
    : public ceres::SizedCostFunction<2, 3, CameraModel::num_params> {
 public:
  AnalyticalReprojErrorConstantPoseCostFunction(const Eigen::Vector2d& point2D,
                                                const Rigid3d& cam_from_world)
      : point2D_(point2D),
        cam_from_world_rotation_(cam_from_world.rotation().toRotationMatrix()),
        cam_from_world_translation_(cam_from_world.translation()) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Eigen::Vector3d> point3D_in_world(parameters[0]);
    const double* camera_params = parameters[1];

    double* J_point = jacobians ? jacobians[0] : nullptr;
    double* J_params = jacobians ? jacobians[1] : nullptr;

    Eigen::Map<Eigen::Vector2d> residuals_vec(residuals);
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point_mat(
        J_point);
    Eigen::Map<
        Eigen::Matrix<double, 2, CameraModel::num_params, Eigen::RowMajor>>
        J_params_mat(J_params);
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_uvw_mat;

    const Eigen::Vector3d point3D_in_cam =
        cam_from_world_rotation_ * point3D_in_world +
        cam_from_world_translation_;

    if (!CameraModel::ImgFromCamWithJac(camera_params,
                                        point3D_in_cam[0],
                                        point3D_in_cam[1],
                                        point3D_in_cam[2],
                                        &residuals[0],
                                        &residuals[1],
                                        J_params,
                                        J_point ? J_uvw_mat.data() : nullptr)) {
      residuals_vec.setZero();
      if (J_point) {
        J_point_mat.setZero();
      }
      if (J_params) {
        J_params_mat.setZero();
      }
      return true;
    }

    residuals_vec -= point2D_;
    // No-op for non-periodic models. The offset is locally constant, so the
    // analytic Jacobian below is unaffected.
    WrapEquirectangularHorizontalSeam<CameraModel>(camera_params, residuals);

    if (J_point) {
      J_point_mat = J_uvw_mat * cam_from_world_rotation_;
    }

    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
  const Eigen::Matrix3d cam_from_world_rotation_;
  const Eigen::Vector3d cam_from_world_translation_;
};

// Standard bundle adjustment cost function for variable
// camera pose, calibration, and point parameters.
template <typename CameraModel>
class ReprojErrorCostFunctor
    : public AutoDiffCostFunctor<ReprojErrorCostFunctor<CameraModel>,
                                 2,
                                 3,
                                 7,
                                 CameraModel::num_params> {
 public:
  explicit ReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D_in_world,
                  const T* const cam_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_world) *
            EigenVector3Map<T>(point3D_in_world) +
        EigenVector3Map<T>(cam_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);
    if (CameraModel::ImgFromCam(camera_params,
                                point3D_in_cam[0],
                                point3D_in_cam[1],
                                point3D_in_cam[2],
                                &residuals[0],
                                &residuals[1])) {
      residuals_vec -= point2D_.cast<T>();
      WrapEquirectangularHorizontalSeam<CameraModel>(camera_params, residuals);
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

// Bundle adjustment cost function for variable camera calibration and point
// parameters, and fixed camera pose. Since the pose is constant, it is stored
// as a precomputed rotation matrix and translation rather than a quaternion:
// applying a fixed rotation as a matrix-vector product is faster than a
// quaternion rotation on every evaluation.
template <typename CameraModel>
class ReprojErrorConstantPoseCostFunctor
    : public AutoDiffCostFunctor<
          ReprojErrorConstantPoseCostFunctor<CameraModel>,
          2,
          3,
          CameraModel::num_params> {
 public:
  // The pose is fixed, so precompute and store the rotation as a 3x3 matrix
  // instead of a quaternion: rotating a point with a matrix is faster than
  // quaternion rotation, and the conversion is done once here rather than on
  // every evaluation. (The analytical variant reuses this matrix directly as
  // the point Jacobian, which additionally avoids a quaternion-to-matrix
  // conversion there.)
  ReprojErrorConstantPoseCostFunctor(const Eigen::Vector2d& point2D,
                                     const Rigid3d& cam_from_world)
      : point2D_(point2D),
        cam_from_world_rotation_(cam_from_world.rotation().toRotationMatrix()),
        cam_from_world_translation_(cam_from_world.translation()) {}

  template <typename T>
  bool operator()(const T* const point3D_in_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        cam_from_world_rotation_.cast<T>() *
            EigenVector3Map<T>(point3D_in_world) +
        cam_from_world_translation_.cast<T>();
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);
    if (CameraModel::ImgFromCam(camera_params,
                                point3D_in_cam[0],
                                point3D_in_cam[1],
                                point3D_in_cam[2],
                                &residuals[0],
                                &residuals[1])) {
      residuals_vec -= point2D_.cast<T>();
      WrapEquirectangularHorizontalSeam<CameraModel>(camera_params, residuals);
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
  const Eigen::Matrix3d cam_from_world_rotation_;
  const Eigen::Vector3d cam_from_world_translation_;
};

// Bundle adjustment cost function for variable
// camera pose and calibration parameters, and fixed point.
template <typename CameraModel>
class ReprojErrorConstantPoint3DCostFunctor
    : public AutoDiffCostFunctor<
          ReprojErrorConstantPoint3DCostFunctor<CameraModel>,
          2,
          7,
          CameraModel::num_params> {
 public:
  ReprojErrorConstantPoint3DCostFunctor(const Eigen::Vector2d& point2D,
                                        const Eigen::Vector3d& point3D_in_world)
      : point3D_in_world_(point3D_in_world), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const cam_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_world = point3D_in_world_.cast<T>();
    return reproj_cost_(
        point3D_in_world.data(), cam_from_world, camera_params, residuals);
  }

 private:
  const Eigen::Vector3d point3D_in_world_;
  const ReprojErrorCostFunctor<CameraModel> reproj_cost_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigReprojErrorCostFunctor
    : public AutoDiffCostFunctor<RigReprojErrorCostFunctor<CameraModel>,
                                 2,
                                 3,
                                 7,
                                 7,
                                 CameraModel::num_params> {
 public:
  explicit RigReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D_in_world,
                  const T* const cam_from_rig,
                  const T* const rig_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_rig) *
            (EigenQuaternionMap<T>(rig_from_world) *
                 EigenVector3Map<T>(point3D_in_world) +
             EigenVector3Map<T>(rig_from_world + 4)) +
        EigenVector3Map<T>(cam_from_rig + 4);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);
    if (CameraModel::ImgFromCam(camera_params,
                                point3D_in_cam[0],
                                point3D_in_cam[1],
                                point3D_in_cam[2],
                                &residuals[0],
                                &residuals[1])) {
      residuals_vec -= point2D_.cast<T>();
      WrapEquirectangularHorizontalSeam<CameraModel>(camera_params, residuals);
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

// Rig bundle adjustment cost function for variable camera pose and camera
// calibration and point parameters but fixed rig extrinsic poses.
template <typename CameraModel>
class RigReprojErrorConstantRigCostFunctor
    : public AutoDiffCostFunctor<
          RigReprojErrorConstantRigCostFunctor<CameraModel>,
          2,
          3,
          7,
          CameraModel::num_params> {
 public:
  RigReprojErrorConstantRigCostFunctor(const Eigen::Vector2d& point2D,
                                       const Rigid3d& cam_from_rig)
      : cam_from_rig_(cam_from_rig), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D_in_world,
                  const T* const rig_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 7, 1> cam_from_rig = cam_from_rig_.params.cast<T>();
    return reproj_cost_(point3D_in_world,
                        cam_from_rig.data(),
                        rig_from_world,
                        camera_params,
                        residuals);
  }

 private:
  const Rigid3d cam_from_rig_;
  const RigReprojErrorCostFunctor<CameraModel> reproj_cost_;
};

// Creates the analytical reprojection error cost function for camera models
// that implement ImgFromCamWithJac(). The overloads are selected via SFINAE so
// that AnalyticalReprojErrorCostFunction<CameraModel> is only ever named (and
// thus instantiated) for qualifying models. This avoids instantiating its
// virtual Evaluate() member for models without an analytical Jacobian, which
// would reference the SFINAE-disabled ImgFromCamWithJac() overload.
template <typename CameraModel, typename... Args>
std::enable_if_t<CameraModel::has_img_from_cam_with_jac, ceres::CostFunction*>
CreateAnalyticalReprojErrorCostFunction(Args&&... args) {
  return new AnalyticalReprojErrorCostFunction<CameraModel>(
      std::forward<Args>(args)...);
}

template <typename CameraModel, typename... Args>
std::enable_if_t<!CameraModel::has_img_from_cam_with_jac, ceres::CostFunction*>
CreateAnalyticalReprojErrorCostFunction(Args&&... /*args*/) {
  // Unreachable: callers guard on has_img_from_cam_with_jac.
  return nullptr;
}

// Same SFINAE pattern as CreateAnalyticalReprojErrorCostFunction, for the
// fixed-pose analytical cost function.
template <typename CameraModel, typename... Args>
std::enable_if_t<CameraModel::has_img_from_cam_with_jac, ceres::CostFunction*>
CreateAnalyticalReprojErrorConstantPoseCostFunction(Args&&... args) {
  return new AnalyticalReprojErrorConstantPoseCostFunction<CameraModel>(
      std::forward<Args>(args)...);
}

template <typename CameraModel, typename... Args>
std::enable_if_t<!CameraModel::has_img_from_cam_with_jac, ceres::CostFunction*>
CreateAnalyticalReprojErrorConstantPoseCostFunction(Args&&... /*args*/) {
  // Unreachable: callers guard on has_img_from_cam_with_jac.
  return nullptr;
}

template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction* CreateCameraCostFunction(
    const CameraModelId camera_model_id, Args&&... args) {
  // NOLINTBEGIN(bugprone-macro-parentheses)
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    if constexpr (std::is_same<CostFunctor<CameraModel>,                       \
                               ReprojErrorCostFunctor<CameraModel>>::value &&  \
                  CameraModel::has_img_from_cam_with_jac) {                    \
      return CreateAnalyticalReprojErrorCostFunction<CameraModel>(             \
          std::forward<Args>(args)...);                                        \
    } else if constexpr (std::is_same<CostFunctor<CameraModel>,                \
                                      ReprojErrorConstantPoseCostFunctor<      \
                                          CameraModel>>::value &&              \
                         CameraModel::has_img_from_cam_with_jac) {             \
      return CreateAnalyticalReprojErrorConstantPoseCostFunction<CameraModel>( \
          std::forward<Args>(args)...);                                        \
    } else {                                                                   \
      return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...);    \
    }                                                                          \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  // NOLINTEND(bugprone-macro-parentheses)
}

}  // namespace colmap
