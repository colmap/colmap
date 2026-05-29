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

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Rotates the point and computes the Jacobian of R(q) * p with respect to Eigen
// quaternions. J_out is a 3x4 matrix in row-major order.
inline Eigen::Vector3d QuaternionRotatePointWithJac(const double* q,
                                                    const double* pt,
                                                    double* J_out) {
  const double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const double px = pt[0], py = pt[1], pz = pt[2];

  // Common sub-expressions.
  const double qx_py = qx * py;
  const double qx_pz = qx * pz;
  const double qy_px = qy * px;
  const double qy_pz = qy * pz;
  const double qz_px = qz * px;
  const double qz_py = qz * py;

  // R(q) * p using the formula: p' = p + 2*w*(v x p) + 2*(v x (v x p)),
  // where v = (qx, qy, qz) is the imaginary part and w = qw is the scalar.

  // First compute v  x  p.
  const double v_x_p0 = qy_pz - qz_py;
  const double v_x_p1 = qz_px - qx_pz;
  const double v_x_p2 = qx_py - qy_px;

  // Then compute v  x  (v  x  p).
  const double v_x_v_x_p0 = qy * v_x_p2 - qz * v_x_p1;
  const double v_x_v_x_p1 = qz * v_x_p0 - qx * v_x_p2;
  const double v_x_v_x_p2 = qx * v_x_p1 - qy * v_x_p0;

  // p' = p + 2*w*(v x p) + 2*(v x (v x p)).
  Eigen::Vector3d pt_out(px + 2.0 * (qw * v_x_p0 + v_x_v_x_p0),
                         py + 2.0 * (qw * v_x_p1 + v_x_v_x_p1),
                         pz + 2.0 * (qw * v_x_p2 + v_x_v_x_p2));

  if (J_out) {
    // Jacobian d(R*p) / dq for Eigen quaternions (x, y, z, w).
    // Must use the ORIGINAL point (px, py, pz), not the rotated point.

    // Common sub-expressions.
    const double qx_px = qx * px;
    const double qx_pz = qx * pz;
    const double qy_px = qy * px;
    const double qy_py = qy * py;
    const double qz_pz = qz * pz;
    const double qw_px = qw * px;
    const double qw_py = qw * py;
    const double qw_pz = qw * pz;

    // d(R*p)_x / d(x,y,z,w)
    J_out[0] = 2.0 * (qy_py + qz_pz);
    J_out[1] = 2.0 * (-2.0 * qy_px + qx_py + qw_pz);
    J_out[2] = 2.0 * (-2.0 * qz_px - qw_py + qx_pz);
    J_out[3] = 2.0 * (-qz_py + qy_pz);

    // d(R*p)_y / d(x,y,z,w)
    J_out[4] = 2.0 * (qy_px - 2.0 * qx_py - qw_pz);
    J_out[5] = 2.0 * (qx_px + qz_pz);
    J_out[6] = 2.0 * (qw_px - 2.0 * qz_py + qy_pz);
    J_out[7] = 2.0 * (qz_px - qx_pz);

    // d(R*p)_z / d(x,y,z,w)
    J_out[8] = 2.0 * (qz_px + qw_py - 2.0 * qx_pz);
    J_out[9] = 2.0 * (-qw_px + qz_py - 2.0 * qy_pz);
    J_out[10] = 2.0 * (qx_px + qy_py);
    J_out[11] = 2.0 * (-qy_px + qx_py);
  }

  return pt_out;
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
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.

template <typename CameraModel>
class ReprojErrorConstantPoseCostFunctor
    : public AutoDiffCostFunctor<
          ReprojErrorConstantPoseCostFunctor<CameraModel>,
          2,
          3,
          CameraModel::num_params> {
 public:
  ReprojErrorConstantPoseCostFunctor(const Eigen::Vector2d& point2D,
                                     const Rigid3d& cam_from_world)
      : cam_from_world_(cam_from_world), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D_in_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 7, 1> cam_from_world =
        cam_from_world_.params.cast<T>();
    return reproj_cost_(
        point3D_in_world, cam_from_world.data(), camera_params, residuals);
  }

 private:
  const Rigid3d cam_from_world_;
  const ReprojErrorCostFunctor<CameraModel> reproj_cost_;
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

template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction* CreateCameraCostFunction(
    const CameraModelId camera_model_id, Args&&... args) {
  // NOLINTBEGIN(bugprone-macro-parentheses)
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                        \
  case CameraModel::model_id:                                                 \
    if constexpr (std::is_same<CostFunctor<CameraModel>,                      \
                               ReprojErrorCostFunctor<CameraModel>>::value && \
                  CameraModel::has_img_from_cam_with_jac) {                   \
      return new AnalyticalReprojErrorCostFunction<CameraModel>(              \
          std::forward<Args>(args)...);                                       \
    } else {                                                                  \
      return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...);   \
    }                                                                         \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  // NOLINTEND(bugprone-macro-parentheses)
}

}  // namespace colmap
