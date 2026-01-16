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

#include "colmap/estimators/cost_function_utils.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"

#include <array>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/conditioned_cost_function.h>
#include <ceres/rotation.h>

namespace colmap {

struct EmptyImgFromCamCostPlaceholder {};

// Computes the Jacobian of R(q)*p with respect to Eigen quaternion q =
// [x,y,z,w]. J_out is a 3x4 matrix in row-major order.
// Also rotates pt in-place: pt_out = R(q) * pt_in
inline void QuaternionRotatePointWithJac(const double* q,
                                         double* pt,
                                         double* J_out) {
  // Eigen quaternion: q = [x, y, z, w] where w is the scalar part
  const double qx = q[0], qy = q[1], qz = q[2], qw = q[3];

  // Save original point for Jacobian computation
  const double px = pt[0], py = pt[1], pz = pt[2];

  // R(q) * p using the formula: p' = p + 2*w*(v x p) + 2*(v x (v x p))
  // where v = (qx, qy, qz) is the imaginary part and w = qw is the scalar
  // First compute v  x  p
  const double vxp0 = qy * pz - qz * py;
  const double vxp1 = qz * px - qx * pz;
  const double vxp2 = qx * py - qy * px;

  // Then compute v  x  (v  x  p)
  const double vxvxp0 = qy * vxp2 - qz * vxp1;
  const double vxvxp1 = qz * vxp0 - qx * vxp2;
  const double vxvxp2 = qx * vxp1 - qy * vxp0;

  // p' = p + 2*w*(v x p) + 2*(v x (v x p))
  pt[0] = px + 2.0 * (qw * vxp0 + vxvxp0);
  pt[1] = py + 2.0 * (qw * vxp1 + vxvxp1);
  pt[2] = pz + 2.0 * (qw * vxp2 + vxvxp2);

  if (J_out) {
    // Jacobian d(R*p)/dq for Eigen quaternion [x,y,z,w]
    // Must use the ORIGINAL point (px, py, pz), not the rotated point

    // d(R*p)_0/d[x,y,z,w]
    J_out[0] = 2.0 * (qy * py + qz * pz);
    J_out[1] = 2.0 * (-2.0 * qy * px + qx * py + qw * pz);
    J_out[2] = 2.0 * (-2.0 * qz * px - qw * py + qx * pz);
    J_out[3] = 2.0 * (-qz * py + qy * pz);

    // d(R*p)_1/d[x,y,z,w]
    J_out[4] = 2.0 * (qy * px - 2.0 * qx * py - qw * pz);
    J_out[5] = 2.0 * (qx * px + qz * pz);
    J_out[6] = 2.0 * (qw * px - 2.0 * qz * py + qy * pz);
    J_out[7] = 2.0 * (qz * px - qx * pz);

    // d(R*p)_2/d[x,y,z,w]
    J_out[8] = 2.0 * (qz * px + qw * py - 2.0 * qx * pz);
    J_out[9] = 2.0 * (-qw * px + qz * py - 2.0 * qy * pz);
    J_out[10] = 2.0 * (qx * px + qy * py);
    J_out[11] = 2.0 * (-qy * px + qx * py);
  }
}

inline Eigen::Matrix3d QuaternionToScaledRotation(
    const double* q) {

  // Make convenient names for elements of q.
  const double qx = q[0];
  const double qy = q[1];
  const double qz = q[2];
  const double qw = q[3];
  // This is not to eliminate common sub-expression, but to
  // make the lines shorter so that they fit in 80 columns!
  const double aa = qw * qw;
  const double ab = qw * qx;
  const double ac = qw * qy;
  const double ad = qw * qz;
  const double bb = qx * qx;
  const double bc = qx * qy;
  const double bd = qx * qz;
  const double cc = qy * qy;
  const double cd = qy * qz;
  const double dd = qz * qz;

  Eigen::Matrix3d R;
  R(0, 0) = aa + bb - cc - dd; R(0, 1) = 2 * (bc - ad);  R(0, 2) = 2 * (ac + bd);
  R(1, 0) = 2 * (ad + bc);  R(1, 1) = aa - bb + cc - dd; R(1, 2) = 2 * (cd - ab);
  R(2, 0) = 2 * (bd - ac);  R(2, 1) = 2 * (ab + cd);  R(2, 2) = aa - bb - cc + dd;
  return R;
}

// Full reprojection error cost function with analytical Jacobians for
// SimpleRadialCameraModel. Computes derivatives for the world-to-camera
// transformation (quaternion rotation + translation).
class SimpleRadialReprojErrorCostFunction
    : public ceres::
          SizedCostFunction<2, 4, 3, 3, SimpleRadialCameraModel::num_params> {
 public:
  explicit SimpleRadialReprojErrorCostFunction(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const double* quat = parameters[0];
    const double* trans = parameters[1];
    const double* point3D = parameters[2];
    const double* camera_params = parameters[3];

    double* J_quat = jacobians ? jacobians[0] : nullptr;
    double* J_trans = jacobians ? jacobians[1] : nullptr;
    double* J_point = jacobians ? jacobians[2] : nullptr;
    double* J_params = jacobians ? jacobians[3] : nullptr;

    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_quat_mat(J_quat);
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_trans_mat(
        J_trans);
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point_mat(
        J_point);
    Eigen::Map<Eigen::Matrix<double,
                             2,
                             SimpleRadialCameraModel::num_params,
                             Eigen::RowMajor>>
        J_params_mat(J_params);

    Eigen::Vector3d point3D_in_cam(point3D[0], point3D[1], point3D[2]);
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> J_Rp_q_mat;
    QuaternionRotatePointWithJac(
        quat, point3D_in_cam.data(), jacobians ? J_Rp_q_mat.data() : nullptr);
    point3D_in_cam += Eigen::Map<const Eigen::Vector3d>(trans);

    if (!SimpleRadialCameraModel::ImgFromCamWithJac(camera_params,
                                                    point3D_in_cam[0],
                                                    point3D_in_cam[1],
                                                    point3D_in_cam[2],
                                                    &residuals[0],
                                                    &residuals[1],
                                                    J_params,
                                                    J_trans)) {
      residuals[0] = 0.0;
      residuals[1] = 0.0;
      if (jacobians) {
        J_quat_mat.setZero();
        J_trans_mat.setZero();
        J_point_mat.setZero();
        J_params_mat.setZero();
      }
      return true;
    }

    Eigen::Map<Eigen::Vector2d> residuals_vec(residuals);
    residuals_vec -= point2D_;

    if (jacobians) {
      // J_quat = J_uvw (2x3) * J_Rp_q (3x4) = 2x4
      J_quat_mat = J_trans_mat * J_Rp_q_mat;
      // J_point = J_uvw (2x3) * R (3x3) = 2x3
      // Note: J_trans_mat holds J_uvw since dp_cam/dtrans = I
      J_point_mat = J_trans_mat * QuaternionToScaledRotation(quat);
    }

    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

class SimpleRadialSizedCostFunction
    : public ceres::
          SizedCostFunction<2, 3, SimpleRadialCameraModel::num_params> {
 public:
  explicit SimpleRadialSizedCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    if (SimpleRadialCameraModel::ImgFromCamWithJac(
            parameters[1],
            parameters[0][0],
            parameters[0][1],
            parameters[0][2],
            &residuals[0],
            &residuals[1],
            jacobians ? jacobians[1] : nullptr,
            jacobians ? jacobians[0] : nullptr)) {
      residuals[0] -= observed_x_;
      residuals[1] -= observed_y_;
    } else {
      residuals[0] = 0;
      residuals[1] = 0;
    }
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Standard bundle adjustment cost function for variable
// camera pose, calibration, and point parameters.
template <typename CameraModel>
class ReprojErrorCostFunctor
    : public AutoDiffCostFunctor<ReprojErrorCostFunctor<CameraModel>,
                                 2,
                                 4,
                                 3,
                                 3,
                                 CameraModel::num_params> {
 public:
  explicit ReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)),
        observed_y_(point2D(1)),
        img_from_cam_cost_([&point2D]() {
          if constexpr (std::is_same<CameraModel,
                                     SimpleRadialCameraModel>::value) {
            return ceres::
                CostFunctionToFunctor<2, 4, 3, 3, CameraModel::num_params>(
                    new SimpleRadialReprojErrorCostFunction(point2D));
          } else {
            (void)point2D;
            return EmptyImgFromCamCostPlaceholder{};
          }
        }()) {}

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    // const Eigen::Matrix<T, 3, 1> point3D_in_cam =
    //     EigenQuaternionMap<T>(cam_from_world_rotation) *
    //         EigenVector3Map<T>(point3D) +
    //     EigenVector3Map<T>(cam_from_world_translation);
    // if constexpr (std::is_same<CameraModel, SimpleRadialCameraModel>::value)
    // {
    //   img_from_cam_cost_(point3D_in_cam.data(), camera_params, residuals);
    // } else {
    //   if (CameraModel::ImgFromCam(camera_params,
    //                               point3D_in_cam[0],
    //                               point3D_in_cam[1],
    //                               point3D_in_cam[2],
    //                               &residuals[0],
    //                               &residuals[1])) {
    //     residuals[0] -= T(observed_x_);
    //     residuals[1] -= T(observed_y_);
    //   } else {
    //     residuals[0] = T(0);
    //     residuals[1] = T(0);
    //   }
    // }
    if constexpr (std::is_same<CameraModel, SimpleRadialCameraModel>::value) {
      img_from_cam_cost_(cam_from_world_rotation,
                         cam_from_world_translation,
                         point3D,
                         camera_params,
                         residuals);
    } else {
      const Eigen::Matrix<T, 3, 1> point3D_in_cam =
          EigenQuaternionMap<T>(cam_from_world_rotation) *
              EigenVector3Map<T>(point3D) +
          EigenVector3Map<T>(cam_from_world_translation);
      if (CameraModel::ImgFromCam(camera_params,
                                  point3D_in_cam[0],
                                  point3D_in_cam[1],
                                  point3D_in_cam[2],
                                  &residuals[0],
                                  &residuals[1])) {
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);
      } else {
        residuals[0] = T(0);
        residuals[1] = T(0);
      }
    }
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
  std::conditional_t<
      std::is_same<CameraModel, SimpleRadialCameraModel>::value,
      ceres::CostFunctionToFunctor<2, 4, 3, 3, CameraModel::num_params>,
      EmptyImgFromCamCostPlaceholder>
      img_from_cam_cost_;
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
  bool operator()(const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Quaternion<T> cam_from_world_rotation =
        cam_from_world_.rotation.cast<T>();
    const Eigen::Matrix<T, 3, 1> cam_from_world_translation =
        cam_from_world_.translation.cast<T>();
    return reproj_cost_(cam_from_world_rotation.coeffs().data(),
                        cam_from_world_translation.data(),
                        point3D,
                        camera_params,
                        residuals);
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
          4,
          3,
          CameraModel::num_params> {
 public:
  ReprojErrorConstantPoint3DCostFunctor(const Eigen::Vector2d& point2D,
                                        const Eigen::Vector3d& point3D)
      : point3D_(point3D), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D = point3D_.cast<T>();
    return reproj_cost_(cam_from_world_rotation,
                        cam_from_world_translation,
                        point3D.data(),
                        camera_params,
                        residuals);
  }

 private:
  const Eigen::Vector3d point3D_;
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
                                 4,
                                 3,
                                 4,
                                 3,
                                 3,
                                 CameraModel::num_params> {
 public:
  explicit RigReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

  template <typename T>
  bool operator()(const T* const cam_from_rig_rotation,
                  const T* const cam_from_rig_translation,
                  const T* const rig_from_world_rotation,
                  const T* const rig_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_rig_rotation) *
            (EigenQuaternionMap<T>(rig_from_world_rotation) *
                 EigenVector3Map<T>(point3D) +
             EigenVector3Map<T>(rig_from_world_translation)) +
        EigenVector3Map<T>(cam_from_rig_translation);
    if (CameraModel::ImgFromCam(camera_params,
                                point3D_in_cam[0],
                                point3D_in_cam[1],
                                point3D_in_cam[2],
                                &residuals[0],
                                &residuals[1])) {
      residuals[0] -= T(observed_x_);
      residuals[1] -= T(observed_y_);
    } else {
      residuals[0] = T(0);
      residuals[1] = T(0);
    }
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
};

// Rig bundle adjustment cost function for variable camera pose and camera
// calibration and point parameters but fixed rig extrinsic poses.
template <typename CameraModel>
class RigReprojErrorConstantRigCostFunctor
    : public AutoDiffCostFunctor<
          RigReprojErrorConstantRigCostFunctor<CameraModel>,
          2,
          4,
          3,
          3,
          CameraModel::num_params> {
 public:
  RigReprojErrorConstantRigCostFunctor(const Eigen::Vector2d& point2D,
                                       const Rigid3d& cam_from_rig)
      : cam_from_rig_(cam_from_rig), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const rig_from_world_rotation,
                  const T* const rig_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Quaternion<T> cam_from_rig_rotation =
        cam_from_rig_.rotation.cast<T>();
    const Eigen::Matrix<T, 3, 1> cam_from_rig_translation =
        cam_from_rig_.translation.cast<T>();
    return reproj_cost_(cam_from_rig_rotation.coeffs().data(),
                        cam_from_rig_translation.data(),
                        rig_from_world_rotation,
                        rig_from_world_translation,
                        point3D,
                        camera_params,
                        residuals);
  }

 private:
  const Rigid3d cam_from_rig_;
  const RigReprojErrorCostFunctor<CameraModel> reproj_cost_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `SphereManifold`.
class SampsonErrorCostFunctor
    : public AutoDiffCostFunctor<SampsonErrorCostFunctor, 1, 4, 3> {
 public:
  SampsonErrorCostFunctor(const Eigen::Vector3d& cam_ray1,
                          const Eigen::Vector3d& cam_ray2)
      : cam_ray1_(cam_ray1), cam_ray2_(cam_ray2) {}

  template <typename T>
  bool operator()(const T* const cam2_from_cam1_rotation,
                  const T* const cam2_from_cam1_translation,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 3> R =
        EigenQuaternionMap<T>(cam2_from_cam1_rotation).toRotationMatrix();

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -cam2_from_cam1_translation[2], cam2_from_cam1_translation[1],
        cam2_from_cam1_translation[2], T(0), -cam2_from_cam1_translation[0],
        -cam2_from_cam1_translation[1], cam2_from_cam1_translation[0], T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> epipolar_line1 = E * cam_ray1_.cast<T>();
    const Eigen::Matrix<T, 3, 1> cam_ray2 = cam_ray2_.cast<T>();
    const T num = cam_ray2.dot(epipolar_line1);
    const Eigen::Matrix<T, 4, 1> denom(cam_ray2.dot(E.col(0)),
                                       cam_ray2.dot(E.col(1)),
                                       epipolar_line1.x(),
                                       epipolar_line1.y());
    const T denom_norm = denom.norm();
    if (denom_norm == static_cast<T>(0)) {
      residuals[0] = static_cast<T>(0);
    } else {
      residuals[0] = num / denom_norm;
    }

    return true;
  }

 private:
  const Eigen::Vector3d cam_ray1_;
  const Eigen::Vector3d cam_ray2_;
};

template <typename T>
inline void EigenQuaternionToAngleAxis(const T* eigen_quaternion,
                                       T* angle_axis) {
  const T quaternion[4] = {eigen_quaternion[3],
                           eigen_quaternion[0],
                           eigen_quaternion[1],
                           eigen_quaternion[2]};
  ceres::QuaternionToAngleAxis(quaternion, angle_axis);
}

// 6-DoF error on the absolute sensor pose. The residual is the log of the error
// pose, splitting SE(3) into SO(3) x R^3. The residual is computed in the
// sensor frame. Its first and last three components correspond to the rotation
// and translation errors, respectively.
struct AbsolutePosePriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePriorCostFunctor, 6, 4, 3> {
 public:
  explicit AbsolutePosePriorCostFunctor(const Rigid3d& sensor_from_world_prior)
      : world_from_sensor_prior_(Inverse(sensor_from_world_prior)) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world_rotation,
                  const T* const sensor_from_world_translation,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_prior_rotation =
        EigenQuaternionMap<T>(sensor_from_world_rotation) *
        world_from_sensor_prior_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(sensor_from_world_translation) +
        EigenQuaternionMap<T>(sensor_from_world_rotation) *
            world_from_sensor_prior_.translation.cast<T>();

    return true;
  }

 private:
  const Rigid3d world_from_sensor_prior_;
};

// 3-DoF error on the sensor position in the world coordinate frame.
struct AbsolutePosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePositionPriorCostFunctor,
                                 3,
                                 4,
                                 3> {
 public:
  explicit AbsolutePosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world_rotation,
                  const T* const sensor_from_world_translation,
                  T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = position_in_world_prior_.cast<T>() +
                EigenQuaternionMap<T>(sensor_from_world_rotation).inverse() *
                    EigenVector3Map<T>(sensor_from_world_translation);
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 3-DoF error on the rig sensor position in the world coordinate frame.
struct AbsoluteRigPosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPosePositionPriorCostFunctor,
                                 3,
                                 4,
                                 3,
                                 4,
                                 3> {
 public:
  explicit AbsoluteRigPosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig_rotation,
                  const T* const sensor_from_rig_translation,
                  const T* const rig_from_world_rotation,
                  const T* const rig_from_world_translation,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig_rotation) *
        EigenQuaternionMap<T>(rig_from_world_rotation);
    const Eigen::Matrix<T, 3, 1> sensor_from_world_translation =
        EigenVector3Map<T>(sensor_from_rig_translation) +
        EigenQuaternionMap<T>(sensor_from_rig_rotation) *
            EigenVector3Map<T>(rig_from_world_translation);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals =
        position_in_world_prior_.cast<T>() +
        sensor_from_world_rotation.inverse() * sensor_from_world_translation;
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 6-DoF error between two absolute camera poses based on a prior on their
// relative pose, with identical scale for the translation. The residual is
// computed in the frame of camera i. Its first and last three components
// correspond to the rotation and translation errors, respectively.
//
// Derivation:
//    i_T_w = ΔT_i·i_T_j·j_T_w
//    where ΔT_i = exp(η_i) is the resjdual in SE(3) and η_i in tangent space.
//    Thus η_i = log(i_T_w·j_T_w⁻¹·j_T_i)
//    Rotation term: ΔR = log(i_R_w·j_R_w⁻¹·j_R_i)
//    Translation term: Δt = i_t_w + i_R_w·j_R_w⁻¹·(j_t_i -j_t_w)
struct RelativePosePriorCostFunctor
    : public AutoDiffCostFunctor<RelativePosePriorCostFunctor, 6, 4, 3, 4, 3> {
 public:
  explicit RelativePosePriorCostFunctor(const Rigid3d& i_from_j_prior)
      : j_from_i_prior_(Inverse(i_from_j_prior)) {}

  template <typename T>
  bool operator()(const T* const i_from_world_rotation,
                  const T* const i_from_world_translation,
                  const T* const j_from_world_rotation,
                  const T* const j_from_world_translation,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> i_from_j_rotation =
        EigenQuaternionMap<T>(i_from_world_rotation) *
        EigenQuaternionMap<T>(j_from_world_rotation).inverse();
    const Eigen::Quaternion<T> param_from_prior_rotation =
        i_from_j_rotation * j_from_i_prior_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    const Eigen::Matrix<T, 3, 1> j_from_i_prior_translation =
        j_from_i_prior_.translation.cast<T>() -
        EigenVector3Map<T>(j_from_world_translation);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(i_from_world_translation) +
        i_from_j_rotation * j_from_i_prior_translation;

    return true;
  }

 private:
  const Rigid3d j_from_i_prior_;
};

// Cost function for aligning one 3D point with a reference 3D point with
// covariance. The Residual is computed in frame b. Coordinate transformation
// convention is equivalent to colmap::Sim3d.
struct Point3DAlignmentCostFunctor
    : public AutoDiffCostFunctor<Point3DAlignmentCostFunctor, 3, 3, 4, 3, 1> {
 public:
  explicit Point3DAlignmentCostFunctor(const Eigen::Vector3d& point_in_b_prior,
                                       bool use_log_scale = true)
      : point_in_b_prior_(point_in_b_prior), use_log_scale_(use_log_scale) {}

  template <typename T>
  bool operator()(
      const T* const point_in_a,
      const T* const b_from_a_rotation,
      const T* const b_from_a_translation,
      const T* const b_from_a_scale_param,  // could be scale or log_scale
                                            // depending on use_log_scale_
      T* residuals_ptr) const {
    // Select whether to exponentiate
    const T b_from_a_scale = use_log_scale_
                                 ? ceres::exp(b_from_a_scale_param[0])
                                 : b_from_a_scale_param[0];

    const Eigen::Matrix<T, 3, 1> point_in_b =
        EigenQuaternionMap<T>(b_from_a_rotation) *
            EigenVector3Map<T>(point_in_a) * b_from_a_scale +
        EigenVector3Map<T>(b_from_a_translation);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = point_in_b - point_in_b_prior_.cast<T>();
    return true;
  }

 private:
  const Eigen::Vector3d point_in_b_prior_;
  const bool use_log_scale_;
};

template <template <typename> class CostFunctor, typename... Args>
ceres::CostFunction* CreateCameraCostFunction(
    const CameraModelId camera_model_id, Args&&... args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::model_id:                                             \
    return CostFunctor<CameraModel>::Create(std::forward<Args>(args)...); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

// Compute polynomial coefficients from cross-products of SVD-derived vectors
// for the Fetzer focal length estimation method. The coefficients encode the
// relationship between the two focal lengths derived from the fundamental
// matrix constraint.
// See: "Stable Intrinsic Auto-Calibration from Fundamental Matrices of Devices
// with Uncorrelated Camera Parameters", Fetzer et al., WACV 2020.
inline Eigen::Vector4d ComputeFetzerPolynomialCoefficients(
    const Eigen::Vector3d& ai,
    const Eigen::Vector3d& bi,
    const Eigen::Vector3d& aj,
    const Eigen::Vector3d& bj,
    const int u,
    const int v) {
  return {ai(u) * aj(v) - ai(v) * aj(u),
          ai(u) * bj(v) - ai(v) * bj(u),
          bi(u) * aj(v) - bi(v) * aj(u),
          bi(u) * bj(v) - bi(v) * bj(u)};
}

// Decompose the fundamental matrix (adjusted by principal points) via SVD and
// compute the polynomial coefficients for the Fetzer focal length method.
// Returns three coefficient vectors used to estimate the two focal lengths.
inline std::array<Eigen::Vector4d, 3> DecomposeFundamentalMatrixForFetzer(
    const Eigen::Matrix3d& i1_F_i0,
    const Eigen::Vector2d& principal_point0,
    const Eigen::Vector2d& principal_point1) {
  Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
  K0(0, 2) = principal_point0(0);
  K0(1, 2) = principal_point0(1);

  Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
  K1(0, 2) = principal_point1(0);
  K1(1, 2) = principal_point1(1);

  const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& s = svd.singularValues();

  const Eigen::Vector3d v0 = svd.matrixV().col(0);
  const Eigen::Vector3d v1 = svd.matrixV().col(1);

  const Eigen::Vector3d u0 = svd.matrixU().col(0);
  const Eigen::Vector3d u1 = svd.matrixU().col(1);

  const Eigen::Vector3d ai(s(0) * s(0) * (v0(0) * v0(0) + v0(1) * v0(1)),
                           s(0) * s(1) * (v0(0) * v1(0) + v0(1) * v1(1)),
                           s(1) * s(1) * (v1(0) * v1(0) + v1(1) * v1(1)));

  const Eigen::Vector3d aj(u1(0) * u1(0) + u1(1) * u1(1),
                           -(u0(0) * u1(0) + u0(1) * u1(1)),
                           u0(0) * u0(0) + u0(1) * u0(1));

  const Eigen::Vector3d bi(s(0) * s(0) * v0(2) * v0(2),
                           s(0) * s(1) * v0(2) * v1(2),
                           s(1) * s(1) * v1(2) * v1(2));

  const Eigen::Vector3d bj(u1(2) * u1(2), -(u0(2) * u1(2)), u0(2) * u0(2));

  const Eigen::Vector4d d01 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 0);
  const Eigen::Vector4d d02 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 0, 2);
  const Eigen::Vector4d d12 =
      ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 2, 1);
  return {d01, d02, d12};
}

// Cost functor for estimating focal lengths from the fundamental matrix using
// the Fetzer method. Used when two images have different cameras (different
// focal lengths). The residual measures the relative error between the
// estimated and expected focal lengths based on the fundamental matrix
// constraint.
class FetzerFocalLengthCostFunctor {
 public:
  FetzerFocalLengthCostFunctor(const Eigen::Matrix3d& i1_F_i0,
                               const Eigen::Vector2d& principal_point0,
                               const Eigen::Vector2d& principal_point1)
      : coeffs_(DecomposeFundamentalMatrixForFetzer(
            i1_F_i0, principal_point0, principal_point1)) {}

  static ceres::CostFunction* Create(const Eigen::Matrix3d& i1_F_i0,
                                     const Eigen::Vector2d& principal_point0,
                                     const Eigen::Vector2d& principal_point1) {
    return new ceres::
        AutoDiffCostFunction<FetzerFocalLengthCostFunctor, 2, 1, 1>(
            new FetzerFocalLengthCostFunctor(
                i1_F_i0, principal_point0, principal_point1));
  }

  template <typename T>
  bool operator()(const T* const fi_, const T* const fj_, T* residuals) const {
    const Eigen::Vector<T, 4> d01_ = coeffs_[0].cast<T>();
    const Eigen::Vector<T, 4> d12_ = coeffs_[2].cast<T>();

    const T fi = fi_[0];
    const T fj = fj_[0];

    T di = (fj * fj * d01_(0) + d01_(1));
    T dj = (fi * fi * d12_(0) + d12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d01_(2) + d01_(3)) / di;
    const T K1_12 = -(fi * fi * d12_(1) + d12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 3> coeffs_;
};

// Cost functor for estimating focal length from the fundamental matrix using
// the Fetzer method. Used when two images share the same camera (same focal
// length). The residual measures the relative error between the estimated and
// expected focal length based on the fundamental matrix constraint.
class FetzerFocalLengthSameCameraCostFunctor {
 public:
  FetzerFocalLengthSameCameraCostFunctor(const Eigen::Matrix3d& i1_F_i0,
                                         const Eigen::Vector2d& principal_point)
      : coeffs_(DecomposeFundamentalMatrixForFetzer(
            i1_F_i0, principal_point, principal_point)) {}

  static ceres::CostFunction* Create(const Eigen::Matrix3d& i1_F_i0,
                                     const Eigen::Vector2d& principal_point) {
    return new ceres::
        AutoDiffCostFunction<FetzerFocalLengthSameCameraCostFunctor, 2, 1>(
            new FetzerFocalLengthSameCameraCostFunctor(i1_F_i0,
                                                       principal_point));
  }

  template <typename T>
  bool operator()(const T* const fi_, T* residuals) const {
    const Eigen::Vector<T, 4> d01_ = coeffs_[0].cast<T>();
    const Eigen::Vector<T, 4> d12_ = coeffs_[2].cast<T>();

    const T fi = fi_[0];
    const T fj = fi_[0];

    T di = (fj * fj * d01_(0) + d01_(1));
    T dj = (fi * fi * d12_(0) + d12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d01_(2) + d01_(3)) / di;
    const T K1_12 = -(fi * fi * d12_(1) + d12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  const std::array<Eigen::Vector4d, 3> coeffs_;
};

}  // namespace colmap
