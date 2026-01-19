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

// Rotates the point using angle-axis representation and computes the Jacobian
// of R(angle_axis) * p with respect to the angle-axis parameters.
// J_out is a 3x3 matrix in row-major order. Uses Rodrigues' formula:
//    p' = p + sinc(θ) * (ω x p) + (1-cos(θ))/θ² * (ω x (ω x p))
inline Eigen::Vector3d AngleAxisRotatePointWithJac(const double* angle_axis,
                                                   const double* pt,
                                                   double* J_out) {
  const double wx = angle_axis[0], wy = angle_axis[1], wz = angle_axis[2];
  const double px = pt[0], py = pt[1], pz = pt[2];

  const double theta2 = wx * wx + wy * wy + wz * wz;

  // Cross product: wxp = w x p
  const double w_x_px = wy * pz - wz * py;
  const double w_x_py = wz * px - wx * pz;
  const double w_x_pz = wx * py - wy * px;

  Eigen::Vector3d pt_out;

  // Use Taylor expansion for small angles to avoid numerical instability.
  // The threshold is chosen such that higher-order terms are negligible.
  constexpr double kSmallAngleThreshold = 1e-8;

  if (theta2 > kSmallAngleThreshold) {
    // Cross product: w_x_w_x_p = w x (w x p)
    const double w_x_w_x_px = wy * w_x_pz - wz * w_x_py;
    const double w_x_w_x_py = wz * w_x_px - wx * w_x_pz;
    const double w_x_w_x_pz = wx * w_x_py - wy * w_x_px;

    const double theta = std::sqrt(theta2);
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    const double sinct = sin_theta / theta;
    const double inv_theta2 = 1.0 / theta2;
    const double cosc = (1.0 - cos_theta) * inv_theta2;

    pt_out[0] = px + sinct * w_x_px + cosc * w_x_w_x_px;
    pt_out[1] = py + sinct * w_x_py + cosc * w_x_w_x_py;
    pt_out[2] = pz + sinct * w_x_pz + cosc * w_x_w_x_pz;

    if (J_out) {
      const double c1 = (cos_theta - sinct) * inv_theta2;
      const double c2 = (sinct - 2.0 * cosc) * inv_theta2;
      const double w_dot_p = wx * px + wy * py + wz * pz;
      const double Bx = c1 * w_x_px + c2 * w_x_w_x_px - cosc * px;
      const double By = c1 * w_x_py + c2 * w_x_w_x_py - cosc * py;
      const double Bz = c1 * w_x_pz + c2 * w_x_w_x_pz - cosc * pz;
      const double cosc_w_dot_p = cosc * w_dot_p;

      // Row 0: dp'_x/dω
      J_out[0] = cosc_w_dot_p + Bx * wx;
      J_out[1] = sinct * pz + cosc * w_x_pz + Bx * wy;
      J_out[2] = -sinct * py - cosc * w_x_py + Bx * wz;

      // Row 1: dp'_y/dω
      J_out[3] = -sinct * pz - cosc * w_x_pz + By * wx;
      J_out[4] = cosc_w_dot_p + By * wy;
      J_out[5] = sinct * px + cosc * w_x_px + By * wz;

      // Row 2: dp'_z/dω
      J_out[6] = sinct * py + cosc * w_x_py + Bz * wx;
      J_out[7] = -sinct * px - cosc * w_x_px + Bz * wy;
      J_out[8] = cosc_w_dot_p + Bz * wz;
    }
  } else {
    // Small angle approximation: p' ~= p + ω x p

    pt_out[0] = px + w_x_px;
    pt_out[1] = py + w_x_py;
    pt_out[2] = pz + w_x_pz;

    if (J_out) {
      J_out[0] = 0;
      J_out[1] = pz;
      J_out[2] = -py;
      J_out[3] = -pz;
      J_out[4] = 0;
      J_out[5] = px;
      J_out[6] = py;
      J_out[7] = -px;
      J_out[8] = 0;
    }
  }

  return pt_out;
}

// Full reprojection error cost function with analytical Jacobians.
// Requires camera model to implement ImgFromCamWithJac().
template <typename CameraModel>
class AnalyticalReprojErrorCostFunction
    : public ceres::SizedCostFunction<2, 3, 6, CameraModel::num_params> {
 public:
  explicit AnalyticalReprojErrorCostFunction(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const double* point3D_in_world = parameters[0];
    const double* cam_from_world = parameters[1];
    const double* camera_params = parameters[2];
    const double* cam_from_world_rotation = cam_from_world;
    const double* cam_from_world_trans = cam_from_world + 3;

    double* J_point = jacobians ? jacobians[0] : nullptr;
    double* J_pose = jacobians ? jacobians[1] : nullptr;
    double* J_params = jacobians ? jacobians[2] : nullptr;

    Eigen::Map<Eigen::Vector2d> residuals_vec(residuals);

    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point_mat(
        J_point);
    Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J_pose_mat(J_pose);
    Eigen::Map<
        Eigen::Matrix<double, 2, CameraModel::num_params, Eigen::RowMajor>>
        J_params_mat(J_params);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> J_Rp_aa_mat;
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_uvw_mat;

    const Eigen::Vector3d rotated_point = AngleAxisRotatePointWithJac(
        cam_from_world_rotation,
        point3D_in_world,
        (J_point || J_pose) ? J_Rp_aa_mat.data() : nullptr);
    const Eigen::Vector3d point3D_in_cam =
        rotated_point + Eigen::Map<const Eigen::Vector3d>(cam_from_world_trans);

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
      if (J_point) {
        J_point_mat.setZero();
      }
      if (J_pose) {
        J_pose_mat.setZero();
      }
      if (J_params) {
        J_params_mat.setZero();
      }
      return true;
    }

    residuals_vec -= point2D_;

    if (J_pose) {
      // Jacobian wrt pose: [angle_axis, translation]
      // d(residual)/d(angle_axis) = J_uvw * J_Rp_aa
      // d(residual)/d(translation) = J_uvw * I = J_uvw
      J_pose_mat.leftCols<3>() = J_uvw_mat * J_Rp_aa_mat;
      J_pose_mat.rightCols<3>() = J_uvw_mat;
    }
    if (J_point) {
      // d(p_cam)/d(p_world) = R (rotation matrix)
      Eigen::Matrix3d R;
      ceres::AngleAxisToRotationMatrix(cam_from_world_rotation, R.data());
      // Note: ceres::AngleAxisToRotationMatrix returns column-major R
      J_point_mat = J_uvw_mat * R;
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
                                 6,
                                 CameraModel::num_params> {
 public:
  explicit ReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const cam_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 1> point3D_in_cam;
    ceres::AngleAxisRotatePoint(cam_from_world, point3D, point3D_in_cam.data());
    point3D_in_cam += EigenVector3Map<T>(cam_from_world + 3);
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
      : cam_from_world_(cam_from_world.Log()), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 6, 1> cam_from_world = cam_from_world_.cast<T>();
    return reproj_cost_(
        point3D, cam_from_world.data(), camera_params, residuals);
  }

 private:
  const Eigen::Vector6d cam_from_world_;
  const ReprojErrorCostFunctor<CameraModel> reproj_cost_;
};

// Bundle adjustment cost function for variable
// camera pose and calibration parameters, and fixed point.
template <typename CameraModel>
class ReprojErrorConstantPoint3DCostFunctor
    : public AutoDiffCostFunctor<
          ReprojErrorConstantPoint3DCostFunctor<CameraModel>,
          2,
          6,
          CameraModel::num_params> {
 public:
  ReprojErrorConstantPoint3DCostFunctor(const Eigen::Vector2d& point2D,
                                        const Eigen::Vector3d& point3D)
      : point3D_(point3D), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const cam_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D = point3D_.cast<T>();
    return reproj_cost_(
        point3D.data(), cam_from_world, camera_params, residuals);
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
                                 3,
                                 6,
                                 6,
                                 CameraModel::num_params> {
 public:
  explicit RigReprojErrorCostFunctor(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const cam_from_rig,
                  const T* const rig_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 1> point3D_in_rig;
    ceres::AngleAxisRotatePoint(rig_from_world, point3D, point3D_in_rig.data());
    point3D_in_rig += EigenVector3Map<T>(rig_from_world + 3);
    Eigen::Matrix<T, 3, 1> point3D_in_cam;
    ceres::AngleAxisRotatePoint(
        cam_from_rig, point3D_in_rig.data(), point3D_in_cam.data());
    point3D_in_cam += EigenVector3Map<T>(cam_from_rig + 3);
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
          6,
          CameraModel::num_params> {
 public:
  RigReprojErrorConstantRigCostFunctor(const Eigen::Vector2d& point2D,
                                       const Rigid3d& cam_from_rig)
      : cam_from_rig_(cam_from_rig.Log()), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const rig_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 6, 1> cam_from_rig = cam_from_rig_.cast<T>();
    return reproj_cost_(
        point3D, cam_from_rig.data(), rig_from_world, camera_params, residuals);
  }

 private:
  const Eigen::Vector6d cam_from_rig_;
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
    : public AutoDiffCostFunctor<SampsonErrorCostFunctor, 1, 6> {
 public:
  SampsonErrorCostFunctor(const Eigen::Vector3d& cam_ray1,
                          const Eigen::Vector3d& cam_ray2)
      : cam_ray1_(cam_ray1), cam_ray2_(cam_ray2) {}

  template <typename T>
  bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    Eigen::Matrix<T, 3, 3> R;
    ceres::AngleAxisToRotationMatrix(cam2_from_cam1, R.data());

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -cam2_from_cam1[5], cam2_from_cam1[4], cam2_from_cam1[5], T(0),
        -cam2_from_cam1[3], -cam2_from_cam1[4], cam2_from_cam1[3], T(0);

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
    : public AutoDiffCostFunctor<AbsolutePosePriorCostFunctor, 6, 6> {
 public:
  explicit AbsolutePosePriorCostFunctor(const Rigid3d& sensor_from_world_prior)
      : world_from_sensor_prior_(Inverse(sensor_from_world_prior).Log()) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_rotation(residuals);
    residual_rotation = EigenVector3Map<T>(sensor_from_world) +
                        world_from_sensor_prior_.head<3>().cast<T>();

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_translation(residuals + 3);
    const Eigen::Matrix<T, 3, 1> world_from_sensor_prior_translation =
        world_from_sensor_prior_.tail<3>().cast<T>();
    Eigen::Matrix<T, 3, 1> neg_world_in_sensor_prior;
    ceres::AngleAxisRotatePoint(sensor_from_world,
                                world_from_sensor_prior_translation.data(),
                                neg_world_in_sensor_prior.data());
    residual_translation =
        EigenVector3Map<T>(sensor_from_world + 3) + neg_world_in_sensor_prior;

    return true;
  }

 private:
  const Eigen::Vector6d world_from_sensor_prior_;
};

// 3-DoF error on the sensor position in the world coordinate frame.
struct AbsolutePosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePositionPriorCostFunctor, 3, 6> {
 public:
  explicit AbsolutePosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals) const {
    const T world_from_sensor_rotation[3] = {
        -sensor_from_world[0], -sensor_from_world[1], -sensor_from_world[2]};
    Eigen::Matrix<T, 3, 1> neg_position_in_world;
    ceres::AngleAxisRotatePoint(world_from_sensor_rotation,
                                sensor_from_world + 3,
                                neg_position_in_world.data());
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = neg_position_in_world + position_in_world_prior_.cast<T>();
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 3-DoF error on the rig sensor position in the world coordinate frame.
struct AbsoluteRigPosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPosePositionPriorCostFunctor,
                                 3,
                                 6,
                                 6> {
 public:
  explicit AbsoluteRigPosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals) const {
    const T rig_from_sensor_rotation[3] = {
        -sensor_from_rig[0], -sensor_from_rig[1], -sensor_from_rig[2]};
    Eigen::Matrix<T, 3, 1> sensor_in_rig;
    ceres::AngleAxisRotatePoint(
        rig_from_sensor_rotation, sensor_from_rig + 3, sensor_in_rig.data());
    sensor_in_rig += EigenVector3Map<T>(rig_from_world + 3);
    const T world_from_rig_rotation[3] = {
        -rig_from_world[0], -rig_from_world[1], -rig_from_world[2]};
    Eigen::Matrix<T, 3, 1> sensor_in_world;
    ceres::AngleAxisRotatePoint(
        world_from_rig_rotation, sensor_in_rig.data(), sensor_in_world.data());
    sensor_in_world = -sensor_in_world;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = sensor_in_world - position_in_world_prior_.cast<T>();
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
