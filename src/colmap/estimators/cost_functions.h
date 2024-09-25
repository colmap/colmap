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

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/conditioned_cost_function.h>
#include <ceres/rotation.h>

namespace colmap {

template <typename T>
using EigenVector3Map = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
template <typename T>
using EigenQuaternionMap = Eigen::Map<const Eigen::Quaternion<T>>;
using EigenMatrix6d = Eigen::Matrix<double, 6, 6>;

enum class CovarianceType {
  IDENTITY = 1,
  DIAGONAL = 2,
  DENSE = 3,
};

template <CovarianceType CTYPE = CovarianceType::DENSE>
inline Eigen::MatrixXd SqrtInformation(const Eigen::MatrixXd& covariance) {
  if constexpr (CTYPE == CovarianceType::IDENTITY) {
    return covariance;
  } else {
    return covariance.inverse().llt().matrixL().transpose();
  }
}

inline CovarianceType GetCovarianceType(const Eigen::MatrixXd& covariance) {
  if (covariance.isDiagonal()) {
    if (covariance.isIdentity()) {
      return CovarianceType::IDENTITY;
    } else {
      return CovarianceType::DIAGONAL;
    }
  } else {
    return CovarianceType::DENSE;
  }
}

template <CovarianceType CTYPE>
inline bool CheckCovarianceByType(const Eigen::MatrixXd& covariance) {
  if constexpr (CTYPE == CovarianceType::IDENTITY) {
    return covariance.isIdentity();
  } else if constexpr (CTYPE == CovarianceType::DIAGONAL) {
    return covariance.isDiagonal();
  } else {
    return true;
  }
}

template <typename T, int N, CovarianceType CTYPE = CovarianceType::IDENTITY>
inline void ApplySqrtInformation(
    T* residuals, const Eigen::Matrix<double, N, N>& sqrt_information) {
  if constexpr (CTYPE == CovarianceType::IDENTITY) {
    return;
  } else if constexpr (CTYPE == CovarianceType::DIAGONAL) {
    for (int i = 0; i < sqrt_information.rows(); ++i) {
      residuals[i] *= sqrt_information(i, i);
    }
  } else if constexpr (CTYPE == CovarianceType::DENSE) {
    Eigen::Map<Eigen::Matrix<T, N, 1>> residuals_map(residuals);
    residuals_map.applyOnTheLeft(sqrt_information.template cast<T>());
  }
}

// Standard bundle adjustment cost function for variable
// camera pose, calibration, and point parameters.
template <typename CameraModel, CovarianceType CTYPE = CovarianceType::IDENTITY>
class ReprojErrorCostFunction {
 public:
  explicit ReprojErrorCostFunction(
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity())
      : observed_x_(point2D(0)),
        observed_y_(point2D(1)),
        sqrt_info_point2D_(SqrtInformation<CTYPE>(point2D_covar)) {
    THROW_CHECK(CheckCovarianceByType<CTYPE>(point2D_covar));
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity()) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorCostFunction<CameraModel, CTYPE>,
            2,
            4,
            3,
            3,
            CameraModel::num_params>(
        new ReprojErrorCostFunction(point2D, point2D_covar)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_world_rotation) *
            EigenVector3Map<T>(point3D) +
        EigenVector3Map<T>(cam_from_world_translation);
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);
    if constexpr (CTYPE != CovarianceType::IDENTITY) {
      ApplySqrtInformation<T, 2, CTYPE>(residuals, sqrt_info_point2D_);
    }
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
  const Eigen::Matrix2d sqrt_info_point2D_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel, CovarianceType CTYPE = CovarianceType::IDENTITY>
class ReprojErrorConstantPoseCostFunction
    : public ReprojErrorCostFunction<CameraModel, CTYPE> {
  using Parent = ReprojErrorCostFunction<CameraModel, CTYPE>;

 public:
  ReprojErrorConstantPoseCostFunction(
      const Rigid3d& cam_from_world,
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity())
      : Parent(point2D, point2D_covar), cam_from_world_(cam_from_world) {}

  static ceres::CostFunction* Create(
      const Rigid3d& cam_from_world,
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity()) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorConstantPoseCostFunction<CameraModel, CTYPE>,
            2,
            3,
            CameraModel::num_params>(new ReprojErrorConstantPoseCostFunction(
        cam_from_world, point2D, point2D_covar)));
  }

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Quaternion<T> cam_from_world_rotation =
        cam_from_world_.rotation.cast<T>();
    const Eigen::Matrix<T, 3, 1> cam_from_world_translation =
        cam_from_world_.translation.cast<T>();
    return Parent::operator()(cam_from_world_rotation.coeffs().data(),
                              cam_from_world_translation.data(),
                              point3D,
                              camera_params,
                              residuals);
  }

 private:
  const Rigid3d& cam_from_world_;
};

// Bundle adjustment cost function for variable
// camera pose and calibration parameters, and fixed point.
template <typename CameraModel, CovarianceType CTYPE = CovarianceType::IDENTITY>
class ReprojErrorConstantPoint3DCostFunction
    : public ReprojErrorCostFunction<CameraModel, CTYPE> {
  using Parent = ReprojErrorCostFunction<CameraModel, CTYPE>;

 public:
  ReprojErrorConstantPoint3DCostFunction(
      const Eigen::Vector2d& point2D,
      const Eigen::Vector3d& point3D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity())
      : Parent(point2D, point2D_covar),
        point3D_x_(point3D(0)),
        point3D_y_(point3D(1)),
        point3D_z_(point3D(2)) {}

  static ceres::CostFunction* Create(
      const Eigen::Vector2d& point2D,
      const Eigen::Vector3d& point3D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity()) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorConstantPoint3DCostFunction<CameraModel, CTYPE>,
            2,
            4,
            3,
            CameraModel::num_params>(new ReprojErrorConstantPoint3DCostFunction(
        point2D, point3D, point2D_covar)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const camera_params,
                  T* residuals) const {
    const T point3D[3] = {T(point3D_x_), T(point3D_y_), T(point3D_z_)};
    return Parent::operator()(cam_from_world_rotation,
                              cam_from_world_translation,
                              point3D,
                              camera_params,
                              residuals);
  }

 private:
  const double point3D_x_;
  const double point3D_y_;
  const double point3D_z_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel, CovarianceType CTYPE = CovarianceType::IDENTITY>
class RigReprojErrorCostFunction {
 public:
  explicit RigReprojErrorCostFunction(
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity())
      : observed_x_(point2D(0)),
        observed_y_(point2D(1)),
        sqrt_info_point2D_(SqrtInformation<CTYPE>(point2D_covar)) {
    THROW_CHECK(CheckCovarianceByType<CTYPE>(point2D_covar));
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity()) {
    return (new ceres::AutoDiffCostFunction<
            RigReprojErrorCostFunction<CameraModel, CTYPE>,
            2,
            4,
            3,
            4,
            3,
            3,
            CameraModel::num_params>(
        new RigReprojErrorCostFunction(point2D, point2D_covar)));
  }

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
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);
    if constexpr (CTYPE != CovarianceType::IDENTITY) {
      ApplySqrtInformation<T, 2, CTYPE>(residuals, sqrt_info_point2D_);
    }
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
  const Eigen::Matrix2d sqrt_info_point2D_;
};

// Rig bundle adjustment cost function for variable camera pose and camera
// calibration and point parameters but fixed rig extrinsic poses.
template <typename CameraModel, CovarianceType CTYPE = CovarianceType::IDENTITY>
class RigReprojErrorConstantRigCostFunction
    : public RigReprojErrorCostFunction<CameraModel, CTYPE> {
  using Parent = RigReprojErrorCostFunction<CameraModel, CTYPE>;

 public:
  RigReprojErrorConstantRigCostFunction(
      const Rigid3d& cam_from_rig,
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity())
      : Parent(point2D, point2D_covar), cam_from_rig_(cam_from_rig) {}

  static ceres::CostFunction* Create(
      const Rigid3d& cam_from_rig,
      const Eigen::Vector2d& point2D,
      const Eigen::Matrix2d& point2D_covar = Eigen::Matrix2d::Identity()) {
    return (new ceres::AutoDiffCostFunction<
            RigReprojErrorConstantRigCostFunction<CameraModel, CTYPE>,
            2,
            4,
            3,
            3,
            CameraModel::num_params>(new RigReprojErrorConstantRigCostFunction(
        cam_from_rig, point2D, point2D_covar)));
  }

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
    return Parent::operator()(cam_from_rig_rotation.coeffs().data(),
                              cam_from_rig_translation.data(),
                              rig_from_world_rotation,
                              rig_from_world_translation,
                              point3D,
                              camera_params,
                              residuals);
  }

 private:
  const Rigid3d& cam_from_rig_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `SphereManifold`.
class SampsonErrorCostFunction {
 public:
  SampsonErrorCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<SampsonErrorCostFunction, 1, 4, 3>(
        new SampsonErrorCostFunction(x1, x2)));
  }

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

    // Homogeneous image coordinates.
    const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
    const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
    const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
    const T x2tEx1 = x2_h.transpose() * Ex1;
    residuals[0] = x2tEx1 * x2tEx1 /
                   (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                    Etx2(1) * Etx2(1));

    return true;
  }

 private:
  const double x1_;
  const double y1_;
  const double x2_;
  const double y2_;
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

// 6-DoF error on the absolute pose. The residual is the log of the error pose,
// splitting SE(3) into SO(3) x R^3. The 6x6 covariance matrix is defined in the
// reference frame of the camera. Its first and last three components correspond
// to the rotation and translation errors, respectively.
template <CovarianceType CTYPE = CovarianceType::DENSE>
struct AbsolutePoseErrorCostFunction {
 public:
  AbsolutePoseErrorCostFunction(const Rigid3d& cam_from_world,
                                const EigenMatrix6d& covariance_cam)
      : world_from_cam_(Inverse(cam_from_world)),
        sqrt_info_cam_(SqrtInformation<CTYPE>(covariance_cam)) {
    THROW_CHECK(CheckCovarianceByType<CTYPE>(covariance_cam));
  }

  static ceres::CostFunction* Create(const Rigid3d& cam_from_world,
                                     const EigenMatrix6d& covariance_cam) {
    return (
        new ceres::
            AutoDiffCostFunction<AbsolutePoseErrorCostFunction<CTYPE>, 6, 4, 3>(
                new AbsolutePoseErrorCostFunction<CTYPE>(cam_from_world,
                                                         covariance_cam)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_q,
                  const T* const cam_from_world_t,
                  T* residuals) const {
    const Eigen::Quaternion<T> param_from_measured_q =
        EigenQuaternionMap<T>(cam_from_world_q) *
        world_from_cam_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_t(residuals + 3);
    param_from_measured_t = EigenVector3Map<T>(cam_from_world_t) +
                            EigenQuaternionMap<T>(cam_from_world_q) *
                                world_from_cam_.translation.cast<T>();

    if constexpr (CTYPE != CovarianceType::IDENTITY) {
      ApplySqrtInformation<T, 6, CTYPE>(residuals, sqrt_info_cam_);
    }
    return true;
  }

 private:
  const Rigid3d world_from_cam_;
  const EigenMatrix6d sqrt_info_cam_;
};

// 6-DoF error between two absolute poses based on a measurement that is their
// relative pose, with identical scale for the translation. The covariance is
// defined in the reference frame of the camera j.
// Its first and last three components correspond to the rotation and
// translation errors, respectively.
//
// Derivation:
// j_T_w = ΔT_j·j_T_i·i_T_w
// where ΔT_j = exp(η_j) is the residual in SE(3) and η_j in tangent space.
// Thus η_j = log(j_T_w·i_T_w⁻¹·i_T_j)
// Rotation term: ΔR = log(j_R_w·i_R_w⁻¹·i_R_j)
// Translation term: Δt = j_t_w + j_R_w·i_R_w⁻¹·(i_t_j -i_t_w)
template <CovarianceType CTYPE = CovarianceType::DENSE>
struct MetricRelativePoseErrorCostFunction {
 public:
  MetricRelativePoseErrorCostFunction(const Rigid3d& i_from_j,
                                      const EigenMatrix6d& covariance_j)
      : i_from_j_(i_from_j),
        sqrt_info_j_(SqrtInformation<CTYPE>(covariance_j)) {
    THROW_CHECK(CheckCovarianceByType<CTYPE>(covariance_j));
  }

  static ceres::CostFunction* Create(const Rigid3d& i_from_j,
                                     const EigenMatrix6d& covariance_j) {
    return (new ceres::AutoDiffCostFunction<
            MetricRelativePoseErrorCostFunction<CTYPE>,
            6,
            4,
            3,
            4,
            3>(new MetricRelativePoseErrorCostFunction<CTYPE>(i_from_j,
                                                              covariance_j)));
  }

  template <typename T>
  bool operator()(const T* const i_from_world_q,
                  const T* const i_from_world_t,
                  const T* const j_from_world_q,
                  const T* const j_from_world_t,
                  T* residuals) const {
    const Eigen::Quaternion<T> j_from_i_q =
        EigenQuaternionMap<T>(j_from_world_q) *
        EigenQuaternionMap<T>(i_from_world_q).inverse();
    const Eigen::Quaternion<T> param_from_measured_q =
        j_from_i_q * i_from_j_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals);

    Eigen::Matrix<T, 3, 1> i_from_jw_t =
        i_from_j_.translation.cast<T>() - EigenVector3Map<T>(i_from_world_t);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_t(residuals + 3);
    param_from_measured_t =
        EigenVector3Map<T>(j_from_world_t) + j_from_i_q * i_from_jw_t;

    if constexpr (CTYPE != CovarianceType::IDENTITY) {
      ApplySqrtInformation<T, 6, CTYPE>(residuals, sqrt_info_j_);
    }
    return true;
  }

 private:
  const Rigid3d& i_from_j_;
  const EigenMatrix6d sqrt_info_j_;
};

// Cost function for aligning one 3D point with a reference 3D point with
// covariance. Convention is similar to colmap::Sim3d
// residual = scale_b_from_a * R_b_from_a * point_in_a + t_b_from_a -
// ref_point_in_b
template <CovarianceType CTYPE = CovarianceType::DENSE>
struct Point3dAlignmentCostFunction {
 public:
  Point3dAlignmentCostFunction(const Eigen::Vector3d& ref_point,
                               const Eigen::Matrix3d& covariance_point)
      : ref_point_(ref_point),
        sqrt_info_point3D_(SqrtInformation<CTYPE>(covariance_point)) {
    THROW_CHECK(CheckCovarianceByType<CTYPE>(covariance_point));
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& ref_point,
                                     const Eigen::Matrix3d& covariance_point) {
    return (new ceres::AutoDiffCostFunction<Point3dAlignmentCostFunction<CTYPE>,
                                            3,
                                            3,
                                            4,
                                            3,
                                            1>(
        new Point3dAlignmentCostFunction<CTYPE>(ref_point, covariance_point)));
  }

  template <typename T>
  bool operator()(const T* const point,
                  const T* const transform_q,
                  const T* const transform_t,
                  const T* const scale,
                  T* residuals) const {
    const Eigen::Quaternion<T> T_q = EigenQuaternionMap<T>(transform_q);
    const Eigen::Matrix<T, 3, 1> transform_point =
        T_q * EigenVector3Map<T>(point) * scale[0] +
        EigenVector3Map<T>(transform_t);
    for (size_t i = 0; i < 3; ++i) {
      residuals[i] = transform_point[i] - T(ref_point_[i]);
    }
    if constexpr (CTYPE != CovarianceType::IDENTITY) {
      ApplySqrtInformation<T, 3, CTYPE>(residuals, sqrt_info_point3D_);
    }
    return true;
  }

 private:
  const Eigen::Vector3d ref_point_;
  const Eigen::Matrix3d sqrt_info_point3D_;
};

template <template <typename, CovarianceType> class CostFunction,
          typename... Args>
ceres::CostFunction* CameraCostFunction(const CameraModelId camera_model_id,
                                        Args&&... args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                  \
  case CameraModel::model_id:                                           \
    return CostFunction<CameraModel, CovarianceType::IDENTITY>::Create( \
        std::forward<Args>(args)...);                                   \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

template <template <typename, CovarianceType> class CostFunction,
          typename... Args>
ceres::CostFunction* WeightedCameraCostFunction(
    const CameraModelId camera_model_id,
    const CovarianceType ctype,
    Args&&... args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::model_id:                                               \
    switch (ctype) {                                                        \
      case CovarianceType::IDENTITY:                                        \
        return CostFunction<CameraModel, CovarianceType::IDENTITY>::Create( \
            std::forward<Args>(args)...);                                   \
      case CovarianceType::DIAGONAL:                                        \
        return CostFunction<CameraModel, CovarianceType::DIAGONAL>::Create( \
            std::forward<Args>(args)...);                                   \
      case CovarianceType::DENSE:                                           \
        return CostFunction<CameraModel, CovarianceType::DENSE>::Create(    \
            std::forward<Args>(args)...);                                   \
      default:                                                              \
        throw std::runtime_error("Covariance type unsupported");            \
    }                                                                       \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

}  // namespace colmap
