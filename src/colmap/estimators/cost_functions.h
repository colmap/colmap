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

template <typename CostFunctor, int kNumResiduals, int... kParameterDims>
ceres::CostFunction* CreateAutoDiffCostFunction(
    CostFunctor* functor, std::integer_sequence<int, kParameterDims...>) {
  return new ceres::AutoDiffCostFunction<CostFunctor,
                                         kNumResiduals,
                                         kParameterDims...>(functor);
}

template <typename CostFunctor>
ceres::CostFunction* CreateAutoDiffCostFunction(CostFunctor* functor) {
  return CreateAutoDiffCostFunction<CostFunctor, CostFunctor::kNumResiduals>(
      functor, typename CostFunctor::kParameterDims{});
}

template <class DerivedCostFunctor, int NumResiduals, int... ParamDims>
class AutoDiffCostFunctor {
 public:
  static constexpr int kNumResiduals = NumResiduals;
  using kParameterDims = std::integer_sequence<int, ParamDims...>;

  template <typename... Args>
  static ceres::CostFunction* Create(Args&&... args) {
    return CreateAutoDiffCostFunction<DerivedCostFunctor>(
        new DerivedCostFunctor(std::forward<Args>(args)...));
  }
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
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

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
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
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
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);
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
  SampsonErrorCostFunctor(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

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

// 6-DoF error on the absolute camera pose. The residual is the log of the error
// pose, splitting SE(3) into SO(3) x R^3. The residual is computed in the
// camera frame. Its first and last three components correspond to the rotation
// and translation errors, respectively.
struct AbsolutePosePriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePriorCostFunctor, 6, 4, 3> {
 public:
  explicit AbsolutePosePriorCostFunctor(const Rigid3d& cam_from_world_prior)
      : world_from_cam_prior_(Inverse(cam_from_world_prior)) {}

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_prior_rotation =
        EigenQuaternionMap<T>(cam_from_world_rotation) *
        world_from_cam_prior_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(cam_from_world_translation) +
        EigenQuaternionMap<T>(cam_from_world_rotation) *
            world_from_cam_prior_.translation.cast<T>();

    return true;
  }

 private:
  const Rigid3d world_from_cam_prior_;
};

// 3-DoF error on the camera position in the world coordinate frame.
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
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = position_in_world_prior_.cast<T>() +
                EigenQuaternionMap<T>(cam_from_world_rotation).inverse() *
                    EigenVector3Map<T>(cam_from_world_translation);
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
  explicit Point3DAlignmentCostFunctor(const Eigen::Vector3d& point_in_b_prior)
      : point_in_b_prior_(point_in_b_prior) {}

  template <typename T>
  bool operator()(const T* const point_in_a,
                  const T* const b_from_a_rotation,
                  const T* const b_from_a_translation,
                  const T* const b_from_a_scale,
                  T* residuals_ptr) const {
    const Eigen::Matrix<T, 3, 1> point_in_b =
        EigenQuaternionMap<T>(b_from_a_rotation) *
            EigenVector3Map<T>(point_in_a) * b_from_a_scale[0] +
        EigenVector3Map<T>(b_from_a_translation);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = point_in_b - point_in_b_prior_.cast<T>();
    return true;
  }

 private:
  const Eigen::Vector3d point_in_b_prior_;
};

template <typename... Args>
auto LastValueParameterPack(Args&&... args) {
  return std::get<sizeof...(Args) - 1>(std::forward_as_tuple(args...));
}

// A cost function that wraps another one and whitens its residuals with a given
// covariance. For example, to weight the reprojection error with a image
// measurement covariance, one can wrap it as:
//
//    using ReprojCostFunctor = ReprojErrorCostFunctor<PinholeCameraModel>;
//    ceres::CostFunction* cost_function =
//        CovarianceWeightedCostFunctor<ReprojCostFunctor>::Create(
//            point2D_cov, point2D));
template <class CostFunctor>
class CovarianceWeightedCostFunctor {
 public:
  static constexpr int kNumResiduals = CostFunctor::kNumResiduals;
  using kParameterDims = typename CostFunctor::kParameterDims;

  // Covariance or sqrt information matrix type.
  using CovMat = Eigen::Matrix<double, kNumResiduals, kNumResiduals>;

  template <typename... Args>
  explicit CovarianceWeightedCostFunctor(const CovMat& cov, Args&&... args)
      : left_sqrt_info_(LeftSqrtInformation(cov)),
        cost_(std::forward<Args>(args)...) {}

  template <typename... Args>
  static ceres::CostFunction* Create(const CovMat& cov, Args&&... args) {
    return CreateAutoDiffCostFunction(
        new CovarianceWeightedCostFunctor<CostFunctor>(
            cov, std::forward<Args>(args)...));
  }

  template <typename... Args>
  bool operator()(Args... args) const {
    if (!cost_(args...)) {
      return false;
    }

    auto residuals_ptr = LastValueParameterPack(args...);
    typedef typename std::remove_reference<decltype(*residuals_ptr)>::type T;
    Eigen::Map<Eigen::Matrix<T, kNumResiduals, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(left_sqrt_info_.template cast<T>());
    return true;
  }

 private:
  CovMat LeftSqrtInformation(const CovMat& cov) {
    return cov.inverse().llt().matrixL().transpose();
  }

  const CovMat left_sqrt_info_;
  const CostFunctor cost_;
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

}  // namespace colmap
