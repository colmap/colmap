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

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/conditioned_cost_function.h>
#include <ceres/rotation.h>

namespace colmap {

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

// Cost function for constraining the depth of a point in the camera frame 
// with depth priors.
struct ScaledDepthErrorCostFunction {
  public:
   ScaledDepthErrorCostFunction(const double depth) : depth_(depth) {}
   static ceres::CostFunction* Create(const double depth) {
     return (
         new ceres::
             AutoDiffCostFunction<ScaledDepthErrorCostFunction, 1, 4, 3, 3, 2>(
                 new ScaledDepthErrorCostFunction(depth)));
   }
   template <typename T>
   bool operator()(const T* const cam_from_world_rotation,
                   const T* const cam_from_world_translation,
                   const T* const point3D,
                   const T* const shift_scale,
                   T* residuals) const {
     *residuals = (EigenQuaternionMap<T>(cam_from_world_rotation) *
                   EigenVector3Map<T>(point3D))[2] +
                  cam_from_world_translation[2] - shift_scale[0] -
                  T(depth_) * exp(shift_scale[1]);
     return true;
   }
  private:
   const double depth_;
 };
 
// Cost function for constraining the depth of a point in the camera frame 
// with depth priors, with a fixed camera pose.
 struct ScaledDepthErrorConstantPoseCostFunction
     : public ScaledDepthErrorCostFunction {
   using Parent = ScaledDepthErrorCostFunction;
  public:
   ScaledDepthErrorConstantPoseCostFunction(const Rigid3d& cam_from_world,
                                            const double depth)
       : Parent(depth), cam_from_world_(cam_from_world) {}
   static ceres::CostFunction* Create(const Rigid3d& cam_from_world,
                                      const double depth) {
     return (new ceres::AutoDiffCostFunction<
             ScaledDepthErrorConstantPoseCostFunction,
             1,
             3,
             2>(
         new ScaledDepthErrorConstantPoseCostFunction(cam_from_world, depth)));
   }
   template <typename T>
   bool operator()(const T* const point3D,
                   const T* const shift_scale,
                   T* residuals) const {
     const Eigen::Quaternion<T> cam_from_world_rotation =
         cam_from_world_.rotation.cast<T>();
     const Eigen::Matrix<T, 3, 1> cam_from_world_translation =
         cam_from_world_.translation.cast<T>();
     return Parent::operator()(cam_from_world_rotation.coeffs().data(),
                               cam_from_world_translation.data(),
                               point3D,
                               shift_scale,
                               residuals);
   }
  private:
   const Rigid3d& cam_from_world_;
 };

 // Cost function for constraining the depth of a point in the camera frame
 // with depth priors, with a fixed camera pose, in log space.
 struct LogScaledDepthErrorCostFunction {
  public:
   LogScaledDepthErrorCostFunction(const double depth) : depth_(depth) {}
 
   static ceres::CostFunction* Create(const double depth) {
     return new ceres::AutoDiffCostFunction<LogScaledDepthErrorCostFunction, 1, 4, 3, 3, 2>(
         new LogScaledDepthErrorCostFunction(depth));
   }
 
   template <typename T>
   bool operator()(const T* const cam_from_world_rotation,
                   const T* const cam_from_world_translation,
                   const T* const point3D,
                   const T* const shift_scale,
                   T* residuals) const {
     // Compute the predicted depth in the camera frame.
     T d_pred = (EigenQuaternionMap<T>(cam_from_world_rotation) *
                 EigenVector3Map<T>(point3D))[2] +
                cam_from_world_translation[2];
 
     if (d_pred <= T(0)) {
       *residuals = T(0);
       return true;
     }
 
     *residuals = ceres::log(d_pred) - (ceres::log(T(depth_)) + shift_scale[1]);
     return true;
   }
 
  private:
   const double depth_;
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
