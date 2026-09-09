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

// Shared machinery for the camera models: model id enum,
// declaration/registration macros, depth guard, and CRTP base classes.
// Per-model structs live in the other sensor/models/ headers.

#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/types.h"

#include <array>
#include <cfloat>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <ceres/jet.h>

namespace colmap {

MAKE_ENUM_CLASS_OVERLOAD_STREAM(CameraModelId,
                                -1,
                                kInvalid,                 // = -1
                                kSimplePinhole,           // = 0
                                kPinhole,                 // = 1
                                kSimpleRadial,            // = 2
                                kRadial,                  // = 3
                                kOpenCV,                  // = 4
                                kOpenCVFisheye,           // = 5
                                kFullOpenCV,              // = 6
                                kFOV,                     // = 7
                                kSimpleRadialFisheye,     // = 8
                                kRadialFisheye,           // = 9
                                kThinPrismFisheye,        // = 10
                                kRadTanThinPrismFisheye,  // = 11
                                kSimpleDivision,          // = 12
                                kDivision,                // = 13
                                kSimpleFisheye,           // = 14
                                kFisheye,                 // = 15
                                kEUCM,                    // = 16
                                kEquirectangular          // = 17
);

// Builds a consecutive parameter index array {Offset, ..., Offset + N - 1}.
// Camera model parameters are stored consecutively by group (focal length,
// principal point, extra), so the group index arrays are derived from the
// group sizes passed to the model definition macros below.
namespace internal {
template <size_t N, size_t Offset>
constexpr std::array<size_t, N> IotaArray() {
  std::array<size_t, N> idxs{};
  for (size_t i = 0; i < N; ++i) {
    idxs[i] = Offset + i;
  }
  return idxs;
}
}  // namespace internal

// Definitions shared by all camera models (perspective and spherical alike).
#ifndef CAMERA_MODEL_SHARED_DEFINITIONS
#define CAMERA_MODEL_SHARED_DEFINITIONS(model_id_val,                  \
                                        model_name_val,                \
                                        params_info_val,               \
                                        num_params_val,                \
                                        has_img_from_cam_with_jac_val) \
  static constexpr size_t num_params = num_params_val;                 \
  static constexpr bool has_img_from_cam_with_jac =                    \
      has_img_from_cam_with_jac_val;                                   \
  static constexpr CameraModelId model_id = model_id_val;              \
  static inline const std::string model_name = model_name_val;         \
  static inline const std::string params_info = params_info_val;       \
  template <bool Enable = has_img_from_cam_with_jac,                   \
            typename std::enable_if<Enable, int>::type = 0>            \
  static inline bool ImgFromCamWithJac(const double* params,           \
                                       const double& u,                \
                                       const double& v,                \
                                       const double& w,                \
                                       double* x,                      \
                                       double* y,                      \
                                       double* J_params,               \
                                       double* J_uvw,                  \
                                       bool check_cheirality = true);
#endif

// Parameter groups specific to perspective camera models: focal length,
// principal point, and extra (distortion) parameters.
#ifndef PERSPECTIVE_CAMERA_MODEL_PARAM_DEFINITIONS
#define PERSPECTIVE_CAMERA_MODEL_PARAM_DEFINITIONS(                           \
    num_focal_params_val, num_pp_params_val, num_extra_params_val)            \
  static constexpr size_t num_focal_params = num_focal_params_val;            \
  static constexpr size_t num_pp_params = num_pp_params_val;                  \
  static constexpr size_t num_extra_params = num_extra_params_val;            \
  static constexpr std::array<size_t, (num_focal_params_val)>                 \
      focal_length_idxs = internal::IotaArray<(num_focal_params_val), 0>();   \
  static constexpr std::array<size_t, (num_pp_params_val)>                    \
      principal_point_idxs =                                                  \
          internal::IotaArray<(num_pp_params_val), (num_focal_params_val)>(); \
  static constexpr std::array<size_t, (num_extra_params_val)>                 \
      extra_params_idxs =                                                     \
          internal::IotaArray<(num_extra_params_val),                         \
                              (num_focal_params_val) + (num_pp_params_val)>();
#endif

// Parameter group specific to spherical (omnidirectional) camera models: the
// metadata parameters (e.g., image dimensions).
#ifndef SPHERICAL_CAMERA_MODEL_PARAM_DEFINITIONS
#define SPHERICAL_CAMERA_MODEL_PARAM_DEFINITIONS(num_metadata_params_val) \
  static constexpr size_t num_metadata_params = num_metadata_params_val;  \
  static constexpr std::array<size_t, (num_metadata_params_val)>          \
      metadata_idxs = internal::IotaArray<(num_metadata_params_val), 0>();
#endif

// Convenience composition macros used in the model class declarations.
#ifndef PERSPECTIVE_CAMERA_MODEL_DEFINITIONS
#define PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(model_id_val,                   \
                                             model_name_val,                 \
                                             params_info_val,                \
                                             num_focal_params_val,           \
                                             num_pp_params_val,              \
                                             num_extra_params_val,           \
                                             has_img_from_cam_with_jac_val)  \
  CAMERA_MODEL_SHARED_DEFINITIONS(                                           \
      model_id_val,                                                          \
      model_name_val,                                                        \
      params_info_val,                                                       \
      (num_focal_params_val) + (num_pp_params_val) + (num_extra_params_val), \
      has_img_from_cam_with_jac_val)                                         \
  PERSPECTIVE_CAMERA_MODEL_PARAM_DEFINITIONS(                                \
      num_focal_params_val, num_pp_params_val, num_extra_params_val)
#endif

#ifndef SPHERICAL_CAMERA_MODEL_DEFINITIONS
#define SPHERICAL_CAMERA_MODEL_DEFINITIONS(model_id_val,                  \
                                           model_name_val,                \
                                           params_info_val,               \
                                           num_metadata_params_val,       \
                                           has_img_from_cam_with_jac_val) \
  CAMERA_MODEL_SHARED_DEFINITIONS(model_id_val,                           \
                                  model_name_val,                         \
                                  params_info_val,                        \
                                  num_metadata_params_val,                \
                                  has_img_from_cam_with_jac_val)          \
  SPHERICAL_CAMERA_MODEL_PARAM_DEFINITIONS(num_metadata_params_val)
#endif

#ifndef PERSPECTIVE_CAMERA_MODEL_CASES
#define PERSPECTIVE_CAMERA_MODEL_CASES              \
  CAMERA_MODEL_CASE(SimplePinholeCameraModel)       \
  CAMERA_MODEL_CASE(PinholeCameraModel)             \
  CAMERA_MODEL_CASE(SimpleRadialCameraModel)        \
  CAMERA_MODEL_CASE(SimpleRadialFisheyeCameraModel) \
  CAMERA_MODEL_CASE(RadialCameraModel)              \
  CAMERA_MODEL_CASE(RadialFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(OpenCVCameraModel)              \
  CAMERA_MODEL_CASE(OpenCVFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(FullOpenCVCameraModel)          \
  CAMERA_MODEL_CASE(FOVCameraModel)                 \
  CAMERA_MODEL_CASE(ThinPrismFisheyeCameraModel)    \
  CAMERA_MODEL_CASE(RadTanThinPrismFisheyeModel)    \
  CAMERA_MODEL_CASE(SimpleDivisionCameraModel)      \
  CAMERA_MODEL_CASE(DivisionCameraModel)            \
  CAMERA_MODEL_CASE(SimpleFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(FisheyeCameraModel)             \
  CAMERA_MODEL_CASE(EUCMCameraModel)
#endif

#ifndef SPHERICAL_CAMERA_MODEL_CASES
#define SPHERICAL_CAMERA_MODEL_CASES \
  CAMERA_MODEL_CASE(EquirectangularCameraModel)
#endif

#ifndef CAMERA_MODEL_CASES
#define CAMERA_MODEL_CASES       \
  PERSPECTIVE_CAMERA_MODEL_CASES \
  SPHERICAL_CAMERA_MODEL_CASES
#endif

#ifndef CAMERA_MODEL_SWITCH_CASES
#define CAMERA_MODEL_SWITCH_CASES         \
  CAMERA_MODEL_CASES                      \
  default:                                \
    CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
    break;
#endif

#define CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
  throw std::domain_error("Camera model does not exist");

// Fisheye camera model macros
#ifndef PERSPECTIVE_FISHEYE_CAMERA_MODEL_CASES
#define PERSPECTIVE_FISHEYE_CAMERA_MODEL_CASES      \
  CAMERA_MODEL_CASE(SimpleRadialFisheyeCameraModel) \
  CAMERA_MODEL_CASE(RadialFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(OpenCVFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(ThinPrismFisheyeCameraModel)    \
  CAMERA_MODEL_CASE(RadTanThinPrismFisheyeModel)    \
  CAMERA_MODEL_CASE(SimpleFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(FisheyeCameraModel)
#endif

// Depth guard shared by the models' `ImgFromCam`. Rejects points at or behind
// the camera plane if `check_cheirality`, otherwise only points on the plane,
// where the projection diverges.
template <typename T>
inline bool HasProjectableDepth(const T& w, const bool check_cheirality) {
  return check_cheirality ? w >= std::numeric_limits<T>::epsilon()
                          : ceres::abs(w) >= std::numeric_limits<T>::epsilon();
}

// The "Curiously Recurring Template Pattern" (CRTP) is used throughout the
// camera model hierarchy so that shared functionality can be reused across
// models. The hierarchy is:
//
//   BaseCameraModel                            (shared by all camera models)
//     - BasePerspectiveCameraModel             (focal length, image plane)
//         - BasePerspectivePinholeCameraModel  (pinhole projection)
//             - Pinhole models
//         - BasePerspectiveFisheyeCameraModel  (fisheye projection)
//             - Fisheye models
//     - BaseSphericalCameraModel               (spherical/omnidirectional)
//        - EquirectangularCameraModel
//
// Whether a model is perspective, and whether its projection is pinhole or
// fisheye, is derived from its position in this hierarchy (see
// CameraModelIsPerspective, CameraModelIsPerspectivePinhole and
// CameraModelIsPerspectiveFisheye), rather than from a separate flag.
template <typename CameraModel>
struct BaseCameraModel {
 private:
  BaseCameraModel() = default;
  friend CameraModel;
};

// Base model for perspective camera models, i.e. models with a finite pinhole
// image plane and a focal length. Provides the shared focal-length / principal-
// point validity checks, the focal-length-based pixel threshold conversion,
// iterative undistortion, and the default forward-hemisphere ray unprojection.
template <typename CameraModel>
struct BasePerspectiveCameraModel : public BaseCameraModel<CameraModel> {
  template <typename T>
  static inline bool HasBogusParams(const std::vector<T>& params,
                                    const size_t width,
                                    const size_t height,
                                    const T min_focal_length_ratio,
                                    const T max_focal_length_ratio,
                                    const T max_extra_param) {
    return CameraModel::HasBogusPrincipalPoint(params, width, height) ||
           CameraModel::HasBogusFocalLength(params,
                                            width,
                                            height,
                                            min_focal_length_ratio,
                                            max_focal_length_ratio) ||
           CameraModel::HasBogusExtraParams(params, max_extra_param);
  }

  template <typename T>
  static inline bool HasBogusFocalLength(const std::vector<T>& params,
                                         const size_t width,
                                         const size_t height,
                                         const T min_focal_length_ratio,
                                         const T max_focal_length_ratio) {
    const T inv_max_size = 1.0 / std::max(width, height);
    for (const size_t idx : CameraModel::focal_length_idxs) {
      const T focal_length_ratio = params[idx] * inv_max_size;
      if (focal_length_ratio < min_focal_length_ratio ||
          focal_length_ratio > max_focal_length_ratio) {
        return true;
      }
    }

    return false;
  }

  template <typename T>
  static inline bool HasBogusPrincipalPoint(const std::vector<T>& params,
                                            const size_t width,
                                            const size_t height) {
    const T cx = params[CameraModel::principal_point_idxs[0]];
    const T cy = params[CameraModel::principal_point_idxs[1]];
    return cx < 0 || cx > width || cy < 0 || cy > height;
  }

  template <typename T>
  static inline bool HasBogusExtraParams(const std::vector<T>& params,
                                         const T max_extra_param) {
    for (const size_t idx : CameraModel::extra_params_idxs) {
      if (std::abs(params[idx]) > max_extra_param) {
        return true;
      }
    }

    return false;
  }

  template <typename T>
  static inline T CamFromImgThreshold(const T* params, const T threshold) {
    T mean_focal_length = 0;
    for (const size_t idx : CameraModel::focal_length_idxs) {
      mean_focal_length += params[idx];
    }
    mean_focal_length /= CameraModel::focal_length_idxs.size();
    return threshold / mean_focal_length;
  }

  static inline bool IterativeUndistortion(const double* params,
                                           double* u,
                                           double* v) {
    // Parameters for Newton iteration. 100 iterations should be enough for
    // complex camera models with higher order terms.
    constexpr size_t kNumIterations = 100;
    constexpr double kMinStepSquaredNorm = 1e-10;
    // Trust region: step_x.norm() <= max(x.norm() * kRelStepRadius,
    // kStepRadius)
    constexpr double kRelStepRadius = 0.1;
    constexpr double kStepRadius = 0.1;

    Eigen::Matrix2d J;
    const Eigen::Vector2d x0(*u, *v);
    Eigen::Vector2d x(*u, *v);
    Eigen::Vector2d dx;

    ceres::Jet<double, 2> params_jet[CameraModel::num_extra_params];
    for (size_t i = 0; i < CameraModel::num_extra_params; ++i) {
      params_jet[i] = ceres::Jet<double, 2>(params[i]);
    }
    for (size_t i = 0; i < kNumIterations; ++i) {
      // Get Jacobian
      ceres::Jet<double, 2> x_jet[2];
      x_jet[0] = ceres::Jet<double, 2>(x(0), 0);
      x_jet[1] = ceres::Jet<double, 2>(x(1), 1);
      ceres::Jet<double, 2> dx_jet[2];
      CameraModel::Distortion(
          params_jet, x_jet[0], x_jet[1], &dx_jet[0], &dx_jet[1]);
      dx[0] = dx_jet[0].a;
      dx[1] = dx_jet[1].a;
      J(0, 0) = dx_jet[0].v[0] + 1;
      J(0, 1) = dx_jet[0].v[1];
      J(1, 0) = dx_jet[1].v[0];
      J(1, 1) = dx_jet[1].v[1] + 1;

      // Update
      Eigen::Vector2d step_x = J.partialPivLu().solve(x + dx - x0);
      const double radius_sqr =
          std::max(x.squaredNorm() * kRelStepRadius * kRelStepRadius,
                   kStepRadius * kStepRadius);
      const double step_x_norm_sqr = step_x.squaredNorm();
      if (step_x_norm_sqr > radius_sqr) {
        step_x *= std::sqrt(radius_sqr / step_x_norm_sqr);
      }
      x -= step_x;
      if (step_x.squaredNorm() < kMinStepSquaredNorm) {
        *u = x(0);
        *v = x(1);
        return true;
      }
    }

    *u = x(0);
    *v = x(1);

    return false;
  }

  // Unproject a pixel to a unit bearing vector in the camera frame.
  //
  // Default implementation: delegates to CameraModel::CamFromImg and
  // normalizes the resulting homogeneous coordinate. Correct for perspective
  // and fisheye-with-FOV<=180° cameras — the returned ray always has rz > 0.
  static inline bool CamRayFromImg(const double* params,
                                   double x,
                                   double y,
                                   double* rx,
                                   double* ry,
                                   double* rz) {
    double u = 0;
    double v = 0;
    if (!CameraModel::CamFromImg(params, x, y, &u, &v)) {
      return false;
    }
    const double norm = std::sqrt(u * u + v * v + 1.0);
    *rx = u / norm;
    *ry = v / norm;
    *rz = 1.0 / norm;
    return true;
  }

  // Rescale the parameters in-place for a new image resolution, given the
  // per-axis scale factors. A single shared focal length scales by the mean
  // factor; separate fx/fy scale independently. The principal point follows
  // the image dimensions. Extra (distortion) parameters are resolution
  // independent and left untouched.
  static inline void Rescale(double scale_x,
                             double scale_y,
                             std::vector<double>* params) {
    if constexpr (CameraModel::num_focal_params == 1) {
      (*params)[CameraModel::focal_length_idxs[0]] *= 0.5 * (scale_x + scale_y);
    } else {
      (*params)[CameraModel::focal_length_idxs[0]] *= scale_x;
      (*params)[CameraModel::focal_length_idxs[1]] *= scale_y;
    }
    (*params)[CameraModel::principal_point_idxs[0]] *= scale_x;
    (*params)[CameraModel::principal_point_idxs[1]] *= scale_y;
  }

 private:
  BasePerspectiveCameraModel() = default;
  friend CameraModel;
};

template <typename CameraModel>
struct BaseSphericalCameraModel : public BaseCameraModel<CameraModel> {
  // Rescale the parameters in-place for a new image resolution. Only the image
  // dimensions (w, h), carried by the metadata group, track the rescaled image;
  // any extra parameters are resolution independent and left untouched.
  static inline void Rescale(double scale_x,
                             double scale_y,
                             std::vector<double>* params) {
    (*params)[CameraModel::metadata_idxs[0]] *= scale_x;
    (*params)[CameraModel::metadata_idxs[1]] *= scale_y;
  }

 private:
  BaseSphericalCameraModel() = default;
  friend CameraModel;
};

// Base model for perspective pinhole camera models, i.e. models that project
// onto the normalized plane as x = X / Z and then apply a deformation of that
// plane. Their calibration therefore acts projectively on the rays, so a
// calibration matrix K describes the projection exactly in the zero-distortion
// limit and remains the exact linearization at the optical axis otherwise.
template <typename CameraModel>
struct BasePerspectivePinholeCameraModel
    : public BasePerspectiveCameraModel<CameraModel> {
 private:
  BasePerspectivePinholeCameraModel() = default;
  friend CameraModel;
};

// Base model for perspective fisheye camera models.
template <typename CameraModel>
struct BasePerspectiveFisheyeCameraModel
    : public BasePerspectiveCameraModel<CameraModel> {
  template <typename T>
  static inline void FisheyeFromNormal(const T& u, const T& v, T* uu, T* vv) {
    *uu = u;
    *vv = v;
    const T r = ceres::sqrt(u * u + v * v);
    if (r > T(std::numeric_limits<double>::epsilon())) {
      const T theta = ceres::atan(r);
      *uu *= theta / r;
      *vv *= theta / r;
    }
  }

  template <typename T>
  static inline void NormalFromFisheye(const T uu, const T vv, T* u, T* v) {
    *u = uu;
    *v = vv;
    const T theta = ceres::sqrt(uu * uu + vv * vv);
    const T theta_cos_theta = theta * ceres::cos(theta);
    if (theta_cos_theta > T(std::numeric_limits<double>::epsilon())) {
      const T scale = ceres::sin(theta) / theta_cos_theta;
      *u *= scale;
      *v *= scale;
    }
  }

 private:
  BasePerspectiveFisheyeCameraModel() = default;
  friend CameraModel;
};

}  // namespace colmap
