// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Runtime dispatch over CameraModelId: free-function API declared
// here and defined inline below. Needs all model structs, so this
// header aggregates the per-family headers; `models.h` adds the
// analytic Jacobians on top.

#include "colmap/sensor/models/base.h"
#include "colmap/sensor/models/division.h"
#include "colmap/sensor/models/eucm.h"
#include "colmap/sensor/models/fisheye.h"
#include "colmap/sensor/models/fisheye_radial.h"
#include "colmap/sensor/models/fov.h"
#include "colmap/sensor/models/opencv.h"
#include "colmap/sensor/models/pinhole.h"
#include "colmap/sensor/models/radial.h"
#include "colmap/sensor/models/spherical.h"
#include "colmap/sensor/models/thin_prism.h"

namespace colmap {

// Check whether camera model with given name or identifier exists.
bool ExistsCameraModelWithName(const std::string& model_name);
bool ExistsCameraModelWithId(CameraModelId model_id);

// Convert camera name to unique camera model identifier.
//
// @param name         Unique name of camera model.
//
// @return             Unique identifier of camera model.
CameraModelId CameraModelNameToId(const std::string& model_name);

// Convert camera model identifier to unique camera model name.
//
// @param model_id     Unique identifier of camera model.
//
// @return             Unique name of camera model.
const std::string& CameraModelIdToName(CameraModelId model_id);

// Initialize camera parameters using given image properties.
//
// Initializes all focal length parameters to the same given focal length and
// sets the principal point to the image center.
//
// @param model_id      Unique identifier of camera model.
// @param focal_length  Focal length, equal for all focal length parameters.
// @param width         Sensor width of the camera.
// @param height        Sensor height of the camera.
std::vector<double> CameraModelInitializeParams(CameraModelId model_id,
                                                double focal_length,
                                                size_t width,
                                                size_t height);

// Get human-readable information about the parameter vector order.
//
// @param model_id     Unique identifier of camera model.
const std::string& CameraModelParamsInfo(CameraModelId model_id);

// Get the indices of the parameter groups in the parameter vector.
//
// @param model_id     Unique identifier of camera model.
span<const size_t> CameraModelFocalLengthIdxs(CameraModelId model_id);
span<const size_t> CameraModelPrincipalPointIdxs(CameraModelId model_id);
span<const size_t> CameraModelExtraParamsIdxs(CameraModelId model_id);
span<const size_t> CameraModelMetaDataParamsIdxs(CameraModelId model_id);

// Get the total number of parameters of a camera model.
size_t CameraModelNumParams(CameraModelId model_id);

// Rescale the camera parameters in-place for a new image resolution, given the
// per-axis scale factors (new_dim / old_dim). Each camera model rescales the
// parameters it owns: focal length and principal point for perspective models,
// the image dimensions for spherical models.
//
// @param model_id     Unique identifier of camera model.
// @param scale_x      Horizontal scale factor (new_width / old_width).
// @param scale_y      Vertical scale factor (new_height / old_height).
// @param params       Array of camera parameters, modified in place.
inline void CameraModelRescale(CameraModelId model_id,
                               double scale_x,
                               double scale_y,
                               std::vector<double>& params);

// Check whether parameters are valid, i.e. the parameter vector has
// the correct dimensions that match the specified camera model.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
bool CameraModelVerifyParams(CameraModelId model_id,
                             const std::vector<double>& params);

// Check whether camera has bogus parameters.
//
// @param model_id                Unique identifier of camera model.
// @param params                  Array of camera parameters.
// @param width                   Sensor width of the camera.
// @param height                  Sensor height of the camera.
// @param min_focal_length_ratio  Minimum ratio of focal length over
//                                maximum sensor dimension.
// @param max_focal_length_ratio  Maximum ratio of focal length over
//                                maximum sensor dimension.
// @param max_extra_param         Maximum magnitude of each extra parameter.
bool CameraModelHasBogusParams(CameraModelId model_id,
                               const std::vector<double>& params,
                               size_t width,
                               size_t height,
                               double min_focal_length_ratio,
                               double max_focal_length_ratio,
                               double max_extra_param);

// Transform camera to image coordinates.
//
// This is the inverse of `CameraModelCamFromImg`.
//
// @param model_id     Unique identifier of camera model.
// @param params       Array of camera parameters.
// @param uvw          Coordinates in camera system as (u, v, w).
//
// @return             Image coordinates in pixels, or std::nullopt on failure.
inline std::optional<Eigen::Vector2d> CameraModelImgFromCam(
    CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector3d& uvw,
    bool check_cheirality = true);

// Transform camera to image coordinates, additionally computing the Jacobian
// of the projection with respect to the camera ray.
//
// Runtime dispatch over the analytic per-model `ImgFromCamWithJac`.
//
// @param model_id     Unique identifier of camera model.
// @param params       Array of camera parameters.
// @param uvw          Coordinates in camera system as (u, v, w).
// @param J_uvw        Output Jacobian d(x, y) / d(u, v, w). May be nullptr, in
//                     which case the Jacobian is not computed.
//
// @return             Image coordinates in pixels, or std::nullopt on failure.
inline std::optional<Eigen::Vector2d> CameraModelImgFromCamWithJac(
    CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector3d& uvw,
    Eigen::Matrix2x3d* J_uvw,
    bool check_cheirality = true);

// The Jacobian of `CameraModelCamRayFromImg`, i.e. d(u, v, w) / d(x, y),
// obtained by inverting the projection Jacobian d(x, y) / d(u, v, w) at a unit
// bearing vector.
//
// Central projection depends only on the direction of the ray, so the ray lies
// in the null space of `J_uvw` and `J_uvw` has rank 2. For a *unit* ray its
// Moore-Penrose pseudo-inverse is exactly the Jacobian of the normalized
// unprojection, and its range is the tangent plane of the unit sphere at the
// ray. No explicit tangent basis is therefore required.
//
// Uses the closed form of Terekhov and Larsson, "Tangent Sampson Error", ICCV
// 2023, Lemma 1:
//
//     J_uvw^+ = 1 / (d . (g_x x g_y)) * [ (g_y x d), (d x g_x) ]
//
// where g_x and g_y are the rows of `J_uvw`. This is cheaper than forming
// J^T (J J^T)^-1 and exposes the rank condition directly as the scalar triple
// product in the denominator.
//
// @param cam_ray      Unit bearing vector at which `J_uvw` was evaluated.
// @param J_uvw        Jacobian d(x, y) / d(u, v, w).
//
// @return             Jacobian d(u, v, w) / d(x, y), or std::nullopt if
//                     `J_uvw` is rank deficient.
inline std::optional<Eigen::Matrix3x2d> CamRayFromImgJac(
    const Eigen::Vector3d& cam_ray, const Eigen::Matrix2x3d& J_uvw);

// Transform image to camera coordinates.
//
// This is the inverse of `CameraModelImgFromCam`.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param xy            Image coordinates in pixels.
//
// @return              Output ray in camera frame (u, v, w).
inline std::optional<Eigen::Vector2d> CameraModelCamFromImg(
    CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& xy);

// Unproject a pixel to a unit 3D bearing vector in the camera frame.
//
// Unlike `CameraModelCamFromImg` (limited to the forward hemisphere via the
// 2D normalized-coordinate representation), this function returns a valid
// bearing for any pixel the camera model can unproject — including back-
// facing rays for omnidirectional cameras.
//
// Prefer this over `CameraModelCamFromImg` followed by homogeneous +
// normalize when downstream code needs a 3D ray.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param xy            Image coordinates in pixels.
//
// @return              Unit bearing vector in camera frame, or std::nullopt
//                      if unprojection fails.
inline std::optional<Eigen::Vector3d> CameraModelCamRayFromImg(
    CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& xy);

// Convert pixel threshold in image plane to camera space by dividing
// the threshold through the mean focal length.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param threshold     Image space threshold in pixels.
//
// @return              Camera space threshold.
inline double CameraModelCamFromImgThreshold(CameraModelId model_id,
                                             const std::vector<double>& params,
                                             double threshold);

// Test if a camera model is a perspective fisheye camera model, i.e. derives
// from BasePerspectiveFisheyeCameraModel.
//
// @param model_id      Unique identifier of camera model.
//
// @return              Whether it is a perspective fisheye camera model.
inline bool CameraModelIsPerspectiveFisheye(CameraModelId model_id);

// Test if a camera model is perspective, i.e. has a focal length and a finite
// pinhole image plane. Omnidirectional models such as EQUIRECTANGULAR are not.
//
// @param model_id      Unique identifier of camera model.
//
// @return              Whether it is a perspective camera model.
inline bool CameraModelIsPerspective(CameraModelId model_id);

// Test if a camera model is a perspective pinhole camera model, i.e. derives
// from BasePerspectivePinholeCameraModel. Such models project as x = X / Z and
// then deform the normalized plane, so their calibration acts projectively on
// the rays and a calibration matrix K is meaningful for them.
//
// @param model_id      Unique identifier of camera model.
//
// @return              Whether it is a perspective pinhole camera model.
inline bool CameraModelIsPerspectivePinhole(CameraModelId model_id);

// Test if a camera model represents a spherical (equirectangular
// omnidirectional panorama) camera.
//
// @param model_id      Unique identifier of camera model.
//
// @return              Whether it is a spherical camera model.
inline bool CameraModelIsSpherical(CameraModelId model_id);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

std::optional<Eigen::Vector2d> CameraModelImgFromCam(
    const CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector3d& uvw,
    const bool check_cheirality) {
  Eigen::Vector2d xy;
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)               \
  case CameraModel::model_id:                        \
    if (CameraModel::ImgFromCam(params.data(),       \
                                uvw.x(),             \
                                uvw.y(),             \
                                uvw.z(),             \
                                &xy.x(),             \
                                &xy.y(),             \
                                check_cheirality)) { \
      return xy;                                     \
    }                                                \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  return std::nullopt;
}

std::optional<Eigen::Vector2d> CameraModelImgFromCamWithJac(
    const CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector3d& uvw,
    Eigen::Matrix2x3d* J_uvw,
    const bool check_cheirality) {
  Eigen::Vector2d xy;
  // 2x3 row-major Jacobian. Zero-init so a kernel that skips an entry can't
  // leak an uninitialized read through the Map below.
  double J_uvw_data[6] = {};
  double* J_uvw_ptr = (J_uvw == nullptr) ? nullptr : J_uvw_data;
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::model_id:                                               \
    static_assert(CameraModel::has_img_from_cam_with_jac,                   \
                  #CameraModel                                              \
                  " does not provide an analytic "                          \
                  "ImgFromCamWithJac, which this dispatch "                 \
                  "requires. Implement it in "                              \
                  "models/jacobian.h.");                                    \
    if (CameraModel::ImgFromCamWithJac(params.data(),                       \
                                       uvw.x(),                             \
                                       uvw.y(),                             \
                                       uvw.z(),                             \
                                       &xy.x(),                             \
                                       &xy.y(),                             \
                                       /*J_params=*/nullptr,                \
                                       J_uvw_ptr,                           \
                                       check_cheirality)) {                 \
      if (J_uvw != nullptr) {                                               \
        *J_uvw =                                                            \
            Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>( \
                J_uvw_data);                                                \
      }                                                                     \
      return xy;                                                            \
    }                                                                       \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  return std::nullopt;
}

std::optional<Eigen::Matrix3x2d> CamRayFromImgJac(
    const Eigen::Vector3d& cam_ray, const Eigen::Matrix2x3d& J_uvw) {
  const Eigen::Vector3d g_x = J_uvw.row(0);
  const Eigen::Vector3d g_y = J_uvw.row(1);
  const double alpha = cam_ray.dot(g_x.cross(g_y));
  // Since the projection is degree-zero homogeneous, g_x x g_y is parallel to
  // the ray, so for a unit ray |alpha| == ||g_x x g_y|| and alpha^2 is exactly
  // det(J J^T), the product of the squared singular values. Requiring
  // |alpha| > kMinRelAlpha * (||g_x||^2 + ||g_y||^2) therefore rejects singular
  // value ratios below kMinRelAlpha, i.e. condition numbers worse than ~1e6.
  // Relative, so the test is invariant to focal length.
  constexpr double kMinRelAlpha = 1e-6;
  if (!(std::abs(alpha) >
        kMinRelAlpha * (g_x.squaredNorm() + g_y.squaredNorm()))) {
    return std::nullopt;
  }
  Eigen::Matrix3x2d J_ray;
  J_ray.col(0) = g_y.cross(cam_ray);
  J_ray.col(1) = cam_ray.cross(g_x);
  return J_ray / alpha;
}

std::optional<Eigen::Vector2d> CameraModelCamFromImg(
    const CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& xy) {
  Eigen::Vector2d uv;
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                          \
  case CameraModel::model_id:                                   \
    if (CameraModel::CamFromImg(                                \
            params.data(), xy.x(), xy.y(), &uv.x(), &uv.y())) { \
      return uv;                                                \
    }                                                           \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  return std::nullopt;
}

// Unproject a pixel to a unit bearing vector in camera coordinates.
//
// Unlike CameraModelCamFromImg (which is limited to the forward hemisphere
// via the 2D normalized-coordinate representation), this function returns a
// valid 3D unit ray for any pixel on the camera model's image plane,
// including back-facing rays on omnidirectional cameras.
//
// Downstream geometry code (two-view geometry, absolute/relative pose,
// triangulation) should prefer this function when a 3D bearing is needed.
std::optional<Eigen::Vector3d> CameraModelCamRayFromImg(
    const CameraModelId model_id,
    const std::vector<double>& params,
    const Eigen::Vector2d& xy) {
  Eigen::Vector3d ray;
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::model_id:                                               \
    if (CameraModel::CamRayFromImg(                                         \
            params.data(), xy.x(), xy.y(), &ray.x(), &ray.y(), &ray.z())) { \
      return ray;                                                           \
    }                                                                       \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  return std::nullopt;
}

double CameraModelCamFromImgThreshold(const CameraModelId model_id,
                                      const std::vector<double>& params,
                                      const double threshold) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::model_id:                                          \
    return CameraModel::CamFromImgThreshold(params.data(), threshold); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return -1;
}

bool CameraModelIsPerspectiveFisheye(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                       \
  case CameraModel::model_id:                                                \
    return std::is_base_of_v<BasePerspectiveFisheyeCameraModel<CameraModel>, \
                             CameraModel>;

    CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
    default:
      return false;
  }

  return false;
}

bool CameraModelIsPerspective(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                \
  case CameraModel::model_id:                                         \
    return std::is_base_of_v<BasePerspectiveCameraModel<CameraModel>, \
                             CameraModel>;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

bool CameraModelIsPerspectivePinhole(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                       \
  case CameraModel::model_id:                                                \
    return std::is_base_of_v<BasePerspectivePinholeCameraModel<CameraModel>, \
                             CameraModel>;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

bool CameraModelIsSpherical(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                              \
  case CameraModel::model_id:                                       \
    return std::is_base_of_v<BaseSphericalCameraModel<CameraModel>, \
                             CameraModel>;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

void CameraModelRescale(const CameraModelId model_id,
                        const double scale_x,
                        const double scale_y,
                        std::vector<double>& params) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)               \
  case CameraModel::model_id:                        \
    CameraModel::Rescale(scale_x, scale_y, &params); \
    return;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

}  // namespace colmap
