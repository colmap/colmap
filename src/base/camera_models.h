// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_BASE_CAMERA_MODELS_H_
#define COLMAP_SRC_BASE_CAMERA_MODELS_H_

#include <cfloat>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

#include <Eigen/Core>

#include <ceres/jet.h>

namespace colmap {

// This file defines several different camera models and arbitrary new camera
// models can be added by the following steps:
//
//  1. Add a new struct in this file which implements all the necessary methods.
//  2. Define an unique name and model_id for the camera model and add it to
//     the struct and update `CameraModelIdToName` and `CameraModelNameToId`.
//  3. Add camera model to `CAMERA_MODEL_CASES` macro in this file.
//  4. Add new template specialization of test case for camera model to
//     `camera_models_test.cc`.
//
// A camera model can have three different types of camera parameters: focal
// length, principal point, extra parameters (abberation parameters). The
// parameter array is split into different groups, so that we can enable or
// disable the refinement of the individual groups during bundle adjustment. It
// is up to the camera model to access the parameters correctly (it is free to
// do so in an arbitrary manner) - the parameters are not accessed from outside.
//
// A camera model must have the following methods:
//
//  - `WorldToImage`: transform normalized camera coordinates to image
//    coordinates (the inverse of `ImageToWorld`). Assumes that the world
//    coordinates are given as (u, v, 1).
//  - `ImageToWorld`: transform image coordinates to normalized camera
//    coordinates (the inverse of `WorldToImage`). Produces world coordinates
//    as (u, v, 1).
//  - `ImageToWorldThreshold`: transform a threshold given in pixels to
//    normalized units (e.g. useful for reprojection error thresholds).
//
// Whenever you specify the camera parameters in a list, they must appear
// exactly in the order as they are accessed in the defined model struct.
//
// The camera models follow the convention that the upper left image corner has
// the coordinate (0, 0), the lower right corner (width, height), i.e. that
// the upper left pixel center has coordinate (0.5, 0.5) and the lower right
// pixel center has the coordinate (width - 0.5, height - 0.5).

static const int kInvalidCameraModelId = -1;

#ifndef CAMERA_MODEL_DEFINITIONS
#define CAMERA_MODEL_DEFINITIONS(model_id_value, num_params_value)             \
  static const int model_id = model_id_value;                                  \
  static const int num_params = num_params_value;                              \
  static const std::string params_info;                                        \
  static const std::vector<size_t> focal_length_idxs;                          \
  static const std::vector<size_t> principal_point_idxs;                       \
  static const std::vector<size_t> extra_params_idxs;                          \
                                                                               \
  static inline std::string InitializeParamsInfo();                            \
  static inline std::vector<size_t> InitializeFocalLengthIdxs();               \
  static inline std::vector<size_t> InitializePrincipalPointIdxs();            \
  static inline std::vector<size_t> InitializeExtraParamsIdxs();               \
                                                                               \
  template <typename T>                                                        \
  static void WorldToImage(const T* params, const T u, const T v, T* x, T* y); \
  template <typename T>                                                        \
  static void ImageToWorld(const T* params, const T x, const T y, T* u, T* v); \
  template <typename T>                                                        \
  static void Distortion(const T* extra_params, const T u, const T v, T* du,   \
                         T* dv);
#endif

#ifndef CAMERA_MODEL_CASES
#define CAMERA_MODEL_CASES                    \
  CAMERA_MODEL_CASE(SimplePinholeCameraModel) \
  CAMERA_MODEL_CASE(PinholeCameraModel)       \
  CAMERA_MODEL_CASE(SimpleRadialCameraModel)  \
  CAMERA_MODEL_CASE(RadialCameraModel)        \
  CAMERA_MODEL_CASE(OpenCVCameraModel)        \
  CAMERA_MODEL_CASE(OpenCVFisheyeCameraModel) \
  CAMERA_MODEL_CASE(FullOpenCVCameraModel)    \
  CAMERA_MODEL_CASE(FOVCameraModel)
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

// The "Curiously Recurring Template Pattern" (CRTP) is used here, so that we
// can reuse some shared functionality between all camera models -
// defined in the BaseCameraModel.
template <typename CameraModel>
struct BaseCameraModel {
  template <typename T>
  static inline bool HasBogusParams(const std::vector<T>& params,
                                    const size_t width, const size_t height,
                                    const T min_focal_length_ratio,
                                    const T max_focal_length_ratio,
                                    const T max_extra_param);

  template <typename T>
  static inline bool HasBogusFocalLength(const std::vector<T>& params,
                                         const size_t width,
                                         const size_t height,
                                         const T min_focal_length_ratio,
                                         const T max_focal_length_ratio);

  template <typename T>
  static inline bool HasBogusPrincipalPoint(const std::vector<T>& params,
                                            const size_t width,
                                            const size_t height);

  template <typename T>
  static inline bool HasBogusExtraParams(const std::vector<T>& params,
                                         const T max_extra_param);

  template <typename T>
  static inline T ImageToWorldThreshold(const T* params, const T threshold);

  template <typename T>
  static inline void IterativeUndistortion(const T* params, T* u, T* v);
};

// Simple Pinhole camera model.
//
// No Distortion is assumed. Only focal length and principal point is modeled.
//
// Parameter list is expected in the following order:
//
//   f, cx, cy
//
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct SimplePinholeCameraModel
    : public BaseCameraModel<SimplePinholeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(0, 3)
};

// Pinhole camera model.
//
// No Distortion is assumed. Only focal length and principal point is modeled.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy
//
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct PinholeCameraModel : public BaseCameraModel<PinholeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(1, 4)
};

// Simple camera model with one focal length and one radial distortion
// parameter.
//
// This model is similar to the camera model that VisualSfM uses with the
// difference that the distortion here is applied to the projections and
// not to the measurements.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k
//
struct SimpleRadialCameraModel
    : public BaseCameraModel<SimpleRadialCameraModel> {
  CAMERA_MODEL_DEFINITIONS(2, 4)
};

// Simple camera model with one focal length and two radial distortion
// parameters.
//
// This model is equivalent to the camera model that Bundler uses
// (except for an inverse z-axis in the camera coordinate system).
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k1, k2
//
struct RadialCameraModel : public BaseCameraModel<RadialCameraModel> {
  CAMERA_MODEL_DEFINITIONS(3, 5)
};

// OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential distortion (up to 2nd degree of coefficients). Not suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVCameraModel : public BaseCameraModel<OpenCVCameraModel> {
  CAMERA_MODEL_DEFINITIONS(4, 8)
};

// OpenCV fish-eye camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential Distortion (up to 2nd degree of coefficients). Suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVFisheyeCameraModel
    : public BaseCameraModel<OpenCVFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(5, 8)
};

// Full OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential Distortion.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct FullOpenCVCameraModel : public BaseCameraModel<FullOpenCVCameraModel> {
  CAMERA_MODEL_DEFINITIONS(6, 12)
};

// FOV camera model.
//
// Based on the pinhole camera model. Additionally models radial distortion.
// This model is for example used by Project Tango for its equidistant
// calibration type.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, omega
//
// See:
// Frederic Devernay, Olivier Faugeras. Straight lines have to be straight:
// Automatic calibration and removal of distortion from scenes of structured
// environments. Machine vision and applications, 2001.
struct FOVCameraModel : public BaseCameraModel<FOVCameraModel> {
  CAMERA_MODEL_DEFINITIONS(7, 5)

  template <typename T>
  static void Undistortion(const T* extra_params, const T u, const T v, T* du,
                           T* dv);
};

// Convert camera name to unique camera model identifier.
//
// @param name         Unique name of camera model.
//
// @return             Unique identifier of camera model.
int CameraModelNameToId(const std::string& name);

// Convert camera model identifier to unique camera model name.
//
// @param model_id     Unique identifier of camera model.
//
// @return             Unique name of camera model.
std::string CameraModelIdToName(const int model_id);

// Initialize camera parameters using given image properties.
//
// Initializes all focal length parameters to the same given focal length and
// sets the principal point to the image center.
//
// @param model_id      Unique identifier of camera model.
// @param focal_length  Focal length, equal for all focal length parameters.
// @param width         Sensor width of the camera.
// @param height        Sensor height of the camera.
// @param params        Array of camera parameters.
void CameraModelInitializeParams(const int model_id, const double focal_length,
                                 const size_t width, const size_t height,
                                 std::vector<double>* params);

// Get human-readable information about the parameter vector order.
//
// @param model_id     Unique identifier of camera model.
std::string CameraModelParamsInfo(const int model_id);

// Get the indices of the parameter groups in the parameter vector.
//
// @param model_id     Unique identifier of camera model.
std::vector<size_t> CameraModelFocalLengthIdxs(const int model_id);
std::vector<size_t> CameraModelPrincipalPointIdxs(const int model_id);
std::vector<size_t> CameraModelExtraParamsIdxs(const int model_id);

// Check whether parameters are valid, i.e. the parameter vector has
// the correct dimensions that match the specified camera model.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
bool CameraModelVerifyParams(const int model_id,
                             const std::vector<double>& params);

// Check whether camera has bogus parameters.
//
// @param model_id                Unique identifier of camera model.
// @param params                  Array of camera parameters.
// @param width                   Sensor width of the camera.
// @param height                  Sensor height of the camera.
// @param min_focal_length_ratio  Minimum ratio of focal length over
//                                maximum sensor dimension.
// @param min_focal_length_ratio  Maximum ratio of focal length over
//                                maximum sensor dimension.
// @param max_extra_param         Maximum magnitude of each extra parameter.
bool CameraModelHasBogusParams(const int model_id,
                               const std::vector<double>& params,
                               const size_t width, const size_t height,
                               const double min_focal_length_ratio,
                               const double max_focal_length_ratio,
                               const double max_extra_param);

// Transform world coordinates in camera coordinate system to image coordinates.
//
// This is the inverse of `CameraModelImageToWorld`.
//
// @param model_id     Unique model_id of camera model as defined in
//                     `CAMERA_MODEL_NAME_TO_CODE`.
// @param params       Array of camera parameters.
// @param u, v         Coordinates in camera system as (u, v, 1).
// @param x, y         Output image coordinates in pixels.
inline void CameraModelWorldToImage(const int model_id,
                                    const std::vector<double>& params,
                                    const double u, const double v, double* x,
                                    double* y);

// Transform image coordinates to world coordinates in camera coordinate system.
//
// This is the inverse of `CameraModelWorldToImage`.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param x, y          Image coordinates in pixels.
// @param v, u          Output Coordinates in camera system as (u, v, 1).
inline void CameraModelImageToWorld(const int model_id,
                                    const std::vector<double>& params,
                                    const double x, const double y, double* u,
                                    double* v);

// Convert pixel threshold in image plane to world space by dividing
// the threshold through the mean focal length.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param threshold     Image space threshold in pixels.
//
// @ return             World space threshold.
inline double CameraModelImageToWorldThreshold(
    const int model_id, const std::vector<double>& params,
    const double threshold);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// BaseCameraModel

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusParams(
    const std::vector<T>& params, const size_t width, const size_t height,
    const T min_focal_length_ratio, const T max_focal_length_ratio,
    const T max_extra_param) {
  if (HasBogusPrincipalPoint(params, width, height)) {
    return true;
  }

  if (HasBogusFocalLength(params, width, height, min_focal_length_ratio,
                          max_focal_length_ratio)) {
    return true;
  }

  if (HasBogusExtraParams(params, max_extra_param)) {
    return true;
  }

  return false;
}

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusFocalLength(
    const std::vector<T>& params, const size_t width, const size_t height,
    const T min_focal_length_ratio, const T max_focal_length_ratio) {
  const size_t max_size = std::max(width, height);

  for (const auto& idx : CameraModel::focal_length_idxs) {
    const T focal_length_ratio = params[idx] / max_size;
    if (focal_length_ratio < min_focal_length_ratio ||
        focal_length_ratio > max_focal_length_ratio) {
      return true;
    }
  }

  return false;
}

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusPrincipalPoint(
    const std::vector<T>& params, const size_t width, const size_t height) {
  const T cx = params[CameraModel::principal_point_idxs[0]];
  const T cy = params[CameraModel::principal_point_idxs[1]];
  return cx < 0 || cx > width || cy < 0 || cy > height;
}

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusExtraParams(
    const std::vector<T>& params, const T max_extra_param) {
  for (const auto& idx : CameraModel::extra_params_idxs) {
    if (std::abs(params[idx]) > max_extra_param) {
      return true;
    }
  }

  return false;
}

template <typename CameraModel>
template <typename T>
T BaseCameraModel<CameraModel>::ImageToWorldThreshold(const T* params,
                                                      const T threshold) {
  T mean_focal_length = 0;
  for (const auto& idx : CameraModel::focal_length_idxs) {
    mean_focal_length += params[idx];
  }
  mean_focal_length /= CameraModel::focal_length_idxs.size();
  return threshold / mean_focal_length;
}

template <typename CameraModel>
template <typename T>
void BaseCameraModel<CameraModel>::IterativeUndistortion(const T* params, T* u,
                                                         T* v) {
  // Number of iterations for iterative undistortion, 100 should be enough
  // even for complex camera models with higher order terms.
  const size_t kNumUndistortionIterations = 100;
  const double kUndistortionEpsilon = 1e-10;

  T uu = *u;
  T vv = *v;
  T du;
  T dv;

  for (size_t i = 0; i < kNumUndistortionIterations; ++i) {
    CameraModel::Distortion(params, uu, vv, &du, &dv);
    const T uu_prev = uu;
    const T vv_prev = vv;
    uu = *u - du;
    vv = *v - dv;
    if (std::abs(uu_prev - uu) < kUndistortionEpsilon &&
        std::abs(vv_prev - vv) < kUndistortionEpsilon) {
      break;
    }
  }

  *u = uu;
  *v = vv;
}

////////////////////////////////////////////////////////////////////////////////
// SimplePinholeCameraModel

std::string SimplePinholeCameraModel::InitializeParamsInfo() {
  return "f, cx, cy";
}

std::vector<size_t> SimplePinholeCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(1);
  idxs[0] = 0;
  return idxs;
}

std::vector<size_t> SimplePinholeCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 1;
  idxs[1] = 2;
  return idxs;
}

std::vector<size_t> SimplePinholeCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs;
  return idxs;
}

template <typename T>
void SimplePinholeCameraModel::WorldToImage(const T* params, const T u,
                                            const T v, T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // No Distortion

  // Transform to image coordinates
  *x = f * u + c1;
  *y = f * v + c2;
}

template <typename T>
void SimplePinholeCameraModel::ImageToWorld(const T* params, const T x,
                                            const T y, T* u, T* v) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  *u = (x - c1) / f;
  *v = (y - c2) / f;
}

////////////////////////////////////////////////////////////////////////////////
// PinholeCameraModel

std::string PinholeCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy";
}

std::vector<size_t> PinholeCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 0;
  idxs[1] = 1;
  return idxs;
}

std::vector<size_t> PinholeCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 2;
  idxs[1] = 3;
  return idxs;
}

std::vector<size_t> PinholeCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs;
  return idxs;
}

template <typename T>
void PinholeCameraModel::WorldToImage(const T* params, const T u, const T v,
                                      T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // No Distortion

  // Transform to image coordinates
  *x = f1 * u + c1;
  *y = f2 * v + c2;
}

template <typename T>
void PinholeCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                      T* u, T* v) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
}

////////////////////////////////////////////////////////////////////////////////
// SimpleRadialCameraModel

std::string SimpleRadialCameraModel::InitializeParamsInfo() {
  return "f, cx, cy, k";
}

std::vector<size_t> SimpleRadialCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(1);
  idxs[0] = 0;
  return idxs;
}

std::vector<size_t> SimpleRadialCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 1;
  idxs[1] = 2;
  return idxs;
}

std::vector<size_t> SimpleRadialCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs(1);
  idxs[0] = 3;
  return idxs;
}

template <typename T>
void SimpleRadialCameraModel::WorldToImage(const T* params, const T u,
                                           const T v, T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Distortion
  T du, dv;
  Distortion(&params[3], u, v, &du, &dv);
  *x = u + du;
  *y = v + dv;

  // Transform to image coordinates
  *x = f * *x + c1;
  *y = f * *y + c2;
}

template <typename T>
void SimpleRadialCameraModel::ImageToWorld(const T* params, const T x,
                                           const T y, T* u, T* v) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;

  IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void SimpleRadialCameraModel::Distortion(const T* extra_params, const T u,
                                         const T v, T* du, T* dv) {
  const T k = extra_params[0];

  const T u2 = u * u;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T radial = k * r2;
  *du = u * radial;
  *dv = v * radial;
}

////////////////////////////////////////////////////////////////////////////////
// RadialCameraModel

std::string RadialCameraModel::InitializeParamsInfo() {
  return "f, cx, cy, k1, k2";
}

std::vector<size_t> RadialCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(1);
  idxs[0] = 0;
  return idxs;
}

std::vector<size_t> RadialCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 1;
  idxs[1] = 2;
  return idxs;
}

std::vector<size_t> RadialCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 3;
  idxs[1] = 4;
  return idxs;
}

template <typename T>
void RadialCameraModel::WorldToImage(const T* params, const T u, const T v,
                                     T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Distortion
  T du, dv;
  Distortion(&params[3], u, v, &du, &dv);
  *x = u + du;
  *y = v + dv;

  // Transform to image coordinates
  *x = f * *x + c1;
  *y = f * *y + c2;
}

template <typename T>
void RadialCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                     T* u, T* v) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;

  IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void RadialCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                   T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];

  const T u2 = u * u;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T radial = k1 * r2 + k2 * r2 * r2;
  *du = u * radial;
  *dv = v * radial;
}

////////////////////////////////////////////////////////////////////////////////
// OpenCVCameraModel

std::string OpenCVCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy, k1, k2, p1, p2";
}

std::vector<size_t> OpenCVCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 0;
  idxs[1] = 1;
  return idxs;
}

std::vector<size_t> OpenCVCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 2;
  idxs[1] = 3;
  return idxs;
}

std::vector<size_t> OpenCVCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs(4);
  idxs[0] = 4;
  idxs[1] = 5;
  idxs[2] = 6;
  idxs[3] = 7;
  return idxs;
}

template <typename T>
void OpenCVCameraModel::WorldToImage(const T* params, const T u, const T v,
                                     T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Distortion
  T du, dv;
  Distortion(&params[4], u, v, &du, &dv);
  *x = u + du;
  *y = v + dv;

  // Transform to image coordinates
  *x = f1 * *x + c1;
  *y = f2 * *y + c2;
}

template <typename T>
void OpenCVCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                     T* u, T* v) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;

  IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void OpenCVCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                   T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];
  const T p1 = extra_params[2];
  const T p2 = extra_params[3];

  const T u2 = u * u;
  const T uv = u * v;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T radial = k1 * r2 + k2 * r2 * r2;
  *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
  *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

////////////////////////////////////////////////////////////////////////////////
// OpenCVFisheyeCameraModel

std::string OpenCVFisheyeCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy, k1, k2, p1, p2";
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 0;
  idxs[1] = 1;
  return idxs;
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 2;
  idxs[1] = 3;
  return idxs;
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs(4);
  idxs[0] = 4;
  idxs[1] = 5;
  idxs[2] = 6;
  idxs[3] = 7;
  return idxs;
}

template <typename T>
void OpenCVFisheyeCameraModel::WorldToImage(const T* params, const T u,
                                            const T v, T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  const T r = ceres::sqrt(u * u + v * v);

  T uu, vv;
  if (r > T(std::numeric_limits<double>::epsilon())) {
    const T theta = ceres::atan2(r, T(1));
    uu = theta * u / r;
    vv = theta * v / r;
  } else {
    uu = u;
    vv = v;
  }

  // Distortion
  T du, dv;
  Distortion(&params[4], uu, vv, &du, &dv);
  *x = uu + du;
  *y = vv + dv;

  // Transform to image coordinates
  *x = f1 * *x + c1;
  *y = f2 * *y + c2;
}

template <typename T>
void OpenCVFisheyeCameraModel::ImageToWorld(const T* params, const T x,
                                            const T y, T* u, T* v) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;

  IterativeUndistortion(&params[4], u, v);

  const T theta = ceres::sqrt(*u * *u + *v * *v);
  const T theta_cos_theta = theta * ceres::cos(theta);
  if (theta_cos_theta > T(std::numeric_limits<double>::epsilon())) {
    const T scale = ceres::sin(theta) / theta_cos_theta;
    *u *= scale;
    *v *= scale;
  }
}

template <typename T>
void OpenCVFisheyeCameraModel::Distortion(const T* extra_params, const T u,
                                          const T v, T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];
  const T p1 = extra_params[2];
  const T p2 = extra_params[3];

  const T u2 = u * u;
  const T uv = u * v;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T radial = k1 * r2 + k2 * r2 * r2;
  *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
  *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

////////////////////////////////////////////////////////////////////////////////
// FullOpenCVCameraModel

std::string FullOpenCVCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6";
}

std::vector<size_t> FullOpenCVCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 0;
  idxs[1] = 1;
  return idxs;
}

std::vector<size_t> FullOpenCVCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 2;
  idxs[1] = 3;
  return idxs;
}

std::vector<size_t> FullOpenCVCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs(8);
  idxs[0] = 4;
  idxs[1] = 5;
  idxs[2] = 6;
  idxs[3] = 7;
  idxs[4] = 8;
  idxs[5] = 9;
  idxs[6] = 10;
  idxs[7] = 11;
  return idxs;
}

template <typename T>
void FullOpenCVCameraModel::WorldToImage(const T* params, const T u, const T v,
                                         T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Distortion
  T du, dv;
  Distortion(&params[4], u, v, &du, &dv);
  *x = u + du;
  *y = v + dv;

  // Transform to image coordinates
  *x = f1 * *x + c1;
  *y = f2 * *y + c2;
}

template <typename T>
void FullOpenCVCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                         T* u, T* v) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;

  IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void FullOpenCVCameraModel::Distortion(const T* extra_params, const T u,
                                       const T v, T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];
  const T p1 = extra_params[2];
  const T p2 = extra_params[3];
  const T k3 = extra_params[4];
  const T k4 = extra_params[5];
  const T k5 = extra_params[6];
  const T k6 = extra_params[7];

  const T u2 = u * u;
  const T uv = u * v;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;
  const T radial = (T(1) + k1 * r2 + k2 * r4 + k3 * r6) /
                   (T(1) + k4 * r2 + k5 * r4 + k6 * r6);
  *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2) - u;
  *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2) - v;
}

////////////////////////////////////////////////////////////////////////////////
// FOVCameraModel

std::string FOVCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy, omega";
}

std::vector<size_t> FOVCameraModel::InitializeFocalLengthIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 0;
  idxs[1] = 1;
  return idxs;
}

std::vector<size_t> FOVCameraModel::InitializePrincipalPointIdxs() {
  std::vector<size_t> idxs(2);
  idxs[0] = 2;
  idxs[1] = 3;
  return idxs;
}

std::vector<size_t> FOVCameraModel::InitializeExtraParamsIdxs() {
  std::vector<size_t> idxs(1);
  idxs[0] = 4;
  return idxs;
}

template <typename T>
void FOVCameraModel::WorldToImage(const T* params, const T u, const T v, T* x,
                                  T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Distortion
  T du, dv;
  Distortion(&params[4], u, v, &du, &dv);
  *x = u + du;
  *y = v + dv;

  // Transform to image coordinates
  *x = f1 * *x + c1;
  *y = f2 * *y + c2;
}

template <typename T>
void FOVCameraModel::ImageToWorld(const T* params, const T x, const T y, T* u,
                                  T* v) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;

  // Undistortion
  T du, dv;
  Undistortion(&params[4], *u, *v, &du, &dv);
  *u = *u + du;
  *v = *v + dv;
}

template <typename T>
void FOVCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                T* du, T* dv) {
  const T omega = extra_params[0];

  const T radius = ceres::sqrt(u * u + v * v);
  T radial;
  const T kEpsilon = T(1e-6);  // Chosen arbitrarily.
  if (radius < kEpsilon) {
    // Derivation of this case with Matlab:
    // syms radius omega;
    // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
    //                  (radius * omega);
    // limit(factor, radius, 0, 'right')
    radial = (T(2) * ceres::tan(omega / T(2))) / omega;
  } else {
    const T numerator = ceres::atan(radius * T(2) * ceres::tan(omega / T(2)));
    radial = numerator / (radius * omega);
  }

  *du = u * radial - u;
  *dv = v * radial - v;
}

template <typename T>
void FOVCameraModel::Undistortion(const T* extra_params, const T u, const T v,
                                  T* du, T* dv) {
  const T omega = extra_params[0];

  const T radius = ceres::sqrt(u * u + v * v);
  T radial;
  const T kEpsilon = T(1e-6);  // Chosen arbitrarily.
  if (radius < kEpsilon) {
    // Derivation of this case with Matlab:
    // syms radius omega;
    // factor(radius) = tan(radius * omega) / ...
    //                  (radius * 2*tan(omega/2));
    // limit(factor, radius, 0, 'right')
    radial = omega / (T(2) * ceres::tan(omega / T(2)));
  } else {
    const T numerator = ceres::tan(radius * omega);
    radial = numerator / (radius * T(2) * ceres::tan(omega / T(2)));
  }

  *du = u * radial - u;
  *dv = v * radial - v;
}

////////////////////////////////////////////////////////////////////////////////

void CameraModelWorldToImage(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, double* x, double* y) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
  case CameraModel::model_id:                             \
    CameraModel::WorldToImage(params.data(), u, v, x, y); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

void CameraModelImageToWorld(const int model_id,
                             const std::vector<double>& params, const double x,
                             const double y, double* u, double* v) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
  case CameraModel::model_id:                             \
    CameraModel::ImageToWorld(params.data(), x, y, u, v); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

double CameraModelImageToWorldThreshold(const int model_id,
                                        const std::vector<double>& params,
                                        const double threshold) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::model_id:                                            \
    return CameraModel::ImageToWorldThreshold(params.data(), threshold); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return -1;
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_CAMERA_MODELS_H_
