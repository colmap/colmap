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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include <cfloat>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace colmap {

// This file defines several different camera models and arbitrary new camera
// models can be added by the following steps:
//
//  1. Add a new struct in this file which implements all the necessary methods.
//  2. Define an unique model_name and model_id for the camera model.
//  3. Add camera model to `CAMERA_MODEL_CASES` macro in this file.
//  4. Add new template specialization of test case for camera model to
//     `camera_models_test.cc`.
//
// A camera model can have three different types of camera parameters: focal
// length, principal point, extra parameters (distortion parameters). The
// parameter array is split into different groups, so that we can enable or
// disable the refinement of the individual groups during bundle adjustment. It
// is up to the camera model to access the parameters correctly (it is free to
// do so in an arbitrary manner) - the parameters are not accessed from outside.
//
// A camera model must have the following methods:
//
//  - `ImgFromCam`: transform normalized camera coordinates to image
//    coordinates (the inverse of `CamFromImg`). Assumes that the camera
//    coordinates are given as (u, v, 1).
//  - `CamFromImg`: transform image coordinates to normalized camera
//    coordinates (the inverse of `ImgFromCam`). Produces camera coordinates
//    as (u, v, 1).
//  - `CamFromImgThreshold`: transform a threshold given in pixels to
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
#define CAMERA_MODEL_DEFINITIONS(                                          \
    model_id_value, model_name_value, num_params_value)                    \
  static const int kModelId = model_id_value;                              \
  static const size_t kNumParams = num_params_value;                       \
  static const int model_id;                                               \
  static const std::string model_name;                                     \
  static const size_t num_params;                                          \
  static const std::string params_info;                                    \
  static const std::vector<size_t> focal_length_idxs;                      \
  static const std::vector<size_t> principal_point_idxs;                   \
  static const std::vector<size_t> extra_params_idxs;                      \
                                                                           \
  static inline int InitializeModelId() { return model_id_value; };        \
  static inline std::string InitializeModelName() {                        \
    return model_name_value;                                               \
  };                                                                       \
  static inline size_t InitializeNumParams() { return num_params_value; }; \
  static inline std::string InitializeParamsInfo();                        \
  static inline std::vector<size_t> InitializeFocalLengthIdxs();           \
  static inline std::vector<size_t> InitializePrincipalPointIdxs();        \
  static inline std::vector<size_t> InitializeExtraParamsIdxs();           \
  static inline std::vector<double> InitializeParams(                      \
      double focal_length, size_t width, size_t height);                   \
                                                                           \
  template <typename T>                                                    \
  static void ImgFromCam(const T* params, T u, T v, T w, T* x, T* y);      \
  template <typename T>                                                    \
  static void CamFromImg(const T* params, T x, T y, T* u, T* v, T* w);     \
  template <typename T>                                                    \
  static void Distortion(const T* extra_params, T u, T v, T* du, T* dv);
#endif

#ifndef CAMERA_MODEL_CASES
#define CAMERA_MODEL_CASES                          \
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
  CAMERA_MODEL_CASE(ThinPrismFisheyeCameraModel)
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
                                    size_t width,
                                    size_t height,
                                    T min_focal_length_ratio,
                                    T max_focal_length_ratio,
                                    T max_extra_param);

  template <typename T>
  static inline bool HasBogusFocalLength(const std::vector<T>& params,
                                         size_t width,
                                         size_t height,
                                         T min_focal_length_ratio,
                                         T max_focal_length_ratio);

  template <typename T>
  static inline bool HasBogusPrincipalPoint(const std::vector<T>& params,
                                            size_t width,
                                            size_t height);

  template <typename T>
  static inline bool HasBogusExtraParams(const std::vector<T>& params,
                                         T max_extra_param);

  template <typename T>
  static inline T CamFromImgThreshold(const T* params, T threshold);

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
  CAMERA_MODEL_DEFINITIONS(0, "SIMPLE_PINHOLE", 3)
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
  CAMERA_MODEL_DEFINITIONS(1, "PINHOLE", 4)
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
  CAMERA_MODEL_DEFINITIONS(2, "SIMPLE_RADIAL", 4)
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
  CAMERA_MODEL_DEFINITIONS(3, "RADIAL", 5)
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
  CAMERA_MODEL_DEFINITIONS(4, "OPENCV", 8)
};

// OpenCV fish-eye camera model.
//
// Based on the pinhole camera model. Additionally models radial distortion
// (up to 4th degree of coefficients). Suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, k3, k4
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVFisheyeCameraModel
    : public BaseCameraModel<OpenCVFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(5, "OPENCV_FISHEYE", 8)
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
  CAMERA_MODEL_DEFINITIONS(6, "FULL_OPENCV", 12)
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
  CAMERA_MODEL_DEFINITIONS(7, "FOV", 5)

  template <typename T>
  static void Undistortion(const T* extra_params, T u, T v, T* du, T* dv);
};

// Simple camera model with one focal length and one radial distortion
// parameter, suitable for fish-eye cameras.
//
// This model is equivalent to the OpenCVFisheyeCameraModel but has only one
// radial distortion coefficient.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k
//
struct SimpleRadialFisheyeCameraModel
    : public BaseCameraModel<SimpleRadialFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(8, "SIMPLE_RADIAL_FISHEYE", 4)
};

// Simple camera model with one focal length and two radial distortion
// parameters, suitable for fish-eye cameras.
//
// This model is equivalent to the OpenCVFisheyeCameraModel but has only two
// radial distortion coefficients.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k1, k2
//
struct RadialFisheyeCameraModel
    : public BaseCameraModel<RadialFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(9, "RADIAL_FISHEYE", 5)
};

// Camera model with radial and tangential distortion coefficients and
// additional coefficients accounting for thin-prism distortion.
//
// This camera model is described in
//
//    "Camera Calibration with Distortion Models and Accuracy Evaluation",
//    J Weng et al., TPAMI, 1992.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
//
struct ThinPrismFisheyeCameraModel
    : public BaseCameraModel<ThinPrismFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(10, "THIN_PRISM_FISHEYE", 12)
};

// Check whether camera model with given name or identifier exists.
bool ExistsCameraModelWithName(const std::string& model_name);
bool ExistsCameraModelWithId(int model_id);

// Convert camera name to unique camera model identifier.
//
// @param name         Unique name of camera model.
//
// @return             Unique identifier of camera model.
int CameraModelNameToId(const std::string& model_name);

// Convert camera model identifier to unique camera model name.
//
// @param model_id     Unique identifier of camera model.
//
// @return             Unique name of camera model.
std::string CameraModelIdToName(int model_id);

// Initialize camera parameters using given image properties.
//
// Initializes all focal length parameters to the same given focal length and
// sets the principal point to the image center.
//
// @param model_id      Unique identifier of camera model.
// @param focal_length  Focal length, equal for all focal length parameters.
// @param width         Sensor width of the camera.
// @param height        Sensor height of the camera.
std::vector<double> CameraModelInitializeParams(int model_id,
                                                double focal_length,
                                                size_t width,
                                                size_t height);

// Get human-readable information about the parameter vector order.
//
// @param model_id     Unique identifier of camera model.
std::string CameraModelParamsInfo(int model_id);

// Get the indices of the parameter groups in the parameter vector.
//
// @param model_id     Unique identifier of camera model.
const std::vector<size_t>& CameraModelFocalLengthIdxs(int model_id);
const std::vector<size_t>& CameraModelPrincipalPointIdxs(int model_id);
const std::vector<size_t>& CameraModelExtraParamsIdxs(int model_id);

// Get the total number of parameters of a camera model.
size_t CameraModelNumParams(int model_id);

// Check whether parameters are valid, i.e. the parameter vector has
// the correct dimensions that match the specified camera model.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
bool CameraModelVerifyParams(int model_id, const std::vector<double>& params);

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
bool CameraModelHasBogusParams(int model_id,
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
// @param model_id     Unique model_id of camera model as defined in
//                     `CAMERA_MODEL_NAME_TO_CODE`.
// @param params       Array of camera parameters.
// @param u, v         Coordinates in camera system as (u, v, 1).
// @param x, y         Output image coordinates in pixels.
inline Eigen::Vector2d CameraModelImgFromCam(int model_id,
                                             const std::vector<double>& params,
                                             const Eigen::Vector3d& uvw);

// Transform image to camera coordinates.
//
// This is the inverse of `CameraModelImgFromCam`.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param xy            Image coordinates in pixels.
//
// @return              Output Coordinates in camera system as (u, v, w=1).
inline Eigen::Vector3d CameraModelCamFromImg(int model_id,
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
inline double CameraModelCamFromImgThreshold(int model_id,
                                             const std::vector<double>& params,
                                             double threshold);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// BaseCameraModel

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusParams(
    const std::vector<T>& params,
    const size_t width,
    const size_t height,
    const T min_focal_length_ratio,
    const T max_focal_length_ratio,
    const T max_extra_param) {
  return HasBogusPrincipalPoint(params, width, height) ||
         HasBogusFocalLength(params,
                             width,
                             height,
                             min_focal_length_ratio,
                             max_focal_length_ratio) ||
         HasBogusExtraParams(params, max_extra_param);
}

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusFocalLength(
    const std::vector<T>& params,
    const size_t width,
    const size_t height,
    const T min_focal_length_ratio,
    const T max_focal_length_ratio) {
  const T inv_max_size = 1.0 / std::max(width, height);
  for (const auto& idx : CameraModel::focal_length_idxs) {
    const T focal_length_ratio = params[idx] * inv_max_size;
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
T BaseCameraModel<CameraModel>::CamFromImgThreshold(const T* params,
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
void BaseCameraModel<CameraModel>::IterativeUndistortion(const T* params,
                                                         T* u,
                                                         T* v) {
  // Parameters for Newton iteration using numerical differentiation with
  // central differences, 100 iterations should be enough even for complex
  // camera models with higher order terms.
  const size_t kNumIterations = 100;
  const double kMaxStepNorm = 1e-10;
  const double kRelStepSize = 1e-6;

  Eigen::Matrix2d J;
  const Eigen::Vector2d x0(*u, *v);
  Eigen::Vector2d x(*u, *v);
  Eigen::Vector2d dx;
  Eigen::Vector2d dx_0b;
  Eigen::Vector2d dx_0f;
  Eigen::Vector2d dx_1b;
  Eigen::Vector2d dx_1f;

  for (size_t i = 0; i < kNumIterations; ++i) {
    const double step0 = std::max(std::numeric_limits<double>::epsilon(),
                                  std::abs(kRelStepSize * x(0)));
    const double step1 = std::max(std::numeric_limits<double>::epsilon(),
                                  std::abs(kRelStepSize * x(1)));
    CameraModel::Distortion(params, x(0), x(1), &dx(0), &dx(1));
    CameraModel::Distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
    CameraModel::Distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
    CameraModel::Distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
    CameraModel::Distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
    J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
    J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
    J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
    J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
    const Eigen::Vector2d step_x = J.inverse() * (x + dx - x0);
    x -= step_x;
    if (step_x.squaredNorm() < kMaxStepNorm) {
      break;
    }
  }

  *u = x(0);
  *v = x(1);
}

////////////////////////////////////////////////////////////////////////////////
// SimplePinholeCameraModel

std::string SimplePinholeCameraModel::InitializeParamsInfo() {
  return "f, cx, cy";
}

std::vector<size_t> SimplePinholeCameraModel::InitializeFocalLengthIdxs() {
  return {0};
}

std::vector<size_t> SimplePinholeCameraModel::InitializePrincipalPointIdxs() {
  return {1, 2};
}

std::vector<size_t> SimplePinholeCameraModel::InitializeExtraParamsIdxs() {
  return {};
}

std::vector<double> SimplePinholeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, width / 2.0, height / 2.0};
}

template <typename T>
void SimplePinholeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // No Distortion

  // Transform to image coordinates
  *x = f * u / w + c1;
  *y = f * v / w + c2;
}

template <typename T>
void SimplePinholeCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  *u = (x - c1) / f;
  *v = (y - c2) / f;
  *w = 1;
}

////////////////////////////////////////////////////////////////////////////////
// PinholeCameraModel

std::string PinholeCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy";
}

std::vector<size_t> PinholeCameraModel::InitializeFocalLengthIdxs() {
  return {0, 1};
}

std::vector<size_t> PinholeCameraModel::InitializePrincipalPointIdxs() {
  return {2, 3};
}

std::vector<size_t> PinholeCameraModel::InitializeExtraParamsIdxs() {
  return {};
}

std::vector<double> PinholeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, focal_length, width / 2.0, height / 2.0};
}

template <typename T>
void PinholeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // No Distortion

  // Transform to image coordinates
  *x = f1 * u / w + c1;
  *y = f2 * v / w + c2;
}

template <typename T>
void PinholeCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
  *w = 1;
}

////////////////////////////////////////////////////////////////////////////////
// SimpleRadialCameraModel

std::string SimpleRadialCameraModel::InitializeParamsInfo() {
  return "f, cx, cy, k";
}

std::vector<size_t> SimpleRadialCameraModel::InitializeFocalLengthIdxs() {
  return {0};
}

std::vector<size_t> SimpleRadialCameraModel::InitializePrincipalPointIdxs() {
  return {1, 2};
}

std::vector<size_t> SimpleRadialCameraModel::InitializeExtraParamsIdxs() {
  return {3};
}

std::vector<double> SimpleRadialCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, width / 2.0, height / 2.0, 0};
}

template <typename T>
void SimpleRadialCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  u /= w;
  v /= w;

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
void SimpleRadialCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;
  *w = 1;

  IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void SimpleRadialCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
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
  return {0};
}

std::vector<size_t> RadialCameraModel::InitializePrincipalPointIdxs() {
  return {1, 2};
}

std::vector<size_t> RadialCameraModel::InitializeExtraParamsIdxs() {
  return {3, 4};
}

std::vector<double> RadialCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, width / 2.0, height / 2.0, 0, 0};
}

template <typename T>
void RadialCameraModel::ImgFromCam(const T* params, T u, T v, T w, T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  u /= w;
  v /= w;

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
void RadialCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;
  *w = 1;

  IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void RadialCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
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
  return {0, 1};
}

std::vector<size_t> OpenCVCameraModel::InitializePrincipalPointIdxs() {
  return {2, 3};
}

std::vector<size_t> OpenCVCameraModel::InitializeExtraParamsIdxs() {
  return {4, 5, 6, 7};
}

std::vector<double> OpenCVCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0};
}

template <typename T>
void OpenCVCameraModel::ImgFromCam(const T* params, T u, T v, T w, T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  u /= w;
  v /= w;

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
void OpenCVCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
  *w = 1;

  IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void OpenCVCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
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
  return "fx, fy, cx, cy, k1, k2, k3, k4";
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializeFocalLengthIdxs() {
  return {0, 1};
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializePrincipalPointIdxs() {
  return {2, 3};
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializeExtraParamsIdxs() {
  return {4, 5, 6, 7};
}

std::vector<double> OpenCVFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0};
}

template <typename T>
void OpenCVFisheyeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  u /= w;
  v /= w;

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
void OpenCVFisheyeCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
  *w = 1;

  IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void OpenCVFisheyeCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];
  const T k3 = extra_params[2];
  const T k4 = extra_params[3];

  const T r = ceres::sqrt(u * u + v * v);

  if (r > T(std::numeric_limits<double>::epsilon())) {
    const T theta = ceres::atan(r);
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T theta6 = theta4 * theta2;
    const T theta8 = theta4 * theta4;
    const T thetad =
        theta * (T(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
    *du = u * thetad / r - u;
    *dv = v * thetad / r - v;
  } else {
    *du = T(0);
    *dv = T(0);
  }
}

////////////////////////////////////////////////////////////////////////////////
// FullOpenCVCameraModel

std::string FullOpenCVCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6";
}

std::vector<size_t> FullOpenCVCameraModel::InitializeFocalLengthIdxs() {
  return {0, 1};
}

std::vector<size_t> FullOpenCVCameraModel::InitializePrincipalPointIdxs() {
  return {2, 3};
}

std::vector<size_t> FullOpenCVCameraModel::InitializeExtraParamsIdxs() {
  return {4, 5, 6, 7, 8, 9, 10, 11};
}

std::vector<double> FullOpenCVCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length,
          focal_length,
          width / 2.0,
          height / 2.0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0};
}

template <typename T>
void FullOpenCVCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  u /= w;
  v /= w;

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
void FullOpenCVCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
  *w = 1;

  IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void FullOpenCVCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
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
  return {0, 1};
}

std::vector<size_t> FOVCameraModel::InitializePrincipalPointIdxs() {
  return {2, 3};
}

std::vector<size_t> FOVCameraModel::InitializeExtraParamsIdxs() { return {4}; }

std::vector<double> FOVCameraModel::InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
  return {focal_length, focal_length, width / 2.0, height / 2.0, 1e-2};
}

template <typename T>
void FOVCameraModel::ImgFromCam(const T* params, T u, T v, T w, T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  u /= w;
  v /= w;

  // Distortion
  Distortion(&params[4], u, v, x, y);

  // Transform to image coordinates
  *x = f1 * *x + c1;
  *y = f2 * *y + c2;
}

template <typename T>
void FOVCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  const T uu = (x - c1) / f1;
  const T vv = (y - c2) / f2;
  *w = 1;

  // Undistortion
  Undistortion(&params[4], uu, vv, u, v);
}

template <typename T>
void FOVCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
  const T omega = extra_params[0];

  // Chosen arbitrarily.
  const T kEpsilon = T(1e-4);

  const T radius2 = u * u + v * v;
  const T omega2 = omega * omega;

  T factor;
  if (omega2 < kEpsilon) {
    // Derivation of this case with Matlab:
    // syms radius omega;
    // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
    //                  (radius * omega);
    // simplify(taylor(factor, omega, 'order', 3))
    factor = (omega2 * radius2) / T(3) - omega2 / T(12) + T(1);
  } else if (radius2 < kEpsilon) {
    // Derivation of this case with Matlab:
    // syms radius omega;
    // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
    //                  (radius * omega);
    // simplify(taylor(factor, radius, 'order', 3))
    const T tan_half_omega = ceres::tan(omega / T(2));
    factor = (T(-2) * tan_half_omega *
              (T(4) * radius2 * tan_half_omega * tan_half_omega - T(3))) /
             (T(3) * omega);
  } else {
    const T radius = ceres::sqrt(radius2);
    const T numerator = ceres::atan(radius * T(2) * ceres::tan(omega / T(2)));
    factor = numerator / (radius * omega);
  }

  *du = u * factor;
  *dv = v * factor;
}

template <typename T>
void FOVCameraModel::Undistortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
  T omega = extra_params[0];

  // Chosen arbitrarily.
  const T kEpsilon = T(1e-4);

  const T radius2 = u * u + v * v;
  const T omega2 = omega * omega;

  T factor;
  if (omega2 < kEpsilon) {
    // Derivation of this case with Matlab:
    // syms radius omega;
    // factor(radius) = tan(radius * omega) / ...
    //                  (radius * 2*tan(omega/2));
    // simplify(taylor(factor, omega, 'order', 3))
    factor = (omega2 * radius2) / T(3) - omega2 / T(12) + T(1);
  } else if (radius2 < kEpsilon) {
    // Derivation of this case with Matlab:
    // syms radius omega;
    // factor(radius) = tan(radius * omega) / ...
    //                  (radius * 2*tan(omega/2));
    // simplify(taylor(factor, radius, 'order', 3))
    factor = (omega * (omega * omega * radius2 + T(3))) /
             (T(6) * ceres::tan(omega / T(2)));
  } else {
    const T radius = ceres::sqrt(radius2);
    const T numerator = ceres::tan(radius * omega);
    factor = numerator / (radius * T(2) * ceres::tan(omega / T(2)));
  }

  *du = u * factor;
  *dv = v * factor;
}

////////////////////////////////////////////////////////////////////////////////
// SimpleRadialFisheyeCameraModel

std::string SimpleRadialFisheyeCameraModel::InitializeParamsInfo() {
  return "f, cx, cy, k";
}

std::vector<size_t>
SimpleRadialFisheyeCameraModel::InitializeFocalLengthIdxs() {
  return {0};
}

std::vector<size_t>
SimpleRadialFisheyeCameraModel::InitializePrincipalPointIdxs() {
  return {1, 2};
}

std::vector<size_t>
SimpleRadialFisheyeCameraModel::InitializeExtraParamsIdxs() {
  return {3};
}

std::vector<double> SimpleRadialFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, width / 2.0, height / 2.0, 0};
}

template <typename T>
void SimpleRadialFisheyeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  u /= w;
  v /= w;

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
void SimpleRadialFisheyeCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;
  *w = 1;

  IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void SimpleRadialFisheyeCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
  const T k = extra_params[0];

  const T r = ceres::sqrt(u * u + v * v);

  if (r > T(std::numeric_limits<double>::epsilon())) {
    const T theta = ceres::atan(r);
    const T theta2 = theta * theta;
    const T thetad = theta * (T(1) + k * theta2);
    *du = u * thetad / r - u;
    *dv = v * thetad / r - v;
  } else {
    *du = T(0);
    *dv = T(0);
  }
}

////////////////////////////////////////////////////////////////////////////////
// RadialFisheyeCameraModel

std::string RadialFisheyeCameraModel::InitializeParamsInfo() {
  return "f, cx, cy, k1, k2";
}

std::vector<size_t> RadialFisheyeCameraModel::InitializeFocalLengthIdxs() {
  return {0};
}

std::vector<size_t> RadialFisheyeCameraModel::InitializePrincipalPointIdxs() {
  return {1, 2};
}

std::vector<size_t> RadialFisheyeCameraModel::InitializeExtraParamsIdxs() {
  return {3, 4};
}

std::vector<double> RadialFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length, width / 2.0, height / 2.0, 0, 0};
}

template <typename T>
void RadialFisheyeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  u /= w;
  v /= w;

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
void RadialFisheyeCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f = params[0];
  const T c1 = params[1];
  const T c2 = params[2];

  // Lift points to normalized plane
  *u = (x - c1) / f;
  *v = (y - c2) / f;
  *w = 1;

  IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void RadialFisheyeCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];

  const T r = ceres::sqrt(u * u + v * v);

  if (r > T(std::numeric_limits<double>::epsilon())) {
    const T theta = ceres::atan(r);
    const T theta2 = theta * theta;
    const T theta4 = theta2 * theta2;
    const T thetad = theta * (T(1) + k1 * theta2 + k2 * theta4);
    *du = u * thetad / r - u;
    *dv = v * thetad / r - v;
  } else {
    *du = T(0);
    *dv = T(0);
  }
}

////////////////////////////////////////////////////////////////////////////////
// ThinPrismFisheyeCameraModel

std::string ThinPrismFisheyeCameraModel::InitializeParamsInfo() {
  return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1";
}

std::vector<size_t> ThinPrismFisheyeCameraModel::InitializeFocalLengthIdxs() {
  return {0, 1};
}

std::vector<size_t>
ThinPrismFisheyeCameraModel::InitializePrincipalPointIdxs() {
  return {2, 3};
}

std::vector<size_t> ThinPrismFisheyeCameraModel::InitializeExtraParamsIdxs() {
  return {4, 5, 6, 7, 8, 9, 10, 11};
}

std::vector<double> ThinPrismFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
  return {focal_length,
          focal_length,
          width / 2.0,
          height / 2.0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0};
}

template <typename T>
void ThinPrismFisheyeCameraModel::ImgFromCam(
    const T* params, T u, T v, T w, T* x, T* y) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  u /= w;
  v /= w;

  const T r = ceres::sqrt(u * u + v * v);

  T uu, vv;
  if (r > T(std::numeric_limits<double>::epsilon())) {
    const T theta = ceres::atan(r);
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
void ThinPrismFisheyeCameraModel::CamFromImg(
    const T* params, const T x, const T y, T* u, T* v, T* w) {
  const T f1 = params[0];
  const T f2 = params[1];
  const T c1 = params[2];
  const T c2 = params[3];

  // Lift points to normalized plane
  *u = (x - c1) / f1;
  *v = (y - c2) / f2;
  *w = 1;

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
void ThinPrismFisheyeCameraModel::Distortion(
    const T* extra_params, const T u, const T v, T* du, T* dv) {
  const T k1 = extra_params[0];
  const T k2 = extra_params[1];
  const T p1 = extra_params[2];
  const T p2 = extra_params[3];
  const T k3 = extra_params[4];
  const T k4 = extra_params[5];
  const T sx1 = extra_params[6];
  const T sy1 = extra_params[7];

  const T u2 = u * u;
  const T uv = u * v;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;
  const T r8 = r6 * r2;
  const T radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
  *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2) + sx1 * r2;
  *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2) + sy1 * r2;
}

////////////////////////////////////////////////////////////////////////////////

Eigen::Vector2d CameraModelImgFromCam(const int model_id,
                                      const std::vector<double>& params,
                                      const Eigen::Vector3d& uvw) {
  Eigen::Vector2d xy;
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                               \
  case CameraModel::kModelId:                                        \
    CameraModel::ImgFromCam(                                         \
        params.data(), uvw.x(), uvw.y(), uvw.z(), &xy.x(), &xy.y()); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  return xy;
}

Eigen::Vector3d CameraModelCamFromImg(const int model_id,
                                      const std::vector<double>& params,
                                      const Eigen::Vector2d& xy) {
  Eigen::Vector3d uvw;
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                \
  case CameraModel::kModelId:                                         \
    CameraModel::CamFromImg(                                          \
        params.data(), xy.x(), xy.y(), &uvw.x(), &uvw.y(), &uvw.z()); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
  return uvw;
}

double CameraModelCamFromImgThreshold(const int model_id,
                                      const std::vector<double>& params,
                                      const double threshold) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::kModelId:                                          \
    return CameraModel::CamFromImgThreshold(params.data(), threshold); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return -1;
}

}  // namespace colmap
