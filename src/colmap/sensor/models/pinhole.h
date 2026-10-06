// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/models/base.h"

namespace colmap {

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
    : public BasePerspectivePinholeCameraModel<SimplePinholeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSimplePinhole,
                                       "SIMPLE_PINHOLE",
                                       "f, cx, cy",
                                       1,
                                       2,
                                       0,
                                       true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, width / 2.0, height / 2.0};
  }

  template <typename T>
  static inline bool ImgFromCam(const T* params,
                                const T& u,
                                const T& v,
                                const T& w,
                                T* x,
                                T* y,
                                const bool check_cheirality = true) {
    if (!HasProjectableDepth(w, check_cheirality)) {
      return false;
    }

    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // No Distortion

    // Transform to image coordinates
    *x = f * u / w + c1;
    *y = f * v / w + c2;

    return true;
  }

  template <typename T>
  static inline bool CamFromImg(
      const T* params, const T& x, const T& y, T* u, T* v) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    *u = (x - c1) / f;
    *v = (y - c2) / f;

    return true;
  }
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
struct PinholeCameraModel
    : public BasePerspectivePinholeCameraModel<PinholeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kPinhole, "PINHOLE", "fx, fy, cx, cy", 2, 2, 0, true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0};
  }

  template <typename T>
  static inline bool ImgFromCam(const T* params,
                                const T& u,
                                const T& v,
                                const T& w,
                                T* x,
                                T* y,
                                const bool check_cheirality = true) {
    if (!HasProjectableDepth(w, check_cheirality)) {
      return false;
    }

    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // No Distortion

    // Transform to image coordinates
    *x = f1 * u / w + c1;
    *y = f2 * v / w + c2;

    return true;
  }

  template <typename T>
  static inline bool CamFromImg(
      const T* params, const T& x, const T& y, T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    return true;
  }
};

// Skewed pinhole camera model.
//
// The pinhole projection with the most general linear (distortion-free)
// calibration matrix, i.e. the PINHOLE model plus a skew parameter:
//
//    K = [ fx   s  cx ]
//        [  0  fy  cy ]
//        [  0   0   1 ]
//
// A non-zero skew arises for cameras synthesized from other sensor models,
// e.g. perspective approximations of the rational polynomial coefficient
// (RPC) models of pushbroom satellite imagery (Zhang et al., "Leveraging
// Vision Reconstruction Pipelines for Satellite Imagery", ICCV Workshops
// 2019), or for sensors with non-orthogonal pixel axes.
//
// The skew is in pixels and forms the extra parameter group, so its
// refinement is controlled by the refine_extra_params options. Because of
// its unit, the extra parameter bound (max_extra_param) is applied to the
// dimensionless shear s / f, with f the mean focal length, and the skew is
// rescaled together with the image width.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, s
//
// See https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
struct SkewedPinholeCameraModel
    : public BasePerspectivePinholeCameraModel<SkewedPinholeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSkewedPinhole,
                                       "SKEWED_PINHOLE",
                                       "fx, fy, cx, cy, s",
                                       2,
                                       2,
                                       1,
                                       true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 0.0};
  }

  // The skew is in pixels, so the bound is applied to the dimensionless shear
  // s / f, with f the mean focal length.
  template <typename T>
  static inline bool HasBogusExtraParams(const std::vector<T>& params,
                                         const T max_extra_param) {
    const T mean_focal_length = (params[0] + params[1]) / T(2);
    return std::abs(params[4]) > max_extra_param * std::abs(mean_focal_length);
  }

  // The skew scales with the image width, like the horizontal focal length.
  static inline void Rescale(double scale_x,
                             double scale_y,
                             std::vector<double>* params) {
    BasePerspectiveCameraModel<SkewedPinholeCameraModel>::Rescale(
        scale_x, scale_y, params);
    (*params)[4] *= scale_x;
  }

  // The skew enters the calibration matrix as K(0, 1).
  static inline Eigen::Matrix3d CalibrationMatrix(
      const std::vector<double>& params) {
    Eigen::Matrix3d K =
        BasePerspectiveCameraModel<SkewedPinholeCameraModel>::CalibrationMatrix(
            params);
    K(0, 1) = params[4];
    return K;
  }

  template <typename T>
  static inline bool ImgFromCam(const T* params,
                                const T& u,
                                const T& v,
                                const T& w,
                                T* x,
                                T* y,
                                const bool check_cheirality = true) {
    if (!HasProjectableDepth(w, check_cheirality)) {
      return false;
    }

    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];
    const T s = params[4];

    // No Distortion

    // Transform to image coordinates
    const T uu = u / w;
    const T vv = v / w;
    *x = f1 * uu + s * vv + c1;
    *y = f2 * vv + c2;

    return true;
  }

  template <typename T>
  static inline bool CamFromImg(
      const T* params, const T& x, const T& y, T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];
    const T s = params[4];

    // v first, since the skew couples u to v.
    *v = (y - c2) / f2;
    *u = (x - c1 - s * *v) / f1;

    return true;
  }
};

}  // namespace colmap
