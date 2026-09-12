// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/models/base.h"

namespace colmap {

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
    : public BasePerspectiveFisheyeCameraModel<OpenCVFisheyeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kOpenCVFisheye,
                                       "OPENCV_FISHEYE",
                                       "fx, fy, cx, cy, k1, k2, k3, k4",
                                       2,
                                       2,
                                       4,
                                       true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0};
  }

  template <typename T>
  static inline void ImgFromFisheye(
      const T* params, const T& uu, const T& vv, T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    *x = f1 * uu + c1;
    *y = f2 * vv + c2;
  }

  template <typename T>
  static inline void FisheyeFromImg(const T* params, T x, T y, T* uu, T* vv) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    *uu = (x - c1) / f1;
    *vv = (y - c2) / f2;
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

    T uu, vv;
    FisheyeFromNormal(u / w, v / w, &uu, &vv);

    // Distortion
    T duu, dvv;
    Distortion(&params[4], uu, vv, &duu, &dvv);

    // Transform to image coordinates
    ImgFromFisheye(params, uu + duu, vv + dvv, x, y);

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    double uu, vv;
    FisheyeFromImg(params, x, y, &uu, &vv);
    if (!IterativeUndistortion(&params[4], &uu, &vv)) {
      return false;
    }
    NormalFromFisheye(uu, vv, u, v);
    return true;
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T k3 = extra_params[2];
    const T k4 = extra_params[3];

    const T theta2 = u * u + v * v;
    const T theta4 = theta2 * theta2;
    const T theta6 = theta4 * theta2;
    const T theta8 = theta4 * theta4;
    const T radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
    *du = u * radial;
    *dv = v * radial;
  }
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
    : public BasePerspectiveFisheyeCameraModel<SimpleRadialFisheyeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSimpleRadialFisheye,
                                       "SIMPLE_RADIAL_FISHEYE",
                                       "f, cx, cy, k",
                                       1,
                                       2,
                                       1,
                                       true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0};
  }

  template <typename T>
  static inline void ImgFromFisheye(
      const T* params, const T& uu, const T& vv, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    *x = f * uu + c1;
    *y = f * vv + c2;
  }

  template <typename T>
  static inline void FisheyeFromImg(const T* params, T x, T y, T* uu, T* vv) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    *uu = (x - c1) / f;
    *vv = (y - c2) / f;
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

    T uu, vv;
    FisheyeFromNormal(u / w, v / w, &uu, &vv);

    // Distortion
    T duu, dvv;
    Distortion(&params[3], uu, vv, &duu, &dvv);

    // Transform to image coordinates
    ImgFromFisheye(params, uu + duu, vv + dvv, x, y);

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    double uu, vv;
    FisheyeFromImg(params, x, y, &uu, &vv);
    if (!IterativeUndistortion(&params[3], &uu, &vv)) {
      return false;
    }
    NormalFromFisheye(uu, vv, u, v);
    return true;
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
    const T k = extra_params[0];

    const T theta2 = u * u + v * v;
    const T radial = k * theta2;
    *du = u * radial;
    *dv = v * radial;
  }
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
    : public BasePerspectiveFisheyeCameraModel<RadialFisheyeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kRadialFisheye,
                                       "RADIAL_FISHEYE",
                                       "f, cx, cy, k1, k2",
                                       1,
                                       2,
                                       2,
                                       true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0, 0};
  }

  template <typename T>
  static inline void ImgFromFisheye(
      const T* params, const T& uu, const T& vv, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    *x = f * uu + c1;
    *y = f * vv + c2;
  }

  template <typename T>
  static inline void FisheyeFromImg(const T* params, T x, T y, T* uu, T* vv) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    *uu = (x - c1) / f;
    *vv = (y - c2) / f;
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

    T uu, vv;
    FisheyeFromNormal(u / w, v / w, &uu, &vv);

    // Distortion
    T duu, dvv;
    Distortion(&params[3], uu, vv, &duu, &dvv);

    // Transform to image coordinates
    ImgFromFisheye(params, uu + duu, vv + dvv, x, y);

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    double uu, vv;
    FisheyeFromImg(params, x, y, &uu, &vv);
    if (!IterativeUndistortion(&params[3], &uu, &vv)) {
      return false;
    }
    NormalFromFisheye(uu, vv, u, v);
    return true;
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];

    const T theta2 = u * u + v * v;
    const T theta4 = theta2 * theta2;
    const T radial = k1 * theta2 + k2 * theta4;
    *du = u * radial;
    *dv = v * radial;
  }
};

}  // namespace colmap
