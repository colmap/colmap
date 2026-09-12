// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/models/base.h"

namespace colmap {

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
    : public BasePerspectivePinholeCameraModel<SimpleRadialCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSimpleRadial,
                                       "SIMPLE_RADIAL",
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

    const T uu = u / w;
    const T vv = v / w;

    // Distortion
    T du, dv;
    Distortion(&params[3], uu, vv, &du, &dv);
    *x = uu + du;
    *y = vv + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double f = params[0];
    const double c1 = params[1];
    const double c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    return IterativeUndistortion(&params[3], u, v);
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
    const T k = extra_params[0];

    const T u2 = u * u;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k * r2;
    *du = u * radial;
    *dv = v * radial;
  }
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
struct RadialCameraModel
    : public BasePerspectivePinholeCameraModel<RadialCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kRadial, "RADIAL", "f, cx, cy, k1, k2", 1, 2, 2, true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0, 0};
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

    const T uu = u / w;
    const T vv = v / w;

    // Distortion
    T du, dv;
    Distortion(&params[3], uu, vv, &du, &dv);
    *x = uu + du;
    *y = vv + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double f = params[0];
    const double c1 = params[1];
    const double c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    return IterativeUndistortion(&params[3], u, v);
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];

    const T u2 = u * u;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    *du = u * radial;
    *dv = v * radial;
  }
};

}  // namespace colmap
