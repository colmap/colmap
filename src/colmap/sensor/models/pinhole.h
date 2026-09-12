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

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double f = params[0];
    const double c1 = params[1];
    const double c2 = params[2];

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

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double f1 = params[0];
    const double f2 = params[1];
    const double c1 = params[2];
    const double c2 = params[3];

    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    return true;
  }
};

}  // namespace colmap
