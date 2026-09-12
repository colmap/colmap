// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/models/base.h"

namespace colmap {

// EUCM camera model
//
// This camera model is described in
//
//      "An Enhanced Unified Camera Model",
//      Bogdan Khomutenko, Gaetan Garcia, Philippe Martinet,  2018
//
//   Parameter list is expected in the following order:
//
//      fx, fy, cx, cy, alpha, beta
//
struct EUCMCameraModel
    : public BasePerspectivePinholeCameraModel<EUCMCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kEUCM,
                                       "EUCM",
                                       "fx, fy, cx, cy, alpha, beta",
                                       2,
                                       2,
                                       2,
                                       true)

  template <typename T>
  static inline bool HasBogusExtraParams(const std::vector<T>& params,
                                         const T max_extra_param) {
    if (BasePerspectiveCameraModel<EUCMCameraModel>::HasBogusExtraParams(
            params, max_extra_param)) {
      return true;
    }

    const T alpha = params[4];
    const T beta = params[5];
    return alpha < T(0) || alpha > T(1) || beta <= T(0);
  }

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 0.0, 1.0};
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

    const T alpha = params[4];
    const T beta = params[5];

    const T rho2 = beta * (u * u + v * v) + w * w;
    if (rho2 < T(0)) {
      return false;
    }
    const T rho = ceres::sqrt(rho2);
    const T den = alpha * rho + (1.0 - alpha) * w;
    if (!HasProjectableDepth(den, check_cheirality)) {
      return false;
    }
    *x = u / den;
    *y = v / den;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;

    return true;
  }

  static inline bool CamFromImg(const double* params,
                                const double x,
                                const double y,
                                double* u,
                                double* v) {
    const double f1 = params[0];
    const double f2 = params[1];
    const double c1 = params[2];
    const double c2 = params[3];

    const double alpha = params[4];
    const double beta = params[5];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    const double r2 = *u * *u + *v * *v;
    const double gamma = 1.0 - alpha;
    const double radicand = 1.0 - (alpha - gamma) * beta * r2;
    if (radicand < 0) {
      return false;
    }
    const double helper_den = alpha * std::sqrt(radicand) + gamma;
    if (helper_den < std::numeric_limits<double>::epsilon()) {
      return false;
    }
    const double helper = (1.0 - alpha * alpha * beta * r2) / helper_den;
    if (helper < std::numeric_limits<double>::epsilon()) {
      return false;
    }

    *u /= helper;
    *v /= helper;

    return true;
  }
};

}  // namespace colmap
