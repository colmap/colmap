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

#include "colmap/sensor/models/base.h"

namespace colmap {

// Simple Division camera model.
//
// One parameter division model from Fitzgibbon with a single focal length.
// This model has a closed-form projection and unprojection.
//
// See: "Simultaneous linear estimation of multiple view geometry and lens
// distortion" by A. Fitzgibbon 2001.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k
//
struct SimpleDivisionCameraModel
    : public BasePerspectivePinholeCameraModel<SimpleDivisionCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSimpleDivision,
                                       "SIMPLE_DIVISION",
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
                                const bool /*check_cheirality*/ = true) {
    // Division model projection:
    // (xp, 1+k*|xp|^2) ~= (x(1:2), x3)
    // Solving the quadratic: rho*k*r2 - x3 * r + rho = 0

    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];
    const T k = params[3];

    const T rho = ceres::sqrt(u * u + v * v);
    const T disc_sq = w * w - T(4) * rho * rho * k;

    if (disc_sq < T(0)) {
      return false;
    }

    const T disc = ceres::sqrt(disc_sq);
    const T r = T(2) / (w + disc);

    *x = f * r * u + c1;
    *y = f * r * v + c2;

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double f = params[0];
    const double c1 = params[1];
    const double c2 = params[2];
    const double k = params[3];

    // Lift to normalized coordinates
    const double x0 = (x - c1) / f;
    const double y0 = (y - c2) / f;
    const double r2 = x0 * x0 + y0 * y0;

    // Closed-form unprojection for division model
    const double denom = 1.0 + k * r2;
    *u = x0 / denom;
    *v = y0 / denom;

    return true;
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
    // The division model doesn't use standard additive distortion,
    // but we need this for compatibility with the iterative undistortion.
    const T k = extra_params[0];
    const T r2 = u * u + v * v;
    const T factor = k * r2 / (T(1) + k * r2);
    *du = -u * factor;
    *dv = -v * factor;
  }
};

// Division camera model.
//
// One parameter division model from Fitzgibbon with separate fx/fy focal
// lengths. This model has a closed-form projection and unprojection.
//
// See: "Simultaneous linear estimation of multiple view geometry and lens
// distortion" by A. Fitzgibbon 2001.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k
//
struct DivisionCameraModel
    : public BasePerspectivePinholeCameraModel<DivisionCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kDivision, "DIVISION", "fx, fy, cx, cy, k", 2, 2, 1, true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 0};
  }

  template <typename T>
  static inline bool ImgFromCam(const T* params,
                                const T& u,
                                const T& v,
                                const T& w,
                                T* x,
                                T* y,
                                const bool /*check_cheirality*/ = true) {
    // Division model projection:
    // (xp, 1+k*|xp|^2) ~= (x(1:2), x3)
    // Solving the quadratic: rho*k*r2 - x3 * r + rho = 0

    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];
    const T k = params[4];

    const T rho = ceres::sqrt(u * u + v * v);
    const T disc_sq = w * w - T(4) * rho * rho * k;

    if (disc_sq < T(0)) {
      return false;
    }

    const T disc = ceres::sqrt(disc_sq);
    const T r = T(2) / (w + disc);

    *x = f1 * r * u + c1;
    *y = f2 * r * v + c2;

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double f1 = params[0];
    const double f2 = params[1];
    const double c1 = params[2];
    const double c2 = params[3];
    const double k = params[4];

    // Lift to normalized coordinates
    const double x0 = (x - c1) / f1;
    const double y0 = (y - c2) / f2;
    const double r2 = x0 * x0 + y0 * y0;

    // Closed-form unprojection for division model
    const double denom = 1.0 + k * r2;
    *u = x0 / denom;
    *v = y0 / denom;

    return true;
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
    // The division model doesn't use standard additive distortion,
    // but we need this for compatibility with the iterative undistortion.
    const T k = extra_params[0];
    const T r2 = u * u + v * v;
    const T factor = k * r2 / (T(1) + k * r2);
    *du = -u * factor;
    *dv = -v * factor;
  }
};

}  // namespace colmap
