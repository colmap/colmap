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
struct FOVCameraModel
    : public BasePerspectivePinholeCameraModel<FOVCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kFOV, "FOV", "fx, fy, cx, cy, omega", 2, 2, 1, true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 1e-2};
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

    // Distortion
    Distortion(&params[4], u / w, v / w, x, y);

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double f1 = params[0];
    const double f2 = params[1];
    const double c1 = params[2];
    const double c2 = params[3];

    // Lift points to normalized plane
    const double uu = (x - c1) / f1;
    const double vv = (y - c2) / f2;

    // Undistortion
    Undistortion(&params[4], uu, vv, u, v);

    return true;
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
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
  static inline void Undistortion(
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
};

}  // namespace colmap
