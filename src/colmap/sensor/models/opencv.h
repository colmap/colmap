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
struct OpenCVCameraModel
    : public BasePerspectivePinholeCameraModel<OpenCVCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kOpenCV,
                                       "OPENCV",
                                       "fx, fy, cx, cy, k1, k2, p1, p2",
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

    const T uu = u / w;
    const T vv = v / w;

    // Distortion
    T du, dv;
    Distortion(&params[4], uu, vv, &du, &dv);
    *x = uu + du;
    *y = vv + dv;

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
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    return IterativeUndistortion(&params[4], u, v);
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
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
struct FullOpenCVCameraModel
    : public BasePerspectivePinholeCameraModel<FullOpenCVCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kFullOpenCV,
      "FULL_OPENCV",
      "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6",
      2,
      2,
      8,
      true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
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

    const T uu = u / w;
    const T vv = v / w;

    // Distortion
    T du, dv;
    Distortion(&params[4], uu, vv, &du, &dv);
    *x = uu + du;
    *y = vv + dv;

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
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    return IterativeUndistortion(&params[4], u, v);
  }

  template <typename T>
  static inline void Distortion(
      const T* extra_params, const T& u, const T& v, T* du, T* dv) {
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
};

}  // namespace colmap
