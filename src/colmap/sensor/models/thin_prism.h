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
    : public BasePerspectiveFisheyeCameraModel<ThinPrismFisheyeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kThinPrismFisheye,
      "THIN_PRISM_FISHEYE",
      "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1",
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
};

// RadTanThinPrismFisheye Camera Model
//
// Camera model with radial and tangential distortion coefficients and
// additional coefficients accounting for thin-prism distortion.
//
// See
// https://facebookresearch.github.io/projectaria_tools/docs/tech_insights/camera_intrinsic_models#the-fisheyeradtanthinprism-fisheye624-model
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3
//
struct RadTanThinPrismFisheyeModel
    : public BasePerspectiveFisheyeCameraModel<RadTanThinPrismFisheyeModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kRadTanThinPrismFisheye,
      "RAD_TAN_THIN_PRISM_FISHEYE",
      "fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3",
      2,
      2,
      12,
      true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    std::vector<double> params(RadTanThinPrismFisheyeModel::num_params);
    params[0] = focal_length;
    params[1] = focal_length;
    params[2] = width / 2.0;
    params[3] = height / 2.0;

    for (size_t i = 4; i < RadTanThinPrismFisheyeModel::num_params; ++i) {
      params[i] = 0;
    }

    return params;
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

    T duu, dvv;
    Distortion(&params[4], uu, vv, &duu, &dvv);
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
    constexpr int kNumRadialParams = 6;
    const T radial_coeffs[kNumRadialParams] = {extra_params[0],
                                               extra_params[1],
                                               extra_params[2],
                                               extra_params[3],
                                               extra_params[4],
                                               extra_params[5]};

    const T p0 = extra_params[6];
    const T p1 = extra_params[7];
    const T s0 = extra_params[8];
    const T s1 = extra_params[9];
    const T s2 = extra_params[10];
    const T s3 = extra_params[11];

    const T theta2 = u * u + v * v;
    T th_radial = T(1);
    T theta_power = T(1);
    for (int i = 0; i < kNumRadialParams; ++i) {
      theta_power *= theta2;
      th_radial += radial_coeffs[i] * theta_power;
    }

    const T x = th_radial * u;
    const T y = th_radial * v;

    const T x2 = x * x;
    const T y2 = y * y;
    const T xy = x * y;
    const T r2 = x2 + y2;
    const T r4 = r2 * r2;

    const T dx_tang = T(2) * p1 * xy + p0 * (r2 + T(2) * x2);
    const T dy_tang = T(2) * p0 * xy + p1 * (r2 + T(2) * y2);

    const T dx_tp = s0 * r2 + s1 * r4;
    const T dy_tp = s2 * r2 + s3 * r4;

    const T x_distorted = x + dx_tang + dx_tp;
    const T y_distorted = y + dy_tang + dy_tp;

    *du = x_distorted - u;
    *dv = y_distorted - v;
  }
};

}  // namespace colmap
