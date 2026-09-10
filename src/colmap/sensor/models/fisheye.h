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

// Simple equidistant fisheye camera model.
//
// Uses the equidistant (theta = r) fisheye projection without distortion
// parameters. Suitable for fish-eye cameras where distortion can be ignored
// or has been pre-corrected. This model has a single focal length.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy
//
struct SimpleFisheyeCameraModel
    : public BasePerspectiveFisheyeCameraModel<SimpleFisheyeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kSimpleFisheye,
                                       "SIMPLE_FISHEYE",
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

    // No distortion

    // Transform to image coordinates
    ImgFromFisheye(params, uu, vv, x, y);

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    double uu, vv;
    FisheyeFromImg(params, x, y, &uu, &vv);
    // No undistortion needed
    NormalFromFisheye(uu, vv, u, v);
    return true;
  }
};

// Equidistant fisheye camera model.
//
// Uses the equidistant (theta = r) fisheye projection without distortion
// parameters. Suitable for fish-eye cameras where distortion can be ignored
// or has been pre-corrected. This model has two focal lengths (fx, fy).
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy
//
struct FisheyeCameraModel
    : public BasePerspectiveFisheyeCameraModel<FisheyeCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(
      CameraModelId::kFisheye, "FISHEYE", "fx, fy, cx, cy", 2, 2, 0, true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0};
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

    // No distortion

    // Transform to image coordinates
    ImgFromFisheye(params, uu, vv, x, y);

    return true;
  }

  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    double uu, vv;
    FisheyeFromImg(params, x, y, &uu, &vv);
    // No undistortion needed
    NormalFromFisheye(uu, vv, u, v);
    return true;
  }
};

}  // namespace colmap
