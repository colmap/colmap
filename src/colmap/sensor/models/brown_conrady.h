// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/models/base.h"

namespace colmap {

// Brown-Conrady camera model with a single focal length.
//
// Based on the pinhole camera model. Models radial distortion with three
// coefficients and tangential distortion with two coefficients, following
// Brown ("Close-range camera calibration", 1971) and Conrady (1919).
//
// This is the single-focal-length counterpart to the OPENCV model extended
// with a third radial coefficient: OPENCV models
// "fx, fy, cx, cy, k1, k2, p1, p2" while this model shares one focal length
// and adds k3. It generalizes SIMPLE_RADIAL ("f, cx, cy, k") and RADIAL
// ("f, cx, cy, k1, k2") with an additional radial term and tangential terms.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k1, k2, k3, p1, p2
//
struct BrownConradyCameraModel
    : public BasePerspectivePinholeCameraModel<BrownConradyCameraModel> {
  PERSPECTIVE_CAMERA_MODEL_DEFINITIONS(CameraModelId::kBrownConrady,
                                       "BROWN_CONRADY",
                                       "f, cx, cy, k1, k2, k3, p1, p2",
                                       1,
                                       2,
                                       5,
                                       true)

  static inline std::vector<double> InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0, 0};
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

  template <typename T>
  static inline bool CamFromImg(
      const T* params, const T& x, const T& y, T* u, T* v) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

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
    const T k3 = extra_params[2];
    const T p1 = extra_params[3];
    const T p2 = extra_params[4];

    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T radial = k1 * r2 + k2 * r4 + k3 * r6;
    *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
  }
};

}  // namespace colmap
