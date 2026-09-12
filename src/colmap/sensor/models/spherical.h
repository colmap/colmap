// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/sensor/models/base.h"

namespace colmap {

// Equirectangular (spherical panorama) camera model.
//
// Maps the full 360°x180° sphere onto an equirectangular image: the azimuth
// spans the image width and the elevation spans the image height. The model
// is fully specified by the image dimensions, so the two parameters are the
// width and height; there is no focal length, principal point, or lens
// distortion.
//
// Parameter list is expected in the following order:
//
//    w, h
//
// This is one specific omnidirectional (spherical) projection; see
// IsSpherical() for the camera-model-agnostic category predicate.
struct EquirectangularCameraModel
    : public BaseSphericalCameraModel<EquirectangularCameraModel> {
  SPHERICAL_CAMERA_MODEL_DEFINITIONS(CameraModelId::kEquirectangular,
                                     "EQUIRECTANGULAR",
                                     "w,h",
                                     /*num_metadata_params=*/2,
                                     true)

  static inline std::vector<double> InitializeParams(
      const double /*focal_length*/, const size_t width, const size_t height) {
    return {static_cast<double>(width), static_cast<double>(height)};
  }

  // Projects camera-frame point (u, v, w) onto the equirectangular image plane.
  // Unlike pinhole/fisheye models that require w > 0, EQUIRECTANGULAR accepts
  // any non-zero direction — all 4π of the sphere are representable.
  template <typename T>
  static inline bool ImgFromCam(const T* params,
                                const T& u,
                                const T& v,
                                const T& w,
                                T* x,
                                T* y,
                                const bool /*check_cheirality*/ = true) {
    const T width = params[0];
    const T height = params[1];

    const T horizontal = ceres::sqrt(u * u + w * w);
    // Degenerate: zero direction vector.
    if (horizontal + ceres::abs(v) <
        T(std::numeric_limits<double>::epsilon())) {
      return false;
    }

    // Azimuth θ ∈ (-π, π], measured from +Z axis (forward). +X is θ = +π/2.
    const T theta = ceres::atan2(u, w);
    // Elevation φ ∈ [-π/2, π/2], measured from the equator. -Y (up) is +π/2.
    const T phi = ceres::atan2(-v, horizontal);

    *x = (theta / T(2.0 * EIGEN_PI) + T(0.5)) * width;
    *y = (T(0.5) - phi / T(EIGEN_PI)) * height;
    return true;
  }

  // Inverse equirectangular projection. Returns the normalized camera
  // coordinates (u = X/Z, v = Y/Z) of the pixel's ray, valid only when the ray
  // falls in the forward hemisphere (Z > 0). Back-hemisphere pixels return
  // false; use CamRayFromImg for the full-sphere 3D bearing.
  static inline bool CamFromImg(
      const double* params, double x, double y, double* u, double* v) {
    const double width = params[0];
    const double height = params[1];

    const double theta = 2.0 * EIGEN_PI * (x / width - 0.5);
    const double phi = EIGEN_PI * (0.5 - y / height);

    const double cos_phi = std::cos(phi);
    const double rx = cos_phi * std::sin(theta);
    const double ry = -std::sin(phi);
    const double rz = cos_phi * std::cos(theta);

    if (rz <= std::numeric_limits<double>::epsilon()) {
      return false;
    }

    *u = rx / rz;
    *v = ry / rz;
    return true;
  }

  template <typename T>
  static inline bool HasBogusParams(const std::vector<T>& /*params*/,
                                    size_t /*width*/,
                                    size_t /*height*/,
                                    T /*min_focal_length_ratio*/,
                                    T /*max_focal_length_ratio*/,
                                    T /*max_extra_param*/) {
    return false;
  }

  // EQUIRECTANGULAR has no focal length, so it cannot use the perspective
  // base's focal-length-based threshold. Convert pixel thresholds to
  // normalized camera-coordinate thresholds using the angular resolution at the
  // equator (2π rad per W pixels in azimuth).
  template <typename T>
  static inline T CamFromImgThreshold(const T* params, T threshold) {
    return threshold * T(2.0 * EIGEN_PI) / params[0];
  }

  // The base's default CamRayFromImg goes through the 2D CamFromImg which fails
  // for back-hemisphere pixels. EQUIRECTANGULAR can produce valid unit bearings
  // for any pixel in the equirectangular image, so we compute the ray directly
  // from the azimuth/elevation parametrization.
  static inline bool CamRayFromImg(const double* params,
                                   double x,
                                   double y,
                                   double* rx,
                                   double* ry,
                                   double* rz) {
    const double width = params[0];
    const double height = params[1];
    const double theta = 2.0 * EIGEN_PI * (x / width - 0.5);
    const double phi = EIGEN_PI * (0.5 - y / height);
    const double cos_phi = std::cos(phi);
    *rx = cos_phi * std::sin(theta);
    *ry = -std::sin(phi);
    *rz = cos_phi * std::cos(theta);
    return true;
  }
};

}  // namespace colmap
