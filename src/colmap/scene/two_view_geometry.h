// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"

#include <optional>

namespace colmap {

// Two-view geometry.
struct TwoViewGeometry {
  // The configuration of the two-view geometry.
  enum ConfigurationType {
    UNDEFINED = 0,
    // Degenerate configuration (e.g., no overlap or not enough inliers).
    DEGENERATE = 1,
    // Essential matrix.
    CALIBRATED = 2,
    // Relative pose (metric) from calibrated (non-panoramic) rig.
    CALIBRATED_RIG = 9,
    // Fundamental matrix. A pair whose focal length(s) were recovered by the
    // two-view solver (e.g. the shared-focal essential-matrix solver) is also
    // UNCALIBRATED: it carries F built from the estimated focal, and exposes
    // the estimated intrinsics via `camera1`/`camera2` (see below) so consumers
    // can seed them without trusting the camera's placeholder focal.
    UNCALIBRATED = 3,
    // Homography, planar scene with baseline.
    PLANAR = 4,
    // Homography, pure rotation without baseline.
    PANORAMIC = 5,
    // Homography, planar or panoramic.
    PLANAR_OR_PANORAMIC = 6,
    // Watermark, pure 2D translation in image borders.
    WATERMARK = 7,
    // Multi-model configuration, i.e. the inlier matches result from multiple
    // individual, non-degenerate configurations.
    MULTIPLE = 8,
  };

  // Defaulted but defined out-of-line to avoid a spurious GCC -Wuninitialized
  // from inlined moves of the std::optional<Camera> members.
  TwoViewGeometry() = default;
  TwoViewGeometry(const TwoViewGeometry&);
  TwoViewGeometry(TwoViewGeometry&&) noexcept;
  TwoViewGeometry& operator=(const TwoViewGeometry&);
  TwoViewGeometry& operator=(TwoViewGeometry&&) noexcept;
  ~TwoViewGeometry();

  // One of `ConfigurationType`.
  int config = ConfigurationType::UNDEFINED;

  // Essential matrix.
  std::optional<Eigen::Matrix3d> E;
  // Fundamental matrix.
  std::optional<Eigen::Matrix3d> F;
  // Homography matrix.
  std::optional<Eigen::Matrix3d> H;

  // Relative pose.
  std::optional<Rigid3d> cam2_from_cam1;

  // Per-side intrinsics recovered by the two-view solver: `cameraN` holds side
  // N's estimated intrinsics, or nullopt if that side was not estimated.
  std::optional<Camera> camera1;
  std::optional<Camera> camera2;

  // Inlier matches of the configuration.
  FeatureMatches inlier_matches;

  // Median triangulation angle.
  double tri_angle = -1;

  // Invert the geometry to match swapped cameras.
  void Invert();
};

}  // namespace colmap
