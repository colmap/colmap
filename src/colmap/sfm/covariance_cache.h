// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/sim3.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/types.h"

#include <optional>

#include <Eigen/Core>

namespace colmap {

// In-memory cache for derived pose and point covariances during incremental
// mapping. 2D measurement covariances live on `Point2D` itself; this cache
// holds quantities that change with estimation: pose covariances (from pose
// refinement or local bundle adjustment marginals) and point covariances
// (from triangulation). Entries are invalidated explicitly by the owner when
// the underlying estimates change. Missing entries mean "unknown", never
// "exact".
class MapperCovarianceCache {
 public:
  // Ceres tangent-space 6x6 pose covariance in [rotation, translation] order.
  void SetPoseCov(image_t image_id, const Eigen::Matrix6d& cov);
  std::optional<Eigen::Matrix6d> PoseCov(image_t image_id) const;
  // Drop a pose covariance, e.g. when its image was de-registered. No-op if
  // missing.
  void ErasePoseCov(image_t image_id);

  // 3x3 point covariance in world coordinates.
  void SetPointCov(point3D_t point3D_id, const Eigen::Matrix3d& cov);
  std::optional<Eigen::Matrix3d> PointCov(point3D_t point3D_id) const;
  // Drop a point covariance, e.g. when its track changed. No-op if missing.
  void ErasePointCov(point3D_t point3D_id);

  // Scale of the true over the modeled measurement variances, e.g., as
  // calibrated from bundle adjustment residuals. The cached covariances are
  // in units of the modeled measurement noise, i.e., their true counterparts
  // are scaled by this factor. Defaults to 1 and is not reset by Clear().
  void SetMeasurementVarianceScale(double scale);
  double MeasurementVarianceScale() const;

  // Drop all entries.
  void Clear();

  // Drop all point covariances while preserving pose covariances. This is used
  // after track-only mutations such as adding or filtering observations.
  void ClearPointCovariances();

  // Propagate all entries through a similarity transform of the world frame.
  // Must be called after `reconstruction` was transformed, as the pose
  // Jacobian depends on the transformed camera poses. Pose entries of images
  // without a pose in `reconstruction` are dropped.
  void Transform(const Sim3d& new_from_old_world,
                 const Reconstruction& reconstruction);

 private:
  FlatHashMap<image_t, Eigen::Matrix6d> pose_covs_;
  FlatHashMap<point3D_t, Eigen::Matrix3d> point_covs_;
  double measurement_variance_scale_ = 1;
};

}  // namespace colmap
