// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sfm/covariance_cache.h"

#include "colmap/geometry/rigid3.h"

namespace colmap {

void MapperCovarianceCache::SetPoseCov(image_t image_id,
                                       const Eigen::Matrix6d& cov) {
  pose_covs_[image_id] = cov;
}

std::optional<Eigen::Matrix6d> MapperCovarianceCache::PoseCov(
    image_t image_id) const {
  const auto it = pose_covs_.find(image_id);
  if (it == pose_covs_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void MapperCovarianceCache::ErasePoseCov(image_t image_id) {
  pose_covs_.erase(image_id);
}

void MapperCovarianceCache::SetPointCov(point3D_t point3D_id,
                                        const Eigen::Matrix3d& cov) {
  point_covs_[point3D_id] = cov;
}

std::optional<Eigen::Matrix3d> MapperCovarianceCache::PointCov(
    point3D_t point3D_id) const {
  const auto it = point_covs_.find(point3D_id);
  if (it == point_covs_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void MapperCovarianceCache::ErasePointCov(point3D_t point3D_id) {
  point_covs_.erase(point3D_id);
}

void MapperCovarianceCache::SetMeasurementVarianceScale(double scale) {
  THROW_CHECK_GT(scale, 0);
  measurement_variance_scale_ = scale;
}

double MapperCovarianceCache::MeasurementVarianceScale() const {
  return measurement_variance_scale_;
}

void MapperCovarianceCache::Clear() {
  pose_covs_.clear();
  point_covs_.clear();
}

void MapperCovarianceCache::ClearPointCovariances() { point_covs_.clear(); }

void MapperCovarianceCache::Transform(const Sim3d& new_from_old_world,
                                      const Reconstruction& reconstruction) {
  const double scale = new_from_old_world.scale();
  const Eigen::Matrix3d rotation =
      new_from_old_world.rotation().toRotationMatrix();

  // With the transformed pose R' = R * Rs^T and t' = s * t - R' * ts, and the
  // left-multiplicative quaternion retraction R = Exp(2 * dr) * R0 (see
  // PropagatePoseCovarianceToImage), the tangent perturbations map as
  // dr' = dr and dt' = s * dt + 2 * [R' * ts]x * dr.
  std::vector<image_t> invalid_image_ids;
  for (auto& [image_id, pose_cov] : pose_covs_) {
    if (!reconstruction.ExistsImage(image_id) ||
        !reconstruction.Image(image_id).HasPose()) {
      invalid_image_ids.push_back(image_id);
      continue;
    }
    const Rigid3d cam_from_world = reconstruction.Image(image_id).CamFromWorld();
    Eigen::Matrix6d J = Eigen::Matrix6d::Identity();
    J.block<3, 3>(3, 0) = 2.0 * CrossProductMatrix(cam_from_world.rotation() *
                                                   new_from_old_world.translation());
    J.block<3, 3>(3, 3) *= scale;
    pose_cov = J * pose_cov * J.transpose();
  }
  for (const image_t image_id : invalid_image_ids) {
    pose_covs_.erase(image_id);
  }

  const Eigen::Matrix3d point_jacobian = scale * rotation;
  for (auto& [_, point_cov] : point_covs_) {
    point_cov = point_jacobian * point_cov * point_jacobian.transpose();
  }
}

}  // namespace colmap
