// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sfm/covariance_cache.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/synthetic.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(MapperCovarianceCache, InitiallyEmpty) {
  MapperCovarianceCache cache;
  EXPECT_FALSE(cache.PoseCov(1).has_value());
  EXPECT_FALSE(cache.PointCov(1).has_value());
}

TEST(MapperCovarianceCache, SetAndGetRoundTrip) {
  MapperCovarianceCache cache;
  const Eigen::Matrix6d pose_cov = 2 * Eigen::Matrix6d::Identity();
  const Eigen::Matrix3d point_cov = 3 * Eigen::Matrix3d::Identity();
  cache.SetPoseCov(7, pose_cov);
  cache.SetPointCov(9, point_cov);
  EXPECT_EQ(cache.PoseCov(7), pose_cov);
  EXPECT_EQ(cache.PointCov(9), point_cov);
  EXPECT_FALSE(cache.PoseCov(8).has_value());
  EXPECT_FALSE(cache.PointCov(10).has_value());
}

TEST(MapperCovarianceCache, Overwrite) {
  MapperCovarianceCache cache;
  cache.SetPoseCov(7, Eigen::Matrix6d::Identity());
  const Eigen::Matrix6d updated = 5 * Eigen::Matrix6d::Identity();
  cache.SetPoseCov(7, updated);
  EXPECT_EQ(cache.PoseCov(7), updated);
}

TEST(MapperCovarianceCache, ErasePointCov) {
  MapperCovarianceCache cache;
  cache.SetPointCov(9, Eigen::Matrix3d::Identity());
  cache.ErasePointCov(9);
  EXPECT_FALSE(cache.PointCov(9).has_value());
  // No-op if missing.
  cache.ErasePointCov(9);
  EXPECT_FALSE(cache.PointCov(9).has_value());
}

TEST(MapperCovarianceCache, ErasePoseCov) {
  MapperCovarianceCache cache;
  cache.SetPoseCov(7, Eigen::Matrix6d::Identity());
  cache.SetPointCov(9, Eigen::Matrix3d::Identity());
  cache.ErasePoseCov(7);
  EXPECT_FALSE(cache.PoseCov(7).has_value());
  EXPECT_TRUE(cache.PointCov(9).has_value());
  // No-op if missing.
  cache.ErasePoseCov(7);
  EXPECT_FALSE(cache.PoseCov(7).has_value());
}

TEST(MapperCovarianceCache, ClearInvalidates) {
  MapperCovarianceCache cache;
  cache.SetPoseCov(7, Eigen::Matrix6d::Identity());
  cache.SetPointCov(9, Eigen::Matrix3d::Identity());
  cache.Clear();
  EXPECT_FALSE(cache.PoseCov(7).has_value());
  EXPECT_FALSE(cache.PointCov(9).has_value());
}

TEST(MapperCovarianceCache, MeasurementVarianceScale) {
  MapperCovarianceCache cache;
  EXPECT_EQ(cache.MeasurementVarianceScale(), 1);
  cache.SetMeasurementVarianceScale(0.04);
  EXPECT_EQ(cache.MeasurementVarianceScale(), 0.04);
  cache.Clear();
  EXPECT_EQ(cache.MeasurementVarianceScale(), 0.04);
  EXPECT_ANY_THROW(cache.SetMeasurementVarianceScale(0));
}

TEST(MapperCovarianceCache, ClearPointCovariancesPreservesPoses) {
  MapperCovarianceCache cache;
  cache.SetPoseCov(7, Eigen::Matrix6d::Identity());
  cache.SetPointCov(9, Eigen::Matrix3d::Identity());
  cache.ClearPointCovariances();
  EXPECT_TRUE(cache.PoseCov(7).has_value());
  EXPECT_FALSE(cache.PointCov(9).has_value());
}

// Applies a tangent perturbation [dr, dt] with Ceres' left-multiplicative
// quaternion retraction.
Rigid3d PerturbPose(const Rigid3d& pose, const Eigen::Vector6d& delta) {
  const Eigen::Vector3d dr = delta.head<3>();
  const double norm = dr.norm();
  Eigen::Quaterniond dq(std::cos(norm), 0, 0, 0);
  dq.vec() = norm > 0 ? (std::sin(norm) / norm * dr).eval() : dr;
  return Rigid3d((dq * pose.rotation()).normalized(),
                 pose.translation() + delta.tail<3>());
}

TEST(MapperCovarianceCache, Transform) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 5;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const Sim3d new_from_old_world(
      2.5,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized())),
      Eigen::Vector3d(0.3, -1.2, 4.0));

  // Use rank-1 covariances v * v^T, such that the transformed covariance
  // must equal w * w^T, where w is the transformed small perturbation v.
  const image_t image_id = reconstruction.RegImageIds().front();
  const point3D_t point3D_id = *reconstruction.Point3DIds().begin();
  Eigen::Vector6d pose_delta;
  pose_delta << 1, -2, 3, -4, 5, -6;
  pose_delta *= 1e-7;
  const Eigen::Vector3d point_delta = 1e-7 * Eigen::Vector3d(1, -2, 3);

  MapperCovarianceCache cache;
  cache.SetPoseCov(image_id, pose_delta * pose_delta.transpose());
  cache.SetPointCov(point3D_id, point_delta * point_delta.transpose());
  cache.SetPoseCov(kInvalidImageId - 1, Eigen::Matrix6d::Identity());

  const Rigid3d old_cam_from_world =
      reconstruction.Image(image_id).CamFromWorld();
  const Eigen::Vector3d old_xyz = reconstruction.Point3D(point3D_id).xyz;
  reconstruction.Transform(new_from_old_world);
  cache.Transform(new_from_old_world, reconstruction);

  const Rigid3d new_cam_from_world =
      reconstruction.Image(image_id).CamFromWorld();
  const Rigid3d new_perturbed_cam_from_world = TransformCameraWorld(
      new_from_old_world, PerturbPose(old_cam_from_world, pose_delta));
  Eigen::Vector6d new_pose_delta;
  new_pose_delta.head<3>() = (new_perturbed_cam_from_world.rotation() *
                              new_cam_from_world.rotation().inverse())
                                 .vec();
  new_pose_delta.tail<3>() = new_perturbed_cam_from_world.translation() -
                             new_cam_from_world.translation();
  const Eigen::Matrix6d expected_pose_cov =
      new_pose_delta * new_pose_delta.transpose();
  ASSERT_TRUE(cache.PoseCov(image_id).has_value());
  EXPECT_LT((*cache.PoseCov(image_id) - expected_pose_cov).norm(),
            1e-5 * expected_pose_cov.norm());

  const Eigen::Vector3d new_point_delta =
      new_from_old_world * (old_xyz + point_delta) -
      reconstruction.Point3D(point3D_id).xyz;
  const Eigen::Matrix3d expected_point_cov =
      new_point_delta * new_point_delta.transpose();
  ASSERT_TRUE(cache.PointCov(point3D_id).has_value());
  EXPECT_LT((*cache.PointCov(point3D_id) - expected_point_cov).norm(),
            1e-5 * expected_point_cov.norm());

  // Entries of unknown images are dropped.
  EXPECT_FALSE(cache.PoseCov(kInvalidImageId - 1).has_value());
}

}  // namespace
}  // namespace colmap
