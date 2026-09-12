// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/pose_prior.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PosePrior, Equals) {
  PosePrior prior;
  prior.pose_prior_id = 0;
  prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 2);
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  prior.gravity = Eigen::Vector3d::UnitZ();
  PosePrior other = prior;
  EXPECT_EQ(prior, other);
  prior.position.x() = 1;
  EXPECT_NE(prior, other);
  other.position.x() = 1;
  EXPECT_EQ(prior, other);
}

TEST(PosePrior, NaNEquals) {
  PosePrior prior;
  PosePrior other = prior;
  EXPECT_EQ(prior, other);
  prior.position =
      Eigen::Vector3d(1, 2, std::numeric_limits<double>::quiet_NaN());
  other.position =
      Eigen::Vector3d(1, 2, std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(prior, other);
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.position_covariance(0, 0) = std::numeric_limits<double>::quiet_NaN();
  other.position_covariance = Eigen::Matrix3d::Identity();
  other.position_covariance(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(prior, other);
  other.position.z() = 3;
  EXPECT_NE(prior, other);
}

TEST(PosePrior, Print) {
  PosePrior prior;
  prior.pose_prior_id = 0;
  prior.corr_data_id = data_t(sensor_t(SensorType::CAMERA, 1), 2);
  prior.position = Eigen::Vector3d::Zero();
  prior.position_covariance = Eigen::Matrix3d::Identity();
  prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  prior.gravity = Eigen::Vector3d::UnitZ();
  std::ostringstream stream;
  stream << prior;
  EXPECT_EQ(stream.str(),
            "PosePrior(pose_prior_id=0, corr_data_id=(CAMERA, 1, 2), "
            "position=[0, 0, 0], "
            "position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1], "
            "coordinate_system=CARTESIAN, gravity=[0, 0, 1])");
}

TEST(PosePrior, GravityFromExifOrientation) {
  EXPECT_EQ(GravityFromExifOrientation(1).value(), Eigen::Vector3d(0, 1, 0));
  EXPECT_EQ(GravityFromExifOrientation(3).value(), Eigen::Vector3d(0, -1, 0));
  EXPECT_EQ(GravityFromExifOrientation(6).value(), Eigen::Vector3d(1, 0, 0));
  EXPECT_EQ(GravityFromExifOrientation(8).value(), Eigen::Vector3d(-1, 0, 0));
  EXPECT_FALSE(GravityFromExifOrientation(2).has_value());
  EXPECT_FALSE(GravityFromExifOrientation(4).has_value());
  EXPECT_FALSE(GravityFromExifOrientation(5).has_value());
  EXPECT_FALSE(GravityFromExifOrientation(7).has_value());
  EXPECT_FALSE(GravityFromExifOrientation(0).has_value());
  EXPECT_FALSE(GravityFromExifOrientation(42).has_value());
  EXPECT_FALSE(GravityFromExifOrientation(-1).has_value());
}

TEST(PosePrior, ComputeRot90FromGravity) {
  Eigen::Vector3d gravity(0, 1, 0);  // Normal
  EXPECT_EQ(ComputeRot90FromGravity(gravity), 0);

  // Gravity is +x (right). Need 90 CW (270 CCW) to make upright.
  gravity = Eigen::Vector3d(1, 0, 0);
  EXPECT_EQ(ComputeRot90FromGravity(gravity), 3);

  // Gravity is -y (up). Need 180 CCW to make upright.
  gravity = Eigen::Vector3d(0, -1, 0);
  EXPECT_EQ(ComputeRot90FromGravity(gravity), 2);

  // Gravity is -x (left). Need 90 CCW to make upright.
  gravity = Eigen::Vector3d(-1, 0, 0);
  EXPECT_EQ(ComputeRot90FromGravity(gravity), 1);

  // Robustness to slight inaccuracies.
  gravity = Eigen::Vector3d(0.01, 0.99, 0.1);
  EXPECT_EQ(ComputeRot90FromGravity(gravity), 0);
  gravity = Eigen::Vector3d(0.99, -0.01, 0.1);
  EXPECT_EQ(ComputeRot90FromGravity(gravity), 3);
}

}  // namespace
}  // namespace colmap
