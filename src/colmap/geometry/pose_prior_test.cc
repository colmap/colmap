// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/pose_prior.h"

#include "colmap/geometry/gps.h"
#include "colmap/util/eigen_matchers.h"

#include <gmock/gmock.h>
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

TEST(PosePrior, ConvertPosePriorsToENU) {
  std::vector<PosePrior> empty_priors;
  EXPECT_FALSE(ConvertPosePriorsToENU(&empty_priors));

  // Gravity-only and already-Cartesian priors are left unchanged.
  PosePrior gravity_only_prior;
  gravity_only_prior.gravity = Eigen::Vector3d(0, 1, 0);
  PosePrior cartesian_prior;
  cartesian_prior.position = Eigen::Vector3d(1, 2, 3);
  cartesian_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  std::vector<PosePrior> non_gps_priors = {gravity_only_prior, cartesian_prior};
  EXPECT_FALSE(ConvertPosePriorsToENU(&non_gps_priors));
  EXPECT_EQ(non_gps_priors[0], gravity_only_prior);
  EXPECT_EQ(non_gps_priors[1], cartesian_prior);

  // Mixed WGS84 + gravity-only (UNDEFINED, NaN position) priors convert only
  // the WGS84 entries and preserve gravity-only entries intact.
  PosePrior gps_prior1;
  gps_prior1.position = Eigen::Vector3d(47.3769, 8.5417, 408.0);
  gps_prior1.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  gps_prior1.gravity = Eigen::Vector3d(0, 1, 0);
  PosePrior gps_prior2;
  gps_prior2.position = Eigen::Vector3d(47.3779, 8.5427, 418.0);
  gps_prior2.coordinate_system = PosePrior::CoordinateSystem::WGS84;

  std::vector<PosePrior> mixed_priors = {
      gravity_only_prior, gps_prior1, gravity_only_prior, gps_prior2};
  EXPECT_TRUE(ConvertPosePriorsToENU(&mixed_priors));
  EXPECT_EQ(mixed_priors[0], gravity_only_prior);
  EXPECT_EQ(mixed_priors[2], gravity_only_prior);
  EXPECT_EQ(mixed_priors[1].coordinate_system,
            PosePrior::CoordinateSystem::CARTESIAN);
  EXPECT_EQ(mixed_priors[1].gravity, gps_prior1.gravity);
  EXPECT_THAT(mixed_priors[1].position,
              EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));
  EXPECT_EQ(mixed_priors[3].coordinate_system,
            PosePrior::CoordinateSystem::CARTESIAN);
  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);
  const Eigen::Vector3d expected_enu2 =
      gps_transform
          .EllipsoidToENU({gps_prior2.position},
                          gps_prior1.position.x(),
                          gps_prior1.position.y(),
                          gps_prior1.position.z())
          .front();
  EXPECT_THAT(mixed_priors[3].position, EigenMatrixNear(expected_enu2, 1e-6));

  // Mixing WGS84 and CARTESIAN finite positions throws.
  std::vector<PosePrior> inconsistent_priors = {gps_prior1, cartesian_prior};
  EXPECT_THAT([&]() { ConvertPosePriorsToENU(&inconsistent_priors); },
              testing::ThrowsMessage<std::invalid_argument>(
                  testing::HasSubstr("Inconsistent coordinate systems")));
}

}  // namespace
}  // namespace colmap
