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

#include "colmap/estimators/cost_functions/pose_prior.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_matchers.h"

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(AbsolutePosePositionPriorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsolutePosePositionPriorCostFunctor::Create(Eigen::Vector3d::Zero()));

  Rigid3d sensor_from_world =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());

  Eigen::Vector3d residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const double* parameters[1] = {sensor_from_world.params.data()};

  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));

  sensor_from_world =
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  const Eigen::Vector3d position_in_world =
      Inverse(sensor_from_world).translation();
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(-position_in_world), 1e-6));

  cost_function.reset(
      AbsolutePosePositionPriorCostFunctor::Create(position_in_world));
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));
}

TEST(AbsoluteRigPosePositionPriorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsoluteRigPosePositionPriorCostFunctor::Create(Eigen::Vector3d::Zero()));

  Rigid3d sensor_from_rig(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d::Zero());
  Rigid3d rig_from_world(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d::Zero());

  Eigen::Vector3d residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const double* parameters[2] = {sensor_from_rig.params.data(),
                                 rig_from_world.params.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));

  sensor_from_rig = Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  rig_from_world = Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  const Rigid3d sensor_from_world = sensor_from_rig * rig_from_world;
  const Eigen::Vector3d position_in_world =
      Inverse(sensor_from_world).translation();
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(-position_in_world), 1e-6));

  cost_function.reset(
      AbsoluteRigPosePositionPriorCostFunctor::Create(position_in_world));
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));
}

TEST(AbsolutePosePriorCostFunctor, Nominal) {
  const Rigid3d cam_from_world_prior;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsolutePosePriorCostFunctor::Create(cam_from_world_prior));

  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double residuals[6];
  const double* parameters[1] = {cam_from_world};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  cam_from_world[4] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 1);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  // Rotation by 90 degrees around the Y axis.
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  Eigen::Map<Eigen::Quaterniond>(static_cast<double*>(cam_from_world)) =
      rotation_matrix;
  cam_from_world[5] = 2;
  cam_from_world[6] = 3;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], 0, 1e-6);
  EXPECT_NEAR(residuals[1], DegToRad(90.0), 1e-6);
  EXPECT_NEAR(residuals[2], 0, 1e-6);
  EXPECT_NEAR(residuals[3], 1, 1e-6);
  EXPECT_NEAR(residuals[4], 2, 1e-6);
  EXPECT_NEAR(residuals[5], 3, 1e-6);
}

TEST(RelativePosePriorCostFunctor, Nominal) {
  Rigid3d i_from_j_prior(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(0, 0, -1));
  std::unique_ptr<ceres::CostFunction> cost_function(
      RelativePosePriorCostFunctor::Create(i_from_j_prior));

  double i_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double j_from_world[7] = {0, 0, 0, 1, 0, 0, 1};
  double residuals[6];
  const double* parameters[2] = {i_from_world, j_from_world};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  i_from_world[6] = 4;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 4);

  j_from_world[4] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], -2);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 4);

  // Rotation by 90 degrees around the Y axis.
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  Eigen::Map<Eigen::Quaterniond>(static_cast<double*>(j_from_world)) =
      rotation_matrix;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], 0, 1e-6);
  EXPECT_NEAR(residuals[1], DegToRad(-90.0), 1e-6);
  EXPECT_NEAR(residuals[2], 0, 1e-6);
  EXPECT_NEAR(residuals[3], 0, 1e-6);
  EXPECT_NEAR(residuals[4], 0, 1e-6);
  EXPECT_NEAR(residuals[5], 2, 1e-6);
}

Eigen::Vector3d EnuDown() { return Eigen::Vector3d(0.0, 0.0, -1.0); }

TEST(AbsoluteGravityPriorCostFunctor, ZeroAtTruth) {
  const Eigen::Vector3d world_down = EnuDown();
  const Eigen::Vector3d measured_down_in_sensor = world_down;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsoluteGravityPriorCostFunctor::Create(world_down,
                                              measured_down_in_sensor));

  // Identity rotation: predicted_down_in_sensor == world_down == measured.
  Rigid3d sensor_from_world =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  Eigen::Vector3d residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const double* parameters[1] = {sensor_from_world.params.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-9));

  // Translation must not affect the residual (rotation-only).
  sensor_from_world.translation() = RandomEigenVectord<3>();
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-9));
}

TEST(AbsoluteGravityPriorCostFunctor, NoFalseMinimumOverFullAngularDomain) {
  // The prior (tangent-plane-projected) residual had norm sin(theta), which
  // is zero both at theta=0 (truth) and theta=180deg (an inverted, upside
  // down prediction). The chordal residual's norm is 2*sin(theta/2): zero
  // only at theta=0, and strictly increasing (monotonic) over [0, 180deg].
  const Eigen::Vector3d world_down = EnuDown();
  const Eigen::Vector3d measured_down_in_sensor = world_down;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsoluteGravityPriorCostFunctor::Create(world_down,
                                              measured_down_in_sensor));

  double previous_norm = -1.0;
  for (const double tilt_deg :
       {0.0, 1.0, 3.0, 5.0, 10.0, 45.0, 90.0, 137.0, 179.0, 180.0}) {
    const double theta = DegToRad(tilt_deg);
    Rigid3d sensor_from_world(
        Eigen::Quaterniond(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitX())),
        Eigen::Vector3d::Zero());
    Eigen::Vector3d residuals;
    const double* parameters[1] = {sensor_from_world.params.data()};
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
    EXPECT_TRUE(residuals.allFinite()) << "tilt_deg=" << tilt_deg;

    const double expected_norm = 2.0 * std::sin(theta / 2.0);
    EXPECT_NEAR(residuals.norm(), expected_norm, 1e-9)
        << "tilt_deg=" << tilt_deg;

    if (tilt_deg == 0.0) {
      EXPECT_NEAR(residuals.norm(), 0.0, 1e-9);
    } else {
      // Strictly positive everywhere except truth -- in particular at
      // 180deg, where the old projected residual was falsely zero.
      EXPECT_GT(residuals.norm(), 1e-6) << "tilt_deg=" << tilt_deg;
    }
    // Monotonic strictly increasing norm over [0, 180] degrees.
    EXPECT_GT(residuals.norm(), previous_norm) << "tilt_deg=" << tilt_deg;
    previous_norm = residuals.norm();
  }
}

TEST(AbsoluteGravityPriorCostFunctor, TiltDirectionChangesResidualDirection) {
  // For equal and opposite tilts about X, the tangential Y component must
  // change sign while the chordal Z component stays equal. Checking only the
  // norm would not catch a sign error in the predicted direction.
  const Eigen::Vector3d world_down = EnuDown();
  const Eigen::Vector3d measured_down_in_sensor = world_down;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsoluteGravityPriorCostFunctor::Create(world_down,
                                              measured_down_in_sensor));

  const double theta = DegToRad(7.0);
  Rigid3d plus_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitX())),
      Eigen::Vector3d::Zero());
  Rigid3d minus_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(-theta, Eigen::Vector3d::UnitX())),
      Eigen::Vector3d::Zero());

  Eigen::Vector3d residuals_plus;
  Eigen::Vector3d residuals_minus;
  const double* parameters_plus[1] = {plus_from_world.params.data()};
  const double* parameters_minus[1] = {minus_from_world.params.data()};
  EXPECT_TRUE(
      cost_function->Evaluate(parameters_plus, residuals_plus.data(), nullptr));
  EXPECT_TRUE(cost_function->Evaluate(
      parameters_minus, residuals_minus.data(), nullptr));
  EXPECT_NEAR(residuals_plus.x(), 0.0, 1e-9);
  EXPECT_NEAR(residuals_minus.x(), 0.0, 1e-9);
  EXPECT_NEAR(residuals_plus.y(), -residuals_minus.y(), 1e-9);
  EXPECT_NEAR(residuals_plus.z(), residuals_minus.z(), 1e-9);
  EXPECT_GT(residuals_plus.y(), 0.0);
  EXPECT_GT(residuals_plus.z(), 0.0);
  EXPECT_NEAR(residuals_plus.norm(), residuals_minus.norm(), 1e-9);
}

TEST(AbsoluteGravityPriorCostFunctor, ExactYawInvariance) {
  // A pure yaw rotation about the sensor's own predicted-down axis must
  // leave the residual exactly unchanged, for a non-trivial base rotation
  // and several yaw angles -- not just to first order.
  const Eigen::Vector3d world_down = EnuDown();
  const Eigen::Vector3d measured_down_in_sensor = world_down;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsoluteGravityPriorCostFunctor::Create(world_down,
                                              measured_down_in_sensor));

  const Eigen::Quaterniond base_rotation = RandomEigenQuaterniond();
  Rigid3d base_sensor_from_world(base_rotation, RandomEigenVectord<3>());
  Eigen::Vector3d base_residuals;
  const double* base_parameters[1] = {base_sensor_from_world.params.data()};
  EXPECT_TRUE(
      cost_function->Evaluate(base_parameters, base_residuals.data(), nullptr));

  const Eigen::Vector3d predicted_down_in_sensor = base_rotation * world_down;

  for (const double yaw_deg : {30.0, 90.0, 180.0, 270.0}) {
    const Eigen::Quaterniond yaw = Eigen::Quaterniond(Eigen::AngleAxisd(
        DegToRad(yaw_deg), predicted_down_in_sensor.normalized()));
    Rigid3d yawed_sensor_from_world(yaw * base_rotation,
                                    base_sensor_from_world.translation());
    Eigen::Vector3d yawed_residuals;
    const double* yawed_parameters[1] = {yawed_sensor_from_world.params.data()};
    EXPECT_TRUE(cost_function->Evaluate(
        yawed_parameters, yawed_residuals.data(), nullptr));
    EXPECT_THAT(yawed_residuals, EigenMatrixNear(base_residuals, 1e-9))
        << "yaw_deg=" << yaw_deg;
  }
}

TEST(AbsoluteRigGravityPriorCostFunctor,
     MatchesNonRigForIdentitySensorFromRig) {
  const Eigen::Vector3d world_down = EnuDown();
  const Eigen::Vector3d measured_down_in_sensor = world_down;
  std::unique_ptr<ceres::CostFunction> rig_cost_function(
      AbsoluteRigGravityPriorCostFunctor::Create(world_down,
                                                 measured_down_in_sensor));
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsoluteGravityPriorCostFunctor::Create(world_down,
                                              measured_down_in_sensor));

  const double theta = DegToRad(5.0);
  Rigid3d sensor_from_rig(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d::Zero());
  Rigid3d rig_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitX())),
      Eigen::Vector3d::Zero());
  Rigid3d sensor_from_world = sensor_from_rig * rig_from_world;

  Eigen::Vector3d rig_residuals;
  const double* rig_parameters[2] = {sensor_from_rig.params.data(),
                                     rig_from_world.params.data()};
  EXPECT_TRUE(rig_cost_function->Evaluate(
      rig_parameters, rig_residuals.data(), nullptr));

  Eigen::Vector3d residuals;
  const double* parameters[1] = {sensor_from_world.params.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));

  EXPECT_THAT(rig_residuals, EigenMatrixNear(residuals, 1e-9));
}

TEST(CovarianceWeightedCostFunctor, AbsolutePosePositionPriorCostFunctor) {
  const Rigid3d cam_from_world(RandomEigenQuaterniond(),
                               RandomEigenVectord<3>());
  const Rigid3d world_from_cam = Inverse(cam_from_world);

  double residuals[3];
  const double* parameters[1] = {cam_from_world.params.data()};

  std::unique_ptr<ceres::CostFunction> cost_function(
      CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
          Create(2 * Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0],
              -0.5 * std::sqrt(2) * world_from_cam.translation()[0],
              1e-6);
  EXPECT_NEAR(residuals[1],
              -0.5 * std::sqrt(2) * world_from_cam.translation()[1],
              1e-6);
  EXPECT_NEAR(residuals[2],
              -0.5 * std::sqrt(2) * world_from_cam.translation()[2],
              1e-6);
}

}  // namespace
}  // namespace colmap
