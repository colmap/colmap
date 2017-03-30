// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "base/cost_functions"
#include "util/testing.h"

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/pose.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestBundleAdjustmentCostFunction) {
  ceres::CostFunction* cost_function =
      BundleAdjustmentCostFunction<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero());
  double qvec[4] = {1, 0, 0, 0};
  double tvec[3] = {0, 0, 0};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[4] = {qvec, tvec, point3D, camera_params};
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 0);

  point3D[1] = 1;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 1);

  camera_params[0] = 2;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 2);

  point3D[0] = -1;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], -2);
  BOOST_CHECK_EQUAL(residuals[1], 2);
}

BOOST_AUTO_TEST_CASE(TestBundleAdjustmentConstantPoseCostFunction) {
  ceres::CostFunction* cost_function = BundleAdjustmentConstantPoseCostFunction<
      SimplePinholeCameraModel>::Create(ComposeIdentityQuaternion(),
                                        Eigen::Vector3d::Zero(),
                                        Eigen::Vector2d::Zero());
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[2] = {point3D, camera_params};
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 0);

  point3D[1] = 1;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 1);

  camera_params[0] = 2;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 2);

  point3D[0] = -1;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], -2);
  BOOST_CHECK_EQUAL(residuals[1], 2);
}

BOOST_AUTO_TEST_CASE(TestRigBundleAdjustmentCostFunction) {
  ceres::CostFunction* cost_function =
      RigBundleAdjustmentCostFunction<SimplePinholeCameraModel>::Create(
          Eigen::Vector2d::Zero());
  double rig_qvec[4] = {1, 0, 0, 0};
  double rig_tvec[3] = {0, 0, -1};
  double rel_qvec[4] = {1, 0, 0, 0};
  double rel_tvec[3] = {0, 0, 1};
  double point3D[3] = {0, 0, 1};
  double camera_params[3] = {1, 0, 0};
  double residuals[2];
  const double* parameters[6] = {rig_qvec, rig_tvec, rel_qvec,
                                 rel_tvec, point3D,  camera_params};
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 0);

  point3D[1] = 1;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 1);

  camera_params[0] = 2;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 2);

  point3D[0] = -1;
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], -2);
  BOOST_CHECK_EQUAL(residuals[1], 2);
}

BOOST_AUTO_TEST_CASE(TestRelativePoseCostFunction) {
  ceres::CostFunction* cost_function = RelativePoseCostFunction::Create(
      Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0));
  double qvec[4] = {1, 0, 0, 0};
  double tvec[3] = {0, 1, 0};
  double residuals[1];
  const double* parameters[2] = {qvec, tvec};
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0);

  cost_function = RelativePoseCostFunction::Create(Eigen::Vector2d(0, 0),
                                                   Eigen::Vector2d(1, 0));
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0.5);

  cost_function = RelativePoseCostFunction::Create(Eigen::Vector2d(0, 0),
                                                   Eigen::Vector2d(1, 1));
  BOOST_CHECK(cost_function->Evaluate(parameters, residuals, nullptr));
  BOOST_CHECK_EQUAL(residuals[0], 0.5);
}
