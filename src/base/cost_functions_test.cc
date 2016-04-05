// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "base/cost_functions"
#include <boost/test/unit_test.hpp>

#include "base/camera_models.h"
#include "base/cost_functions.h"

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
      SimplePinholeCameraModel>::Create(Eigen::Vector4d(1, 0, 0, 0),
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

BOOST_AUTO_TEST_CASE(TestUnitTranslationPlus) {
  UnitTranslationPlus function;
  Eigen::Vector3d x(0, 0, 1);
  Eigen::Vector3d delta(0, 0, 1);
  Eigen::Vector3d x_plus_delta(0, 0, 1);
  BOOST_CHECK(function(x.data(), delta.data(), x_plus_delta.data()));
  BOOST_CHECK_EQUAL(x_plus_delta.norm(), 1);
  BOOST_CHECK_EQUAL((x_plus_delta - x).norm(), 0);

  delta = Eigen::Vector3d(0, 1, 0);
  BOOST_CHECK(function(x.data(), delta.data(), x_plus_delta.data()));
  BOOST_CHECK_LT(std::abs(x_plus_delta.norm() - 1), 1e-6);
  BOOST_CHECK_EQUAL(
      (x_plus_delta - Eigen::Vector3d(0, 1, 1).normalized()).norm(), 0);
}
