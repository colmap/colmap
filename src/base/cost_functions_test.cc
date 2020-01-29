// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

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
