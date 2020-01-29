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

#define TEST_NAME "estimators/utils"
#include "util/testing.h"

#include "base/essential_matrix.h"
#include "estimators/utils.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestCenterAndNormalizeImagePoints) {
  std::vector<Eigen::Vector2d> points;
  for (size_t i = 0; i < 11; ++i) {
    points.emplace_back(i, i);
  }

  std::vector<Eigen::Vector2d> normed_points;
  Eigen::Matrix3d matrix;
  CenterAndNormalizeImagePoints(points, &normed_points, &matrix);

  BOOST_CHECK_EQUAL(matrix(0, 0), 0.31622776601683794);
  BOOST_CHECK_EQUAL(matrix(1, 1), 0.31622776601683794);
  BOOST_CHECK_EQUAL(matrix(0, 2), -1.5811388300841898);
  BOOST_CHECK_EQUAL(matrix(1, 2), -1.5811388300841898);

  Eigen::Vector2d mean_point(0, 0);
  for (const auto& point : normed_points) {
    mean_point += point;
  }
  BOOST_CHECK_LT(std::abs(mean_point[0]), 1e-6);
  BOOST_CHECK_LT(std::abs(mean_point[1]), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestComputeSquaredSampsonError) {
  std::vector<Eigen::Vector2d> points1;
  points1.emplace_back(0, 0);
  points1.emplace_back(0, 0);
  points1.emplace_back(0, 0);
  std::vector<Eigen::Vector2d> points2;
  points2.emplace_back(2, 0);
  points2.emplace_back(2, 1);
  points2.emplace_back(2, 2);

  const Eigen::Matrix3d E = EssentialMatrixFromPose(Eigen::Matrix3d::Identity(),
                                                    Eigen::Vector3d(1, 0, 0));

  std::vector<double> residuals;
  ComputeSquaredSampsonError(points1, points2, E, &residuals);

  BOOST_CHECK_EQUAL(residuals.size(), 3);
  BOOST_CHECK_EQUAL(residuals[0], 0);
  BOOST_CHECK_EQUAL(residuals[1], 0.5);
  BOOST_CHECK_EQUAL(residuals[2], 2);
}
