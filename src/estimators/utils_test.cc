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
