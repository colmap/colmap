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

#define TEST_NAME "estimators/essential_matrix"
#include "util/testing.h"

#include <Eigen/Core>

#include "base/camera_models.h"
#include "base/essential_matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "estimators/essential_matrix.h"
#include "optim/ransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestFivePoint) {
  const double points1_raw[] = {
      0.4964, 1.0577, 0.3650,  -0.0919, -0.5412, 0.0159, -0.5239, 0.9467,
      0.3467, 0.5301, 0.2797,  0.0012,  -0.1986, 0.0460, -0.1622, 0.5347,
      0.0796, 0.2379, -0.3946, 0.7969,  0.2,     0.7,    0.6,     0.3};

  const double points2_raw[] = {
      0.7570, 2.7340, 0.3961,  0.6981, -0.6014, 0.7110, -0.7385, 2.2712,
      0.4177, 1.2132, 0.3052,  0.4835, -0.2171, 0.5057, -0.2059, 1.1583,
      0.0946, 0.7013, -0.6236, 3.0253, 0.5,     0.9,    0.9,     0.2};

  const size_t num_points = 12;

  std::vector<Eigen::Vector2d> points1(num_points);
  std::vector<Eigen::Vector2d> points2(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
    points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
  }

  // Enforce repeatable tests
  SetPRNGSeed(0);

  RANSACOptions options;
  options.max_error = 0.02;
  options.confidence = 0.9999;
  options.min_inlier_ratio = 0.1;

  RANSAC<EssentialMatrixFivePointEstimator> ransac(options);

  const auto report = ransac.Estimate(points1, points2);

  std::vector<double> residuals;
  EssentialMatrixFivePointEstimator::Residuals(points1, points2, report.model,
                                               &residuals);

  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_LE(residuals[i], options.max_error * options.max_error);
  }

  BOOST_CHECK(!report.inlier_mask[10]);
  BOOST_CHECK(!report.inlier_mask[11]);
}

BOOST_AUTO_TEST_CASE(TestEightPoint) {
  const double points1_raw[] = {1.839035, 1.924743, 0.543582,  0.375221,
                                0.473240, 0.142522, 0.964910,  0.598376,
                                0.102388, 0.140092, 15.994343, 9.622164,
                                0.285901, 0.430055, 0.091150,  0.254594};

  const double points2_raw[] = {
      1.002114, 1.129644, 1.521742, 1.846002, 1.084332, 0.275134,
      0.293328, 0.588992, 0.839509, 0.087290, 1.779735, 1.116857,
      0.878616, 0.602447, 0.642616, 1.028681,
  };

  const size_t kNumPoints = 8;
  std::vector<Eigen::Vector2d> points1(kNumPoints);
  std::vector<Eigen::Vector2d> points2(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
    points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
  }

  EssentialMatrixEightPointEstimator estimator;
  const auto E = estimator.Estimate(points1, points2)[0];

  // Reference values obtained from Matlab.
  BOOST_CHECK(std::abs(E(0, 0) - -0.0811666) < 1e-5);
  BOOST_CHECK(std::abs(E(0, 1) - 0.255449) < 1e-5);
  BOOST_CHECK(std::abs(E(0, 2) - -0.0478999) < 1e-5);
  BOOST_CHECK(std::abs(E(1, 0) - -0.192392) < 1e-5);
  BOOST_CHECK(std::abs(E(1, 1) - -0.0531675) < 1e-5);
  BOOST_CHECK(std::abs(E(1, 2) - 0.119547) < 1e-5);
  BOOST_CHECK(std::abs(E(2, 0) - 0.177784) < 1e-5);
  BOOST_CHECK(std::abs(E(2, 1) - -0.22008) < 1e-5);
  BOOST_CHECK(std::abs(E(2, 2) - -0.015203) < 1e-5);
}
