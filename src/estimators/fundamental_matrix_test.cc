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
#define BOOST_TEST_MODULE "estimators/fundamental_matrix"
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>

#include "estimators/fundamental_matrix.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestSevenPoint) {
  const double points1_raw[] = {0.4964, 1.0577,  0.3650,  -0.0919, -0.5412,
                                0.0159, -0.5239, 0.9467,  0.3467,  0.5301,
                                0.2797, 0.0012,  -0.1986, 0.0460};

  const double points2_raw[] = {0.7570, 2.7340,  0.3961,  0.6981, -0.6014,
                                0.7110, -0.7385, 2.2712,  0.4177, 1.2132,
                                0.3052, 0.4835,  -0.2171, 0.5057};

  const size_t kNumPoints = 7;

  std::vector<Eigen::Vector2d> points1(kNumPoints);
  std::vector<Eigen::Vector2d> points2(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    points1[i] = Eigen::Vector2d(points1_raw[2 * i], points1_raw[2 * i + 1]);
    points2[i] = Eigen::Vector2d(points2_raw[2 * i], points2_raw[2 * i + 1]);
  }

  FundamentalMatrixSevenPointEstimator estimator;
  const auto F = estimator.Estimate(points1, points2)[0];

  // Reference values obtained from Matlab
  BOOST_CHECK_CLOSE(F(0, 0), 4.81441976, 1e-6);
  BOOST_CHECK_CLOSE(F(0, 1), -8.16978909, 1e-6);
  BOOST_CHECK_CLOSE(F(0, 2), 6.73133404, 1e-6);
  BOOST_CHECK_CLOSE(F(1, 0), 5.16247992, 1e-6);
  BOOST_CHECK_CLOSE(F(1, 1), 0.19325606, 1e-6);
  BOOST_CHECK_CLOSE(F(1, 2), -2.87239381, 1e-6);
  BOOST_CHECK_CLOSE(F(2, 0), -9.92570126, 1e-6);
  BOOST_CHECK_CLOSE(F(2, 1), 3.64159554, 1e-6);
  BOOST_CHECK_CLOSE(F(2, 2), 1., 1e-6);
}
#include <iostream>
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

  FundamentalMatrixEightPointEstimator estimator;
  const auto F = estimator.Estimate(points1, points2)[0];

  // Reference values obtained from Matlab.
  BOOST_CHECK(std::abs(F(0, 0) - -0.217859) < 1e-5);
  BOOST_CHECK(std::abs(F(0, 1) - 0.419282) < 1e-5);
  BOOST_CHECK(std::abs(F(0, 2) - -0.0343075) < 1e-5);
  BOOST_CHECK(std::abs(F(1, 0) - -0.0717941) < 1e-5);
  BOOST_CHECK(std::abs(F(1, 1) - 0.0451643) < 1e-5);
  BOOST_CHECK(std::abs(F(1, 2) - 0.0216073) < 1e-5);
  BOOST_CHECK(std::abs(F(2, 0) - 0.248062) < 1e-5);
  BOOST_CHECK(std::abs(F(2, 1) - -0.429478) < 1e-5);
  BOOST_CHECK(std::abs(F(2, 2) - 0.0221019) < 1e-5);
}
