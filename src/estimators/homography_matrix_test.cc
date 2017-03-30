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

#define TEST_NAME "estimators/homography_matrix"
#include "util/testing.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "estimators/homography_matrix.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEstimate) {
  for (double x = 0; x < 10; ++x) {
    Eigen::Matrix3d H0;
    H0 << x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1;

    std::vector<Eigen::Vector2d> src;
    src.emplace_back(x, 0);
    src.emplace_back(1, 0);
    src.emplace_back(2, 1);
    src.emplace_back(10, 30);

    std::vector<Eigen::Vector2d> dst;

    for (size_t i = 0; i < 4; ++i) {
      const Eigen::Vector3d dsth = H0 * src[i].homogeneous();
      dst.push_back(dsth.hnormalized());
    }

    HomographyMatrixEstimator est_tform;
    const auto models = est_tform.Estimate(src, dst);

    std::vector<double> residuals;
    est_tform.Residuals(src, dst, models[0], &residuals);

    for (size_t i = 0; i < 4; ++i) {
      BOOST_CHECK(residuals[i] < 1e-6);
    }
  }
}
