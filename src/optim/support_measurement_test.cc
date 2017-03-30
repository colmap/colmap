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

#define TEST_NAME "optim/support_measurement"
#include "util/testing.h"

#include <unordered_set>

#include "optim/support_measurement.h"
#include "util/math.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestInlierSupportMeasuremer) {
  InlierSupportMeasurer::Support support1;
  BOOST_CHECK_EQUAL(support1.num_inliers, 0);
  BOOST_CHECK_EQUAL(support1.residual_sum, std::numeric_limits<double>::max());
  InlierSupportMeasurer measurer;
  std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  BOOST_CHECK_EQUAL(support1.num_inliers, 3);
  BOOST_CHECK_EQUAL(support1.residual_sum, 0.0);
  InlierSupportMeasurer::Support support2;
  support2.num_inliers = 2;
  BOOST_CHECK(measurer.Compare(support1, support2));
  BOOST_CHECK(!measurer.Compare(support2, support1));
  support2.residual_sum = support1.residual_sum;
  BOOST_CHECK(measurer.Compare(support1, support2));
  BOOST_CHECK(!measurer.Compare(support2, support1));
  support2.num_inliers = support1.num_inliers;
  support2.residual_sum += 0.01;
  BOOST_CHECK(measurer.Compare(support1, support2));
  BOOST_CHECK(!measurer.Compare(support2, support1));
  support2.residual_sum -= 0.01;
  BOOST_CHECK(!measurer.Compare(support1, support2));
  BOOST_CHECK(!measurer.Compare(support2, support1));
  support2.residual_sum -= 0.01;
  BOOST_CHECK(!measurer.Compare(support1, support2));
  BOOST_CHECK(measurer.Compare(support2, support1));
}

BOOST_AUTO_TEST_CASE(TestMEstimatorSupportMeasurer) {
  MEstimatorSupportMeasurer::Support support1;
  BOOST_CHECK_EQUAL(support1.num_inliers, 0);
  BOOST_CHECK_EQUAL(support1.score, std::numeric_limits<double>::max());
  MEstimatorSupportMeasurer measurer;
  std::vector<double> residuals = {-1.0, 0.0, 1.0, 2.0};
  support1 = measurer.Evaluate(residuals, 1.0);
  BOOST_CHECK_EQUAL(support1.num_inliers, 3);
  BOOST_CHECK_EQUAL(support1.score, 1.0);
  MEstimatorSupportMeasurer::Support support2 = support1;
  BOOST_CHECK(!measurer.Compare(support1, support2));
  BOOST_CHECK(!measurer.Compare(support2, support1));
  support2.num_inliers -= 1;
  support2.score += 0.01;
  BOOST_CHECK(measurer.Compare(support1, support2));
  BOOST_CHECK(!measurer.Compare(support2, support1));
  support2.score -= 0.01;
  BOOST_CHECK(!measurer.Compare(support1, support2));
  BOOST_CHECK(!measurer.Compare(support2, support1));
  support2.score -= 0.01;
  BOOST_CHECK(!measurer.Compare(support1, support2));
  BOOST_CHECK(measurer.Compare(support2, support1));
}
