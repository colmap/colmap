// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
