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

#define TEST_NAME "optim/ransac"
#include "util/testing.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/pose.h"
#include "base/similarity_transform.h"
#include "estimators/similarity_transform.h"
#include "optim/ransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestOptions) {
  RANSACOptions options;
  BOOST_CHECK_EQUAL(options.max_error, 0);
  BOOST_CHECK_EQUAL(options.min_inlier_ratio, 0.1);
  BOOST_CHECK_EQUAL(options.confidence, 0.99);
  BOOST_CHECK_EQUAL(options.min_num_trials, 0);
  BOOST_CHECK_EQUAL(options.max_num_trials, std::numeric_limits<size_t>::max());
}

BOOST_AUTO_TEST_CASE(TestReport) {
  RANSAC<SimilarityTransformEstimator<3>>::Report report;
  BOOST_CHECK_EQUAL(report.success, false);
  BOOST_CHECK_EQUAL(report.num_trials, 0);
  BOOST_CHECK_EQUAL(report.support.num_inliers, 0);
  BOOST_CHECK_EQUAL(report.support.residual_sum,
                    std::numeric_limits<double>::max());
  BOOST_CHECK_EQUAL(report.inlier_mask.size(), 0);
}

BOOST_AUTO_TEST_CASE(TestNumTrials) {
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        1, 100, 0.99, 1.0),
                    4605168);
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        10, 100, 0.99, 1.0),
                    4603);
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        10, 100, 0.999, 1.0),
                    6905);
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        10, 100, 0.999, 2.0),
                    13809);
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        100, 100, 0.99, 1.0),
                    1);
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        100, 100, 0.999, 1.0),
                    1);
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        100, 100, 0, 1.0),
                    1);
}

BOOST_AUTO_TEST_CASE(TestSimilarityTransform) {
  SetPRNGSeed(0);

  const size_t num_samples = 1000;
  const size_t num_outliers = 400;

  // Create some arbitrary transformation.
  const SimilarityTransform3 orig_tform(2, ComposeIdentityQuaternion(),
                                        Eigen::Vector3d(100, 10, 10));

  // Generate exact data.
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_samples; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    dst.push_back(src.back());
    orig_tform.TransformPoint(&dst.back());
  }

  // Add some faulty data.
  for (size_t i = 0; i < num_outliers; ++i) {
    dst[i] = Eigen::Vector3d(RandomReal(-3000.0, -2000.0),
                             RandomReal(-4000.0, -3000.0),
                             RandomReal(-5000.0, -4000.0));
  }

  // Robustly estimate transformation using RANSAC.
  RANSACOptions options;
  options.max_error = 10;
  RANSAC<SimilarityTransformEstimator<3>> ransac(options);
  const auto report = ransac.Estimate(src, dst);

  BOOST_CHECK_EQUAL(report.success, true);
  BOOST_CHECK_GT(report.num_trials, 0);

  // Make sure outliers were detected correctly.
  BOOST_CHECK_EQUAL(report.support.num_inliers, num_samples - num_outliers);
  for (size_t i = 0; i < num_samples; ++i) {
    if (i < num_outliers) {
      BOOST_CHECK(!report.inlier_mask[i]);
    } else {
      BOOST_CHECK(report.inlier_mask[i]);
    }
  }

  // Make sure original transformation is estimated correctly.
  const double matrix_diff =
      (orig_tform.Matrix().topLeftCorner<3, 4>() - report.model).norm();
  BOOST_CHECK(std::abs(matrix_diff) < 1e-6);
}
