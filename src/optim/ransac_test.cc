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
#define BOOST_TEST_MODULE "optim/ransac"
#include <boost/test/unit_test.hpp>

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

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
  BOOST_CHECK_EQUAL(
      RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(1, 100, 0.99),
      4605168);
  BOOST_CHECK_EQUAL(
      RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(10, 100, 0.99),
      4603);
  BOOST_CHECK_EQUAL(
      RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(10, 100, 0.999),
      6905);
  BOOST_CHECK_EQUAL(
      RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(100, 100, 0.99),
      1);
  BOOST_CHECK_EQUAL(RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(
                        100, 100, 0.999),
                    1);
  BOOST_CHECK_EQUAL(
      RANSAC<SimilarityTransformEstimator<3>>::ComputeNumTrials(100, 100, 0),
      1);
}

BOOST_AUTO_TEST_CASE(TestSimilarityTransform) {
  SetPRNGSeed(0);

  const size_t num_samples = 1000;
  const size_t num_outliers = 400;

  // Create some arbitrary transformation
  SimilarityTransform3 orig_tform(2, 0, 0, 0, 1, 100, 10, 10);

  // Generate exact data
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_samples; ++i) {
    src.emplace_back(i, std::sqrt(i) + 2, std::sqrt(2 * i + 2));
    dst.push_back(src.back());
    orig_tform.TransformPoint(&dst.back());
  }

  // Add some faulty data
  for (size_t i = 0; i < num_outliers; ++i) {
    dst[i] = Eigen::Vector3d(RandomReal(-3000.0, -2000.0),
                             RandomReal(-4000.0, -3000.0),
                             RandomReal(-5000.0, -4000.0));
  }

  // Robustly estimate transformation using RANSAC
  SetPRNGSeed(0);

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

  // Make sure original transformation is estimated correctly
  const double matrix_diff =
      (orig_tform.Matrix().topLeftCorner<3, 4>() - report.model).norm();
  BOOST_CHECK(std::abs(matrix_diff) < 1e-6);
}
