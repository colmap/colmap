// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/geometry/essential_matrix.h"

#include "colmap/estimators/essential_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/math/random.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void RandomEpipolarCorrespondences(const Rigid3d& cam2_from_cam1,
                                   size_t num_rays,
                                   std::vector<Eigen::Vector3d>& rays1,
                                   std::vector<Eigen::Vector3d>& rays2) {
  for (size_t i = 0; i < num_rays; ++i) {
    rays1.push_back(Eigen::Vector3d::Random().normalized());
    const double random_depth = RandomUniformReal<double>(0.2, 2.0);
    rays2.push_back(
        (cam2_from_cam1 * (random_depth * rays1.back())).normalized());
  }
}

template <typename Estimator>
void ExpectAtLeastOneValidModel(const Estimator& estimator,
                                const std::vector<Eigen::Vector3d>& rays1,
                                const std::vector<Eigen::Vector3d>& rays2,
                                Eigen::Matrix3d& expected_E,
                                std::vector<Eigen::Matrix3d>& models,
                                double E_eps = 1e-4,
                                double r_eps = 1e-5) {
  expected_E /= expected_E(2, 2);
  for (size_t i = 0; i < models.size(); ++i) {
    Eigen::Matrix3d E = models[i];
    E /= E(2, 2);
    if (!E.isApprox(expected_E, E_eps)) {
      continue;
    }

    bool all_residuals_small = true;
    std::vector<double> residuals;
    estimator.Residuals(rays1, rays2, E, &residuals);
    for (const double residual : residuals) {
      if (residual > r_eps) {
        all_residuals_small = false;
        break;
      }
    }

    if (all_residuals_small) {
      return;
    }
  }

  ADD_FAILURE() << "No good solution found.";
}

class EssentialMatrixFivePointEstimatorTests
    : public ::testing::TestWithParam<size_t> {};

TEST_P(EssentialMatrixFivePointEstimatorTests, Nominal) {
  const size_t kNumrays = GetParam();
  for (size_t k = 0; k < 100; ++k) {
    const Rigid3d cam2_from_cam1(Eigen::Quaterniond::UnitRandom(),
                                 Eigen::Vector3d::Random());
    Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(cam2_from_cam1, kNumrays, rays1, rays2);

    EssentialMatrixFivePointEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(rays1, rays2, &models);

    ExpectAtLeastOneValidModel(estimator, rays1, rays2, expected_E, models);
  }
}

INSTANTIATE_TEST_SUITE_P(EssentialMatrixFivePointEstimator,
                         EssentialMatrixFivePointEstimatorTests,
                         ::testing::Values(5, 20, 1000));

TEST(EssentialMatrixEightPointEstimator, Reference) {
  const double rays1_raw[] = {1.839035,
                              1.924743,
                              0.543582,
                              0.375221,
                              0.473240,
                              0.142522,
                              0.964910,
                              0.598376,
                              0.102388,
                              0.140092,
                              15.994343,
                              9.622164,
                              0.285901,
                              0.430055,
                              0.091150,
                              0.254594};

  const double rays2_raw[] = {
      1.002114,
      1.129644,
      1.521742,
      1.846002,
      1.084332,
      0.275134,
      0.293328,
      0.588992,
      0.839509,
      0.087290,
      1.779735,
      1.116857,
      0.878616,
      0.602447,
      0.642616,
      1.028681,
  };

  const size_t kNumrays = 8;
  std::vector<Eigen::Vector3d> rays1(kNumrays);
  std::vector<Eigen::Vector3d> rays2(kNumrays);
  for (size_t i = 0; i < kNumrays; ++i) {
    rays1[i] = Eigen::Vector3d(rays1_raw[2 * i], rays1_raw[2 * i + 1], 1);
    rays2[i] = Eigen::Vector3d(rays2_raw[2 * i], rays2_raw[2 * i + 1], 1);
  }

  EssentialMatrixEightPointEstimator estimator;
  std::vector<Eigen::Matrix3d> models;
  estimator.Estimate(rays1, rays2, &models);

  Eigen::Matrix3d expected_E;
  expected_E << -0.315968, 0.604935, -0.0538653, -0.103596, 0.0652994,
      0.0208669, 0.355946, -0.622327, 0.043552;
  expected_E /= expected_E(2, 2);

  ASSERT_EQ(models.size(), 1);
  Eigen::Matrix3d& E = models[0];
  E /= E(2, 2);

  EXPECT_THAT(E, EigenMatrixNear(expected_E, 1e-5));

  // Check that the internal constraint is satisfied (two singular values equal
  // and one zero).
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
  Eigen::Vector3d s = svd.singularValues();
  EXPECT_GT(s(0), 1e-5);
  EXPECT_GT(s(1), 1e-5);
  EXPECT_NEAR(s(2), 0, 1e-5);
}

class EssentialMatrixEightPointEstimatorTests
    : public ::testing::TestWithParam<size_t> {};

TEST_P(EssentialMatrixEightPointEstimatorTests, Nominal) {
  const size_t kNumrays = GetParam();
  for (size_t k = 0; k < 1; ++k) {
    const Rigid3d cam2_from_cam1(Eigen::Quaterniond::UnitRandom(),
                                 Eigen::Vector3d::Random());
    Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(cam2_from_cam1, kNumrays, rays1, rays2);

    EssentialMatrixEightPointEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(rays1, rays2, &models);

    ExpectAtLeastOneValidModel(estimator, rays1, rays2, expected_E, models);
  }
}

INSTANTIATE_TEST_SUITE_P(EssentialMatrixEightPointEstimator,
                         EssentialMatrixEightPointEstimatorTests,
                         ::testing::Values(8, 64, 1024));

}  // namespace
}  // namespace colmap
