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

#include "colmap/estimators/solvers/essential_matrix.h"
#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Generates a random relative pose with a unit-norm baseline. Bounding the
// baseline away from zero avoids the (near) pure-rotation degeneracy, where the
// essential matrix vanishes and the cheirality of every correspondence becomes
// ill-defined.
Rigid3d TestCam2FromCam1() {
  return Rigid3d(Eigen::Quaterniond::UnitRandom(),
                 Eigen::Vector3d::Random().normalized());
}

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
  expected_E.normalize();
  for (size_t i = 0; i < models.size(); ++i) {
    const Eigen::Matrix3d E = models[i].normalized();
    if (std::min((E - expected_E).norm(), (E + expected_E).norm()) > E_eps) {
      continue;
    }

    std::vector<double> residuals;
    estimator.Residuals(rays1, rays2, E, &residuals);
    for (size_t j = 0; j < rays1.size(); ++j) {
      EXPECT_LT(residuals[j], r_eps);
    }

    return;
  }
  ADD_FAILURE() << "No essential matrix is equal up to scale.";
}

class EssentialMatrixFivePointEstimatorTests
    : public ::testing::TestWithParam<size_t> {};

TEST_P(EssentialMatrixFivePointEstimatorTests, Nominal) {
  SetPRNGSeed(0);
  const size_t kNumRays = GetParam();
  // The minimal case has no redundancy, so its recovered E is only accurate to
  // the solver's numerical limit and is checked with a looser tolerance.
  const bool is_minimal =
      kNumRays == EssentialMatrixFivePointEstimator::kMinNumSamples;
  for (size_t k = 0; k < 100; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(cam2_from_cam1, kNumRays, rays1, rays2);

    EssentialMatrixFivePointEstimator estimator;
    std::vector<Eigen::Matrix3d> models;
    estimator.Estimate(rays1, rays2, &models);

    ExpectAtLeastOneValidModel(estimator,
                               rays1,
                               rays2,
                               expected_E,
                               models,
                               /*E_eps=*/is_minimal ? 5e-3 : 1e-4);
  }
}

INSTANTIATE_TEST_SUITE_P(EssentialMatrixFivePointEstimator,
                         EssentialMatrixFivePointEstimatorTests,
                         ::testing::Values(5, 20, 1000));

class EssentialMatrixEightPointEstimatorTests
    : public ::testing::TestWithParam<size_t> {};

TEST_P(EssentialMatrixEightPointEstimatorTests, Nominal) {
  SetPRNGSeed(0);
  const size_t kNumRays = GetParam();
  for (size_t k = 0; k < 1; ++k) {
    const Rigid3d cam2_from_cam1 = TestCam2FromCam1();
    Eigen::Matrix3d expected_E = EssentialMatrixFromPose(cam2_from_cam1);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    RandomEpipolarCorrespondences(cam2_from_cam1, kNumRays, rays1, rays2);

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
