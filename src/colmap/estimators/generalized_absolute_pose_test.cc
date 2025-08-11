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

#include "colmap/estimators/generalized_absolute_pose.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <array>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

class ParameterizedGP3PEstimatorTests
    : public ::testing::TestWithParam<
          std::pair</*num_cams=*/int, /*panoramic=*/bool>> {};

TEST_P(ParameterizedGP3PEstimatorTests, Nominal) {
  // Note that we can estimate the minimal problem from only 3 points but we
  // need a 4th point to choose the correct solution. In theory, we don't need
  // RANSAC as we generate exact correspondences, but we use it in this test to
  // do the choosing of the best solution for us.
  constexpr int kNumPoints = 4;
  constexpr int kNumTrials = 10;
  const auto [kNumCams, kPanoramic] = GetParam();

  for (int i = 0; i < kNumTrials; ++i) {
    const Rigid3d rig_from_world(Eigen::Quaterniond::UnitRandom(),
                                 Eigen::Vector3d::Random());
    const Rigid3d world_from_rig = Inverse(rig_from_world);

    std::vector<Rigid3d> cams_from_world(kNumCams);
    std::vector<Rigid3d> cams_from_rig(kNumCams);
    for (int i = 0; i < kNumCams; ++i) {
      if (kPanoramic) {
        const Eigen::Quaterniond cam_from_rig_rotation =
            Eigen::Quaterniond::UnitRandom();
        cams_from_rig[i] =
            Rigid3d(cam_from_rig_rotation,
                    cam_from_rig_rotation * Eigen::Vector3d(1, 2, 3));
        cams_from_world[i] = cams_from_rig[i] * rig_from_world;
      } else {
        cams_from_world[i] = Rigid3d(Eigen::Quaterniond::UnitRandom(),
                                     Eigen::Vector3d::Random());
        cams_from_rig[i] = cams_from_world[i] * world_from_rig;
      }
    }

    std::vector<GP3PEstimator::X_t> points2D;
    std::vector<GP3PEstimator::Y_t> points3D;
    std::vector<GP3PEstimator::Y_t> points3D_outlier;
    for (int i = 0; i < kNumPoints; ++i) {
      points2D.emplace_back();
      points2D.back().cam_from_rig = cams_from_rig[i % kNumCams];
      points2D.back().ray_in_cam =
          Eigen::Vector3d(RandomUniformReal<double>(-0.5, 0.5),
                          RandomUniformReal<double>(-0.5, 0.5),
                          1)
              .normalized();
      const Eigen::Vector3d point3D_in_cam =
          points2D.back().ray_in_cam * RandomUniformReal<double>(0.1, 10);
      const Rigid3d world_from_cam = Inverse(cams_from_world[i % kNumCams]);
      points3D.push_back(world_from_cam * point3D_in_cam);
      points3D_outlier.push_back(
          world_from_cam *
          (Eigen::AngleAxisd(EIGEN_PI / 2, Eigen::Vector3d::UnitX()) *
           point3D_in_cam));
    }

    for (const auto residual_type :
         {GP3PEstimator::ResidualType::CosineDistance,
          GP3PEstimator::ResidualType::ReprojectionError}) {
      RANSACOptions options;
      options.max_error = 1e-5;
      RANSAC<GP3PEstimator> ransac(options, GP3PEstimator(residual_type));

      const auto report = ransac.Estimate(points2D, points3D);

      EXPECT_TRUE(report.success);
      EXPECT_THAT(report.model,
                  Rigid3dNear(rig_from_world, /*rtol=*/1e-6, /*ttol=*/1e-6));

      // Test residuals of inlier points.
      std::vector<double> residuals;
      ransac.estimator.Residuals(points2D, points3D, report.model, &residuals);
      EXPECT_EQ(residuals.size(), points2D.size());
      for (size_t i = 0; i < residuals.size(); ++i) {
        EXPECT_LT(residuals[i], 1e-10);
      }

      // Test residuals of outlier points.
      std::vector<double> residuals_outlier;
      ransac.estimator.Residuals(
          points2D, points3D_outlier, report.model, &residuals_outlier);
      EXPECT_EQ(residuals_outlier.size(), points2D.size());
      for (size_t i = 0; i < residuals_outlier.size(); ++i) {
        EXPECT_GT(residuals_outlier[i], 1e-2);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(GP3PEstimatorTests,
                         ParameterizedGP3PEstimatorTests,
                         ::testing::Values(std::make_pair(1, false),
                                           std::make_pair(2, false),
                                           std::make_pair(3, false),
                                           std::make_pair(4, false),
                                           std::make_pair(1, true),
                                           std::make_pair(2, true)));

}  // namespace
}  // namespace colmap
