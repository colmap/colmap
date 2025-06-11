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

#include "colmap/estimators/generalized_relative_pose.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"

#include <array>
#include <tuple>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct GeneralizedRelativePoseProblem {
  std::vector<GRNPObservation> points1;
  std::vector<GRNPObservation> points2;
  Rigid3d rig2_from_rig1;
};

GeneralizedRelativePoseProblem CreateGeneralizedRelativePoseProblem(
    int num_points,
    int num_cameras1,
    int num_cameras2,
    bool panoramic1,
    bool panoramic2) {
  GeneralizedRelativePoseProblem problem;

  const std::array<Rigid3d, 2> rigs_from_world = {
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random()),
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random())};

  problem.rig2_from_rig1 = rigs_from_world[1] * Inverse(rigs_from_world[0]);

  std::vector<Rigid3d> cams_from_rig1(num_cameras1);
  for (int i = 0; i < num_cameras1; ++i) {
    const Eigen::Quaterniond cam1_from_rig_rotation =
        Eigen::Quaterniond::UnitRandom();
    cams_from_rig1[i] =
        Rigid3d(cam1_from_rig_rotation,
                panoramic1 ? cam1_from_rig_rotation * Eigen::Vector3d(1, 2, 3)
                           : Eigen::Vector3d(Eigen::Vector3d::Random()));
  }

  std::vector<Rigid3d> cams_from_rig2(num_cameras2);
  for (int i = 0; i < num_cameras2; ++i) {
    const Eigen::Quaterniond cam2_from_rig_rotation =
        Eigen::Quaterniond::UnitRandom();
    cams_from_rig2[i] = Rigid3d(
        cam2_from_rig_rotation,
        panoramic2 ? cam2_from_rig_rotation * Eigen::Vector3d(-3, -2, -1)
                   : Eigen::Vector3d(Eigen::Vector3d::Random()));
  }

  std::vector<Eigen::Vector3d> points3D;
  points3D.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    points3D.emplace_back(Eigen::Vector3d::Random());
  }

  problem.points1.reserve(num_points);
  problem.points2.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const size_t cam_idx1 = i % num_cameras1;
    const size_t cam_idx2 = i % num_cameras2;
    const Eigen::Vector3d point3D_in_cam1 =
        cams_from_rig1[cam_idx1] * (rigs_from_world[0] * points3D[i]);
    const Eigen::Vector3d point3D_in_cam2 =
        cams_from_rig2[cam_idx2] * (rigs_from_world[1] * points3D[i]);
    if (point3D_in_cam1.norm() < 1e-8 || point3D_in_cam2.norm() < 1e-8) {
      continue;
    }

    auto& point1 = problem.points1.emplace_back();
    point1.cam_from_rig = cams_from_rig1[cam_idx1];
    point1.ray_in_cam = point3D_in_cam1.normalized();

    auto& point2 = problem.points2.emplace_back();
    point2.cam_from_rig = cams_from_rig2[cam_idx2];
    point2.ray_in_cam = point3D_in_cam2.normalized();
  }

  return problem;
}

class ParameterizedGRNPEstimatorTests
    : public ::testing::TestWithParam<std::tuple</*num_cams1=*/int,
                                                 /*num_cams2=*/int,
                                                 /*panoramic1=*/bool,
                                                 /*panoramic2=*/bool>> {};

TEST_P(ParameterizedGRNPEstimatorTests, GR6P) {
  // Note that we can estimate the minimal problem from only 6 points but we
  // need a 7th point to choose the correct solution. In theory, we don't need
  // RANSAC as we generate exact correspondences, but we use it in this test to
  // do the choosing of the best solution for us.
  constexpr int kNumPoints = 7;
  constexpr int kNumTrials = 10;
  const auto [kNumCams1, kNumCams2, kPanoramic1, kPanoramic2] = GetParam();

  for (int i = 0; i < kNumTrials; ++i) {
    const auto problem = CreateGeneralizedRelativePoseProblem(
        kNumPoints, kNumCams1, kNumCams2, kPanoramic1, kPanoramic2);

    RANSACOptions options;
    options.max_error = 1e-3;
    RANSAC<GR6PEstimator> ransac(options);
    const auto report = ransac.Estimate(problem.points1, problem.points2);

    EXPECT_TRUE(report.success);
    EXPECT_THAT(report.model,
                Rigid3dNear(problem.rig2_from_rig1,
                            /*rtol=*/1e-4,
                            /*ttol=*/1e-4));

    std::vector<double> residuals;
    GR6PEstimator::Residuals(
        problem.points1, problem.points2, report.model, &residuals);
    for (size_t i = 0; i < residuals.size(); ++i) {
      EXPECT_LE(residuals[i], options.max_error);
    }
  }
}

TEST_P(ParameterizedGRNPEstimatorTests, GR8P) {
  // In theory, we don't need RANSAC as we generate exact correspondences, but
  // we use it in this test to do the choosing of the best solution for us.
  constexpr int kNumPoints = 8;
  constexpr int kNumTrials = 10;
  const auto [kNumCams1, kNumCams2, kPanoramic1, kPanoramic2] = GetParam();

  for (int i = 0; i < kNumTrials; ++i) {
    const auto problem = CreateGeneralizedRelativePoseProblem(
        kNumPoints, kNumCams1, kNumCams2, kPanoramic1, kPanoramic2);

    RANSACOptions options;
    options.max_error = 1e-3;
    RANSAC<GR6PEstimator> ransac(options);
    const auto report = ransac.Estimate(problem.points1, problem.points2);

    EXPECT_TRUE(report.success);
    EXPECT_THAT(report.model,
                Rigid3dNear(problem.rig2_from_rig1,
                            /*rtol=*/1e-4,
                            /*ttol=*/1e-4));

    std::vector<double> residuals;
    GR6PEstimator::Residuals(
        problem.points1, problem.points2, report.model, &residuals);
    for (size_t i = 0; i < residuals.size(); ++i) {
      EXPECT_LE(residuals[i], options.max_error);
    }
  }
}

// Note that only one of the cameras must be panoramic, otherwise the
// generalized relative pose problem is ill-posed, as we cannot estimate the
// scale between the rigs.
INSTANTIATE_TEST_SUITE_P(GRNPEstimatorTests,
                         ParameterizedGRNPEstimatorTests,
                         ::testing::Values(std::make_tuple(1, 2, false, false),
                                           std::make_tuple(2, 1, false, false),
                                           std::make_tuple(2, 2, false, false),
                                           std::make_tuple(3, 3, false, false),
                                           std::make_tuple(4, 4, false, false),
                                           std::make_tuple(4, 4, false, true),
                                           std::make_tuple(4, 4, true, false)));

}  // namespace
}  // namespace colmap
