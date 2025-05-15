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

#include "colmap/estimators/pose.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct AbsolutePoseProblem {
  Reconstruction reconstruction;
  Image image;
  Camera camera;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
};

AbsolutePoseProblem CreateAbsolutePoseTestData() {
  AbsolutePoseProblem problem;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.point2D_stddev = 0;
  SynthesizeDataset(synthetic_dataset_options, &problem.reconstruction);

  problem.image = problem.reconstruction.Image(1);
  problem.camera = *problem.image.CameraPtr();
  CHECK_EQ(problem.camera.model_id, CameraModelId::kSimpleRadial);
  for (const auto& point2D : problem.image.Points2D()) {
    if (point2D.HasPoint3D()) {
      problem.points2D.push_back(point2D.xy);
      problem.points3D.push_back(
          problem.reconstruction.Point3D(point2D.point3D_id).xyz);
    }
  }

  return problem;
}

TEST(EstimateAbsolutePose, Nominal) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();

  AbsolutePoseEstimationOptions options;
  Rigid3d cam_from_world;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  Camera camera = problem.camera;
  EXPECT_TRUE(EstimateAbsolutePose(options,
                                   problem.points2D,
                                   problem.points3D,
                                   &cam_from_world,
                                   &camera,
                                   &num_inliers,
                                   &inlier_mask));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-6, /*ttol=*/1e-6));
  EXPECT_EQ(camera, problem.camera);
  EXPECT_EQ(num_inliers, problem.points2D.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
}

TEST(EstimateAbsolutePose, EstimateFocalLength) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();

  AbsolutePoseEstimationOptions options;
  options.estimate_focal_length = true;
  Rigid3d cam_from_world;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  Camera camera = problem.camera;
  EXPECT_TRUE(EstimateAbsolutePose(options,
                                   problem.points2D,
                                   problem.points3D,
                                   &cam_from_world,
                                   &camera,
                                   &num_inliers,
                                   &inlier_mask));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-3, /*ttol=*/1e-2));
  EXPECT_NEAR(camera.FocalLength(), problem.camera.FocalLength(), 5);
  camera.SetFocalLength(problem.camera.FocalLength());
  EXPECT_EQ(camera, problem.camera);
  EXPECT_EQ(num_inliers, problem.points2D.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
}

TEST(EstimateRelativePose, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0.1, 0.2).normalized());

  std::vector<Eigen::Vector3d> points3D(100);
  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D[i] = Eigen::Vector3d::Random();
  }

  std::vector<Eigen::Vector2d> points1(points3D.size());
  std::vector<Eigen::Vector2d> points2(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    points1[i] = (cam1_from_world * points3D[i]).hnormalized();
    points2[i] = (cam2_from_world * points3D[i]).hnormalized();
  }

  RANSACOptions options;
  options.max_error = 1e-3;
  Rigid3d cam2_from_cam1;
  size_t num_inliers = 0;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateRelativePose(
      options, points1, points2, &cam2_from_cam1, &num_inliers, &inlier_mask));

  EXPECT_THAT(cam2_from_cam1,
              Rigid3dNear(cam2_from_world * Inverse(cam1_from_world),
                          /*rtol=*/1e-3,
                          /*ttol=*/1e-3));
  EXPECT_EQ(num_inliers, points3D.size());
  EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
}

TEST(RefineAbsolutePose, Nominal) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions options;
  Rigid3d cam_from_world = problem.image.CamFromWorld();
  cam_from_world =
      cam_from_world * Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                                   0.1, Eigen::Vector3d::Random())),
                               0.1 * Eigen::Vector3d::Random());
  Camera camera = problem.camera;
  Eigen::Matrix6d cam_from_world_cov = Eigen::Matrix6d::Zero();
  EXPECT_TRUE(RefineAbsolutePose(options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &cam_from_world,
                                 &camera,
                                 &cam_from_world_cov));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-6, /*ttol=*/1e-6));
  EXPECT_EQ(camera, problem.camera);
  EXPECT_NE(cam_from_world_cov, Eigen::Matrix6d::Zero());
}

TEST(RefineAbsolutePose, RefineFocalLength) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = true;
  Rigid3d cam_from_world = problem.image.CamFromWorld();
  Camera camera = problem.camera;
  camera.SetFocalLength(0.9 * camera.FocalLength());
  Eigen::Matrix6d cam_from_world_cov = Eigen::Matrix6d::Zero();
  EXPECT_TRUE(RefineAbsolutePose(options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &cam_from_world,
                                 &camera,
                                 &cam_from_world_cov));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-3, /*ttol=*/1e-3));
  EXPECT_NEAR(camera.FocalLength(), problem.camera.FocalLength(), 5);
  camera.SetFocalLength(problem.camera.FocalLength());
  EXPECT_EQ(camera, problem.camera);
  EXPECT_NE(cam_from_world_cov, Eigen::Matrix6d::Zero());
}

TEST(RefineAbsolutePose, RefineExtraParams) {
  const AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
  std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions options;
  options.refine_extra_params = true;
  Rigid3d cam_from_world = problem.image.CamFromWorld();
  Camera camera = problem.camera;
  camera.params.at(3) += 0.1;
  Eigen::Matrix6d cam_from_world_cov = Eigen::Matrix6d::Zero();
  EXPECT_TRUE(RefineAbsolutePose(options,
                                 inlier_mask,
                                 problem.points2D,
                                 problem.points3D,
                                 &cam_from_world,
                                 &camera,
                                 &cam_from_world_cov));
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.image.CamFromWorld(), /*rtol=*/1e-3, /*ttol=*/1e-3));
  EXPECT_NEAR(camera.params.at(3), problem.camera.params.at(3), 1e-3);
  camera.params.at(3) = problem.camera.params.at(3);
  EXPECT_EQ(camera, problem.camera);
  EXPECT_NE(cam_from_world_cov, Eigen::Matrix6d::Zero());
}

TEST(RefineEssentialMatrix, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Eigen::Matrix3d E =
      EssentialMatrixFromPose(cam2_from_world * Inverse(cam1_from_world));

  std::vector<Eigen::Vector3d> points3D(150);
  for (size_t i = 0; i < points3D.size() / 3; ++i) {
    points3D[3 * i + 0] = Eigen::Vector3d(i * 0.01, 0, 1);
    points3D[3 * i + 1] = Eigen::Vector3d(0, i * 0.01, 1);
    points3D[3 * i + 2] = Eigen::Vector3d(i * 0.01, i * 0.01, 1);
  }

  std::vector<Eigen::Vector2d> points1(points3D.size());
  std::vector<Eigen::Vector2d> points2(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    points1[i] = (cam1_from_world * points3D[i]).hnormalized();
    points2[i] = (cam2_from_world * points3D[i]).hnormalized();
  }

  const Rigid3d cam2_from_world_perturbed(
      Eigen::Quaterniond::Identity(),
      Eigen::Vector3d(1.02, 0.02, 0.01).normalized());
  const Eigen::Matrix3d E_pertubated =
      EssentialMatrixFromPose(cam2_from_world * Inverse(cam1_from_world));

  Eigen::Matrix3d E_refined = E_pertubated;
  ceres::Solver::Options options;
  RefineEssentialMatrix(options,
                        points1,
                        points2,
                        std::vector<char>(points1.size(), true),
                        &E_refined);

  EXPECT_LE((E - E_refined).norm(), (E - E_pertubated).norm());
}

}  // namespace
}  // namespace colmap
