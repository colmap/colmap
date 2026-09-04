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

#include "colmap/estimators/generalized_pose.h"

#include "colmap/estimators/solvers/generalized_absolute_pose.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/geometry/sim3.h"
#include "colmap/geometry/sim3_matchers.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/hash_containers.h"

#include <numeric>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct GeneralizedAbsolutePoseProblem {
  Rigid3d gt_rig_from_world;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  std::vector<size_t> point3D_ids;
  std::vector<size_t> camera_idxs;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;
};

GeneralizedAbsolutePoseProblem BuildGeneralizedAbsolutePoseProblem() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  GeneralizedAbsolutePoseProblem problem;
  problem.gt_rig_from_world =
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>());
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    for (const auto& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        problem.points2D.push_back(point2D.xy);
        problem.points3D.push_back(
            reconstruction.Point3D(point2D.point3D_id).xyz);
        problem.point3D_ids.push_back(point2D.point3D_id);
        problem.camera_idxs.push_back(problem.cameras.size());
      }
    }
    problem.cameras.push_back(*image.CameraPtr());
    problem.cams_from_rig.push_back(image.CamFromWorld() *
                                    Inverse(problem.gt_rig_from_world));
  }
  return problem;
}

struct ScaledGeneralizedAbsolutePoseProblem {
  Sim3d gt_rig_from_world;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  std::vector<size_t> point3D_ids;
  std::vector<size_t> camera_idxs;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;
};

ScaledGeneralizedAbsolutePoseProblem
BuildScaledGeneralizedAbsolutePoseProblem() {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  ScaledGeneralizedAbsolutePoseProblem problem;
  problem.gt_rig_from_world = Sim3d(RandomUniformReal<double>(0.5, 2),
                                    RandomEigenQuaterniond(),
                                    RandomEigenVectord<3>());
  for (const image_t image_id : reconstruction.RegImageIds()) {
    const auto& image = reconstruction.Image(image_id);
    for (const auto& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        problem.points2D.push_back(point2D.xy);
        problem.points3D.push_back(
            reconstruction.Point3D(point2D.point3D_id).xyz);
        problem.point3D_ids.push_back(point2D.point3D_id);
        problem.camera_idxs.push_back(problem.cameras.size());
      }
    }
    problem.cameras.push_back(*image.CameraPtr());
    // Rigid camera pose in the scaled rig frame. The uniform scaling of the
    // camera frame leaves the image projections unchanged.
    problem.cams_from_rig.push_back(
        TransformCameraWorld(problem.gt_rig_from_world, image.CamFromWorld()));
  }
  return problem;
}

void MovePointBehindCamera(ScaledGeneralizedAbsolutePoseProblem* problem,
                           const size_t i) {
  const Rigid3d& cam_from_rig = problem->cams_from_rig[problem->camera_idxs[i]];
  Eigen::Vector3d point3D_in_cam =
      cam_from_rig * (problem->gt_rig_from_world * problem->points3D[i]);
  point3D_in_cam.z() = -std::abs(point3D_in_cam.z());
  problem->points3D[i] = Inverse(problem->gt_rig_from_world) *
                         (Inverse(cam_from_rig) * point3D_in_cam);
}

Sim3d PerturbSim3d(const Sim3d& tform) {
  const double rotation_noise_degree = 1;
  const double translation_noise = 0.1;
  const double scale_noise = 1.05;
  const Sim3d perturbation(scale_noise,
                           Eigen::Quaterniond(Eigen::AngleAxisd(
                               DegToRad(rotation_noise_degree),
                               RandomEigenVectord<3>().normalized())),
                           RandomEigenVectord<3>() * translation_noise);
  return perturbation * tform;
}

TEST(EstimateGeneralizedAbsolutePose, Nominal) {
  GeneralizedAbsolutePoseProblem problem =
      BuildGeneralizedAbsolutePoseProblem();
  const size_t num_points = problem.points2D.size();

  const double gt_inlier_ratio = 0.8;
  const double outlier_distance = 50;
  const size_t gt_num_inliers =
      std::max(static_cast<size_t>(gt_inlier_ratio * num_points),
               static_cast<size_t>(GP3PEstimator::kMinNumSamples));
  std::vector<size_t> shuffled_idxs(num_points);
  std::iota(shuffled_idxs.begin(), shuffled_idxs.end(), 0);
  std::shuffle(shuffled_idxs.begin(), shuffled_idxs.end(), *PRNG);

  FlatHashSet<size_t> unique_inlier_ids;
  unique_inlier_ids.reserve(gt_num_inliers);
  for (size_t i = 0; i < gt_num_inliers; ++i) {
    unique_inlier_ids.insert(problem.point3D_ids[shuffled_idxs[i]]);
  }

  std::vector<char> gt_inlier_mask(num_points, true);
  for (size_t i = gt_num_inliers; i < num_points; ++i) {
    problem.points2D[shuffled_idxs[i]] +=
        RandomEigenVectord<2>().normalized() * outlier_distance;
    gt_inlier_mask[shuffled_idxs[i]] = false;
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 2;
  ransac_options.min_inlier_ratio = gt_inlier_ratio / 2;
  ransac_options.confidence = 0.99999;

  Rigid3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateGeneralizedAbsolutePose(ransac_options,
                                              problem.points2D,
                                              problem.points3D,
                                              problem.camera_idxs,
                                              problem.cams_from_rig,
                                              problem.cameras,
                                              &rig_from_world,
                                              &num_inliers,
                                              &inlier_mask));
  EXPECT_EQ(num_inliers, unique_inlier_ids.size());
  EXPECT_EQ(inlier_mask, gt_inlier_mask);
  EXPECT_THAT(
      rig_from_world,
      Rigid3dNear(problem.gt_rig_from_world, /*rtol=*/1e-6, /*ttol=*/1e-6));
}

TEST(EstimateScaledGeneralizedAbsolutePose, Nominal) {
  ScaledGeneralizedAbsolutePoseProblem problem =
      BuildScaledGeneralizedAbsolutePoseProblem();
  const size_t num_points = problem.points2D.size();

  const double gt_inlier_ratio = 0.8;
  const double outlier_distance = 50;
  const size_t gt_num_inliers =
      std::max(static_cast<size_t>(gt_inlier_ratio * num_points),
               static_cast<size_t>(GP4PSEstimator::kMinNumSamples));
  std::vector<size_t> shuffled_idxs(num_points);
  std::iota(shuffled_idxs.begin(), shuffled_idxs.end(), 0);
  std::shuffle(shuffled_idxs.begin(), shuffled_idxs.end(), *PRNG);

  FlatHashSet<size_t> unique_inlier_ids;
  unique_inlier_ids.reserve(gt_num_inliers);
  for (size_t i = 0; i < gt_num_inliers; ++i) {
    unique_inlier_ids.insert(problem.point3D_ids[shuffled_idxs[i]]);
  }

  std::vector<char> gt_inlier_mask(num_points, true);
  for (size_t i = gt_num_inliers; i < num_points; ++i) {
    problem.points2D[shuffled_idxs[i]] +=
        RandomEigenVectord<2>().normalized() * outlier_distance;
    gt_inlier_mask[shuffled_idxs[i]] = false;
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 2;
  ransac_options.min_inlier_ratio = gt_inlier_ratio / 2;
  ransac_options.confidence = 0.99999;

  Sim3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateScaledGeneralizedAbsolutePose(ransac_options,
                                                    problem.points2D,
                                                    problem.points3D,
                                                    problem.camera_idxs,
                                                    problem.cams_from_rig,
                                                    problem.cameras,
                                                    &rig_from_world,
                                                    &num_inliers,
                                                    &inlier_mask));
  EXPECT_EQ(num_inliers, unique_inlier_ids.size());
  EXPECT_EQ(inlier_mask, gt_inlier_mask);
  EXPECT_THAT(rig_from_world,
              Sim3dNear(problem.gt_rig_from_world,
                        /*stol=*/1e-5,
                        /*rtol=*/1e-5,
                        /*ttol=*/1e-5));
}

TEST(EstimateScaledGeneralizedAbsolutePose, PanoramicRigFails) {
  ScaledGeneralizedAbsolutePoseProblem problem =
      BuildScaledGeneralizedAbsolutePoseProblem();

  // Move all cameras to a shared projection center, making the rig geometry
  // scale unobservable.
  const Eigen::Vector3d center = RandomEigenVectord<3>();
  for (Rigid3d& cam_from_rig : problem.cams_from_rig) {
    cam_from_rig.translation() = cam_from_rig.rotation() * -center;
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 2;

  Sim3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_FALSE(EstimateScaledGeneralizedAbsolutePose(ransac_options,
                                                     problem.points2D,
                                                     problem.points3D,
                                                     problem.camera_idxs,
                                                     problem.cams_from_rig,
                                                     problem.cameras,
                                                     &rig_from_world,
                                                     &num_inliers,
                                                     &inlier_mask));
}

TEST(EstimateScaledGeneralizedAbsolutePose, SingleCenterConsensusFails) {
  // The first camera at the rig origin observing a scene in front of
  // it, the second facing away from the scene and placed behind the first.
  // For any model that is consistent with the observations of the first
  // camera, the scene is scaled about the first camera's center, so it stays
  // behind the second camera. The consensus set therefore only contains
  // observations from a single projection center, for which the scale is
  // unobservable, even though the input has multiple centers.
  const Camera camera = Camera::CreateFromModelId(
      /*camera_id=*/1, CameraModelId::kPinhole, 500, 640, 480);
  const std::vector<Camera> cameras = {camera, camera};
  const std::vector<Rigid3d> cams_from_rig = {
      Rigid3d(),
      Rigid3d(Eigen::Quaterniond(
                  Eigen::AngleAxisd(EIGEN_PI, Eigen::Vector3d::UnitX())),
              Eigen::Vector3d(0, 0, -1))};
  const Sim3d gt_rig_from_world(RandomUniformReal<double>(0.5, 2),
                                RandomEigenQuaterniond(),
                                RandomEigenVectord<3>());

  constexpr int kNumPoints = 50;
  std::vector<Eigen::Vector2d> points2D;
  std::vector<Eigen::Vector3d> points3D;
  std::vector<size_t> camera_idxs;
  for (int i = 0; i < kNumPoints; ++i) {
    const Eigen::Vector2d point2D(RandomUniformReal<double>(0, camera.width),
                                  RandomUniformReal<double>(0, camera.height));
    const Eigen::Vector3d point3D_in_rig =
        RandomUniformReal<double>(1, 10) *
        Eigen::Vector3d(camera.CamFromImg(point2D)->homogeneous());
    const Eigen::Vector3d point3D = Inverse(gt_rig_from_world) * point3D_in_rig;
    points2D.push_back(point2D);
    points3D.push_back(point3D);
    camera_idxs.push_back(0);
    // The same point cannot be observed by the second camera; its arbitrary
    // observation is an outlier for any model that fits the first camera.
    points2D.emplace_back(RandomUniformReal<double>(0, camera.width),
                          RandomUniformReal<double>(0, camera.height));
    points3D.push_back(point3D);
    camera_idxs.push_back(1);
  }

  RANSACOptions ransac_options;
  ransac_options.max_error = 2;
  ransac_options.min_inlier_ratio = 0.1;

  Sim3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_FALSE(EstimateScaledGeneralizedAbsolutePose(ransac_options,
                                                     points2D,
                                                     points3D,
                                                     camera_idxs,
                                                     cams_from_rig,
                                                     cameras,
                                                     &rig_from_world,
                                                     &num_inliers,
                                                     &inlier_mask));
}

TEST(RefineGeneralizedAbsolutePose, Nominal) {
  GeneralizedAbsolutePoseProblem problem =
      BuildGeneralizedAbsolutePoseProblem();
  const std::vector<char> gt_inlier_mask(problem.points2D.size(), true);

  const double rotation_noise_degree = 1;
  const double translation_noise = 0.1;
  const Rigid3d rig_from_gt_rig(Eigen::Quaterniond(Eigen::AngleAxisd(
                                    DegToRad(rotation_noise_degree),
                                    RandomEigenVectord<3>().normalized())),
                                RandomEigenVectord<3>() * translation_noise);
  Rigid3d rig_from_world = rig_from_gt_rig * problem.gt_rig_from_world;

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  Eigen::Matrix6d rig_from_world_cov = Eigen::Matrix6d::Zero();
  EXPECT_TRUE(RefineGeneralizedAbsolutePose(options,
                                            gt_inlier_mask,
                                            problem.points2D,
                                            problem.points3D,
                                            problem.camera_idxs,
                                            problem.cams_from_rig,
                                            &rig_from_world,
                                            &problem.cameras,
                                            &rig_from_world_cov));
  EXPECT_THAT(
      rig_from_world,
      Rigid3dNear(problem.gt_rig_from_world, /*rtol=*/1e-6, /*ttol=*/1e-6));
  EXPECT_NEAR(rig_from_world.rotation().norm(), 1.0, 1e-6);
  EXPECT_NE(rig_from_world_cov, Eigen::Matrix6d::Zero());
}

TEST(RefineScaledGeneralizedAbsolutePose, Nominal) {
  ScaledGeneralizedAbsolutePoseProblem problem =
      BuildScaledGeneralizedAbsolutePoseProblem();
  const std::vector<char> gt_inlier_mask(problem.points2D.size(), true);

  Sim3d rig_from_world = PerturbSim3d(problem.gt_rig_from_world);

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  Eigen::Matrix7d rig_from_world_cov = Eigen::Matrix7d::Zero();
  const std::vector<Camera> gt_cameras = problem.cameras;
  EXPECT_TRUE(RefineScaledGeneralizedAbsolutePose(options,
                                                  gt_inlier_mask,
                                                  problem.points2D,
                                                  problem.points3D,
                                                  problem.camera_idxs,
                                                  problem.cams_from_rig,
                                                  &rig_from_world,
                                                  &problem.cameras,
                                                  &rig_from_world_cov));
  EXPECT_THAT(rig_from_world,
              Sim3dNear(problem.gt_rig_from_world,
                        /*stol=*/1e-6,
                        /*rtol=*/1e-6,
                        /*ttol=*/1e-6));
  EXPECT_GT(rig_from_world.scale(), 0);
  EXPECT_NEAR(rig_from_world.rotation().norm(), 1.0, 1e-6);
  EXPECT_NE(rig_from_world_cov, Eigen::Matrix7d::Zero());
  EXPECT_TRUE(rig_from_world_cov.allFinite());
  EXPECT_TRUE(rig_from_world_cov.isApprox(rig_from_world_cov.transpose()));
  // Cameras are not refined and must be returned unchanged.
  for (size_t i = 0; i < gt_cameras.size(); ++i) {
    EXPECT_EQ(problem.cameras[i].params, gt_cameras[i].params);
  }
}

TEST(RefineScaledGeneralizedAbsolutePose, StaleInliersAreIgnored) {
  ScaledGeneralizedAbsolutePoseProblem problem =
      BuildScaledGeneralizedAbsolutePoseProblem();

  // Inlier observations that do not project at the initial estimate must
  // neither abort the refinement nor bias it: the remaining observations are
  // exact, so the refinement must still converge to the ground truth.
  for (const size_t i : {0, 7, 20}) {
    MovePointBehindCamera(&problem, i);
  }

  Sim3d rig_from_world = PerturbSim3d(problem.gt_rig_from_world);

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;
  const std::vector<char> inlier_mask(problem.points2D.size(), true);
  EXPECT_TRUE(RefineScaledGeneralizedAbsolutePose(options,
                                                  inlier_mask,
                                                  problem.points2D,
                                                  problem.points3D,
                                                  problem.camera_idxs,
                                                  problem.cams_from_rig,
                                                  &rig_from_world,
                                                  &problem.cameras));
  EXPECT_THAT(rig_from_world,
              Sim3dNear(problem.gt_rig_from_world,
                        /*stol=*/1e-6,
                        /*rtol=*/1e-6,
                        /*ttol=*/1e-6));
}

TEST(EstimateScaledGeneralizedAbsolutePose, EmptyInputsFail) {
  ScaledGeneralizedAbsolutePoseProblem problem =
      BuildScaledGeneralizedAbsolutePoseProblem();

  RANSACOptions ransac_options;
  ransac_options.max_error = 2;

  Sim3d rig_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_FALSE(EstimateScaledGeneralizedAbsolutePose(ransac_options,
                                                     /*points2D=*/{},
                                                     /*points3D=*/{},
                                                     /*camera_idxs=*/{},
                                                     problem.cams_from_rig,
                                                     problem.cameras,
                                                     &rig_from_world,
                                                     &num_inliers,
                                                     &inlier_mask));
}

TEST(RefineScaledGeneralizedAbsolutePose, PanoramicInliersFail) {
  ScaledGeneralizedAbsolutePoseProblem problem =
      BuildScaledGeneralizedAbsolutePoseProblem();

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;

  // The scale is unobservable if the inlier mask only selects observations
  // from a single projection center, so the arbitrary initial scale must not
  // be reported as successfully refined.
  const double initial_scale = 2;
  Sim3d rig_from_world(initial_scale,
                       problem.gt_rig_from_world.rotation(),
                       problem.gt_rig_from_world.translation());
  std::vector<char> single_camera_inlier_mask(problem.points2D.size());
  for (size_t i = 0; i < problem.camera_idxs.size(); ++i) {
    single_camera_inlier_mask[i] = problem.camera_idxs[i] == 0;
  }
  EXPECT_FALSE(RefineScaledGeneralizedAbsolutePose(options,
                                                   single_camera_inlier_mask,
                                                   problem.points2D,
                                                   problem.points3D,
                                                   problem.camera_idxs,
                                                   problem.cams_from_rig,
                                                   &rig_from_world,
                                                   &problem.cameras));
  EXPECT_EQ(rig_from_world.scale(), initial_scale);

  // Same for a rig whose cameras share one projection center.
  std::vector<Rigid3d> central_cams_from_rig = problem.cams_from_rig;
  const Eigen::Vector3d center = RandomEigenVectord<3>();
  for (Rigid3d& cam_from_rig : central_cams_from_rig) {
    cam_from_rig.translation() = cam_from_rig.rotation() * -center;
  }
  const std::vector<char> all_inlier_mask(problem.points2D.size(), true);
  EXPECT_FALSE(RefineScaledGeneralizedAbsolutePose(options,
                                                   all_inlier_mask,
                                                   problem.points2D,
                                                   problem.points3D,
                                                   problem.camera_idxs,
                                                   central_cams_from_rig,
                                                   &rig_from_world,
                                                   &problem.cameras));
  EXPECT_EQ(rig_from_world.scale(), initial_scale);

  // An empty inlier set leaves the problem unconstrained.
  const std::vector<char> empty_inlier_mask(problem.points2D.size(), false);
  EXPECT_FALSE(RefineScaledGeneralizedAbsolutePose(options,
                                                   empty_inlier_mask,
                                                   problem.points2D,
                                                   problem.points3D,
                                                   problem.camera_idxs,
                                                   problem.cams_from_rig,
                                                   &rig_from_world,
                                                   &problem.cameras));
  EXPECT_EQ(rig_from_world.scale(), initial_scale);
}

TEST(RefineScaledGeneralizedAbsolutePose, PointsBehindCamerasFail) {
  ScaledGeneralizedAbsolutePoseProblem problem =
      BuildScaledGeneralizedAbsolutePoseProblem();

  // Move every 3D point behind its observing camera. The reprojection cost
  // cannot be evaluated at the initial estimate, so refinement must fail
  // instead of accepting the invalid pose as a perfect fit.
  for (size_t i = 0; i < problem.points3D.size(); ++i) {
    const Rigid3d& cam_from_rig = problem.cams_from_rig[problem.camera_idxs[i]];
    Eigen::Vector3d point3D_in_cam =
        cam_from_rig * (problem.gt_rig_from_world * problem.points3D[i]);
    point3D_in_cam.z() = -std::abs(point3D_in_cam.z());
    problem.points3D[i] = Inverse(problem.gt_rig_from_world) *
                          (Inverse(cam_from_rig) * point3D_in_cam);
  }

  AbsolutePoseRefinementOptions options;
  options.refine_focal_length = false;
  options.refine_extra_params = false;

  Sim3d rig_from_world = problem.gt_rig_from_world;
  const std::vector<char> inlier_mask(problem.points2D.size(), true);
  EXPECT_FALSE(RefineScaledGeneralizedAbsolutePose(options,
                                                   inlier_mask,
                                                   problem.points2D,
                                                   problem.points3D,
                                                   problem.camera_idxs,
                                                   problem.cams_from_rig,
                                                   &rig_from_world,
                                                   &problem.cameras));
}

TEST(RefineGeneralizedAbsolutePose, PositionPrior) {
  GeneralizedAbsolutePoseProblem problem =
      BuildGeneralizedAbsolutePoseProblem();
  // Isolate the position-prior-only refinement path without reprojection terms.
  const std::vector<char> inlier_mask(problem.points2D.size(), false);

  AbsolutePoseRefinementOptions options;
  options.use_position_prior = true;
  options.position_prior_in_world = Eigen::Vector3d(1.0, 2.0, 3.0);
  options.position_prior_covariance = Eigen::Matrix3d::Identity();
  Rigid3d rig_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(0.3, -0.5, 0.7));
  auto compute_position_error = [&](const Rigid3d& rig_from_world_to_check) {
    return (Inverse(rig_from_world_to_check).translation() -
            options.position_prior_in_world)
        .norm();
  };
  const double initial_error = compute_position_error(rig_from_world);
  EXPECT_TRUE(RefineGeneralizedAbsolutePose(options,
                                            inlier_mask,
                                            problem.points2D,
                                            problem.points3D,
                                            problem.camera_idxs,
                                            problem.cams_from_rig,
                                            &rig_from_world,
                                            &problem.cameras));
  EXPECT_LT(compute_position_error(rig_from_world), initial_error);
  EXPECT_NEAR(rig_from_world.rotation().norm(), 1.0, 1e-6);
}

TEST(RefineGeneralizedAbsolutePose, PositionPriorCovariance) {
  GeneralizedAbsolutePoseProblem problem =
      BuildGeneralizedAbsolutePoseProblem();
  const std::vector<char> inlier_mask(problem.points2D.size(), true);

  AbsolutePoseRefinementOptions weak_prior_options;
  weak_prior_options.use_position_prior = true;
  weak_prior_options.position_prior_in_world =
      Inverse(problem.gt_rig_from_world).translation() +
      Eigen::Vector3d(1.0, -0.7, 0.5);
  // Large covariance = weak prior (high uncertainty).
  weak_prior_options.position_prior_covariance = Eigen::Matrix3d::Identity();

  AbsolutePoseRefinementOptions strong_prior_options = weak_prior_options;
  // Small covariance = strong prior (low uncertainty).
  strong_prior_options.position_prior_covariance =
      0.01 * Eigen::Matrix3d::Identity();

  const double rotation_noise_degree = 1;
  const double translation_noise = 0.1;
  const Rigid3d rig_from_gt_rig(Eigen::Quaterniond(Eigen::AngleAxisd(
                                    DegToRad(rotation_noise_degree),
                                    RandomEigenVectord<3>().normalized())),
                                RandomEigenVectord<3>() * translation_noise);
  const Rigid3d initial_rig_from_world =
      rig_from_gt_rig * problem.gt_rig_from_world;

  std::vector<Camera> weak_prior_cameras = problem.cameras;
  std::vector<Camera> strong_prior_cameras = problem.cameras;
  Rigid3d weak_prior_rig_from_world = initial_rig_from_world;
  Rigid3d strong_prior_rig_from_world = initial_rig_from_world;

  EXPECT_TRUE(RefineGeneralizedAbsolutePose(weak_prior_options,
                                            inlier_mask,
                                            problem.points2D,
                                            problem.points3D,
                                            problem.camera_idxs,
                                            problem.cams_from_rig,
                                            &weak_prior_rig_from_world,
                                            &weak_prior_cameras));
  EXPECT_NEAR(weak_prior_rig_from_world.rotation().norm(), 1.0, 1e-6);
  EXPECT_TRUE(RefineGeneralizedAbsolutePose(strong_prior_options,
                                            inlier_mask,
                                            problem.points2D,
                                            problem.points3D,
                                            problem.camera_idxs,
                                            problem.cams_from_rig,
                                            &strong_prior_rig_from_world,
                                            &strong_prior_cameras));
  EXPECT_NEAR(strong_prior_rig_from_world.rotation().norm(), 1.0, 1e-6);

  auto compute_position_error = [&](const Rigid3d& rig_from_world_to_check) {
    return (Inverse(rig_from_world_to_check).translation() -
            weak_prior_options.position_prior_in_world)
        .norm();
  };
  const double weak_prior_error =
      compute_position_error(weak_prior_rig_from_world);
  const double strong_prior_error =
      compute_position_error(strong_prior_rig_from_world);
  EXPECT_LT(strong_prior_error, weak_prior_error);
}

struct GeneralizedRelativePoseProblem {
  Rigid3d gt_rig2_from_rig1;
  std::vector<Eigen::Vector2d> points2D1;
  std::vector<Eigen::Vector2d> points2D2;
  std::vector<size_t> camera_idxs1;
  std::vector<size_t> camera_idxs2;
  std::vector<Rigid3d> cams_from_rig;
  std::vector<Camera> cameras;
};

GeneralizedRelativePoseProblem BuildGeneralizedRelativePoseProblem(
    int num_cameras_per_rig1,
    int num_cameras_per_rig2,
    double sensor_from_rig_translation_stddev) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig =
      std::max(num_cameras_per_rig1, num_cameras_per_rig2);
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      sensor_from_rig_translation_stddev;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 10;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const Frame& frame1 = reconstruction.Frame(1);
  const Frame& frame2 = reconstruction.Frame(2);
  CHECK_NE(frame1.RigId(), frame2.RigId());

  GeneralizedRelativePoseProblem problem;
  problem.gt_rig2_from_rig1 =
      frame2.RigFromWorld() * Inverse(frame1.RigFromWorld());

  FlatHashMap<point3D_t, std::vector<std::pair<const Image*, point2D_t>>>
      observations2;
  for (const data_t& data_id : frame2.ImageIds()) {
    const auto& image = reconstruction.Image(data_id.id);
    for (size_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const auto& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        observations2[point2D.point3D_id].emplace_back(&image, point2D_idx);
      }
    }
    if (--num_cameras_per_rig2 == 0) {
      break;
    }
  }

  NodeHashMap<camera_t, size_t> camera_id_to_idx;
  for (const data_t& data_id : frame1.ImageIds()) {
    const auto& image1 = reconstruction.Image(data_id.id);
    for (size_t point2D_idx1 = 0; point2D_idx1 < image1.NumPoints2D();
         ++point2D_idx1) {
      const auto& point2D1 = image1.Point2D(point2D_idx1);
      const auto observation_it = observations2.find(point2D1.point3D_id);
      if (observation_it == observations2.end()) {
        continue;
      }

      auto maybe_add_and_get_camera = [&problem,
                                       &camera_id_to_idx](const Image& image) {
        auto [it, inserted] =
            camera_id_to_idx.emplace(image.CameraId(), problem.cameras.size());
        if (inserted) {
          problem.cameras.push_back(*image.CameraPtr());
          const Rig& rig = *image.FramePtr()->RigPtr();
          if (rig.IsRefSensor(image.CameraPtr()->SensorId())) {
            problem.cams_from_rig.push_back(Rigid3d());
          } else {
            problem.cams_from_rig.push_back(
                image.FramePtr()->RigPtr()->SensorFromRig(
                    image.CameraPtr()->SensorId()));
          }
        }
        return it->second;
      };

      for (const auto& [image2_ptr, point2D_idx2] : observation_it->second) {
        problem.points2D1.push_back(point2D1.xy);
        problem.points2D2.push_back(image2_ptr->Point2D(point2D_idx2).xy);
        problem.camera_idxs1.push_back(maybe_add_and_get_camera(image1));
        problem.camera_idxs2.push_back(maybe_add_and_get_camera(*image2_ptr));
      }
    }

    if (--num_cameras_per_rig1 == 0) {
      break;
    }
  }

  return problem;
}

TEST(EstimateGeneralizedRelativePose, Nominal) {
  for (const int num_cameras_per_rig1 : {1, 2, 3}) {
    for (const int num_cameras_per_rig2 : {1, 2, 3}) {
      // A meaningful inter-camera baseline is needed to recover metric scale
      // reliably in non-panoramic configurations.
      for (const double sensor_from_rig_translation_stddev : {0.0, 0.2}) {
        GeneralizedRelativePoseProblem problem =
            BuildGeneralizedRelativePoseProblem(
                num_cameras_per_rig1,
                num_cameras_per_rig2,
                sensor_from_rig_translation_stddev);

        RANSACOptions ransac_options;
        ransac_options.max_error = 1;

        std::optional<Rigid3d> rig2_from_rig1;
        std::optional<Rigid3d> pano2_from_pano1;
        size_t num_inliers;
        std::vector<char> inlier_mask;
        EXPECT_TRUE(EstimateGeneralizedRelativePose(ransac_options,
                                                    problem.points2D1,
                                                    problem.points2D2,
                                                    problem.camera_idxs1,
                                                    problem.camera_idxs2,
                                                    problem.cams_from_rig,
                                                    problem.cameras,
                                                    &rig2_from_rig1,
                                                    &pano2_from_pano1,
                                                    &num_inliers,
                                                    &inlier_mask));
        EXPECT_EQ(num_inliers, problem.points2D1.size());
        EXPECT_THAT(inlier_mask, testing::Each(testing::Eq(true)));
        if ((num_cameras_per_rig1 == 1 && num_cameras_per_rig2 == 1) ||
            sensor_from_rig_translation_stddev == 0) {
          // Panoramic pairs do not allow for recovery of translation scale.
          ASSERT_FALSE(rig2_from_rig1.has_value());
          ASSERT_TRUE(pano2_from_pano1.has_value());
          EXPECT_THAT(
              *pano2_from_pano1,
              Rigid3dNear(
                  Rigid3d(problem.gt_rig2_from_rig1.rotation(),
                          problem.gt_rig2_from_rig1.translation().normalized()),
                  /*rtol=*/1e-6,
                  /*ttol=*/1e-6));
        } else {
          ASSERT_TRUE(rig2_from_rig1.has_value());
          ASSERT_FALSE(pano2_from_pano1.has_value());
          EXPECT_THAT(
              *rig2_from_rig1,
              Rigid3dNear(
                  problem.gt_rig2_from_rig1, /*rtol=*/1e-3, /*ttol=*/2e-3));
        }
      }
    }
  }
}

struct StructureLessAbsolutePoseProblem {
  Rigid3d gt_cam_from_world;
  std::vector<Eigen::Vector2d> world_points2D;
  std::vector<Eigen::Vector2d> query_points2D;
  std::vector<size_t> world_camera_idxs;
  std::vector<Rigid3d> world_cams_from_world;
  std::vector<Camera> world_cameras;
  Camera query_camera;
};

StructureLessAbsolutePoseProblem BuildStructureLessAbsolutePoseProblem(
    int num_world_cams) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = num_world_cams + 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  const image_t query_image_id = reconstruction.RegImageIds()[0];
  const Image& query_image = reconstruction.Image(query_image_id);

  StructureLessAbsolutePoseProblem problem;
  problem.gt_cam_from_world = query_image.CamFromWorld();
  problem.query_camera = *query_image.CameraPtr();

  // Build mapping of world cameras
  NodeHashMap<image_t, size_t> world_image_id_to_camera_idx;
  FlatHashMap<point3D_t, std::vector<std::pair<const Image*, point2D_t>>>
      world_obs;

  for (const image_t world_image_id : reconstruction.RegImageIds()) {
    if (world_image_id == query_image_id) {
      continue;
    }

    const auto& world_image = reconstruction.Image(world_image_id);

    if (world_image_id_to_camera_idx.find(world_image.ImageId()) ==
        world_image_id_to_camera_idx.end()) {
      world_image_id_to_camera_idx[world_image.ImageId()] =
          problem.world_cameras.size();
      problem.world_cameras.push_back(*world_image.CameraPtr());
      problem.world_cams_from_world.push_back(world_image.CamFromWorld());
    }

    for (size_t point2D_idx = 0; point2D_idx < world_image.NumPoints2D();
         ++point2D_idx) {
      const auto& point2D = world_image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        world_obs[point2D.point3D_id].emplace_back(&world_image, point2D_idx);
      }
    }
  }

  for (size_t point2D_idx = 0; point2D_idx < query_image.NumPoints2D();
       ++point2D_idx) {
    const auto& query_point2D = query_image.Point2D(point2D_idx);
    if (!query_point2D.HasPoint3D()) {
      continue;
    }

    const auto world_obs_it = world_obs.find(query_point2D.point3D_id);
    if (world_obs_it == world_obs.end()) {
      continue;
    }

    for (const auto& [world_image, world_point2D_idx] : world_obs_it->second) {
      const auto& world_point2D = world_image->Point2D(world_point2D_idx);
      problem.world_points2D.push_back(world_point2D.xy);
      problem.query_points2D.push_back(query_point2D.xy);
      problem.world_camera_idxs.push_back(
          world_image_id_to_camera_idx[world_image->ImageId()]);
    }
  }

  return problem;
}

TEST(EstimateStructureLessAbsolutePose, Nominal) {
  const StructureLessAbsolutePoseProblem problem =
      BuildStructureLessAbsolutePoseProblem(/*num_world_cams=*/5);

  StructureLessAbsolutePoseEstimationOptions options;
  Rigid3d cam_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateStructureLessAbsolutePose(options,
                                                problem.query_points2D,
                                                problem.world_points2D,
                                                problem.world_camera_idxs,
                                                problem.world_cams_from_world,
                                                problem.world_cameras,
                                                problem.query_camera,
                                                &cam_from_world,
                                                &num_inliers,
                                                &inlier_mask));
  EXPECT_EQ(num_inliers, problem.world_points2D.size());
  EXPECT_EQ(inlier_mask.size(), problem.world_points2D.size());
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.gt_cam_from_world, /*rtol=*/1e-6, /*ttol=*/1e-6));
}

TEST(EstimateStructureLessAbsolutePose, WithOutliers) {
  StructureLessAbsolutePoseProblem problem =
      BuildStructureLessAbsolutePoseProblem(/*num_world_cams=*/10);

  // Add outliers by perturbing some query observations.
  const double kOutlierRatio = 0.3;
  const size_t num_outliers =
      static_cast<size_t>(kOutlierRatio * problem.query_points2D.size());
  std::vector<size_t> shuffled_idxs(problem.query_points2D.size());
  std::iota(shuffled_idxs.begin(), shuffled_idxs.end(), 0);
  std::shuffle(shuffled_idxs.begin(), shuffled_idxs.end(), *PRNG);
  for (size_t i = 0; i < num_outliers; ++i) {
    problem.query_points2D[shuffled_idxs[i]] += Eigen::Vector2d(1000, 1000);
  }

  StructureLessAbsolutePoseEstimationOptions options;
  options.ransac_options.max_error = 1.0;  // pixels
  Rigid3d cam_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_TRUE(EstimateStructureLessAbsolutePose(options,
                                                problem.query_points2D,
                                                problem.world_points2D,
                                                problem.world_camera_idxs,
                                                problem.world_cams_from_world,
                                                problem.world_cameras,
                                                problem.query_camera,
                                                &cam_from_world,
                                                &num_inliers,
                                                &inlier_mask));
  EXPECT_GT(num_inliers,
            problem.world_points2D.size() * (1 - kOutlierRatio) * 0.9);
  EXPECT_THAT(
      cam_from_world,
      Rigid3dNear(problem.gt_cam_from_world, /*rtol=*/5e-3, /*ttol=*/5e-3));
}

TEST(EstimateStructureLessAbsolutePose, PanoramicWorldCameras) {
  const StructureLessAbsolutePoseProblem problem =
      BuildStructureLessAbsolutePoseProblem(/*num_world_cams=*/1);

  StructureLessAbsolutePoseEstimationOptions options;
  Rigid3d cam_from_world;
  size_t num_inliers;
  std::vector<char> inlier_mask;
  EXPECT_FALSE(EstimateStructureLessAbsolutePose(options,
                                                 problem.query_points2D,
                                                 problem.world_points2D,
                                                 problem.world_camera_idxs,
                                                 problem.world_cams_from_world,
                                                 problem.world_cameras,
                                                 problem.query_camera,
                                                 &cam_from_world,
                                                 &num_inliers,
                                                 &inlier_mask));
}

}  // namespace
}  // namespace colmap
