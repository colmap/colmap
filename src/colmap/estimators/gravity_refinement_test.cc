// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/gravity_refinement.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/random.h"
#include "colmap/math/random_eigen.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void LoadReconstructionAndPoseGraph(const Database& database,
                                    Reconstruction* reconstruction,
                                    PoseGraph* pose_graph) {
  DatabaseCache database_cache;
  DatabaseCache::Options options;
  database_cache.Load(database, options);
  reconstruction->Load(database_cache);
  pose_graph->Load(*database_cache.CorrespondenceGraph());
}

void SynthesizeGravityOutliers(std::vector<PosePrior>& pose_priors,
                               double outlier_ratio = 0.0) {
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.HasGravity() &&
        RandomUniformReal<double>(0, 1) < outlier_ratio) {
      pose_prior.gravity = RandomEigenVectord<3>().normalized();
    }
  }
}

void ExpectEqualGravity(const Eigen::Vector3d& gravity_in_world,
                        const Reconstruction& gt,
                        const std::vector<PosePrior>& pose_priors,
                        const double max_gravity_error_deg) {
  const double max_gravity_error_rad = DegToRad(max_gravity_error_deg);
  NodeHashMap<image_t, const PosePrior*> image_to_pose_prior;
  for (const auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      image_to_pose_prior.emplace(pose_prior.corr_data_id.id, &pose_prior);
    }
  }
  for (const auto& image_id : gt.RegImageIds()) {
    const auto& image = gt.Image(image_id);
    if (!image.IsRefInFrame()) {
      continue;
    }
    const Eigen::Vector3d gravity_gt =
        gt.Image(image_id).CamFromWorld().rotation() * gravity_in_world;
    const Eigen::Vector3d gravity_computed =
        image_to_pose_prior.at(image_id)->gravity;
    const double gravity_error_rad =
        CalculateAngleBetweenVectors(gravity_gt, gravity_computed);
    EXPECT_LT(gravity_error_rad, max_gravity_error_rad);
  }
}

TEST(GravityRefinement, RefineGravity) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GravityRefinerOptions opt_grav_refine;
  RunGravityRefinement(
      opt_grav_refine, pose_graph, reconstruction, pose_priors);

  ExpectEqualGravity(synthetic_dataset_options.prior_gravity_in_world,
                     gt_reconstruction,
                     pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

TEST(GravityRefinement, RefineGravityWithNonTrivialRigs) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  Reconstruction reconstruction;
  PoseGraph pose_graph;
  LoadReconstructionAndPoseGraph(*database, &reconstruction, &pose_graph);

  std::vector<PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GravityRefinerOptions opt_grav_refine;
  RunGravityRefinement(
      opt_grav_refine, pose_graph, reconstruction, pose_priors);

  ExpectEqualGravity(synthetic_dataset_options.prior_gravity_in_world,
                     gt_reconstruction,
                     pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

}  // namespace
}  // namespace colmap
