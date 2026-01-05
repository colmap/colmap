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

#include "glomap/estimators/gravity_refinement.h"

#include "colmap/geometry/triangulation.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include "glomap/scene/view_graph.h"

#include <gtest/gtest.h>

namespace glomap {
namespace {

void LoadReconstructionAndViewGraph(const colmap::Database& database,
                                    colmap::Reconstruction* reconstruction,
                                    ViewGraph* view_graph) {
  colmap::DatabaseCache database_cache;
  database_cache.Load(database, /*min_num_matches=*/0);
  reconstruction->Load(database_cache);
  view_graph->LoadFromDatabase(database);
}

void SynthesizeGravityOutliers(std::vector<colmap::PosePrior>& pose_priors,
                               double outlier_ratio = 0.0) {
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.HasGravity() &&
        colmap::RandomUniformReal<double>(0, 1) < outlier_ratio) {
      pose_prior.gravity = Eigen::Vector3d::Random().normalized();
    }
  }
}

void ExpectEqualGravity(const Eigen::Vector3d& gravity_in_world,
                        const colmap::Reconstruction& gt,
                        const std::vector<colmap::PosePrior>& pose_priors,
                        const double max_gravity_error_deg) {
  const double max_gravity_error_rad = colmap::DegToRad(max_gravity_error_deg);
  std::unordered_map<image_t, const colmap::PosePrior*> image_to_pose_prior;
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
        gt.Image(image_id).CamFromWorld().rotation * gravity_in_world;
    const Eigen::Vector3d gravity_computed =
        image_to_pose_prior.at(image_id)->gravity;
    const double gravity_error_rad =
        colmap::CalculateAngleBetweenVectors(gravity_gt, gravity_computed);
    EXPECT_LT(gravity_error_rad, max_gravity_error_rad);
  }
}

TEST(GravityRefinement, RefineGravity) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  colmap::Reconstruction reconstruction;
  ViewGraph view_graph;
  LoadReconstructionAndViewGraph(*database, &reconstruction, &view_graph);

  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(view_graph, reconstruction, pose_priors);

  ExpectEqualGravity(synthetic_dataset_options.prior_gravity_in_world,
                     gt_reconstruction,
                     pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

TEST(GravityRefinement, RefineGravityWithNonTrivialRigs) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 25;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.prior_gravity = true;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  colmap::Reconstruction reconstruction;
  ViewGraph view_graph;
  LoadReconstructionAndViewGraph(*database, &reconstruction, &view_graph);

  std::vector<colmap::PosePrior> pose_priors = database->ReadAllPosePriors();
  SynthesizeGravityOutliers(pose_priors, /*outlier_ratio=*/0.3);

  GravityRefinerOptions opt_grav_refine;
  GravityRefiner grav_refiner(opt_grav_refine);
  grav_refiner.RefineGravity(view_graph, reconstruction, pose_priors);

  ExpectEqualGravity(synthetic_dataset_options.prior_gravity_in_world,
                     gt_reconstruction,
                     pose_priors,
                     /*max_gravity_error_deg=*/1e-2);
}

}  // namespace
}  // namespace glomap
