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

#include "colmap/controllers/global_pipeline.h"

#include "colmap/estimators/view_graph_calibration.h"
#include "colmap/math/random_eigen.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// TODO(jsch): Create parameterized tests for the different mapper
// implementations (incremental, hierarchical, global)
TEST(GlobalPipeline, Nominal) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  GlobalPipelineOptions options;
  ViewGraphCalibrationOptions vgc_options;
  CalibrateViewGraph(vgc_options, database.get());
  GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  auto reconstruction = reconstruction_manager->Get(0);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));

  // After the pipeline runs, point3D.error must be in pixel units, i.e.
  // equal to what UpdatePoint3DErrors would recompute.
  ASSERT_GT(reconstruction->NumPoints3D(), 0u);
  const double mean_after_run = reconstruction->ComputeMeanReprojectionError();
  reconstruction->UpdatePoint3DErrors();
  EXPECT_DOUBLE_EQ(mean_after_run,
                   reconstruction->ComputeMeanReprojectionError());
}

TEST(GlobalPipeline, SfMWithRandomSeedStability) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    GlobalPipelineOptions options;
    options.num_threads = num_threads;
    options.random_seed = random_seed;
    ViewGraphCalibrationOptions vgc_options;
    vgc_options.random_seed = random_seed;
    vgc_options.solver_options.num_threads = num_threads;
    CalibrateViewGraph(vgc_options, database.get());
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
    mapper.Run();
    EXPECT_EQ(reconstruction_manager->Size(), 1);
    return reconstruction_manager;
  };

  constexpr int kRandomSeed = 42;

  // Single-threaded execution.
  {
    auto reconstruction_manager0 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    auto reconstruction_manager1 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    EXPECT_THAT(*reconstruction_manager0->Get(0),
                ReconstructionEq(*reconstruction_manager1->Get(0)));
  }

  // Multi-threaded execution.
  {
    auto reconstruction_manager0 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    auto reconstruction_manager1 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    // Same seed should produce similar results, up to floating-point variations
    // in optimization.
    EXPECT_THAT(*reconstruction_manager0->Get(0),
                ReconstructionNear(*reconstruction_manager1->Get(0),
                                   /*max_rotation_error_deg=*/1e-9,
                                   /*max_proj_center_error=*/1e-9,
                                   /*max_scale_error=*/std::nullopt,
                                   /*num_obs_tolerance=*/0.01,
                                   /*align=*/false));
  }
}

TEST(GlobalPipeline, WithExistingRelativePoses) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  GlobalPipelineOptions options;
  ViewGraphCalibrationOptions vgc_options;
  CalibrateViewGraph(vgc_options, database.get());
  GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

// To test relative pose re-estimation from view graph calibration.
TEST(GlobalPipeline, WithNoisyExistingRelativePoses) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Replace relative poses with completely random values.
  for (auto& [pair_id, two_view_geometry] : database->ReadTwoViewGeometries()) {
    if (!two_view_geometry.cam2_from_cam1.has_value()) {
      continue;
    }
    two_view_geometry.cam2_from_cam1->rotation() = RandomEigenQuaterniond();
    two_view_geometry.cam2_from_cam1->translation() =
        RandomEigenVectord<3>().normalized();

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    database->UpdateTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  }

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  GlobalPipelineOptions options;
  ViewGraphCalibrationOptions vgc_options;
  CalibrateViewGraph(vgc_options, database.get());
  GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  // Expect slightly worse accuracy due to noisy input poses.
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction_manager->Get(0),
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

// Returns the set of registered image ids for each reconstruction managed by
// `reconstruction_manager`.
std::vector<std::unordered_set<image_t>> RegImageIdSetsPerReconstruction(
    const ReconstructionManager& reconstruction_manager) {
  std::vector<std::unordered_set<image_t>> image_id_sets;
  for (size_t i = 0; i < reconstruction_manager.Size(); ++i) {
    const std::vector<image_t> reg_image_ids =
        reconstruction_manager.Get(i)->RegImageIds();
    image_id_sets.emplace_back(reg_image_ids.begin(), reg_image_ids.end());
  }
  return image_id_sets;
}

// Groups the registered images of `reconstruction` by their rig.
std::vector<std::unordered_set<image_t>> GroupImageIdsByRig(
    const Reconstruction& reconstruction) {
  std::unordered_map<rig_t, std::unordered_set<image_t>> images_by_rig;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    for (const data_t& data_id : frame.ImageIds()) {
      images_by_rig[frame.RigId()].insert(data_id.id);
    }
  }
  std::vector<std::unordered_set<image_t>> groups;
  for (auto& [rig_id, image_ids] : images_by_rig) {
    groups.push_back(std::move(image_ids));
  }
  return groups;
}

// Builds a ground-truth sub-reconstruction restricted to `group_image_ids` by
// de-registering all frames outside the group and tearing down the leftover
// images, frames, rigs, and cameras. The group must align with frame/rig
// boundaries (e.g. one full rig) so the result is internally consistent.
Reconstruction ExtractGroundTruthSubset(
    const Reconstruction& gt_reconstruction,
    const std::unordered_set<image_t>& group_image_ids) {
  Reconstruction subset = gt_reconstruction;
  std::vector<frame_t> frames_to_deregister;
  for (const auto& [frame_id, frame] : subset.Frames()) {
    const bool in_group =
        std::any_of(frame.ImageIds().begin(),
                    frame.ImageIds().end(),
                    [&](const data_t& data_id) {
                      return group_image_ids.count(data_id.id) > 0;
                    });
    if (!in_group) {
      frames_to_deregister.push_back(frame_id);
    }
  }
  for (const frame_t frame_id : frames_to_deregister) {
    subset.DeRegisterFrame(frame_id);
  }
  subset.TearDown();
  return subset;
}

// Deletes all two-view geometries and matches connecting images from different
// groups so the view graph in the database splits into disconnected components.
void DisconnectDatabaseComponents(
    const std::vector<std::unordered_set<image_t>>& groups,
    Database& database) {
  std::unordered_map<image_t, int> image_to_group;
  for (int group = 0; group < static_cast<int>(groups.size()); ++group) {
    for (const image_t image_id : groups[group]) {
      image_to_group[image_id] = group;
    }
  }
  for (const auto& [pair_id, two_view_geometry] :
       database.ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (image_to_group.at(image_id1) != image_to_group.at(image_id2)) {
      database.DeleteTwoViewGeometry(image_id1, image_id2);
      database.DeleteInlierMatches(image_id1, image_id2);
      database.DeleteMatches(image_id1, image_id2);
    }
  }
}

// Bridges the given image groups with `num_outlier_edges` cross-group two-view
// geometries whose relative rotations are randomized (outliers), and deletes
// all other cross-group edges. The kept outlier edges connect the groups into a
// single initial connected component that rotation averaging must split by
// filtering the outliers.
void BridgeGroupsWithOutlierEdges(
    const std::vector<std::unordered_set<image_t>>& groups,
    int num_outlier_edges,
    Database& database) {
  std::unordered_map<image_t, int> image_to_group;
  for (int group = 0; group < static_cast<int>(groups.size()); ++group) {
    for (const image_t image_id : groups[group]) {
      image_to_group[image_id] = group;
    }
  }
  int num_kept = 0;
  for (auto [pair_id, two_view_geometry] : database.ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (image_to_group.at(image_id1) == image_to_group.at(image_id2)) {
      continue;  // Keep intra-group edges untouched.
    }
    if (num_kept < num_outlier_edges &&
        two_view_geometry.cam2_from_cam1.has_value()) {
      // Corrupt the relative rotation so this bridge edge is an outlier.
      two_view_geometry.cam2_from_cam1->rotation() = RandomEigenQuaterniond();
      database.UpdateTwoViewGeometry(image_id1, image_id2, two_view_geometry);
      ++num_kept;
    } else {
      database.DeleteTwoViewGeometry(image_id1, image_id2);
      database.DeleteInlierMatches(image_id1, image_id2);
      database.DeleteMatches(image_id1, image_id2);
    }
  }
}

// End-to-end: a database whose view graph splits into two disconnected
// components should yield one reconstruction per component, each registering
// exactly that component's images.
TEST(GlobalPipeline, MultiComponents) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.camera_has_prior_focal_length = false;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Split the images into two groups (one per rig) and cut all cross-group
  // matches so the view graph decomposes into two connected components.
  // Grouping by rig keeps each component's ground truth well-defined.
  const std::vector<std::unordered_set<image_t>> expected_components =
      GroupImageIdsByRig(gt_reconstruction);
  ASSERT_EQ(expected_components.size(), 2);
  ASSERT_EQ(expected_components[0].size(), 5);
  ASSERT_EQ(expected_components[1].size(), 5);
  DisconnectDatabaseComponents(expected_components, *database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  GlobalPipelineOptions options;
  ASSERT_TRUE(options.reconstruct_all_components);
  ViewGraphCalibrationOptions vgc_options;
  CalibrateViewGraph(vgc_options, database.get());
  GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
  mapper.Run();

  // Expect one reconstruction per component, each covering its own images.
  ASSERT_EQ(reconstruction_manager->Size(), 2);
  EXPECT_THAT(RegImageIdSetsPerReconstruction(*reconstruction_manager),
              testing::UnorderedElementsAreArray(expected_components));

  // Each recovered component must also match the ground truth of its cluster.
  for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
    const Reconstruction& reconstruction = *reconstruction_manager->Get(i);
    const std::vector<image_t> reg_image_ids = reconstruction.RegImageIds();
    const std::unordered_set<image_t> reconstruction_image_ids(
        reg_image_ids.begin(), reg_image_ids.end());
    const auto group_it = std::find(expected_components.begin(),
                                    expected_components.end(),
                                    reconstruction_image_ids);
    ASSERT_NE(group_it, expected_components.end());
    const Reconstruction gt_subset =
        ExtractGroundTruthSubset(gt_reconstruction, *group_it);
    EXPECT_THAT(gt_subset,
                ReconstructionNear(reconstruction,
                                   /*max_rotation_error_deg=*/1e-2,
                                   /*max_proj_center_error=*/1e-4));
  }
}

// End-to-end: two clusters bridged only by a couple of outlier edges (with
// bogus relative rotations) are still recovered as two separate
// reconstructions. The two mutually-inconsistent bridges cannot both be
// satisfied by any global rotation solution, so rotation averaging leaves each
// with a large residual and FilterEdgesByRelativeRotation removes them,
// splitting the view graph.
TEST(GlobalPipeline, MultiComponentsWithOutlierEdges) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  // Known focal lengths let us skip view graph calibration, which would
  // otherwise re-estimate (and thereby "fix") the injected outlier edges.
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Split the images into two groups.
  std::vector<image_t> image_ids = gt_reconstruction.RegImageIds();
  std::sort(image_ids.begin(), image_ids.end());
  ASSERT_EQ(image_ids.size(), 10);
  const std::vector<std::unordered_set<image_t>> expected_components = {
      {image_ids.begin(), image_ids.begin() + 5},
      {image_ids.begin() + 5, image_ids.end()}};

  // Connect the two groups only through two outlier bridge edges.
  BridgeGroupsWithOutlierEdges(
      expected_components, /*num_outlier_edges=*/2, *database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  GlobalPipelineOptions options;
  ASSERT_TRUE(options.reconstruct_all_components);
  GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
  mapper.Run();

  // The outlier bridges must be rejected, recovering the two clusters.
  ASSERT_EQ(reconstruction_manager->Size(), 2);
  EXPECT_THAT(RegImageIdSetsPerReconstruction(*reconstruction_manager),
              testing::UnorderedElementsAreArray(expected_components));
}

// End-to-end (gravity variant): even when *every* cross-cluster edge is a
// bogus-rotation outlier - a regime the default gravity-free solver cannot
// resolve because the inter-cluster orientation gauge is free - gravity priors
// anchor each cluster to the vertical. The random bridge rotations then exceed
// the rotation error threshold and are filtered, recovering the two clusters.
TEST(GlobalPipeline, MultiComponentsWithOutlierEdgesUsingGravity) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 100;
  // Known focal lengths let us skip view graph calibration, which would
  // otherwise re-estimate (and thereby "fix") the injected outlier edges.
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  synthetic_dataset_options.prior_gravity = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Split the images into two groups.
  std::vector<image_t> image_ids = gt_reconstruction.RegImageIds();
  std::sort(image_ids.begin(), image_ids.end());
  ASSERT_EQ(image_ids.size(), 10);
  const std::vector<std::unordered_set<image_t>> expected_components = {
      {image_ids.begin(), image_ids.begin() + 5},
      {image_ids.begin() + 5, image_ids.end()}};

  // Corrupt every cross-cluster edge into an outlier bridge (passing a count
  // larger than the number of cross pairs keeps and randomizes all of them).
  BridgeGroupsWithOutlierEdges(
      expected_components,
      /*num_outlier_edges=*/
      static_cast<int>(image_ids.size() * image_ids.size()),
      *database);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  GlobalPipelineOptions options;
  ASSERT_TRUE(options.reconstruct_all_components);
  // Gravity priors pin each cluster to the vertical, making the random-rotation
  // bridges detectable regardless of the otherwise free inter-cluster gauge.
  options.mapper.rotation_averaging.use_gravity = true;
  GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
  mapper.Run();

  // Despite every cross edge being an outlier, gravity lets rotation averaging
  // filter them all and recover the two clusters.
  ASSERT_EQ(reconstruction_manager->Size(), 2);
  EXPECT_THAT(RegImageIdSetsPerReconstruction(*reconstruction_manager),
              testing::UnorderedElementsAreArray(expected_components));
}

// End-to-end: with no matches at all, the view graph is empty and the pipeline
// produces no reconstructions.
TEST(GlobalPipeline, MultiComponentsEmptyViewGraph) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 3;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Remove all matches so the view graph is empty.
  database->ClearTwoViewGeometries();
  database->ClearMatches();

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  GlobalPipelineOptions options;
  GlobalPipeline mapper(std::move(options), database, reconstruction_manager);
  mapper.Run();

  EXPECT_EQ(reconstruction_manager->Size(), 0);
}

}  // namespace
}  // namespace colmap
