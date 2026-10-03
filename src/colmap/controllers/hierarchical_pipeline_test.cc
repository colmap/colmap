// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/hierarchical_pipeline.h"

#include "colmap/estimators/alignment.h"
#include "colmap/scene/database.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

void ExpectEqualReconstructions(const Reconstruction& gt,
                                const Reconstruction& computed,
                                const double max_rotation_error_deg,
                                const double max_proj_center_error,
                                const double num_obs_tolerance) {
  EXPECT_EQ(computed.NumCameras(), gt.NumCameras());
  EXPECT_EQ(computed.NumImages(), gt.NumImages());
  EXPECT_EQ(computed.NumRegImages(), gt.NumRegImages());
  EXPECT_GE(computed.ComputeNumObservations(),
            (1 - num_obs_tolerance) * gt.ComputeNumObservations());

  Sim3d gt_from_computed;
  ASSERT_TRUE(AlignReconstructionsViaProjCenters(computed,
                                                 gt,
                                                 /*max_proj_center_error=*/0.1,
                                                 &gt_from_computed));

  const std::vector<ImageAlignmentError> errors =
      ComputeImageAlignmentError(computed, gt, gt_from_computed);
  EXPECT_EQ(errors.size(), gt.NumImages());
  for (const auto& error : errors) {
    EXPECT_LT(error.rotation_error_deg, max_rotation_error_deg);
    EXPECT_LT(error.proj_center_error, max_proj_center_error);
  }
}

TEST(HierarchicalPipeline, WithoutNoise) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 20;
  synthetic_dataset_options.num_points3D = 100;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 5;
  mapper_options.clustering_options.image_overlap = 3;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  auto reconstruction = reconstruction_manager->Get(0);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/5e-4,
                             /*num_obs_tolerance=*/0);

  // After the pipeline runs, point3D.error must be in pixel units, i.e.
  // equal to what UpdatePoint3DErrors would recompute.
  ASSERT_GT(reconstruction->NumPoints3D(), 0u);
  const double mean_after_run = reconstruction->ComputeMeanReprojectionError();
  reconstruction->UpdatePoint3DErrors();
  EXPECT_DOUBLE_EQ(mean_after_run,
                   reconstruction->ComputeMeanReprojectionError());
}

TEST(HierarchicalPipeline, WithoutNoiseAndNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.05;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 10;
  mapper_options.clustering_options.image_overlap = 3;
  // Note that the hierarchical mapper does not work well when the
  // sensor_from_rig poses are inconsistently refined in different clusters,
  // because then the merging does not work well.
  mapper_options.incremental_options.ba_refine_sensor_from_rig = false;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-3,
                             /*num_obs_tolerance=*/0);
}

TEST(HierarchicalPipeline, WithoutNoiseAndPanoramicNonTrivialFrames) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 30;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 10;
  mapper_options.clustering_options.image_overlap = 3;
  // Note that the hierarchical mapper does not work well when the
  // sensor_from_rig poses are inconsistently refined in different clusters,
  // because then the merging does not work well.
  mapper_options.incremental_options.ba_refine_sensor_from_rig = false;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 1);
  ExpectEqualReconstructions(gt_reconstruction,
                             *reconstruction_manager->Get(0),
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-3,
                             /*num_obs_tolerance=*/0);
}

TEST(HierarchicalPipeline, MultiReconstruction) {
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction1;
  Reconstruction gt_reconstruction2;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction1, database.get());
  synthetic_dataset_options.num_frames_per_rig = 4;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction2, database.get());

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  HierarchicalPipelineOptions mapper_options;
  mapper_options.clustering_options.leaf_max_num_images = 5;
  mapper_options.clustering_options.image_overlap = 3;
  HierarchicalPipeline mapper(mapper_options, database, reconstruction_manager);
  mapper.Run();

  ASSERT_EQ(reconstruction_manager->Size(), 2);
  Reconstruction* computed_reconstruction1 = nullptr;
  Reconstruction* computed_reconstruction2 = nullptr;
  if (reconstruction_manager->Get(0)->NumRegImages() == 5) {
    computed_reconstruction1 = reconstruction_manager->Get(0).get();
    computed_reconstruction2 = reconstruction_manager->Get(1).get();
  } else {
    computed_reconstruction1 = reconstruction_manager->Get(1).get();
    computed_reconstruction2 = reconstruction_manager->Get(0).get();
  }
  ExpectEqualReconstructions(gt_reconstruction1,
                             *computed_reconstruction1,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);
  ExpectEqualReconstructions(gt_reconstruction2,
                             *computed_reconstruction2,
                             /*max_rotation_error_deg=*/1e-2,
                             /*max_proj_center_error=*/1e-4,
                             /*num_obs_tolerance=*/0);

  // After the pipeline runs, point3D.error must be in pixel units for every
  // reconstruction in the manager, i.e. equal to what UpdatePoint3DErrors
  // would recompute.
  for (Reconstruction* reconstruction :
       {computed_reconstruction1, computed_reconstruction2}) {
    ASSERT_GT(reconstruction->NumPoints3D(), 0u);
    const double mean_after_run =
        reconstruction->ComputeMeanReprojectionError();
    reconstruction->UpdatePoint3DErrors();
    EXPECT_DOUBLE_EQ(mean_after_run,
                     reconstruction->ComputeMeanReprojectionError());
  }
}

}  // namespace
}  // namespace colmap
