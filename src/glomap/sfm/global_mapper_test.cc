#include "glomap/sfm/global_mapper.h"

#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include "glomap/io/colmap_io.h"

#include <gtest/gtest.h>

namespace glomap {
namespace {

// TODO(jsch): Add tests for pose priors.

GlobalMapperOptions CreateTestOptions() {
  GlobalMapperOptions options;
  options.skip_view_graph_calibration = false;
  options.skip_relative_pose_estimation = false;
  options.skip_rotation_averaging = false;
  options.skip_track_establishment = false;
  options.skip_global_positioning = false;
  options.skip_bundle_adjustment = false;
  options.skip_retriangulation = false;
  return options;
}

TEST(GlobalMapper, WithoutNoise) {
  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  colmap::Reconstruction reconstruction;
  std::vector<colmap::PosePrior> pose_priors;

  InitializeGlomapFromDatabase(*database, reconstruction, view_graph);

  GlobalMapper global_mapper(CreateTestOptions());
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(
      database.get(), view_graph, reconstruction, pose_priors, cluster_ids);

  EXPECT_THAT(gt_reconstruction,
              colmap::ReconstructionNear(reconstruction,
                                         /*max_rotation_error_deg=*/1e-2,
                                         /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialKnownRig) {
  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  colmap::Reconstruction reconstruction;
  std::vector<colmap::PosePrior> pose_priors;

  InitializeGlomapFromDatabase(*database, reconstruction, view_graph);

  GlobalMapper global_mapper(CreateTestOptions());
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(
      database.get(), view_graph, reconstruction, pose_priors, cluster_ids);

  EXPECT_THAT(gt_reconstruction,
              colmap::ReconstructionNear(reconstruction,
                                         /*max_rotation_error_deg=*/1e-2,
                                         /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialUnknownRig) {
  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise

  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  colmap::Reconstruction reconstruction;
  std::vector<colmap::PosePrior> pose_priors;

  InitializeGlomapFromDatabase(*database, reconstruction, view_graph);

  // Set the rig sensors to be unknown
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction.Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  GlobalMapper global_mapper(CreateTestOptions());
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(
      database.get(), view_graph, reconstruction, pose_priors, cluster_ids);

  EXPECT_THAT(gt_reconstruction,
              colmap::ReconstructionNear(reconstruction,
                                         /*max_rotation_error_deg=*/1e-2,
                                         /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithNoiseAndOutliers) {
  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  colmap::SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  colmap::SynthesizeNoise(
      synthetic_noise_options, &gt_reconstruction, database.get());

  ViewGraph view_graph;
  colmap::Reconstruction reconstruction;
  std::vector<colmap::PosePrior> pose_priors;

  InitializeGlomapFromDatabase(*database, reconstruction, view_graph);

  GlobalMapper global_mapper(CreateTestOptions());
  std::unordered_map<frame_t, int> cluster_ids;
  global_mapper.Solve(
      database.get(), view_graph, reconstruction, pose_priors, cluster_ids);

  EXPECT_THAT(gt_reconstruction,
              colmap::ReconstructionNear(reconstruction,
                                         /*max_rotation_error_deg=*/1e-1,
                                         /*max_proj_center_error=*/1e-1,
                                         /*max_scale_error=*/std::nullopt,
                                         /*num_obs_tolerance=*/0.02));
}

TEST(GlobalMapper, WithRandomSeedStability) {
  colmap::SetPRNGSeed(1);

  const std::string database_path = colmap::CreateTestDir() + "/database.db";

  auto database = colmap::Database::Open(database_path);
  colmap::Reconstruction gt_reconstruction;
  colmap::SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  colmap::SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  colmap::SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  colmap::SynthesizeNoise(
      synthetic_noise_options, &gt_reconstruction, database.get());

  auto run_mapper = [&](int num_threads, int random_seed) {
    ViewGraph view_graph;
    auto reconstruction = std::make_shared<colmap::Reconstruction>();
    std::vector<colmap::PosePrior> pose_priors;

    InitializeGlomapFromDatabase(*database, *reconstruction, view_graph);

    GlobalMapperOptions options = CreateTestOptions();
    options.num_threads = num_threads;
    options.random_seed = random_seed;
    GlobalMapper global_mapper(options);
    std::unordered_map<frame_t, int> cluster_ids;
    global_mapper.Solve(
        database.get(), view_graph, *reconstruction, pose_priors, cluster_ids);
    return reconstruction;
  };

  constexpr int kRandomSeed = 42;

  // Single-threaded execution.
  {
    auto reconstruction0 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    auto reconstruction1 =
        run_mapper(/*num_threads=*/1, /*random_seed=*/kRandomSeed);
    EXPECT_THAT(*reconstruction0,
                colmap::ReconstructionNear(*reconstruction1,
                                           /*max_rotation_error_deg=*/1e-10,
                                           /*max_proj_center_error=*/1e-10));
  }

  // Multi-threaded execution.
  {
    auto reconstruction0 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    auto reconstruction1 =
        run_mapper(/*num_threads=*/3, /*random_seed=*/kRandomSeed);
    // Same seed should produce similar results, up to floating-point variations
    // in optimization.
    EXPECT_THAT(*reconstruction0,
                colmap::ReconstructionNear(*reconstruction1,
                                           /*max_rotation_error_deg=*/1e-6,
                                           /*max_proj_center_error=*/1e-6));
  }
}

}  // namespace
}  // namespace glomap
