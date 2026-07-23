#include "colmap/sfm/global_mapper.h"

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/math.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

TEST(GlobalMapper, WithoutNoise) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialKnownRig) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialUnknownRig) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise

  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  // Set the rig sensors to be unknown
  for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction->Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithNoiseAndOutliers) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

// Rotation-gauge case with non-identity sensor_from_rig
// (num_cameras_per_rig=2); covariance weighting and one gross orientation
// outlier; the recovered gauge matches the known perturbation; every
// pairwise relative frame rotation is unchanged before vs. after; and, in a
// second sub-case, absent orientations produce a requested-but-not-engaged
// no-op.
TEST(GlobalMapper, InitializeRotationGaugeFromPosePriors) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 8;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev = 0.1;
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Full-orientation pose priors in the ground-truth ("prior world") frame,
  // one per frame (first image of each frame), with a small finite
  // rotation_covariance (exercising covariance weighting) and one gross
  // outlier.
  bool injected_outlier = false;
  for (const auto& [frame_id, frame] : gt_reconstruction.Frames()) {
    const data_t image_data_id = *frame.ImageIds().begin();
    const image_t image_id = static_cast<image_t>(image_data_id.id);
    const Image& image = gt_reconstruction.Image(image_id);

    PosePrior prior;
    prior.corr_data_id = image.DataId();
    prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
    prior.rotation = image.CamFromWorld().rotation();
    prior.rotation_covariance = Eigen::Matrix3d::Identity() * 1e-4;
    if (!injected_outlier) {
      // A gross 90 degree outlier on one frame's prior. The literal radian
      // value avoids M_PI, which is undefined on MSVC without
      // _USE_MATH_DEFINES.
      constexpr double kNinetyDegRad = 1.5707963267948966;
      prior.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(
                           kNinetyDegRad, Eigen::Vector3d::UnitX())) *
                       prior.rotation;
      injected_outlier = true;
    }
    database->WritePosePrior(prior);
  }
  ASSERT_TRUE(injected_outlier);

  // A known global rotation gauge, simulating rotation averaging having
  // solved everything correctly up to one unknown global rotation:
  // rig_from_solver = rig_from_prior_world * Inverse(solver_from_prior_world),
  // which preserves every pairwise relative rotation exactly (each frame is
  // right-multiplied by the same fixed rotation).
  const Eigen::Quaterniond solver_from_prior_world_true = Eigen::Quaterniond(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()));

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  for (const auto& [frame_id, gt_frame] : gt_reconstruction.Frames()) {
    Frame& frame = reconstruction->Frame(frame_id);
    const Eigen::Quaterniond perturbed_rotation =
        gt_frame.RigFromWorld().rotation() *
        solver_from_prior_world_true.inverse();
    frame.SetRigFromWorld(
        Rigid3d(perturbed_rotation, gt_frame.RigFromWorld().translation()));
  }

  // Snapshot one pairwise relative rotation before the gauge is applied.
  const auto frame_ids_vec = [&] {
    std::vector<frame_t> ids;
    for (const auto& [frame_id, _] : reconstruction->Frames()) {
      ids.push_back(frame_id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  }();
  ASSERT_GE(frame_ids_vec.size(), 2u);
  const Eigen::Quaterniond relative_before =
      reconstruction->Frame(frame_ids_vec[0]).RigFromWorld().rotation() *
      reconstruction->Frame(frame_ids_vec[1])
          .RigFromWorld()
          .rotation()
          .inverse();

  ASSERT_TRUE(global_mapper.InitializeRotationGaugeFromPosePriors(
      RotationEstimatorOptions()));

  // The gauge must recover (the inverse of) the known perturbation: every
  // frame's rotation should now match ground truth, despite the outlier.
  for (const auto& [frame_id, gt_frame] : gt_reconstruction.Frames()) {
    EXPECT_THAT(reconstruction->Frame(frame_id).RigFromWorld(),
                Rigid3dNear(gt_frame.RigFromWorld(),
                            /*rtol=*/DegToRad(1.0),
                            /*ttol=*/1e-6));
  }

  // Every pairwise relative frame rotation is unchanged.
  const Eigen::Quaterniond relative_after =
      reconstruction->Frame(frame_ids_vec[0]).RigFromWorld().rotation() *
      reconstruction->Frame(frame_ids_vec[1])
          .RigFromWorld()
          .rotation()
          .inverse();
  EXPECT_NEAR(relative_before.angularDistance(relative_after), 0.0, 1e-9);

  // Absent orientations: a fresh reconstruction/database with no pose priors
  // produces a requested-but-not-engaged no-op, not an error.
  {
    auto empty_database = Database::Open(CreateTestDir() / "empty_database.db");
    Reconstruction unused_gt;
    SynthesizeDataset(
        synthetic_dataset_options, &unused_gt, empty_database.get());
    auto no_prior_reconstruction = std::make_shared<Reconstruction>();
    GlobalMapper no_prior_mapper(CreateDatabaseCache(*empty_database));
    no_prior_mapper.BeginReconstruction(no_prior_reconstruction);
    for (const auto& [frame_id, gt_frame] : unused_gt.Frames()) {
      no_prior_reconstruction->Frame(frame_id).SetRigFromWorld(
          gt_frame.RigFromWorld());
    }
    EXPECT_TRUE(no_prior_mapper.InitializeRotationGaugeFromPosePriors(
        RotationEstimatorOptions()));
    for (const auto& [frame_id, gt_frame] : unused_gt.Frames()) {
      EXPECT_THAT(no_prior_reconstruction->Frame(frame_id).RigFromWorld(),
                  Rigid3dEq(gt_frame.RigFromWorld()));
    }
  }
}

// Regression test for the bug where
// GlobalMapper::IterativeRetriangulateAndRefine left
// IncrementalMapper::Options::use_prior_position at its default false, so
// IncrementalMapper::IterativeGlobalRefinement's own inner Normalize() call
// unconditionally discarded the metric gauge that an engaged `optimize`
// pose-prior position solve had just established -- even though GlobalMapper's
// own outer Normalize() calls were already correctly guarded by
// pose_prior_position_engaged_. Before the fix, this test's final scale would
// drift by roughly an order of magnitude (matching the ~20x drift observed on
// real data); after the fix it must stay within a tight tolerance of ground
// truth through the full Solve(), including retriangulation.
TEST(GlobalMapper, PosePriorPositionOptimizeSurvivesRetriangulation) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 12;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  for (const auto& [image_id, image] : gt_reconstruction.Images()) {
    PosePrior prior;
    prior.corr_data_id = image.DataId();
    prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
    prior.position = image.ProjectionCenter();
    database->WritePosePrior(prior);
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  GlobalMapperOptions options;
  options.global_positioning.use_gpu = false;
  options.global_positioning.random_seed = 42;
  options.global_positioning.pose_prior_position_mode =
      PosePriorPositionMode::optimize;
  // No position_covariance on the synthetic priors above, so alignment and
  // BA both fall back to this declared stddev (matches the pattern used in
  // GlobalPositioning.PosePriorPositionMode).
  options.global_positioning.pose_prior_position_fallback_stddev = 3.0;

  ASSERT_TRUE(global_mapper.Solve(options));

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/0.05,
                                 /*num_obs_tolerance=*/0.05));
}

// End-to-end wiring test for the soft gravity BA residual through the full
// GlobalMapper::Solve() pipeline (CLI-level PosePriorGravityBAMode ->
// GlobalMapperOptions -> RunBundleAdjustment()'s prior branch ->
// PosePriorBundleAdjuster, and also through IterativeRetriangulateAndRefine's
// mapper_options per M2's fix). The hard ra_use_gravity rotation-averaging
// reduction is deliberately left off (rotation_averaging.use_gravity =
// false), so any rotation accuracy here is attributable to the soft BA
// residual and the reprojection/position-prior terms, not the legacy hard
// mechanism -- matching the requirement to not silently keep relying on
// ra_use_gravity once the soft path is engaged. Precise isolation of
// gravity's own marginal contribution (yaw invariance, robustness to one bad
// reading, exact tilt magnitude) is covered at the functor level
// (pose_prior_test.cc) and the single-BA-solve integration level
// (bundle_adjustment_ceres_test.cc); this test's job is only to prove the
// full pipeline wiring is correct and stays metric/near-truth end-to-end.
TEST(GlobalMapper, PosePriorGravityOptimizeSurvivesFullPipeline) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 12;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  synthetic_dataset_options.prior_position = true;
  synthetic_dataset_options.prior_gravity = true;
  // Match AbsoluteGravityPriorCostFunctor's fixed ENU-down convention.
  synthetic_dataset_options.prior_gravity_in_world = Eigen::Vector3d(0, 0, -1);
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  GlobalMapperOptions options;
  options.rotation_averaging.use_gravity = false;
  options.global_positioning.use_gpu = false;
  options.global_positioning.random_seed = 42;
  options.global_positioning.pose_prior_position_mode =
      PosePriorPositionMode::optimize;
  options.global_positioning.pose_prior_position_fallback_stddev = 3.0;
  options.pose_prior_gravity_ba_mode = PosePriorGravityBAMode::optimize;
  options.pose_prior_gravity_stddev_deg = 2.0;

  ASSERT_TRUE(global_mapper.Solve(options));

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/0.5,
                                 /*max_proj_center_error=*/0.5,
                                 /*max_scale_error=*/0.05,
                                 /*num_obs_tolerance=*/0.05));
}

TEST(GlobalMapperOptions, RefineSensorFromRigPropagatesToSubOptions) {
  GlobalMapperOptions options;
  options.refine_sensor_from_rig = false;
  // Sub-options keep their own defaults (true) until accessed.
  EXPECT_TRUE(options.rotation_averaging.refine_sensor_from_rig);
  EXPECT_TRUE(options.global_positioning.refine_sensor_from_rig);
  EXPECT_TRUE(options.bundle_adjustment.refine_sensor_from_rig);
  // Accessors return resolved sub-options with the top-level flag applied.
  EXPECT_FALSE(options.RotationAveraging().refine_sensor_from_rig);
  EXPECT_FALSE(options.GlobalPositioning().refine_sensor_from_rig);
  EXPECT_FALSE(options.BundleAdjustment().refine_sensor_from_rig);
}

}  // namespace
}  // namespace colmap
