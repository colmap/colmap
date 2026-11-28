// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/sfm.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(PointTriangulator, IncludesRegisteredImagesOutsideImageList) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 10;
  SynthesizeDataset(synthetic_options, reconstruction.get(), database.get());

  ASSERT_EQ(reconstruction->NumRegImages(), 3);
  IncrementalPipelineOptions options;
  options.image_names = {reconstruction->Image(1).Name(),
                         reconstruction->Image(2).Name()};
  options.ba_global_max_refinements = 0;
  options.num_threads = 1;

  database.reset();
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  EXPECT_NO_THROW(RunPointTriangulatorImpl(reconstruction,
                                           database_path,
                                           test_path,
                                           output_path,
                                           options,
                                           /*clear_points=*/false,
                                           /*refine_intrinsics=*/false));
  EXPECT_EQ(reconstruction->NumRegImages(), 3);
}

TEST(PointTriangulator, WritesPartialReconstructionWhenCancelled) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, reconstruction.get(), database.get());
  database.reset();

  const size_t num_points3D = reconstruction->NumPoints3D();
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  IncrementalPipelineOptions options;
  options.extract_colors = false;
  bool stop_checked = false;
  RunPointTriangulatorImpl(reconstruction,
                           database_path,
                           test_path,
                           output_path,
                           options,
                           /*clear_points=*/false,
                           /*refine_intrinsics=*/false,
                           [&stop_checked]() {
                             stop_checked = true;
                             return true;
                           });

  EXPECT_TRUE(stop_checked);
  EXPECT_EQ(reconstruction->NumPoints3D(), num_points3D);
  EXPECT_FALSE(ExistsFile(output_path / "cameras.bin"));
  EXPECT_FALSE(ExistsFile(output_path / "images.bin"));
  EXPECT_FALSE(ExistsFile(output_path / "points3D.bin"));
  const auto partial_output_path = test_path / "output.partial";
  EXPECT_TRUE(ExistsFile(partial_output_path / "cameras.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "images.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "points3D.bin"));
}

TEST(PointTriangulator, DoesNotOverwriteInputWhenCancelledAfterClearingPoints) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 20;
  SynthesizeDataset(synthetic_options, reconstruction.get(), database.get());
  database.reset();

  const Reconstruction expected_reconstruction = *reconstruction;
  const auto model_path = test_path / "model";
  CreateDirIfNotExists(model_path);
  reconstruction->Write(model_path);

  IncrementalPipelineOptions options;
  options.extract_colors = false;
  RunPointTriangulatorImpl(reconstruction,
                           database_path,
                           test_path,
                           model_path,
                           options,
                           /*clear_points=*/true,
                           /*refine_intrinsics=*/false,
                           []() { return true; });

  Reconstruction written_reconstruction;
  written_reconstruction.Read(model_path);
  EXPECT_THAT(written_reconstruction,
              ReconstructionEq(expected_reconstruction));

  const auto partial_output_path = test_path / "model.partial";
  EXPECT_TRUE(ExistsFile(partial_output_path / "cameras.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "images.bin"));
  EXPECT_TRUE(ExistsFile(partial_output_path / "points3D.bin"));
}

TEST(IncrementalMapper, WritesInterruptedReconstruction) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());
  database.reset();

  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->extract_colors = false;
  options->multiple_models = false;
  options->num_threads = 1;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  bool stop_requested = false;
  EXPECT_TRUE(RunIncrementalMapperImpl(
      database_path,
      test_path,
      output_path,
      options,
      reconstruction_manager,
      [&stop_requested]() { stop_requested = true; },
      {},
      [&stop_requested]() { return stop_requested; }));

  EXPECT_TRUE(stop_requested);
  ASSERT_EQ(reconstruction_manager->Size(), 1);
  EXPECT_TRUE(ExistsFile(output_path / "0" / "cameras.bin"));
  EXPECT_TRUE(ExistsFile(output_path / "0" / "images.bin"));
  EXPECT_TRUE(ExistsFile(output_path / "0" / "points3D.bin"));
}

TEST(IncrementalMapper, DiscardsInterruptBeforeInitialPair) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());
  database.reset();

  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);
  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->extract_colors = false;
  options->multiple_models = false;
  options->num_threads = 1;
  auto reconstruction_manager = std::make_shared<ReconstructionManager>();

  int num_stop_checks = 0;
  EXPECT_FALSE(RunIncrementalMapperImpl(
      database_path,
      test_path,
      output_path,
      options,
      reconstruction_manager,
      {},
      {},
      [&num_stop_checks]() { return ++num_stop_checks >= 2; }));

  EXPECT_GE(num_stop_checks, 2);
  EXPECT_EQ(reconstruction_manager->Size(), 0);
  EXPECT_FALSE(ExistsDir(output_path / "0"));
}

TEST(BundleAdjusterExe, WithDatabasePositionPriors) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 6;
  synthetic_options.num_points3D = 80;
  synthetic_options.prior_position = true;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  // Store tiny covariances in the database so alignment would fail without
  // overwrite_prior_position_covariance, and verify that overriding them in
  // memory does not mutate the database on disk.
  const Eigen::Matrix3d orig_db_cov = 1e-8 * Eigen::Matrix3d::Identity();
  for (PosePrior prior : database->ReadAllPosePriors()) {
    prior.position_covariance = orig_db_cov;
    database->UpdatePosePrior(prior);
  }
  database.reset();

  Reconstruction unaligned = gt_reconstruction;
  SyntheticNoiseOptions noise_options;
  noise_options.point2D_stddev = 0.2;
  noise_options.point3D_stddev = 0.05;
  noise_options.rig_from_world_rotation_stddev = 0.5;
  noise_options.rig_from_world_translation_stddev = 0.02;
  SynthesizeNoise(noise_options, &unaligned);
  unaligned.Transform(Sim3d(
      0.3,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(5.0, -3.0, 2.0)));
  const auto input_path = test_path / "input";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(input_path);
  CreateDirIfNotExists(output_path);
  unaligned.Write(input_path);

  const auto run_bundle_adjuster = [&](std::vector<std::string> args) {
    args.insert(args.begin(),
                {"bundle_adjuster",
                 "--input_path=" + input_path.string(),
                 "--output_path=" + output_path.string(),
                 "--BundleAdjustment.use_prior_position=1"});
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) {
      argv.push_back(arg.data());
    }
    return RunBundleAdjuster(argv.size(), argv.data());
  };
  EXPECT_EQ(run_bundle_adjuster({}), EXIT_FAILURE);
  ASSERT_EQ(run_bundle_adjuster(
                {"--database_path=" + database_path.string(),
                 "--BundleAdjustment.overwrite_prior_position_covariance=1",
                 "--BundleAdjustment.prior_position_stddev_x=0.1",
                 "--BundleAdjustment.prior_position_stddev_y=0.1",
                 "--BundleAdjustment.prior_position_stddev_z=0.1"}),
            EXIT_SUCCESS);

  Reconstruction refined;
  refined.Read(output_path);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(refined,
                                 /*max_rotation_error_deg=*/0.2,
                                 /*max_proj_center_error=*/0.05,
                                 /*max_scale_error=*/1e-2,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));

  const auto reopened_db = Database::Open(database_path);
  for (const PosePrior& prior : reopened_db->ReadAllPosePriors()) {
    EXPECT_EQ(prior.position_covariance, orig_db_cov);
  }
}

TEST(PosePriorMapperExe, DeprecatedShimPreservesDatabaseCovariances) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  const auto image_path = test_path / "images";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 80;
  synthetic_options.prior_position = true;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());
  const std::vector<PosePrior> orig_priors = database->ReadAllPosePriors();
  database.reset();

  const auto project_path = test_path / "project.ini";
  OptionManager project_options;
  project_options.AddAllOptions();
  *project_options.database_path = database_path;
  *project_options.image_path = image_path;
  ASSERT_FALSE(project_options.mapper->use_prior_position);
  project_options.mapper->prior_position_stddev =
      Eigen::Vector3d::Constant(2.5);
  project_options.mapper->prior_position_loss_scale = 4.5;
  project_options.Write(project_path);

  // Invalid --prior_position_std_* is rejected even when
  // --overwrite_priors_covariance=0.
  std::vector<std::string> invalid_args = {
      "pose_prior_mapper",
      "--project_path=" + project_path.string(),
      "--output_path=" + output_path.string(),
      "--prior_position_std_x=-1.0",
  };
  std::vector<char*> invalid_argv;
  invalid_argv.reserve(invalid_args.size());
  for (std::string& arg : invalid_args) {
    invalid_argv.push_back(arg.data());
  }
  EXPECT_EQ(RunPosePriorMapper(invalid_argv.size(), invalid_argv.data()),
            EXIT_FAILURE);

  std::vector<std::string> args = {
      "pose_prior_mapper",
      "--project_path=" + project_path.string(),
      "--output_path=" + output_path.string(),
      "--Mapper.extract_colors=0",
      "--Mapper.min_model_size=3",
      "--Mapper.num_threads=1",
      "--overwrite_priors_covariance=1",
      "--prior_position_std_x=0.1",
      "--prior_position_std_y=0.1",
      "--prior_position_std_z=1.0",
      "--use_robust_loss_on_prior_position=1",
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  ASSERT_EQ(RunPosePriorMapper(argv.size(), argv.data()), EXIT_SUCCESS);

  std::filesystem::path saved_input_path;
  std::filesystem::path saved_output_path;
  OptionManager saved_options;
  saved_options.AddAllOptions();
  saved_options.AddDefaultOption("input_path", &saved_input_path);
  saved_options.AddDefaultOption("output_path", &saved_output_path);
  ASSERT_TRUE(saved_options.Read(output_path / "0" / "project.ini",
                                 /*allow_unregistered=*/false));
  EXPECT_TRUE(saved_options.mapper->use_prior_position);
  EXPECT_TRUE(saved_options.mapper->overwrite_prior_position_covariance);
  EXPECT_EQ(saved_options.mapper->prior_position_stddev,
            Eigen::Vector3d(0.1, 0.1, 1.0));
  EXPECT_TRUE(saved_options.mapper->use_robust_loss_on_prior_position);
  EXPECT_DOUBLE_EQ(saved_options.mapper->prior_position_loss_scale, 4.5);

  auto check_db = Database::Open(database_path);
  EXPECT_EQ(check_db->ReadAllPosePriors(), orig_priors);
  Reconstruction mapped;
  mapped.Read(output_path / "0");
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(mapped,
                                 /*max_rotation_error_deg=*/0.2,
                                 /*max_proj_center_error=*/0.1,
                                 /*max_scale_error=*/1e-2,
                                 /*num_obs_tolerance=*/0.05,
                                 /*align=*/false));
}

TEST(IncrementalMapper, FixExistingFramesWithPositionPriorsKeepsMetricFrame) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(output_path);

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 6;
  synthetic_options.num_points3D = 80;
  synthetic_options.prior_position = true;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());

  const std::vector<frame_t> frame_ids = gt_reconstruction.RegFrameIds();
  ASSERT_EQ(frame_ids.size(), 6);
  // Shift one unregistered frame's prior with tight covariance to ensure active
  // prior BA runs during continuation rather than only a post-hoc alignment.
  const image_t shifted_image_id =
      gt_reconstruction.Frame(frame_ids[5]).ImageIds().begin()->id;
  Eigen::Vector3d shifted_prior_position = Eigen::Vector3d::Zero();
  for (PosePrior prior : database->ReadAllPosePriors()) {
    if (prior.corr_data_id.id == shifted_image_id) {
      prior.position += Eigen::Vector3d(0.01, 0.0, 0.0);
      prior.position_covariance = 1e-12 * Eigen::Matrix3d::Identity();
      shifted_prior_position = prior.position;
    } else {
      prior.position_covariance = 1e-6 * Eigen::Matrix3d::Identity();
    }
    database->UpdatePosePrior(prior);
  }
  database.reset();

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  Reconstruction& initial =
      *reconstruction_manager->Get(reconstruction_manager->Add());
  initial = gt_reconstruction;
  initial.DeRegisterFrame(frame_ids[4]);
  initial.DeRegisterFrame(frame_ids[5]);
  initial.Transform(Sim3d(
      2.5,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(50.0, -25.0, 10.0)));

  auto options = std::make_shared<IncrementalPipelineOptions>();
  options->extract_colors = false;
  options->fix_existing_frames = true;
  options->use_prior_position = true;
  options->ba_global_frames_freq = 1;
  options->num_threads = 1;
  options->random_seed = 0;

  ASSERT_TRUE(RunIncrementalMapperImpl(
      database_path, test_path, output_path, options, reconstruction_manager));

  Reconstruction continued;
  continued.Read(output_path);
  EXPECT_EQ(continued.NumRegFrames(), 6);
  EXPECT_LT((continued.Image(shifted_image_id).ProjectionCenter() -
             shifted_prior_position)
                .norm(),
            1e-3);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(continued,
                                 /*max_rotation_error_deg=*/0.5,
                                 /*max_proj_center_error=*/0.05,
                                 /*max_scale_error=*/1e-2,
                                 /*num_obs_tolerance=*/0.05,
                                 /*align=*/false));
}

}  // namespace
}  // namespace colmap
