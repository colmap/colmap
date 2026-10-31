// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/sfm.h"

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
  synthetic_options.num_frames_per_rig = 5;
  synthetic_options.num_points3D = 60;
  synthetic_options.prior_position = true;
  SynthesizeDataset(synthetic_options, &gt_reconstruction, database.get());
  database.reset();

  Reconstruction unaligned = gt_reconstruction;
  unaligned.Transform(Sim3d(
      0.3,
      Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ())),
      Eigen::Vector3d(5.0, -3.0, 2.0)));
  const auto input_path = test_path / "input";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(input_path);
  CreateDirIfNotExists(output_path);
  unaligned.Write(input_path);

  std::vector<std::string> args = {
      "bundle_adjuster",
      "--input_path=" + input_path.string(),
      "--output_path=" + output_path.string(),
      "--database_path=" + database_path.string(),
      "--BundleAdjustment.use_prior_position=1",
      "--BundleAdjustment.prior_position_overwrite_covariance=1",
      "--BundleAdjustment.prior_position_stddev_x=0.05",
      "--BundleAdjustment.prior_position_stddev_y=0.05",
      "--BundleAdjustment.prior_position_stddev_z=0.05",
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  ASSERT_EQ(RunBundleAdjuster(argv.size(), argv.data()), EXIT_SUCCESS);

  Reconstruction refined;
  refined.Read(output_path);
  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(refined,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.05,
                                 /*max_scale_error=*/1e-2,
                                 /*num_obs_tolerance=*/0.02,
                                 /*align=*/false));
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

  std::vector<std::string> args = {
      "pose_prior_mapper",
      "--database_path=" + database_path.string(),
      "--image_path=" + image_path.string(),
      "--output_path=" + output_path.string(),
      "--Mapper.extract_colors=0",
      "--Mapper.min_model_size=3",
      "--Mapper.num_threads=1",
      "--overwrite_priors_covariance=1",
      "--prior_position_std_x=0.1",
      "--prior_position_std_y=0.1",
      "--prior_position_std_z=0.1",
      "--use_robust_loss_on_prior_position=1",
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  ASSERT_EQ(RunPosePriorMapper(argv.size(), argv.data()), EXIT_SUCCESS);

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

}  // namespace
}  // namespace colmap
