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

#include "colmap/controllers/option_manager.h"

#include "colmap/controllers/global_pipeline.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/controllers/pairing.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/gravity_refinement.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/types.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/scene/reconstruction_clustering.h"
#include "colmap/ui/render_options.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <limits>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Helper to build a fake argv from a vector of strings.
std::vector<char*> MakeArgv(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  return argv;
}

// --------------------------------------------------------------------------
// Existing tests (kept as-is)
// --------------------------------------------------------------------------

TEST(OptionManager, Reset) {
  OptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.AddDatabaseOptions();
  options.AddImageOptions();
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");

  options.Reset();

  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");
}

TEST(OptionManager, ResetOptions) {
  OptionManager options;
  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  const int original_num_threads = options.feature_extraction->num_threads;
  options.feature_extraction->num_threads = original_num_threads + 42;

  options.ResetOptions(/*reset_paths=*/true);
  EXPECT_EQ(*options.database_path, "");
  EXPECT_EQ(*options.image_path, "");
  EXPECT_EQ(options.feature_extraction->num_threads, original_num_threads);

  *options.database_path = "/test/path";
  *options.image_path = "/test/images";
  options.feature_extraction->num_threads = original_num_threads + 42;
  options.ResetOptions(/*reset_paths=*/false);
  EXPECT_EQ(*options.database_path, "/test/path");
  EXPECT_EQ(*options.image_path, "/test/images");
  EXPECT_EQ(options.feature_extraction->num_threads, original_num_threads);
}

TEST(OptionManager, AddOptionsIdempotent) {
  OptionManager options;

  // Adding options multiple times should not cause issues
  options.AddLogOptions();
  options.AddLogOptions();

  options.AddRandomOptions();
  options.AddRandomOptions();

  options.AddFeatureExtractionOptions();
  options.AddFeatureExtractionOptions();

  options.AddFeatureMatchingOptions();
  options.AddFeatureMatchingOptions();

  options.AddMapperOptions();
  options.AddMapperOptions();

  // If idempotency is not maintained, the above would cause errors
  SUCCEED();
}

TEST(OptionManager, AddAllOptions) {
  OptionManager options;
  options.AddAllOptions();

  // Verify that at least some key options are initialized
  EXPECT_NE(options.image_reader, nullptr);
  EXPECT_NE(options.feature_extraction, nullptr);
  EXPECT_NE(options.feature_matching, nullptr);
  EXPECT_NE(options.bundle_adjustment, nullptr);
  EXPECT_NE(options.mapper, nullptr);
  EXPECT_NE(options.patch_match_stereo, nullptr);
}

TEST(OptionManager, WriteAndRead) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir / "images");

  // Create and configure an OptionManager
  OptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  options_write.AddFeatureExtractionOptions();
  options_write.AddMapperOptions();
  options_write.AddGlobalMapperOptions();

  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  options_write.feature_extraction->max_image_size = 2048;
  options_write.feature_extraction->sift->max_num_features = 4096;
  options_write.mapper->min_num_matches = 20;

  // Write to file
  options_write.Write(config_path);
  EXPECT_TRUE(ExistsFile(config_path));

  // Read from file
  OptionManager options_read;
  options_read.AddDatabaseOptions();
  options_read.AddImageOptions();
  options_read.AddFeatureExtractionOptions();
  options_read.AddMapperOptions();
  options_read.AddGlobalMapperOptions();

  EXPECT_TRUE(options_read.Read(config_path));

  // Verify that values were read correctly
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
  EXPECT_EQ(options_read.feature_extraction->max_image_size,
            options_write.feature_extraction->max_image_size);
  EXPECT_EQ(options_read.feature_extraction->sift->max_num_features,
            options_write.feature_extraction->sift->max_num_features);
  EXPECT_EQ(options_read.mapper->min_num_matches,
            options_write.mapper->min_num_matches);
}

TEST(OptionManager, ReRead) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir / "images");

  // Create and write initial config
  OptionManager options_write;
  options_write.AddAllOptions();
  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  options_write.feature_extraction->max_image_size = 2048;
  options_write.Write(config_path);

  // Read with ReRead
  OptionManager options_read;
  EXPECT_TRUE(options_read.ReRead(config_path));

  // Verify values
  EXPECT_EQ(*options_read.database_path, *options_write.database_path);
  EXPECT_EQ(*options_read.image_path, *options_write.image_path);
  EXPECT_EQ(options_read.feature_extraction->max_image_size, 2048);
}

TEST(OptionManager, ReadNonExistentFile) {
  OptionManager options;
  options.AddAllOptions();

  EXPECT_FALSE(options.Read("/path/that/does/not/exist.ini"));
}

TEST(OptionManager, Check) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  // Should fail with non-existent paths
  *options.database_path = test_dir / "database.db";
  *options.image_path = "/path/that/does/not/exist";
  EXPECT_FALSE(options.Check());

  // Should succeed with valid paths
  CreateDirIfNotExists(test_dir / "images");
  *options.image_path = test_dir / "images";
  EXPECT_TRUE(options.Check());
}

TEST(OptionManager, CheckDatabaseParentDir) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();

  // Should succeed when database parent dir exists
  *options.database_path = test_dir / "database.db";
  EXPECT_TRUE(options.Check());

  // Should fail when database path is a directory
  CreateDirIfNotExists(test_dir / "bad_database");
  *options.database_path = test_dir / "bad_database";
  EXPECT_FALSE(options.Check());
}

TEST(OptionManager, ParseWithOptions) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddFeatureExtractionOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  // Create argv with additional options
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--FeatureExtraction.max_image_size",
      "1024",
      "--SiftExtraction.max_num_features",
      "2048",
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Verify parsed values
  EXPECT_EQ(*options.database_path, database_path);
  EXPECT_EQ(*options.image_path, image_path);
  EXPECT_EQ(options.feature_extraction->max_image_size, 1024);
  EXPECT_EQ(options.feature_extraction->sift->max_num_features, 2048);
}

TEST(OptionManager, ParseWithProjectPath) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";
  CreateDirIfNotExists(test_dir / "images");

  // Create and write a config file
  OptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  options_write.AddFeatureExtractionOptions();

  *options_write.database_path = test_dir / "database.db";
  *options_write.image_path = test_dir / "images";
  options_write.feature_extraction->max_image_size = 3000;
  options_write.Write(config_path);

  // Parse using project_path
  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddFeatureExtractionOptions();

  const std::vector<std::string> args = {
      "colmap",
      "--project_path",
      config_path.string(),
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // Verify values were loaded from config file
  EXPECT_EQ(*options.database_path, *options_write.database_path);
  EXPECT_EQ(*options.image_path, *options_write.image_path);
  EXPECT_EQ(options.feature_extraction->max_image_size, 3000);
}

TEST(OptionManager, ParseEmptyArguments) {
  OptionManager options;

  const std::vector<std::string> args = {"colmap"};
  auto argv = MakeArgv(args);

  // Should succeed with no required options
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));
}

TEST(OptionManager, ParseUnknownArgumentsFails) {
  const auto test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();

  const auto database_path = test_dir / "database.db";

  // Create argv with an unknown option
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--unknown_option",
      "value",
  };

  auto argv = MakeArgv(args);

  // Should return false when encountering unknown option
  EXPECT_FALSE(options.Parse(argv.size(), argv.data()));
}

// --------------------------------------------------------------------------
// Quality modifier tests
// --------------------------------------------------------------------------

TEST(OptionManager, ModifyForIndividualData) {
  OptionManager options;

  options.ModifyForIndividualData();

  EXPECT_DOUBLE_EQ(options.mapper->min_focal_length_ratio, 0.1);
  EXPECT_DOUBLE_EQ(options.mapper->max_focal_length_ratio, 10.0);
  EXPECT_EQ(options.mapper->max_extra_param,
            std::numeric_limits<double>::max());
}

TEST(OptionManager, ModifyForVideoData) {
  OptionManager options;
  const IncrementalPipelineOptions defaults;

  options.ModifyForVideoData();

  EXPECT_DOUBLE_EQ(options.mapper->mapper.init_min_tri_angle,
                   defaults.mapper.init_min_tri_angle / 2);
  EXPECT_DOUBLE_EQ(options.mapper->ba_global_frames_ratio, 1.4);
  EXPECT_DOUBLE_EQ(options.mapper->ba_global_points_ratio, 1.4);
  EXPECT_DOUBLE_EQ(options.mapper->min_focal_length_ratio, 0.1);
  EXPECT_DOUBLE_EQ(options.mapper->max_focal_length_ratio, 10.0);
  EXPECT_EQ(options.mapper->max_extra_param,
            std::numeric_limits<double>::max());
  EXPECT_EQ(options.stereo_fusion->min_num_pixels, 15);
}

TEST(OptionManager, ModifyForInternetData) {
  OptionManager options;

  options.ModifyForInternetData();

  EXPECT_EQ(options.stereo_fusion->min_num_pixels, 10);
}

TEST(OptionManager, ModifyForLowQuality) {
  OptionManager options;
  const FeatureExtractionOptions fe_defaults;
  const SequentialPairingOptions sp_defaults;
  const VocabTreePairingOptions vt_defaults;
  const IncrementalPipelineOptions mp_defaults;
  const mvs::PatchMatchOptions pm_defaults;
  const mvs::StereoFusionOptions sf_defaults;

  options.ModifyForLowQuality();

  EXPECT_EQ(options.feature_extraction->max_image_size, 1000);
  EXPECT_EQ(options.feature_extraction->sift->max_num_features, 2048);
  EXPECT_EQ(options.sequential_pairing->loop_detection_num_images,
            sp_defaults.loop_detection_num_images / 2);
  EXPECT_EQ(options.vocab_tree_pairing->max_num_features, 256);
  EXPECT_EQ(options.vocab_tree_pairing->num_images, vt_defaults.num_images / 2);
  EXPECT_EQ(options.mapper->ba_local_max_num_iterations,
            mp_defaults.ba_local_max_num_iterations / 2);
  EXPECT_EQ(options.mapper->ba_global_max_num_iterations,
            mp_defaults.ba_global_max_num_iterations / 2);
  EXPECT_DOUBLE_EQ(options.mapper->ba_global_frames_ratio,
                   mp_defaults.ba_global_frames_ratio * 1.2);
  EXPECT_DOUBLE_EQ(options.mapper->ba_global_points_ratio,
                   mp_defaults.ba_global_points_ratio * 1.2);
  EXPECT_EQ(options.mapper->ba_global_max_refinements, 2);
  EXPECT_EQ(options.patch_match_stereo->max_image_size, 1000);
  EXPECT_EQ(options.patch_match_stereo->window_radius, 4);
  EXPECT_EQ(options.patch_match_stereo->window_step, 2);
  EXPECT_EQ(options.patch_match_stereo->num_samples,
            pm_defaults.num_samples / 2);
  EXPECT_EQ(options.patch_match_stereo->num_iterations, 3);
  EXPECT_FALSE(options.patch_match_stereo->geom_consistency);
  EXPECT_EQ(options.stereo_fusion->check_num_images,
            sf_defaults.check_num_images / 2);
  EXPECT_EQ(options.stereo_fusion->max_image_size, 1000);
}

TEST(OptionManager, ModifyForMediumQuality) {
  OptionManager options;
  const IncrementalPipelineOptions mp_defaults;
  const mvs::PatchMatchOptions pm_defaults;
  const mvs::StereoFusionOptions sf_defaults;

  options.ModifyForMediumQuality();

  EXPECT_EQ(options.feature_extraction->max_image_size, 1600);
  EXPECT_EQ(options.feature_extraction->sift->max_num_features, 4096);
  EXPECT_EQ(options.mapper->ba_global_max_refinements, 2);
  EXPECT_EQ(options.patch_match_stereo->max_image_size, 1600);
  EXPECT_EQ(options.patch_match_stereo->window_radius, 4);
  EXPECT_EQ(options.patch_match_stereo->window_step, 2);
  EXPECT_EQ(options.patch_match_stereo->num_iterations, 5);
  EXPECT_FALSE(options.patch_match_stereo->geom_consistency);
  EXPECT_EQ(options.stereo_fusion->max_image_size, 1600);
}

TEST(OptionManager, ModifyForHighQuality) {
  OptionManager options;

  options.ModifyForHighQuality();

  EXPECT_TRUE(options.feature_extraction->sift->estimate_affine_shape);
  EXPECT_EQ(options.feature_extraction->max_image_size, 2400);
  EXPECT_EQ(options.feature_extraction->sift->max_num_features, 8192);
  EXPECT_TRUE(options.feature_matching->guided_matching);
  EXPECT_EQ(options.vocab_tree_pairing->max_num_features, 4096);
  EXPECT_EQ(options.mapper->ba_local_max_num_iterations, 30);
  EXPECT_EQ(options.mapper->ba_local_max_refinements, 3);
  EXPECT_EQ(options.mapper->ba_global_max_num_iterations, 75);
  EXPECT_EQ(options.patch_match_stereo->max_image_size, 2400);
  EXPECT_EQ(options.stereo_fusion->max_image_size, 2400);
}

TEST(OptionManager, ModifyForExtremeQuality) {
  OptionManager options;

  options.ModifyForExtremeQuality();

  EXPECT_TRUE(options.feature_extraction->sift->estimate_affine_shape);
  EXPECT_TRUE(options.feature_extraction->sift->domain_size_pooling);
  EXPECT_TRUE(options.feature_matching->guided_matching);
  EXPECT_EQ(options.mapper->ba_local_max_num_iterations, 40);
  EXPECT_EQ(options.mapper->ba_local_max_refinements, 3);
  EXPECT_EQ(options.mapper->ba_global_max_num_iterations, 100);
}

// --------------------------------------------------------------------------
// Idempotency for remaining Add*Options methods
// --------------------------------------------------------------------------

TEST(OptionManager, AddAllOptionsIdempotent) {
  OptionManager options;

  options.AddTwoViewGeometryOptions();
  options.AddTwoViewGeometryOptions();

  options.AddExhaustivePairingOptions();
  options.AddExhaustivePairingOptions();

  options.AddSequentialPairingOptions();
  options.AddSequentialPairingOptions();

  options.AddVocabTreePairingOptions();
  options.AddVocabTreePairingOptions();

  options.AddSpatialPairingOptions();
  options.AddSpatialPairingOptions();

  options.AddTransitivePairingOptions();
  options.AddTransitivePairingOptions();

  options.AddImportedPairingOptions();
  options.AddImportedPairingOptions();

  options.AddBundleAdjustmentOptions();
  options.AddBundleAdjustmentOptions();

  options.AddGlobalMapperOptions();
  options.AddGlobalMapperOptions();

  options.AddGravityRefinerOptions();
  options.AddGravityRefinerOptions();

  options.AddReconstructionClustererOptions();
  options.AddReconstructionClustererOptions();

  options.AddPatchMatchStereoOptions();
  options.AddPatchMatchStereoOptions();

  options.AddStereoFusionOptions();
  options.AddStereoFusionOptions();

  options.AddPoissonMeshingOptions();
  options.AddPoissonMeshingOptions();

  options.AddDelaunayMeshingOptions();
  options.AddDelaunayMeshingOptions();

  options.AddRenderOptions();
  options.AddRenderOptions();

  SUCCEED();
}

// --------------------------------------------------------------------------
// Composite Add* methods pull in their dependencies
// --------------------------------------------------------------------------

TEST(OptionManager, ExhaustivePairingAddsMatchingAndTwoViewGeometry) {
  OptionManager options;
  options.AddExhaustivePairingOptions();

  // Write and read back to verify matching and two-view options are registered.
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";
  options.feature_matching->guided_matching = true;
  options.two_view_geometry->min_num_inliers = 42;
  options.exhaustive_pairing->block_size = 77;
  options.Write(config_path);

  OptionManager reader;
  reader.AddExhaustivePairingOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_TRUE(reader.feature_matching->guided_matching);
  EXPECT_EQ(reader.two_view_geometry->min_num_inliers, 42);
  EXPECT_EQ(reader.exhaustive_pairing->block_size, 77);
}

// --------------------------------------------------------------------------
// Write/Read round-trip for additional option categories
// --------------------------------------------------------------------------

TEST(OptionManager, WriteAndReadBundleAdjustmentOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddBundleAdjustmentOptions();
  writer.bundle_adjustment->refine_focal_length = false;
  writer.bundle_adjustment->refine_principal_point = true;
  writer.bundle_adjustment->min_track_length = 5;
  writer.bundle_adjustment->ceres->solver_options.max_num_iterations = 200;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddBundleAdjustmentOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_FALSE(reader.bundle_adjustment->refine_focal_length);
  EXPECT_TRUE(reader.bundle_adjustment->refine_principal_point);
  EXPECT_EQ(reader.bundle_adjustment->min_track_length, 5);
  EXPECT_EQ(reader.bundle_adjustment->ceres->solver_options.max_num_iterations,
            200);
}

TEST(OptionManager, WriteAndReadStereoOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddPatchMatchStereoOptions();
  writer.AddStereoFusionOptions();
  writer.patch_match_stereo->max_image_size = 999;
  writer.patch_match_stereo->window_radius = 7;
  writer.stereo_fusion->min_num_pixels = 13;
  writer.stereo_fusion->max_reproj_error = 1.5;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddPatchMatchStereoOptions();
  reader.AddStereoFusionOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.patch_match_stereo->max_image_size, 999);
  EXPECT_EQ(reader.patch_match_stereo->window_radius, 7);
  EXPECT_EQ(reader.stereo_fusion->min_num_pixels, 13);
  EXPECT_DOUBLE_EQ(reader.stereo_fusion->max_reproj_error, 1.5);
}

TEST(OptionManager, WriteAndReadMeshingOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddPoissonMeshingOptions();
  writer.AddDelaunayMeshingOptions();
  writer.poisson_meshing->depth = 11;
  writer.poisson_meshing->trim = 8.0;
  writer.delaunay_meshing->max_proj_dist = 25.0;
  writer.delaunay_meshing->num_threads = 4;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddPoissonMeshingOptions();
  reader.AddDelaunayMeshingOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.poisson_meshing->depth, 11);
  EXPECT_DOUBLE_EQ(reader.poisson_meshing->trim, 8.0);
  EXPECT_DOUBLE_EQ(reader.delaunay_meshing->max_proj_dist, 25.0);
  EXPECT_EQ(reader.delaunay_meshing->num_threads, 4);
}

TEST(OptionManager, WriteAndReadRenderOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddRenderOptions();
  writer.render->min_track_len = 5;
  writer.render->max_error = 3.0;
  writer.render->refresh_rate = 2;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddRenderOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.render->min_track_len, 5);
  EXPECT_DOUBLE_EQ(reader.render->max_error, 3.0);
  EXPECT_EQ(reader.render->refresh_rate, 2);
}

TEST(OptionManager, WriteAndReadGravityRefinerOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddGravityRefinerOptions();
  writer.gravity_refiner->max_outlier_ratio = 0.3;
  writer.gravity_refiner->max_gravity_error = 5.0;
  writer.gravity_refiner->min_num_neighbors = 10;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddGravityRefinerOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_DOUBLE_EQ(reader.gravity_refiner->max_outlier_ratio, 0.3);
  EXPECT_DOUBLE_EQ(reader.gravity_refiner->max_gravity_error, 5.0);
  EXPECT_EQ(reader.gravity_refiner->min_num_neighbors, 10);
}

TEST(OptionManager, WriteAndReadReconstructionClustererOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddReconstructionClustererOptions();
  writer.reconstruction_clusterer->min_covisibility_count = 42;
  writer.reconstruction_clusterer->min_num_reg_frames = 7;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddReconstructionClustererOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.reconstruction_clusterer->min_covisibility_count, 42);
  EXPECT_EQ(reader.reconstruction_clusterer->min_num_reg_frames, 7);
}

TEST(OptionManager, WriteAndReadSequentialPairingOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddSequentialPairingOptions();
  writer.sequential_pairing->overlap = 15;
  writer.sequential_pairing->quadratic_overlap = true;
  writer.sequential_pairing->loop_detection = true;
  writer.sequential_pairing->loop_detection_num_images = 100;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddSequentialPairingOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.sequential_pairing->overlap, 15);
  EXPECT_TRUE(reader.sequential_pairing->quadratic_overlap);
  EXPECT_TRUE(reader.sequential_pairing->loop_detection);
  EXPECT_EQ(reader.sequential_pairing->loop_detection_num_images, 100);
}

TEST(OptionManager, WriteAndReadSpatialPairingOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddSpatialPairingOptions();
  writer.spatial_pairing->ignore_z = true;
  writer.spatial_pairing->max_num_neighbors = 100;
  writer.spatial_pairing->max_distance = 200.0;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddSpatialPairingOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_TRUE(reader.spatial_pairing->ignore_z);
  EXPECT_EQ(reader.spatial_pairing->max_num_neighbors, 100);
  EXPECT_DOUBLE_EQ(reader.spatial_pairing->max_distance, 200.0);
}

TEST(OptionManager, WriteAndReadTransitivePairingOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddTransitivePairingOptions();
  writer.transitive_pairing->batch_size = 500;
  writer.transitive_pairing->num_iterations = 7;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddTransitivePairingOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.transitive_pairing->batch_size, 500);
  EXPECT_EQ(reader.transitive_pairing->num_iterations, 7);
}

TEST(OptionManager, WriteAndReadTwoViewGeometryOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddTwoViewGeometryOptions();
  writer.two_view_geometry->min_num_inliers = 30;
  writer.two_view_geometry->detect_watermark = true;
  writer.two_view_geometry->ransac_options.max_error = 5.0;
  writer.two_view_geometry->ransac_options.confidence = 0.9;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddTwoViewGeometryOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.two_view_geometry->min_num_inliers, 30);
  EXPECT_TRUE(reader.two_view_geometry->detect_watermark);
  EXPECT_DOUBLE_EQ(reader.two_view_geometry->ransac_options.max_error, 5.0);
  EXPECT_DOUBLE_EQ(reader.two_view_geometry->ransac_options.confidence, 0.9);
}

TEST(OptionManager, WriteAndReadGlobalMapperOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  OptionManager writer;
  writer.AddGlobalMapperOptions();
  writer.global_mapper->min_num_matches = 30;
  writer.global_mapper->num_threads = 8;
  writer.global_mapper->mapper.ba_num_iterations = 42;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddGlobalMapperOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.global_mapper->min_num_matches, 30);
  EXPECT_EQ(reader.global_mapper->num_threads, 8);
  EXPECT_EQ(reader.global_mapper->mapper.ba_num_iterations, 42);
}

// --------------------------------------------------------------------------
// Read INI with unregistered options (warning path, should still succeed)
// --------------------------------------------------------------------------

TEST(OptionManager, ReadWithUnregisteredOptionsInINI) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Write an INI file that includes keys not registered with OptionManager
  {
    std::ofstream file(config_path);
    file << "SiftExtraction.max_num_features=1234\n";
    file << "UnknownSection.unknown_key=some_value\n";
  }

  OptionManager options;
  options.AddFeatureExtractionOptions();

  // allow_unregistered=true (default) should succeed
  EXPECT_TRUE(options.Read(config_path, /*allow_unregistered=*/true));
  EXPECT_EQ(options.feature_extraction->sift->max_num_features, 1234);
}

TEST(OptionManager, ReadWithDisallowedUnregisteredOptionsInINI) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  {
    std::ofstream file(config_path);
    file << "SiftExtraction.max_num_features=1234\n";
    file << "UnknownSection.unknown_key=some_value\n";
  }

  OptionManager options;
  options.AddFeatureExtractionOptions();

  // allow_unregistered=false should fail on unrecognized option
  EXPECT_FALSE(options.Read(config_path, /*allow_unregistered=*/false));
}

// --------------------------------------------------------------------------
// PostParse: image list, constant rig list, constant camera list
// --------------------------------------------------------------------------

TEST(OptionManager, PostParseMapperImageList) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");
  const auto image_list_path = test_dir / "image_list.txt";
  const auto config_path = test_dir / "config.ini";

  // Write an image list file
  {
    std::ofstream file(image_list_path);
    file << "img001.jpg\n";
    file << "img002.jpg\n";
    file << "img003.jpg\n";
  }

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddMapperOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--Mapper.image_list_path",
      image_list_path.string(),
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // PostParse should have populated image_names from the file
  ASSERT_EQ(options.mapper->image_names.size(), 3);
  EXPECT_EQ(options.mapper->image_names[0], "img001.jpg");
  EXPECT_EQ(options.mapper->image_names[1], "img002.jpg");
  EXPECT_EQ(options.mapper->image_names[2], "img003.jpg");
}

TEST(OptionManager, PostParseGlobalMapperImageList) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");
  const auto image_list_path = test_dir / "global_image_list.txt";
  const auto config_path = test_dir / "config.ini";

  {
    std::ofstream file(image_list_path);
    file << "a.jpg\n";
    file << "b.jpg\n";
  }

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddGlobalMapperOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--GlobalMapper.image_list_path",
      image_list_path.string(),
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  ASSERT_EQ(options.global_mapper->image_names.size(), 2);
  EXPECT_EQ(options.global_mapper->image_names[0], "a.jpg");
  EXPECT_EQ(options.global_mapper->image_names[1], "b.jpg");
}

TEST(OptionManager, PostParseConstantRigAndCameraLists) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");
  const auto rig_list_path = test_dir / "constant_rigs.txt";
  const auto camera_list_path = test_dir / "constant_cameras.txt";

  {
    std::ofstream file(rig_list_path);
    file << "1\n";
    file << "3\n";
    file << "5\n";
  }
  {
    std::ofstream file(camera_list_path);
    file << "10\n";
    file << "20\n";
  }

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddMapperOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--Mapper.constant_rig_list_path",
      rig_list_path.string(),
      "--Mapper.constant_camera_list_path",
      camera_list_path.string(),
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  EXPECT_EQ(options.mapper->constant_rigs.size(), 3);
  EXPECT_TRUE(options.mapper->constant_rigs.count(1));
  EXPECT_TRUE(options.mapper->constant_rigs.count(3));
  EXPECT_TRUE(options.mapper->constant_rigs.count(5));

  EXPECT_EQ(options.mapper->constant_cameras.size(), 2);
  EXPECT_TRUE(options.mapper->constant_cameras.count(10));
  EXPECT_TRUE(options.mapper->constant_cameras.count(20));
}

TEST(OptionManager, PostParseEmptyPaths) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddMapperOptions();
  options.AddGlobalMapperOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  // No image_list_path or constant_*_list_path specified
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  // With no list paths, the collections should remain empty
  EXPECT_TRUE(options.mapper->image_names.empty());
  EXPECT_TRUE(options.global_mapper->image_names.empty());
  EXPECT_TRUE(options.mapper->constant_rigs.empty());
  EXPECT_TRUE(options.mapper->constant_cameras.empty());
}

// --------------------------------------------------------------------------
// Parsing feature matching options via command-line
// --------------------------------------------------------------------------

TEST(OptionManager, ParseFeatureMatchingOptions) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddFeatureMatchingOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--FeatureMatching.guided_matching",
      "1",
      "--FeatureMatching.max_num_matches",
      "50000",
      "--SiftMatching.max_ratio",
      "0.7",
      "--SiftMatching.cross_check",
      "1",
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  EXPECT_TRUE(options.feature_matching->guided_matching);
  EXPECT_EQ(options.feature_matching->max_num_matches, 50000);
  EXPECT_NEAR(options.feature_matching->sift->max_ratio, 0.7, 1e-6);
  EXPECT_TRUE(options.feature_matching->sift->cross_check);
}

// --------------------------------------------------------------------------
// Parsing mapper sub-options via command-line
// --------------------------------------------------------------------------

TEST(OptionManager, ParseMapperSubOptions) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddMapperOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  // Test triangulation and incremental mapper sub-options
  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--Mapper.init_min_num_inliers",
      "200",
      "--Mapper.abs_pose_max_error",
      "10.0",
      "--Mapper.tri_min_angle",
      "2.0",
      "--Mapper.tri_max_transitivity",
      "3",
      "--Mapper.fix_existing_frames",
      "1",
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  EXPECT_EQ(options.mapper->mapper.init_min_num_inliers, 200);
  EXPECT_DOUBLE_EQ(options.mapper->mapper.abs_pose_max_error, 10.0);
  EXPECT_DOUBLE_EQ(options.mapper->triangulation.min_angle, 2.0);
  EXPECT_EQ(options.mapper->triangulation.max_transitivity, 3);
  EXPECT_TRUE(options.mapper->fix_existing_frames);
}

// --------------------------------------------------------------------------
// Reset then re-add works correctly
// --------------------------------------------------------------------------

TEST(OptionManager, ResetThenReAdd) {
  OptionManager options;
  options.AddFeatureExtractionOptions();
  options.feature_extraction->max_image_size = 999;

  options.Reset();

  // After Reset, all added_* flags are cleared.
  // Re-adding and writing/reading should work.
  options.AddFeatureExtractionOptions();
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  options.feature_extraction->max_image_size = 777;
  options.Write(config_path);

  OptionManager reader;
  reader.AddFeatureExtractionOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.feature_extraction->max_image_size, 777);
}

// --------------------------------------------------------------------------
// Check() validates sub-option structs
// --------------------------------------------------------------------------

TEST(OptionManager, CheckDefaultOptionsPass) {
  // With no database or image options added, Check() should pass on defaults
  OptionManager options;
  EXPECT_TRUE(options.Check());
}

// --------------------------------------------------------------------------
// ResetOptions resets all sub-option structs to defaults
// --------------------------------------------------------------------------

TEST(OptionManager, ResetOptionsResetsAllSubOptions) {
  OptionManager options;

  // Modify several sub-option structs
  const mvs::PatchMatchOptions pm_defaults;
  options.patch_match_stereo->max_image_size = 12345;
  options.stereo_fusion->min_num_pixels = 99;
  options.poisson_meshing->depth = 42;
  options.bundle_adjustment->refine_focal_length = false;

  options.ResetOptions(/*reset_paths=*/false);

  EXPECT_EQ(options.patch_match_stereo->max_image_size,
            pm_defaults.max_image_size);
  EXPECT_NE(options.stereo_fusion->min_num_pixels, 99);
  EXPECT_NE(options.poisson_meshing->depth, 42);
  EXPECT_TRUE(options.bundle_adjustment->refine_focal_length);
}

// --------------------------------------------------------------------------
// Constructor with add_project_options=false
// --------------------------------------------------------------------------

TEST(OptionManager, ConstructorWithoutProjectOptions) {
  OptionManager options(/*add_project_options=*/false);

  // Should still have valid sub-option pointers
  EXPECT_NE(options.image_reader, nullptr);
  EXPECT_NE(options.feature_extraction, nullptr);
  EXPECT_NE(options.mapper, nullptr);

  // Should be able to add and use options
  options.AddFeatureExtractionOptions();
  options.feature_extraction->max_image_size = 512;

  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";
  options.Write(config_path);

  OptionManager reader(/*add_project_options=*/false);
  reader.AddFeatureExtractionOptions();
  EXPECT_TRUE(reader.Read(config_path));
  EXPECT_EQ(reader.feature_extraction->max_image_size, 512);
}

// --------------------------------------------------------------------------
// Write/Read round-trip with all options preserves values
// --------------------------------------------------------------------------

TEST(OptionManager, WriteAndReadAllOptions) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";
  CreateDirIfNotExists(test_dir / "images");

  OptionManager writer;
  writer.AddAllOptions();
  writer.AddGravityRefinerOptions();
  writer.AddReconstructionClustererOptions();
  *writer.database_path = test_dir / "database.db";
  *writer.image_path = test_dir / "images";
  writer.feature_extraction->max_image_size = 1111;
  writer.feature_matching->max_num_matches = 22222;
  writer.two_view_geometry->min_num_inliers = 33;
  writer.exhaustive_pairing->block_size = 44;
  writer.sequential_pairing->overlap = 55;
  writer.bundle_adjustment->min_track_length = 4;
  writer.mapper->min_num_matches = 66;
  writer.patch_match_stereo->num_iterations = 7;
  writer.stereo_fusion->min_num_pixels = 8;
  writer.poisson_meshing->depth = 9;
  writer.delaunay_meshing->num_threads = 2;
  writer.render->min_track_len = 11;
  writer.gravity_refiner->min_num_neighbors = 12;
  writer.reconstruction_clusterer->min_num_reg_frames = 13;
  writer.Write(config_path);

  OptionManager reader;
  reader.AddAllOptions();
  reader.AddGravityRefinerOptions();
  reader.AddReconstructionClustererOptions();
  EXPECT_TRUE(reader.Read(config_path));

  EXPECT_EQ(reader.feature_extraction->max_image_size, 1111);
  EXPECT_EQ(reader.feature_matching->max_num_matches, 22222);
  EXPECT_EQ(reader.two_view_geometry->min_num_inliers, 33);
  EXPECT_EQ(reader.exhaustive_pairing->block_size, 44);
  EXPECT_EQ(reader.sequential_pairing->overlap, 55);
  EXPECT_EQ(reader.bundle_adjustment->min_track_length, 4);
  EXPECT_EQ(reader.mapper->min_num_matches, 66);
  EXPECT_EQ(reader.patch_match_stereo->num_iterations, 7);
  EXPECT_EQ(reader.stereo_fusion->min_num_pixels, 8);
  EXPECT_EQ(reader.poisson_meshing->depth, 9);
  EXPECT_EQ(reader.delaunay_meshing->num_threads, 2);
  EXPECT_EQ(reader.render->min_track_len, 11);
  EXPECT_EQ(reader.gravity_refiner->min_num_neighbors, 12);
  EXPECT_EQ(reader.reconstruction_clusterer->min_num_reg_frames, 13);
}

// --------------------------------------------------------------------------
// ModifyForVideoData resets options first, then modifies
// --------------------------------------------------------------------------

TEST(OptionManager, ModifyForVideoDataResetsFirst) {
  OptionManager options;

  // Set a non-default value that should get reset by ModifyForVideoData
  options.feature_extraction->max_image_size = 12345;

  options.ModifyForVideoData();

  // ModifyForVideoData calls ResetOptions(false), so feature_extraction
  // should be back to defaults
  const FeatureExtractionOptions fe_defaults;
  EXPECT_EQ(options.feature_extraction->max_image_size,
            fe_defaults.max_image_size);
}

// --------------------------------------------------------------------------
// Parsing PatchMatchStereo options via command-line
// --------------------------------------------------------------------------

TEST(OptionManager, ParsePatchMatchStereoOptions) {
  OptionManager options;
  options.AddPatchMatchStereoOptions();

  const std::vector<std::string> args = {
      "colmap",
      "--PatchMatchStereo.max_image_size",
      "800",
      "--PatchMatchStereo.window_radius",
      "3",
      "--PatchMatchStereo.geom_consistency",
      "1",
      "--PatchMatchStereo.filter",
      "0",
      "--PatchMatchStereo.num_iterations",
      "10",
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  EXPECT_EQ(options.patch_match_stereo->max_image_size, 800);
  EXPECT_EQ(options.patch_match_stereo->window_radius, 3);
  EXPECT_TRUE(options.patch_match_stereo->geom_consistency);
  EXPECT_FALSE(options.patch_match_stereo->filter);
  EXPECT_EQ(options.patch_match_stereo->num_iterations, 10);
}

// --------------------------------------------------------------------------
// Read with Check failure (Read calls Check and returns false)
// --------------------------------------------------------------------------

TEST(OptionManager, ReadFailsOnCheckFailure) {
  const auto test_dir = CreateTestDir();
  const auto config_path = test_dir / "config.ini";

  // Write config with an image_path that doesn't exist
  OptionManager writer;
  writer.AddDatabaseOptions();
  writer.AddImageOptions();
  *writer.database_path = test_dir / "database.db";
  *writer.image_path = test_dir / "nonexistent_images";
  writer.Write(config_path);

  OptionManager reader;
  reader.AddDatabaseOptions();
  reader.AddImageOptions();

  // OptionManager::Read calls Check() which should fail
  // because image_path directory doesn't exist
  EXPECT_FALSE(reader.Read(config_path));
}

// --------------------------------------------------------------------------
// Parsing global mapper sub-options via command-line
// --------------------------------------------------------------------------

TEST(OptionManager, ParseGlobalMapperSubOptions) {
  const auto test_dir = CreateTestDir();
  CreateDirIfNotExists(test_dir / "images");

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddGlobalMapperOptions();

  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";

  const std::vector<std::string> args = {
      "colmap",
      "--database_path",
      database_path.string(),
      "--image_path",
      image_path.string(),
      "--GlobalMapper.min_num_matches",
      "50",
      "--GlobalMapper.ba_num_iterations",
      "25",
      "--GlobalMapper.skip_rotation_averaging",
      "1",
      "--GlobalMapper.tri_min_angle",
      "3.0",
      "--GlobalMapper.ra_max_rotation_error_deg",
      "7.5",
  };

  auto argv = MakeArgv(args);
  EXPECT_TRUE(options.Parse(argv.size(), argv.data()));

  EXPECT_EQ(options.global_mapper->min_num_matches, 50);
  EXPECT_EQ(options.global_mapper->mapper.ba_num_iterations, 25);
  EXPECT_TRUE(options.global_mapper->mapper.skip_rotation_averaging);
  EXPECT_DOUBLE_EQ(options.global_mapper->mapper.retriangulation.min_angle,
                   3.0);
  EXPECT_DOUBLE_EQ(
      options.global_mapper->mapper.rotation_averaging.max_rotation_error_deg,
      7.5);
}

// --------------------------------------------------------------------------
// Quality modifiers are cumulative on pre-modified options
// --------------------------------------------------------------------------

TEST(OptionManager, QualityModifiersApplyToCurrentState) {
  OptionManager options;

  // First apply high quality
  options.ModifyForHighQuality();
  EXPECT_EQ(options.mapper->ba_local_max_num_iterations, 30);

  // Then apply low quality on top - should halve the already-set value
  // (low quality divides by 2, so 30/2 = 15)
  options.ModifyForLowQuality();
  EXPECT_EQ(options.mapper->ba_local_max_num_iterations, 15);
}

}  // namespace
}  // namespace colmap
