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

#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/pairing.h"
#include "colmap/feature/sift.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/ui/render_options.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

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
  const std::string test_dir = CreateTestDir();
  const std::string config_path = test_dir + "/config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir + "/images");

  // Create and configure an OptionManager
  OptionManager options_write;
  options_write.AddDatabaseOptions();
  options_write.AddImageOptions();
  options_write.AddFeatureExtractionOptions();
  options_write.AddMapperOptions();

  *options_write.database_path = test_dir + "/database.db";
  *options_write.image_path = test_dir + "/images";
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
  const std::string test_dir = CreateTestDir();
  const std::string config_path = test_dir + "/config.ini";

  // Create necessary directories
  CreateDirIfNotExists(test_dir + "/images");

  // Create and write initial config
  OptionManager options_write;
  options_write.AddAllOptions();
  *options_write.database_path = test_dir + "/database.db";
  *options_write.image_path = test_dir + "/images";
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
  const std::string test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();

  // Should fail with non-existent paths
  *options.database_path = test_dir + "/database.db";
  *options.image_path = "/path/that/does/not/exist";
  EXPECT_FALSE(options.Check());

  // Should succeed with valid paths
  CreateDirIfNotExists(test_dir + "/images");
  *options.image_path = test_dir + "/images";
  EXPECT_TRUE(options.Check());
}

TEST(OptionManager, CheckDatabaseParentDir) {
  const std::string test_dir = CreateTestDir();

  OptionManager options;
  options.AddDatabaseOptions();

  // Should succeed when database parent dir exists
  *options.database_path = test_dir + "/database.db";
  EXPECT_TRUE(options.Check());

  // Should fail when database path is a directory
  CreateDirIfNotExists(test_dir + "/bad_database");
  *options.database_path = test_dir + "/bad_database";
  EXPECT_FALSE(options.Check());
}

}  // namespace
}  // namespace colmap
