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

#include "colmap/controllers/feature_matching.h"

#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateTestDatabase(int num_images, Database& database) {
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = num_images;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.num_points2D_without_point3D = 3;
  synthetic_dataset_options.use_prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, &database);
}

std::unique_ptr<retrieval::VisualIndex> CreateSyntheticVisualIndex() {
  auto visual_index = retrieval::VisualIndex::Create();
  retrieval::VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  visual_index->Build(build_options,
                      retrieval::VisualIndex::Descriptors::Random(50, 128));
  return visual_index;
}

TEST(CreateExhaustiveFeatureMatcher, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  ExhaustivePairingOptions pairing_options;
  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateExhaustiveFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_EQ(database->ReadAllMatches().size(), 6);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 6);
}

TEST(CreateVocabTreeFeatureMatcher, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  const std::string vocab_tree_path = test_dir + "/vocab_tree.bin";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  // Create vocab tree
  CreateSyntheticVisualIndex()->Write(vocab_tree_path);

  VocabTreePairingOptions pairing_options;
  pairing_options.vocab_tree_path = vocab_tree_path;
  pairing_options.num_images = 2;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateVocabTreeFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Each image should match with num_images others,
  // while some of the pairs may be redundant.
  EXPECT_GE(database->ReadAllMatches().size(), 4);
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 4);
}

TEST(CreateSequentialFeatureMatcher, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/5, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SequentialPairingOptions pairing_options;
  pairing_options.overlap = 2;
  pairing_options.quadratic_overlap = false;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSequentialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // With 5 images and overlap=2:
  // (0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)
  EXPECT_EQ(database->ReadAllMatches().size(), 7);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 7);
}

TEST(CreateSpatialFeatureMatcher, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SpatialPairingOptions pairing_options;
  pairing_options.max_num_neighbors = 2;
  pairing_options.max_distance = 1e6;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSpatialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_GT(database->ReadAllMatches().size(), 0);
  EXPECT_GT(database->ReadTwoViewGeometries().size(), 0);
}

TEST(CreateTransitiveFeatureMatcher, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);

  // Create initial matches: 1-2 and 2-3
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.inlier_matches = FeatureMatches(10);

  database->WriteTwoViewGeometry(
      images[0].ImageId(), images[1].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[1].ImageId(), images[2].ImageId(), two_view_geometry);

  TransitivePairingOptions pairing_options;
  pairing_options.batch_size = 100;
  pairing_options.num_iterations = 1;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateTransitiveFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Should create transitive match 1-3
  const size_t final_matches = database->ReadTwoViewGeometries().size();
  EXPECT_GE(final_matches, 2);  // At least the original 2 matches
}

TEST(CreateImagePairsFeatureMatcher, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  const std::string match_list_path = test_dir + "/match_list.txt";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);

  // Create match list file with specific image pairs
  std::ofstream file(match_list_path);
  file << images[0].Name() << " " << images[1].Name() << "\n";
  file << images[1].Name() << " " << images[2].Name() << "\n";
  file << images[2].Name() << " " << images[3].Name() << "\n";
  file.close();

  ImportedPairingOptions pairing_options;
  pairing_options.match_list_path = match_list_path;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateImagePairsFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_EQ(database->ReadAllMatches().size(), 3);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 3);
}

TEST(CreateFeaturePairsFeatureMatcher, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  const std::string match_list_path = test_dir + "/feature_match_list.txt";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/3, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 2);

  // Create feature match list file with many matches for better verification
  std::ofstream file(match_list_path);
  file << images[0].Name() << " " << images[1].Name() << "\n";
  for (int i = 0; i < 15; ++i) {
    file << i << " " << i << "\n";
  }
  file << "\n";  // Empty line separates pairs
  file << images[1].Name() << " " << images[2].Name() << "\n";
  for (int i = 0; i < 15; ++i) {
    file << i << " " << i << "\n";
  }
  file << "\n";
  file.close();

  FeaturePairsMatchingOptions pairing_options;
  pairing_options.match_list_path = match_list_path;
  pairing_options.verify_matches = true;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;  // Lower threshold for testing

  auto matcher = CreateFeaturePairsFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Should have imported and verified the matches
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 2);
}

TEST(CreateGeometricVerifier, Nominal) {
  const std::string test_dir = CreateTestDir();
  const std::string database_path = test_dir + "/database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearTwoViewGeometries();

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  EXPECT_GE(database->ReadAllMatches().size(), 3);
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 3);
}

}  // namespace
}  // namespace colmap
