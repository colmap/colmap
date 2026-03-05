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

#include "colmap/controllers/pairing.h"

#include "colmap/feature/types.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateSyntheticDatabase(int num_images, Database& database) {
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = num_images;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, &database);
}

TEST(ExhaustivePairGenerator, Nominal) {
  constexpr int kNumImages = 34;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  ExhaustivePairingOptions options;
  options.block_size = 10;
  ExhaustivePairGenerator generator(options, database);
  const int num_expected_blocks =
      std::ceil(static_cast<double>(kNumImages) / options.block_size) *
      std::ceil(static_cast<double>(kNumImages) / options.block_size);
  std::set<std::pair<image_t, image_t>> pairs;
  for (int i = 0; i < num_expected_blocks; ++i) {
    for (const auto& pair : generator.Next()) {
      pairs.insert(pair);
    }
  }
  EXPECT_EQ(pairs.size(), kNumImages * (kNumImages - 1) / 2);
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(ExhaustivePairGenerator, AllPairs) {
  constexpr int kNumImages = 10;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  ExhaustivePairingOptions options;
  options.block_size = 3;
  ExhaustivePairGenerator generator(options, database);
  const auto all_pairs = generator.AllPairs();

  std::set<std::pair<image_t, image_t>> unique_pairs(all_pairs.begin(),
                                                     all_pairs.end());
  EXPECT_EQ(unique_pairs.size(), kNumImages * (kNumImages - 1) / 2);
  EXPECT_TRUE(generator.HasFinished());
}

TEST(ExhaustivePairGenerator, Reset) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  ExhaustivePairingOptions options;
  options.block_size = 3;
  ExhaustivePairGenerator generator(options, database);

  const auto all_pairs_first = generator.AllPairs();
  EXPECT_TRUE(generator.HasFinished());

  generator.Reset();
  EXPECT_FALSE(generator.HasFinished());

  const auto all_pairs_second = generator.AllPairs();
  EXPECT_EQ(all_pairs_first.size(), all_pairs_second.size());
}

TEST(ExhaustivePairGenerator, SingleImage) {
  constexpr int kNumImages = 1;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  ExhaustivePairingOptions options;
  options.block_size = 10;
  ExhaustivePairGenerator generator(options, database);
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(ExhaustivePairGenerator, BlockSizeEqualToNumImages) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  ExhaustivePairingOptions options;
  options.block_size = kNumImages;
  ExhaustivePairGenerator generator(options, database);
  const auto all_pairs = generator.AllPairs();

  std::set<std::pair<image_t, image_t>> unique_pairs(all_pairs.begin(),
                                                     all_pairs.end());
  EXPECT_EQ(unique_pairs.size(), kNumImages * (kNumImages - 1) / 2);
}

TEST(ExhaustivePairGenerator, BlockSizeLargerThanNumImages) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  ExhaustivePairingOptions options;
  options.block_size = 100;
  ExhaustivePairGenerator generator(options, database);
  const auto all_pairs = generator.AllPairs();

  std::set<std::pair<image_t, image_t>> unique_pairs(all_pairs.begin(),
                                                     all_pairs.end());
  EXPECT_EQ(unique_pairs.size(), kNumImages * (kNumImages - 1) / 2);
}

std::unique_ptr<retrieval::VisualIndex> CreateSyntheticVisualIndex() {
  auto visual_index = retrieval::VisualIndex::Create();
  retrieval::VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  visual_index->Build(
      build_options,
      FeatureDescriptorsFloat(FeatureExtractorType::SIFT,
                              FeatureDescriptorsFloatData::Random(50, 128)));
  return visual_index;
}

TEST(VocabTreePairGenerator, Nominal) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  VocabTreePairingOptions options;
  options.vocab_tree_path = CreateTestDir() / "vocab_tree.txt";

  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  CreateSyntheticVisualIndex()->Write(options.vocab_tree_path);

  {
    options.num_images = 3;
    VocabTreePairGenerator generator(options, database);
    for (int i = 0; i < kNumImages; ++i) {
      const auto pairs = generator.Next();
      EXPECT_EQ(pairs.size(), options.num_images);
      EXPECT_EQ(
          (std::set<std::pair<image_t, image_t>>(pairs.begin(), pairs.end())
               .size()),
          pairs.size());
    }
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }

  {
    options.num_images = 100;
    VocabTreePairGenerator generator(options, database);
    for (int i = 0; i < kNumImages; ++i) {
      const auto pairs = generator.Next();
      EXPECT_EQ(pairs.size(), kNumImages);
    }
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }
}

TEST(VocabTreePairGenerator, Reset) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  VocabTreePairingOptions options;
  options.vocab_tree_path = CreateTestDir() / "vocab_tree_reset.txt";
  options.num_images = 2;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  CreateSyntheticVisualIndex()->Write(options.vocab_tree_path);

  VocabTreePairGenerator generator(options, database);

  size_t first_run_count = 0;
  while (!generator.HasFinished()) {
    first_run_count += generator.Next().size();
  }
  EXPECT_GT(first_run_count, 0);

  generator.Reset();
  EXPECT_FALSE(generator.HasFinished());

  size_t second_run_count = 0;
  while (!generator.HasFinished()) {
    second_run_count += generator.Next().size();
  }
  EXPECT_EQ(first_run_count, second_run_count);
}

TEST(VocabTreePairGenerator, MatchListPath) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  const auto test_dir = CreateTestDir();
  VocabTreePairingOptions options;
  options.vocab_tree_path = test_dir / "vocab_tree_ml.txt";
  options.num_images = 3;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  CreateSyntheticVisualIndex()->Write(options.vocab_tree_path);

  // Write a match list with a subset of images (2 out of 5).
  options.match_list_path = test_dir / "match_list.txt";
  {
    std::ofstream file(options.match_list_path);
    file << images[0].Name() << "\n";
    file << "# This is a comment\n";
    file << "\n";
    file << images[2].Name() << "\n";
    file.close();
  }

  VocabTreePairGenerator generator(options, database);
  size_t query_count = 0;
  while (!generator.HasFinished()) {
    generator.Next();
    ++query_count;
  }
  // Only 2 images are queried, so we expect exactly 2 calls that produce
  // results.
  EXPECT_EQ(query_count, 2);
}

TEST(SequentialPairGenerator, Linear) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  SequentialPairingOptions options;
  options.overlap = 3;
  options.quadratic_overlap = false;
  SequentialPairGenerator generator(options, database);
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[0].ImageId(), images[1].ImageId()),
                  std::make_pair(images[0].ImageId(), images[2].ImageId()),
                  std::make_pair(images[0].ImageId(), images[3].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[1].ImageId(), images[2].ImageId()),
                  std::make_pair(images[1].ImageId(), images[3].ImageId()),
                  std::make_pair(images[1].ImageId(), images[4].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[2].ImageId(), images[3].ImageId()),
                  std::make_pair(images[2].ImageId(), images[4].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[3].ImageId(), images[4].ImageId())));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(SequentialPairGenerator, LinearRig) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 3;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, database.get());
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(),
           synthetic_dataset_options.num_cameras_per_rig *
               synthetic_dataset_options.num_frames_per_rig);

  SequentialPairingOptions options;
  options.overlap = 1;
  options.quadratic_overlap = false;
  SequentialPairGenerator generator(options, database);
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[0].ImageId(), images[1].ImageId()),
                  std::make_pair(images[0].ImageId(), images[2].ImageId()),
                  std::make_pair(images[0].ImageId(), images[3].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[2].ImageId(), images[3].ImageId()),
                  std::make_pair(images[2].ImageId(), images[4].ImageId()),
                  std::make_pair(images[2].ImageId(), images[5].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[4].ImageId(), images[5].ImageId()),
                  std::make_pair(images[4].ImageId(), images[1].ImageId()),
                  std::make_pair(images[4].ImageId(), images[0].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[1].ImageId(), images[0].ImageId()),
                  std::make_pair(images[1].ImageId(), images[3].ImageId()),
                  std::make_pair(images[1].ImageId(), images[2].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[3].ImageId(), images[2].ImageId()),
                  std::make_pair(images[3].ImageId(), images[5].ImageId()),
                  std::make_pair(images[3].ImageId(), images[4].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[5].ImageId(), images[4].ImageId())));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(SequentialPairGenerator, Quadratic) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  SequentialPairingOptions options;
  options.overlap = 3;
  options.quadratic_overlap = true;
  SequentialPairGenerator generator(options, database);
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[0].ImageId(), images[1].ImageId()),
                  std::make_pair(images[0].ImageId(), images[2].ImageId()),
                  std::make_pair(images[0].ImageId(), images[4].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[1].ImageId(), images[2].ImageId()),
                  std::make_pair(images[1].ImageId(), images[3].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[2].ImageId(), images[3].ImageId()),
                  std::make_pair(images[2].ImageId(), images[4].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[3].ImageId(), images[4].ImageId())));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(SequentialPairGenerator, Reset) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  SequentialPairingOptions options;
  options.overlap = 2;
  options.quadratic_overlap = false;
  SequentialPairGenerator generator(options, database);

  const auto first_run = generator.AllPairs();
  EXPECT_TRUE(generator.HasFinished());

  generator.Reset();
  EXPECT_FALSE(generator.HasFinished());

  const auto second_run = generator.AllPairs();
  EXPECT_EQ(first_run.size(), second_run.size());
}

TEST(SequentialPairGenerator, NoRigExpansion) {
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 3;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, database.get());
  const std::vector<Image> images = database->ReadAllImages();

  SequentialPairingOptions options;
  options.overlap = 1;
  options.quadratic_overlap = false;
  options.expand_rig_images = false;
  SequentialPairGenerator generator(options, database);

  const auto all_pairs = generator.AllPairs();
  // With expand_rig_images=false on 6 images, overlap=1, we get exactly 5
  // sequential pairs (each image paired with the next one).
  EXPECT_EQ(all_pairs.size(), images.size() - 1);
}

TEST(SequentialPairGenerator, OverlapLargerThanNumImages) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);

  SequentialPairingOptions options;
  options.overlap = 100;
  options.quadratic_overlap = false;
  SequentialPairGenerator generator(options, database);
  const auto all_pairs = generator.AllPairs();

  std::set<std::pair<image_t, image_t>> unique_pairs(all_pairs.begin(),
                                                     all_pairs.end());
  // Each image is paired with all later images.
  EXPECT_EQ(unique_pairs.size(), kNumImages * (kNumImages - 1) / 2);
}

TEST(SpatialPairGenerator, Nominal) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = images[0].DataId();
  pose_prior1.position = Eigen::Vector3d(1, 2, 3);
  database->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = images[1].DataId();
  pose_prior2.position = Eigen::Vector3d(2, 3, 4);
  database->WritePosePrior(pose_prior2);

  PosePrior pose_prior3;
  pose_prior3.corr_data_id = images[2].DataId();
  pose_prior3.position = Eigen::Vector3d(2, 4, 12);
  database->WritePosePrior(pose_prior3);

  SpatialPairingOptions options;
  options.max_num_neighbors = 1;
  options.max_distance = 1000;
  options.ignore_z = false;

  {
    SpatialPairGenerator generator(options, database);

    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[0].ImageId(), images[1].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[1].ImageId(), images[0].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[2].ImageId(), images[1].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }

  {
    options.ignore_z = true;
    SpatialPairGenerator generator(options, database);
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[0].ImageId(), images[1].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[1].ImageId(), images[2].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[2].ImageId(), images[1].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }

  {
    options.ignore_z = false;
    options.max_distance = 5;
    SpatialPairGenerator generator(options, database);
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[0].ImageId(), images[1].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[1].ImageId(), images[0].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }

  {
    options.max_num_neighbors = 2;
    options.max_distance = 1000;
    SpatialPairGenerator generator(options, database);
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[0].ImageId(), images[1].ImageId()),
                    std::make_pair(images[0].ImageId(), images[2].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[1].ImageId(), images[0].ImageId()),
                    std::make_pair(images[1].ImageId(), images[2].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[2].ImageId(), images[1].ImageId()),
                    std::make_pair(images[2].ImageId(), images[0].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }
}

TEST(SpatialPairGenerator, LargeCoordinates) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = images[0].DataId();
  pose_prior1.position =
      Eigen::Vector3d(1, 2, 3) + Eigen::Vector3d::Constant(1e16);
  database->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = images[1].DataId();
  pose_prior2.position =
      Eigen::Vector3d(2, 3, 4) + Eigen::Vector3d::Constant(1e16);
  database->WritePosePrior(pose_prior2);

  PosePrior pose_prior3;
  pose_prior3.corr_data_id = images[2].DataId();
  pose_prior3.position =
      Eigen::Vector3d(2, 4, 12) + Eigen::Vector3d::Constant(1e16);
  database->WritePosePrior(pose_prior3);

  SpatialPairingOptions options;
  options.max_num_neighbors = 1;
  options.max_distance = 1000;
  options.ignore_z = false;

  SpatialPairGenerator generator(options, database);

  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[0].ImageId(), images[1].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[1].ImageId(), images[0].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[2].ImageId(), images[1].ImageId())));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(SpatialPairGenerator, MinNumNeighborsControlsMatchingDistance) {
  constexpr int kNumImages = 4;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const auto images = database->ReadAllImages();

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = images[0].DataId();
  pose_prior1.position = Eigen::Vector3d(1, 1, 2);
  database->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = images[1].DataId();
  pose_prior2.position = Eigen::Vector3d(1, 2, 3);
  database->WritePosePrior(pose_prior2);

  PosePrior pose_prior3;
  pose_prior3.corr_data_id = images[2].DataId();
  pose_prior3.position = Eigen::Vector3d(2, 3, 4);
  database->WritePosePrior(pose_prior3);

  PosePrior pose_prior4;
  pose_prior4.corr_data_id = images[3].DataId();
  pose_prior4.position = Eigen::Vector3d(2, 4, 12);
  database->WritePosePrior(pose_prior4);

  SpatialPairingOptions options;
  options.ignore_z = false;
  options.max_num_neighbors = kNumImages;
  options.max_distance = 0.0;

  {
    options.min_num_neighbors = 0;
    EXPECT_FALSE(options.Check());
  }
  {
    options.min_num_neighbors = 1;
    SpatialPairGenerator generator(options, database);
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[0].ImageId(), images[1].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[1].ImageId(), images[0].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[2].ImageId(), images[1].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[3].ImageId(), images[2].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
  }
  {
    options.min_num_neighbors = 2;
    SpatialPairGenerator generator(options, database);
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[0].ImageId(), images[1].ImageId()),
                    std::make_pair(images[0].ImageId(), images[2].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[1].ImageId(), images[0].ImageId()),
                    std::make_pair(images[1].ImageId(), images[2].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[2].ImageId(), images[1].ImageId()),
                    std::make_pair(images[2].ImageId(), images[0].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[3].ImageId(), images[2].ImageId()),
                    std::make_pair(images[3].ImageId(), images[1].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
  }
  {
    options.min_num_neighbors = 3;
    SpatialPairGenerator generator(options, database);
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[0].ImageId(), images[1].ImageId()),
                    std::make_pair(images[0].ImageId(), images[2].ImageId()),
                    std::make_pair(images[0].ImageId(), images[3].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[1].ImageId(), images[0].ImageId()),
                    std::make_pair(images[1].ImageId(), images[2].ImageId()),
                    std::make_pair(images[1].ImageId(), images[3].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[2].ImageId(), images[1].ImageId()),
                    std::make_pair(images[2].ImageId(), images[0].ImageId()),
                    std::make_pair(images[2].ImageId(), images[3].ImageId())));
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[3].ImageId(), images[2].ImageId()),
                    std::make_pair(images[3].ImageId(), images[1].ImageId()),
                    std::make_pair(images[3].ImageId(), images[0].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
  }
}

TEST(SpatialPairGenerator, ReadPositionPriorData) {
  {
    constexpr int kNumImages = 3;
    auto database = Database::Open(kInMemorySqliteDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);

    PosePrior pose_prior1;
    pose_prior1.corr_data_id = images[0].DataId();
    pose_prior1.position = Eigen::Vector3d(1, 2, 3);
    database->WritePosePrior(pose_prior1);

    PosePrior pose_prior2;
    pose_prior2.corr_data_id = images[1].DataId();
    pose_prior2.position = Eigen::Vector3d(2, 3, 4);
    database->WritePosePrior(pose_prior2);

    PosePrior pose_prior3;
    pose_prior3.corr_data_id = images[2].DataId();
    pose_prior3.position = Eigen::Vector3d(2, 4, 12);
    database->WritePosePrior(pose_prior3);

    SpatialPairingOptions options;
    options.max_num_neighbors = 1;
    options.max_distance = 1000;
    options.ignore_z = false;

    auto cache = std::make_shared<FeatureMatcherCache>(
        options.CacheSize(), THROW_CHECK_NOTNULL(database));
    SpatialPairGenerator generator(options, cache);

    Eigen::RowMajorMatrixXf position_matrix =
        generator.ReadPositionPriorData(*cache);
    EXPECT_EQ(position_matrix.rows(), 3);
  }

  {
    // Test that the position prior data is read correctly when some images
    // don't have a pose prior.

    constexpr int kNumImages = 4;
    auto database = Database::Open(kInMemorySqliteDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);

    database->ClearPosePriors();

    PosePrior pose_prior1;
    pose_prior1.corr_data_id = images[0].DataId();
    pose_prior1.position = Eigen::Vector3d(1, 2, 3);
    database->WritePosePrior(pose_prior1);

    PosePrior pose_prior2;
    pose_prior2.corr_data_id = images[1].DataId();
    pose_prior2.position = Eigen::Vector3d(2, 3, 4);
    database->WritePosePrior(pose_prior2);

    PosePrior pose_prior4;
    pose_prior4.corr_data_id = images[3].DataId();
    pose_prior4.position = Eigen::Vector3d(2, 4, 12);
    database->WritePosePrior(pose_prior4);

    SpatialPairingOptions options;
    options.max_num_neighbors = 1;
    options.max_distance = 1000;
    options.ignore_z = false;

    auto cache = std::make_shared<FeatureMatcherCache>(
        options.CacheSize(), THROW_CHECK_NOTNULL(database));
    SpatialPairGenerator generator(options, cache);

    Eigen::RowMajorMatrixXf position_matrix =
        generator.ReadPositionPriorData(*cache);
    EXPECT_EQ(position_matrix.rows(), 3);
  }

  {
    constexpr int kNumImages = 3;
    auto database = Database::Open(kInMemorySqliteDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);

    PosePrior pose_prior1;
    pose_prior1.corr_data_id = images[0].DataId();
    pose_prior1.position =
        Eigen::Vector3d(0, 0, std::numeric_limits<double>::quiet_NaN());
    database->WritePosePrior(pose_prior1);

    PosePrior pose_prior2;
    pose_prior2.corr_data_id = images[1].DataId();
    pose_prior2.position = Eigen::Vector3d(2, 3, 4);
    database->WritePosePrior(pose_prior2);

    PosePrior pose_prior3;
    pose_prior3.corr_data_id = images[2].DataId();
    pose_prior3.position = Eigen::Vector3d(2, 4, 12);
    database->WritePosePrior(pose_prior3);

    SpatialPairingOptions options;
    options.max_num_neighbors = 1;
    options.max_distance = 1000;
    options.ignore_z = false;

    auto cache = std::make_shared<FeatureMatcherCache>(
        options.CacheSize(), THROW_CHECK_NOTNULL(database));
    SpatialPairGenerator generator(options, cache);

    Eigen::RowMajorMatrixXf position_matrix =
        generator.ReadPositionPriorData(*cache);
    EXPECT_EQ(position_matrix.rows(), 2);
  }

  {
    constexpr int kNumImages = 3;
    auto database = Database::Open(kInMemorySqliteDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);

    PosePrior pose_prior1;
    pose_prior1.corr_data_id = images[0].DataId();
    pose_prior1.position =
        Eigen::Vector3d(0, 0, std::numeric_limits<double>::quiet_NaN());
    database->WritePosePrior(pose_prior1);

    PosePrior pose_prior2;
    pose_prior2.corr_data_id = images[1].DataId();
    pose_prior2.position = Eigen::Vector3d(2, 3, 4);
    database->WritePosePrior(pose_prior2);

    PosePrior pose_prior3;
    pose_prior3.corr_data_id = images[2].DataId();
    pose_prior3.position = Eigen::Vector3d(2, 4, 12);
    database->WritePosePrior(pose_prior3);

    SpatialPairingOptions options;
    options.max_num_neighbors = 1;
    options.max_distance = 1000;
    options.ignore_z = true;

    auto cache = std::make_shared<FeatureMatcherCache>(
        options.CacheSize(), THROW_CHECK_NOTNULL(database));
    SpatialPairGenerator generator(options, cache);

    Eigen::RowMajorMatrixXf position_matrix =
        generator.ReadPositionPriorData(*cache);
    EXPECT_EQ(position_matrix.rows(), 3);
  }

  {
    constexpr int kNumImages = 3;
    auto database = Database::Open(kInMemorySqliteDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);

    PosePrior pose_prior1;
    pose_prior1.corr_data_id = images[0].DataId();
    pose_prior1.position =
        Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
    database->WritePosePrior(pose_prior1);

    PosePrior pose_prior2;
    pose_prior2.corr_data_id = images[1].DataId();
    pose_prior2.position = Eigen::Vector3d(2, 3, 4);
    database->WritePosePrior(pose_prior2);

    PosePrior pose_prior3;
    pose_prior3.corr_data_id = images[2].DataId();
    pose_prior3.position = Eigen::Vector3d(2, 4, 12);
    database->WritePosePrior(pose_prior3);

    SpatialPairingOptions options;
    options.max_num_neighbors = 1;
    options.max_distance = 1000;
    options.ignore_z = false;

    auto cache = std::make_shared<FeatureMatcherCache>(
        options.CacheSize(), THROW_CHECK_NOTNULL(database));
    SpatialPairGenerator generator(options, cache);

    Eigen::RowMajorMatrixXf position_matrix =
        generator.ReadPositionPriorData(*cache);
    EXPECT_EQ(position_matrix.rows(), 2);
  }

  {
    constexpr int kNumImages = 3;
    auto database = Database::Open(kInMemorySqliteDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);

    PosePrior pose_prior1;
    pose_prior1.corr_data_id = images[0].DataId();
    pose_prior1.position =
        Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
    database->WritePosePrior(pose_prior1);

    PosePrior pose_prior2;
    pose_prior2.corr_data_id = images[1].DataId();
    pose_prior2.position = Eigen::Vector3d(2, 3, 4);
    database->WritePosePrior(pose_prior2);

    PosePrior pose_prior3;
    pose_prior3.corr_data_id = images[2].DataId();
    pose_prior3.position = Eigen::Vector3d(2, 4, 12);
    database->WritePosePrior(pose_prior3);

    SpatialPairingOptions options;
    options.max_num_neighbors = 1;
    options.max_distance = 1000;
    options.ignore_z = true;

    auto cache = std::make_shared<FeatureMatcherCache>(
        options.CacheSize(), THROW_CHECK_NOTNULL(database));
    SpatialPairGenerator generator(options, cache);

    Eigen::RowMajorMatrixXf position_matrix =
        generator.ReadPositionPriorData(*cache);
    EXPECT_EQ(position_matrix.rows(), 2);
  }
}

TEST(SpatialPairGenerator, NoPositionPriors) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  database->ClearPosePriors();

  SpatialPairingOptions options;
  options.max_num_neighbors = 2;
  options.max_distance = 1000;

  SpatialPairGenerator generator(options, database);
  EXPECT_TRUE(generator.HasFinished());
  EXPECT_TRUE(generator.Next().empty());
}

TEST(SpatialPairGenerator, Reset) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = images[0].DataId();
  pose_prior1.position = Eigen::Vector3d(1, 2, 3);
  database->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = images[1].DataId();
  pose_prior2.position = Eigen::Vector3d(2, 3, 4);
  database->WritePosePrior(pose_prior2);

  PosePrior pose_prior3;
  pose_prior3.corr_data_id = images[2].DataId();
  pose_prior3.position = Eigen::Vector3d(4, 5, 6);
  database->WritePosePrior(pose_prior3);

  SpatialPairingOptions options;
  options.max_num_neighbors = 1;
  options.max_distance = 1000;

  SpatialPairGenerator generator(options, database);
  const auto first_run = generator.AllPairs();
  EXPECT_TRUE(generator.HasFinished());

  generator.Reset();
  EXPECT_FALSE(generator.HasFinished());
  const auto second_run = generator.AllPairs();
  EXPECT_EQ(first_run.size(), second_run.size());
}

TEST(SpatialPairGenerator, WGS84Coordinates) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  // Use GPS coordinates (lat, lon, alt) with WGS84 coordinate system.
  // These are converted to ECEF internally.
  PosePrior pose_prior1;
  pose_prior1.corr_data_id = images[0].DataId();
  pose_prior1.position = Eigen::Vector3d(47.3769, 8.5417, 408);
  pose_prior1.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  database->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = images[1].DataId();
  pose_prior2.position = Eigen::Vector3d(47.3770, 8.5418, 409);
  pose_prior2.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  database->WritePosePrior(pose_prior2);

  PosePrior pose_prior3;
  pose_prior3.corr_data_id = images[2].DataId();
  pose_prior3.position = Eigen::Vector3d(48.0, 9.0, 500);
  pose_prior3.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  database->WritePosePrior(pose_prior3);

  SpatialPairingOptions options;
  options.max_num_neighbors = 1;
  options.max_distance = 100000;
  options.ignore_z = false;

  SpatialPairGenerator generator(options, database);

  // The first two images are very close together (a few meters apart),
  // so they should be each other's nearest neighbor. The third image is
  // far away (~70km).
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[0].ImageId(), images[1].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[1].ImageId(), images[0].ImageId())));
  EXPECT_FALSE(generator.HasFinished());
  generator.Next();  // consume the last one
  EXPECT_TRUE(generator.HasFinished());
}

TEST(SpatialPairGenerator, WGS84IgnoreZ) {
  constexpr int kNumImages = 2;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  PosePrior pose_prior1;
  pose_prior1.corr_data_id = images[0].DataId();
  pose_prior1.position = Eigen::Vector3d(47.3769, 8.5417, 408);
  pose_prior1.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  database->WritePosePrior(pose_prior1);

  PosePrior pose_prior2;
  pose_prior2.corr_data_id = images[1].DataId();
  pose_prior2.position = Eigen::Vector3d(47.3770, 8.5418, 10000);
  pose_prior2.coordinate_system = PosePrior::CoordinateSystem::WGS84;
  database->WritePosePrior(pose_prior2);

  SpatialPairingOptions options;
  options.max_num_neighbors = 1;
  options.max_distance = 100000;
  options.ignore_z = true;

  SpatialPairGenerator generator(options, database);
  // Both images should be paired because z is ignored.
  const auto all_pairs = generator.AllPairs();
  EXPECT_EQ(all_pairs.size(), 2);
}

TEST(TransitivePairGenerator, Nominal) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches.resize(10);

  database->ClearTwoViewGeometries();
  database->WriteTwoViewGeometry(
      images[0].ImageId(), images[1].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[0].ImageId(), images[2].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[1].ImageId(), images[3].ImageId(), two_view_geometry);

  TransitivePairingOptions options;
  TransitivePairGenerator generator(options, database);
  const auto pairs1 = generator.Next();
  EXPECT_THAT(pairs1,
              testing::UnorderedElementsAre(
                  std::make_pair(images[1].ImageId(), images[2].ImageId()),
                  std::make_pair(images[0].ImageId(), images[3].ImageId())));
  for (const auto& pair : pairs1) {
    database->WriteTwoViewGeometry(pair.first, pair.second, two_view_geometry);
  }
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[2].ImageId(), images[3].ImageId())));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(TransitivePairGenerator, Reset) {
  constexpr int kNumImages = 4;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches.resize(5);

  database->ClearTwoViewGeometries();
  database->WriteTwoViewGeometry(
      images[0].ImageId(), images[1].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[1].ImageId(), images[2].ImageId(), two_view_geometry);

  TransitivePairingOptions options;
  options.num_iterations = 1;
  TransitivePairGenerator generator(options, database);

  const auto first_run = generator.AllPairs();
  EXPECT_TRUE(generator.HasFinished());
  EXPECT_FALSE(first_run.empty());

  generator.Reset();
  EXPECT_FALSE(generator.HasFinished());
}

TEST(TransitivePairGenerator, BatchSplitting) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  TwoViewGeometry two_view_geometry;
  two_view_geometry.inlier_matches.resize(5);

  // Create a chain: 0-1, 1-2, 2-3, 3-4
  database->ClearTwoViewGeometries();
  database->WriteTwoViewGeometry(
      images[0].ImageId(), images[1].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[1].ImageId(), images[2].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[2].ImageId(), images[3].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[3].ImageId(), images[4].ImageId(), two_view_geometry);

  TransitivePairingOptions options;
  options.num_iterations = 1;
  // Use a small batch size to force batch splitting.
  options.batch_size = 2;
  TransitivePairGenerator generator(options, database);

  // Transitive closure on a chain 0-1-2-3-4 produces pairs like
  // (0,2), (1,3), (2,4), (0,3), (1,4), (0,4) depending on iteration.
  // With batch_size=2, these should be split across multiple Next() calls.
  std::vector<std::pair<image_t, image_t>> all_pairs;
  while (!generator.HasFinished()) {
    const auto batch = generator.Next();
    if (!batch.empty()) {
      EXPECT_LE(batch.size(), static_cast<size_t>(options.batch_size));
      all_pairs.insert(all_pairs.end(), batch.begin(), batch.end());
    }
  }
  // The chain 0-1-2-3-4 has transitive pairs: (0,2), (1,3), (2,4).
  EXPECT_GE(all_pairs.size(), 3);
}

TEST(TransitivePairGenerator, NoExistingPairs) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  database->ClearTwoViewGeometries();

  TransitivePairingOptions options;
  options.num_iterations = 1;
  TransitivePairGenerator generator(options, database);

  // With no existing pairs, the transitive closure produces nothing.
  const auto pairs = generator.AllPairs();
  EXPECT_TRUE(pairs.empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(ImportedPairGenerator, Nominal) {
  constexpr int kNumImages = 10;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  ImportedPairingOptions options;
  options.match_list_path = CreateTestDir() / "pairs.txt";

  {
    std::ofstream match_list_file(options.match_list_path);
    match_list_file.close();

    ImportedPairGenerator generator(options, database);
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }

  {
    std::ofstream match_list_file(options.match_list_path);
    match_list_file << images[2].Name() << " " << images[4].Name() << '\n';
    match_list_file << images[1].Name() << " " << images[3].Name() << '\n';
    match_list_file << images[2].Name() << " " << images[9].Name() << '\n';
    match_list_file.close();

    ImportedPairGenerator generator(options, database);
    EXPECT_THAT(generator.Next(),
                testing::ElementsAre(
                    std::make_pair(images[2].ImageId(), images[4].ImageId()),
                    std::make_pair(images[1].ImageId(), images[3].ImageId()),
                    std::make_pair(images[2].ImageId(), images[9].ImageId())));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.HasFinished());
  }
}

TEST(ImportedPairGenerator, CommentsAndBlankLines) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  const auto test_dir = CreateTestDir();
  ImportedPairingOptions options;
  options.match_list_path = test_dir / "pairs_comments.txt";

  {
    std::ofstream file(options.match_list_path);
    file << "# This is a comment\n";
    file << "\n";
    file << images[0].Name() << " " << images[1].Name() << "\n";
    file << "# Another comment\n";
    file << "  \n";
    file << images[2].Name() << " " << images[3].Name() << "\n";
    file.close();
  }

  ImportedPairGenerator generator(options, database);
  EXPECT_THAT(generator.Next(),
              testing::ElementsAre(
                  std::make_pair(images[0].ImageId(), images[1].ImageId()),
                  std::make_pair(images[2].ImageId(), images[3].ImageId())));
  EXPECT_TRUE(generator.HasFinished());
}

TEST(ImportedPairGenerator, DuplicatePairsDeduplication) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  const auto test_dir = CreateTestDir();
  ImportedPairingOptions options;
  options.match_list_path = test_dir / "pairs_dup.txt";

  {
    std::ofstream file(options.match_list_path);
    file << images[0].Name() << " " << images[1].Name() << "\n";
    file << images[0].Name() << " " << images[1].Name() << "\n";
    file << images[2].Name() << " " << images[3].Name() << "\n";
    file.close();
  }

  ImportedPairGenerator generator(options, database);
  const auto pairs = generator.AllPairs();
  // The duplicate pair should be deduplicated.
  EXPECT_EQ(pairs.size(), 2);
}

TEST(ImportedPairGenerator, UnknownImageSkipped) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  const auto test_dir = CreateTestDir();
  ImportedPairingOptions options;
  options.match_list_path = test_dir / "pairs_unknown.txt";

  {
    std::ofstream file(options.match_list_path);
    file << "nonexistent_image.jpg " << images[1].Name() << "\n";
    file << images[0].Name() << " nonexistent_image2.jpg\n";
    file << images[2].Name() << " " << images[3].Name() << "\n";
    file.close();
  }

  ImportedPairGenerator generator(options, database);
  const auto pairs = generator.AllPairs();
  // Only the valid pair should remain.
  EXPECT_EQ(pairs.size(), 1);
  EXPECT_EQ(pairs[0], std::make_pair(images[2].ImageId(), images[3].ImageId()));
}

TEST(ImportedPairGenerator, MultipleBlocks) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  const auto test_dir = CreateTestDir();
  ImportedPairingOptions options;
  options.match_list_path = test_dir / "pairs_blocks.txt";
  options.block_size = 2;

  {
    std::ofstream file(options.match_list_path);
    file << images[0].Name() << " " << images[1].Name() << "\n";
    file << images[1].Name() << " " << images[2].Name() << "\n";
    file << images[2].Name() << " " << images[3].Name() << "\n";
    file << images[3].Name() << " " << images[4].Name() << "\n";
    file << images[0].Name() << " " << images[4].Name() << "\n";
    file.close();
  }

  ImportedPairGenerator generator(options, database);

  // First block: 2 pairs.
  auto block1 = generator.Next();
  EXPECT_EQ(block1.size(), 2);
  EXPECT_FALSE(generator.HasFinished());

  // Second block: 2 pairs.
  auto block2 = generator.Next();
  EXPECT_EQ(block2.size(), 2);
  EXPECT_FALSE(generator.HasFinished());

  // Third block: 1 remaining pair.
  auto block3 = generator.Next();
  EXPECT_EQ(block3.size(), 1);

  // Fourth call: empty, finished.
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(ImportedPairGenerator, Reset) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  const auto test_dir = CreateTestDir();
  ImportedPairingOptions options;
  options.match_list_path = test_dir / "pairs_reset.txt";

  {
    std::ofstream file(options.match_list_path);
    file << images[0].Name() << " " << images[1].Name() << "\n";
    file << images[2].Name() << " " << images[3].Name() << "\n";
    file.close();
  }

  ImportedPairGenerator generator(options, database);
  const auto first_run = generator.AllPairs();
  EXPECT_TRUE(generator.HasFinished());

  generator.Reset();
  EXPECT_FALSE(generator.HasFinished());
  const auto second_run = generator.AllPairs();
  EXPECT_EQ(first_run.size(), second_run.size());
}

TEST(ExistingMatchedPairGenerator, Nominal) {
  constexpr int kNumImages = 5;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  database->ClearMatches();
  database->WriteMatches(
      images[0].ImageId(), images[1].ImageId(), FeatureMatches(1));
  database->WriteMatches(
      images[0].ImageId(), images[2].ImageId(), FeatureMatches(2));
  database->WriteMatches(
      images[1].ImageId(), images[3].ImageId(), FeatureMatches(3));
  database->WriteMatches(
      images[2].ImageId(), images[3].ImageId(), FeatureMatches(0));

  ExistingMatchedPairingOptions options;
  options.batch_size = 2;
  ExistingMatchedPairGenerator generator(options, database);
  EXPECT_THAT(generator.Next(),
              testing::UnorderedElementsAre(
                  std::make_pair(images[0].ImageId(), images[1].ImageId()),
                  std::make_pair(images[0].ImageId(), images[2].ImageId())));
  EXPECT_THAT(generator.Next(),
              testing::UnorderedElementsAre(
                  std::make_pair(images[1].ImageId(), images[3].ImageId())));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.HasFinished());
}

TEST(ExistingMatchedPairGenerator, NoMatches) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  database->ClearMatches();

  ExistingMatchedPairingOptions options;
  ExistingMatchedPairGenerator generator(options, database);
  EXPECT_TRUE(generator.HasFinished());
  EXPECT_TRUE(generator.Next().empty());
}

TEST(ExistingMatchedPairGenerator, Reset) {
  constexpr int kNumImages = 3;
  auto database = Database::Open(kInMemorySqliteDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();

  database->ClearMatches();
  database->WriteMatches(
      images[0].ImageId(), images[1].ImageId(), FeatureMatches(1));
  database->WriteMatches(
      images[1].ImageId(), images[2].ImageId(), FeatureMatches(2));

  ExistingMatchedPairingOptions options;
  options.batch_size = 2;
  ExistingMatchedPairGenerator generator(options, database);

  const auto first_run = generator.AllPairs();
  EXPECT_EQ(first_run.size(), 2);
  EXPECT_TRUE(generator.HasFinished());

  generator.Reset();
  EXPECT_FALSE(generator.HasFinished());
  const auto second_run = generator.AllPairs();
  EXPECT_EQ(first_run.size(), second_run.size());
}

TEST(OptionsCheck, ExhaustivePairingOptions) {
  ExhaustivePairingOptions options;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(options.CacheSize(), 2 * options.block_size);
}

TEST(OptionsCheck, VocabTreePairingOptions) {
  VocabTreePairingOptions options;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(options.CacheSize(), 5 * options.num_images);
}

TEST(OptionsCheck, SequentialPairingOptions) {
  SequentialPairingOptions options;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(
      options.CacheSize(),
      std::max(5 * options.loop_detection_num_images, 5 * options.overlap));
}

TEST(OptionsCheck, SpatialPairingOptions) {
  {
    SpatialPairingOptions options;
    EXPECT_TRUE(options.Check());
    EXPECT_EQ(options.CacheSize(), 5 * options.max_num_neighbors);
  }
  {
    // max_distance=0 and min_num_neighbors=0 is invalid.
    SpatialPairingOptions options;
    options.max_distance = 0;
    options.min_num_neighbors = 0;
    EXPECT_FALSE(options.Check());
  }
  {
    // min_num_neighbors > max_num_neighbors is invalid.
    SpatialPairingOptions options;
    options.min_num_neighbors = 100;
    options.max_num_neighbors = 10;
    EXPECT_FALSE(options.Check());
  }
}

TEST(OptionsCheck, TransitivePairingOptions) {
  TransitivePairingOptions options;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(options.CacheSize(), 2 * options.batch_size);
}

TEST(OptionsCheck, ImportedPairingOptions) {
  ImportedPairingOptions options;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(options.CacheSize(), static_cast<size_t>(options.block_size));
}

TEST(OptionsCheck, FeaturePairsMatchingOptions) {
  FeaturePairsMatchingOptions options;
  EXPECT_TRUE(options.Check());
}

TEST(OptionsCheck, ExistingMatchedPairingOptions) {
  ExistingMatchedPairingOptions options;
  EXPECT_TRUE(options.Check());
  EXPECT_EQ(options.CacheSize(),
            std::max<size_t>(
                10, static_cast<size_t>(2 * std::sqrt(options.batch_size))));
}

TEST(SequentialPairingOptions, VocabTreeOptionsConversion) {
  SequentialPairingOptions seq_options;
  seq_options.loop_detection_num_images = 42;
  seq_options.loop_detection_num_nearest_neighbors = 7;
  seq_options.loop_detection_num_checks = 128;
  seq_options.loop_detection_num_images_after_verification = 10;
  seq_options.loop_detection_max_num_features = 500;
  seq_options.vocab_tree_path = "/path/to/vocab.bin";
  seq_options.num_threads = 4;

  VocabTreePairingOptions vt_options = seq_options.VocabTreeOptions();
  EXPECT_EQ(vt_options.num_images, 42);
  EXPECT_EQ(vt_options.num_nearest_neighbors, 7);
  EXPECT_EQ(vt_options.num_checks, 128);
  EXPECT_EQ(vt_options.num_images_after_verification, 10);
  EXPECT_EQ(vt_options.max_num_features, 500);
  EXPECT_EQ(vt_options.vocab_tree_path, "/path/to/vocab.bin");
  EXPECT_EQ(vt_options.num_threads, 4);
}

}  // namespace
}  // namespace colmap
