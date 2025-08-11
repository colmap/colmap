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

#include "colmap/feature/pairing.h"

#include "colmap/retrieval/visual_index.h"
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
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
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

std::unique_ptr<retrieval::VisualIndex> CreateSyntheticVisualIndex() {
  auto visual_index = retrieval::VisualIndex::Create();
  retrieval::VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  visual_index->Build(build_options,
                      retrieval::VisualIndex::Descriptors::Random(50, 128));
  return visual_index;
}

TEST(VocabTreePairGenerator, Nominal) {
  constexpr int kNumImages = 5;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  VocabTreePairingOptions options;
  options.vocab_tree_path = CreateTestDir() + "/vocab_tree.txt";

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

TEST(SequentialPairGenerator, Linear) {
  constexpr int kNumImages = 5;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
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
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
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
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
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

TEST(SpatialPairGenerator, Nominal) {
  constexpr int kNumImages = 3;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);
  database->WritePosePrior(images[0].ImageId(),
                           PosePrior(Eigen::Vector3d(1, 2, 3)));
  database->WritePosePrior(images[1].ImageId(),
                           PosePrior(Eigen::Vector3d(2, 3, 4)));
  database->WritePosePrior(images[2].ImageId(),
                           PosePrior(Eigen::Vector3d(2, 4, 12)));

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
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);
  database->WritePosePrior(
      images[0].ImageId(),
      PosePrior(Eigen::Vector3d(1, 2, 3) + Eigen::Vector3d::Constant(1e16)));
  database->WritePosePrior(
      images[1].ImageId(),
      PosePrior(Eigen::Vector3d(2, 3, 4) + Eigen::Vector3d::Constant(1e16)));
  database->WritePosePrior(
      images[2].ImageId(),
      PosePrior(Eigen::Vector3d(2, 4, 12) + Eigen::Vector3d::Constant(1e16)));

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
  auto db = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *db);
  const auto images = db->ReadAllImages();

  db->WritePosePrior(images[0].ImageId(), PosePrior(Eigen::Vector3d(1, 1, 2)));
  db->WritePosePrior(images[1].ImageId(), PosePrior(Eigen::Vector3d(1, 2, 3)));
  db->WritePosePrior(images[2].ImageId(), PosePrior(Eigen::Vector3d(2, 3, 4)));
  db->WritePosePrior(images[3].ImageId(), PosePrior(Eigen::Vector3d(2, 4, 12)));

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
    SpatialPairGenerator generator(options, db);
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
    SpatialPairGenerator generator(options, db);
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
    SpatialPairGenerator generator(options, db);
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
    auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);
    database->WritePosePrior(images[0].ImageId(),
                             PosePrior(Eigen::Vector3d(1, 2, 3)));
    database->WritePosePrior(images[1].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 3, 4)));
    database->WritePosePrior(images[2].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 4, 12)));

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
    constexpr int kNumImages = 4;
    auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);
    database->WritePosePrior(images[0].ImageId(),
                             PosePrior(Eigen::Vector3d(1, 2, 3)));
    database->WritePosePrior(images[1].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 3, 4)));
    database->WritePosePrior(images[2].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 4, 12)));

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
    auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);
    database->WritePosePrior(images[0].ImageId(),
                             PosePrior(Eigen::Vector3d(0, 0, 2)));
    database->WritePosePrior(images[1].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 3, 4)));
    database->WritePosePrior(images[2].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 4, 12)));

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
    auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);
    database->WritePosePrior(images[0].ImageId(),
                             PosePrior(Eigen::Vector3d(0, 0, 2)));
    database->WritePosePrior(images[1].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 3, 4)));
    database->WritePosePrior(images[2].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 4, 12)));

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

  {
    constexpr int kNumImages = 3;
    auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);
    database->WritePosePrior(images[0].ImageId(),
                             PosePrior(Eigen::Vector3d(0, 0, 0)));
    database->WritePosePrior(images[1].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 3, 4)));
    database->WritePosePrior(images[2].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 4, 12)));

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
    auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
    CreateSyntheticDatabase(kNumImages, *database);
    const std::vector<Image> images = database->ReadAllImages();
    CHECK_EQ(images.size(), kNumImages);
    database->WritePosePrior(images[0].ImageId(),
                             PosePrior(Eigen::Vector3d(0, 0, 0)));
    database->WritePosePrior(images[1].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 3, 4)));
    database->WritePosePrior(images[2].ImageId(),
                             PosePrior(Eigen::Vector3d(2, 4, 12)));

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

TEST(TransitivePairGenerator, Nominal) {
  constexpr int kNumImages = 5;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
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

TEST(ImportedPairGenerator, Nominal) {
  constexpr int kNumImages = 10;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  ImportedPairingOptions options;
  options.match_list_path = CreateTestDir() + "/pairs.txt";

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

}  // namespace
}  // namespace colmap
