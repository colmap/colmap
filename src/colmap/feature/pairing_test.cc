// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateSyntheticDatabase(int num_images, Database& database) {
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_cameras = num_images;
  synthetic_dataset_options.num_images = num_images;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, &database);
}

TEST(ExhaustivePairGenerator, Nominal) {
  constexpr int kNumImages = 34;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  ExhaustiveMatchingOptions options;
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
  EXPECT_TRUE(generator.Next().empty());
}

retrieval::VisualIndex<> CreateSyntheticVisualIndex() {
  retrieval::VisualIndex<> visual_index;
  retrieval::VisualIndex<>::BuildOptions build_options;
  build_options.num_visual_words = 5;
  build_options.branching = 5;
  visual_index.Build(build_options,
                     retrieval::VisualIndex<>::DescType::Random(50, 128));
  return visual_index;
}

TEST(VocabTreePairGenerator, Nominal) {
  constexpr int kNumImages = 5;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  VocabTreeMatchingOptions options;
  options.vocab_tree_path = CreateTestDir() + "/vocab_tree.txt";
  CreateSyntheticVisualIndex().Write(options.vocab_tree_path);

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
    EXPECT_TRUE(generator.Next().empty());
  }

  {
    options.num_images = 100;
    VocabTreePairGenerator generator(options, database);
    for (int i = 0; i < kNumImages; ++i) {
      const auto pairs = generator.Next();
      EXPECT_EQ(pairs.size(), kNumImages);
    }
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.Next().empty());
  }
}

TEST(SequentialPairGenerator, Linear) {
  constexpr int kNumImages = 5;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  SequentialMatchingOptions options;
  options.overlap = 3;
  options.quadratic_overlap = false;
  SequentialPairGenerator generator(options, database);
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[0].ImageId(), images[1].ImageId()},
                {images[0].ImageId(), images[2].ImageId()},
                {images[0].ImageId(), images[3].ImageId()}}));
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[1].ImageId(), images[2].ImageId()},
                {images[1].ImageId(), images[3].ImageId()},
                {images[1].ImageId(), images[4].ImageId()}}));
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[2].ImageId(), images[3].ImageId()},
                {images[2].ImageId(), images[4].ImageId()}}));
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[3].ImageId(), images[4].ImageId()}}));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.Next().empty());
}

TEST(SequentialPairGenerator, Quadratic) {
  constexpr int kNumImages = 5;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  SequentialMatchingOptions options;
  options.overlap = 3;
  options.quadratic_overlap = true;
  SequentialPairGenerator generator(options, database);
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[0].ImageId(), images[1].ImageId()},
                {images[0].ImageId(), images[2].ImageId()},
                {images[0].ImageId(), images[4].ImageId()}}));
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[1].ImageId(), images[2].ImageId()},
                {images[1].ImageId(), images[3].ImageId()}}));
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[2].ImageId(), images[3].ImageId()},
                {images[2].ImageId(), images[4].ImageId()}}));
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[3].ImageId(), images[4].ImageId()}}));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.Next().empty());
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

  SpatialMatchingOptions options;
  options.max_num_neighbors = 1;
  options.max_distance = 1000;
  options.ignore_z = false;
  {
    SpatialPairGenerator generator(options, database);
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[2].ImageId(), images[1].ImageId()}}));
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[1].ImageId(), images[0].ImageId()}}));
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[0].ImageId(), images[1].ImageId()}}));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.Next().empty());
  }

  {
    options.ignore_z = true;
    SpatialPairGenerator generator(options, database);
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[2].ImageId(), images[1].ImageId()}}));
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[1].ImageId(), images[2].ImageId()}}));
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[0].ImageId(), images[1].ImageId()}}));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.Next().empty());
  }

  {
    options.ignore_z = false;
    options.max_distance = 5;
    SpatialPairGenerator generator(options, database);
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[1].ImageId(), images[0].ImageId()}}));
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[0].ImageId(), images[1].ImageId()}}));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.Next().empty());
  }

  {
    options.max_num_neighbors = 2;
    options.max_distance = 1000;
    SpatialPairGenerator generator(options, database);
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[2].ImageId(), images[1].ImageId()},
                  {images[2].ImageId(), images[0].ImageId()}}));
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[1].ImageId(), images[0].ImageId()},
                  {images[1].ImageId(), images[2].ImageId()}}));
    EXPECT_EQ(generator.Next(),
              (std::vector<std::pair<image_t, image_t>>{
                  {images[0].ImageId(), images[1].ImageId()},
                  {images[0].ImageId(), images[2].ImageId()}}));
    EXPECT_TRUE(generator.Next().empty());
    EXPECT_TRUE(generator.Next().empty());
  }
}

TEST(ImportedPairGenerator, Nominal) {
  constexpr int kNumImages = 10;
  auto database = std::make_shared<Database>(Database::kInMemoryDatabasePath);
  CreateSyntheticDatabase(kNumImages, *database);
  const std::vector<Image> images = database->ReadAllImages();
  CHECK_EQ(images.size(), kNumImages);

  const std::string match_list_path = CreateTestDir() + "/pairs.txt";
  std::ofstream match_list_file(match_list_path);
  match_list_file << images[2].Name() << " " << images[4].Name() << "\n";
  match_list_file << images[1].Name() << " " << images[3].Name() << "\n";
  match_list_file << images[2].Name() << " " << images[9].Name() << "\n";
  match_list_file.close();

  ImagePairsMatchingOptions options;
  options.match_list_path = match_list_path;
  ImportedPairGenerator generator(options, database);
  EXPECT_EQ(generator.Next(),
            (std::vector<std::pair<image_t, image_t>>{
                {images[2].ImageId(), images[4].ImageId()},
                {images[1].ImageId(), images[3].ImageId()},
                {images[2].ImageId(), images[9].ImageId()}}));
  EXPECT_TRUE(generator.Next().empty());
  EXPECT_TRUE(generator.Next().empty());
}

}  // namespace
}  // namespace colmap
