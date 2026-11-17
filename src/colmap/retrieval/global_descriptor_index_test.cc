// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/global_descriptor_index.h"

#include "colmap/util/testing.h"

#include <cmath>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

constexpr int kDescriptorDim = 4;

// Images at increasing angles in the plane of the first two axes, so that the
// cosine similarity ranking of the neighbors of every image is known and has
// no ties.
const std::vector<image_t> kImageIds = {10, 20, 30, 40};
const std::vector<double> kAngles = {0.0, 0.1, 0.3, 0.7};

std::vector<float> CreateDescriptor(const double angle) {
  std::vector<float> descriptor(kDescriptorDim, 0.0f);
  descriptor[0] = static_cast<float>(std::cos(angle));
  descriptor[1] = static_cast<float>(std::sin(angle));
  return descriptor;
}

GlobalDescriptorIndex CreateTestIndex() {
  GlobalDescriptorIndex index(kDescriptorDim);
  for (size_t i = 0; i < kImageIds.size(); ++i) {
    index.Add(kImageIds[i], CreateDescriptor(kAngles[i]));
  }
  index.Prepare();
  return index;
}

std::vector<image_t> QueryImageIds(
    const GlobalDescriptorIndex& index,
    const GlobalDescriptorIndex::QueryOptions& options,
    const image_t query_image_id) {
  std::vector<ImageScore> scores;
  index.Query(options, query_image_id, &scores);
  std::vector<image_t> image_ids;
  image_ids.reserve(scores.size());
  for (const ImageScore& score : scores) {
    image_ids.push_back(score.image_id);
  }
  return image_ids;
}

TEST(GlobalDescriptorIndex, Nominal) {
  GlobalDescriptorIndex index(kDescriptorDim);
  EXPECT_EQ(index.NumImages(), 0);
  EXPECT_FALSE(index.IsPrepared());
  for (size_t i = 0; i < kImageIds.size(); ++i) {
    index.Add(kImageIds[i], CreateDescriptor(kAngles[i]));
  }
  EXPECT_EQ(index.NumImages(), kImageIds.size());
  index.Prepare();
  EXPECT_TRUE(index.IsPrepared());

  GlobalDescriptorIndex::QueryOptions options;
  options.max_num_images = 2;
  std::vector<ImageScore> scores;
  index.Query(options, /*query_image_id=*/10, &scores);
  ASSERT_EQ(scores.size(), 2);
  // Neighbors in descending similarity, excluding the query image itself.
  EXPECT_EQ(scores[0].image_id, 20);
  EXPECT_EQ(scores[1].image_id, 30);
  EXPECT_NEAR(scores[0].score, std::cos(0.1), 1e-5);
  EXPECT_NEAR(scores[1].score, std::cos(0.3), 1e-5);

  EXPECT_EQ(QueryImageIds(index, options, 20), (std::vector<image_t>{10, 30}));
  EXPECT_EQ(QueryImageIds(index, options, 30), (std::vector<image_t>{20, 10}));
  EXPECT_EQ(QueryImageIds(index, options, 40), (std::vector<image_t>{30, 20}));

  // Requesting more images than indexed returns all other images.
  options.max_num_images = 100;
  EXPECT_EQ(QueryImageIds(index, options, 10),
            (std::vector<image_t>{20, 30, 40}));
}

TEST(GlobalDescriptorIndex, QueryWithFilter) {
  const GlobalDescriptorIndex index = CreateTestIndex();
  GlobalDescriptorIndex::QueryOptions options;
  options.max_num_images = 2;
  // Filtered images do not consume the requested budget.
  options.image_id_filter = [](const image_t image_id) {
    return image_id != 20;
  };
  EXPECT_EQ(QueryImageIds(index, options, 10), (std::vector<image_t>{30, 40}));
  options.image_id_filter = [](const image_t) { return false; };
  EXPECT_TRUE(QueryImageIds(index, options, 10).empty());
}

TEST(GlobalDescriptorIndex, DuplicateImageIsIgnored) {
  GlobalDescriptorIndex index(kDescriptorDim);
  index.Add(10, CreateDescriptor(0.0));
  index.Add(10, CreateDescriptor(0.5));
  EXPECT_EQ(index.NumImages(), 1);
}

TEST(GlobalDescriptorIndex, InvalidUsage) {
  GlobalDescriptorIndex index(kDescriptorDim);
  EXPECT_ANY_THROW(index.Add(10, std::vector<float>(kDescriptorDim + 1, 0.f)));
  EXPECT_ANY_THROW(index.Prepare());
  index.Add(10, CreateDescriptor(0.0));
  index.Add(20, CreateDescriptor(0.1));
  std::vector<ImageScore> scores;
  EXPECT_ANY_THROW(index.Query({}, 10, &scores));
  index.Prepare();
  EXPECT_ANY_THROW(index.Prepare());
  EXPECT_ANY_THROW(index.Add(30, CreateDescriptor(0.2)));
  EXPECT_ANY_THROW(index.Query({}, 30, &scores));
  GlobalDescriptorIndex::QueryOptions options;
  options.max_num_images = 0;
  EXPECT_ANY_THROW(index.Query(options, 10, &scores));
}

TEST(GlobalDescriptorIndex, WriteRead) {
  const std::filesystem::path path = CreateTestDir() / "index.bin";
  const GlobalDescriptorIndex index = CreateTestIndex();
  index.Write(path);

  // The descriptor dimension is restored from the file.
  GlobalDescriptorIndex read_index(/*descriptor_dim=*/1);
  read_index.Read(path);
  EXPECT_EQ(read_index.NumImages(), kImageIds.size());
  EXPECT_TRUE(read_index.IsPrepared());

  GlobalDescriptorIndex::QueryOptions options;
  options.max_num_images = 3;
  for (const image_t image_id : kImageIds) {
    std::vector<ImageScore> scores;
    index.Query(options, image_id, &scores);
    std::vector<ImageScore> read_scores;
    read_index.Query(options, image_id, &read_scores);
    ASSERT_EQ(scores.size(), read_scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
      EXPECT_EQ(scores[i].image_id, read_scores[i].image_id);
      EXPECT_EQ(scores[i].score, read_scores[i].score);
    }
  }
}

TEST(GlobalDescriptorIndex, ReadInvalidFile) {
  const std::filesystem::path test_dir = CreateTestDir();
  const std::filesystem::path path = test_dir / "index.bin";
  CreateTestIndex().Write(path);

  GlobalDescriptorIndex index(kDescriptorDim);
  EXPECT_ANY_THROW(index.Read(test_dir / "missing.bin"));

  const std::filesystem::path truncated_path = test_dir / "truncated.bin";
  std::filesystem::copy_file(path, truncated_path);
  std::filesystem::resize_file(truncated_path,
                               std::filesystem::file_size(path) - 1);
  EXPECT_ANY_THROW(index.Read(truncated_path));

  const std::filesystem::path header_path = test_dir / "header_only.bin";
  std::filesystem::copy_file(path, header_path);
  std::filesystem::resize_file(header_path, 8);
  EXPECT_ANY_THROW(index.Read(header_path));

  const std::filesystem::path invalid_path = test_dir / "invalid.bin";
  {
    std::ofstream file(invalid_path, std::ios::binary);
    const std::string content(64, 'x');
    file.write(content.data(), content.size());
  }
  EXPECT_ANY_THROW(index.Read(invalid_path));
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
