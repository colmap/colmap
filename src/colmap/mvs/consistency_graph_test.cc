// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"

#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(ConsistencyGraph, Empty) {
  const std::vector<int> data;
  ConsistencyGraph consistency_graph(2, 2, data);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      int num_images;
      const int* image_idxs;
      consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
      EXPECT_EQ(num_images, 0);
      EXPECT_TRUE(image_idxs == nullptr);
    }
  }
  EXPECT_EQ(consistency_graph.GetNumBytes(), 16);
}

TEST(ConsistencyGraph, Partial) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 3);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 7);
  EXPECT_EQ(image_idxs[2], 33);
  consistency_graph.GetImageIdxs(0, 1, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
  EXPECT_TRUE(image_idxs == nullptr);
  EXPECT_EQ(consistency_graph.GetNumBytes(), 32);
}

TEST(ConsistencyGraph, Zero) {
  const std::vector<int> data = {0, 0, 0};
  ConsistencyGraph consistency_graph(2, 1, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
  EXPECT_TRUE(image_idxs == nullptr);
  consistency_graph.GetImageIdxs(0, 1, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
  EXPECT_TRUE(image_idxs == nullptr);
  EXPECT_EQ(consistency_graph.GetNumBytes(), 20);
}

TEST(ConsistencyGraph, Full) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33, 0, 1, 1, 100};
  ConsistencyGraph consistency_graph(1, 2, data);
  int num_images;
  const int* image_idxs;
  consistency_graph.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 3);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 7);
  EXPECT_EQ(image_idxs[2], 33);
  consistency_graph.GetImageIdxs(1, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 1);
  EXPECT_EQ(image_idxs[0], 100);
  EXPECT_EQ(consistency_graph.GetNumBytes(), 48);
}

TEST(ConsistencyGraph, DefaultConstructor) {
  ConsistencyGraph consistency_graph;
  EXPECT_EQ(consistency_graph.GetNumBytes(), 0);
}

TEST(ConsistencyGraph, WriteReadRoundtrip) {
  const std::vector<int> data = {0, 0, 3, 5, 7, 33, 0, 1, 1, 100};
  ConsistencyGraph original(1, 2, data);

  const auto test_dir = CreateTestDir();
  const auto path = test_dir / "consistency_graph.bin";
  original.Write(path);

  ConsistencyGraph loaded;
  loaded.Read(path);

  EXPECT_EQ(loaded.GetNumBytes(), original.GetNumBytes());

  // Verify data is preserved
  int num_images;
  const int* image_idxs;
  loaded.GetImageIdxs(0, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 3);
  EXPECT_EQ(image_idxs[0], 5);
  EXPECT_EQ(image_idxs[1], 7);
  EXPECT_EQ(image_idxs[2], 33);
  loaded.GetImageIdxs(1, 0, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 1);
  EXPECT_EQ(image_idxs[0], 100);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
