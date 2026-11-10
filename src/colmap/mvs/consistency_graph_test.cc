// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"

#include "colmap/util/testing.h"

#include <fstream>
#include <iterator>
#include <limits>
#include <string>

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

TEST(ConsistencyGraph, RejectsTruncatedRecordPayload) {
  for (const std::vector<int>& payload : {std::vector<int>{0, 0, 1},
                                          std::vector<int>{0, 0, 2, 7},
                                          std::vector<int>{0, 0, 2147483647}}) {
    EXPECT_THROW(ConsistencyGraph(3, 2, payload), std::exception);
  }
}

TEST(ConsistencyGraph, RejectsUnrepresentableDimensions) {
  const size_t too_large =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  const std::vector<int> data;
  EXPECT_THROW(ConsistencyGraph(too_large, 1, data), std::exception);
  EXPECT_THROW(ConsistencyGraph(1, too_large, data), std::exception);
}

TEST(ConsistencyGraph, RejectsPartialIntegerTail) {
  const auto path = CreateTestDir() / "partial.bin";
  for (int tail = 1; tail < 4; ++tail) {
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << "3&2&1&";
    file.write("\0\0\0", tail);
    file.close();
    ConsistencyGraph graph;
    EXPECT_THROW(graph.Read(path), std::exception);
  }
}

TEST(ConsistencyGraph, IndependentWireOrderAndWhitespacePayload) {
  const auto path = CreateTestDir() / "wire.bin";
  const std::string expected =
      std::string("33&2&1&") +
      std::string("\x20\0\0\0\x01\0\0\0\x02\0\0\0\x0a\0\0\0\x20\0\0\0", 20);
  {
    std::ofstream fixture(path, std::ios::binary);
    fixture.write(expected.data(), expected.size());
  }
  std::ifstream file(path, std::ios::binary);
  const std::string actual((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  EXPECT_EQ(actual, expected);
  file.close();
  ConsistencyGraph loaded;
  loaded.Read(path);
  int count;
  const int* ids;
  loaded.GetImageIdxs(1, 32, &count, &ids);
  ASSERT_EQ(count, 2);
  EXPECT_EQ(ids[0], 10);
  EXPECT_EQ(ids[1], 32);
  ConsistencyGraph graph(33, 2, std::vector<int>{32, 1, 2, 10, 32});
  graph.Write(path);
  std::ifstream written(path, std::ios::binary);
  const std::string actual_written((std::istreambuf_iterator<char>(written)),
                                   std::istreambuf_iterator<char>());
  EXPECT_EQ(actual_written, expected);
}

TEST(ConsistencyGraph, ReadsEmptyPayload) {
  const auto path = CreateTestDir() / "empty.bin";
  {
    std::ofstream file(path, std::ios::binary);
    file << "3&2&1&";
  }
  ConsistencyGraph graph;
  graph.Read(path);
  int count = -1;
  const int* ids = nullptr;
  graph.GetImageIdxs(1, 2, &count, &ids);
  EXPECT_EQ(count, 0);
  EXPECT_EQ(ids, nullptr);
}

TEST(ConsistencyGraph, PreservesGraphAfterFailedRead) {
  const auto path = CreateTestDir() / "failed-read.bin";
  ConsistencyGraph graph(3, 2, std::vector<int>{2, 1, 2, 7, 8});
  const std::vector<std::string> bad_inputs = {
      std::string("2147483648&2&1&") +
          std::string("\0\0\0\0\0\0\0\0\x64\0\0\0\x01\0\0\0", 16),
      std::string("3&2&1&") + std::string("\0\0\0\0\0\0\0\0\x02\0\0\0", 12),
      std::string("3&2&1&") + std::string("\0", 1),
      "3&2&-1&"};
  for (const auto& input : bad_inputs) {
    {
      std::ofstream file(path, std::ios::binary);
      file.write(input.data(), input.size());
    }
    EXPECT_THROW(graph.Read(path), std::exception);
    int count = -1;
    const int* ids = nullptr;
    graph.GetImageIdxs(1, 2, &count, &ids);
    ASSERT_EQ(count, 2);
    ASSERT_NE(ids, nullptr);
    EXPECT_EQ(ids[0], 7);
    EXPECT_EQ(ids[1], 8);
  }
  {
    std::ofstream file(path, std::ios::binary);
    file << "3&2&1&";
  }
  graph.Read(path);
  int count = -1;
  const int* ids = nullptr;
  graph.GetImageIdxs(1, 2, &count, &ids);
  EXPECT_EQ(count, 0);
  EXPECT_EQ(ids, nullptr);
}

TEST(ConsistencyGraph, RejectsMalformedHeader) {
  const auto path = CreateTestDir() / "header.bin";
  for (const char* header :
       {"3:2&1&", "3&2&", "0&2&1&", "-3&2&1&", "3&-2&1&", "3&2&-1&"}) {
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << header;
    file.close();
    ConsistencyGraph graph;
    EXPECT_THROW(graph.Read(path), std::exception);
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
