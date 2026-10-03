// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/inverted_file_entry.h"

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

TEST(InvertedFileEntry, Empty) {
  InvertedFileEntry<10> entry;
  EXPECT_EQ(entry.image_id, -1);
  EXPECT_EQ(entry.feature_idx, -1);
  EXPECT_EQ(entry.geometry.x, 0);
  EXPECT_EQ(entry.geometry.y, 0);
  EXPECT_EQ(entry.geometry.scale, 0);
  EXPECT_EQ(entry.geometry.orientation, 0);
  EXPECT_EQ(entry.descriptor.size(), 10);
}

TEST(InvertedFileEntry, ReadWrite) {
  InvertedFileEntry<10> entry;
  entry.image_id = 99;
  entry.feature_idx = 100;
  entry.geometry.x = 0.123;
  entry.geometry.y = 0.456;
  entry.geometry.scale = 0.789;
  entry.geometry.orientation = -0.1;
  for (size_t i = 0; i < entry.descriptor.size(); ++i) {
    entry.descriptor[i] = (i % 2) == 0;
  }
  std::stringstream file;
  entry.Write(&file);

  InvertedFileEntry<10> read_entry;
  read_entry.Read(&file);
  EXPECT_EQ(entry.image_id, read_entry.image_id);
  EXPECT_EQ(entry.feature_idx, read_entry.feature_idx);
  EXPECT_EQ(entry.geometry.x, read_entry.geometry.x);
  EXPECT_EQ(entry.geometry.y, read_entry.geometry.y);
  EXPECT_EQ(entry.geometry.scale, read_entry.geometry.scale);
  EXPECT_EQ(entry.geometry.orientation, read_entry.geometry.orientation);
  for (size_t i = 0; i < entry.descriptor.size(); ++i) {
    EXPECT_EQ(entry.descriptor[i], read_entry.descriptor[i]);
  }
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
