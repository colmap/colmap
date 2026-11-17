// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/util/testing.h"

#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(Mat, Empty) {
  Mat<int> mat;
  EXPECT_EQ(mat.GetWidth(), 0);
  EXPECT_EQ(mat.GetHeight(), 0);
  EXPECT_EQ(mat.GetDepth(), 0);
  EXPECT_EQ(mat.GetNumBytes(), 0);
}

TEST(Mat, NonEmpty) {
  Mat<int> mat(1, 2, 3);
  EXPECT_EQ(mat.GetWidth(), 1);
  EXPECT_EQ(mat.GetHeight(), 2);
  EXPECT_EQ(mat.GetDepth(), 3);
  EXPECT_EQ(mat.GetNumBytes(), 24);
}

TEST(Mat, GetSet) {
  Mat<int> mat(1, 2, 3);

  EXPECT_EQ(mat.GetNumBytes(), 24);

  mat.Set(0, 0, 0, 1);
  mat.Set(0, 0, 1, 2);
  mat.Set(0, 0, 2, 3);
  mat.Set(1, 0, 0, 4);
  mat.Set(1, 0, 1, 5);
  mat.Set(1, 0, 2, 6);

  EXPECT_EQ(mat.Get(0, 0, 0), 1);
  EXPECT_EQ(mat.Get(0, 0, 1), 2);
  EXPECT_EQ(mat.Get(0, 0, 2), 3);
  EXPECT_EQ(mat.Get(1, 0, 0), 4);
  EXPECT_EQ(mat.Get(1, 0, 1), 5);
  EXPECT_EQ(mat.Get(1, 0, 2), 6);

  int slice[3];
  mat.GetSlice(0, 0, slice);
  EXPECT_EQ(slice[0], 1);
  EXPECT_EQ(slice[1], 2);
  EXPECT_EQ(slice[2], 3);
  mat.GetSlice(1, 0, slice);
  EXPECT_EQ(slice[0], 4);
  EXPECT_EQ(slice[1], 5);
  EXPECT_EQ(slice[2], 6);
}

TEST(Mat, Fill) {
  Mat<int> mat(1, 2, 3);

  EXPECT_EQ(mat.GetNumBytes(), 24);

  mat.Fill(10);
  mat.Set(0, 0, 0, 10);
  mat.Set(0, 0, 1, 10);
  mat.Set(0, 0, 2, 10);
  mat.Set(1, 0, 0, 10);
  mat.Set(1, 0, 1, 10);
  mat.Set(1, 0, 2, 10);
}

TEST(Mat, RejectsTruncatedFloatPayloadAndCanBeRetried) {
  const auto path = CreateTestDir() / "matrix.bin";
  Mat<float> expected(3, 2, 2);
  for (size_t i = 0; i < expected.GetData().size(); ++i) {
    expected.GetPtr()[i] = static_cast<float>(i) + 0.25f;
  }
  for (const size_t missing_bytes : {sizeof(float), size_t{1}}) {
    SCOPED_TRACE(missing_bytes);
    expected.Write(path);
    std::filesystem::resize_file(
        path, std::filesystem::file_size(path) - missing_bytes);
    Mat<float> actual;
    EXPECT_THAT([&] { actual.Read(path); },
                testing::ThrowsMessage<std::ios_base::failure>(
                    testing::HasSubstr(path.filename().string())));

    expected.Write(path);
    actual.Read(path);
    EXPECT_EQ(actual.GetWidth(), expected.GetWidth());
    EXPECT_EQ(actual.GetHeight(), expected.GetHeight());
    EXPECT_EQ(actual.GetDepth(), expected.GetDepth());
    EXPECT_EQ(actual.GetData(), expected.GetData());
  }

  // Keep accepting unused bytes after the declared payload.
  {
    std::ofstream file(path, std::ios::binary | std::ios::app);
    file << "tail";
    ASSERT_TRUE(file.good());
  }
  Mat<float> actual;
  actual.Read(path);
  EXPECT_EQ(actual.GetData(), expected.GetData());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
