// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

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

}  // namespace
}  // namespace mvs
}  // namespace colmap
