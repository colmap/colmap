// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/depth_map.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(DepthMap, Empty) {
  DepthMap depth_map;
  EXPECT_EQ(depth_map.GetWidth(), 0);
  EXPECT_EQ(depth_map.GetHeight(), 0);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), -1);
  EXPECT_EQ(depth_map.GetDepthMax(), -1);
}

TEST(DepthMap, NonEmpty) {
  DepthMap depth_map(1, 2, 0, 1);
  EXPECT_EQ(depth_map.GetWidth(), 1);
  EXPECT_EQ(depth_map.GetHeight(), 2);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, Rescale) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Rescale(0.5);
  EXPECT_EQ(depth_map.GetWidth(), 3);
  EXPECT_EQ(depth_map.GetHeight(), 4);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, Downsize) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Downsize(2, 4);
  EXPECT_EQ(depth_map.GetWidth(), 2);
  EXPECT_EQ(depth_map.GetHeight(), 2);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, KeepsThinDimensions) {
  for (const bool portrait : {false, true}) {
    for (const bool downsize : {false, true}) {
      DepthMap map(portrait ? 1 : 100, portrait ? 100 : 1, 0.1f, 10.0f);
      map.Fill(1.5f);
      if (downsize) {
        map.Downsize(10, 10);
      } else {
        map.Rescale(0.1f);
      }
      EXPECT_EQ(map.GetWidth(), portrait ? 1 : 10);
      EXPECT_EQ(map.GetHeight(), portrait ? 10 : 1);
      ASSERT_EQ(map.GetData().size(), 10);
      for (const float depth : map.GetData()) {
        EXPECT_NEAR(depth, 1.5f, 1e-5f);
      }
      EXPECT_FLOAT_EQ(map.GetDepthMin(), 0.1f);
      EXPECT_FLOAT_EQ(map.GetDepthMax(), 10.0f);
    }
  }
}

TEST(DepthMap, ToBitmap) {
  DepthMap depth_map(2, 2, 0.1, 0.9);
  depth_map.Fill(0.9);
  depth_map.Set(0, 0, 0, 0.1);
  depth_map.Set(0, 1, 0, 0.5);
  const Bitmap bitmap = depth_map.ToBitmap(0, 100);
  EXPECT_EQ(bitmap.Width(), depth_map.GetWidth());
  EXPECT_EQ(bitmap.Height(), depth_map.GetHeight());
  EXPECT_TRUE(bitmap.IsRGB());
  EXPECT_EQ(bitmap.GetPixel(0, 0).value(), BitmapColor<uint8_t>(0, 0, 128));
  EXPECT_EQ(bitmap.GetPixel(0, 1).value(), BitmapColor<uint8_t>(128, 0, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 0).value(), BitmapColor<uint8_t>(128, 255, 127));
  EXPECT_EQ(bitmap.GetPixel(1, 1).value(), BitmapColor<uint8_t>(128, 0, 0));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
