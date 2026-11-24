// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/normal_map.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(NormalMap, Empty) {
  NormalMap normal_map;
  EXPECT_EQ(normal_map.GetWidth(), 0);
  EXPECT_EQ(normal_map.GetHeight(), 0);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, NonEmpty) {
  NormalMap normal_map(1, 2);
  EXPECT_EQ(normal_map.GetWidth(), 1);
  EXPECT_EQ(normal_map.GetHeight(), 2);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, Rescale) {
  NormalMap normal_map(6, 7);
  normal_map.Rescale(0.5);
  EXPECT_EQ(normal_map.GetWidth(), 3);
  EXPECT_EQ(normal_map.GetHeight(), 4);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, Downsize) {
  NormalMap normal_map(6, 7);
  normal_map.Downsize(2, 4);
  EXPECT_EQ(normal_map.GetWidth(), 2);
  EXPECT_EQ(normal_map.GetHeight(), 2);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, KeepsThinDimensions) {
  for (const bool portrait : {false, true}) {
    for (const bool downsize : {false, true}) {
      NormalMap map(portrait ? 1 : 100, portrait ? 100 : 1);
      map.Fill(0.0f);
      for (size_t y = 0; y < map.GetHeight(); ++y) {
        for (size_t x = 0; x < map.GetWidth(); ++x) {
          map.Set(y, x, 2, -1.0f);
        }
      }
      if (downsize) {
        map.Downsize(10, 10);
      } else {
        map.Rescale(0.1f);
      }
      EXPECT_EQ(map.GetWidth(), portrait ? 1 : 10);
      EXPECT_EQ(map.GetHeight(), portrait ? 10 : 1);
      ASSERT_EQ(map.GetData().size(), 30);
      for (size_t y = 0; y < map.GetHeight(); ++y) {
        for (size_t x = 0; x < map.GetWidth(); ++x) {
          EXPECT_FLOAT_EQ(map.Get(y, x, 0), 0.0f);
          EXPECT_FLOAT_EQ(map.Get(y, x, 1), 0.0f);
          EXPECT_FLOAT_EQ(map.Get(y, x, 2), -1.0f);
        }
      }
    }
  }
}

TEST(NormalMap, ToBitmap) {
  NormalMap normal_map(2, 2);
  normal_map.Set(0, 0, 0, 0);
  normal_map.Set(0, 0, 1, 0);
  normal_map.Set(0, 0, 2, 1);
  normal_map.Set(0, 1, 0, 0);
  normal_map.Set(0, 1, 1, 1);
  normal_map.Set(0, 1, 2, 0);
  normal_map.Set(1, 0, 0, 1);
  normal_map.Set(1, 0, 1, 0);
  normal_map.Set(1, 0, 2, 0);
  normal_map.Set(1, 1, 0, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 1, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 2, 0);
  const Bitmap bitmap = normal_map.ToBitmap();
  EXPECT_EQ(bitmap.Width(), normal_map.GetWidth());
  EXPECT_EQ(bitmap.Height(), normal_map.GetHeight());
  EXPECT_TRUE(bitmap.IsRGB());
  EXPECT_EQ(bitmap.GetPixel(0, 0).value(), BitmapColor<uint8_t>(128, 128, 0));
  EXPECT_EQ(bitmap.GetPixel(0, 1).value(), BitmapColor<uint8_t>(0, 128, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 0).value(), BitmapColor<uint8_t>(128, 0, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 1).value(), BitmapColor<uint8_t>(37, 37, 0));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
