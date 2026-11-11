// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/normal_map.h"

#include "colmap/image/warp.h"
#include "colmap/util/testing.h"

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

TEST(NormalMap, RescaleKeepsUnitNormals) {
  NormalMap map(8, 6);
  map.Fill(2.0f);
  map.Rescale(0.5f);
  for (const float component : map.GetData()) {
    EXPECT_NEAR(component, 1.0f / std::sqrt(3.0f), 1e-5f);
  }
}

TEST(NormalMap, RescaleValuesAndReuse) {
  for (const bool zero : {false, true}) {
    NormalMap map(7, 5);
    for (size_t r = 0; r < map.GetHeight(); ++r) {
      for (size_t c = 0; c < map.GetWidth(); ++c) {
        map.Set(r, c, 0, zero ? 0.0f : static_cast<float>(c + 1));
        map.Set(r, c, 1, zero ? 0.0f : -static_cast<float>(r + 1));
        map.Set(r, c, 2, zero ? 0.0f : 2.0f);
      }
    }
    for (const float factor : {0.5f, 1.0f, 0.75f, 0.5f}) {
      const size_t width = std::round(map.GetWidth() * factor);
      const size_t height = std::round(map.GetHeight() * factor);
      Mat<float> expected(width, height, 3);
      for (size_t d = 0; d < 3; ++d) {
        DownsampleImage(map.GetPtr() + d * map.GetWidth() * map.GetHeight(),
                        map.GetHeight(),
                        map.GetWidth(),
                        height,
                        width,
                        expected.GetPtr() + d * width * height);
      }
      for (size_t r = 0; r < height; ++r) {
        for (size_t c = 0; c < width; ++c) {
          Eigen::Vector3f normal(expected.Get(r, c, 0),
                                 expected.Get(r, c, 1),
                                 expected.Get(r, c, 2));
          const float squared_norm = normal.squaredNorm();
          if (squared_norm > 0) {
            normal /= std::sqrt(squared_norm);
          }
          for (size_t d = 0; d < 3; ++d) {
            expected.Set(r, c, d, normal(d));
          }
        }
      }
      map.Rescale(factor);
      EXPECT_EQ(map.GetWidth(), width);
      EXPECT_EQ(map.GetHeight(), height);
      EXPECT_EQ(map.GetDepth(), 3);
      EXPECT_EQ(map.GetData(), expected.GetData());
    }

    const auto path = CreateTestDir() / "rescaled.bin";
    map.Write(path);
    NormalMap loaded;
    loaded.Read(path);
    EXPECT_EQ(loaded.GetWidth(), map.GetWidth());
    EXPECT_EQ(loaded.GetHeight(), map.GetHeight());
    EXPECT_EQ(loaded.GetDepth(), 3);
    EXPECT_EQ(loaded.GetData(), map.GetData());
    loaded.Downsize(1, 1);
    EXPECT_EQ(loaded.GetWidth(), 1);
    EXPECT_EQ(loaded.GetHeight(), 1);
    EXPECT_EQ(loaded.GetData().size(), 3);
  }

  NormalMap empty;
  empty.Rescale(0.5f);
  EXPECT_TRUE(empty.GetData().empty());
}

TEST(NormalMap, Downsize) {
  NormalMap normal_map(6, 7);
  normal_map.Downsize(2, 4);
  EXPECT_EQ(normal_map.GetWidth(), 2);
  EXPECT_EQ(normal_map.GetHeight(), 2);
  EXPECT_EQ(normal_map.GetDepth(), 3);
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
