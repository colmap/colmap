// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/depth_map.h"

#include "colmap/image/warp.h"
#include "colmap/util/testing.h"

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

TEST(DepthMap, RescalePreservesConstantDepth) {
  DepthMap map(8, 6, 0, 2);
  map.Fill(1.75f);
  map.Rescale(0.5f);
  for (const float depth : map.GetData()) {
    EXPECT_NEAR(depth, 1.75f, 1e-5f);
  }
}

TEST(DepthMap, RescaleValuesAndReuse) {
  DepthMap map(7, 5, 0.25f, 40.0f);
  for (size_t i = 0; i < map.GetData().size(); ++i) {
    map.GetPtr()[i] = static_cast<float>(i) + 0.5f;
  }
  for (const float factor : {0.5f, 1.0f, 0.75f, 0.5f}) {
    const size_t width = std::round(map.GetWidth() * factor);
    const size_t height = std::round(map.GetHeight() * factor);
    std::vector<float> expected(width * height);
    DownsampleImage(map.GetPtr(),
                    map.GetHeight(),
                    map.GetWidth(),
                    height,
                    width,
                    expected.data());
    map.Rescale(factor);
    EXPECT_EQ(map.GetWidth(), width);
    EXPECT_EQ(map.GetHeight(), height);
    EXPECT_EQ(map.GetDepth(), 1);
    EXPECT_EQ(map.GetData(), expected);
    EXPECT_EQ(map.GetDepthMin(), 0.25f);
    EXPECT_EQ(map.GetDepthMax(), 40.0f);
  }

  const auto path = CreateTestDir() / "rescaled.bin";
  map.Write(path);
  DepthMap loaded;
  loaded.Read(path);
  EXPECT_EQ(loaded.GetWidth(), map.GetWidth());
  EXPECT_EQ(loaded.GetHeight(), map.GetHeight());
  EXPECT_EQ(loaded.GetDepth(), 1);
  EXPECT_EQ(loaded.GetData(), map.GetData());
  loaded.Downsize(1, 1);
  EXPECT_EQ(loaded.GetWidth(), 1);
  EXPECT_EQ(loaded.GetHeight(), 1);
  EXPECT_EQ(loaded.GetData().size(), 1);

  DepthMap empty;
  empty.Rescale(0.5f);
  EXPECT_TRUE(empty.GetData().empty());
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
