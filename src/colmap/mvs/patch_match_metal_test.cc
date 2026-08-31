// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in COPYING.txt
// are met.

#include "colmap/mvs/patch_match.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

Image MakeImage(const int index,
                const size_t width,
                const size_t height,
                const float translation_x) {
  const float K[9] = {8.0f,
                      0.0f,
                      (width - 1) / 2.0f,
                      0.0f,
                      8.0f,
                      (height - 1) / 2.0f,
                      0.0f,
                      0.0f,
                      1.0f};
  const float R[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  const float T[3] = {translation_x, 0.0f, 0.0f};
  Image image(std::to_string(index) + ".png", width, height, K, R, T);

  Bitmap bitmap(width, height, false);
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      bitmap.RowMajorData()[row * width + col] =
          static_cast<uint8_t>(17 * col + 11 * row);
    }
  }
  image.SetBitmap(std::move(bitmap));
  return image;
}

TEST(PatchMatchMetal, SyntheticGeometricProblem) {
  constexpr size_t kWidth = 8;
  constexpr size_t kHeight = 6;
  std::vector<Image> images;
  images.push_back(MakeImage(0, kWidth, kHeight, 0.0f));
  images.push_back(MakeImage(1, kWidth, kHeight, 0.1f));
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  for (size_t i = 0; i < images.size(); ++i) {
    depth_maps.emplace_back(kWidth, kHeight, 1.0f, 4.0f);
    depth_maps.back().Fill(2.0f);
    normal_maps.emplace_back(kWidth, kHeight);
    for (size_t row = 0; row < kHeight; ++row) {
      for (size_t col = 0; col < kWidth; ++col) {
        normal_maps.back().Set(row, col, 0, 0.0f);
        normal_maps.back().Set(row, col, 1, 0.0f);
        normal_maps.back().Set(row, col, 2, -1.0f);
      }
    }
  }

  PatchMatchOptions options;
  options.depth_min = 1.0;
  options.depth_max = 4.0;
  options.sigma_spatial = 1.0;
  options.sigma_color = 0.2;
  options.window_radius = 1;
  options.window_step = 1;
  options.num_samples = 1;
  options.num_iterations = 1;
  options.filter = false;
  options.geom_consistency = true;
  options.gpu_index = "0";

  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1};
  problem.images = &images;
  problem.depth_maps = &depth_maps;
  problem.normal_maps = &normal_maps;

  PatchMatch patch_match(options, problem);
  patch_match.Run();
  const DepthMap depth = patch_match.GetDepthMap();
  const NormalMap normal = patch_match.GetNormalMap();
  const Mat<float> probability = patch_match.GetSelProbMap();

  ASSERT_EQ(depth.GetWidth(), kWidth);
  ASSERT_EQ(depth.GetHeight(), kHeight);
  ASSERT_EQ(normal.GetDepth(), 3);
  ASSERT_EQ(probability.GetDepth(), 1);
  for (size_t row = 0; row < kHeight; ++row) {
    for (size_t col = 0; col < kWidth; ++col) {
      EXPECT_TRUE(std::isfinite(depth.Get(row, col)));
      // Match CUDA semantics: the configured range initializes depth, while
      // plane propagation and multiplicative refinement may leave that range.
      const float normal_length =
          std::sqrt(normal.Get(row, col, 0) * normal.Get(row, col, 0) +
                    normal.Get(row, col, 1) * normal.Get(row, col, 1) +
                    normal.Get(row, col, 2) * normal.Get(row, col, 2));
      EXPECT_NEAR(normal_length, 1.0f, 1e-4f);
      EXPECT_TRUE(std::isfinite(probability.Get(row, col)));
      EXPECT_GE(probability.Get(row, col), 0.0f);
      EXPECT_LE(probability.Get(row, col), 1.0f);
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
