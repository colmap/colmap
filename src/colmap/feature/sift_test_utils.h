// SPDX-License-Identifier: BSD-3-Clause

// Test fixtures shared by the SIFT and cusift test binaries.

#pragma once

#include "colmap/feature/extractor.h"
#include "colmap/feature/types.h"
#include "colmap/feature/utils.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/sensor/bitmap.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {

// A black image with a white square in its middle.
inline Bitmap CreateImageWithSquare(const int size) {
  Bitmap bitmap(size, size, false);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap.SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

// A synthetic scene of many overlapping squares, so that the extractor finds
// a few hundred keypoints across several octaves rather than the handful a
// single square gives. (dx, dy) shifts the whole scene, which yields a second
// view with real correspondences for the matcher tests.
inline Bitmap CreateTexturedImage(const int width,
                                  const int height,
                                  const int dx = 0,
                                  const int dy = 0) {
  Bitmap bitmap(width, height, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(40));
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> pos_x(0, width - 1);
  std::uniform_int_distribution<int> pos_y(0, height - 1);
  std::uniform_int_distribution<int> side(4, 28);
  std::uniform_int_distribution<int> value(70, 255);
  for (int k = 0; k < 150; ++k) {
    const int x0 = pos_x(rng) + dx;
    const int y0 = pos_y(rng) + dy;
    const int s = side(rng);
    const BitmapColor<uint8_t> color(static_cast<uint8_t>(value(rng)));
    for (int y = std::max(y0, 0); y < std::min(y0 + s, height); ++y) {
      for (int x = std::max(x0, 0); x < std::min(x0 + s, width); ++x) {
        bitmap.SetPixel(x, y, color);
      }
    }
  }
  return bitmap;
}

inline FeatureExtractionOptions CreateCusiftOptions() {
  FeatureExtractionOptions options(FeatureExtractorType::CUSIFT);
  options.use_gpu = true;
  return options;
}

inline FeatureDescriptors CreateEmptyDescriptors() {
  return FeatureDescriptors(FeatureExtractorType::SIFT,
                            FeatureDescriptorsData(0, 128));
}

// The descriptors in reverse row order, for matcher symmetry tests.
inline FeatureDescriptors CreateReversedDescriptors(
    const FeatureDescriptors& src) {
  return FeatureDescriptors(src.type, src.data.colwise().reverse());
}

// Each row is a sparse random unit vector (10 of 128 dims set to 1, then
// L2-normalized and quantized), so distinct rows are near-orthogonal and
// cross-row matches are clearly separated from a row's match with itself.
inline FeatureDescriptors CreateRandomFeatureDescriptors(
    const size_t num_features) {
  SetPRNGSeed(0);
  FeatureDescriptorsFloatData descriptors_float =
      FeatureDescriptorsFloatData::Zero(num_features, 128);
  std::vector<int> dims(128);
  std::iota(dims.begin(), dims.end(), 0);
  for (size_t i = 0; i < num_features; ++i) {
    std::shuffle(dims.begin(), dims.end(), *PRNG);
    for (size_t j = 0; j < 10; ++j) {
      descriptors_float(i, dims[j]) = 1.0f;
    }
  }
  L2NormalizeFeatureDescriptors(&descriptors_float);
  return FeatureDescriptors(
      FeatureExtractorType::SIFT,
      FeatureDescriptorsToUnsignedByte(descriptors_float));
}

inline void ValidateKeypoints(const FeatureKeypoints& keypoints,
                              const Bitmap& bitmap) {
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }
}

// Every row has COLMAP's SIFT descriptor norm of 512, within rounding.
inline void ValidateDescriptorNorms(const FeatureDescriptors& descriptors) {
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

// Matches of two descriptors against their reversed copies: 0-1 and 1-0.
inline void ExpectReversedMatches(const FeatureMatches& matches) {
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 1);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 0);
}

}  // namespace colmap
