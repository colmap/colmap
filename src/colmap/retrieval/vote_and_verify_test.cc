// SPDX-License-Identifier: BSD-3-Clause

#define _USE_MATH_DEFINES

#include "colmap/retrieval/vote_and_verify.h"

#include "colmap/math/random.h"
#include "colmap/retrieval/geometry.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

struct SyntheticData {
  FeatureGeometryTransform img2_from_img1;
  std::vector<FeatureGeometryMatch> matches;
};

SyntheticData SynthesizeData(size_t num_inliers, size_t num_outliers) {
  SyntheticData data;
  data.img2_from_img1.scale = RandomUniformReal<float>(0.1, 10);
  data.img2_from_img1.angle = RandomUniformReal<float>(0, 2 * M_PI);
  data.img2_from_img1.tx = RandomUniformReal<float>(-100, 100);
  data.img2_from_img1.ty = RandomUniformReal<float>(-100, 100);

  const float sin_angle = std::sin(data.img2_from_img1.angle);
  const float cos_angle = std::cos(data.img2_from_img1.angle);

  data.matches.resize(num_inliers + num_outliers);
  for (size_t i = 0; i < num_inliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    match.geometry2.scale = data.img2_from_img1.scale * match.geometry1.scale;
    match.geometry2.orientation =
        data.img2_from_img1.angle + match.geometry1.orientation;
    match.geometry2.x =
        data.img2_from_img1.scale *
        (cos_angle * match.geometry1.x - sin_angle * match.geometry1.y);
    match.geometry2.y =
        data.img2_from_img1.scale *
        (sin_angle * match.geometry1.x + cos_angle * match.geometry1.y);
  }

  for (size_t i = 0; i < num_outliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i + num_inliers];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    match.geometry2.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry2.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry2.x = RandomUniformReal<float>(-100, 100);
    match.geometry2.y = RandomUniformReal<float>(-100, 100);
  }

  return data;
}

TEST(VoteAndVerify, NoMatches) {
  EXPECT_EQ(VoteAndVerify(VoteAndVerifyOptions(), {}), 0);
}

TEST(VoteAndVerify, NoEffectiveInliers) {
  const size_t kNumInliers = 100;
  const size_t kNumOutliers = 50;
  const auto data = SynthesizeData(kNumInliers, kNumOutliers);
  VoteAndVerifyOptions options;
  options.eff_inlier_count = false;
  const int num_inliers = VoteAndVerify(options, data.matches);
  EXPECT_EQ(num_inliers, kNumInliers);
}

TEST(VoteAndVerify, EffectiveInliers) {
  const size_t kNumInliers = 100;
  const size_t kNumOutliers = 50;
  const auto data = SynthesizeData(kNumInliers, kNumOutliers);
  VoteAndVerifyOptions options;
  options.eff_inlier_count = true;
  const int num_inliers = VoteAndVerify(options, data.matches);
  EXPECT_GT(num_inliers, 0.8 * kNumInliers);
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
