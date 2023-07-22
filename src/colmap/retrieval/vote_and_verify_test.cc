// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/retrieval/visual_index.h"

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {

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

}  // namespace retrieval
}  // namespace colmap
