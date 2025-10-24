// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/feature/xfeat.h"

#include "colmap/math/random.h"
#include "colmap/sensor/bitmap.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateImageWithSquare(const int width, const int height, Bitmap* bitmap) {
  SetPRNGSeed(42);
  bitmap->Allocate(width, height, /*as_rgb=*/true);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      bitmap->SetPixel(r,
                       c,
                       BitmapColor<uint8_t>(RandomUniformInteger(0, 255),
                                            RandomUniformInteger(0, 255),
                                            RandomUniformInteger(0, 255)));
    }
  }
}

TEST(XFeat, Nominal) {
  Bitmap image;
  CreateImageWithSquare(1024, 768, &image);

  FeatureExtractionOptions extraction_options(FeatureExtractorType::XFEAT);
  extraction_options.use_gpu = false;
  auto extractor = CreateXFeatFeatureExtractor(extraction_options);
  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(extractor->Extract(image, keypoints.get(), descriptors.get()));
  // Different platforms lead to slightly different number of keypoints.
  EXPECT_NEAR(keypoints->size(), 2048, 20);
  EXPECT_EQ(keypoints->size(), descriptors->rows());
  EXPECT_EQ(descriptors->cols(), 64 * sizeof(float));
  for (const auto& keypoint : *keypoints) {
    EXPECT_GE(keypoint.x, -5);
    EXPECT_GE(keypoint.y, -5);
    EXPECT_LE(keypoint.x, image.Width() + 5);
    EXPECT_LE(keypoint.y, image.Height() + 5);
  }

  for (const auto& matcher_type : {FeatureMatcherType::XFEAT_BRUTEFORCE,
                                   FeatureMatcherType::XFEAT_LIGHTERGLUE}) {
    FeatureMatchingOptions matching_options(matcher_type);
    matching_options.use_gpu = false;
    auto matcher = CreateXFeatFeatureMatcher(matching_options);
    FeatureMatches matches;
    matcher->Match({/*image_id=*/1,
                    /*image_width=*/image.Width(),
                    /*image_height=*/image.Height(),
                    keypoints,
                    descriptors},
                   {/*image_id=*/2,
                    /*image_width=*/image.Width(),
                    /*image_height=*/image.Height(),
                    keypoints,
                    descriptors},
                   &matches);
    EXPECT_NEAR(matches.size(), keypoints->size(), 0.05 * keypoints->size());
    for (const auto& match : matches) {
      EXPECT_GE(match.point2D_idx1, 0);
      EXPECT_GE(match.point2D_idx2, 0);
      EXPECT_LT(match.point2D_idx1, keypoints->size());
      EXPECT_LT(match.point2D_idx2, keypoints->size());
    }
  }
}

}  // namespace
}  // namespace colmap
