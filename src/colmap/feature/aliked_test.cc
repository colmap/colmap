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

#include "colmap/feature/aliked.h"

#include "colmap/sensor/bitmap.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateImageWithSquare(const int width, const int height, Bitmap* bitmap) {
  bitmap->Allocate(width, height, false);
  bitmap->Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = height / 2 - height / 8; r < height / 2 + height / 8; ++r) {
    for (int c = width / 2 - width / 8; c < width / 2 + width / 8; ++c) {
      bitmap->SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
}

TEST(ALIKED, Nominal) {
  Bitmap image;
  CreateImageWithSquare(512, 512, &image);

  FeatureExtractionOptions extraction_options(FeatureExtractorType::ALIKED);
  auto extractor = CreateALIKEDFeatureExtractor(extraction_options);
  auto keypoints = std::make_shared<FeatureKeypoints>();
  auto descriptors = std::make_shared<FeatureDescriptors>();
  ASSERT_TRUE(extractor->Extract(image, keypoints.get(), descriptors.get()));
  EXPECT_EQ(keypoints->size(), 24);
  EXPECT_EQ(keypoints->size(), descriptors->rows());
  EXPECT_EQ(descriptors->cols(), 128 * sizeof(float));

  for (const auto& matching_options :
       {FeatureMatchingOptions(FeatureMatcherType::ALIKED)}) {
    auto matcher = CreateALIKEDFeatureMatcher(matching_options);
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
    EXPECT_GT(matches.size(), 0);
  }
}

}  // namespace
}  // namespace colmap
