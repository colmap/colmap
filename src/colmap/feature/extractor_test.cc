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

#include "colmap/feature/extractor.h"

#include "colmap/feature/aliked.h"
#include "colmap/feature/sift.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FeatureExtractionOptions, Copy) {
  FeatureExtractionOptions options;
  options.max_image_size += 100;
  options.sift->max_num_features += 100;
  options.aliked->max_num_features += 100;

  FeatureExtractionOptions copy = options;

  // Verify fields are copied
  EXPECT_EQ(copy.max_image_size, options.max_image_size);
  EXPECT_EQ(copy.sift->max_num_features, options.sift->max_num_features);
  EXPECT_EQ(copy.aliked->max_num_features, options.aliked->max_num_features);

  // Verify deep copy of shared_ptr (different pointer instances)
  EXPECT_NE(options.sift.get(), copy.sift.get());
  EXPECT_NE(options.aliked.get(), copy.aliked.get());
}

TEST(FeatureExtractionOptions, EffMaxImageSize) {
  FeatureExtractionOptions options;

  // When max_image_size is explicitly set, use that value.
  options.max_image_size = 2000;
  for (const auto& type : {FeatureExtractorType::SIFT,
                           FeatureExtractorType::ALIKED_N16ROT,
                           FeatureExtractorType::ALIKED_N32}) {
    options.type = type;
    EXPECT_EQ(options.EffMaxImageSize(), 2000);
  }

  // When max_image_size is non-positive, use type-specific defaults.
  options.max_image_size = -1;
  options.type = FeatureExtractorType::SIFT;
  EXPECT_EQ(options.EffMaxImageSize(), 3200);
  options.type = FeatureExtractorType::ALIKED_N16ROT;
  EXPECT_EQ(options.EffMaxImageSize(), 1600);
  options.type = FeatureExtractorType::ALIKED_N32;
  EXPECT_EQ(options.EffMaxImageSize(), 1600);

  options.max_image_size = 0;
  options.type = FeatureExtractorType::SIFT;
  EXPECT_EQ(options.EffMaxImageSize(), 3200);
  options.type = FeatureExtractorType::ALIKED_N16ROT;
  EXPECT_EQ(options.EffMaxImageSize(), 1600);
  options.type = FeatureExtractorType::ALIKED_N32;
  EXPECT_EQ(options.EffMaxImageSize(), 1600);
}

TEST(FeatureExtractionOptions, CopyAssignment) {
  FeatureExtractionOptions options;
  options.max_image_size = 999;
  options.sift->max_num_features += 200;
  options.aliked->max_num_features += 300;

  // Test copy assignment into a default-constructed instance.
  FeatureExtractionOptions assigned;
  assigned = options;

  EXPECT_EQ(assigned.max_image_size, 999);
  EXPECT_EQ(assigned.sift->max_num_features, options.sift->max_num_features);
  EXPECT_EQ(assigned.aliked->max_num_features,
            options.aliked->max_num_features);

  // Verify deep copy (different pointer instances).
  EXPECT_NE(assigned.sift.get(), options.sift.get());
  EXPECT_NE(assigned.aliked.get(), options.aliked.get());

  // Mutating the copy must not affect the original.
  assigned.sift->max_num_features += 1;
  EXPECT_NE(assigned.sift->max_num_features, options.sift->max_num_features);

  // Test self-assignment (assign via const ref to avoid -Wself-assign).
  const auto* sift_ptr_before = options.sift.get();
  const auto* aliked_ptr_before = options.aliked.get();
  const auto max_image_size_before = options.max_image_size;
  const auto& self_ref = options;
  options = self_ref;
  EXPECT_EQ(options.sift.get(), sift_ptr_before);
  EXPECT_EQ(options.aliked.get(), aliked_ptr_before);
  EXPECT_EQ(options.max_image_size, max_image_size_before);
}

TEST(FeatureExtractionOptions, RequiresRGB) {
  FeatureExtractionOptions options;

  const std::vector<std::pair<FeatureExtractorType, bool>> kTestCases = {
      {FeatureExtractorType::SIFT, false},
      {FeatureExtractorType::ALIKED_N16ROT, true},
      {FeatureExtractorType::ALIKED_N32, true},
  };

  for (const auto& [type, expected] : kTestCases) {
    options.type = type;
    EXPECT_EQ(options.RequiresRGB(), expected);
  }
}

TEST(FeatureExtractionOptions, CheckAndRequiresOpenGLWithNoGpu) {
  FeatureExtractionOptions options;
  options.use_gpu = false;

  const std::vector<FeatureExtractorType> kTypes = {
      FeatureExtractorType::SIFT,
      FeatureExtractorType::ALIKED_N16ROT,
      FeatureExtractorType::ALIKED_N32,
  };

  for (const auto& type : kTypes) {
    options.type = type;
    EXPECT_TRUE(options.Check());
    EXPECT_FALSE(options.RequiresOpenGL());
  }
}

}  // namespace
}  // namespace colmap
