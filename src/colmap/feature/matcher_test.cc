// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/matcher.h"

#include "colmap/feature/aliked.h"
#include "colmap/feature/cusift_matcher.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/loma.h"
#include "colmap/feature/onnx_matchers.h"
#include "colmap/feature/sift.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FeatureMatchingOptions, Copy) {
  FeatureMatchingOptions options;
  options.max_num_matches += 100;
  options.sift->max_ratio *= 0.1;
  options.aliked->brute_force.min_cossim *= 0.1;
  options.cusift->max_ratio *= 0.1;

  FeatureMatchingOptions copy = options;

  // Verify fields are copied
  EXPECT_EQ(copy.max_num_matches, options.max_num_matches);
  EXPECT_EQ(copy.sift->max_ratio, options.sift->max_ratio);
  EXPECT_EQ(copy.aliked->brute_force.min_cossim,
            options.aliked->brute_force.min_cossim);
  EXPECT_EQ(copy.cusift->max_ratio, options.cusift->max_ratio);

  // Verify deep copy of shared_ptr (different pointer instances)
  EXPECT_NE(options.sift.get(), copy.sift.get());
  EXPECT_NE(options.aliked.get(), copy.aliked.get());
  EXPECT_NE(options.cusift.get(), copy.cusift.get());
}

TEST(FeatureMatchingOptions, CopyAssignment) {
  FeatureMatchingOptions options;
  options.max_num_matches += 100;
  options.sift->max_ratio *= 0.1;
  options.aliked->brute_force.min_cossim *= 0.1;
  options.cusift->max_ratio *= 0.1;

  FeatureMatchingOptions assigned;
  assigned = options;

  // Verify fields are copied.
  EXPECT_EQ(assigned.max_num_matches, options.max_num_matches);
  EXPECT_EQ(assigned.sift->max_ratio, options.sift->max_ratio);
  EXPECT_EQ(assigned.aliked->brute_force.min_cossim,
            options.aliked->brute_force.min_cossim);
  EXPECT_EQ(assigned.cusift->max_ratio, options.cusift->max_ratio);

  // Verify deep copy (different pointer instances).
  EXPECT_NE(options.sift.get(), assigned.sift.get());
  EXPECT_NE(options.aliked.get(), assigned.aliked.get());
  EXPECT_NE(options.cusift.get(), assigned.cusift.get());

  // Test self-assignment (assign via const ref to avoid -Wself-assign).
  const auto* sift_ptr = options.sift.get();
  const auto* aliked_ptr = options.aliked.get();
  const auto* cusift_ptr = options.cusift.get();
  const auto prev_max_num_matches = options.max_num_matches;
  const auto& self_ref = options;
  options = self_ref;
  EXPECT_EQ(options.sift.get(), sift_ptr);
  EXPECT_EQ(options.aliked.get(), aliked_ptr);
  EXPECT_EQ(options.cusift.get(), cusift_ptr);
  EXPECT_EQ(options.max_num_matches, prev_max_num_matches);
}

TEST(FeatureMatchingOptions, RequiresOpenGL) {
  FeatureMatchingOptions options;
  options.use_gpu = false;

  const FeatureMatcherType types[] = {FeatureMatcherType::SIFT_BRUTEFORCE,
                                      FeatureMatcherType::SIFT_LIGHTGLUE,
                                      FeatureMatcherType::ALIKED_BRUTEFORCE,
                                      FeatureMatcherType::ALIKED_LIGHTGLUE,
                                      FeatureMatcherType::LOMA_BRUTEFORCE,
                                      FeatureMatcherType::LOMA_B,
                                      FeatureMatcherType::LOMA_B128,
                                      FeatureMatcherType::LOMA_R,
                                      FeatureMatcherType::LOMA_L,
                                      FeatureMatcherType::LOMA_G,
                                      FeatureMatcherType::CUSIFT_GEMM};

  for (const auto type : types) {
    options.type = type;
    EXPECT_FALSE(options.RequiresOpenGL());
  }
}

TEST(FeatureMatchingOptions, Check) {
  FeatureMatchingOptions options;
  options.use_gpu = false;

  const FeatureMatcherType types[] = {FeatureMatcherType::SIFT_BRUTEFORCE,
                                      FeatureMatcherType::SIFT_LIGHTGLUE,
                                      FeatureMatcherType::ALIKED_BRUTEFORCE,
                                      FeatureMatcherType::ALIKED_LIGHTGLUE,
                                      FeatureMatcherType::LOMA_BRUTEFORCE,
                                      FeatureMatcherType::LOMA_B,
                                      FeatureMatcherType::LOMA_B128,
                                      FeatureMatcherType::LOMA_R,
                                      FeatureMatcherType::LOMA_L,
                                      FeatureMatcherType::LOMA_G};

  for (const auto type : types) {
    options.type = type;
    EXPECT_TRUE(options.Check());
  }

  // cusift has no CPU implementation, so it is the one type that must reject
  // use_gpu = false instead of falling back -- same rule
  // FeatureExtractionOptions applies to FeatureExtractorType::CUSIFT
  // (extractor_test.cc, CheckAndRequiresOpenGLWithNoGpu).
  options.type = FeatureMatcherType::CUSIFT_GEMM;
  EXPECT_FALSE(options.Check());
}

}  // namespace
}  // namespace colmap
