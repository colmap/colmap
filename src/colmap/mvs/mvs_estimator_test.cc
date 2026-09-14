// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mvs_estimator.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(MVSEstimatorOptions, Defaults) {
  const MVSEstimator::Options options;
  EXPECT_EQ(options.type, MVSEstimator::Type::PATCH_MATCH);
  ASSERT_NE(options.patch_match, nullptr);
  ASSERT_NE(options.mvsformer_pp, nullptr);
  EXPECT_TRUE(options.Check());
}

TEST(MVSEstimatorOptions, DeepCopy) {
  MVSEstimator::Options options;
  options.patch_match->max_image_size = 1234;
  options.mvsformer_pp->num_views = 10;

  MVSEstimator::Options copy = options;
  EXPECT_NE(copy.patch_match, options.patch_match);
  EXPECT_NE(copy.mvsformer_pp, options.mvsformer_pp);
  EXPECT_EQ(copy.patch_match->max_image_size, 1234);
  EXPECT_EQ(copy.mvsformer_pp->num_views, 10);
}

TEST(MVSFormerPlusPlusOptions, Check) {
  MVSFormerPlusPlus::Options options;
  EXPECT_TRUE(options.Check());
  options.num_views = 6;
  EXPECT_FALSE(options.Check());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
