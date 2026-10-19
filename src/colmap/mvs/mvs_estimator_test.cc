// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mvs_estimator.h"

#include <memory>
#include <stdexcept>

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

  // Mutating the copy must not affect the original.
  copy.patch_match->max_image_size = 4321;
  copy.mvsformer_pp->num_views = 5;
  EXPECT_EQ(options.patch_match->max_image_size, 1234);
  EXPECT_EQ(options.mvsformer_pp->num_views, 10);

  copy = options;
  EXPECT_EQ(copy.type, options.type);
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

TEST(MVSEstimator, CreatePatchMatch) {
  const MVSEstimator::Options options(MVSEstimator::Type::PATCH_MATCH);
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  const std::unique_ptr<MVSEstimator> estimator =
      MVSEstimator::Create(options, /*device_index=*/0);
  ASSERT_NE(estimator, nullptr);
  EXPECT_FALSE(estimator->GetCapabilities().requires_rgb);
  // The caller must resolve the depth range of the problem.
  const MVSEstimator::Problem problem;
  EXPECT_THROW(estimator->Estimate(problem, MVSEstimator::Pass::PHOTOMETRIC),
               std::invalid_argument);
#else
  EXPECT_THROW(MVSEstimator::Create(options, /*device_index=*/0),
               std::runtime_error);
#endif
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
