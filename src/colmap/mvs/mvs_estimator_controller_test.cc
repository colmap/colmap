// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mvs_estimator_controller.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(MVSEstimatorController, PatchMatchOutputTypes) {
  MVSEstimator::Options options(MVSEstimator::Type::PATCH_MATCH);
  MVSEstimatorController controller(options, "", "COLMAP", "");
  EXPECT_EQ(controller.OutputType(MVSEstimator::Pass::PHOTOMETRIC),
            "photometric");
  EXPECT_EQ(controller.OutputType(MVSEstimator::Pass::GEOMETRIC), "geometric");
}

TEST(MVSEstimatorController, MVSFormerOutputTypes) {
  MVSEstimator::Options options(MVSEstimator::Type::MVSFORMER_PP);
  options.mvsformer_pp->num_views = 10;
  MVSEstimatorController controller(options, "", "COLMAP", "");
  EXPECT_EQ(controller.OutputType(MVSEstimator::Pass::PHOTOMETRIC),
            "mvsformer_pp_10.photometric");
  EXPECT_EQ(controller.OutputType(MVSEstimator::Pass::GEOMETRIC),
            "mvsformer_pp_10.geometric");
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
