// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/translation_transform.h"

#include "colmap/math/random.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(TranslationTransform, Estimate) {
  constexpr size_t kNumPoints = 100;

  std::vector<Eigen::Vector2d> src;
  src.reserve(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    src.emplace_back(RandomUniformReal(-1000.0, 1000.0),
                     RandomUniformReal(-1000.0, 1000.0));
  }

  Eigen::Vector2d translation(RandomUniformReal(-1000.0, 1000.0),
                              RandomUniformReal(-1000.0, 1000.0));

  std::vector<Eigen::Vector2d> dst;
  dst.reserve(kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    dst.push_back(src[i] + translation);
  }

  std::vector<Eigen::Vector2d> models;
  TranslationTransformEstimator<2>::Estimate(src, dst, &models);

  ASSERT_EQ(models.size(), 1);
  const Eigen::Vector2d& estimated_translation = models[0];

  EXPECT_NEAR(translation(0), estimated_translation(0), 1e-6);
  EXPECT_NEAR(translation(1), estimated_translation(1), 1e-6);

  std::vector<double> residuals;
  TranslationTransformEstimator<2>::Residuals(
      src, dst, estimated_translation, &residuals);

  for (size_t i = 0; i < residuals.size(); ++i) {
    EXPECT_TRUE(residuals[i] < 1e-6);
  }
}

}  // namespace
}  // namespace colmap
