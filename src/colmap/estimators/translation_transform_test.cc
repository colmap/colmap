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

#include "colmap/estimators/translation_transform.h"

#include "colmap/math/random.h"
#include "colmap/optim/ransac.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(TranslationTransform, Estimate) {
  SetPRNGSeed(0);

  std::vector<Eigen::Vector2d> src;
  for (size_t i = 0; i < 100; ++i) {
    src.emplace_back(RandomUniformReal(-1000.0, 1000.0),
                     RandomUniformReal(-1000.0, 1000.0));
  }

  Eigen::Vector2d translation(RandomUniformReal(-1000.0, 1000.0),
                              RandomUniformReal(-1000.0, 1000.0));

  std::vector<Eigen::Vector2d> dst;
  for (size_t i = 0; i < src.size(); ++i) {
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
