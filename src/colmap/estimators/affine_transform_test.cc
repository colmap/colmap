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

#include "colmap/estimators/affine_transform.h"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {

TEST(AffineTransform, Nominal) {
  for (int x = 0; x < 10; ++x) {
    Eigen::Matrix<double, 2, 3> A;
    A << x / 10.0, 0.2, 0.3, 30, 0.2, 0.1;

    std::vector<Eigen::Vector2d> src;
    src.emplace_back(x, 0);
    src.emplace_back(1, 0);
    src.emplace_back(2, 1);

    std::vector<Eigen::Vector2d> dst;
    for (size_t i = 0; i < 3; ++i) {
      dst.push_back(A * src[i].homogeneous());
    }

    AffineTransformEstimator estimator;
    const auto models = estimator.Estimate(src, dst);

    EXPECT_EQ(models.size(), 1);

    std::vector<double> residuals;
    estimator.Residuals(src, dst, models[0], &residuals);

    EXPECT_EQ(residuals.size(), 3);

    for (size_t i = 0; i < 3; ++i) {
      EXPECT_LT(residuals[i], 1e-6);
    }
  }
}

}  // namespace colmap
