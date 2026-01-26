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

#include "colmap/estimators/cost_functions/focal_length.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// TODO(jsch): Add meaningful tests for FetzerFocalLengthCostFunctor.

TEST(FetzerFocalLengthCostFunctor, CreateCostFunction) {
  Eigen::Matrix3d F;
  F << 0, 0, 0.1, 0, 0, 0.2, -0.1, -0.2, 0;
  const Eigen::Vector2d pp0(320, 240);
  const Eigen::Vector2d pp1(320, 240);

  std::unique_ptr<ceres::CostFunction> cost_function(
      FetzerFocalLengthCostFunctor::Create(F, pp0, pp1));
  ASSERT_NE(cost_function, nullptr);
}

// TODO(jsch): Add meaningful tests for FetzerFocalLengthSameCameraCostFunctor.

TEST(FetzerFocalLengthSameCameraCostFunctor, CreateCostFunction) {
  Eigen::Matrix3d F;
  F << 0, 0, 0.1, 0, 0, 0.2, -0.1, -0.2, 0;
  const Eigen::Vector2d pp(320, 240);

  std::unique_ptr<ceres::CostFunction> cost_function(
      FetzerFocalLengthSameCameraCostFunctor::Create(F, pp));
  ASSERT_NE(cost_function, nullptr);
}

}  // namespace
}  // namespace colmap
