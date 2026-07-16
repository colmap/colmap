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

#include "colmap/math/geometric_median.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(GeometricMedian, SymmetricPointsAtCentroid) {
  const std::vector<Eigen::Vector3d> points = {
      Eigen::Vector3d(1, 0, 0),
      Eigen::Vector3d(-1, 0, 0),
      Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(0, -1, 0),
  };
  const Eigen::Vector3d median = GeometricMedian(points);
  EXPECT_NEAR(median.norm(), 0.0, 1e-6);
}

TEST(GeometricMedian, RobustToOutlier) {
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 9; ++i) {
    points.emplace_back(1.0, 0.0, 0.0);
  }
  points.emplace_back(1000.0, 0.0, 0.0);  // One gross outlier.

  const Eigen::Vector3d median = GeometricMedian(points);
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    mean += p;
  }
  mean /= static_cast<double>(points.size());

  // The median stays near the inlier cluster; the mean does not.
  EXPECT_NEAR(median.x(), 1.0, 1e-3);
  EXPECT_GT(mean.x(), 50.0);
}

}  // namespace
}  // namespace colmap
