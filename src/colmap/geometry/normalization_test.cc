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

#include "colmap/geometry/normalization.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ComputeBoundingBoxAndCentroid, SingleCoord) {
  const auto [bbox, centroid] =
      ComputeBoundingBoxAndCentroid(0, 1, {1}, {2}, {3});
  EXPECT_EQ(bbox.min(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(bbox.max(), Eigen::Vector3d(1, 2, 3));
  EXPECT_EQ(centroid, Eigen::Vector3d(1, 2, 3));
}

TEST(ComputeBoundingBoxAndCentroid, TwoCoords) {
  const auto [bbox, centroid] =
      ComputeBoundingBoxAndCentroid(0, 1, {2, -1}, {3, -2}, {4, -3});
  EXPECT_EQ(bbox.min(), Eigen::Vector3d(-1, -2, -3));
  EXPECT_EQ(bbox.max(), Eigen::Vector3d(2, 3, 4));
  EXPECT_EQ(centroid, Eigen::Vector3d(0.5, 0.5, 0.5));
}

TEST(ComputeBoundingBoxAndCentroid, ThreeCoords) {
  const auto [bbox, centroid] =
      ComputeBoundingBoxAndCentroid(0, 1, {2, -1, 5}, {3, -2, 5}, {4, -3, 5});
  EXPECT_EQ(bbox.min(), Eigen::Vector3d(-1, -2, -3));
  EXPECT_EQ(bbox.max(), Eigen::Vector3d(5, 5, 5));
  EXPECT_THAT(centroid, EigenMatrixNear(Eigen::Vector3d(2, 2, 2), 1e-6));
}

TEST(ComputeBoundingBoxAndCentroid, FiveCoords) {
  const auto [bbox1, centroid1] =
      ComputeBoundingBoxAndCentroid(0,
                                    1,
                                    {2, -1, 5, 100, -100},
                                    {3, -2, 5, 100, -100},
                                    {4, -3, 5, 100, -100});
  EXPECT_EQ(bbox1.min(), Eigen::Vector3d(-100, -100, -100));
  EXPECT_EQ(bbox1.max(), Eigen::Vector3d(100, 100, 100));
  EXPECT_THAT(centroid1, EigenMatrixNear(Eigen::Vector3d(1.2, 1.2, 1.2), 1e-6));

  const auto [bbox2, centroid2] =
      ComputeBoundingBoxAndCentroid(0.3,
                                    0.7,
                                    {2, -1, 5, 100, -100},
                                    {3, -2, 5, 100, -100},
                                    {4, -3, 5, 100, -100});
  EXPECT_EQ(bbox2.min(), Eigen::Vector3d(-1, -2, -3));
  EXPECT_EQ(bbox2.max(), Eigen::Vector3d(5, 5, 5));
  EXPECT_THAT(centroid2, EigenMatrixNear(Eigen::Vector3d(2, 2, 2), 1e-6));
}

}  // namespace
}  // namespace colmap
