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

#include "colmap/geometry/bbox.h"

#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ComputeEqualPartsBboxes, Split1x1x2) {
  const Eigen::AlignedBox3d bbox(Eigen::Vector3d(0, 0, 0),
                                 Eigen::Vector3d(1, 1, 1));
  const Eigen::Vector3i split(1, 1, 1);

  const auto bboxes = ComputeEqualPartsBboxes(bbox, split);

  ASSERT_EQ(bboxes.size(), 1);
  EXPECT_THAT(bboxes[0].min(),
              EigenMatrixNear(Eigen::Vector3d(0.0, 0.0, 0.0), 1e-10));
  EXPECT_THAT(bboxes[0].max(),
              EigenMatrixNear(Eigen::Vector3d(1.0, 1.0, 1.0), 1e-10));
}

TEST(ComputeEqualPartsBboxes, Split2x2x2) {
  const Eigen::AlignedBox3d bbox(Eigen::Vector3d(0, 0, 0),
                                 Eigen::Vector3d(2, 2, 2));
  const Eigen::Vector3i split(2, 2, 2);

  const auto bboxes = ComputeEqualPartsBboxes(bbox, split);

  ASSERT_EQ(bboxes.size(), 8);

  for (const auto& sub_bbox : bboxes) {
    Eigen::Vector3d diag = sub_bbox.diagonal();
    EXPECT_THAT(diag, EigenMatrixNear(Eigen::Vector3d(1.0, 1.0, 1.0), 1e-10));
  }

  Eigen::AlignedBox3d covered;
  for (const auto& sub_bbox : bboxes) {
    covered.extend(sub_bbox);
  }
  EXPECT_THAT(covered.min(), EigenMatrixNear(bbox.min(), 1e-10));
  EXPECT_THAT(covered.max(), EigenMatrixNear(bbox.max(), 1e-10));
}

TEST(ComputeEqualPartsBboxes, Asymmetric) {
  const Eigen::AlignedBox3d bbox(Eigen::Vector3d(0, 0, 0),
                                 Eigen::Vector3d(6, 4, 2));
  const Eigen::Vector3i split(3, 2, 1);

  const auto bboxes = ComputeEqualPartsBboxes(bbox, split);

  ASSERT_EQ(bboxes.size(), 6);

  for (const auto& sub_bbox : bboxes) {
    Eigen::Vector3d diag = sub_bbox.diagonal();
    EXPECT_THAT(diag, EigenMatrixNear(Eigen::Vector3d(2.0, 2.0, 2.0), 1e-10));
  }
}

TEST(ComputeEqualPartsBboxes, WithOffset) {
  const Eigen::AlignedBox3d bbox(Eigen::Vector3d(10, 20, 30),
                                 Eigen::Vector3d(20, 30, 40));
  const Eigen::Vector3i split(2, 2, 2);

  const auto bboxes = ComputeEqualPartsBboxes(bbox, split);

  ASSERT_EQ(bboxes.size(), 8);

  // Check that sub-boxes cover the original box
  Eigen::AlignedBox3d covered;
  for (const auto& sub_bbox : bboxes) {
    covered.extend(sub_bbox);
  }
  EXPECT_THAT(covered.min(), EigenMatrixNear(bbox.min(), 1e-10));
  EXPECT_THAT(covered.max(), EigenMatrixNear(bbox.max(), 1e-10));
}

}  // namespace
}  // namespace colmap
