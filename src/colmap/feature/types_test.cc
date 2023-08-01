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

#include "colmap/feature/types.h"

#include "colmap/math/math.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(FeatureKeypoints, Nominal) {
  FeatureKeypoint keypoint;
  EXPECT_EQ(keypoint.x, 0.0f);
  EXPECT_EQ(keypoint.y, 0.0f);
  EXPECT_EQ(keypoint.a11, 1.0f);
  EXPECT_EQ(keypoint.a12, 0.0f);
  EXPECT_EQ(keypoint.a21, 0.0f);
  EXPECT_EQ(keypoint.a22, 1.0f);

  FeatureKeypoints keypoints(1);
  EXPECT_EQ(keypoints.size(), 1);
  EXPECT_EQ(keypoints[0].x, 0.0f);
  EXPECT_EQ(keypoints[0].y, 0.0f);
  EXPECT_EQ(keypoints[0].a11, 1.0f);
  EXPECT_EQ(keypoints[0].a12, 0.0f);
  EXPECT_EQ(keypoints[0].a21, 0.0f);
  EXPECT_EQ(keypoints[0].a22, 1.0f);

  keypoint = FeatureKeypoint(1, 2);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_EQ(keypoint.a11, 1.0f);
  EXPECT_EQ(keypoint.a12, 0.0f);
  EXPECT_EQ(keypoint.a21, 0.0f);
  EXPECT_EQ(keypoint.a22, 1.0f);
  EXPECT_NEAR(keypoint.ComputeScale(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), 0.0f, 1e-6);

  keypoint = FeatureKeypoint(1, 2, 0, 0);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_EQ(keypoint.a11, 0.0f);
  EXPECT_EQ(keypoint.a12, 0.0f);
  EXPECT_EQ(keypoint.a21, 0.0f);
  EXPECT_EQ(keypoint.a22, 0.0f);

  keypoint = FeatureKeypoint(1, 2, 1, 0);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_EQ(keypoint.a11, 1.0f);
  EXPECT_EQ(keypoint.a12, 0.0f);
  EXPECT_EQ(keypoint.a21, 0.0f);
  EXPECT_EQ(keypoint.a22, 1.0f);
  EXPECT_NEAR(keypoint.ComputeScale(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), 0.0f, 1e-6);

  keypoint = FeatureKeypoint(1, 2, 1, M_PI / 2);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, -1.0f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.a22, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 1.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), M_PI / 2, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), 0.0f, 1e-6);

  keypoint = FeatureKeypoint(1, 2, 2, M_PI / 2);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, -2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a22, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), M_PI / 2, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), 0.0f, 1e-6);

  keypoint = FeatureKeypoint(1, 2, 2, M_PI);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, -2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 0.0, 1e-6);
  EXPECT_NEAR(keypoint.a22, -2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 2.0f, 1e-6);
  EXPECT_TRUE(std::abs(keypoint.ComputeOrientation() - M_PI) < 1e-6 ||
              std::abs(keypoint.ComputeOrientation() + M_PI) < 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), 0.0f, 1e-6);

  keypoint = FeatureKeypoint::FromShapeParameters(1, 2, 2, 2, M_PI, 0);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, -2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 0.0, 1e-6);
  EXPECT_NEAR(keypoint.a22, -2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 2.0f, 1e-6);
  EXPECT_TRUE(std::abs(keypoint.ComputeOrientation() - M_PI) < 1e-6 ||
              std::abs(keypoint.ComputeOrientation() + M_PI) < 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), 0.0f, 1e-6);

  keypoint = FeatureKeypoint::FromShapeParameters(1, 2, 2, 3, M_PI, 0);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, -2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 0.0, 1e-6);
  EXPECT_NEAR(keypoint.a22, -3.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 2.5f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 3.0f, 1e-6);
  EXPECT_TRUE(std::abs(keypoint.ComputeOrientation() - M_PI) < 1e-6 ||
              std::abs(keypoint.ComputeOrientation() + M_PI) < 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), 0.0f, 1e-6);

  keypoint =
      FeatureKeypoint::FromShapeParameters(1, 2, 2, 3, -M_PI / 2, M_PI / 4);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, 2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.a21, -2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a22, 2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 2.5f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 3.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), -M_PI / 2, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), M_PI / 4, 1e-6);

  keypoint =
      FeatureKeypoint::FromShapeParameters(1, 2, 2, 3, M_PI / 2, M_PI / 4);
  EXPECT_EQ(keypoint.x, 1.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, -2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a22, -2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 2.5f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 3.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), M_PI / 2, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), M_PI / 4, 1e-6);

  keypoint.Rescale(2, 2);
  EXPECT_EQ(keypoint.x, 2.0f);
  EXPECT_EQ(keypoint.y, 4.0f);
  EXPECT_NEAR(keypoint.a11, 2 * 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, 2 * -2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 2 * 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.a22, 2 * -2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 2 * 2.5f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX(), 2 * 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 2 * 3.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), M_PI / 2, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), M_PI / 4, 1e-6);

  keypoint.Rescale(1, 0.5);
  EXPECT_EQ(keypoint.x, 2.0f);
  EXPECT_EQ(keypoint.y, 2.0f);
  EXPECT_NEAR(keypoint.a11, 0.0f, 1e-6);
  EXPECT_NEAR(keypoint.a12, -2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.a21, 4.0f, 1e-6);
  EXPECT_NEAR(keypoint.a22, -2.12132025f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScale(), 3.5f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleX() - 2, 2.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeScaleY(), 3.0f, 1e-6);
  EXPECT_NEAR(keypoint.ComputeOrientation(), M_PI / 2, 1e-6);
  EXPECT_NEAR(keypoint.ComputeShear(), M_PI / 4, 1e-6);
}

TEST(FeatureDescriptors, Nominal) {
  FeatureDescriptors descriptors = FeatureDescriptors::Random(2, 3);
  EXPECT_EQ(descriptors.rows(), 2);
  EXPECT_EQ(descriptors.cols(), 3);
  EXPECT_EQ(descriptors(0, 0), descriptors.data()[0]);
  EXPECT_EQ(descriptors(0, 1), descriptors.data()[1]);
  EXPECT_EQ(descriptors(0, 2), descriptors.data()[2]);
  EXPECT_EQ(descriptors(1, 0), descriptors.data()[3]);
  EXPECT_EQ(descriptors(1, 1), descriptors.data()[4]);
  EXPECT_EQ(descriptors(1, 2), descriptors.data()[5]);
}

TEST(FeatureMatches, Nominal) {
  FeatureMatch match;
  EXPECT_EQ(match.point2D_idx1, kInvalidPoint2DIdx);
  EXPECT_EQ(match.point2D_idx2, kInvalidPoint2DIdx);
  FeatureMatches matches(1);
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].point2D_idx1, kInvalidPoint2DIdx);
  EXPECT_EQ(matches[0].point2D_idx2, kInvalidPoint2DIdx);
}

}  // namespace colmap
