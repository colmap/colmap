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

#include "colmap/feature/types.h"

#include "colmap/math/math.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

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

  EXPECT_EQ(keypoint, keypoint);
  EXPECT_NE(keypoint, FeatureKeypoint(1, 2, 1, 0));
}

TEST(FeatureKeypoint, Rot90) {
  FeatureKeypoint kp(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
  int w = 10, h = 20;

  FeatureKeypoint kp1 = kp;
  kp1.Rot90(1, w, h);  // 90 CCW
  EXPECT_FLOAT_EQ(kp1.x, 2.0f);
  EXPECT_FLOAT_EQ(kp1.y, 10.0f - 1.0f);
  EXPECT_FLOAT_EQ(kp1.a11, 5.0f);
  EXPECT_FLOAT_EQ(kp1.a12, 6.0f);
  EXPECT_FLOAT_EQ(kp1.a21, -3.0f);
  EXPECT_FLOAT_EQ(kp1.a22, -4.0f);

  FeatureKeypoint kp2 = kp;
  kp2.Rot90(2, w, h);  // 180 CCW
  EXPECT_FLOAT_EQ(kp2.x, 10.0f - 1.0f);
  EXPECT_FLOAT_EQ(kp2.y, 20.0f - 2.0f);
  EXPECT_FLOAT_EQ(kp2.a11, -3.0f);
  EXPECT_FLOAT_EQ(kp2.a12, -4.0f);
  EXPECT_FLOAT_EQ(kp2.a21, -5.0f);
  EXPECT_FLOAT_EQ(kp2.a22, -6.0f);

  FeatureKeypoint kp3 = kp;
  kp3.Rot90(3, w, h);  // 270 CCW
  EXPECT_FLOAT_EQ(kp3.x, 20.0f - 2.0f);
  EXPECT_FLOAT_EQ(kp3.y, 1.0f);
  EXPECT_FLOAT_EQ(kp3.a11, -5.0f);
  EXPECT_FLOAT_EQ(kp3.a12, -6.0f);
  EXPECT_FLOAT_EQ(kp3.a21, 3.0f);
  EXPECT_FLOAT_EQ(kp3.a22, 4.0f);

  FeatureKeypoint kp_identity = kp;
  kp_identity.Rot90(0, w, h);
  EXPECT_EQ(kp_identity, kp);

  FeatureKeypoint kp_identity4 = kp;
  kp_identity4.Rot90(4, w, h);
  EXPECT_EQ(kp_identity4, kp);

  FeatureKeypoint kp_neg1 = kp;
  kp_neg1.Rot90(-1, w, h);  // same as 3
  EXPECT_EQ(kp_neg1, kp3);
}

TEST(FeatureDescriptors, Nominal) {
  FeatureDescriptors descriptors(FeatureExtractorType::SIFT,
                                 FeatureDescriptorsData::Random(2, 3));
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  EXPECT_EQ(descriptors.data.rows(), 2);
  EXPECT_EQ(descriptors.data.cols(), 3);
  EXPECT_EQ(descriptors.data(0, 0), descriptors.data.data()[0]);
  EXPECT_EQ(descriptors.data(0, 1), descriptors.data.data()[1]);
  EXPECT_EQ(descriptors.data(0, 2), descriptors.data.data()[2]);
  EXPECT_EQ(descriptors.data(1, 0), descriptors.data.data()[3]);
  EXPECT_EQ(descriptors.data(1, 1), descriptors.data.data()[4]);
  EXPECT_EQ(descriptors.data(1, 2), descriptors.data.data()[5]);
}

TEST(FeatureDescriptors, SiftConversion) {
  // SIFT uses value cast (uint8 <-> float)
  const FeatureDescriptors original(FeatureExtractorType::SIFT,
                                    FeatureDescriptorsData::Random(10, 128));
  const FeatureDescriptorsFloat as_float = original.ToFloat();
  EXPECT_EQ(as_float.type, FeatureExtractorType::SIFT);
  EXPECT_EQ(as_float.data.cols(), original.data.cols());
  EXPECT_EQ(as_float.data, original.data.cast<float>());

  const FeatureDescriptors recovered = as_float.ToBytes();
  EXPECT_EQ(recovered.type, FeatureExtractorType::SIFT);
  EXPECT_EQ(recovered.data, original.data);
}

TEST(FeatureDescriptors, AlikedConversion) {
  // ALIKED uses reinterpret cast (float32 bytes <-> float)
  const FeatureDescriptors original(FeatureExtractorType::ALIKED_N16ROT,
                                    FeatureDescriptorsData::Random(10, 512));
  const FeatureDescriptorsFloat as_float = original.ToFloat();
  EXPECT_EQ(as_float.type, FeatureExtractorType::ALIKED_N16ROT);
  EXPECT_EQ(as_float.data.cols() * sizeof(float), original.data.cols());

  const FeatureDescriptors recovered = as_float.ToBytes();
  EXPECT_EQ(recovered.type, FeatureExtractorType::ALIKED_N16ROT);
  EXPECT_EQ(recovered.data, original.data);
}

TEST(FeatureMatches, Nominal) {
  FeatureMatch match;
  EXPECT_EQ(match.point2D_idx1, kInvalidPoint2DIdx);
  EXPECT_EQ(match.point2D_idx2, kInvalidPoint2DIdx);
  FeatureMatches matches(1);
  EXPECT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].point2D_idx1, kInvalidPoint2DIdx);
  EXPECT_EQ(matches[0].point2D_idx2, kInvalidPoint2DIdx);

  EXPECT_EQ(match, match);
  EXPECT_NE(match, FeatureMatch(0, 1));
}

TEST(KeypointsMatrixConversion, Roundtrip) {
  FeatureKeypoints keypoints;
  keypoints.emplace_back(1.0f, 2.0f, 3.0f, 0.5f);
  keypoints.emplace_back(4.0f, 5.0f, 1.0f, -0.3f);
  keypoints.emplace_back(10.0f, 20.0f, 0.5f, 0.0f);
  const FeatureKeypoints recovered =
      KeypointsFromMatrix(KeypointsToMatrix(keypoints));
  ASSERT_EQ(recovered.size(), keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_EQ(recovered[i].x, keypoints[i].x);
    EXPECT_EQ(recovered[i].y, keypoints[i].y);
    EXPECT_NEAR(recovered[i].ComputeScale(), keypoints[i].ComputeScale(), 1e-5);
    EXPECT_NEAR(recovered[i].ComputeOrientation(),
                keypoints[i].ComputeOrientation(),
                1e-5);
  }
}

TEST(MatchesMatrixConversion, Roundtrip) {
  FeatureMatches matches;
  matches.emplace_back(0, 5);
  matches.emplace_back(3, 7);
  matches.emplace_back(100, 200);
  EXPECT_EQ(MatchesFromMatrix(MatchesToMatrix(matches)), matches);
}

}  // namespace
}  // namespace colmap
