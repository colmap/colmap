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

#include "colmap/geometry/similarity_transform.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {

SimilarityTransform3 TestSimilarityTransform3() {
  return SimilarityTransform3(1.23,
                              NormalizeQuaternion(Eigen::Vector4d(1, 2, 3, 4)),
                              Eigen::Vector3d(1, 2, 3));
}

TEST(SimilarityTransform3, Default) {
  const SimilarityTransform3 tform;
  EXPECT_EQ(tform.Scale(), 1);
  EXPECT_EQ(tform.Rotation(), ComposeIdentityQuaternion());
  EXPECT_EQ(tform.Translation(), Eigen::Vector3d::Zero());
}

TEST(SimilarityTransform3, Initialization) {
  const SimilarityTransform3 tform = TestSimilarityTransform3();
  const SimilarityTransform3 tform2(tform.Matrix());
  EXPECT_EQ(tform.Scale(), tform2.Scale());
  EXPECT_EQ(tform.Rotation(), tform2.Rotation());
  EXPECT_EQ(tform.Translation(), tform2.Translation());
}

void TestEstimationWithNumCoords(const size_t num_coords) {
  const SimilarityTransform3 original = TestSimilarityTransform3();

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_coords; ++i) {
    src.emplace_back(i, i + 2, i * i);
    dst.push_back(original * src.back());
  }

  SimilarityTransform3 estimated;
  EXPECT_TRUE(estimated.Estimate(src, dst));
  EXPECT_TRUE((original.Matrix() - estimated.Matrix()).norm() < 1e-6);
}

TEST(SimilarityTransform3, EstimateMinimal) { TestEstimationWithNumCoords(3); }

TEST(SimilarityTransform3, EstimateOverDetermined) {
  TestEstimationWithNumCoords(100);
}

TEST(SimilarityTransform3, EstimateDegenerate) {
  std::vector<Eigen::Vector3d> invalid_src_dst(3, Eigen::Vector3d::Zero());
  EXPECT_FALSE(
      SimilarityTransform3().Estimate(invalid_src_dst, invalid_src_dst));
}

TEST(SimilarityTransform3, Inverse) {
  const SimilarityTransform3 bFromA = TestSimilarityTransform3();
  const SimilarityTransform3 aFromB = bFromA.Inverse();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    const Eigen::Vector3d x_in_b = bFromA * x_in_a;
    EXPECT_LT((aFromB * x_in_b - x_in_a).norm(), 1e-6);
  }
}

TEST(SimilarityTransform3, ToFromFile) {
  const std::string path = CreateTestDir() + "/file.txt";
  const SimilarityTransform3 written = TestSimilarityTransform3();
  written.ToFile(path);
  const SimilarityTransform3 read = SimilarityTransform3::FromFile(path);
  EXPECT_EQ(written.Scale(), read.Scale());
  EXPECT_EQ(written.Rotation(), read.Rotation());
  EXPECT_EQ(written.Translation(), read.Translation());
}

}  // namespace colmap
