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

#include <fstream>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {

TEST(SimilarityTransform, Default) {
  const SimilarityTransform3 tform;

  EXPECT_EQ(tform.Scale(), 1);

  EXPECT_EQ(tform.Rotation()[0], 1);
  EXPECT_EQ(tform.Rotation()[1], 0);
  EXPECT_EQ(tform.Rotation()[2], 0);
  EXPECT_EQ(tform.Rotation()[3], 0);

  EXPECT_EQ(tform.Translation()[0], 0);
  EXPECT_EQ(tform.Translation()[1], 0);
  EXPECT_EQ(tform.Translation()[2], 0);
}

TEST(SimilarityTransform, Initialization) {
  const Eigen::Vector4d qvec =
      NormalizeQuaternion(Eigen::Vector4d(0.1, 0.3, 0.2, 0.4));

  const SimilarityTransform3 tform(2, qvec, Eigen::Vector3d(100, 10, 0.5));

  EXPECT_NEAR(tform.Scale(), 2, 1e-10);

  EXPECT_NEAR(tform.Rotation()[0], qvec(0), 1e-10);
  EXPECT_NEAR(tform.Rotation()[1], qvec(1), 1e-10);
  EXPECT_NEAR(tform.Rotation()[2], qvec(2), 1e-10);
  EXPECT_NEAR(tform.Rotation()[3], qvec(3), 1e-10);

  EXPECT_NEAR(tform.Translation()[0], 100, 1e-10);
  EXPECT_NEAR(tform.Translation()[1], 10, 1e-10);
  EXPECT_NEAR(tform.Translation()[2], 0.5, 1e-10);
}

void TestEstimationWithNumCoords(const size_t num_coords) {
  const SimilarityTransform3 orig_tform(
      2, Eigen::Vector4d(0.1, 0.3, 0.2, 0.4), Eigen::Vector3d(100, 10, 0.5));

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;

  for (size_t i = 0; i < num_coords; ++i) {
    src.emplace_back(i, i + 2, i * i);
    dst.push_back(src.back());
    orig_tform.TransformPoint(&dst.back());
  }

  SimilarityTransform3 est_tform;
  EXPECT_TRUE(est_tform.Estimate(src, dst));

  EXPECT_TRUE((orig_tform.Matrix() - est_tform.Matrix()).norm() < 1e-6);

  std::vector<Eigen::Vector3d> invalid_src_dst(3, Eigen::Vector3d::Zero());
  EXPECT_FALSE(est_tform.Estimate(invalid_src_dst, invalid_src_dst));
}

TEST(SimilarityTransform, Estimation) {
  TestEstimationWithNumCoords(3);
  TestEstimationWithNumCoords(100);
}

TEST(SimilarityTransform, FromFile) {
  // Create transform file
  const std::string path = "test_from_file_transform.txt";
  {
    std::ofstream out(path);
    out << "0.0 2.0 0.0 3.0 0.0 0.0 2.0 4.0 2.0 0.0 0.0 5.0 0.0 0.0 0.0 1.0"
        << std::endl;
  }
  SimilarityTransform3 tform = SimilarityTransform3::FromFile(path);
  EXPECT_NEAR(tform.Scale(), 2.0, 1e-10);
  EXPECT_LE((tform.Translation() - Eigen::Vector3d(3.0, 4.0, 5.0)).norm(),
            1e-6);
  EXPECT_LE((tform.Rotation() - Eigen::Vector4d(-0.5, 0.5, 0.5, 0.5)).norm(),
            1e-6);
}

}  // namespace colmap
