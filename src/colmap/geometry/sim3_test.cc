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

#include "colmap/geometry/sim3.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {

Sim3d TestSim3d() {
  return Sim3d(1.23,
                              NormalizeQuaternion(Eigen::Vector4d(1, 2, 3, 4)),
                              Eigen::Vector3d(1, 2, 3));
}

TEST(Sim3d, Default) {
  const Sim3d tform;
  EXPECT_EQ(tform.Scale(), 1);
  EXPECT_EQ(tform.Rotation(), ComposeIdentityQuaternion());
  EXPECT_EQ(tform.Translation(), Eigen::Vector3d::Zero());
}

TEST(Sim3d, Initialization) {
  const Sim3d tform = TestSim3d();
  const Sim3d tform2(tform.Matrix());
  EXPECT_EQ(tform.Scale(), tform2.Scale());
  EXPECT_EQ(tform.Rotation(), tform2.Rotation());
  EXPECT_EQ(tform.Translation(), tform2.Translation());
}

void TestEstimationWithNumCoords(const size_t num_coords) {
  const Sim3d original = TestSim3d();

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_coords; ++i) {
    src.emplace_back(i, i + 2, i * i);
    dst.push_back(original * src.back());
  }

  Sim3d estimated;
  EXPECT_TRUE(estimated.Estimate(src, dst));
  EXPECT_TRUE((original.Matrix() - estimated.Matrix()).norm() < 1e-6);
}

TEST(Sim3d, EstimateMinimal) { TestEstimationWithNumCoords(3); }

TEST(Sim3d, EstimateOverDetermined) {
  TestEstimationWithNumCoords(100);
}

TEST(Sim3d, EstimateDegenerate) {
  std::vector<Eigen::Vector3d> invalid_src_dst(3, Eigen::Vector3d::Zero());
  EXPECT_FALSE(
      Sim3d().Estimate(invalid_src_dst, invalid_src_dst));
}

TEST(Sim3d, Inverse) {
  const Sim3d bFromA = TestSim3d();
  const Sim3d aFromB = bFromA.Inverse();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    const Eigen::Vector3d x_in_b = bFromA * x_in_a;
    EXPECT_LT((aFromB * x_in_b - x_in_a).norm(), 1e-6);
  }
}

TEST(Sim3d, ToFromFile) {
  const std::string path = CreateTestDir() + "/file.txt";
  const Sim3d written = TestSim3d();
  written.ToFile(path);
  const Sim3d read = Sim3d::FromFile(path);
  EXPECT_EQ(written.Scale(), read.Scale());
  EXPECT_EQ(written.Rotation(), read.Rotation());
  EXPECT_EQ(written.Translation(), read.Translation());
}

}  // namespace colmap
