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
#include "colmap/math/random.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {

Sim3d TestSim3d() {
  return Sim3d(RandomUniformReal<double>(0.1, 10),
               Eigen::Quaterniond::UnitRandom(),
               Eigen::Vector3d::Random());
}

TEST(Sim3d, Default) {
  const Sim3d tform;
  EXPECT_EQ(tform.scale, 1);
  EXPECT_EQ(tform.rotation.coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(tform.translation, Eigen::Vector3d::Zero());
}

TEST(Sim3d, Inverse) {
  const Sim3d b_from_a = TestSim3d();
  const Sim3d a_from_b = Inverse(b_from_a);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
    EXPECT_LT((a_from_b * x_in_b - x_in_a).norm(), 1e-6);
  }
}

TEST(Sim3d, Matrix) {
  const Sim3d b_from_a = TestSim3d();
  const Eigen::Matrix3x4d b_from_a_mat = b_from_a.ToMatrix();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    EXPECT_LT((b_from_a * x_in_a - b_from_a_mat * x_in_a.homogeneous()).norm(),
              1e-6);
  }
}

TEST(Sim3d, ApplyScaleOnly) {
  const Sim3d b_from_a(
      2, Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, 4, 6)).norm(),
      1e-6);
}

TEST(Sim3d, ApplyTranslationOnly) {
  const Sim3d b_from_a(
      1, Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, 4, 6)).norm(),
      1e-6);
}

TEST(Sim3d, ApplyRotationOnly) {
  const Sim3d b_from_a(1,
                       Eigen::Quaterniond(Eigen::AngleAxisd(
                           EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                       Eigen::Vector3d::Zero());
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(1, -3, 2)).norm(),
      1e-6);
}

TEST(Sim3d, ApplyScaleRotationTranslation) {
  const Sim3d b_from_a(2,
                       Eigen::Quaterniond(Eigen::AngleAxisd(
                           EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                       Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(3, -4, 7)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyChain) {
  const Sim3d b_from_a = TestSim3d();
  const Sim3d c_from_b = TestSim3d();
  const Sim3d d_from_c = TestSim3d();
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_EQ((d_from_c * (c_from_b * (b_from_a * x_in_a))), x_in_d);
}

TEST(Sim3d, Compose) {
  const Sim3d b_from_a = TestSim3d();
  const Sim3d c_from_b = TestSim3d();
  const Sim3d d_from_c = TestSim3d();
  const Sim3d d_from_a = d_from_c * c_from_b * b_from_a;
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_LT((d_from_a * x_in_a - x_in_d).norm(), 1e-6);
}

void TestEstimationWithNumCoords(const size_t num_coords) {
  const Sim3d gt_tgt_from_src = TestSim3d();

  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < num_coords; ++i) {
    src.emplace_back(i, i + 2, i * i);
    dst.push_back(gt_tgt_from_src * src.back());
  }

  Sim3d tgt_from_src;
  EXPECT_TRUE(tgt_from_src.Estimate(src, dst));
  EXPECT_NEAR(gt_tgt_from_src.scale, tgt_from_src.scale, 1e-6);
  EXPECT_LT(gt_tgt_from_src.rotation.angularDistance(tgt_from_src.rotation),
            1e-6);
  EXPECT_LT((gt_tgt_from_src.translation - tgt_from_src.translation).norm(),
            1e-6);
}

TEST(Sim3d, EstimateMinimal) { TestEstimationWithNumCoords(3); }

TEST(Sim3d, EstimateOverDetermined) { TestEstimationWithNumCoords(100); }

TEST(Sim3d, EstimateDegenerate) {
  std::vector<Eigen::Vector3d> invalid_src_dst(3, Eigen::Vector3d::Zero());
  EXPECT_FALSE(Sim3d().Estimate(invalid_src_dst, invalid_src_dst));
}

TEST(Sim3d, ToFromFile) {
  const std::string path = CreateTestDir() + "/file.txt";
  const Sim3d written = TestSim3d();
  written.ToFile(path);
  const Sim3d read = Sim3d::FromFile(path);
  EXPECT_EQ(written.scale, read.scale);
  EXPECT_EQ(written.rotation.coeffs(), read.rotation.coeffs());
  EXPECT_EQ(written.translation, read.translation);
}

}  // namespace colmap
