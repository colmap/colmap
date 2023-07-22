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

#include "colmap/geometry/rigid3.h"

#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {

Rigid3d TestRigid3d() {
  return Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
}

TEST(Rigid3d, Default) {
  const Rigid3d tform;
  EXPECT_EQ(tform.rotation.coeffs(), Eigen::Quaterniond::Identity().coeffs());
  EXPECT_EQ(tform.translation, Eigen::Vector3d::Zero());
}

TEST(Rigid3d, Inverse) {
  const Rigid3d b_from_a = TestRigid3d();
  const Rigid3d a_from_b = Inverse(b_from_a);
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
    EXPECT_LT((a_from_b * x_in_b - x_in_a).norm(), 1e-6);
  }
}

TEST(Rigid3d, Matrix) {
  const Rigid3d b_from_a = TestRigid3d();
  const Eigen::Matrix3x4d b_from_a_mat = b_from_a.ToMatrix();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    EXPECT_LT((b_from_a * x_in_a - b_from_a_mat * x_in_a.homogeneous()).norm(),
              1e-6);
  }
}

TEST(Rigid3d, ApplyNoRotation) {
  const Rigid3d b_from_a(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, 4, 6)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyNoTranslation) {
  const Rigid3d b_from_a(Eigen::Quaterniond(Eigen::AngleAxisd(
                             EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                         Eigen::Vector3d::Zero());
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(1, -3, 2)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyRotationTranslation) {
  const Rigid3d b_from_a(Eigen::Quaterniond(Eigen::AngleAxisd(
                             EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                         Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (b_from_a * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, -1, 5)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyChain) {
  const Rigid3d b_from_a = TestRigid3d();
  const Rigid3d c_from_b = TestRigid3d();
  const Rigid3d d_from_c = TestRigid3d();
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_EQ((d_from_c * (c_from_b * (b_from_a * x_in_a))), x_in_d);
}

TEST(Rigid3d, Compose) {
  const Rigid3d b_from_a = TestRigid3d();
  const Rigid3d c_from_b = TestRigid3d();
  const Rigid3d d_from_c = TestRigid3d();
  const Rigid3d d_from_a = d_from_c * c_from_b * b_from_a;
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = b_from_a * x_in_a;
  const Eigen::Vector3d x_in_c = c_from_b * x_in_b;
  const Eigen::Vector3d x_in_d = d_from_c * x_in_c;
  EXPECT_LT((d_from_a * x_in_a - x_in_d).norm(), 1e-6);
}

}  // namespace colmap
