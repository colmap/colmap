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
  const Rigid3d bFromA = TestRigid3d();
  const Rigid3d aFromB = bFromA.Inverse();
  for (int i = 0; i < 100; ++i) {
    const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
    const Eigen::Vector3d x_in_b = bFromA * x_in_a;
    EXPECT_LT((aFromB * x_in_b - x_in_a).norm(), 1e-6);
  }
}

TEST(Rigid3d, ApplyNoRotation) {
  const Rigid3d bFromA(Eigen::Quaterniond::Identity(),
                       Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (bFromA * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, 4, 6)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyNoTranslation) {
  const Rigid3d bFromA(Eigen::Quaterniond(Eigen::AngleAxisd(
                           EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                       Eigen::Vector3d::Zero());
  EXPECT_LT(
      (bFromA * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(1, -3, 2)).norm(),
      1e-6);
}

TEST(Rigid3d, ApplyRotationTranslation) {
  const Rigid3d bFromA(Eigen::Quaterniond(Eigen::AngleAxisd(
                           EIGEN_PI / 2, Eigen::Vector3d::UnitX())),
                       Eigen::Vector3d(1, 2, 3));
  EXPECT_LT(
      (bFromA * Eigen::Vector3d(1, 2, 3) - Eigen::Vector3d(2, -1, 5)).norm(),
      1e-6);
}

TEST(Rigid3d, Concatenate) {
  const Rigid3d bFromA = TestRigid3d();
  const Rigid3d cFromB = TestRigid3d();
  const Rigid3d dFromC = TestRigid3d();
  const Rigid3d dFromA = dFromC * cFromB * bFromA;
  const Eigen::Vector3d x_in_a = Eigen::Vector3d::Random();
  const Eigen::Vector3d x_in_b = bFromA * x_in_a;
  const Eigen::Vector3d x_in_c = cFromB * x_in_b;
  const Eigen::Vector3d x_in_d = dFromC * x_in_c;
  EXPECT_LT((dFromA * x_in_a - x_in_d).norm(), 1e-6);
}

}  // namespace colmap
