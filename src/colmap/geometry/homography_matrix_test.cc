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

#include "colmap/geometry/homography_matrix.h"

#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/logging.h"

#include <cmath>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Note that the test case values are obtained from OpenCV.
TEST(DecomposeHomographyMatrix, Nominal) {
  Eigen::Matrix3d H;
  H << 2.649157564634028, 4.583875997496426, 70.694447785121326,
      -1.072756858861583, 3.533262150437228, 1513.656999614321649,
      0.001303887589576, 0.003042206876298, 1;
  H *= 3;

  Eigen::Matrix3d K;
  K << 640, 0, 320, 0, 640, 240, 0, 0, 1;

  std::vector<Rigid3d> cams2_from_cams1;
  std::vector<Eigen::Vector3d> normals;
  DecomposeHomographyMatrix(H, K, K, &cams2_from_cams1, &normals);

  EXPECT_EQ(cams2_from_cams1.size(), 4);
  EXPECT_EQ(normals.size(), 4);

  Eigen::Matrix3d ref_rotation;
  ref_rotation << 0.43307983549125, 0.545749113549648, -0.717356090899523,
      -0.85630229674426, 0.497582023798831, -0.138414255706431,
      0.281404038139784, 0.67421809131173, 0.682818960388909;
  const Eigen::Vector3d ref_translation(
      1.826751712278038, 1.264718492450820, 0.195080809998819);
  const Eigen::Vector3d ref_normal(
      -0.244875830334816, -0.480857890778889, -0.841909446789566);

  bool ref_solution_exists = false;
  for (size_t i = 0; i < 4; ++i) {
    const double kEps = 1e-6;
    if ((cams2_from_cams1[i].rotation.toRotationMatrix() - ref_rotation)
                .norm() < kEps &&
        (cams2_from_cams1[i].translation - ref_translation).norm() < kEps &&
        (normals[i] - ref_normal).norm() < kEps) {
      ref_solution_exists = true;
      break;
    }
  }
  EXPECT_TRUE(ref_solution_exists);
}

TEST(DecomposeHomographyMatrix, Random) {
  constexpr int kNumIters = 100;

  const double epsilon = 1e-6;

  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();

  for (int i = 0; i < kNumIters; ++i) {
    const Eigen::Matrix3d H = Eigen::Matrix3d::Random();

    if (std::abs(H.determinant()) < epsilon) {
      continue;
    }

    std::vector<Rigid3d> cams2_from_cams1;
    std::vector<Eigen::Vector3d> normals;
    DecomposeHomographyMatrix(
        H, identity, identity, &cams2_from_cams1, &normals);

    EXPECT_EQ(cams2_from_cams1.size(), 4);
    EXPECT_EQ(normals.size(), 4);

    // Test that each candidate rotation is a rotation
    for (const Rigid3d& cam2_from_cam1 : cams2_from_cams1) {
      const Eigen::Matrix3d orthog_error =
          cam2_from_cam1.rotation.toRotationMatrix().transpose() *
              cam2_from_cam1.rotation.toRotationMatrix() -
          identity;

      // Check that cam2_from_cam1.rotation.toRotationMatrix() is an orthognal
      // matrix
      EXPECT_LT(orthog_error.lpNorm<Eigen::Infinity>(), epsilon);

      // Check determinant is 1
      EXPECT_NEAR(cam2_from_cam1.rotation.toRotationMatrix().determinant(),
                  1.0,
                  epsilon);
    }
  }
}

TEST(PoseFromHomographyMatrix, Nominal) {
  const Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
  const Eigen::Quaterniond ref_rotation =
      Eigen::Quaterniond(1, 0.1, 0.2, 0.3).normalized();
  const Eigen::Vector3d ref_translation(1, 0, 0);
  const Eigen::Vector3d ref_normal(0, 0, -1);
  const Eigen::Matrix3d H = HomographyMatrixFromPose(
      K1, K2, ref_rotation.matrix(), ref_translation, ref_normal, 1);

  std::vector<Eigen::Vector3d> rays1;
  rays1.push_back(Eigen::Vector3d(0.1, 0.1, 1).normalized());
  rays1.push_back(Eigen::Vector3d(0.4, 0.1, 1).normalized());
  rays1.push_back(Eigen::Vector3d(0.1, 0.4, 1).normalized());
  rays1.push_back(Eigen::Vector3d(0.4, 0.4, 1).normalized());
  rays1.push_back(Eigen::Vector3d(0.0, 0.0, 1).normalized());

  std::vector<Eigen::Vector3d> rays2;
  for (const auto& ray1 : rays1) {
    const Eigen::Vector3d ray2 = H * ray1;
    CHECK_GT(ray2.z(), 0);
    rays2.push_back(ray2.normalized());
  }

  Rigid3d cam2_from_cam1;
  Eigen::Vector3d normal;
  std::vector<Eigen::Vector3d> points3D;
  PoseFromHomographyMatrix(
      H, K1, K2, rays1, rays2, &cam2_from_cam1, &normal, &points3D);

  EXPECT_THAT(
      Rigid3d(cam2_from_cam1.rotation, cam2_from_cam1.translation.normalized()),
      Rigid3dNear(
          Rigid3d(ref_rotation, ref_translation.normalized()), 1e-6, 1e-6));

  EXPECT_THAT(normal, EigenMatrixNear(ref_normal, 1e-5));
  EXPECT_EQ(points3D.size(), rays1.size());
}

TEST(HomographyMatrixFromPose, PureRotation) {
  const Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(0, 0, 0);
  const Eigen::Vector3d n(-1, 0, 0);
  const double d = 1;
  const Eigen::Matrix3d H = HomographyMatrixFromPose(K1, K2, R, t, n, d);
  EXPECT_EQ(H, Eigen::Matrix3d::Identity());
}

TEST(HomographyMatrixFromPose, PlanarScene) {
  const Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K2 = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(1, 0, 0);
  const Eigen::Vector3d n(-1, 0, 0);
  const double d = 1;
  const Eigen::Matrix3d H = HomographyMatrixFromPose(K1, K2, R, t, n, d);
  Eigen::Matrix3d H_ref;
  H_ref << 2, 0, 0, 0, 1, 0, 0, 0, 1;
  EXPECT_EQ(H, H_ref);
}

}  // namespace
}  // namespace colmap
