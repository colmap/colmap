// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/estimators/solvers/utils.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/util/eigen_matchers.h"

#include <vector>

#include <Eigen/SVD>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SolveEpipolarConstraintMatrix, ExactRecoveryIsRankTwo) {
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 0, 0));
  const Eigen::Matrix3d E_expected = EssentialMatrixFromPose(cam2_from_cam1);

  for (const int num_points : {8, 64}) {
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(num_points, 9);
    std::vector<Eigen::Vector3d> rays1(num_points);
    std::vector<Eigen::Vector3d> rays2(num_points);
    for (int i = 0; i < num_points; ++i) {
      // Deterministic non-coplanar spread in front of both cameras (a planar
      // point set would leave the epipolar matrix under-determined).
      const Eigen::Vector3d point_in_cam1(
          (i % 8) - 3.5, ((i * 3) % 8) - 3.5, 4.0 + (i % 5) * 0.5);
      rays1[i] = point_in_cam1.normalized();
      rays2[i] = (cam2_from_cam1 * point_in_cam1).normalized();
      A.row(i) << rays2[i].x() * rays1[i].transpose(),
          rays2[i].y() * rays1[i].transpose(),
          rays2[i].z() * rays1[i].transpose();
    }

    const Eigen::Matrix3d E = SolveEpipolarConstraintMatrix(A);
    EXPECT_LT(std::min((E.normalized() - E_expected.normalized()).norm(),
                       (E.normalized() + E_expected.normalized()).norm()),
              1e-6);
    EXPECT_LT(Eigen::JacobiSVD<Eigen::Matrix3d>(E).singularValues()(2), 1e-12);
    for (int i = 0; i < num_points; ++i) {
      EXPECT_LT(std::abs(rays2[i].dot(E * rays1[i])), 1e-10);
    }
  }
}

TEST(CalibratedRays, Nominal) {
  const std::vector<Eigen::Vector2d> points = {{100.0, 0.0}, {0.0, -200.0}};
  const std::vector<Eigen::Vector3d> rays = CalibratedRays(points, 100.0);
  ASSERT_EQ(rays.size(), 2);
  EXPECT_THAT(rays[0],
              EigenMatrixNear(Eigen::Vector3d(1, 0, 1).normalized(), 1e-12));
  EXPECT_THAT(rays[1],
              EigenMatrixNear(Eigen::Vector3d(0, -2, 1).normalized(), 1e-12));
}

TEST(RaysFromCamRaysWithJac, Nominal) {
  const std::vector<CamRayWithJac> rays_with_jac = {
      {Eigen::Vector3d(1, 0, 0), Eigen::Matrix3x2d::Ones()},
      {Eigen::Vector3d(0, 1, 0), Eigen::Matrix3x2d::Zero()}};
  const std::vector<Eigen::Vector3d> rays =
      RaysFromCamRaysWithJac(rays_with_jac);
  ASSERT_EQ(rays.size(), 2);
  EXPECT_EQ(rays[0], Eigen::Vector3d(1, 0, 0));
  EXPECT_EQ(rays[1], Eigen::Vector3d(0, 1, 0));
}

}  // namespace
}  // namespace colmap
