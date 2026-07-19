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

#include "colmap/geometry/essential_matrix.h"

#include "colmap/geometry/pose.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/math/random_eigen.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/eigen_matchers.h"

#include <limits>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(DecomposeEssentialMatrix, Nominal) {
  const Rigid3d cam2_from_cam1(RandomEigenQuaterniond(),
                               Eigen::Vector3d(0.5, 1, 1).normalized());
  const Eigen::Matrix3d cam2_from_cam1_rot_mat =
      cam2_from_cam1.rotation().toRotationMatrix();
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  Eigen::Vector3d t;
  DecomposeEssentialMatrix(E, &R1, &R2, &t);

  EXPECT_TRUE((R1 - cam2_from_cam1_rot_mat).norm() < 1e-10 ||
              (R2 - cam2_from_cam1_rot_mat).norm() < 1e-10);
  EXPECT_TRUE((t - cam2_from_cam1.translation()).norm() < 1e-10 ||
              (t + cam2_from_cam1.translation()).norm() < 1e-10);
}

TEST(EssentialMatrixFromPose, Nominal) {
  EXPECT_EQ(EssentialMatrixFromPose(Rigid3d(Eigen::Quaterniond::Identity(),
                                            Eigen::Vector3d(0, 0, 1))),
            (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
  EXPECT_EQ(EssentialMatrixFromPose(Rigid3d(Eigen::Quaterniond::Identity(),
                                            Eigen::Vector3d(0, 0, 2))),
            (Eigen::MatrixXd(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 0).finished());
}

TEST(PoseFromEssentialMatrix, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  std::vector<Eigen::Vector3d> rays1(4);
  std::vector<Eigen::Vector3d> rays2(4);
  for (size_t i = 0; i < points3D.size(); ++i) {
    rays1[i] = (cam1_from_world * points3D[i]).normalized();
    rays2[i] = (cam2_from_world * points3D[i]).normalized();
  }

  Rigid3d cam2_from_cam1_est;
  std::vector<int> valid_indices;
  PoseFromEssentialMatrix(E, rays1, rays2, &cam2_from_cam1_est, &valid_indices);

  EXPECT_EQ(valid_indices.size(), 4);

  EXPECT_THAT(cam2_from_cam1_est, Rigid3dNear(cam2_from_cam1));
}

TEST(FindOptimalImageObservations, Nominal) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Eigen::Matrix3d E =
      EssentialMatrixFromPose(cam2_from_world * Inverse(cam1_from_world));

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  // Test if perfect projection is equivalent to optimal image observations.
  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector2d point1 =
        (cam1_from_world * points3D[i]).hnormalized();
    const Eigen::Vector2d point2 =
        (cam2_from_world * points3D[i]).hnormalized();
    Eigen::Vector2d optimal_point1;
    Eigen::Vector2d optimal_point2;
    FindOptimalImageObservations(
        E, point1, point2, &optimal_point1, &optimal_point2);
    EXPECT_THAT(point1, EigenMatrixNear(optimal_point1));
    EXPECT_THAT(point2, EigenMatrixNear(optimal_point2));
  }
}

TEST(EpipoleFromEssentialMatrix, Nominal) {
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(0, 0, -1).normalized());
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  const Eigen::Vector3d left_epipole = EpipoleFromEssentialMatrix(E, true);
  const Eigen::Vector3d right_epipole = EpipoleFromEssentialMatrix(E, false);
  EXPECT_THAT(left_epipole, EigenMatrixNear(Eigen::Vector3d(0, 0, 1)));
  EXPECT_THAT(right_epipole, EigenMatrixNear(Eigen::Vector3d(0, 0, 1)));
}

TEST(InvertEssentialMatrix, Nominal) {
  for (size_t i = 1; i < 10; ++i) {
    const Rigid3d cam2_from_cam1(
        Eigen::Quaterniond(EulerAnglesToRotationMatrix(0, 0.1, 0)),
        Eigen::Vector3d(0, 0, i).normalized());
    const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);
    const Eigen::Matrix3d inv_inv_E =
        InvertEssentialMatrix(InvertEssentialMatrix(E));
    EXPECT_THAT(E, EigenMatrixNear(inv_inv_E));
  }
}

TEST(FundamentalFromEssentialMatrix, Nominal) {
  const Eigen::Matrix3d E = EssentialMatrixFromPose(
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>()));
  const Eigen::Matrix3d K1 =
      (Eigen::Matrix3d() << 2, 0, 1, 0, 3, 2, 0, 0, 1).finished();
  const Eigen::Matrix3d K2 =
      (Eigen::Matrix3d() << 3, 0, 2, 0, 4, 1, 0, 0, 1).finished();
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(K2, E, K1);
  const Eigen::Vector3d x(3, 2, 1);
  EXPECT_THAT(K2.transpose().inverse() * E * x,
              EigenMatrixNear(Eigen::Vector3d(F * K1 * x)));
  EXPECT_THAT(E * K1.inverse() * x,
              EigenMatrixNear(Eigen::Vector3d(K2.transpose() * F * x)));
}

TEST(EssentialFromFundamentalMatrix, Nominal) {
  const Eigen::Matrix3d E = EssentialMatrixFromPose(
      Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>()));
  const Eigen::Matrix3d K1 =
      (Eigen::Matrix3d() << 2, 0, 1, 0, 3, 2, 0, 0, 1).finished();
  const Eigen::Matrix3d K2 =
      (Eigen::Matrix3d() << 3, 0, 2, 0, 4, 1, 0, 0, 1).finished();
  const Eigen::Matrix3d F = FundamentalFromEssentialMatrix(K2, E, K1);
  EXPECT_THAT(EssentialFromFundamentalMatrix(K2, F, K1),
              EigenMatrixNear(E, 1e-6));
}

TEST(ComputeSquaredSampsonError, Nominal) {
  std::vector<Eigen::Vector2d> points1;
  points1.emplace_back(0, 0);
  points1.emplace_back(0, 0);
  points1.emplace_back(0, 0);
  std::vector<Eigen::Vector2d> points2;
  points2.emplace_back(2, 0);
  points2.emplace_back(2, 1);
  points2.emplace_back(2, 2);

  const Eigen::Matrix3d E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));

  std::vector<double> residuals;
  ComputeSquaredSampsonError(points1, points2, E, &residuals);

  EXPECT_EQ(residuals.size(), 3);
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0.5);
  EXPECT_EQ(residuals[2], 2);
}

TEST(ComputeSquaredSampsonErrorWithCheirality, AllCorrespondencesInFront) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Eigen::Matrix3d E =
      EssentialMatrixFromPose(cam2_from_world * Inverse(cam1_from_world));

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  std::vector<Eigen::Vector3d> rays1(points3D.size());
  std::vector<Eigen::Vector3d> rays2(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    rays1[i] = (cam1_from_world * points3D[i]).normalized();
    rays2[i] = (cam2_from_world * points3D[i]).normalized();
  }

  // All correspondences triangulate in front of both cameras, so enforcing
  // cheirality must leave every residual identical to the plain Sampson error.
  std::vector<double> sampson_residuals;
  ComputeSquaredSampsonError(rays1, rays2, E, &sampson_residuals);
  std::vector<double> cheiral_residuals;
  ComputeSquaredSampsonErrorWithCheirality(rays1, rays2, E, &cheiral_residuals);

  ASSERT_EQ(cheiral_residuals.size(), sampson_residuals.size());
  for (size_t i = 0; i < sampson_residuals.size(); ++i) {
    EXPECT_EQ(cheiral_residuals[i], sampson_residuals[i]);
  }
}

TEST(ComputeSquaredSampsonErrorWithCheirality, CorrespondenceBehindCamera) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Eigen::Matrix3d E =
      EssentialMatrixFromPose(cam2_from_world * Inverse(cam1_from_world));

  std::vector<Eigen::Vector3d> points3D(4);
  points3D[0] = Eigen::Vector3d(0, 0, 1);
  points3D[1] = Eigen::Vector3d(0, 0.1, 1);
  points3D[2] = Eigen::Vector3d(0.1, 0, 1);
  points3D[3] = Eigen::Vector3d(0.1, 0.1, 1);

  std::vector<Eigen::Vector3d> rays1(points3D.size());
  std::vector<Eigen::Vector3d> rays2(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    rays1[i] = (cam1_from_world * points3D[i]).normalized();
    rays2[i] = (cam2_from_world * points3D[i]).normalized();
  }

  // Negating a ray keeps it on the same epipolar line (the Sampson error is
  // invariant to the sign of the ray) but flips its triangulated depth, so it
  // ends up behind the cameras.
  rays2[1] = -rays2[1];

  // The plain Sampson error stays finite (and near zero) for the flipped
  // correspondence because it ignores cheirality.
  std::vector<double> sampson_residuals;
  ComputeSquaredSampsonError(rays1, rays2, E, &sampson_residuals);
  EXPECT_LT(sampson_residuals[1], 1e-10);

  // Enforcing cheirality rejects the flipped correspondence with an infinite
  // residual while leaving the remaining ones equal to the Sampson error.
  std::vector<double> cheiral_residuals;
  ComputeSquaredSampsonErrorWithCheirality(rays1, rays2, E, &cheiral_residuals);
  ASSERT_EQ(cheiral_residuals.size(), 4);
  EXPECT_EQ(cheiral_residuals[1], std::numeric_limits<double>::max());
  EXPECT_EQ(cheiral_residuals[0], sampson_residuals[0]);
  EXPECT_EQ(cheiral_residuals[2], sampson_residuals[2]);
  EXPECT_EQ(cheiral_residuals[3], sampson_residuals[3]);
}

namespace {

// A pinhole geometry to exercise the tangent Sampson error against a known
// closed-form answer.
constexpr double kFocal = 650.0;
constexpr double kPrincipalX = 512.0;
constexpr double kPrincipalY = 384.0;

Eigen::Vector2d PinholeImgFromCam(const Eigen::Vector3d& cam_point) {
  return Eigen::Vector2d(kFocal * cam_point.x() / cam_point.z() + kPrincipalX,
                         kFocal * cam_point.y() / cam_point.z() + kPrincipalY);
}

// Normalized image plane representative (u, v, 1) of a pixel.
Eigen::Vector3d PinholeNormalizedFromImg(const Eigen::Vector2d& image_point) {
  return Eigen::Vector3d((image_point.x() - kPrincipalX) / kFocal,
                         (image_point.y() - kPrincipalY) / kFocal,
                         1.0);
}

// d(u, v, 1) / d(x, y) for a pinhole: constant and diagonal.
Eigen::Matrix<double, 3, 2> PinholeNormalizedJacobian() {
  Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
  J(0, 0) = 1.0 / kFocal;
  J(1, 1) = 1.0 / kFocal;
  return J;
}

// d(unit ray) / d(x, y) for a pinhole, via the normalization quotient rule.
Eigen::Matrix<double, 3, 2> PinholeUnitRayJacobian(
    const Eigen::Vector3d& normalized) {
  const double norm = normalized.norm();
  const Eigen::Matrix3d dnormalize =
      (Eigen::Matrix3d::Identity() -
       normalized * normalized.transpose() / (norm * norm)) /
      norm;
  return dnormalize * PinholeNormalizedJacobian();
}

}  // namespace

// With the normalized image plane representative (u, v, 1), whose Jacobian
// w.r.t. pixels is the constant 1/f, the tangent Sampson error is *exactly*
// f^2 times the classical Sampson error. This is an algebraic identity, so it
// pins down the whole formula - numerator, both gradient chains, and the
// denominator - to machine precision.
TEST(ComputeSquaredTangentSampsonError, PinholeMatchesScaledSampsonExactly) {
  const Rigid3d cam2_from_cam1(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(0.15, Eigen::Vector3d(0.3, 1, 0.2).normalized())),
      Eigen::Vector3d(1.0, 0.2, 0.1).normalized());
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  const Eigen::Matrix<double, 3, 2> J_norm = PinholeNormalizedJacobian();

  // Deliberately includes badly mismatched pairs: the identity is exact for
  // arbitrary inputs, not only for near-inliers.
  for (const double x1 : {20.0, 512.0, 1000.0}) {
    for (const double y1 : {30.0, 384.0, 740.0}) {
      for (const double x2 : {45.0, 512.0, 980.0}) {
        for (const double y2 : {60.0, 384.0, 700.0}) {
          const Eigen::Vector3d m1 =
              PinholeNormalizedFromImg(Eigen::Vector2d(x1, y1));
          const Eigen::Vector3d m2 =
              PinholeNormalizedFromImg(Eigen::Vector2d(x2, y2));

          const double tangent_sampson =
              ComputeSquaredTangentSampsonError(m1, J_norm, m2, J_norm, E);
          const double scaled_sampson =
              kFocal * kFocal * ComputeSquaredSampsonError(m1, m2, E);

          ASSERT_GT(scaled_sampson, 0.0);
          EXPECT_LE(std::abs(tangent_sampson - scaled_sampson) / scaled_sampson,
                    1e-14);
        }
      }
    }
  }
}

// With unit bearing vectors the agreement is only first order: rescaling the
// homogeneous representative by a function of the measurements changes the
// Sampson approximation by a term proportional to the residual itself. The
// relative discrepancy is therefore expected to shrink linearly as the
// correspondence approaches the epipolar variety. This test documents that
// behaviour rather than hiding it, since it is the reason the error is called
// an approximation.
TEST(ComputeSquaredTangentSampsonError, UnitRaysAgreeToFirstOrder) {
  const Rigid3d cam2_from_cam1(
      Eigen::Quaterniond(
          Eigen::AngleAxisd(0.15, Eigen::Vector3d(0.3, 1, 0.2).normalized())),
      Eigen::Vector3d(1.0, 0.2, 0.1).normalized());
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  const Eigen::Vector3d point3D_in_cam1(0.35, -0.2, 4.0);
  const Eigen::Vector2d image_point1 = PinholeImgFromCam(point3D_in_cam1);
  const Eigen::Vector2d image_point2 =
      PinholeImgFromCam(cam2_from_cam1 * point3D_in_cam1);

  // Displace image 2 perpendicular to the epipolar line of image_point1, so
  // the offset is entirely "epipolar error" rather than a slide along the
  // line. In normalized coordinates the line is l = E * m1 and the pixel-space
  // gradient of the constraint is proportional to (l.x, l.y).
  const Eigen::Vector3d m1 = PinholeNormalizedFromImg(image_point1);
  const Eigen::Vector3d epipolar_line2 = E * m1;
  const Eigen::Vector2d perpendicular = epipolar_line2.head<2>().normalized();

  double prev_rel_diff = std::numeric_limits<double>::max();
  double prev_ratio = 0.0;
  for (const double offset : {1.0, 0.1, 0.01, 0.001}) {
    const Eigen::Vector3d m2 =
        PinholeNormalizedFromImg(image_point2 + offset * perpendicular);

    const double tangent_sampson =
        ComputeSquaredTangentSampsonError(m1.normalized(),
                                          PinholeUnitRayJacobian(m1),
                                          m2.normalized(),
                                          PinholeUnitRayJacobian(m2),
                                          E);
    const double scaled_sampson =
        kFocal * kFocal * ComputeSquaredSampsonError(m1, m2, E);

    // The residual is a squared distance, so it must scale quadratically with
    // the displacement. Note sqrt(residual) is strictly below `offset`: Sampson
    // measures distance to the epipolar variety in the joint 4-D measurement
    // space, which distributes the correction over both images rather than
    // charging it entirely to the one that was displaced.
    EXPECT_LT(std::sqrt(tangent_sampson), offset);
    const double ratio = tangent_sampson / (offset * offset);
    if (prev_ratio > 0.0) {
      EXPECT_LE(std::abs(ratio - prev_ratio) / prev_ratio, 1e-2);
    }
    prev_ratio = ratio;

    const double rel_diff =
        std::abs(tangent_sampson - scaled_sampson) / scaled_sampson;
    // Each tenfold reduction of the residual must reduce the discrepancy,
    // confirming it is a first-order effect and not a constant bias.
    EXPECT_LT(rel_diff, prev_rel_diff);
    prev_rel_diff = rel_diff;
  }
  // At a milli-pixel residual the two formulations are indistinguishable.
  EXPECT_LT(prev_rel_diff, 1e-5);
}

TEST(ComputeSquaredTangentSampsonError, DegenerateDenominatorReturnsMax) {
  const Eigen::Matrix<double, 3, 2> J_norm = PinholeNormalizedJacobian();
  EXPECT_EQ(ComputeSquaredTangentSampsonError(Eigen::Vector3d(0, 0, 1),
                                              J_norm,
                                              Eigen::Vector3d(0, 0, 1),
                                              J_norm,
                                              Eigen::Matrix3d::Zero()),
            std::numeric_limits<double>::max());
}

TEST(ComputeSquaredTangentSampsonError, VectorOverloadAndCheirality) {
  const Rigid3d cam1_from_world;
  const Rigid3d cam2_from_world(Eigen::Quaterniond::Identity(),
                                Eigen::Vector3d(1, 0, 0).normalized());
  const Rigid3d cam2_from_cam1 = cam2_from_world * Inverse(cam1_from_world);
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  const std::vector<Eigen::Vector3d> points3D = {Eigen::Vector3d(0, 0, 1),
                                                 Eigen::Vector3d(0, 0.1, 1),
                                                 Eigen::Vector3d(0.1, 0, 1),
                                                 Eigen::Vector3d(0.1, 0.1, 1)};

  std::vector<Eigen::Vector3d> rays1(points3D.size());
  std::vector<Eigen::Vector3d> rays2(points3D.size());
  std::vector<Eigen::Matrix<double, 3, 2>> J_rays1(points3D.size());
  std::vector<Eigen::Matrix<double, 3, 2>> J_rays2(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    const Eigen::Vector3d cam1_point = cam1_from_world * points3D[i];
    const Eigen::Vector3d cam2_point = cam2_from_world * points3D[i];
    rays1[i] = cam1_point.normalized();
    rays2[i] = cam2_point.normalized();
    J_rays1[i] = PinholeUnitRayJacobian(cam1_point / cam1_point.z());
    J_rays2[i] = PinholeUnitRayJacobian(cam2_point / cam2_point.z());
  }

  std::vector<double> residuals;
  ComputeSquaredTangentSampsonError(
      rays1, J_rays1, rays2, J_rays2, E, &residuals);
  ASSERT_EQ(residuals.size(), points3D.size());
  for (const double residual : residuals) {
    EXPECT_LT(residual, 1e-16);
  }

  // Flipping one correspondence behind both cameras leaves the epipolar
  // constraint satisfied but must be rejected once cheirality is enforced.
  rays1[1] = -rays1[1];
  rays2[1] = -rays2[1];

  std::vector<double> plain_residuals;
  ComputeSquaredTangentSampsonError(
      rays1, J_rays1, rays2, J_rays2, E, &plain_residuals);
  EXPECT_LT(plain_residuals[1], 1e-16);

  std::vector<CamRayWithJac> cam_rays1(points3D.size());
  std::vector<CamRayWithJac> cam_rays2(points3D.size());
  for (size_t i = 0; i < points3D.size(); ++i) {
    cam_rays1[i] = {rays1[i], J_rays1[i]};
    cam_rays2[i] = {rays2[i], J_rays2[i]};
  }

  std::vector<double> cheiral_residuals;
  ComputeSquaredTangentSampsonErrorWithCheirality(
      cam_rays1, cam_rays2, E, &cheiral_residuals);
  ASSERT_EQ(cheiral_residuals.size(), points3D.size());
  EXPECT_EQ(cheiral_residuals[1], std::numeric_limits<double>::max());
  EXPECT_EQ(cheiral_residuals[0], plain_residuals[0]);
  EXPECT_EQ(cheiral_residuals[2], plain_residuals[2]);
  EXPECT_EQ(cheiral_residuals[3], plain_residuals[3]);
}

}  // namespace
}  // namespace colmap
