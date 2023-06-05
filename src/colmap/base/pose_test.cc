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

#define TEST_NAME "base/pose"
#include "colmap/base/pose.h"

#include "colmap/base/projection.h"
#include "colmap/util/math.h"
#include "colmap/util/testing.h"

#include <Eigen/Core>

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestCrossProductMatrix) {
  BOOST_CHECK_EQUAL(CrossProductMatrix(Eigen::Vector3d(0, 0, 0)),
                    Eigen::Matrix3d::Zero());
  Eigen::Matrix3d ref_matrix;
  ref_matrix << 0, -3, 2, 3, 0, -1, -2, 1, 0;
  BOOST_CHECK_EQUAL(CrossProductMatrix(Eigen::Vector3d(1, 2, 3)), ref_matrix);
}

BOOST_AUTO_TEST_CASE(TestEulerAnglesX) {
  const double rx = 0.3;
  const double ry = 0;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  BOOST_CHECK_CLOSE(rx, rxx, 1e-6);
  BOOST_CHECK_CLOSE(ry, ryy, 1e-6);
  BOOST_CHECK_CLOSE(rz, rzz, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestEulerAnglesY) {
  const double rx = 0;
  const double ry = 0.3;
  const double rz = 0;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  BOOST_CHECK_CLOSE(rx, rxx, 1e-6);
  BOOST_CHECK_CLOSE(ry, ryy, 1e-6);
  BOOST_CHECK_CLOSE(rz, rzz, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestEulerAnglesZ) {
  const double rx = 0;
  const double ry = 0;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  BOOST_CHECK_CLOSE(rx, rxx, 1e-6);
  BOOST_CHECK_CLOSE(ry, ryy, 1e-6);
  BOOST_CHECK_CLOSE(rz, rzz, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestEulerAnglesXYZ) {
  const double rx = 0.1;
  const double ry = 0.2;
  const double rz = 0.3;
  double rxx, ryy, rzz;

  RotationMatrixToEulerAngles(
      EulerAnglesToRotationMatrix(rx, ry, rz), &rxx, &ryy, &rzz);

  BOOST_CHECK_CLOSE(rx, rxx, 1e-6);
  BOOST_CHECK_CLOSE(ry, ryy, 1e-6);
  BOOST_CHECK_CLOSE(rz, rzz, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestQuaternionToRotationMatrix) {
  const double rx = 0;
  const double ry = 0;
  const double rz = 0.3;
  const Eigen::Matrix3d rot_mat0 = EulerAnglesToRotationMatrix(rx, ry, rz);
  const Eigen::Matrix3d rot_mat1 =
      QuaternionToRotationMatrix(RotationMatrixToQuaternion(rot_mat0));
  BOOST_CHECK(rot_mat0.isApprox(rot_mat1));
}

BOOST_AUTO_TEST_CASE(TestComposeIdentityQuaternion) {
  BOOST_CHECK_EQUAL(ComposeIdentityQuaternion(), Eigen::Vector4d(1, 0, 0, 0));
}

BOOST_AUTO_TEST_CASE(TestNormalizeQuaternion) {
  BOOST_CHECK_EQUAL(NormalizeQuaternion(ComposeIdentityQuaternion()),
                    ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(NormalizeQuaternion(Eigen::Vector4d(2, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(NormalizeQuaternion(Eigen::Vector4d(0.5, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(NormalizeQuaternion(Eigen::Vector4d(0, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  BOOST_CHECK(
      NormalizeQuaternion(Eigen::Vector4d(1, 1, 0, 0))
          .isApprox(Eigen::Vector4d(std::sqrt(2) / 2, std::sqrt(2) / 2, 0, 0)));
  BOOST_CHECK(
      NormalizeQuaternion(Eigen::Vector4d(0.5, 0.5, 0, 0))
          .isApprox(Eigen::Vector4d(std::sqrt(2) / 2, std::sqrt(2) / 2, 0, 0)));
}

BOOST_AUTO_TEST_CASE(TestInvertQuaternion) {
  BOOST_CHECK_EQUAL(InvertQuaternion(ComposeIdentityQuaternion()),
                    Eigen::Vector4d(1, -0, -0, -0));
  BOOST_CHECK_EQUAL(InvertQuaternion(Eigen::Vector4d(2, 0, 0, 0)),
                    Eigen::Vector4d(2, -0, -0, -0));
  BOOST_CHECK_EQUAL(
      InvertQuaternion(InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4))),
      Eigen::Vector4d(0.1, 0.2, 0.3, 0.4));
}

BOOST_AUTO_TEST_CASE(TestConcatenateQuaternions) {
  BOOST_CHECK_EQUAL(ConcatenateQuaternions(ComposeIdentityQuaternion(),
                                           ComposeIdentityQuaternion()),
                    ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(ConcatenateQuaternions(Eigen::Vector4d(2, 0, 0, 0),
                                           ComposeIdentityQuaternion()),
                    ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(ConcatenateQuaternions(ComposeIdentityQuaternion(),
                                           Eigen::Vector4d(2, 0, 0, 0)),
                    ComposeIdentityQuaternion());
  BOOST_CHECK(ConcatenateQuaternions(
                  Eigen::Vector4d(0.1, 0.2, 0.3, 0.4),
                  InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4)))
                  .isApprox(ComposeIdentityQuaternion()));
  BOOST_CHECK(ConcatenateQuaternions(
                  InvertQuaternion(Eigen::Vector4d(0.1, 0.2, 0.3, 0.4)),
                  Eigen::Vector4d(0.1, 0.2, 0.3, 0.4))
                  .isApprox(ComposeIdentityQuaternion()));
}

BOOST_AUTO_TEST_CASE(TestQuaternionRotatePoint) {
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(ComposeIdentityQuaternion(),
                                          Eigen::Vector3d(0, 0, 0)),
                    Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(Eigen::Vector4d(0.1, 0, 0, 0),
                                          Eigen::Vector3d(0, 0, 0)),
                    Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(ComposeIdentityQuaternion(),
                                          Eigen::Vector3d(1, 1, 0)),
                    Eigen::Vector3d(1, 1, 0));
  BOOST_CHECK_EQUAL(QuaternionRotatePoint(Eigen::Vector4d(0.1, 0, 0, 0),
                                          Eigen::Vector3d(1, 1, 0)),
                    Eigen::Vector3d(1, 1, 0));
  BOOST_CHECK(
      QuaternionRotatePoint(
          RotationMatrixToQuaternion(EulerAnglesToRotationMatrix(M_PI, 0, 0)),
          Eigen::Vector3d(1, 1, 0))
          .isApprox(Eigen::Vector3d(1, -1, 0)));
}

BOOST_AUTO_TEST_CASE(TestAverageQuaternions) {
  std::vector<Eigen::Vector4d> qvecs;
  std::vector<double> weights;

  qvecs = {{ComposeIdentityQuaternion()}};
  weights = {1.0};
  BOOST_CHECK_EQUAL(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion()};
  weights = {2.0};
  BOOST_CHECK_EQUAL(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), ComposeIdentityQuaternion()};
  weights = {1.0, 1.0};
  BOOST_CHECK_EQUAL(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), ComposeIdentityQuaternion()};
  weights = {1.0, 2.0};
  BOOST_CHECK_EQUAL(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), Eigen::Vector4d(2, 0, 0, 0)};
  weights = {1.0, 2.0};
  BOOST_CHECK_EQUAL(AverageQuaternions(qvecs, weights),
                    ComposeIdentityQuaternion());

  qvecs = {ComposeIdentityQuaternion(), Eigen::Vector4d(1, 1, 0, 0)};
  weights = {1.0, 1.0};
  BOOST_CHECK(AverageQuaternions(qvecs, weights)
                  .isApprox(Eigen::Vector4d(0.92388, 0.382683, 0, 0), 1e-6));

  qvecs = {ComposeIdentityQuaternion(), Eigen::Vector4d(1, 1, 0, 0)};
  weights = {1.0, 2.0};
  BOOST_CHECK(AverageQuaternions(qvecs, weights)
                  .isApprox(Eigen::Vector4d(0.850651, 0.525731, 0, 0), 1e-6));
}

BOOST_AUTO_TEST_CASE(TestRotationFromUnitVectors) {
  BOOST_CHECK_EQUAL(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 1),
                                            Eigen::Vector3d(0, 0, 1)),
                    Eigen::Matrix3d::Identity());
  BOOST_CHECK_EQUAL(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 2),
                                            Eigen::Vector3d(0, 0, 2)),
                    Eigen::Matrix3d::Identity());

  Eigen::Matrix3d ref_matrix1;
  ref_matrix1 << 1, 0, 0, 0, 0, 1, 0, -1, 0;
  BOOST_CHECK_EQUAL(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 1),
                                            Eigen::Vector3d(0, 1, 0)),
                    ref_matrix1);
  BOOST_CHECK_EQUAL(ref_matrix1 * Eigen::Vector3d(0, 0, 1),
                    Eigen::Vector3d(0, 1, 0));
  BOOST_CHECK_EQUAL(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 2),
                                            Eigen::Vector3d(0, 2, 0)),
                    ref_matrix1);
  BOOST_CHECK_EQUAL(ref_matrix1 * Eigen::Vector3d(0, 0, 2),
                    Eigen::Vector3d(0, 2, 0));

  BOOST_CHECK_EQUAL(RotationFromUnitVectors(Eigen::Vector3d(0, 0, 1),
                                            Eigen::Vector3d(0, 0, -1)),
                    Eigen::Matrix3d::Identity());
}

BOOST_AUTO_TEST_CASE(TestPoseFromProjectionMatrix) {
  const Eigen::Vector4d qvec = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec(3, 4, 5);
  const Eigen::Matrix3x4d proj_matrix = ComposeProjectionMatrix(qvec, tvec);
  const Eigen::Matrix3x4d inv_proj_matrix = InvertProjectionMatrix(proj_matrix);
  const Eigen::Vector3d pose = ProjectionCenterFromMatrix(proj_matrix);
  BOOST_CHECK_CLOSE((inv_proj_matrix.rightCols<1>() - pose).norm(), 0, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestPoseFromProjectionParameters) {
  const Eigen::Vector4d qvec = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec(3, 4, 5);
  const Eigen::Matrix3x4d proj_matrix = ComposeProjectionMatrix(qvec, tvec);
  const Eigen::Matrix3x4d inv_proj_matrix = InvertProjectionMatrix(proj_matrix);
  const Eigen::Vector3d pose = ProjectionCenterFromPose(qvec, tvec);
  BOOST_CHECK((inv_proj_matrix.rightCols<1>() - pose).norm() < 1e-6);
}

BOOST_AUTO_TEST_CASE(TestComputeRelativePose) {
  Eigen::Vector4d qvec12;
  Eigen::Vector3d tvec12;

  ComputeRelativePose(ComposeIdentityQuaternion(),
                      Eigen::Vector3d(0, 0, 0),
                      ComposeIdentityQuaternion(),
                      Eigen::Vector3d(0, 0, 0),
                      &qvec12,
                      &tvec12);
  BOOST_CHECK_EQUAL(qvec12, ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(0, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(),
                      Eigen::Vector3d(0, 0, 0),
                      ComposeIdentityQuaternion(),
                      Eigen::Vector3d(1, 0, 0),
                      &qvec12,
                      &tvec12);
  BOOST_CHECK_EQUAL(qvec12, ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(1, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(),
                      Eigen::Vector3d(0, 0, 0),
                      Eigen::Vector4d(1, 1, 0, 0),
                      Eigen::Vector3d(0, 0, 0),
                      &qvec12,
                      &tvec12);
  BOOST_CHECK(qvec12.isApprox(Eigen::Vector4d(0.707107, 0.707107, 0, 0), 1e-6));
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(0, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(),
                      Eigen::Vector3d(0, 0, 0),
                      Eigen::Vector4d(1, 1, 0, 0),
                      Eigen::Vector3d(1, 0, 0),
                      &qvec12,
                      &tvec12);
  BOOST_CHECK(qvec12.isApprox(Eigen::Vector4d(0.707107, 0.707107, 0, 0), 1e-6));
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(1, 0, 0));

  ComputeRelativePose(Eigen::Vector4d(1, 1, 0, 0),
                      Eigen::Vector3d(0, 0, 0),
                      Eigen::Vector4d(1, 1, 0, 0),
                      Eigen::Vector3d(1, 0, 0),
                      &qvec12,
                      &tvec12);
  BOOST_CHECK(qvec12.isApprox(ComposeIdentityQuaternion()));
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(1, 0, 0));

  ComputeRelativePose(ComposeIdentityQuaternion(),
                      Eigen::Vector3d(0, 0, 1),
                      Eigen::Vector4d(1, 1, 0, 0),
                      Eigen::Vector3d(0, 0, 0),
                      &qvec12,
                      &tvec12);
  BOOST_CHECK(qvec12.isApprox(Eigen::Vector4d(0.707107, 0.707107, 0, 0), 1e-6));
  BOOST_CHECK(tvec12.isApprox(Eigen::Vector3d(0, 1, 0)));
}

BOOST_AUTO_TEST_CASE(TestConcatenatePoses) {
  Eigen::Vector4d qvec12;
  Eigen::Vector3d tvec12;

  ConcatenatePoses(ComposeIdentityQuaternion(),
                   Eigen::Vector3d(0, 0, 0),
                   ComposeIdentityQuaternion(),
                   Eigen::Vector3d(0, 0, 0),
                   &qvec12,
                   &tvec12);
  BOOST_CHECK_EQUAL(qvec12, ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(0, 0, 0));

  ConcatenatePoses(ComposeIdentityQuaternion(),
                   Eigen::Vector3d(0, 0, 0),
                   ComposeIdentityQuaternion(),
                   Eigen::Vector3d(0, 1, 2),
                   &qvec12,
                   &tvec12);
  BOOST_CHECK_EQUAL(qvec12, ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(0, 1, 2));

  ConcatenatePoses(ComposeIdentityQuaternion(),
                   Eigen::Vector3d(0, 1, 2),
                   ComposeIdentityQuaternion(),
                   Eigen::Vector3d(3, 4, 5),
                   &qvec12,
                   &tvec12);
  BOOST_CHECK_EQUAL(qvec12, ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(tvec12, Eigen::Vector3d(3, 5, 7));

  Eigen::Vector4d rel_qvec12;
  Eigen::Vector3d rel_tvec12;
  ComputeRelativePose(Eigen::Vector4d(1, 1, 0, 0),
                      Eigen::Vector3d(0, 1, 2),
                      Eigen::Vector4d(1, 3, 0, 0),
                      Eigen::Vector3d(3, 4, 5),
                      &rel_qvec12,
                      &rel_tvec12);
  ConcatenatePoses(Eigen::Vector4d(1, 1, 0, 0),
                   Eigen::Vector3d(0, 1, 2),
                   rel_qvec12,
                   rel_tvec12,
                   &qvec12,
                   &tvec12);
  BOOST_CHECK(
      qvec12.isApprox(NormalizeQuaternion(Eigen::Vector4d(1, 3, 0, 0))));
  BOOST_CHECK(tvec12.isApprox(Eigen::Vector3d(3, 4, 5)));
}

BOOST_AUTO_TEST_CASE(TestInvertPose) {
  Eigen::Vector4d inv_qvec;
  Eigen::Vector3d inv_tvec;
  InvertPose(ComposeIdentityQuaternion(),
             Eigen::Vector3d(0, 0, 0),
             &inv_qvec,
             &inv_tvec);
  BOOST_CHECK_EQUAL(inv_qvec, ComposeIdentityQuaternion());
  BOOST_CHECK_EQUAL(inv_tvec, Eigen::Vector3d(0, 0, 0));
  InvertPose(Eigen::Vector4d(0, 1, 2, 3),
             Eigen::Vector3d(0, 1, 2),
             &inv_qvec,
             &inv_tvec);
  Eigen::Vector4d inv_inv_qvec;
  Eigen::Vector3d inv_inv_tvec;
  InvertPose(inv_qvec, inv_tvec, &inv_inv_qvec, &inv_inv_tvec);
  BOOST_CHECK_EQUAL(inv_inv_qvec, Eigen::Vector4d(0, 1, 2, 3));
  BOOST_CHECK(inv_inv_tvec.isApprox(Eigen::Vector3d(0, 1, 2)));
}

BOOST_AUTO_TEST_CASE(TestInterpolatePose) {
  const Eigen::Vector4d qvec1 = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec1 = Eigen::Vector3d::Random();
  const Eigen::Vector4d qvec2 = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec2 = Eigen::Vector3d::Random();

  Eigen::Vector4d qveci;
  Eigen::Vector3d tveci;

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 0, &qveci, &tveci);
  BOOST_CHECK(tvec1.isApprox(tveci));

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 1, &qveci, &tveci);
  BOOST_CHECK(tvec2.isApprox(tveci));

  InterpolatePose(qvec1, tvec1, qvec2, tvec2, 0.5, &qveci, &tveci);
  BOOST_CHECK(((tvec1 + tvec2) / 2).isApprox(tveci));
}

BOOST_AUTO_TEST_CASE(TestCalculateBaseline) {
  Eigen::Vector4d qvec1(1, 0, 0, 0);
  Eigen::Vector4d qvec2(1, 0, 0, 0);

  Eigen::Vector3d tvec1(0, 0, 0);
  Eigen::Vector3d tvec2(0, 0, 1);

  const double baseline1 = CalculateBaseline(qvec1, tvec1, qvec2, tvec2).norm();
  BOOST_CHECK_CLOSE(baseline1, 1, 1e-10);

  tvec2(2) = 2;

  const double baseline2 = CalculateBaseline(qvec1, tvec1, qvec2, tvec2).norm();
  BOOST_CHECK_CLOSE(baseline2, 2, 1e-10);
}

BOOST_AUTO_TEST_CASE(TestCheckCheirality) {
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t(1, 0, 0);

  std::vector<Eigen::Vector2d> points1;
  std::vector<Eigen::Vector2d> points2;
  std::vector<Eigen::Vector3d> points3D;

  points1.emplace_back(0, 0);
  points2.emplace_back(0.1, 0);
  BOOST_CHECK(CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 1);

  points1.emplace_back(0, 0);
  points2.emplace_back(-0.1, 0);
  BOOST_CHECK(CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 1);

  points2[1][0] = 0.2;
  BOOST_CHECK(CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 2);

  points2[0][0] = -0.2;
  points2[1][0] = -0.2;
  BOOST_CHECK(!CheckCheirality(R, t, points1, points2, &points3D));
  BOOST_CHECK_EQUAL(points3D.size(), 0);
}
