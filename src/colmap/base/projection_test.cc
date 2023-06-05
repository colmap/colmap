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

#define TEST_NAME "base/projection"
#include "colmap/base/projection.h"

#include "colmap/base/camera_models.h"
#include "colmap/base/pose.h"
#include "colmap/util/math.h"
#include "colmap/util/testing.h"

#include <Eigen/Core>

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestComposeProjectionMatrix) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 1, 2);
  const Eigen::Vector4d qvec = RotationMatrixToQuaternion(R);
  const Eigen::Vector3d tvec = Eigen::Vector3d::Random();

  const auto proj_matrix1 = ComposeProjectionMatrix(qvec, tvec);
  const auto proj_matrix2 = ComposeProjectionMatrix(R, tvec);

  BOOST_CHECK((proj_matrix1 - proj_matrix2).norm() < 1e-6);
  BOOST_CHECK((proj_matrix1.leftCols<3>() - R).norm() < 1e-6);
  BOOST_CHECK_CLOSE((proj_matrix1.rightCols<1>() - tvec).norm(), 0, 1e-6);
  BOOST_CHECK_CLOSE((proj_matrix2.leftCols<3>() - R).norm(), 0, 1e-6);
  BOOST_CHECK_CLOSE((proj_matrix2.rightCols<1>() - tvec).norm(), 0, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestInvertProjectionMatrix) {
  const Eigen::Matrix3d R = EulerAnglesToRotationMatrix(0, 1, 2);
  const Eigen::Vector3d tvec = Eigen::Vector3d::Random();

  const auto proj_matrix = ComposeProjectionMatrix(R, tvec);
  const auto inv_proj_matrix = InvertProjectionMatrix(proj_matrix);
  const auto inv_inv_proj_matrix = InvertProjectionMatrix(inv_proj_matrix);

  BOOST_CHECK((proj_matrix - inv_inv_proj_matrix).norm() < 1e-6);
}

BOOST_AUTO_TEST_CASE(TestComputeClosestRotationMatrix) {
  const Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  BOOST_CHECK_LT((ComputeClosestRotationMatrix(A) - A).norm(), 1e-6);
  BOOST_CHECK_LT((ComputeClosestRotationMatrix(2 * A) - A).norm(), 1e-6);
}

BOOST_AUTO_TEST_CASE(TestDecomposeProjectionMatrix) {
  for (int i = 1; i < 100; ++i) {
    Eigen::Matrix3d ref_K = i * Eigen::Matrix3d::Identity();
    ref_K(0, 2) = i;
    ref_K(1, 2) = 2 * i;
    const Eigen::Matrix3d ref_R = EulerAnglesToRotationMatrix(i, 2 * i, 3 * i);
    const Eigen::Vector3d ref_T = Eigen::Vector3d::Random();
    const Eigen::Matrix3x4d ref_P =
        ref_K * ComposeProjectionMatrix(ref_R, ref_T);
    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    DecomposeProjectionMatrix(ref_P, &K, &R, &T);
    BOOST_CHECK(ref_K.isApprox(K, 1e-6));
    BOOST_CHECK(ref_R.isApprox(R, 1e-6));
    BOOST_CHECK(ref_T.isApprox(T, 1e-6));
  }
}

BOOST_AUTO_TEST_CASE(TestCalculateSquaredReprojectionError) {
  const Eigen::Vector4d qvec = ComposeIdentityQuaternion();
  const Eigen::Vector3d tvec = Eigen::Vector3d::Zero();

  const auto proj_matrix = ComposeProjectionMatrix(qvec, tvec);

  const Eigen::Vector3d point3D = Eigen::Vector3d::Random().cwiseAbs();
  const Eigen::Vector3d point2D_h = proj_matrix * point3D.homogeneous();
  const Eigen::Vector2d point2D = point2D_h.hnormalized();

  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 0, 0);

  const double error1 =
      CalculateSquaredReprojectionError(point2D, point3D, qvec, tvec, camera);
  BOOST_CHECK_EQUAL(error1, 0);

  const double error2 =
      CalculateSquaredReprojectionError(point2D, point3D, proj_matrix, camera);
  BOOST_CHECK_GE(error2, 0);
  BOOST_CHECK_LT(error2, 1e-6);

  const double error3 = CalculateSquaredReprojectionError(
      point2D.array() + 1, point3D, qvec, tvec, camera);
  BOOST_CHECK_CLOSE(error3, 2, 1e-6);

  const double error4 = CalculateSquaredReprojectionError(
      point2D.array() + 1, point3D, proj_matrix, camera);
  BOOST_CHECK_CLOSE(error4, 2, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestCalculateAngularError) {
  const Eigen::Vector4d qvec = ComposeIdentityQuaternion();
  const Eigen::Vector3d tvec = Eigen::Vector3d(0, 0, 0);

  const auto proj_matrix = ComposeProjectionMatrix(qvec, tvec);
  Camera camera;
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  camera.Params() = {1, 0, 0};

  const double error1 = CalculateAngularError(
      Eigen::Vector2d(0, 0), Eigen::Vector3d(0, 0, 1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error1, 0, 1e-6);

  const double error2 = CalculateAngularError(
      Eigen::Vector2d(0, 0), Eigen::Vector3d(0, 1, 1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error2, M_PI / 4, 1e-6);

  const double error3 = CalculateAngularError(
      Eigen::Vector2d(0, 0), Eigen::Vector3d(0, 5, 5), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error3, M_PI / 4, 1e-6);

  const double error4 = CalculateAngularError(
      Eigen::Vector2d(1, 0), Eigen::Vector3d(0, 0, 1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error4, M_PI / 4, 1e-6);

  const double error5 = CalculateAngularError(
      Eigen::Vector2d(2, 0), Eigen::Vector3d(0, 0, 1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error5, 1.10714872, 1e-6);

  const double error6 = CalculateAngularError(
      Eigen::Vector2d(2, 0), Eigen::Vector3d(1, 0, 1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error6, 1.10714872 - M_PI / 4, 1e-6);

  const double error7 = CalculateAngularError(
      Eigen::Vector2d(2, 0), Eigen::Vector3d(5, 0, 5), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error7, 1.10714872 - M_PI / 4, 1e-6);

  const double error8 = CalculateAngularError(
      Eigen::Vector2d(1, 0), Eigen::Vector3d(-1, 0, 1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error8, M_PI / 2, 1e-6);

  const double error9 = CalculateAngularError(
      Eigen::Vector2d(1, 0), Eigen::Vector3d(-1, 0, 0), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error9, M_PI * 3 / 4, 1e-6);

  const double error10 = CalculateAngularError(
      Eigen::Vector2d(1, 0), Eigen::Vector3d(-1, 0, -1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error10, M_PI, 1e-6);

  const double error11 = CalculateAngularError(
      Eigen::Vector2d(1, 0), Eigen::Vector3d(0, 0, -1), proj_matrix, camera);
  BOOST_CHECK_CLOSE(error11, M_PI * 3 / 4, 1e-6);
}

BOOST_AUTO_TEST_CASE(TestCalculateDepth) {
  const Eigen::Vector4d qvec(1, 0, 0, 0);
  const Eigen::Vector3d tvec(0, 0, 0);
  const auto proj_matrix = ComposeProjectionMatrix(qvec, tvec);

  // In the image plane
  const double depth1 = CalculateDepth(proj_matrix, Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK_CLOSE(depth1, 0, 1e-10);
  const double depth2 = CalculateDepth(proj_matrix, Eigen::Vector3d(0, 2, 0));
  BOOST_CHECK_CLOSE(depth2, 0, 1e-10);

  // Infront of camera
  const double depth3 = CalculateDepth(proj_matrix, Eigen::Vector3d(0, 0, 1));
  BOOST_CHECK_CLOSE(depth3, 1, 1e-10);

  // Behind camera
  const double depth4 = CalculateDepth(proj_matrix, Eigen::Vector3d(0, 0, -1));
  BOOST_CHECK_CLOSE(depth4, -1, 1e-10);
}

BOOST_AUTO_TEST_CASE(TestHasPointPositiveDepth) {
  const Eigen::Vector4d qvec(1, 0, 0, 0);
  const Eigen::Vector3d tvec(0, 0, 0);
  const auto proj_matrix = ComposeProjectionMatrix(qvec, tvec);

  // In the image plane
  const bool check1 =
      HasPointPositiveDepth(proj_matrix, Eigen::Vector3d(0, 0, 0));
  BOOST_CHECK(!check1);
  const bool check2 =
      HasPointPositiveDepth(proj_matrix, Eigen::Vector3d(0, 2, 0));
  BOOST_CHECK(!check2);

  // Infront of camera
  const bool check3 =
      HasPointPositiveDepth(proj_matrix, Eigen::Vector3d(0, 0, 1));
  BOOST_CHECK(check3);

  // Behind camera
  const bool check4 =
      HasPointPositiveDepth(proj_matrix, Eigen::Vector3d(0, 0, -1));
  BOOST_CHECK(!check4);
}
