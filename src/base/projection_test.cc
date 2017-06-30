// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define TEST_NAME "base/projection"
#include "util/testing.h"

#include <Eigen/Core>

#include "base/camera_models.h"
#include "base/pose.h"
#include "base/projection.h"
#include "util/math.h"

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

BOOST_AUTO_TEST_CASE(TestCalculateReprojectionError) {
  const Eigen::Vector4d qvec = Eigen::Vector4d::Random().normalized();
  const Eigen::Vector3d tvec = Eigen::Vector3d::Random();

  const auto proj_matrix = ComposeProjectionMatrix(qvec, tvec);

  const Eigen::Vector3d point3D = Eigen::Vector3d::Random();
  const Eigen::Vector3d point2D_h = proj_matrix * point3D.homogeneous();
  const Eigen::Vector2d point2D = point2D_h.hnormalized();

  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1, 0, 0);

  const double error1 =
      CalculateReprojectionError(point2D, point3D, proj_matrix, camera);
  BOOST_CHECK_CLOSE(error1, 0, 1e-6);

  const double error2 = CalculateReprojectionError(point2D.array() + 1, point3D,
                                                   proj_matrix, camera);
  BOOST_CHECK_CLOSE(error2, std::sqrt(2), 1e-6);
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
