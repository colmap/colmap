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

#include "colmap/scene/projection.h"

#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"

#include <cmath>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CalculateSquaredReprojectionError, Nominal) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());
  const Eigen::Matrix3x4d cam_from_world_mat = cam_from_world.ToMatrix();

  const Eigen::Vector3d point3D = Eigen::Vector3d::Random().cwiseAbs();
  const Eigen::Vector3d point2D_h = cam_from_world_mat * point3D.homogeneous();
  const Eigen::Vector2d point2D = point2D_h.hnormalized();

  Camera camera =
      Camera::CreateFromModelId(1, SimplePinholeCameraModel::model_id, 1, 0, 0);

  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D, point3D, cam_from_world, camera),
              0,
              1e-6);
  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D, point3D, cam_from_world_mat, camera),
              0,
              1e-6);

  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D.array() + 1, point3D, cam_from_world, camera),
              2,
              1e-6);
  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  point2D.array() + 1, point3D, cam_from_world_mat, camera),
              2,
              1e-6);
}

TEST(CalculateSquaredReprojectionError, Spherical) {
  const Camera camera = Camera::CreateFromModelId(
      1, EquirectangularCameraModel::model_id, /*focal_length=*/0.0, 1000, 500);
  const Rigid3d cam_from_world;
  const double px_per_rad =
      static_cast<double>(camera.width) / (2.0 * EIGEN_PI);

  // Exact observations have ~0 error, including a back-hemisphere match that
  // straddles the azimuth seam (observed at x = 0, point projects to x = w),
  // which a raw pixel error would penalize by ~width^2.
  for (const Eigen::Vector3d& cam_point : {Eigen::Vector3d(0, 0, 1),
                                           Eigen::Vector3d(0, 0, -1),
                                           Eigen::Vector3d(1, 0, 0)}) {
    const Eigen::Vector2d img_point = *camera.ImgFromCam(cam_point);
    EXPECT_NEAR(CalculateSquaredReprojectionError(
                    img_point, cam_point, cam_from_world, camera),
                0.0,
                1e-9);
  }

  // Check that the error is continuous across the seam.
  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  Eigen::Vector2d(0.0, camera.height / 2.0),
                  Eigen::Vector3d(0, 0, -1),
                  cam_from_world,
                  camera),
              0.0,
              1e-9);

  // Pole invariance: the same angular offset yields the same squared error at
  // the equator and near the pole. Raw equirectangular pixel error would
  // diverge towards the pole, dropping otherwise-valid observations.
  constexpr double kOffset = 0.01;  // radians
  const Eigen::Vector3d equator_point(0, 0, 1);
  const Eigen::Vector2d equator_obs = *camera.ImgFromCam(
      Eigen::Vector3d(std::sin(kOffset), 0, std::cos(kOffset)));
  const Eigen::Vector3d pole_point(0, -1, 0);
  const Eigen::Vector2d pole_obs = *camera.ImgFromCam(
      Eigen::Vector3d(0, -std::cos(kOffset), std::sin(kOffset)));
  const double expected = (kOffset * px_per_rad) * (kOffset * px_per_rad);
  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  equator_obs, equator_point, cam_from_world, camera),
              expected,
              1e-6);
  EXPECT_NEAR(CalculateSquaredReprojectionError(
                  pole_obs, pole_point, cam_from_world, camera),
              expected,
              1e-6);
}

TEST(CalculateAngularReprojectionError, Nominal) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());
  const Eigen::Matrix3x4d cam_from_world_mat = cam_from_world.ToMatrix();

  Camera camera;
  camera.model_id = SimplePinholeCameraModel::model_id;
  camera.params = {1, 0, 0};

  const double error1 =
      CalculateAngularReprojectionError(Eigen::Vector2d(0, 0),
                                        Eigen::Vector3d(0, 0, 1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error1, 0, 1e-6);

  const double error2 =
      CalculateAngularReprojectionError(Eigen::Vector2d(0, 0),
                                        Eigen::Vector3d(0, 1, 1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error2, EIGEN_PI / 4, 1e-6);

  const double error3 =
      CalculateAngularReprojectionError(Eigen::Vector2d(0, 0),
                                        Eigen::Vector3d(0, 5, 5),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error3, EIGEN_PI / 4, 1e-6);

  const double error4 =
      CalculateAngularReprojectionError(Eigen::Vector2d(1, 0),
                                        Eigen::Vector3d(0, 0, 1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error4, EIGEN_PI / 4, 1e-6);

  const double error5 =
      CalculateAngularReprojectionError(Eigen::Vector2d(2, 0),
                                        Eigen::Vector3d(0, 0, 1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error5, 1.10714872, 1e-6);

  const double error6 =
      CalculateAngularReprojectionError(Eigen::Vector2d(2, 0),
                                        Eigen::Vector3d(1, 0, 1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error6, 1.10714872 - EIGEN_PI / 4, 1e-6);

  const double error7 =
      CalculateAngularReprojectionError(Eigen::Vector2d(2, 0),
                                        Eigen::Vector3d(5, 0, 5),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error7, 1.10714872 - EIGEN_PI / 4, 1e-6);

  const double error8 =
      CalculateAngularReprojectionError(Eigen::Vector2d(1, 0),
                                        Eigen::Vector3d(-1, 0, 1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error8, EIGEN_PI / 2, 1e-6);

  const double error9 =
      CalculateAngularReprojectionError(Eigen::Vector2d(1, 0),
                                        Eigen::Vector3d(-1, 0, 0),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error9, EIGEN_PI * 3 / 4, 1e-6);

  const double error10 =
      CalculateAngularReprojectionError(Eigen::Vector2d(1, 0),
                                        Eigen::Vector3d(-1, 0, -1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error10, EIGEN_PI, 1e-6);

  const double error11 =
      CalculateAngularReprojectionError(Eigen::Vector2d(1, 0),
                                        Eigen::Vector3d(0, 0, -1),
                                        cam_from_world_mat,
                                        camera);
  EXPECT_NEAR(error11, EIGEN_PI * 3 / 4, 1e-6);
}

TEST(HasPointPositiveDepth, Nominal) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d::Zero());
  const Eigen::Matrix3x4d cam_from_world_mat = cam_from_world.ToMatrix();

  // In the image plane
  const bool check1 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 0, 0));
  EXPECT_FALSE(check1);
  const bool check2 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 2, 0));
  EXPECT_FALSE(check2);

  // Infront of camera
  const bool check3 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 0, 1));
  EXPECT_TRUE(check3);

  // Behind camera
  const bool check4 =
      HasPointPositiveDepth(cam_from_world_mat, Eigen::Vector3d(0, 0, -1));
  EXPECT_FALSE(check4);
}

}  // namespace
}  // namespace colmap
