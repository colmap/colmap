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

#include "colmap/scene/camera.h"

#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(Camera, Empty) {
  Camera camera;
  EXPECT_EQ(camera.camera_id, kInvalidCameraId);
  EXPECT_EQ(camera.model_id, CameraModelId::kInvalid);
  EXPECT_EQ(camera.ModelName(), "");
  EXPECT_EQ(camera.width, 0);
  EXPECT_EQ(camera.height, 0);
  EXPECT_FALSE(camera.has_prior_focal_length);
  EXPECT_THROW(camera.FocalLengthIdxs(), std::domain_error);
  EXPECT_THROW(camera.ParamsInfo(), std::domain_error);
  EXPECT_EQ(camera.ParamsToString(), "");
  EXPECT_EQ(camera.params.size(), 0);
  EXPECT_EQ(camera.params.size(), 0);
  EXPECT_EQ(camera.params.data(), camera.params.data());
}

TEST(Camera, Equals) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  Camera other = camera;
  EXPECT_EQ(camera, other);
  camera.SetFocalLength(2.);
  EXPECT_NE(camera, other);
  other.SetFocalLength(2.);
  EXPECT_EQ(camera, other);
}

TEST(Camera, Print) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  std::ostringstream stream;
  stream << camera;
  EXPECT_EQ(stream.str(),
            "Camera(camera_id=1, model=SIMPLE_PINHOLE, width=1, height=1, "
            "params=[1, 0.5, 0.5] (f, cx, cy))");
}

TEST(Camera, CameraId) {
  Camera camera;
  EXPECT_EQ(camera.camera_id, kInvalidCameraId);
  camera.camera_id = 1;
  EXPECT_EQ(camera.camera_id, 1);
}

TEST(Camera, FocalLength) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.FocalLengthX(), 1.0);
  EXPECT_EQ(camera.FocalLengthY(), 1.0);
  camera.SetFocalLength(2.0);
  EXPECT_EQ(camera.FocalLength(), 2.0);
  EXPECT_EQ(camera.FocalLengthX(), 2.0);
  EXPECT_EQ(camera.FocalLengthY(), 2.0);
  camera =
      Camera::CreateFromModelId(1, PinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.FocalLengthX(), 1.0);
  EXPECT_EQ(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthX(2.0);
  EXPECT_EQ(camera.FocalLengthX(), 2.0);
  EXPECT_EQ(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthY(2.0);
  EXPECT_EQ(camera.FocalLengthX(), 2.0);
  EXPECT_EQ(camera.FocalLengthY(), 2.0);
}

TEST(Camera, PrincipalPoint) {
  Camera camera =
      Camera::CreateFromModelId(1, PinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  camera.SetPrincipalPointX(2.0);
  EXPECT_EQ(camera.PrincipalPointX(), 2.0);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  camera.SetPrincipalPointY(2.0);
  EXPECT_EQ(camera.PrincipalPointX(), 2.0);
  EXPECT_EQ(camera.PrincipalPointY(), 2.0);
}

TEST(Camera, ParamIdxs) {
  Camera camera;
  EXPECT_THROW(camera.FocalLengthIdxs(), std::domain_error);
  EXPECT_THROW(camera.PrincipalPointIdxs(), std::domain_error);
  EXPECT_THROW(camera.ExtraParamsIdxs(), std::domain_error);
  camera.model_id = FullOpenCVCameraModel::model_id;
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 2);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 8);
}

TEST(Camera, CalibrationMatrix) {
  Camera camera =
      Camera::CreateFromModelId(1, PinholeCameraModel::model_id, 1.0, 1, 1);
  const Eigen::Matrix3d K = camera.CalibrationMatrix();
  Eigen::Matrix3d K_ref;
  K_ref << 1, 0, 0.5, 0, 1, 0.5, 0, 0, 1;
  EXPECT_EQ(K, K_ref);
}

TEST(Camera, ParamsInfo) {
  Camera camera;
  EXPECT_THROW(camera.ParamsInfo(), std::domain_error);
  camera.model_id = SimpleRadialCameraModel::model_id;
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy, k");
}

TEST(Camera, ParamsToString) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
}

TEST(Camera, ParamsFromString) {
  Camera camera;
  camera.model_id = SimplePinholeCameraModel::model_id;
  EXPECT_TRUE(camera.SetParamsFromString("1, 0.5, 0.5"));
  const std::vector<double> params{1.0, 0.5, 0.5};
  EXPECT_EQ(camera.params, params);
  EXPECT_FALSE(camera.SetParamsFromString("1, 0.5"));
  EXPECT_EQ(camera.params, params);
}

TEST(Camera, VerifyParams) {
  Camera camera;
  EXPECT_THROW(camera.VerifyParams(), std::domain_error);
  camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.VerifyParams());
  camera.params.resize(2);
  EXPECT_FALSE(camera.VerifyParams());
}

TEST(Camera, IsUndistorted) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera = Camera::CreateFromModelId(
      1, SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.params = {1.0, 0.5, 0.5, 0.005};
  EXPECT_FALSE(camera.IsUndistorted());
  camera = Camera::CreateFromModelId(1, RadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.params = {1.0, 0.5, 0.5, 0.0, 0.005};
  EXPECT_FALSE(camera.IsUndistorted());
  camera = Camera::CreateFromModelId(1, OpenCVCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.params = {1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.001};
  EXPECT_FALSE(camera.IsUndistorted());
  camera =
      Camera::CreateFromModelId(1, FullOpenCVCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.params = {
      1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001};
  EXPECT_FALSE(camera.IsUndistorted());
}

TEST(Camera, HasBogusParams) {
  Camera camera;
  EXPECT_THROW(camera.HasBogusParams(0.0, 0.0, 0.0), std::domain_error);
  camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 1.0));
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 0.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.99, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(1.01, 1.1, 1.0));
  camera = Camera::CreateFromModelId(
      1, SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 1.0));
  camera.params[3] = 1.01;
  EXPECT_TRUE(camera.HasBogusParams(0.1, 1.1, 1.0));
  camera.params[3] = -0.5;
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 1.0));
  camera.params[3] = -1.01;
  EXPECT_TRUE(camera.HasBogusParams(0.1, 1.1, 1.0));
}

TEST(Camera, CreateFromModelId) {
  Camera camera = Camera::CreateFromModelId(
      1, SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.camera_id, 1);
  EXPECT_EQ(camera.model_id, SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.width, 1);
  EXPECT_EQ(camera.height, 1);
  EXPECT_FALSE(camera.has_prior_focal_length);
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_FALSE(camera.HasBogusParams(0.1, 2.0, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.5, 1.0));
  EXPECT_EQ(camera.params.size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.params.size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(Camera, CreateFromModelName) {
  Camera camera = Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.camera_id, 1);
  EXPECT_EQ(camera.model_id, SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.width, 1);
  EXPECT_EQ(camera.height, 1);
  EXPECT_FALSE(camera.has_prior_focal_length);
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1, 0.5, 0.5");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_FALSE(camera.HasBogusParams(0.1, 2.0, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.5, 1.0));
  EXPECT_EQ(camera.params.size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.params.size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(Camera, CamFromImg) {
  Camera camera;
  EXPECT_THROW(camera.CamFromImg(Eigen::Vector2d::Zero()), std::domain_error);
  camera = Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.0, 0.0))(0), -0.5);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.0, 0.0))(1), -0.5);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.5, 0.5))(0), 0.0);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.5, 0.5))(1), 0.0);
}

TEST(Camera, CamFromImgThreshold) {
  Camera camera;
  EXPECT_THROW(camera.CamFromImgThreshold(0), std::domain_error);
  camera = Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CamFromImgThreshold(0), 0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 1);
  camera.SetFocalLength(2.0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 0.5);
  camera = Camera::CreateFromModelName(1, "PINHOLE", 1.0, 1, 1);
  camera.SetFocalLengthY(3.0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 0.5);
}

TEST(Camera, ImgFromCam) {
  Camera camera;
  EXPECT_THROW(camera.ImgFromCam(Eigen::Vector2d::Zero()), std::domain_error);
  camera = Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(0.0, 0.0))(0), 0.5);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(0.0, 0.0))(1), 0.5);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(-0.5, -0.5))(0), 0.0);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(-0.5, -0.5))(1), 0.0);
}

TEST(Camera, Rescale) {
  Camera camera = Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  EXPECT_EQ(camera.width, 2);
  EXPECT_EQ(camera.height, 2);
  EXPECT_EQ(camera.FocalLength(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 1);
  EXPECT_EQ(camera.PrincipalPointY(), 1);

  camera = Camera::CreateFromModelName(1, "PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  EXPECT_EQ(camera.width, 2);
  EXPECT_EQ(camera.height, 2);
  EXPECT_EQ(camera.FocalLengthX(), 2);
  EXPECT_EQ(camera.FocalLengthY(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 1);
  EXPECT_EQ(camera.PrincipalPointY(), 1);

  camera = Camera::CreateFromModelName(1, "PINHOLE", 1.0, 2, 2);
  camera.Rescale(0.5);
  EXPECT_EQ(camera.width, 1);
  EXPECT_EQ(camera.height, 1);
  EXPECT_EQ(camera.FocalLengthX(), 0.5);
  EXPECT_EQ(camera.FocalLengthY(), 0.5);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);

  camera = Camera::CreateFromModelName(1, "PINHOLE", 1.0, 2, 2);
  camera.Rescale(1, 1);
  EXPECT_EQ(camera.width, 1);
  EXPECT_EQ(camera.height, 1);
  EXPECT_EQ(camera.FocalLengthX(), 0.5);
  EXPECT_EQ(camera.FocalLengthY(), 0.5);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);

  camera = Camera::CreateFromModelName(1, "PINHOLE", 1.0, 2, 2);
  camera.Rescale(4, 4);
  EXPECT_EQ(camera.width, 4);
  EXPECT_EQ(camera.height, 4);
  EXPECT_EQ(camera.FocalLengthX(), 2);
  EXPECT_EQ(camera.FocalLengthY(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 2);
  EXPECT_EQ(camera.PrincipalPointY(), 2);
}

}  // namespace
}  // namespace colmap
