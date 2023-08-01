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

#include "colmap/scene/camera.h"

#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {

TEST(Camera, Empty) {
  Camera camera;
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  EXPECT_EQ(camera.ModelId(), kInvalidCameraModelId);
  EXPECT_EQ(camera.ModelName(), "");
  EXPECT_EQ(camera.Width(), 0);
  EXPECT_EQ(camera.Height(), 0);
  EXPECT_FALSE(camera.HasPriorFocalLength());
  EXPECT_THROW(camera.FocalLengthIdxs(), std::domain_error);
  EXPECT_THROW(camera.ParamsInfo(), std::domain_error);
  EXPECT_EQ(camera.ParamsToString(), "");
  EXPECT_EQ(camera.NumParams(), 0);
  EXPECT_EQ(camera.Params().size(), 0);
  EXPECT_EQ(camera.ParamsData(), camera.Params().data());
}

TEST(Camera, CameraId) {
  Camera camera;
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  camera.SetCameraId(1);
  EXPECT_EQ(camera.CameraId(), 1);
}

TEST(Camera, ModelId) {
  Camera camera;
  EXPECT_EQ(camera.ModelId(), kInvalidCameraModelId);
  EXPECT_EQ(camera.ModelName(), "");
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  EXPECT_EQ(camera.ModelId(),
            static_cast<int>(SimplePinholeCameraModel::model_id));
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.NumParams(), SimplePinholeCameraModel::num_params);
  camera.SetModelIdFromName("SIMPLE_RADIAL");
  EXPECT_EQ(camera.ModelId(),
            static_cast<int>(SimpleRadialCameraModel::model_id));
  EXPECT_EQ(camera.ModelName(), "SIMPLE_RADIAL");
  EXPECT_EQ(camera.NumParams(), SimpleRadialCameraModel::num_params);
}

TEST(Camera, WidthHeight) {
  Camera camera;
  EXPECT_EQ(camera.Width(), 0);
  EXPECT_EQ(camera.Height(), 0);
  camera.SetWidth(1);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 0);
  camera.SetHeight(1);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
}

TEST(Camera, FocalLength) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.FocalLength(), 1.0);
  camera.SetFocalLength(2.0);
  EXPECT_EQ(camera.FocalLength(), 2.0);
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.FocalLengthX(), 1.0);
  EXPECT_EQ(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthX(2.0);
  EXPECT_EQ(camera.FocalLengthX(), 2.0);
  EXPECT_EQ(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthY(2.0);
  EXPECT_EQ(camera.FocalLengthX(), 2.0);
  EXPECT_EQ(camera.FocalLengthY(), 2.0);
}

TEST(Camera, PriorFocalLength) {
  Camera camera;
  EXPECT_FALSE(camera.HasPriorFocalLength());
  camera.SetPriorFocalLength(true);
  EXPECT_TRUE(camera.HasPriorFocalLength());
  camera.SetPriorFocalLength(false);
  EXPECT_FALSE(camera.HasPriorFocalLength());
}

TEST(Camera, PrincipalPoint) {
  Camera camera;
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
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
  camera.SetModelId(FullOpenCVCameraModel::model_id);
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 2);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 8);
}

TEST(Camera, CalibrationMatrix) {
  Camera camera;
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  const Eigen::Matrix3d K = camera.CalibrationMatrix();
  Eigen::Matrix3d K_ref;
  K_ref << 1, 0, 0.5, 0, 1, 0.5, 0, 0, 1;
  EXPECT_EQ(K, K_ref);
}

TEST(Camera, ParamsInfo) {
  Camera camera;
  EXPECT_THROW(camera.ParamsInfo(), std::domain_error);
  camera.SetModelId(SimpleRadialCameraModel::model_id);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy, k");
}

TEST(Camera, Params) {
  Camera camera;
  EXPECT_EQ(camera.NumParams(), 0);
  EXPECT_EQ(camera.Params().size(), camera.NumParams());
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.NumParams(), 3);
  EXPECT_EQ(camera.Params().size(), camera.NumParams());
  EXPECT_EQ(camera.ParamsData(), camera.Params().data());
  EXPECT_EQ(camera.Params(0), 1.0);
  EXPECT_EQ(camera.Params(1), 0.5);
  EXPECT_EQ(camera.Params(2), 0.5);
  EXPECT_EQ(camera.Params()[0], 1.0);
  EXPECT_EQ(camera.Params()[1], 0.5);
  EXPECT_EQ(camera.Params()[2], 0.5);
  camera.SetParams({2.0, 1.0, 1.0});
  EXPECT_EQ(camera.Params(0), 2.0);
  EXPECT_EQ(camera.Params(1), 1.0);
  EXPECT_EQ(camera.Params(2), 1.0);
}

TEST(Camera, ParamsToString) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.ParamsToString(), "1.000000, 0.500000, 0.500000");
}

TEST(Camera, ParamsFromString) {
  Camera camera;
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  EXPECT_TRUE(camera.SetParamsFromString("1.000000, 0.500000, 0.500000"));
  const std::vector<double> params{1.0, 0.5, 0.5};
  EXPECT_EQ(camera.Params(), params);
  EXPECT_FALSE(camera.SetParamsFromString("1.000000, 0.500000"));
  EXPECT_EQ(camera.Params(), params);
}

TEST(Camera, VerifyParams) {
  Camera camera;
  EXPECT_THROW(camera.VerifyParams(), std::domain_error);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.VerifyParams());
  camera.Params().resize(2);
  EXPECT_FALSE(camera.VerifyParams());
}

TEST(Camera, IsUndistorted) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.InitializeWithId(SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams({1.0, 0.5, 0.5, 0.005});
  EXPECT_FALSE(camera.IsUndistorted());
  camera.InitializeWithId(RadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams({1.0, 0.5, 0.5, 0.0, 0.005});
  EXPECT_FALSE(camera.IsUndistorted());
  camera.InitializeWithId(OpenCVCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams({1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.001});
  EXPECT_FALSE(camera.IsUndistorted());
  camera.InitializeWithId(FullOpenCVCameraModel::model_id, 1.0, 1, 1);
  EXPECT_TRUE(camera.IsUndistorted());
  camera.SetParams(
      {1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001});
  EXPECT_FALSE(camera.IsUndistorted());
}

TEST(Camera, HasBogusParams) {
  Camera camera;
  EXPECT_THROW(camera.HasBogusParams(0.0, 0.0, 0.0), std::domain_error);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 1.0));
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 0.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.99, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(1.01, 1.1, 1.0));
  camera.InitializeWithId(SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 1.0));
  camera.Params(3) = 1.01;
  EXPECT_TRUE(camera.HasBogusParams(0.1, 1.1, 1.0));
  camera.Params(3) = -0.5;
  EXPECT_FALSE(camera.HasBogusParams(0.1, 1.1, 1.0));
  camera.Params(3) = -1.01;
  EXPECT_TRUE(camera.HasBogusParams(0.1, 1.1, 1.0));
}

TEST(Camera, InitializeWithId) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  EXPECT_EQ(camera.ModelId(),
            static_cast<int>(SimplePinholeCameraModel::model_id));
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_FALSE(camera.HasPriorFocalLength());
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1.000000, 0.500000, 0.500000");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_FALSE(camera.HasBogusParams(0.1, 2.0, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.5, 1.0));
  EXPECT_EQ(camera.NumParams(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.Params().size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(Camera, InitializeWithName) {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CameraId(), kInvalidCameraId);
  EXPECT_EQ(camera.ModelId(),
            static_cast<int>(SimplePinholeCameraModel::model_id));
  EXPECT_EQ(camera.ModelName(), "SIMPLE_PINHOLE");
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_FALSE(camera.HasPriorFocalLength());
  EXPECT_EQ(camera.FocalLengthIdxs().size(), 1);
  EXPECT_EQ(camera.PrincipalPointIdxs().size(), 2);
  EXPECT_EQ(camera.ExtraParamsIdxs().size(), 0);
  EXPECT_EQ(camera.ParamsInfo(), "f, cx, cy");
  EXPECT_EQ(camera.ParamsToString(), "1.000000, 0.500000, 0.500000");
  EXPECT_EQ(camera.FocalLength(), 1.0);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_FALSE(camera.HasBogusParams(0.1, 2.0, 1.0));
  EXPECT_TRUE(camera.HasBogusParams(0.1, 0.5, 1.0));
  EXPECT_EQ(camera.NumParams(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
  EXPECT_EQ(camera.Params().size(),
            static_cast<int>(SimplePinholeCameraModel::num_params));
}

TEST(Camera, CamFromImg) {
  Camera camera;
  EXPECT_THROW(camera.CamFromImg(Eigen::Vector2d::Zero()), std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.0, 0.0))(0), -0.5);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.0, 0.0))(1), -0.5);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.5, 0.5))(0), 0.0);
  EXPECT_EQ(camera.CamFromImg(Eigen::Vector2d(0.5, 0.5))(1), 0.0);
}

TEST(Camera, CamFromImgThreshold) {
  Camera camera;
  EXPECT_THROW(camera.CamFromImgThreshold(0), std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.CamFromImgThreshold(0), 0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 1);
  camera.SetFocalLength(2.0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 0.5);
  camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  camera.SetFocalLengthY(3.0);
  EXPECT_EQ(camera.CamFromImgThreshold(1), 0.5);
}

TEST(Camera, ImgFromCam) {
  Camera camera;
  EXPECT_THROW(camera.ImgFromCam(Eigen::Vector2d::Zero()), std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(0.0, 0.0))(0), 0.5);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(0.0, 0.0))(1), 0.5);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(-0.5, -0.5))(0), 0.0);
  EXPECT_EQ(camera.ImgFromCam(Eigen::Vector2d(-0.5, -0.5))(1), 0.0);
}

TEST(Camera, Rescale) {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  EXPECT_EQ(camera.Width(), 2);
  EXPECT_EQ(camera.Height(), 2);
  EXPECT_EQ(camera.FocalLength(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 1);
  EXPECT_EQ(camera.PrincipalPointY(), 1);

  camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  EXPECT_EQ(camera.Width(), 2);
  EXPECT_EQ(camera.Height(), 2);
  EXPECT_EQ(camera.FocalLengthX(), 2);
  EXPECT_EQ(camera.FocalLengthY(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 1);
  EXPECT_EQ(camera.PrincipalPointY(), 1);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(0.5);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_EQ(camera.FocalLengthX(), 0.5);
  EXPECT_EQ(camera.FocalLengthY(), 0.5);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(1, 1);
  EXPECT_EQ(camera.Width(), 1);
  EXPECT_EQ(camera.Height(), 1);
  EXPECT_EQ(camera.FocalLengthX(), 0.5);
  EXPECT_EQ(camera.FocalLengthY(), 0.5);
  EXPECT_EQ(camera.PrincipalPointX(), 0.5);
  EXPECT_EQ(camera.PrincipalPointY(), 0.5);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(4, 4);
  EXPECT_EQ(camera.Width(), 4);
  EXPECT_EQ(camera.Height(), 4);
  EXPECT_EQ(camera.FocalLengthX(), 2);
  EXPECT_EQ(camera.FocalLengthY(), 2);
  EXPECT_EQ(camera.PrincipalPointX(), 2);
  EXPECT_EQ(camera.PrincipalPointY(), 2);
}

}  // namespace colmap
