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

#define TEST_NAME "base/camera"
#include "util/testing.h"

#include "base/camera.h"
#include "base/camera_models.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestEmpty) {
  Camera camera;
  BOOST_CHECK_EQUAL(camera.CameraId(), kInvalidCameraId);
  BOOST_CHECK_EQUAL(camera.ModelId(), kInvalidCameraModelId);
  BOOST_CHECK_EQUAL(camera.ModelName(), "");
  BOOST_CHECK_EQUAL(camera.Width(), 0);
  BOOST_CHECK_EQUAL(camera.Height(), 0);
  BOOST_CHECK_EQUAL(camera.HasPriorFocalLength(), false);
  BOOST_CHECK_THROW(camera.FocalLengthIdxs(), std::domain_error);
  BOOST_CHECK_THROW(camera.ParamsInfo(), std::domain_error);
  BOOST_CHECK_EQUAL(camera.ParamsToString(), "");
  BOOST_CHECK_EQUAL(camera.NumParams(), 0);
  BOOST_CHECK_EQUAL(camera.Params().size(), 0);
  BOOST_CHECK_EQUAL(camera.ParamsData(), camera.Params().data());
}

BOOST_AUTO_TEST_CASE(TestCameraId) {
  Camera camera;
  BOOST_CHECK_EQUAL(camera.CameraId(), kInvalidCameraId);
  camera.SetCameraId(1);
  BOOST_CHECK_EQUAL(camera.CameraId(), 1);
}

BOOST_AUTO_TEST_CASE(TestModelId) {
  Camera camera;
  BOOST_CHECK_EQUAL(camera.ModelId(), kInvalidCameraModelId);
  BOOST_CHECK_EQUAL(camera.ModelName(), "");
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  BOOST_CHECK_EQUAL(camera.ModelId(),
                    static_cast<int>(SimplePinholeCameraModel::model_id));
  BOOST_CHECK_EQUAL(camera.ModelName(), "SIMPLE_PINHOLE");
  BOOST_CHECK_EQUAL(camera.NumParams(), SimplePinholeCameraModel::num_params);
  camera.SetModelIdFromName("SIMPLE_RADIAL");
  BOOST_CHECK_EQUAL(camera.ModelId(),
                    static_cast<int>(SimpleRadialCameraModel::model_id));
  BOOST_CHECK_EQUAL(camera.ModelName(), "SIMPLE_RADIAL");
  BOOST_CHECK_EQUAL(camera.NumParams(), SimpleRadialCameraModel::num_params);
}

BOOST_AUTO_TEST_CASE(TestWidthHeight) {
  Camera camera;
  BOOST_CHECK_EQUAL(camera.Width(), 0);
  BOOST_CHECK_EQUAL(camera.Height(), 0);
  camera.SetWidth(1);
  BOOST_CHECK_EQUAL(camera.Width(), 1);
  BOOST_CHECK_EQUAL(camera.Height(), 0);
  camera.SetHeight(1);
  BOOST_CHECK_EQUAL(camera.Width(), 1);
  BOOST_CHECK_EQUAL(camera.Height(), 1);
}

BOOST_AUTO_TEST_CASE(TestFocalLength) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.FocalLength(), 1.0);
  camera.SetFocalLength(2.0);
  BOOST_CHECK_EQUAL(camera.FocalLength(), 2.0);
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.FocalLengthX(), 1.0);
  BOOST_CHECK_EQUAL(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthX(2.0);
  BOOST_CHECK_EQUAL(camera.FocalLengthX(), 2.0);
  BOOST_CHECK_EQUAL(camera.FocalLengthY(), 1.0);
  camera.SetFocalLengthY(2.0);
  BOOST_CHECK_EQUAL(camera.FocalLengthX(), 2.0);
  BOOST_CHECK_EQUAL(camera.FocalLengthY(), 2.0);
}

BOOST_AUTO_TEST_CASE(TestPriorFocalLength) {
  Camera camera;
  BOOST_CHECK_EQUAL(camera.HasPriorFocalLength(), false);
  camera.SetPriorFocalLength(true);
  BOOST_CHECK_EQUAL(camera.HasPriorFocalLength(), true);
  camera.SetPriorFocalLength(false);
  BOOST_CHECK_EQUAL(camera.HasPriorFocalLength(), false);
}

BOOST_AUTO_TEST_CASE(TestPrincipalPoint) {
  Camera camera;
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 0.5);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 0.5);
  camera.SetPrincipalPointX(2.0);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 2.0);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 0.5);
  camera.SetPrincipalPointY(2.0);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 2.0);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 2.0);
}

BOOST_AUTO_TEST_CASE(TestParamIdxs) {
  Camera camera;
  BOOST_CHECK_THROW(camera.FocalLengthIdxs(), std::domain_error);
  BOOST_CHECK_THROW(camera.PrincipalPointIdxs(), std::domain_error);
  BOOST_CHECK_THROW(camera.ExtraParamsIdxs(), std::domain_error);
  camera.SetModelId(FullOpenCVCameraModel::model_id);
  BOOST_CHECK_EQUAL(camera.FocalLengthIdxs().size(), 2);
  BOOST_CHECK_EQUAL(camera.PrincipalPointIdxs().size(), 2);
  BOOST_CHECK_EQUAL(camera.ExtraParamsIdxs().size(), 8);
}

BOOST_AUTO_TEST_CASE(TestCalibrationMatrix) {
  Camera camera;
  camera.InitializeWithId(PinholeCameraModel::model_id, 1.0, 1, 1);
  const Eigen::Matrix3d K = camera.CalibrationMatrix();
  Eigen::Matrix3d K_ref;
  K_ref << 1, 0, 0.5, 0, 1, 0.5, 0, 0, 1;
  BOOST_CHECK_EQUAL(K, K_ref);
}

BOOST_AUTO_TEST_CASE(TestParamsInfo) {
  Camera camera;
  BOOST_CHECK_THROW(camera.ParamsInfo(), std::domain_error);
  camera.SetModelId(SimpleRadialCameraModel::model_id);
  BOOST_CHECK_EQUAL(camera.ParamsInfo(), "f, cx, cy, k");
}

BOOST_AUTO_TEST_CASE(TestParams) {
  Camera camera;
  BOOST_CHECK_EQUAL(camera.NumParams(), 0);
  BOOST_CHECK_EQUAL(camera.Params().size(), camera.NumParams());
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.NumParams(), 3);
  BOOST_CHECK_EQUAL(camera.Params().size(), camera.NumParams());
  BOOST_CHECK_EQUAL(camera.ParamsData(), camera.Params().data());
  BOOST_CHECK_EQUAL(camera.Params(0), 1.0);
  BOOST_CHECK_EQUAL(camera.Params(1), 0.5);
  BOOST_CHECK_EQUAL(camera.Params(2), 0.5);
  BOOST_CHECK_EQUAL(camera.Params()[0], 1.0);
  BOOST_CHECK_EQUAL(camera.Params()[1], 0.5);
  BOOST_CHECK_EQUAL(camera.Params()[2], 0.5);
  camera.SetParams({2.0, 1.0, 1.0});
  BOOST_CHECK_EQUAL(camera.Params(0), 2.0);
  BOOST_CHECK_EQUAL(camera.Params(1), 1.0);
  BOOST_CHECK_EQUAL(camera.Params(2), 1.0);
}

BOOST_AUTO_TEST_CASE(TestParamsToString) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.ParamsToString(), "1.000000, 0.500000, 0.500000");
}

BOOST_AUTO_TEST_CASE(TestParamsFromString) {
  Camera camera;
  camera.SetModelId(SimplePinholeCameraModel::model_id);
  BOOST_CHECK(camera.SetParamsFromString("1.000000, 0.500000, 0.500000"));
  const std::vector<double> params{1.0, 0.5, 0.5};
  BOOST_CHECK_EQUAL_COLLECTIONS(camera.Params().begin(), camera.Params().end(),
                                params.begin(), params.end());
  BOOST_CHECK(!camera.SetParamsFromString("1.000000, 0.500000"));
  BOOST_CHECK_EQUAL_COLLECTIONS(camera.Params().begin(), camera.Params().end(),
                                params.begin(), params.end());
}

BOOST_AUTO_TEST_CASE(TestVerifyParams) {
  Camera camera;
  BOOST_CHECK_THROW(camera.VerifyParams(), std::domain_error);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.VerifyParams(), true);
  camera.Params().resize(2);
  BOOST_CHECK_EQUAL(camera.VerifyParams(), false);
}

BOOST_AUTO_TEST_CASE(TestIsUndistorted) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK(camera.IsUndistorted());
  camera.InitializeWithId(SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK(camera.IsUndistorted());
  camera.SetParams({1.0, 0.5, 0.5, 0.005});
  BOOST_CHECK(!camera.IsUndistorted());
  camera.InitializeWithId(RadialCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK(camera.IsUndistorted());
  camera.SetParams({1.0, 0.5, 0.5, 0.0, 0.005});
  BOOST_CHECK(!camera.IsUndistorted());
  camera.InitializeWithId(OpenCVCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK(camera.IsUndistorted());
  camera.SetParams({1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.001});
  BOOST_CHECK(!camera.IsUndistorted());
  camera.InitializeWithId(FullOpenCVCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK(camera.IsUndistorted());
  camera.SetParams({1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001});
  BOOST_CHECK(!camera.IsUndistorted());
}

BOOST_AUTO_TEST_CASE(TestHasBogusParams) {
  Camera camera;
  BOOST_CHECK_THROW(camera.HasBogusParams(0.0, 0.0, 0.0), std::domain_error);
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 1.1, 1.0), false);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 1.1, 0.0), false);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 0.99, 1.0), true);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(1.01, 1.1, 1.0), true);
  camera.InitializeWithId(SimpleRadialCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 1.1, 1.0), false);
  camera.Params(3) = 1.01;
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 1.1, 1.0), true);
  camera.Params(3) = -0.5;
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 1.1, 1.0), false);
  camera.Params(3) = -1.01;
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 1.1, 1.0), true);
}

BOOST_AUTO_TEST_CASE(TestInitializeWithId) {
  Camera camera;
  camera.InitializeWithId(SimplePinholeCameraModel::model_id, 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.CameraId(), kInvalidCameraId);
  BOOST_CHECK_EQUAL(camera.ModelId(),
                    static_cast<int>(SimplePinholeCameraModel::model_id));
  BOOST_CHECK_EQUAL(camera.ModelName(), "SIMPLE_PINHOLE");
  BOOST_CHECK_EQUAL(camera.Width(), 1);
  BOOST_CHECK_EQUAL(camera.Height(), 1);
  BOOST_CHECK_EQUAL(camera.HasPriorFocalLength(), false);
  BOOST_CHECK_EQUAL(camera.FocalLengthIdxs().size(), 1);
  BOOST_CHECK_EQUAL(camera.PrincipalPointIdxs().size(), 2);
  BOOST_CHECK_EQUAL(camera.ExtraParamsIdxs().size(), 0);
  BOOST_CHECK_EQUAL(camera.ParamsInfo(), "f, cx, cy");
  BOOST_CHECK_EQUAL(camera.ParamsToString(), "1.000000, 0.500000, 0.500000");
  BOOST_CHECK_EQUAL(camera.FocalLength(), 1.0);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 0.5);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 0.5);
  BOOST_CHECK_EQUAL(camera.VerifyParams(), true);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 2.0, 1.0), false);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 0.5, 1.0), true);
  BOOST_CHECK_EQUAL(camera.NumParams(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
  BOOST_CHECK_EQUAL(camera.Params().size(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
}

BOOST_AUTO_TEST_CASE(TestInitializeWithName) {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.CameraId(), kInvalidCameraId);
  BOOST_CHECK_EQUAL(camera.ModelId(),
                    static_cast<int>(SimplePinholeCameraModel::model_id));
  BOOST_CHECK_EQUAL(camera.ModelName(), "SIMPLE_PINHOLE");
  BOOST_CHECK_EQUAL(camera.Width(), 1);
  BOOST_CHECK_EQUAL(camera.Height(), 1);
  BOOST_CHECK_EQUAL(camera.HasPriorFocalLength(), false);
  BOOST_CHECK_EQUAL(camera.FocalLengthIdxs().size(), 1);
  BOOST_CHECK_EQUAL(camera.PrincipalPointIdxs().size(), 2);
  BOOST_CHECK_EQUAL(camera.ExtraParamsIdxs().size(), 0);
  BOOST_CHECK_EQUAL(camera.ParamsInfo(), "f, cx, cy");
  BOOST_CHECK_EQUAL(camera.ParamsToString(), "1.000000, 0.500000, 0.500000");
  BOOST_CHECK_EQUAL(camera.FocalLength(), 1.0);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 0.5);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 0.5);
  BOOST_CHECK_EQUAL(camera.VerifyParams(), true);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 2.0, 1.0), false);
  BOOST_CHECK_EQUAL(camera.HasBogusParams(0.1, 0.5, 1.0), true);
  BOOST_CHECK_EQUAL(camera.NumParams(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
  BOOST_CHECK_EQUAL(camera.Params().size(),
                    static_cast<int>(SimplePinholeCameraModel::num_params));
}

BOOST_AUTO_TEST_CASE(TestImageToWorld) {
  Camera camera;
  BOOST_CHECK_THROW(camera.ImageToWorld(Eigen::Vector2d::Zero()),
                    std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.ImageToWorld(Eigen::Vector2d(0.0, 0.0))(0), -0.5);
  BOOST_CHECK_EQUAL(camera.ImageToWorld(Eigen::Vector2d(0.0, 0.0))(1), -0.5);
  BOOST_CHECK_EQUAL(camera.ImageToWorld(Eigen::Vector2d(0.5, 0.5))(0), 0.0);
  BOOST_CHECK_EQUAL(camera.ImageToWorld(Eigen::Vector2d(0.5, 0.5))(1), 0.0);
}

BOOST_AUTO_TEST_CASE(TestImageToWorldThreshold) {
  Camera camera;
  BOOST_CHECK_THROW(camera.ImageToWorldThreshold(0), std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.ImageToWorldThreshold(0), 0);
  BOOST_CHECK_EQUAL(camera.ImageToWorldThreshold(1), 1);
  camera.SetFocalLength(2.0);
  BOOST_CHECK_EQUAL(camera.ImageToWorldThreshold(1), 0.5);
  camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  camera.SetFocalLengthY(3.0);
  BOOST_CHECK_EQUAL(camera.ImageToWorldThreshold(1), 0.5);
}

BOOST_AUTO_TEST_CASE(TestWorldToImage) {
  Camera camera;
  BOOST_CHECK_THROW(camera.WorldToImage(Eigen::Vector2d::Zero()),
                    std::domain_error);
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  BOOST_CHECK_EQUAL(camera.WorldToImage(Eigen::Vector2d(0.0, 0.0))(0), 0.5);
  BOOST_CHECK_EQUAL(camera.WorldToImage(Eigen::Vector2d(0.0, 0.0))(1), 0.5);
  BOOST_CHECK_EQUAL(camera.WorldToImage(Eigen::Vector2d(-0.5, -0.5))(0), 0.0);
  BOOST_CHECK_EQUAL(camera.WorldToImage(Eigen::Vector2d(-0.5, -0.5))(1), 0.0);
}

BOOST_AUTO_TEST_CASE(TestRescale) {
  Camera camera;
  camera.InitializeWithName("SIMPLE_PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  BOOST_CHECK_EQUAL(camera.Width(), 2);
  BOOST_CHECK_EQUAL(camera.Height(), 2);
  BOOST_CHECK_EQUAL(camera.FocalLength(), 2);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 1);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 1);

  camera.InitializeWithName("PINHOLE", 1.0, 1, 1);
  camera.Rescale(2.0);
  BOOST_CHECK_EQUAL(camera.Width(), 2);
  BOOST_CHECK_EQUAL(camera.Height(), 2);
  BOOST_CHECK_EQUAL(camera.FocalLengthX(), 2);
  BOOST_CHECK_EQUAL(camera.FocalLengthY(), 2);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 1);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 1);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(0.5);
  BOOST_CHECK_EQUAL(camera.Width(), 1);
  BOOST_CHECK_EQUAL(camera.Height(), 1);
  BOOST_CHECK_EQUAL(camera.FocalLengthX(), 0.5);
  BOOST_CHECK_EQUAL(camera.FocalLengthY(), 0.5);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 0.5);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 0.5);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(1, 1);
  BOOST_CHECK_EQUAL(camera.Width(), 1);
  BOOST_CHECK_EQUAL(camera.Height(), 1);
  BOOST_CHECK_EQUAL(camera.FocalLengthX(), 0.5);
  BOOST_CHECK_EQUAL(camera.FocalLengthY(), 0.5);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 0.5);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 0.5);

  camera.InitializeWithName("PINHOLE", 1.0, 2, 2);
  camera.Rescale(4, 4);
  BOOST_CHECK_EQUAL(camera.Width(), 4);
  BOOST_CHECK_EQUAL(camera.Height(), 4);
  BOOST_CHECK_EQUAL(camera.FocalLengthX(), 2);
  BOOST_CHECK_EQUAL(camera.FocalLengthY(), 2);
  BOOST_CHECK_EQUAL(camera.PrincipalPointX(), 2);
  BOOST_CHECK_EQUAL(camera.PrincipalPointY(), 2);
}
