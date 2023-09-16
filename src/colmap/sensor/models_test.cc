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

#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

template <typename CameraModel>
void TestCamToCamFromImg(const std::vector<double>& params,
                         const double u0,
                         const double v0,
                         const double w0) {
  double u, v, w, x, y;
  CameraModel::ImgFromCam(params.data(), u0, v0, w0, &x, &y);
  const Eigen::Vector2d xy = CameraModelImgFromCam(
      CameraModel::model_id, params, Eigen::Vector3d(u0, v0, w0));
  EXPECT_EQ(x, xy.x());
  EXPECT_EQ(y, xy.y());
  CameraModel::CamFromImg(params.data(), x, y, &u, &v, &w);
  EXPECT_NEAR(u, u0 / w0, 1e-6);
  EXPECT_NEAR(v, v0 / w0, 1e-6);
}

template <typename CameraModel>
void TestCamFromImgToImg(const std::vector<double>& params,
                         const double x0,
                         const double y0) {
  double u, v, w, x, y;
  CameraModel::CamFromImg(params.data(), x0, y0, &u, &v, &w);
  const Eigen::Vector3d uvw = CameraModelCamFromImg(
      CameraModel::model_id, params, Eigen::Vector2d(x0, y0));
  EXPECT_EQ(u, uvw.x());
  EXPECT_EQ(v, uvw.y());
  EXPECT_EQ(w, uvw.z());
  CameraModel::ImgFromCam(params.data(), u, v, w, &x, &y);
  EXPECT_NEAR(x, x0, 1e-6);
  EXPECT_NEAR(y, y0, 1e-6);
}

template <typename CameraModel>
void TestModel(const std::vector<double>& params) {
  EXPECT_TRUE(CameraModelVerifyParams(CameraModel::model_id, params));

  const std::vector<double> default_params =
      CameraModelInitializeParams(CameraModel::model_id, 100, 100, 100);
  EXPECT_TRUE(CameraModelVerifyParams(CameraModel::model_id, default_params));

  EXPECT_EQ(CameraModelParamsInfo(CameraModel::model_id),
            CameraModel::params_info);
  EXPECT_EQ(&CameraModelFocalLengthIdxs(CameraModel::model_id),
            &CameraModel::focal_length_idxs);
  EXPECT_EQ(&CameraModelPrincipalPointIdxs(CameraModel::model_id),
            &CameraModel::principal_point_idxs);
  EXPECT_EQ(&CameraModelExtraParamsIdxs(CameraModel::model_id),
            &CameraModel::extra_params_idxs);
  EXPECT_EQ(CameraModelNumParams(CameraModel::model_id),
            CameraModel::num_params);

  EXPECT_FALSE(CameraModelHasBogusParams(
      CameraModel::model_id, default_params, 100, 100, 0.1, 2.0, 1.0));
  EXPECT_TRUE(CameraModelHasBogusParams(
      CameraModel::model_id, default_params, 100, 100, 0.1, 0.5, 1.0));
  EXPECT_TRUE(CameraModelHasBogusParams(
      CameraModel::model_id, default_params, 100, 100, 1.5, 2.0, 1.0));
  if (CameraModel::extra_params_idxs.size() > 0) {
    EXPECT_TRUE(CameraModelHasBogusParams(
        CameraModel::model_id, default_params, 100, 100, 0.1, 2.0, -0.1));
  }

  EXPECT_EQ(CameraModelCamFromImgThreshold(CameraModel::model_id, params, 0),
            0);
  EXPECT_GT(CameraModelCamFromImgThreshold(CameraModel::model_id, params, 1),
            0);
  EXPECT_EQ(
      CameraModelCamFromImgThreshold(CameraModel::model_id, default_params, 1),
      1.0 / 100.0);

  EXPECT_TRUE(ExistsCameraModelWithName(CameraModel::model_name));
  EXPECT_FALSE(ExistsCameraModelWithName(CameraModel::model_name + "FOO"));

  EXPECT_TRUE(ExistsCameraModelWithId(CameraModel::model_id));
  EXPECT_FALSE(ExistsCameraModelWithId(CameraModel::model_id + 123456789));

  EXPECT_EQ(CameraModelNameToId(CameraModelIdToName(CameraModel::model_id)),
            CameraModel::model_id);
  EXPECT_EQ(CameraModelIdToName(CameraModelNameToId(CameraModel::model_name)),
            CameraModel::model_name);

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double u = -0.5; u <= 0.5; u += 0.1) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double v = -0.5; v <= 0.5; v += 0.1) {
      for (double w : {1, 2}) {
        TestCamToCamFromImg<CameraModel>(params, u, v, w);
      }
    }
  }

  for (int x = 0; x <= 800; x += 50) {
    for (int y = 0; y <= 800; y += 50) {
      TestCamFromImgToImg<CameraModel>(params, x, y);
    }
  }

  const auto pp_idxs = CameraModel::principal_point_idxs;
  TestCamFromImgToImg<CameraModel>(
      params, params[pp_idxs.at(0)], params[pp_idxs.at(1)]);
}

TEST(SimplePinhole, Nominal) {
  TestModel<SimplePinholeCameraModel>({655.123, 386.123, 511.123});
}

TEST(Pinhole, Nominal) {
  TestModel<PinholeCameraModel>({651.123, 655.123, 386.123, 511.123});
}

TEST(SimpleRadial, Nominal) {
  TestModel<SimpleRadialCameraModel>({651.123, 386.123, 511.123, 0});
  TestModel<SimpleRadialCameraModel>({651.123, 386.123, 511.123, 0.1});
}

TEST(Radial, Nominal) {
  TestModel<RadialCameraModel>({651.123, 386.123, 511.123, 0, 0});
  TestModel<RadialCameraModel>({651.123, 386.123, 511.123, 0.1, 0});
  TestModel<RadialCameraModel>({651.123, 386.123, 511.12, 0, 0.05});
  TestModel<RadialCameraModel>({651.123, 386.123, 511.123, 0.05, 0.03});
}

TEST(OpenCV, Nominal) {
  TestModel<OpenCVCameraModel>(
      {651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001});
}

TEST(OpenCVFisheye, Nominal) {
  TestModel<OpenCVFisheyeCameraModel>(
      {651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001});
}

TEST(FullOpenCV, Nominal) {
  TestModel<FullOpenCVCameraModel>({651.123,
                                    655.123,
                                    386.123,
                                    511.123,
                                    -0.471,
                                    0.223,
                                    -0.001,
                                    0.001,
                                    0.001,
                                    0.02,
                                    -0.02,
                                    0.001});
}

TEST(FOV, Nominal) {
  TestModel<FOVCameraModel>({651.123, 655.123, 386.123, 511.123, 0});
  TestModel<FOVCameraModel>({651.123, 655.123, 386.123, 511.123, 0.9});
  TestModel<FOVCameraModel>({651.123, 655.123, 386.123, 511.123, 1e-6});
  TestModel<FOVCameraModel>({651.123, 655.123, 386.123, 511.123, 1e-2});
  EXPECT_EQ(CameraModelInitializeParams(FOVCameraModel::model_id, 100, 100, 100)
                .back(),
            1e-2);
}

TEST(SimpleRadialFisheye, Nominal) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0};
  TestModel<SimpleRadialFisheyeCameraModel>({651.123, 386.123, 511.123, 0});
  TestModel<SimpleRadialFisheyeCameraModel>({651.123, 386.123, 511.123, 0.1});
}

TEST(RadialFisheye, Nominal) {
  TestModel<RadialFisheyeCameraModel>({651.123, 386.123, 511.123, 0, 0});
  TestModel<RadialFisheyeCameraModel>({651.123, 386.123, 511.123, 0, 0.1});
  TestModel<RadialFisheyeCameraModel>({651.123, 386.123, 511.123, 0, 0.05});
  TestModel<RadialFisheyeCameraModel>({651.123, 386.123, 511.123, 0, 0.03});
}

TEST(ThinPrismFisheye, Nominal) {
  TestModel<ThinPrismFisheyeCameraModel>({651.123,
                                          655.123,
                                          386.123,
                                          511.123,
                                          -0.471,
                                          0.223,
                                          -0.001,
                                          0.001,
                                          0.001,
                                          0.02,
                                          -0.02,
                                          0.001});
}

}  // namespace
}  // namespace colmap
