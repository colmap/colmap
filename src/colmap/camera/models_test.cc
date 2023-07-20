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

#include "colmap/camera/models.h"

#include <gtest/gtest.h>

namespace colmap {

template <typename CameraModel>
void TestCamToImgToWorld(const std::vector<double> params,
                         const double u0,
                         const double v0) {
  double u, v, x, y, xx, yy;
  CameraModel::CamToImg(params.data(), u0, v0, &x, &y);
  CameraModelCamToImg(CameraModel::model_id, params, u0, v0, &xx, &yy);
  EXPECT_EQ(x, xx);
  EXPECT_EQ(y, yy);
  CameraModel::ImgToCam(params.data(), x, y, &u, &v);
  EXPECT_LT(std::abs(u - u0), 1e-6);
  EXPECT_LT(std::abs(v - v0), 1e-6);
}

template <typename CameraModel>
void TestImageToCamToImg(const std::vector<double> params,
                         const double x0,
                         const double y0) {
  double u, v, x, y, uu, vv;
  CameraModel::ImgToCam(params.data(), x0, y0, &u, &v);
  CameraModelImgToCam(CameraModel::model_id, params, x0, y0, &uu, &vv);
  EXPECT_EQ(u, uu);
  EXPECT_EQ(v, vv);
  CameraModel::CamToImg(params.data(), u, v, &x, &y);
  EXPECT_LT(std::abs(x - x0), 1e-6);
  EXPECT_LT(std::abs(y - y0), 1e-6);
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

  EXPECT_EQ(CameraModelImgToCamThreshold(CameraModel::model_id, params, 0), 0);
  EXPECT_GT(CameraModelImgToCamThreshold(CameraModel::model_id, params, 1), 0);
  EXPECT_EQ(
      CameraModelImgToCamThreshold(CameraModel::model_id, default_params, 1),
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
      TestCamToImgToWorld<CameraModel>(params, u, v);
    }
  }

  for (int x = 0; x <= 800; x += 50) {
    for (int y = 0; y <= 800; y += 50) {
      TestImageToCamToImg<CameraModel>(params, x, y);
    }
  }

  const auto pp_idxs = CameraModel::principal_point_idxs;
  TestImageToCamToImg<CameraModel>(
      params, params[pp_idxs.at(0)], params[pp_idxs.at(1)]);
}

TEST(SimplePinhole, Nominal) {
  std::vector<double> params = {655.123, 386.123, 511.123};
  TestModel<SimplePinholeCameraModel>(params);
}

TEST(Pinhole, Nominal) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123};
  TestModel<PinholeCameraModel>(params);
}

TEST(SimpleRadial, Nominal) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0};
  TestModel<SimpleRadialCameraModel>(params);
  params[3] = 0.1;
  TestModel<SimpleRadialCameraModel>(params);
}

TEST(Radial, Nominal) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0, 0};
  TestModel<RadialCameraModel>(params);
  params[3] = 0.1;
  TestModel<RadialCameraModel>(params);
  params[3] = 0.05;
  TestModel<RadialCameraModel>(params);
  params[4] = 0.03;
  TestModel<RadialCameraModel>(params);
}

TEST(OpenCV, Nominal) {
  std::vector<double> params = {
      651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001};
  TestModel<OpenCVCameraModel>(params);
}

TEST(OpenCVFisheye, Nominal) {
  std::vector<double> params = {
      651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001};
  TestModel<OpenCVFisheyeCameraModel>(params);
}

TEST(FullOpenCV, Nominal) {
  std::vector<double> params = {651.123,
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
                                0.001};
  TestModel<FullOpenCVCameraModel>(params);
}

TEST(FOV, Nominal) {
  std::vector<double> params = {651.123, 655.123, 386.123, 511.123, 0.9};
  TestModel<FOVCameraModel>(params);
  params[4] = 0;
  TestModel<FOVCameraModel>(params);
  params[4] = 1e-6;
  TestModel<FOVCameraModel>(params);
  params[4] = 1e-2;
  TestModel<FOVCameraModel>(params);
  EXPECT_EQ(CameraModelInitializeParams(FOVCameraModel::model_id, 100, 100, 100)
                .back(),
            1e-2);
}

TEST(SimpleRadialFisheye, Nominal) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0};
  TestModel<SimpleRadialFisheyeCameraModel>(params);
  params[3] = 0.1;
  TestModel<SimpleRadialFisheyeCameraModel>(params);
}

TEST(RadialFisheye, Nominal) {
  std::vector<double> params = {651.123, 386.123, 511.123, 0, 0};
  TestModel<RadialFisheyeCameraModel>(params);
  params[3] = 0.1;
  TestModel<RadialFisheyeCameraModel>(params);
  params[3] = 0.05;
  TestModel<RadialFisheyeCameraModel>(params);
  params[4] = 0.03;
  TestModel<RadialFisheyeCameraModel>(params);
}

TEST(ThinPrismFisheye, Nominal) {
  std::vector<double> params = {651.123,
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
                                0.001};
  TestModel<ThinPrismFisheyeCameraModel>(params);
}

}  // namespace colmap
