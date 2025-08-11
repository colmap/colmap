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

#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

bool FisheyeCameraModelIsValidPixel(const CameraModelId model_id,
                                    const std::vector<double>& params,
                                    const Eigen::Vector2d& xy) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::model_id: {                                           \
    double uu, vv;                                                        \
    CameraModel::FisheyeFromImg(params.data(), xy.x(), xy.y(), &uu, &vv); \
    const double theta = std::sqrt(uu * uu + vv * vv);                    \
    if (theta < M_PI / 2.0) {                                             \
      return true;                                                        \
    } else {                                                              \
      return false;                                                       \
    }                                                                     \
  }

    FISHEYE_CAMERA_MODEL_CASES
    default:
      throw std::domain_error(
          "Camera model does not exist or is not a fisheye camera");

#undef CAMERA_MODEL_CASE
  }

  return false;
}

template <typename CameraModel>
void TestCamToCamFromImg(const std::vector<double>& params,
                         const double u0,
                         const double v0,
                         const double w0) {
  double u = 0;
  double v = 0;
  double x = 0;
  double y = 0;
  CameraModel::ImgFromCam(params.data(), u0, v0, w0, &x, &y);
  const std::optional<Eigen::Vector2d> xy = CameraModelImgFromCam(
      CameraModel::model_id, params, Eigen::Vector3d(u0, v0, w0));
  ASSERT_TRUE(xy.has_value());
  EXPECT_EQ(x, xy->x());
  EXPECT_EQ(y, xy->y());
  CameraModel::CamFromImg(params.data(), x, y, &u, &v);
  EXPECT_NEAR(u, u0 / w0, 1e-6);
  EXPECT_NEAR(v, v0 / w0, 1e-6);
}

template <typename CameraModel>
void TestCamFromImgToImg(const std::vector<double>& params,
                         const double x0,
                         const double y0) {
  double u = 0;
  double v = 0;
  double x = 0;
  double y = 0;
  CameraModel::CamFromImg(params.data(), x0, y0, &u, &v);
  const std::optional<Eigen::Vector2d> uv = CameraModelCamFromImg(
      CameraModel::model_id, params, Eigen::Vector2d(x0, y0));
  ASSERT_TRUE(uv.has_value());
  EXPECT_EQ(u, uv->x());
  EXPECT_EQ(v, uv->y());
  for (const double w : {0.5, 1.0, 2.0}) {
    ASSERT_TRUE(
        CameraModel::ImgFromCam(params.data(), w * u, w * v, w, &x, &y));
    EXPECT_NEAR(x, x0, 1e-6);
    EXPECT_NEAR(y, y0, 1e-6);
  }
}

template <typename CameraModel>
void TestModel(const std::vector<double>& params) {
  EXPECT_TRUE(CameraModelVerifyParams(CameraModel::model_id, params));

  const std::vector<double> default_params =
      CameraModelInitializeParams(CameraModel::model_id, 100, 100, 100);
  EXPECT_TRUE(CameraModelVerifyParams(CameraModel::model_id, default_params));

  EXPECT_EQ(CameraModelParamsInfo(CameraModel::model_id),
            CameraModel::params_info);
  EXPECT_EQ(std::vector<size_t>(
                CameraModelFocalLengthIdxs(CameraModel::model_id).begin(),
                CameraModelFocalLengthIdxs(CameraModel::model_id).end()),
            std::vector<size_t>(CameraModel::focal_length_idxs.begin(),
                                CameraModel::focal_length_idxs.end()));
  EXPECT_EQ(std::vector<size_t>(
                CameraModelPrincipalPointIdxs(CameraModel::model_id).begin(),
                CameraModelPrincipalPointIdxs(CameraModel::model_id).end()),
            std::vector<size_t>(CameraModel::principal_point_idxs.begin(),
                                CameraModel::principal_point_idxs.end()));
  EXPECT_EQ(std::vector<size_t>(
                CameraModelExtraParamsIdxs(CameraModel::model_id).begin(),
                CameraModelExtraParamsIdxs(CameraModel::model_id).end()),
            std::vector<size_t>(CameraModel::extra_params_idxs.begin(),
                                CameraModel::extra_params_idxs.end()));
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
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_FALSE(ExistsCameraModelWithId(static_cast<CameraModelId>(123456789)));

  EXPECT_EQ(CameraModelNameToId(CameraModelIdToName(CameraModel::model_id)),
            CameraModel::model_id);
  EXPECT_EQ(CameraModelIdToName(CameraModelNameToId(CameraModel::model_name)),
            CameraModel::model_name);

  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double u = -0.5; u <= 0.5; u += 0.1) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double v = -0.5; v <= 0.5; v += 0.1) {
      for (const double w : {0.5, 1.0, 2.0}) {
        TestCamToCamFromImg<CameraModel>(params, u, v, w);
      }
    }
  }

  for (int x = 0; x <= 800; x += 50) {
    for (int y = 0; y <= 800; y += 50) {
      if (CameraModelIsFisheye(CameraModel::model_id) &&
          !FisheyeCameraModelIsValidPixel(
              CameraModel::model_id, params, Eigen::Vector2d(x, y))) {
        continue;
      }
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

TEST(RadTanThinPrismFisheye, Nominal) {
  std::vector<double> params = {651.123,
                                655.123,
                                386.123,
                                511.123,
                                -0.0232,
                                0.0924,
                                -0.0591,
                                0.003,
                                0.0048,
                                -0.0009,
                                0.0002,
                                0.0005,
                                -0.0009,
                                -0.0001,
                                0.00007,
                                -0.00017};

  TestModel<RadTanThinPrismFisheyeModel>(params);
}

}  // namespace
}  // namespace colmap
