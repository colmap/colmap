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
    if (theta < EIGEN_PI / 2.0) {                                         \
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

// Round-trip a pixel through the 3D bearing interface: CamRayFromImg yields a
// unit ray, ImgFromCam must project it back to the same pixel.
template <typename CameraModel>
void TestCamRayFromImgToImg(const std::vector<double>& params,
                            const double x0,
                            const double y0) {
  const std::optional<Eigen::Vector3d> ray = CameraModelCamRayFromImg(
      CameraModel::model_id, params, Eigen::Vector2d(x0, y0));
  ASSERT_TRUE(ray.has_value());
  EXPECT_NEAR(ray->norm(), 1.0, 1e-12);
  const std::optional<Eigen::Vector2d> xy =
      CameraModelImgFromCam(CameraModel::model_id, params, *ray);
  ASSERT_TRUE(xy.has_value());
  // The pixel round-trip is floored by the iterative Newton undistortion in
  // CamFromImg (~1e-7 worst case); matches the tolerance of the sibling
  // CamFromImg/ImgFromCam round-trip in TestCamFromImgToImg.
  EXPECT_NEAR(xy->x(), x0, 1e-6);
  EXPECT_NEAR(xy->y(), y0, 1e-6);
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
  EXPECT_TRUE(CameraModelMetaDataParamsIdxs(CameraModel::model_id).empty());
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
      if (CameraModelIsPerspectiveFisheye(CameraModel::model_id) &&
          !FisheyeCameraModelIsValidPixel(
              CameraModel::model_id, params, Eigen::Vector2d(x, y))) {
        continue;
      }
      TestCamFromImgToImg<CameraModel>(params, x, y);
      TestCamRayFromImgToImg<CameraModel>(params, x, y);
    }
  }

  const auto pp_idxs = CameraModel::principal_point_idxs;
  TestCamFromImgToImg<CameraModel>(
      params, params[pp_idxs.at(0)], params[pp_idxs.at(1)]);
  TestCamRayFromImgToImg<CameraModel>(
      params, params[pp_idxs.at(0)], params[pp_idxs.at(1)]);

  // Analytic ImgFromCamWithJac is validated separately in
  // models_jacobian_test.cc.
}

TEST(SimplePinhole, Nominal) {
  TestModel<SimplePinholeCameraModel>({655.123, 386.123, 511.123});
}

TEST(Pinhole, Nominal) {
  TestModel<PinholeCameraModel>({651.123, 655.123, 386.123, 511.123});
}

TEST(Spherical, Nominal) {
  // params = (w, h) of the equirectangular image.
  const std::vector<double> params = {800, 400};
  EXPECT_TRUE(
      CameraModelVerifyParams(EquirectangularCameraModel::model_id, params));

  EXPECT_EQ(CameraModelParamsInfo(EquirectangularCameraModel::model_id), "w,h");
  EXPECT_TRUE(
      CameraModelFocalLengthIdxs(EquirectangularCameraModel::model_id).empty());
  EXPECT_TRUE(
      CameraModelPrincipalPointIdxs(EquirectangularCameraModel::model_id)
          .empty());
  EXPECT_TRUE(
      CameraModelExtraParamsIdxs(EquirectangularCameraModel::model_id).empty());
  EXPECT_EQ(
      std::vector<size_t>(
          CameraModelMetaDataParamsIdxs(EquirectangularCameraModel::model_id)
              .begin(),
          CameraModelMetaDataParamsIdxs(EquirectangularCameraModel::model_id)
              .end()),
      (std::vector<size_t>{0, 1}));
  EXPECT_EQ(CameraModelNumParams(EquirectangularCameraModel::model_id), 2u);

  // Perspective models have no metadata parameters.
  EXPECT_TRUE(
      CameraModelMetaDataParamsIdxs(PinholeCameraModel::model_id).empty());

  // EQUIRECTANGULAR is non-perspective, spherical, and never has bogus
  // parameters.
  EXPECT_FALSE(CameraModelIsPerspective(EquirectangularCameraModel::model_id));
  EXPECT_TRUE(CameraModelIsPerspective(PinholeCameraModel::model_id));
  EXPECT_TRUE(CameraModelIsSpherical(EquirectangularCameraModel::model_id));
  EXPECT_FALSE(CameraModelIsSpherical(PinholeCameraModel::model_id));
  EXPECT_FALSE(CameraModelIsSpherical(OpenCVFisheyeCameraModel::model_id));
  EXPECT_FALSE(CameraModelHasBogusParams(
      EquirectangularCameraModel::model_id, params, 800, 400, 0.1, 2.0, 1.0));

  // InitializeParams ignores the focal length and returns (w, h).
  EXPECT_EQ(
      CameraModelInitializeParams(
          EquirectangularCameraModel::model_id, /*focal_length=*/123, 800, 400),
      params);

  // Full-sphere bearing round-trip CamRayFromImg -> ImgFromCam over the image
  // interior (avoiding the azimuth seam at x in {0, w} and the poles at
  // y in {0, h}, where the azimuth is undefined).
  for (int xi = 40; xi <= 760; xi += 40) {
    for (int yi = 40; yi <= 360; yi += 40) {
      TestCamRayFromImgToImg<EquirectangularCameraModel>(
          params, static_cast<double>(xi), static_cast<double>(yi));
    }
  }

  // Back-hemisphere pixels (azimuth near +/-pi) have no forward 2D
  // representation, so the 2D CamFromImg fails there while CamRayFromImg still
  // yields a valid unit bearing.
  EXPECT_FALSE(CameraModelCamFromImg(EquirectangularCameraModel::model_id,
                                     params,
                                     Eigen::Vector2d(0, 200))
                   .has_value());
  EXPECT_TRUE(CameraModelCamRayFromImg(EquirectangularCameraModel::model_id,
                                       params,
                                       Eigen::Vector2d(0, 200))
                  .has_value());

  // A forward-hemisphere pixel (azimuth ~0, image center column) also
  // round-trips through the 2D CamFromImg / ImgFromCam path.
  TestCamFromImgToImg<EquirectangularCameraModel>(params, 400, 200);
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

TEST(SimpleDivision, Nominal) {
  TestModel<SimpleDivisionCameraModel>({651.123, 386.123, 511.123, 0});
  TestModel<SimpleDivisionCameraModel>({651.123, 386.123, 511.123, 0.1});
  TestModel<SimpleDivisionCameraModel>({651.123, 386.123, 511.123, -0.1});
}

TEST(Division, Nominal) {
  TestModel<DivisionCameraModel>({651.123, 655.123, 386.123, 511.123, 0});
  TestModel<DivisionCameraModel>({651.123, 655.123, 386.123, 511.123, 0.1});
  TestModel<DivisionCameraModel>({651.123, 655.123, 386.123, 511.123, -0.1});
}

TEST(SimpleFisheyeCamera, Nominal) {
  TestModel<SimpleFisheyeCameraModel>({651.123, 386.123, 511.123});
}

TEST(FisheyeCamera, Nominal) {
  TestModel<FisheyeCameraModel>({651.123, 655.123, 386.123, 511.123});
}

TEST(EUCMCamera, Nominal) {
  TestModel<EUCMCameraModel>({651.123, 655.123, 386.123, 511.123, 0.56, 0.87});
  TestModel<EUCMCameraModel>({400, 400, 400, 400, 0.88, 0.64});
  TestModel<EUCMCameraModel>({651.123, 655.123, 386.123, 511.123, 0.0, 1.0});
  TestModel<EUCMCameraModel>({651.123, 655.123, 386.123, 511.123, 0.5, 1.0});
}

TEST(EUCMCamera, RejectsInvalidExtraParams) {
  EXPECT_TRUE(CameraModelHasBogusParams(
      EUCMCameraModel::model_id,
      {651.123, 655.123, 386.123, 511.123, -0.01, 0.87},
      1000,
      1000,
      0.1,
      2.0,
      1.0));
  EXPECT_TRUE(CameraModelHasBogusParams(
      EUCMCameraModel::model_id,
      {651.123, 655.123, 386.123, 511.123, 1.01, 0.87},
      1000,
      1000,
      0.1,
      2.0,
      1.0));
  EXPECT_TRUE(CameraModelHasBogusParams(
      EUCMCameraModel::model_id,
      {651.123, 655.123, 386.123, 511.123, 0.56, 0.00},
      1000,
      1000,
      0.1,
      2.0,
      1.0));
  EXPECT_TRUE(CameraModelHasBogusParams(
      EUCMCameraModel::model_id,
      {651.123, 655.123, 386.123, 511.123, 0.56, -0.3},
      1000,
      1000,
      0.1,
      2.0,
      1.0));
}

TEST(CameraModelRescale, Perspective) {
  // Distinct per-axis scale factors to verify each is applied to the right
  // parameter; all results are exactly representable.
  const double scale_x = 2.0;
  const double scale_y = 3.0;

  // Two focal lengths (fx, fy): each scales along its own axis, as does the
  // principal point (cx, cy).
  {
    std::vector<double> params = {100, 200, 50, 80};  // fx, fy, cx, cy
    CameraModelRescale(PinholeCameraModel::model_id, scale_x, scale_y, params);
    EXPECT_EQ(params, (std::vector<double>{200, 600, 100, 240}));
  }

  // Single shared focal length scales by the mean of the two factors.
  {
    std::vector<double> params = {100, 50, 80};  // f, cx, cy
    CameraModelRescale(
        SimplePinholeCameraModel::model_id, scale_x, scale_y, params);
    EXPECT_EQ(params, (std::vector<double>{250, 100, 240}));  // f *= 2.5
  }

  // Extra (distortion) parameters are resolution independent and untouched.
  {
    std::vector<double> params = {100, 50, 80, 0.3};  // f, cx, cy, k
    CameraModelRescale(
        SimpleRadialCameraModel::model_id, scale_x, scale_y, params);
    EXPECT_EQ(params, (std::vector<double>{250, 100, 240, 0.3}));
  }
}

TEST(CameraModelRescale, Spherical) {
  // The (w, h) image-size parameters track the rescaled image dimensions.
  std::vector<double> params = {800, 400};  // w, h
  CameraModelRescale(EquirectangularCameraModel::model_id,
                     /*scale_x=*/2.0,
                     /*scale_y=*/0.5,
                     params);
  EXPECT_EQ(params, (std::vector<double>{1600, 200}));
}

}  // namespace
}  // namespace colmap
