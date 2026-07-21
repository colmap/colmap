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

#include <ceres/ceres.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Validate ImgFromCamWithJac against ImgFromCam using Ceres Jets.
template <typename CameraModel>
void TestImgFromCamWithJac(const std::vector<double>& params,
                           const double u,
                           const double v,
                           const double w) {
  constexpr size_t kNumParams = CameraModel::num_params;
  constexpr size_t kNumUvw = 3;
  constexpr size_t kNumDerivs = kNumParams + kNumUvw;

  // Compute using ImgFromCamWithJac
  double x_jac = 0, y_jac = 0;
  double J_params[2 * kNumParams];
  double J_uvw[2 * kNumUvw];
  ASSERT_TRUE(CameraModel::ImgFromCamWithJac(
      params.data(), u, v, w, &x_jac, &y_jac, J_params, J_uvw));

  // Compute using ImgFromCam with Ceres Jets for auto-differentiation
  // Jets track derivatives: first kNumParams for params, next 3 for u, v, w.
  using JetT = ceres::Jet<double, kNumDerivs>;

  JetT params_jet[kNumParams];
  for (size_t i = 0; i < kNumParams; ++i) {
    params_jet[i] = JetT(params[i], i);
  }

  JetT u_jet(u, kNumParams);
  JetT v_jet(v, kNumParams + 1);
  JetT w_jet(w, kNumParams + 2);

  JetT x_jet, y_jet;
  ASSERT_TRUE(
      CameraModel::ImgFromCam(params_jet, u_jet, v_jet, w_jet, &x_jet, &y_jet));

  // Compare function values
  EXPECT_NEAR(x_jac, x_jet.a, 1e-10);
  EXPECT_NEAR(y_jac, y_jet.a, 1e-10);

  // Compare Jacobian w.r.t. params (2 x num_params, row-major)
  for (size_t i = 0; i < kNumParams; ++i) {
    EXPECT_NEAR(J_params[i], x_jet.v[i], 1e-10)
        << "J_params mismatch at dx/dparam[" << i << "]";
    EXPECT_NEAR(J_params[kNumParams + i], y_jet.v[i], 1e-10)
        << "J_params mismatch at dy/dparam[" << i << "]";
  }

  // Compare Jacobian w.r.t. uvw (2 x 3, row-major)
  for (size_t i = 0; i < kNumUvw; ++i) {
    EXPECT_NEAR(J_uvw[i], x_jet.v[kNumParams + i], 1e-10)
        << "J_uvw mismatch at dx/d(uvw)[" << i << "]";
    EXPECT_NEAR(J_uvw[kNumUvw + i], y_jet.v[kNumParams + i], 1e-10)
        << "J_uvw mismatch at dy/d(uvw)[" << i << "]";
  }
}

// Validate the analytic ImgFromCamWithJac over a grid of camera-space points.
template <typename CameraModel>
void TestModelImgFromCamWithJac(const std::vector<double>& params) {
  static_assert(CameraModel::has_img_from_cam_with_jac,
                "Model does not provide an analytic ImgFromCamWithJac");
  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double u = -0.5; u <= 0.5; u += 0.1) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double v = -0.5; v <= 0.5; v += 0.1) {
      for (const double w : {0.5, 1.0, 2.0}) {
        TestImgFromCamWithJac<CameraModel>(params, u, v, w);
      }
    }
  }
}

TEST(SimplePinhole, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<SimplePinholeCameraModel>(
      {655.123, 386.123, 511.123});
}

TEST(Pinhole, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<PinholeCameraModel>(
      {651.123, 655.123, 386.123, 511.123});
}

TEST(SimpleRadial, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<SimpleRadialCameraModel>(
      {651.123, 386.123, 511.123, 0});
  TestModelImgFromCamWithJac<SimpleRadialCameraModel>(
      {651.123, 386.123, 511.123, 0.1});
}

TEST(Radial, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<RadialCameraModel>(
      {651.123, 386.123, 511.123, 0, 0});
  TestModelImgFromCamWithJac<RadialCameraModel>(
      {651.123, 386.123, 511.123, 0.1, 0});
  TestModelImgFromCamWithJac<RadialCameraModel>(
      {651.123, 386.123, 511.12, 0, 0.05});
  TestModelImgFromCamWithJac<RadialCameraModel>(
      {651.123, 386.123, 511.123, 0.05, 0.03});
}

TEST(OpenCV, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<OpenCVCameraModel>(
      {651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001});
}

TEST(FullOpenCV, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<FullOpenCVCameraModel>({651.123,
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

TEST(FOV, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<FOVCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0.9});
  TestModelImgFromCamWithJac<FOVCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0.5});
}

TEST(SimpleRadialFisheye, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<SimpleRadialFisheyeCameraModel>(
      {651.123, 386.123, 511.123, 0});
  TestModelImgFromCamWithJac<SimpleRadialFisheyeCameraModel>(
      {651.123, 386.123, 511.123, 0.1});
}

TEST(RadialFisheye, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<RadialFisheyeCameraModel>(
      {651.123, 386.123, 511.123, 0, 0});
  TestModelImgFromCamWithJac<RadialFisheyeCameraModel>(
      {651.123, 386.123, 511.123, 0.1, 0.02});
}

TEST(OpenCVFisheye, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<OpenCVFisheyeCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0, 0, 0, 0});
  TestModelImgFromCamWithJac<OpenCVFisheyeCameraModel>(
      {651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001});
}

TEST(ThinPrismFisheye, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<ThinPrismFisheyeCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0, 0, 0, 0, 0, 0, 0, 0});
  TestModelImgFromCamWithJac<ThinPrismFisheyeCameraModel>({651.123,
                                                           655.123,
                                                           386.123,
                                                           511.123,
                                                           -0.05,
                                                           0.02,
                                                           -0.001,
                                                           0.001,
                                                           0.001,
                                                           0.002,
                                                           0.001,
                                                           -0.001});
}

TEST(RadTanThinPrismFisheye, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<RadTanThinPrismFisheyeModel>(
      {651.123, 655.123, 386.123, 511.123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  TestModelImgFromCamWithJac<RadTanThinPrismFisheyeModel>({651.123,
                                                           655.123,
                                                           386.123,
                                                           511.123,
                                                           -0.05,
                                                           0.02,
                                                           -0.005,
                                                           0.001,
                                                           0.0005,
                                                           0.0002,
                                                           -0.001,
                                                           0.001,
                                                           0.001,
                                                           -0.001,
                                                           0.0005,
                                                           -0.0005});
}

TEST(SimpleFisheye, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<SimpleFisheyeCameraModel>(
      {651.123, 386.123, 511.123});
}

TEST(Fisheye, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<FisheyeCameraModel>(
      {651.123, 655.123, 386.123, 511.123});
}

TEST(SimpleDivision, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<SimpleDivisionCameraModel>(
      {651.123, 386.123, 511.123, 0});
  TestModelImgFromCamWithJac<SimpleDivisionCameraModel>(
      {651.123, 386.123, 511.123, 0.1});
  TestModelImgFromCamWithJac<SimpleDivisionCameraModel>(
      {651.123, 386.123, 511.123, -0.1});
}

TEST(Division, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<DivisionCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0});
  TestModelImgFromCamWithJac<DivisionCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0.1});
  TestModelImgFromCamWithJac<DivisionCameraModel>(
      {651.123, 655.123, 386.123, 511.123, -0.1});
}

TEST(EUCM, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<EUCMCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0.0, 1.0});
  TestModelImgFromCamWithJac<EUCMCameraModel>(
      {651.123, 655.123, 386.123, 511.123, 0.6, 1.2});
}

TEST(Equirectangular, ImgFromCamWithJac) {
  TestModelImgFromCamWithJac<EquirectangularCameraModel>({1000, 500});
}

}  // namespace
}  // namespace colmap
