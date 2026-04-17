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

  if constexpr (CameraModel::has_img_from_cam_with_jac) {
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

// SphereCameraModel tests. SPHERE has no focal length or principal point —
// its only parameters are the image dimensions — so the generic TestModel
// helper doesn't apply. We test its invariants directly.

TEST(Sphere, Metadata) {
  const std::vector<double> params = {4096.0, 2048.0};

  EXPECT_TRUE(CameraModelVerifyParams(SphereCameraModel::model_id, params));
  EXPECT_EQ(CameraModelParamsInfo(SphereCameraModel::model_id),
            SphereCameraModel::params_info);
  EXPECT_EQ(SphereCameraModel::num_params, 2u);
  EXPECT_EQ(SphereCameraModel::focal_length_idxs.size(), 0u);
  EXPECT_EQ(SphereCameraModel::principal_point_idxs.size(), 0u);
  EXPECT_EQ(SphereCameraModel::extra_params_idxs.size(), 2u);

  EXPECT_TRUE(ExistsCameraModelWithName("SPHERE"));
  EXPECT_EQ(CameraModelNameToId("SPHERE"), SphereCameraModel::model_id);
  EXPECT_EQ(CameraModelIdToName(SphereCameraModel::model_id), "SPHERE");

  const std::vector<double> init_params =
      CameraModelInitializeParams(SphereCameraModel::model_id, 100, 4096, 2048);
  EXPECT_EQ(init_params, std::vector<double>({4096.0, 2048.0}));

  // HasBogusParams is overridden to always return false for SPHERE (the only
  // parameters are image dimensions, always valid by construction).
  EXPECT_FALSE(CameraModelHasBogusParams(
      SphereCameraModel::model_id, params, 4096, 2048, 0.1, 2.0, 1.0));
}

TEST(Sphere, CardinalDirections) {
  const std::vector<double> params = {4096.0, 2048.0};
  double x = 0, y = 0;

  // Forward (+Z) -> image center.
  ASSERT_TRUE(
      SphereCameraModel::ImgFromCam(params.data(), 0.0, 0.0, 1.0, &x, &y));
  EXPECT_NEAR(x, 2048.0, 1e-9);
  EXPECT_NEAR(y, 1024.0, 1e-9);

  // Right (+X) -> column 3W/4.
  ASSERT_TRUE(
      SphereCameraModel::ImgFromCam(params.data(), 1.0, 0.0, 0.0, &x, &y));
  EXPECT_NEAR(x, 3072.0, 1e-9);
  EXPECT_NEAR(y, 1024.0, 1e-9);

  // Left (-X) -> column W/4.
  ASSERT_TRUE(
      SphereCameraModel::ImgFromCam(params.data(), -1.0, 0.0, 0.0, &x, &y));
  EXPECT_NEAR(x, 1024.0, 1e-9);
  EXPECT_NEAR(y, 1024.0, 1e-9);

  // Up (-Y) -> row 0 (top).
  ASSERT_TRUE(
      SphereCameraModel::ImgFromCam(params.data(), 0.0, -1.0, 0.0, &x, &y));
  EXPECT_NEAR(x, 2048.0, 1e-9);
  EXPECT_NEAR(y, 0.0, 1e-9);

  // Down (+Y) -> row H (bottom).
  ASSERT_TRUE(
      SphereCameraModel::ImgFromCam(params.data(), 0.0, 1.0, 0.0, &x, &y));
  EXPECT_NEAR(x, 2048.0, 1e-9);
  EXPECT_NEAR(y, 2048.0, 1e-9);

  // Back (-Z) -> column 0 or W (image wraps).
  ASSERT_TRUE(
      SphereCameraModel::ImgFromCam(params.data(), 0.0, 0.0, -1.0, &x, &y));
  EXPECT_TRUE(std::abs(x) < 1e-9 || std::abs(x - 4096.0) < 1e-9);
  EXPECT_NEAR(y, 1024.0, 1e-9);

  // Zero vector -> rejected.
  EXPECT_FALSE(
      SphereCameraModel::ImgFromCam(params.data(), 0.0, 0.0, 0.0, &x, &y));
}

TEST(Sphere, RoundTripFrontHemisphere) {
  // ImgFromCam -> CamFromImg -> reconstructed ray.
  // CamFromImg is limited to the forward hemisphere (Z > 0) via the 2D
  // normalized-coordinate API, so restrict the sampled grid accordingly.
  const std::vector<double> params = {4096.0, 2048.0};
  int samples = 0;
  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (double u = -0.8; u <= 0.8; u += 0.2) {
    // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
    for (double v = -0.8; v <= 0.8; v += 0.2) {
      for (const double w : {0.5, 1.0, 2.0}) {
        double x = 0, y = 0;
        ASSERT_TRUE(
            SphereCameraModel::ImgFromCam(params.data(), u, v, w, &x, &y));
        double un = 0, vn = 0;
        ASSERT_TRUE(
            SphereCameraModel::CamFromImg(params.data(), x, y, &un, &vn));
        // (un, vn, 1) normalized must match (u, v, w) normalized.
        const double norm_uvw = std::sqrt(u * u + v * v + w * w);
        const double norm_n = std::sqrt(un * un + vn * vn + 1.0);
        EXPECT_NEAR(un / norm_n, u / norm_uvw, 1e-9);
        EXPECT_NEAR(vn / norm_n, v / norm_uvw, 1e-9);
        EXPECT_NEAR(1.0 / norm_n, w / norm_uvw, 1e-9);
        ++samples;
      }
    }
  }
  EXPECT_GT(samples, 0);
}

TEST(Sphere, CamFromImgRejectsBackHemisphere) {
  // Pixels near the horizontal edges of the image correspond to rays in the
  // back hemisphere (Z <= 0). The 2D normalized-coord interface cannot
  // represent those rays, so CamFromImg must return false.
  const std::vector<double> params = {4096.0, 2048.0};
  double u = 0, v = 0;
  EXPECT_FALSE(SphereCameraModel::CamFromImg(params.data(), 10.0, 1024.0, &u, &v));
  EXPECT_FALSE(
      SphereCameraModel::CamFromImg(params.data(), 4090.0, 1024.0, &u, &v));
  // Image center -> forward ray -> (0, 0) in normalized coords.
  ASSERT_TRUE(
      SphereCameraModel::CamFromImg(params.data(), 2048.0, 1024.0, &u, &v));
  EXPECT_NEAR(u, 0.0, 1e-9);
  EXPECT_NEAR(v, 0.0, 1e-9);
}

TEST(Sphere, CamFromImgThresholdIsAngular) {
  // For a W=4096 panorama, 1 pixel at the equator corresponds to
  // 2π / 4096 ≈ 0.00153 radians of azimuth.
  const std::vector<double> params = {4096.0, 2048.0};
  const double threshold_1px = CameraModelCamFromImgThreshold(
      SphereCameraModel::model_id, params, 1.0);
  EXPECT_NEAR(threshold_1px, 2.0 * M_PI / 4096.0, 1e-12);

  const double threshold_4px = CameraModelCamFromImgThreshold(
      SphereCameraModel::model_id, params, 4.0);
  EXPECT_NEAR(threshold_4px, 4.0 * threshold_1px, 1e-12);
}

TEST(Sphere, CamFromImgRayFullSphere) {
  // ImgFromCam -> CamFromImgRay should round-trip for ANY direction (all 4π
  // sr), unlike the 2D CamFromImg path which only covers the forward
  // hemisphere.
  const std::vector<double> params = {4096.0, 2048.0};
  int samples = 0;
  for (int ix = -5; ix <= 5; ++ix) {
    for (int iy = -5; iy <= 5; ++iy) {
      for (int iz = -5; iz <= 5; ++iz) {
        const double u = ix * 0.1;
        const double v = iy * 0.1;
        const double w = iz * 0.1;
        const double norm = std::sqrt(u * u + v * v + w * w);
        if (norm < 1e-6) continue;
        double x = 0, y = 0;
        ASSERT_TRUE(
            SphereCameraModel::ImgFromCam(params.data(), u, v, w, &x, &y));
        const std::optional<Eigen::Vector3d> ray = CameraModelCamFromImgRay(
            SphereCameraModel::model_id, params, Eigen::Vector2d(x, y));
        ASSERT_TRUE(ray.has_value());
        // Ray should be unit-length and parallel to the input direction.
        EXPECT_NEAR(ray->norm(), 1.0, 1e-12);
        EXPECT_NEAR(ray->x(), u / norm, 1e-9);
        EXPECT_NEAR(ray->y(), v / norm, 1e-9);
        EXPECT_NEAR(ray->z(), w / norm, 1e-9);
        ++samples;
      }
    }
  }
  EXPECT_GT(samples, 0);
}

TEST(Sphere, CamFromImgRayBackHemisphere) {
  // Back-hemisphere pixels fail via the 2D CamFromImg path but produce valid
  // unit rays via CamFromImgRay. This is the key reason for the new
  // interface.
  const std::vector<double> params = {4096.0, 2048.0};
  double u = 0, v = 0;
  // Near the left edge -> ray points backward.
  EXPECT_FALSE(
      SphereCameraModel::CamFromImg(params.data(), 10.0, 1024.0, &u, &v));
  const std::optional<Eigen::Vector3d> ray = CameraModelCamFromImgRay(
      SphereCameraModel::model_id, params, Eigen::Vector2d(10.0, 1024.0));
  ASSERT_TRUE(ray.has_value());
  EXPECT_NEAR(ray->norm(), 1.0, 1e-12);
  EXPECT_LT(ray->z(), 0.0);  // back hemisphere
}

// Default CamFromImgRay for perspective models should match the legacy
// CamFromImg -> homogeneous -> normalized conversion path used everywhere
// else in the codebase before this interface existed.
TEST(Perspective, CamFromImgRayMatchesLegacy) {
  struct Case {
    CameraModelId id;
    std::vector<double> params;
    const char* name;
  };
  const Case cases[] = {
      {SimplePinholeCameraModel::model_id,
       {655.123, 386.123, 511.123},
       "SimplePinhole"},
      {PinholeCameraModel::model_id,
       {651.123, 655.123, 386.123, 511.123},
       "Pinhole"},
      {SimpleRadialCameraModel::model_id,
       {651.123, 386.123, 511.123, 0.1},
       "SimpleRadial"},
      {OpenCVCameraModel::model_id,
       {651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001},
       "OpenCV"},
  };
  for (const auto& c : cases) {
    for (int xi = 100; xi <= 700; xi += 100) {
      for (int yi = 100; yi <= 700; yi += 100) {
        const Eigen::Vector2d xy(xi, yi);
        const std::optional<Eigen::Vector3d> ray =
            CameraModelCamFromImgRay(c.id, c.params, xy);
        const std::optional<Eigen::Vector2d> cam_point =
            CameraModelCamFromImg(c.id, c.params, xy);
        ASSERT_EQ(ray.has_value(), cam_point.has_value())
            << c.name << " at (" << xi << "," << yi << ")";
        if (ray.has_value()) {
          const Eigen::Vector3d expected =
              cam_point->homogeneous().normalized();
          EXPECT_NEAR(ray->x(), expected.x(), 1e-12) << c.name;
          EXPECT_NEAR(ray->y(), expected.y(), 1e-12) << c.name;
          EXPECT_NEAR(ray->z(), expected.z(), 1e-12) << c.name;
          EXPECT_NEAR(ray->norm(), 1.0, 1e-12) << c.name;
        }
      }
    }
  }
}

TEST(Sphere, ImgFromCamHandlesAllDirections) {
  // ImgFromCam must accept any non-zero direction, including Z <= 0
  // (unlike pinhole/fisheye which require the point to be in front of the
  // camera). This is a key property of the spherical camera.
  const std::vector<double> params = {4096.0, 2048.0};
  const double directions[][3] = {
      {0, 0, 1},    // forward
      {0, 0, -1},   // backward  <-- would fail for pinhole
      {1, 0, 0},    // right
      {-1, 0, 0},   // left
      {0, 1, 0},    // down
      {0, -1, 0},   // up
      {1, 1, 1},    // diagonal
      {-1, -1, -1}, // diagonal, back
  };
  for (const auto& d : directions) {
    double x = 0, y = 0;
    EXPECT_TRUE(SphereCameraModel::ImgFromCam(
        params.data(), d[0], d[1], d[2], &x, &y))
        << "direction (" << d[0] << ", " << d[1] << ", " << d[2] << ")";
    EXPECT_GE(x, 0.0);
    EXPECT_LE(x, 4096.0);
    EXPECT_GE(y, 0.0);
    EXPECT_LE(y, 2048.0);
  }
}

}  // namespace
}  // namespace colmap
