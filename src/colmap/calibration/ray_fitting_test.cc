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

#include "colmap/calibration/ray_fitting.h"

#include "colmap/scene/camera.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Unproject a pixel-center grid with known parameters to synthesize the
// dense image-point/camera-ray correspondences the network would predict.
void SynthesizeCorrespondences(const Camera& camera,
                               int width,
                               int height,
                               std::vector<Eigen::Vector2d>* img_points,
                               std::vector<Eigen::Vector3d>* cam_rays) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Eigen::Vector2d img_point(x + 0.5, y + 0.5);
      const std::optional<Eigen::Vector3d> cam_ray =
          camera.CamRayFromImg(img_point);
      if (cam_ray.has_value()) {
        img_points->push_back(img_point);
        cam_rays->push_back(cam_ray->normalized());
      }
    }
  }
}

struct RayFittingCase {
  std::string model_name;
  std::vector<double> params;
  // Tolerance for distortion parameters. High-order polynomial coefficients
  // (FULL_OPENCV's k1..k6 up to r^12) are weakly identifiable, admitting
  // projection-equivalent solutions; the projection check below is the
  // binding assertion for those.
  double distortion_tol = 1e-4;
};

class RayFittingTest : public ::testing::TestWithParam<RayFittingCase> {};

TEST_P(RayFittingTest, RoundTripRecoversParameters) {
  const RayFittingCase& test_case = GetParam();
  const CameraModelId model_id = CameraModelNameToId(test_case.model_name);
  Camera camera;
  camera.model_id = model_id;
  camera.width = 64;
  camera.height = 64;
  camera.params = test_case.params;
  ASSERT_TRUE(camera.VerifyParams());

  std::vector<Eigen::Vector2d> img_points;
  std::vector<Eigen::Vector3d> cam_rays;
  SynthesizeCorrespondences(
      camera, camera.width, camera.height, &img_points, &cam_rays);
  ASSERT_GT(img_points.size(), 1000);

  RayFittingOptions options;
  const FittedCamera fitted =
      FitCameraFromRays(model_id, img_points, cam_rays, options);
  ASSERT_TRUE(fitted.success);
  ASSERT_EQ(fitted.params.size(), test_case.params.size());
  EXPECT_TRUE(std::all_of(
      fitted.params.begin(), fitted.params.end(), [](const double param) {
        return std::isfinite(param);
      }));
  EXPECT_LT(fitted.final_cost, fitted.initial_cost);
  EXPECT_LT(fitted.final_cost, 1e-10);
  const span<const size_t> extra_idxs = CameraModelExtraParamsIdxs(model_id);
  const std::unordered_set<size_t> extra_set(extra_idxs.begin(),
                                             extra_idxs.end());
  for (size_t i = 0; i < test_case.params.size(); ++i) {
    const double tol = extra_set.count(i) ? test_case.distortion_tol : 1e-4;
    EXPECT_NEAR(fitted.params[i], test_case.params[i], tol)
        << "param " << i << " of " << test_case.model_name;
  }
  // The fitted model must project identically to the ground truth.
  for (size_t i = 0; i < img_points.size(); ++i) {
    const std::optional<Eigen::Vector2d> projected =
        CameraModelImgFromCam(model_id, fitted.params, cam_rays[i]);
    ASSERT_TRUE(projected.has_value());
    EXPECT_LT((projected.value() - img_points[i]).norm(), 1e-3)
        << "projection mismatch for " << test_case.model_name;
  }
}

TEST(RayFittingTest, EUCMConstraintsHoldForMismatchedRays) {
  Camera source_camera;
  source_camera.model_id = CameraModelId::kFOV;
  source_camera.width = 64;
  source_camera.height = 64;
  source_camera.params = {60, 61, 31, 33, 0.7};

  std::vector<Eigen::Vector2d> img_points;
  std::vector<Eigen::Vector3d> cam_rays;
  SynthesizeCorrespondences(source_camera,
                            source_camera.width,
                            source_camera.height,
                            &img_points,
                            &cam_rays);

  const FittedCamera fitted = FitCameraFromRays(
      CameraModelId::kEUCM, img_points, cam_rays, RayFittingOptions());
  ASSERT_TRUE(fitted.success);
  ASSERT_EQ(fitted.params.size(), EUCMCameraModel::num_params);
  EXPECT_GE(fitted.params[EUCMCameraModel::extra_params_idxs[0]], 0.0);
  EXPECT_LE(fitted.params[EUCMCameraModel::extra_params_idxs[0]], 1.0);
  EXPECT_GT(fitted.params[EUCMCameraModel::extra_params_idxs[1]], 0.0);
}

TEST(RayFittingTest, FocalLengthPriorPullsEstimate) {
  Camera camera;
  camera.model_id = CameraModelId::kSimplePinhole;
  camera.width = 64;
  camera.height = 64;
  camera.params = {60, 32, 32};

  std::vector<Eigen::Vector2d> img_points;
  std::vector<Eigen::Vector3d> cam_rays;
  SynthesizeCorrespondences(
      camera, camera.width, camera.height, &img_points, &cam_rays);

  RayFittingOptions options;
  options.prior_focal_length_weight = 1.0;
  const std::vector<double> prior_focal_lengths = {80};
  const FittedCamera fitted = FitCameraFromRays(CameraModelId::kSimplePinhole,
                                                img_points,
                                                cam_rays,
                                                options,
                                                prior_focal_lengths);

  ASSERT_TRUE(fitted.success);
  EXPECT_GT(fitted.params[0], camera.params[0]);
  EXPECT_LT(fitted.params[0], prior_focal_lengths[0]);
}

INSTANTIATE_TEST_SUITE_P(
    AllPerspectiveModels,
    RayFittingTest,
    ::testing::Values(
        RayFittingCase{"SIMPLE_PINHOLE", {60, 32, 32}},
        RayFittingCase{"PINHOLE", {60, 61, 31, 33}},
        RayFittingCase{"SIMPLE_RADIAL", {60, 32, 32, 0.05}},
        RayFittingCase{"RADIAL", {60, 32, 32, 0.05, -0.01}},
        RayFittingCase{"OPENCV", {60, 61, 31, 33, 0.05, -0.01, 0.001, -0.002}},
        RayFittingCase{"FULL_OPENCV",
                       {60,
                        61,
                        31,
                        33,
                        0.05,
                        -0.01,
                        0.001,
                        -0.002,
                        0.001,
                        -0.001,
                        0.0005,
                        0.0002},
                       /*distortion_tol=*/1e-2},
        RayFittingCase{"FOV", {60, 61, 31, 33, 0.7}},
        RayFittingCase{"FOV", {60, 61, 31, 33, 0.05}},
        RayFittingCase{"SIMPLE_DIVISION", {60, 32, 32, -1e-5}},
        RayFittingCase{"DIVISION", {60, 61, 31, 33, -1e-5}},
        RayFittingCase{"SIMPLE_FISHEYE", {60, 32, 32}},
        RayFittingCase{"FISHEYE", {60, 61, 31, 33}},
        RayFittingCase{"OPENCV_FISHEYE", {60, 61, 31, 33, 0.01, -0.001, 0, 0}},
        RayFittingCase{"SIMPLE_RADIAL_FISHEYE", {60, 32, 32, 0.01}},
        RayFittingCase{"RADIAL_FISHEYE", {60, 32, 32, 0.01, -0.001}},
        RayFittingCase{"THIN_PRISM_FISHEYE",
                       {60,
                        61,
                        31,
                        33,
                        0.01,
                        -0.001,
                        0.0005,
                        -0.0005,
                        0.0002,
                        -0.0002,
                        0.0001,
                        0.0001}},
        RayFittingCase{"RAD_TAN_THIN_PRISM_FISHEYE",
                       {60,
                        61,
                        31,
                        33,
                        0.01,
                        -0.001,
                        0.0005,
                        -0.0005,
                        0.0002,
                        -0.0002,
                        0.0001,
                        0.0001,
                        0.00005,
                        -0.00005,
                        0.00002,
                        0.00002}},
        RayFittingCase{"EUCM", {60, 61, 31, 33, 0.6, 2.0}}));

TEST(RayFittingDeathTest, RejectsNonPerspectiveModel) {
  const CameraModelId model_id = CameraModelId::kEquirectangular;
  const std::vector<Eigen::Vector2d> img_points(10, Eigen::Vector2d(1, 1));
  const std::vector<Eigen::Vector3d> cam_rays(10, Eigen::Vector3d(0, 0, 1));
  const FittedCamera fitted =
      FitCameraFromRays(model_id, img_points, cam_rays, RayFittingOptions());
  EXPECT_FALSE(fitted.success);
}

TEST(ReverseScaleAndShiftParamsTest, InvertsDigitizingTransform) {
  // 640x480 image center-cropped to 480-square and resized to 322.
  const Eigen::Vector2d scale(322.0 / 480, 322.0 / 480);
  const Eigen::Vector2d shift(-80 * 322.0 / 480, 0);
  const std::vector<double> params322 = {300, 161, 161, 0.05};
  const std::vector<double> params = ReverseScaleAndShiftParams(
      CameraModelId::kSimpleRadial, params322, scale, shift);
  ASSERT_EQ(params.size(), 4);
  EXPECT_NEAR(params[0], 300 / scale.x(), 1e-9);
  EXPECT_NEAR(params[1], (161 - shift.x()) / scale.x(), 1e-9);
  EXPECT_NEAR(params[2], 161 / scale.y(), 1e-9);
  // Distortion is scale-invariant.
  EXPECT_EQ(params[3], 0.05);
}

TEST(StrideSubsampleIndicesTest, CoversBounds) {
  EXPECT_EQ(StrideSubsampleIndices(0, 100).size(), 0);
  const auto all = StrideSubsampleIndices(10, 100);
  ASSERT_EQ(all.size(), 10);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(all[i], i);
  }
  const auto sub = StrideSubsampleIndices(103684, 16384);
  EXPECT_LE(sub.size(), 16384);
  EXPECT_GT(sub.size(), 10000);
  EXPECT_EQ(sub.front(), 0);
  for (size_t i = 1; i < sub.size(); ++i) {
    EXPECT_GT(sub[i], sub[i - 1]);
  }
}

}  // namespace
}  // namespace colmap
