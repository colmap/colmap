// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/image/warp.h"

#include "colmap/math/random.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {
namespace {

const Bitmap GenerateRandomBitmap(const int width,
                                  const int height,
                                  const bool as_rgb) {
  Bitmap bitmap(width, height, as_rgb);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      BitmapColor<uint8_t> color;
      color.r = RandomUniformInteger<int>(0, 255);
      color.g = RandomUniformInteger<int>(0, 255);
      color.b = RandomUniformInteger<int>(0, 255);
      bitmap.SetPixel(x, y, color);
    }
  }
  return bitmap;
}

void CheckBitmapsExactlyEqual(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  EXPECT_EQ(bitmap1.Width(), bitmap2.Width());
  EXPECT_EQ(bitmap1.Height(), bitmap2.Height());
  EXPECT_EQ(bitmap1.IsRGB(), bitmap2.IsRGB());
  EXPECT_EQ(bitmap1.RowMajorData(), bitmap2.RowMajorData());
}

WarpImageOptions DirectWarpMinScaleOptions(const double min_scale) {
  WarpImageOptions options;
  options.direct_warp_min_scale = min_scale;
  return options;
}

// Check that the two bitmaps are equal, ignoring a 1px boundary.
void CheckBitmapsEqual(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  ASSERT_EQ(bitmap1.IsGrey(), bitmap2.IsGrey());
  ASSERT_EQ(bitmap1.IsRGB(), bitmap2.IsRGB());
  ASSERT_EQ(bitmap1.Width(), bitmap2.Width());
  ASSERT_EQ(bitmap1.Height(), bitmap2.Height());
  for (int x = 1; x < bitmap1.Width() - 1; ++x) {
    for (int y = 1; y < bitmap1.Height() - 1; ++y) {
      const auto color1 = bitmap1.GetPixel(x, y);
      const auto color2 = bitmap2.GetPixel(x, y);
      ASSERT_TRUE(color1.has_value());
      ASSERT_TRUE(color2.has_value());
      EXPECT_EQ(*color1, *color2);
    }
  }
}

// Check that the two bitmaps are equal, ignoring a 1px boundary.
void CheckBitmapsTransposed(const Bitmap& bitmap1, const Bitmap& bitmap2) {
  ASSERT_EQ(bitmap1.IsGrey(), bitmap2.IsGrey());
  ASSERT_EQ(bitmap1.IsRGB(), bitmap2.IsRGB());
  ASSERT_EQ(bitmap1.Width(), bitmap2.Width());
  ASSERT_EQ(bitmap1.Height(), bitmap2.Height());
  for (int x = 1; x < bitmap1.Width() - 1; ++x) {
    for (int y = 1; y < bitmap1.Height() - 1; ++y) {
      const auto color1 = bitmap1.GetPixel(x, y);
      const auto color2 = bitmap2.GetPixel(y, x);
      ASSERT_TRUE(color1.has_value());
      ASSERT_TRUE(color2.has_value());
      EXPECT_EQ(*color1, *color2);
    }
  }
}

}  // namespace

TEST(Warp, IdenticalCameras) {
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 1, 100, 100);
  const Bitmap source_image_gray = GenerateRandomBitmap(100, 100, false);
  Bitmap target_image_gray;
  WarpImageBetweenCameras(WarpImageOptions(),
                          camera,
                          camera,
                          source_image_gray,
                          &target_image_gray);
  CheckBitmapsEqual(source_image_gray, target_image_gray);
  const Bitmap source_image_rgb = GenerateRandomBitmap(100, 100, true);
  Bitmap target_image_rgb;
  WarpImageBetweenCameras(
      WarpImageOptions(), camera, camera, source_image_rgb, &target_image_rgb);
  CheckBitmapsEqual(source_image_rgb, target_image_rgb);
}

TEST(Warp, DirectWarpMinScale) {
  const Camera source_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 100, 100, 100);
  const Bitmap source_image = GenerateRandomBitmap(100, 100, true);

  Camera target_camera = source_camera;
  target_camera.Rescale(50, 50);
  Bitmap default_image;
  WarpImageBetweenCameras(WarpImageOptions(),
                          source_camera,
                          target_camera,
                          source_image,
                          &default_image);
  Bitmap direct_image;
  WarpImageBetweenCameras(DirectWarpMinScaleOptions(0.0),
                          source_camera,
                          target_camera,
                          source_image,
                          &direct_image);
  Bitmap resized_image;
  WarpImageBetweenCameras(DirectWarpMinScaleOptions(1.0),
                          source_camera,
                          target_camera,
                          source_image,
                          &resized_image);
  CheckBitmapsExactlyEqual(default_image, direct_image);
  EXPECT_NE(default_image.RowMajorData(), resized_image.RowMajorData());

  target_camera = source_camera;
  target_camera.Rescale(49, 49);
  WarpImageBetweenCameras(WarpImageOptions(),
                          source_camera,
                          target_camera,
                          source_image,
                          &default_image);
  WarpImageBetweenCameras(DirectWarpMinScaleOptions(1.0),
                          source_camera,
                          target_camera,
                          source_image,
                          &resized_image);
  CheckBitmapsExactlyEqual(default_image, resized_image);

  target_camera = source_camera;
  target_camera.Rescale(95, 49);
  WarpImageBetweenCameras(WarpImageOptions(),
                          source_camera,
                          target_camera,
                          source_image,
                          &default_image);
  WarpImageBetweenCameras(DirectWarpMinScaleOptions(1.0),
                          source_camera,
                          target_camera,
                          source_image,
                          &resized_image);
  CheckBitmapsExactlyEqual(default_image, resized_image);

  EXPECT_ANY_THROW(WarpImageBetweenCameras(DirectWarpMinScaleOptions(-0.1),
                                           source_camera,
                                           target_camera,
                                           source_image,
                                           &default_image));
}

TEST(Warp, Interpolation) {
  const Camera source_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 4, 4, 4);
  Camera target_camera = source_camera;
  target_camera.SetPrincipalPointX(1.5);

  Bitmap source_image(4, 4, /*as_rgb=*/false);
  for (int y = 0; y < source_image.Height(); ++y) {
    source_image.SetPixel(0, y, BitmapColor<uint8_t>(0));
    source_image.SetPixel(1, y, BitmapColor<uint8_t>(100));
  }

  WarpImageOptions options;
  Bitmap bilinear_image;
  WarpImageBetweenCameras(
      options, source_camera, target_camera, source_image, &bilinear_image);
  ASSERT_TRUE(bilinear_image.GetPixel(0, 0).has_value());
  EXPECT_EQ(bilinear_image.GetPixel(0, 0)->r, 50);

  options.interpolation = WarpImageOptions::Interpolation::kNearestNeighbor;
  Bitmap nearest_image;
  WarpImageBetweenCameras(
      options, source_camera, target_camera, source_image, &nearest_image);
  ASSERT_TRUE(nearest_image.GetPixel(0, 0).has_value());
  EXPECT_EQ(nearest_image.GetPixel(0, 0)->r, 100);

  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  options.interpolation = static_cast<WarpImageOptions::Interpolation>(-1);
  EXPECT_ANY_THROW(WarpImageBetweenCameras(
      options, source_camera, target_camera, source_image, &nearest_image));
}

TEST(Warp, ShiftedCameras) {
  const Camera source_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 1, 100, 100);
  Camera target_camera = source_camera;
  target_camera.SetPrincipalPointX(0.0);
  const Bitmap source_image_gray = GenerateRandomBitmap(100, 100, true);
  Bitmap target_image_gray;
  WarpImageBetweenCameras(WarpImageOptions(),
                          source_camera,
                          target_camera,
                          source_image_gray,
                          &target_image_gray);
  for (int x = 0; x < target_image_gray.Width(); ++x) {
    for (int y = 0; y < target_image_gray.Height(); ++y) {
      const auto color = target_image_gray.GetPixel(x, y);
      ASSERT_TRUE(color.has_value());
      if (x >= 50) {
        EXPECT_EQ(*color, BitmapColor<uint8_t>(0));
      } else {
        const auto source_color = source_image_gray.GetPixel(x + 50, y);
        if (source_color && *color != BitmapColor<uint8_t>(0)) {
          EXPECT_EQ(*color, *source_color);
        }
      }
    }
  }
}

TEST(Warp, WarpImageWithHomographyIdentity) {
  const Bitmap source_image_gray = GenerateRandomBitmap(100, 100, false);
  Bitmap target_image_gray(100, 100, false);
  WarpImageWithHomography(
      Eigen::Matrix3d::Identity(), source_image_gray, &target_image_gray);
  CheckBitmapsEqual(source_image_gray, target_image_gray);

  const Bitmap source_image_rgb = GenerateRandomBitmap(100, 100, true);
  Bitmap target_image_rgb(100, 100, true);
  WarpImageWithHomography(
      Eigen::Matrix3d::Identity(), source_image_rgb, &target_image_rgb);
  CheckBitmapsEqual(source_image_rgb, target_image_rgb);
}

TEST(Warp, WarpImageWithHomographyTransposed) {
  Eigen::Matrix3d H;
  H << 0, 1, 0, 1, 0, 0, 0, 0, 1;

  const Bitmap source_image_gray = GenerateRandomBitmap(100, 100, false);
  Bitmap target_image_gray(100, 100, false);
  WarpImageWithHomography(H, source_image_gray, &target_image_gray);
  CheckBitmapsTransposed(source_image_gray, target_image_gray);

  const Bitmap source_image_rgb = GenerateRandomBitmap(100, 100, true);
  Bitmap target_image_rgb(100, 100, true);
  WarpImageWithHomography(H, source_image_rgb, &target_image_rgb);
  CheckBitmapsTransposed(source_image_rgb, target_image_rgb);
}

TEST(Warp, WarpImageWithHomographyBetweenCamerasIdentity) {
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 1, 100, 100);
  const Bitmap source_image_gray = GenerateRandomBitmap(100, 100, false);
  Bitmap target_image_gray;
  WarpImageWithHomographyBetweenCameras(WarpImageOptions(),
                                        Eigen::Matrix3d::Identity(),
                                        camera,
                                        camera,
                                        source_image_gray,
                                        &target_image_gray);
  CheckBitmapsEqual(source_image_gray, target_image_gray);

  const Bitmap source_image_rgb = GenerateRandomBitmap(100, 100, true);
  Bitmap target_image_rgb;
  WarpImageWithHomographyBetweenCameras(WarpImageOptions(),
                                        Eigen::Matrix3d::Identity(),
                                        camera,
                                        camera,
                                        source_image_rgb,
                                        &target_image_rgb);
  CheckBitmapsEqual(source_image_rgb, target_image_rgb);
}

TEST(Warp, HomographyDirectWarpMinScale) {
  const Camera source_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 100, 100, 100);
  const Bitmap source_image = GenerateRandomBitmap(100, 100, false);

  Camera target_camera = source_camera;
  target_camera.Rescale(50, 50);
  Bitmap default_image;
  WarpImageWithHomographyBetweenCameras(WarpImageOptions(),
                                        Eigen::Matrix3d::Identity(),
                                        source_camera,
                                        target_camera,
                                        source_image,
                                        &default_image);
  Bitmap direct_image;
  WarpImageWithHomographyBetweenCameras(DirectWarpMinScaleOptions(0.0),
                                        Eigen::Matrix3d::Identity(),
                                        source_camera,
                                        target_camera,
                                        source_image,
                                        &direct_image);
  Bitmap resized_image;
  WarpImageWithHomographyBetweenCameras(DirectWarpMinScaleOptions(1.0),
                                        Eigen::Matrix3d::Identity(),
                                        source_camera,
                                        target_camera,
                                        source_image,
                                        &resized_image);
  CheckBitmapsExactlyEqual(default_image, direct_image);
  EXPECT_NE(default_image.RowMajorData(), resized_image.RowMajorData());

  target_camera = source_camera;
  target_camera.Rescale(49, 49);
  WarpImageWithHomographyBetweenCameras(WarpImageOptions(),
                                        Eigen::Matrix3d::Identity(),
                                        source_camera,
                                        target_camera,
                                        source_image,
                                        &default_image);
  WarpImageWithHomographyBetweenCameras(DirectWarpMinScaleOptions(1.0),
                                        Eigen::Matrix3d::Identity(),
                                        source_camera,
                                        target_camera,
                                        source_image,
                                        &resized_image);
  CheckBitmapsExactlyEqual(default_image, resized_image);

  EXPECT_ANY_THROW(
      WarpImageWithHomographyBetweenCameras(DirectWarpMinScaleOptions(-0.1),
                                            Eigen::Matrix3d::Identity(),
                                            source_camera,
                                            target_camera,
                                            source_image,
                                            &default_image));
}

TEST(Warp, HomographyDownsamplingIdentity) {
  const Camera source_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 100, 100, 80);
  const Bitmap source_image = GenerateRandomBitmap(100, 80, true);
  for (const size_t height : {20, 32}) {
    Camera target_camera = source_camera;
    target_camera.Rescale(40, height);
    for (const auto interpolation :
         {WarpImageOptions::Interpolation::kNearestNeighbor,
          WarpImageOptions::Interpolation::kBilinear}) {
      WarpImageOptions options;
      options.interpolation = interpolation;
      Bitmap expected;
      WarpImageBetweenCameras(
          options, source_camera, target_camera, source_image, &expected);
      Bitmap actual;
      WarpImageWithHomographyBetweenCameras(options,
                                            Eigen::Matrix3d::Identity(),
                                            source_camera,
                                            target_camera,
                                            source_image,
                                            &actual);
      CheckBitmapsExactlyEqual(expected, actual);
    }
  }
}

TEST(Warp, HomographyDownsamplingCoordinates) {
  const Camera source_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 100, 100, 80);
  Bitmap source_image(100, 80, false);
  for (int y = 0; y < source_image.Height(); ++y) {
    for (int x = 0; x < source_image.Width(); ++x) {
      source_image.SetPixel(x, y, BitmapColor<uint8_t>(x + y));
    }
  }
  for (const size_t height : {20, 32}) {
    Camera target_camera = source_camera;
    target_camera.Rescale(40, height);
    for (const double perspective : {0.0, 0.001}) {
      Eigen::Matrix3d H;
      H << 1, 0.1, 2.5, -0.1, 1, 1.25, perspective, 0, 1;
      Bitmap direct_image;
      WarpImageWithHomographyBetweenCameras(DirectWarpMinScaleOptions(0.0),
                                            H,
                                            source_camera,
                                            target_camera,
                                            source_image,
                                            &direct_image);
      Bitmap resized_image;
      WarpImageWithHomographyBetweenCameras(WarpImageOptions(),
                                            H,
                                            source_camera,
                                            target_camera,
                                            source_image,
                                            &resized_image);
      // Only compare the interior: resampling near the borders differs
      // between the direct and resize paths, and more so on the right where
      // H shifts samples out of bounds.
      for (int y = 4; y < resized_image.Height() - 4; ++y) {
        for (int x = 4; x < resized_image.Width() - 8; ++x) {
          EXPECT_NEAR(direct_image.GetPixel(x, y)->r,
                      resized_image.GetPixel(x, y)->r,
                      2);
        }
      }
    }
  }
}

TEST(Warp, HomographyInterpolation) {
  const Camera source_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 4, 4, 4);
  Camera target_camera = source_camera;
  target_camera.SetPrincipalPointX(1.5);

  Bitmap source_image(4, 4, /*as_rgb=*/false);
  for (int y = 0; y < source_image.Height(); ++y) {
    source_image.SetPixel(0, y, BitmapColor<uint8_t>(0));
    source_image.SetPixel(1, y, BitmapColor<uint8_t>(100));
  }

  WarpImageOptions options;
  Bitmap bilinear_image;
  WarpImageWithHomographyBetweenCameras(options,
                                        Eigen::Matrix3d::Identity(),
                                        source_camera,
                                        target_camera,
                                        source_image,
                                        &bilinear_image);
  ASSERT_TRUE(bilinear_image.GetPixel(0, 0).has_value());
  EXPECT_EQ(bilinear_image.GetPixel(0, 0)->r, 50);

  options.interpolation = WarpImageOptions::Interpolation::kNearestNeighbor;
  Bitmap nearest_image;
  WarpImageWithHomographyBetweenCameras(options,
                                        Eigen::Matrix3d::Identity(),
                                        source_camera,
                                        target_camera,
                                        source_image,
                                        &nearest_image);
  ASSERT_TRUE(nearest_image.GetPixel(0, 0).has_value());
  EXPECT_EQ(nearest_image.GetPixel(0, 0)->r, 100);

  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  options.interpolation = static_cast<WarpImageOptions::Interpolation>(-1);
  EXPECT_ANY_THROW(
      WarpImageWithHomographyBetweenCameras(options,
                                            Eigen::Matrix3d::Identity(),
                                            source_camera,
                                            target_camera,
                                            source_image,
                                            &nearest_image));
}

TEST(Warp, WarpImageWithHomographyBetweenCamerasTransposed) {
  const Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 1, 100, 100);

  Eigen::Matrix3d H;
  H << 0, 1, 0, 1, 0, 0, 0, 0, 1;

  const Bitmap source_image_gray = GenerateRandomBitmap(100, 100, false);
  Bitmap target_image_gray;
  WarpImageWithHomographyBetweenCameras(WarpImageOptions(),
                                        H,
                                        camera,
                                        camera,
                                        source_image_gray,
                                        &target_image_gray);
  CheckBitmapsTransposed(source_image_gray, target_image_gray);

  const Bitmap source_image_rgb = GenerateRandomBitmap(100, 100, true);
  Bitmap target_image_rgb;
  WarpImageWithHomographyBetweenCameras(WarpImageOptions(),
                                        H,
                                        camera,
                                        camera,
                                        source_image_rgb,
                                        &target_image_rgb);
  CheckBitmapsTransposed(source_image_rgb, target_image_rgb);
}

TEST(Warp, ResampleImageBilinear) {
  std::vector<float> image(16);
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = i;
  }

  std::vector<float> resampled(4);
  ResampleImageBilinear(image.data(), 4, 4, 2, 2, resampled.data());

  EXPECT_EQ(resampled[0], 2.5);
  EXPECT_EQ(resampled[1], 4.5);
  EXPECT_EQ(resampled[2], 10.5);
  EXPECT_EQ(resampled[3], 12.5);
}

TEST(Warp, SmoothImage) {
  std::vector<float> image(16);
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = i;
  }

  std::vector<float> smoothed(16);
  SmoothImage(image.data(), 4, 4, 1, 1, smoothed.data());

  EXPECT_NEAR(smoothed[0], 1.81673253, 1e-3);
  EXPECT_NEAR(smoothed[1], 2.51182437, 1e-3);
  EXPECT_NEAR(smoothed[2], 3.39494729, 1e-3);
  EXPECT_NEAR(smoothed[3], 4.09003973, 1e-3);
  EXPECT_NEAR(smoothed[4], 4.59710073, 1e-3);
  EXPECT_NEAR(smoothed[5], 5.29219341, 1e-3);
  EXPECT_NEAR(smoothed[6], 6.17531633, 1e-3);
  EXPECT_NEAR(smoothed[7], 6.87040806, 1e-3);
}

TEST(Warp, DownsampleImage) {
  std::vector<float> image(16);
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = i;
  }

  std::vector<float> downsampled(4);
  DownsampleImage(image.data(), 4, 4, 2, 2, downsampled.data());

  EXPECT_NEAR(downsampled[0], 2.76810598, 1e-3);
  EXPECT_NEAR(downsampled[1], 4.66086388, 1e-3);
  EXPECT_NEAR(downsampled[2], 10.3391361, 1e-3);
  EXPECT_NEAR(downsampled[3], 12.2318935, 1e-3);
}

TEST(Warp, CameraModelsMatchPerPixelProjection) {
  const Bitmap source_image = GenerateRandomBitmap(37, 29, /*as_rgb=*/true);

  Camera opencv_src =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 25.0, 37, 29);
  opencv_src.params.at(opencv_src.ExtraParamsIdxs()[0]) = -0.15;
  Camera pinhole_tgt =
      Camera::CreateFromModelId(2, CameraModelId::kPinhole, 18.3, 31, 23);
  pinhole_tgt.SetFocalLengthY(19.1);
  pinhole_tgt.SetPrincipalPointX(14.0);

  // Outer rays fail SimpleDivision::ImgFromCam (negative discriminant).
  Camera division_src = Camera::CreateFromModelId(
      1, CameraModelId::kSimpleDivision, 20.0, 37, 29);
  division_src.params.at(division_src.ExtraParamsIdxs()[0]) = 0.5;
  Camera fov_tgt =
      Camera::CreateFromModelId(2, CameraModelId::kFOV, 12.0, 31, 23);
  fov_tgt.params.at(fov_tgt.ExtraParamsIdxs()[0]) = 0.8;

  // Back-hemisphere pixels fail Equirectangular::CamFromImg.
  Camera radial_src =
      Camera::CreateFromModelId(1, CameraModelId::kSimpleRadial, 22.0, 37, 29);
  radial_src.params.at(radial_src.ExtraParamsIdxs()[0]) = 0.1;
  const Camera equirect_tgt = Camera::CreateFromModelId(
      2, CameraModelId::kEquirectangular, 1.0, 31, 23);

  Eigen::Matrix3d H;
  // Column x = 10 hits warped.z() == 0 under direct warping.
  H << 1.0, 0.05, 1.5, -0.05, 1.0, -0.75, -1.0 / 10.5, 0.0, 1.0;

  const auto warp_reference = [&](const WarpImageOptions::Interpolation interp,
                                  const Eigen::Matrix3d* homography,
                                  const Camera& src_cam,
                                  const Camera& tgt_cam) {
    Bitmap expected(tgt_cam.width, tgt_cam.height, source_image.IsRGB());
    for (int y = 0; y < expected.Height(); ++y) {
      for (int x = 0; x < expected.Width(); ++x) {
        Eigen::Vector2d img_pt(x + 0.5, y + 0.5);
        if (homography != nullptr) {
          const Eigen::Vector3d warped = *homography * img_pt.homogeneous();
          if (warped.z() == 0) {
            continue;
          }
          img_pt = warped.hnormalized();
        }
        const auto cam_pt = tgt_cam.CamFromImg(img_pt);
        const auto src_pt =
            cam_pt ? src_cam.ImgFromCam(cam_pt->homogeneous()) : std::nullopt;
        if (!src_pt) {
          continue;
        }
        if (interp == WarpImageOptions::Interpolation::kNearestNeighbor) {
          if (const auto c = source_image.InterpolateNearestNeighbor(
                  src_pt->x() - 0.5, src_pt->y() - 0.5)) {
            expected.SetPixel(x, y, *c);
          }
        } else if (const auto c = source_image.InterpolateBilinear(
                       src_pt->x() - 0.5, src_pt->y() - 0.5)) {
          expected.SetPixel(x, y, c->Cast<uint8_t>());
        }
      }
    }
    return expected;
  };

  for (const auto& [src_cam, tgt_cam] :
       std::vector<std::pair<Camera, Camera>>{{opencv_src, pinhole_tgt},
                                              {division_src, fov_tgt},
                                              {radial_src, equirect_tgt}}) {
    bool saw_nonzero = false;
    bool saw_zero = false;
    for (const auto interpolation :
         {WarpImageOptions::Interpolation::kNearestNeighbor,
          WarpImageOptions::Interpolation::kBilinear}) {
      WarpImageOptions options;
      options.interpolation = interpolation;

      Bitmap actual;
      WarpImageBetweenCameras(options, src_cam, tgt_cam, source_image, &actual);
      CheckBitmapsExactlyEqual(
          actual, warp_reference(interpolation, nullptr, src_cam, tgt_cam));

      Bitmap actual_H;
      WarpImageWithHomographyBetweenCameras(
          options, H, src_cam, tgt_cam, source_image, &actual_H);
      CheckBitmapsExactlyEqual(
          actual_H, warp_reference(interpolation, &H, src_cam, tgt_cam));
      for (const uint8_t v : actual_H.RowMajorData()) {
        saw_nonzero |= (v != 0);
        saw_zero |= (v == 0);
      }
    }
    EXPECT_TRUE(saw_nonzero);
    EXPECT_TRUE(saw_zero);
  }
}

}  // namespace
}  // namespace colmap
