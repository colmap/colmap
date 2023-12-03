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

#include "colmap/image/undistortion.h"

#include "colmap/geometry/pose.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(UndistortCamera, Nominal) {
  UndistortCameraOptions options;
  Camera distorted_camera;
  Camera undistorted_camera;

  distorted_camera = Camera::CreateFromModelName(1, "SIMPLE_PINHOLE", 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 1);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 1);
  EXPECT_EQ(undistorted_camera.width, 1);
  EXPECT_EQ(undistorted_camera.height, 1);

  distorted_camera = Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 1);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 1);
  EXPECT_EQ(undistorted_camera.width, 1);
  EXPECT_EQ(undistorted_camera.height, 1);

  distorted_camera =
      Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.params[3] = 0.5;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.width, 84);
  EXPECT_EQ(undistorted_camera.height, 84);

  options.blank_pixels = 1;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.width, 90);
  EXPECT_EQ(undistorted_camera.height, 90);

  options.max_scale = 0.75;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.width, 75);
  EXPECT_EQ(undistorted_camera.height, 75);

  options.max_scale = 1.0;
  options.roi_min_x = 0.1;
  options.roi_min_y = 0.2;
  options.roi_max_x = 0.9;
  options.roi_max_y = 0.8;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.width, 80);
  EXPECT_EQ(undistorted_camera.height, 60);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 40);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 30);
}

TEST(UndistortCamera, BlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 1;

  Camera distorted_camera;
  distorted_camera =
      Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.params[3] = 0.5;

  Bitmap distorted_image;
  distorted_image.Allocate(100, 100, false);
  distorted_image.Fill(BitmapColor<uint8_t>(255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options,
                 distorted_image,
                 distorted_camera,
                 &undistorted_image,
                 &undistorted_camera);

  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 90.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 90.0 / 2.0);
  EXPECT_EQ(undistorted_camera.width, 90);
  EXPECT_EQ(undistorted_camera.height, 90);

  // Make sure that there is no blank pixel.
  size_t num_blank_pixels = 0;
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(undistorted_image.GetPixel(x, y, &color));
      if (color == BitmapColor<uint8_t>(0)) {
        num_blank_pixels += 1;
      }
    }
  }

  EXPECT_GT(num_blank_pixels, 0);
}

TEST(UndistortCamera, NoBlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 0;

  Camera distorted_camera;
  distorted_camera =
      Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.params[3] = 0.5;

  Bitmap distorted_image;
  distorted_image.Allocate(100, 100, false);
  distorted_image.Fill(BitmapColor<uint8_t>(255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options,
                 distorted_image,
                 distorted_camera,
                 &undistorted_image,
                 &undistorted_camera);

  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.FocalLengthX(), 100);
  EXPECT_EQ(undistorted_camera.FocalLengthY(), 100);
  EXPECT_EQ(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  EXPECT_EQ(undistorted_camera.width, 84);
  EXPECT_EQ(undistorted_camera.height, 84);

  // Make sure that there is no blank pixel.
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      BitmapColor<uint8_t> color;
      EXPECT_TRUE(undistorted_image.GetPixel(x, y, &color));
      EXPECT_NE(color.r, 0);
      EXPECT_EQ(color.g, 0);
      EXPECT_EQ(color.b, 0);
    }
  }
}

TEST(UndistortReconstruction, Nominal) {
  const size_t kNumImages = 10;
  const size_t kNumPoints2D = 10;

  Reconstruction reconstruction;

  Camera camera = Camera::CreateFromModelName(1, "OPENCV", 1, 1, 1);
  camera.params[4] = 1.0;
  reconstruction.AddCamera(camera);

  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Ones()));
    reconstruction.AddImage(image);
    reconstruction.RegisterImage(image_id);
  }

  UndistortCameraOptions options;
  UndistortReconstruction(options, &reconstruction);
  for (const auto& camera : reconstruction.Cameras()) {
    EXPECT_EQ(camera.second.ModelName(), "PINHOLE");
  }

  for (const auto& image : reconstruction.Images()) {
    for (const auto& point2D : image.second.Points2D()) {
      EXPECT_NE(point2D.xy, Eigen::Vector2d::Ones());
    }
  }
}

TEST(RectifyStereoCameras, Nominal) {
  Camera camera1;
  camera1 = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);

  Camera camera2;
  camera2 = Camera::CreateFromModelName(1, "PINHOLE", 1, 1, 1);

  const Rigid3d cam2_from_cam1(
      Eigen::Quaterniond(EulerAnglesToRotationMatrix(0.1, 0.2, 0.3)),
      Eigen::Vector3d(0.1, 0.2, 0.3));

  Camera rectified_camera1;
  Camera rectified_camera2;
  Eigen::Matrix3d H1;
  Eigen::Matrix3d H2;
  Eigen::Matrix4d Q;
  RectifyStereoCameras(camera1, camera2, cam2_from_cam1, &H1, &H2, &Q);

  Eigen::Matrix3d H1_ref;
  H1_ref << -0.202759, -0.815848, -0.897034, 0.416329, 0.733069, -0.199657,
      0.910839, -0.175408, 0.942638;
  EXPECT_TRUE(H1.isApprox(H1_ref.transpose(), 1e-5));

  Eigen::Matrix3d H2_ref;
  H2_ref << -0.082173, -1.01288, -0.698868, 0.301854, 0.472844, -0.465336,
      0.963533, 0.292411, 1.12528;
  EXPECT_TRUE(H2.isApprox(H2_ref.transpose(), 1e-5));

  Eigen::Matrix4d Q_ref;
  Q_ref << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2.67261, -0.5, -0.5, 1, 0;
  EXPECT_TRUE(Q.isApprox(Q_ref, 1e-5));
}

}  // namespace
}  // namespace colmap
