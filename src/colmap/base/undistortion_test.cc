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

#define TEST_NAME "base/undistortion"
#include "colmap/base/undistortion.h"

#include "colmap/base/pose.h"
#include "colmap/util/testing.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(TestUndistortCamera) {
  UndistortCameraOptions options;
  Camera distorted_camera;
  Camera undistorted_camera;

  distorted_camera.InitializeWithName("SIMPLE_PINHOLE", 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 1);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 1);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 1);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 1);

  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 1, 1, 1);
  undistorted_camera = UndistortCamera(options, distorted_camera);
  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 1);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 1);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 1);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 1);

  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.Params(3) = 0.5;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 84);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 84);

  options.blank_pixels = 1;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 90);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 90);

  options.max_scale = 0.75;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 75);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 75);

  options.max_scale = 1.0;
  options.roi_min_x = 0.1;
  options.roi_min_y = 0.2;
  options.roi_max_x = 0.9;
  options.roi_max_y = 0.8;
  undistorted_camera = UndistortCamera(options, distorted_camera);
  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 80);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 60);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointX(), 40);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointY(), 30);
}

BOOST_AUTO_TEST_CASE(TestUndistortCameraBlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 1;

  Camera distorted_camera;
  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.Params(3) = 0.5;

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

  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointX(), 90.0 / 2.0);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointY(), 90.0 / 2.0);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 90);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 90);

  // Make sure that there is no blank pixel.
  size_t num_blank_pixels = 0;
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      BitmapColor<uint8_t> color;
      BOOST_CHECK(undistorted_image.GetPixel(x, y, &color));
      if (color == BitmapColor<uint8_t>(0)) {
        num_blank_pixels += 1;
      }
    }
  }

  BOOST_CHECK_GT(num_blank_pixels, 0);
}

BOOST_AUTO_TEST_CASE(TestUndistortCameraNoBlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 0;

  Camera distorted_camera;
  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.Params(3) = 0.5;

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

  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthX(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.FocalLengthY(), 100);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointX(), 84.0 / 2.0);
  BOOST_CHECK_EQUAL(undistorted_camera.PrincipalPointY(), 84.0 / 2.0);
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), 84);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), 84);

  // Make sure that there is no blank pixel.
  for (int y = 0; y < undistorted_image.Height(); ++y) {
    for (int x = 0; x < undistorted_image.Width(); ++x) {
      BitmapColor<uint8_t> color;
      BOOST_CHECK(undistorted_image.GetPixel(x, y, &color));
      BOOST_CHECK_NE(color.r, 0);
      BOOST_CHECK_EQUAL(color.g, 0);
      BOOST_CHECK_EQUAL(color.b, 0);
    }
  }
}

BOOST_AUTO_TEST_CASE(TestUndistortReconstruction) {
  const size_t kNumImages = 10;
  const size_t kNumPoints2D = 10;

  Reconstruction reconstruction;

  Camera camera;
  camera.SetCameraId(1);
  camera.InitializeWithName("OPENCV", 1, 1, 1);
  camera.Params(4) = 1.0;
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
    BOOST_CHECK_EQUAL(camera.second.ModelName(), "PINHOLE");
  }

  for (const auto& image : reconstruction.Images()) {
    for (const auto& point2D : image.second.Points2D()) {
      BOOST_CHECK_NE(point2D.XY(), Eigen::Vector2d::Ones());
    }
  }
}

BOOST_AUTO_TEST_CASE(TestRectifyStereoCameras) {
  Camera camera1;
  camera1.SetCameraId(1);
  camera1.InitializeWithName("PINHOLE", 1, 1, 1);

  Camera camera2;
  camera2.SetCameraId(1);
  camera2.InitializeWithName("PINHOLE", 1, 1, 1);

  const Eigen::Vector4d qvec =
      RotationMatrixToQuaternion(EulerAnglesToRotationMatrix(0.1, 0.2, 0.3));
  const Eigen::Vector3d tvec(0.1, 0.2, 0.3);

  Camera rectified_camera1;
  Camera rectified_camera2;
  Eigen::Matrix3d H1;
  Eigen::Matrix3d H2;
  Eigen::Matrix4d Q;
  RectifyStereoCameras(camera1, camera2, qvec, tvec, &H1, &H2, &Q);

  Eigen::Matrix3d H1_ref;
  H1_ref << -0.202759, -0.815848, -0.897034, 0.416329, 0.733069, -0.199657,
      0.910839, -0.175408, 0.942638;
  BOOST_CHECK(H1.isApprox(H1_ref.transpose(), 1e-5));

  Eigen::Matrix3d H2_ref;
  H2_ref << -0.082173, -1.01288, -0.698868, 0.301854, 0.472844, -0.465336,
      0.963533, 0.292411, 1.12528;
  BOOST_CHECK(H2.isApprox(H2_ref.transpose(), 1e-5));

  Eigen::Matrix4d Q_ref;
  Q_ref << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2.67261, -0.5, -0.5, 1, 0;
  BOOST_CHECK(Q.isApprox(Q_ref, 1e-5));
}
