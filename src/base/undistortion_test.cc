// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "base/undistortion"
#include <boost/test/unit_test.hpp>

#include "base/undistortion.h"

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
}

BOOST_AUTO_TEST_CASE(TestUndistortCameraBlankPixels) {
  UndistortCameraOptions options;
  options.blank_pixels = 1;

  Camera distorted_camera;
  distorted_camera.InitializeWithName("SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera.Params(3) = 0.5;

  Bitmap distorted_image;
  distorted_image.Allocate(100, 100, false);
  distorted_image.Fill(BitmapColor<uint8_t>(255, 255, 255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options, distorted_image, distorted_camera, &undistorted_image,
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
      if (color == BitmapColor<uint8_t>(0, 0, 0)) {
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
  distorted_image.Fill(BitmapColor<uint8_t>(255, 255, 255));

  Bitmap undistorted_image;
  Camera undistorted_camera;
  UndistortImage(options, distorted_image, distorted_camera, &undistorted_image,
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
