// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#define TEST_NAME "base/undistortion"
#include "util/testing.h"

#include "base/pose.h"
#include "base/undistortion.h"
#include "util/misc.h"

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
  distorted_image.Fill(BitmapColor<uint8_t>(255));

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

BOOST_AUTO_TEST_CASE(TestUndistortFOVEstimation) {
  // Due to generic nature of FOV estimation estimates will be slightly smaller,
  // than calculated values; using 2% tolerance
  const auto tolerance = boost::test_tools::tolerance(2.0);
  const double f = 100.0, w = 100.0, h = 200.0;
  const double w2 = w / 2.0, h2 = h / 2.0;
  const double diag2 = sqrt(w2 * w2 + h2 * h2);

  UndistortCameraOptions options;
  options.estimate_focal_length_from_fov = true;
  options.max_fov = 180.0 - 1e-5;

  Camera distorted_camera;
  distorted_camera.InitializeWithName("OPENCV_FISHEYE", f, w, h);

  Camera undistorted_camera_default =
      UndistortCamera(options, distorted_camera);
  const double expected_default = diag2 / tan(diag2 / f);
  BOOST_CHECK_EQUAL(undistorted_camera_default.ModelName(), "PINHOLE");
  BOOST_CHECK_EQUAL(undistorted_camera_default.FocalLengthX(),
                    undistorted_camera_default.FocalLengthY());
  BOOST_TEST(expected_default == undistorted_camera_default.FocalLengthX(),
             tolerance);

  options.max_horizontal_fov = 2.0 * RadToDeg(atan(h2 / f));
  Camera undistorted_camera_hfov = UndistortCamera(options, distorted_camera);
  const double expected_hfov = h2 / tan(h2 / f);
  BOOST_TEST(expected_hfov == undistorted_camera_hfov.FocalLengthX(),
             tolerance);

  options.max_horizontal_fov = 180.0;
  options.max_vertical_fov = 2.0 * RadToDeg(atan(w2 / f));
  Camera undistorted_camera_vfov = UndistortCamera(options, distorted_camera);
  const double expected_vfov = w2 / tan(w2 / f);
  BOOST_TEST(expected_vfov == undistorted_camera_vfov.FocalLengthX(),
             tolerance);

  options.max_fov = 90.0;
  options.max_vertical_fov = 180.0;
  Camera undistorted_camera_90fov = UndistortCamera(options, distorted_camera);
  const double expected_90fov = diag2 / tan(DegToRad(options.max_fov) / 2.0);
  BOOST_TEST(expected_90fov == undistorted_camera_90fov.FocalLengthX(),
             tolerance);
}

BOOST_AUTO_TEST_CASE(TestUndistortionForced) {
  UndistortCameraOptions options;
  options.camera_model_override = "RADIAL";
  options.camera_model_override_params = "1.0,2.0,3.0,4.0,5.0";

  Camera distorted_camera;
  distorted_camera.InitializeWithName("FOV", 1, 10, 20);

  Camera undistorted_camera = UndistortCamera(options, distorted_camera);

  BOOST_CHECK_EQUAL(undistorted_camera.ModelName(),
                    options.camera_model_override);
  BOOST_CHECK_EQUAL(undistorted_camera.Height(), distorted_camera.Height());
  BOOST_CHECK_EQUAL(undistorted_camera.Width(), distorted_camera.Width());

  const std::vector<double> params_expected =
      CSVToVector<double>(options.camera_model_override_params);
  const std::vector<double> params = undistorted_camera.Params();

  BOOST_TEST(params == params_expected);
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
