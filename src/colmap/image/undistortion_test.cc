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

#include "colmap/image/undistortion.h"

#include "colmap/geometry/pose.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/eigen_matchers.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

Reconstruction CreateSyntheticReconstructionWithBitmaps(
    const std::string& image_path,
    int num_images = 2,
    int image_width = 100,
    int image_height = 100) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_images;
  synthetic_dataset_options.camera_width = image_width;
  synthetic_dataset_options.camera_height = image_height;

  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Create dummy images.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap(image_width, image_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));
    bitmap.Write(JoinPaths(image_path, image.Name()));
  }

  return reconstruction;
}

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

  Bitmap distorted_image(100, 100, false);
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

  Bitmap distorted_image(100, 100, false);
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
      ASSERT_NE(color.r, 0);
      ASSERT_NE(color.g, 0);
      ASSERT_NE(color.b, 0);
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
  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, 1));
  reconstruction.AddRig(rig);

  for (image_t image_id = 1; image_id <= kNumImages; ++image_id) {
    Frame frame;
    frame.SetRigId(1);
    frame.SetFrameId(image_id);
    frame.SetRigFromWorld(Rigid3d());
    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetFrameId(frame.FrameId());
    image.SetName("image" + std::to_string(image_id));
    image.SetPoints2D(
        std::vector<Eigen::Vector2d>(kNumPoints2D, Eigen::Vector2d::Ones()));
    frame.AddDataId(image.DataId());
    reconstruction.AddFrame(frame);
    reconstruction.AddImage(image);
    reconstruction.RegisterFrame(frame.FrameId());
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
  EXPECT_THAT(H1, EigenMatrixNear<Eigen::Matrix3d>(H1_ref.transpose(), 1e-5));

  Eigen::Matrix3d H2_ref;
  H2_ref << -0.082173, -1.01288, -0.698868, 0.301854, 0.472844, -0.465336,
      0.963533, 0.292411, 1.12528;
  EXPECT_THAT(H2, EigenMatrixNear<Eigen::Matrix3d>(H2_ref.transpose(), 1e-5));

  Eigen::Matrix4d Q_ref;
  Q_ref << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2.67261, -0.5, -0.5, 1, 0;
  EXPECT_THAT(Q, EigenMatrixNear(Q_ref, 1e-5));
}

TEST(RectifyAndUndistortStereoImages, Nominal) {
  UndistortCameraOptions options;

  // Create two distorted cameras with radial distortion.
  Camera distorted_camera1 =
      Camera::CreateFromModelName(1, "SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera1.params[3] = 0.1;  // Add some radial distortion

  Camera distorted_camera2 =
      Camera::CreateFromModelName(2, "SIMPLE_RADIAL", 100, 100, 100);
  distorted_camera2.params[3] = 0.1;  // Add some radial distortion

  // Create dummy distorted images.
  Bitmap distorted_image1(100, 100, true);
  distorted_image1.Fill(BitmapColor<uint8_t>(255, 0, 0));  // Red image

  Bitmap distorted_image2(100, 100, true);
  distorted_image2.Fill(BitmapColor<uint8_t>(0, 255, 0));  // Green image

  // Create relative pose between cameras (typical stereo baseline).
  const Rigid3d cam2_from_cam1(
      Eigen::Quaterniond(EulerAnglesToRotationMatrix(0.0, 0.05, 0.0)),
      Eigen::Vector3d(0.1, 0.0, 0.0));  // 0.1m baseline

  Bitmap undistorted_image1;
  Bitmap undistorted_image2;
  Camera undistorted_camera;
  Eigen::Matrix4d Q;

  // Rectify and undistort stereo images.
  RectifyAndUndistortStereoImages(options,
                                  distorted_image1,
                                  distorted_image2,
                                  distorted_camera1,
                                  distorted_camera2,
                                  cam2_from_cam1,
                                  &undistorted_image1,
                                  &undistorted_image2,
                                  &undistorted_camera,
                                  &Q);
  EXPECT_EQ(undistorted_camera.ModelName(), "PINHOLE");
  EXPECT_EQ(undistorted_camera.width, undistorted_image1.Width());
  EXPECT_EQ(undistorted_camera.height, undistorted_image1.Height());
  EXPECT_EQ(undistorted_image1.Width(), undistorted_image2.Width());
  EXPECT_EQ(undistorted_image1.Height(), undistorted_image2.Height());
}

TEST(COLMAPUndistorter, Integration) {
  std::string temp_dir = CreateTestDir();
  std::string image_path = JoinPaths(temp_dir, "input_images");
  std::string output_path = JoinPaths(temp_dir, "output");
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run COLMAP undistorter.
  UndistortCameraOptions options;
  COLMAPUndistorter undistorter(
      options, reconstruction, image_path, output_path);
  undistorter.Run();

  // Verify output directories were created.
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, "images")));
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, "sparse")));
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, "stereo")));

  // Verify undistorted images were written.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(ExistsFile(JoinPaths(output_path, "images", image.Name())));
  }
}

TEST(PMVSUndistorter, Integration) {
  std::string temp_dir = CreateTestDir();
  std::string image_path = JoinPaths(temp_dir, "input_images");
  std::string output_path = JoinPaths(temp_dir, "pmvs_output");
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run PMVS undistorter.
  UndistortCameraOptions options;
  PMVSUndistorter undistorter(options, reconstruction, image_path, output_path);
  undistorter.Run();

  // Verify PMVS output structure was created (under pmvs/ subdirectory).
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, "pmvs")));
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, "pmvs", "models")));
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, "pmvs", "txt")));
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, "pmvs", "visualize")));

  // Verify undistorted images were written with numbered names.
  // PMVS writes images as 00000000.jpg, 00000001.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 0; i < num_images; ++i) {
    const std::string image_name = StringPrintf("%08zu.jpg", i);
    EXPECT_TRUE(
        ExistsFile(JoinPaths(output_path, "pmvs", "visualize", image_name)));
  }
}

TEST(CMPMVSUndistorter, Integration) {
  std::string temp_dir = CreateTestDir();
  std::string image_path = JoinPaths(temp_dir, "input_images");
  std::string output_path = JoinPaths(temp_dir, "cmpmvs_output");
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run CMP-MVS undistorter.
  UndistortCameraOptions options;
  CMPMVSUndistorter undistorter(
      options, reconstruction, image_path, output_path);
  undistorter.Run();

  // Verify CMP-MVS output structure was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written with sequential numbering.
  // CMP-MVS writes images as 00001.jpg, 00002.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 1; i <= num_images; ++i) {
    const std::string image_name = StringPrintf("%05zu.jpg", i);
    EXPECT_TRUE(ExistsFile(JoinPaths(output_path, image_name)));
  }
}

TEST(PureImageUndistorter, Integration) {
  std::string temp_dir = CreateTestDir();
  std::string image_path = JoinPaths(temp_dir, "input_images");
  std::string output_path = JoinPaths(temp_dir, "pure_output");
  CreateDirIfNotExists(image_path);

  // Create test images and cameras.
  std::vector<std::pair<std::string, Camera>> image_names_and_cameras;
  for (int i = 1; i <= 2; ++i) {
    std::string image_name = "image" + std::to_string(i) + ".png";
    Camera camera =
        Camera::CreateFromModelName(i, "SIMPLE_RADIAL", 1.0, 100, 100);
    image_names_and_cameras.emplace_back(image_name, camera);

    // Create dummy image.
    Bitmap bitmap(100, 100, true);
    bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));
    bitmap.Write(JoinPaths(image_path, image_name));
  }

  // Run pure image undistorter.
  UndistortCameraOptions options;
  PureImageUndistorter undistorter(
      options, image_path, output_path, image_names_and_cameras);
  undistorter.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written.
  for (const auto& [image_name, camera] : image_names_and_cameras) {
    EXPECT_TRUE(ExistsFile(JoinPaths(output_path, image_name)));
  }
}

TEST(StereoImageRectifier, Integration) {
  std::string temp_dir = CreateTestDir();
  std::string image_path = JoinPaths(temp_dir, "input_images");
  std::string output_path = JoinPaths(temp_dir, "stereo_output");
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Create stereo pair from first two images.
  std::vector<std::pair<image_t, image_t>> stereo_pairs;
  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  ASSERT_GE(image_ids.size(), 2);
  stereo_pairs.emplace_back(image_ids[0], image_ids[1]);

  // Run stereo image rectifier.
  UndistortCameraOptions options;
  StereoImageRectifier rectifier(
      options, reconstruction, image_path, output_path, stereo_pairs);
  rectifier.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify rectified images were written.
  // StereoImageRectifier creates a subdirectory for each stereo pair.
  const auto& image1 = reconstruction.Image(image_ids[0]);
  const auto& image2 = reconstruction.Image(image_ids[1]);
  const std::string stereo_pair_name =
      StringPrintf("%s-%s", image1.Name().c_str(), image2.Name().c_str());
  EXPECT_TRUE(ExistsDir(JoinPaths(output_path, stereo_pair_name)));
  EXPECT_TRUE(
      ExistsFile(JoinPaths(output_path, stereo_pair_name, image1.Name())));
  EXPECT_TRUE(
      ExistsFile(JoinPaths(output_path, stereo_pair_name, image2.Name())));
}

}  // namespace
}  // namespace colmap
