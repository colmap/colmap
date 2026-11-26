// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/undistorters.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Reconstruction CreateSyntheticReconstructionWithBitmaps(
    const std::filesystem::path& image_path,
    int num_images = 2,
    int image_width = 100,
    int image_height = 100,
    const std::string& image_extension = ".png",
    CameraModelId camera_model_id = SimpleRadialCameraModel::model_id,
    const std::vector<double>& camera_params = {1280, 512, 384, 0.05}) {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_images;
  synthetic_dataset_options.camera_width = image_width;
  synthetic_dataset_options.camera_height = image_height;
  synthetic_dataset_options.camera_model_id = camera_model_id;
  synthetic_dataset_options.camera_params = camera_params;
  synthetic_dataset_options.image_extension = image_extension;

  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Create dummy images.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap(image_width, image_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));
    bitmap.Write(image_path / image.Name());
  }

  return reconstruction;
}

TEST(COLMAPUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run COLMAP undistorter.
  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify output directories were created.
  EXPECT_TRUE(ExistsDir(output_path / "images"));
  EXPECT_TRUE(ExistsDir(output_path / "sparse"));
  EXPECT_TRUE(ExistsDir(output_path / "stereo"));
  EXPECT_FALSE(ExistsDir(output_path / "masks"));

  // Verify undistorted images were written.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(ExistsFile(output_path / "images" / image.Name()));
  }

  // Expect dense reconstruction files to be written.
  EXPECT_TRUE(ExistsFile(output_path / "stereo/patch-match.cfg"));
  EXPECT_TRUE(ExistsFile(output_path / "stereo/fusion.cfg"));
}

TEST(COLMAPUndistorter, UndistortsZeroDistortionFisheyeCameras) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // SIMPLE_FISHEYE has no distortion parameters, but MVS needs pinhole cameras.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(
          image_path,
          /*num_images=*/1,
          /*image_width=*/100,
          /*image_height=*/100,
          /*image_extension=*/".png",
          SimpleFisheyeCameraModel::model_id,
          /*camera_params=*/{50, 50, 50});

  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  Reconstruction undistorted_reconstruction;
  undistorted_reconstruction.Read(output_path / "sparse");
  EXPECT_EQ(undistorted_reconstruction.Camera(1).model_id,
            PinholeCameraModel::model_id);
}

TEST(COLMAPUndistorter, StopsPendingWork) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path, /*num_images=*/10);
  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  bool stop_checked = false;
  undistorter.SetCheckIfStoppedFunc([&stop_checked]() {
    stop_checked = true;
    return true;
  });
  undistorter.Run();

  EXPECT_TRUE(stop_checked);
  EXPECT_FALSE(ExistsFile(output_path / "stereo/patch-match.cfg"));
  EXPECT_FALSE(ExistsFile(output_path / "stereo/fusion.cfg"));
}

TEST(COLMAPUndistorter, SpecificImages) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path,
                                               /*num_images=*/2,
                                               /*image_width=*/100,
                                               /*image_height=*/100,
                                               /*image_extension=*/".jpg");

  const Image& image = reconstruction.Image(reconstruction.RegImageIds()[0]);

  // Run COLMAP undistorter.
  COLMAPUndistorter::Options options;
  options.image_ids = {image.ImageId()};
  COLMAPUndistorter undistorter(options,
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify that only the specified image was written.
  EXPECT_THAT(
      GetRecursiveFileList(output_path / "images"),
      testing::UnorderedElementsAre(output_path / "images" / image.Name()));
}

TEST(COLMAPUndistorter, JpegQuality) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path,
                                               /*num_images=*/1,
                                               /*image_width=*/100,
                                               /*image_height=*/100,
                                               /*image_extension=*/".jpg");

  // Run COLMAP undistorter.
  COLMAPUndistorter::Options options;
  options.jpeg_quality = 50;
  COLMAPUndistorter undistorter(options,
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify undistorted images were written.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(ExistsFile(output_path / "images" / image.Name()));
  }
}

TEST(COLMAPUndistorter, MasksFollowOutputCamera) {
  for (const auto model_id : {CameraModelId::kSimpleRadial,
                              CameraModelId::kPinhole,
                              CameraModelId::kEquirectangular}) {
    for (const int max_image_size : {-1, 24}) {
      SCOPED_TRACE(static_cast<int>(model_id));
      SCOPED_TRACE(max_image_size);
      const auto temp_dir =
          CreateTestDir() / (std::to_string(static_cast<int>(model_id)) + "_" +
                             std::to_string(max_image_size));
      CreateDirIfNotExists(temp_dir);
      const auto image_path = temp_dir / "images";
      const auto mask_path = temp_dir / "masks";
      const auto output_path = temp_dir / "output";
      CreateDirIfNotExists(image_path);
      CreateDirIfNotExists(mask_path);
      CreateDirIfNotExists(output_path);
      Reconstruction reconstruction = CreateSyntheticReconstructionWithBitmaps(
          image_path, 2, 100, 80, ".jpg");
      Image& image = reconstruction.Image(reconstruction.RegImageIds()[0]);
      Camera& camera = reconstruction.Camera(image.CameraId());
      camera =
          Camera::CreateFromModelId(camera.camera_id, model_id, 70, 100, 80);
      if (model_id == CameraModelId::kSimpleRadial) {
        camera.params.back() = 0.3;
      }
      CreateDirIfNotExists(image_path / "nested");
      CreateDirIfNotExists(mask_path / "nested");
      const std::string image_name = "nested/" + image.Name();
      std::filesystem::rename(image_path / image.Name(),
                              image_path / image_name);
      image.SetName(image_name);

      Bitmap mask(100, 80, false);
      for (int y = 0; y < mask.Height(); ++y) {
        for (int x = 0; x < mask.Width(); ++x) {
          mask.SetPixel(x, y, BitmapColor<uint8_t>((x + y) % 3 * 127));
        }
      }
      // Exercise both accepted input naming conventions.
      auto input_mask_path = mask_path / (image.Name() + ".png");
      if (max_image_size > 0) {
        input_mask_path = mask_path / image.Name();
        input_mask_path.replace_extension(".png");
      }
      ASSERT_TRUE(mask.Write(input_mask_path));
      COLMAPUndistorter::Options options;
      options.mask_path = mask_path;
      options.image_ids = {image.ImageId()};
      options.num_threads = 1;
      UndistortCameraOptions camera_options;
      camera_options.max_image_size = max_image_size;
      camera_options.roi_min_x = 0.1;
      camera_options.roi_max_x = 0.9;
      camera_options.roi_min_y = 0.2;
      camera_options.roi_max_y = 0.8;
      COLMAPUndistorter(
          options, camera_options, reconstruction, image_path, output_path)
          .Run();

      Bitmap output_mask;
      ASSERT_TRUE(output_mask.Read(
          output_path / "masks" / (image.Name() + ".png"), false));
      Bitmap output_image;
      ASSERT_TRUE(output_image.Read(output_path / "images" / image.Name()));
      EXPECT_EQ(output_mask.Width(), output_image.Width());
      EXPECT_EQ(output_mask.Height(), output_image.Height());
      Reconstruction output_reconstruction;
      output_reconstruction.Read(output_path / "sparse");
      const Camera& output_camera =
          output_reconstruction.Camera(image.CameraId());
      EXPECT_EQ(output_mask.Width(), output_camera.width);
      EXPECT_EQ(output_mask.Height(), output_camera.height);
      EXPECT_EQ(GetRecursiveFileList(output_path / "masks").size(), 1);
      int num_nonzero = 0;
      for (int y = 0; y < output_mask.Height(); ++y) {
        for (int x = 0; x < output_mask.Width(); ++x) {
          Eigen::Vector2d source_point;
          if (camera.IsSpherical()) {
            source_point = Eigen::Vector2d(
                (x + 0.5) * camera.width / output_camera.width,
                (y + 0.5) * camera.height / output_camera.height);
          } else {
            const auto ray = output_camera.CamFromImg({x + 0.5, y + 0.5});
            ASSERT_TRUE(ray);
            const auto point = camera.ImgFromCam(ray->homogeneous());
            ASSERT_TRUE(point);
            source_point = *point;
          }
          const auto source_color =
              mask.GetPixel(static_cast<int>(std::floor(source_point.x())),
                            static_cast<int>(std::floor(source_point.y())));
          const auto color = output_mask.GetPixel(x, y);
          ASSERT_TRUE(color);
          EXPECT_EQ(color->r, source_color ? source_color->r : 0);
          EXPECT_TRUE(color->r == 0 || color->r == 127 || color->r == 254);
          num_nonzero += color->r != 0;
        }
      }
      EXPECT_GT(num_nonzero, 0);
    }
  }
}

TEST(COLMAPUndistorter, InvalidMasks) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "images";
  const auto mask_path = temp_dir / "masks";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path, 1);
  COLMAPUndistorter::Options options;
  options.mask_path = mask_path;
  options.num_threads = 1;
  EXPECT_ANY_THROW(COLMAPUndistorter(options,
                                     UndistortCameraOptions(),
                                     reconstruction,
                                     image_path,
                                     output_path));
  CreateDirIfNotExists(mask_path);
  const auto run = [&]() {
    COLMAPUndistorter(options,
                      UndistortCameraOptions(),
                      reconstruction,
                      image_path,
                      output_path)
        .Run();
  };
  // Missing masks must not silently produce an unmasked workspace.
  EXPECT_ANY_THROW(run());
  const Image& image = reconstruction.Image(reconstruction.RegImageIds()[0]);
  const auto input_mask_path = mask_path / (image.Name() + ".png");
  Bitmap wrong_size(10, 10, false);
  ASSERT_TRUE(wrong_size.Write(input_mask_path));
  EXPECT_ANY_THROW(run());
  {
    std::ofstream file(input_mask_path);
    file << "not an image";
  }
  EXPECT_ANY_THROW(run());
}

TEST(COLMAPUndistorter, TargetCamera) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Already-undistorted images are also undistorted to the target camera
  // instead of being copied.
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);
  for (const auto& [camera_id, _] : reconstruction.Cameras()) {
    reconstruction.Camera(camera_id) = Camera::CreateFromModelId(
        camera_id, CameraModelId::kPinhole, 100, 100, 100);
  }

  UndistortCameraOptions camera_options;
  camera_options.target_camera =
      Camera::CreateFromModelId(2, CameraModelId::kPinhole, 50, 60, 40);
  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                camera_options,
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(output_path / "images" / image.Name()));
    EXPECT_EQ(bitmap.Width(), 60);
    EXPECT_EQ(bitmap.Height(), 40);
  }
  Reconstruction undistorted_reconstruction;
  undistorted_reconstruction.Read(output_path / "sparse");
  EXPECT_EQ(undistorted_reconstruction.NumCameras(),
            reconstruction.NumCameras());
  for (const auto& [camera_id, camera] : undistorted_reconstruction.Cameras()) {
    EXPECT_TRUE(reconstruction.ExistsCamera(camera_id));
    EXPECT_EQ(camera.width, 60);
    EXPECT_EQ(camera.height, 40);
    EXPECT_EQ(camera.params, camera_options.target_camera->params);
  }
}

TEST(PMVSUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "pmvs_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run PMVS undistorter.
  PMVSUndistorter undistorter(PMVSUndistorter::Options(),
                              UndistortCameraOptions(),
                              reconstruction,
                              image_path,
                              output_path);
  undistorter.Run();

  // Verify PMVS output structure was created (under pmvs/ subdirectory).
  EXPECT_TRUE(ExistsDir(output_path / "pmvs"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "models"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "txt"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "visualize"));

  // Verify undistorted images were written with numbered names.
  // PMVS writes images as 00000000.jpg, 00000001.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 0; i < num_images; ++i) {
    const std::string image_name = StringPrintf("%08zu.jpg", i);
    EXPECT_TRUE(ExistsFile(output_path / "pmvs" / "visualize" / image_name));
  }
}

TEST(CMPMVSUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "cmpmvs_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run CMP-MVS undistorter.
  CMPMVSUndistorter undistorter(CMPMVSUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify CMP-MVS output structure was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written with sequential numbering.
  // CMP-MVS writes images as 00001.jpg, 00002.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 1; i <= num_images; ++i) {
    const std::string image_name = StringPrintf("%05zu.jpg", i);
    EXPECT_TRUE(ExistsFile(output_path / image_name));
  }
}

TEST(StandaloneImageUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "pure_output";
  CreateDirIfNotExists(image_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  StandaloneImageUndistorter::Options options;
  for (const auto& [_, image] : reconstruction.Images()) {
    options.image_names_and_cameras.emplace_back(image.Name(),
                                                 *image.CameraPtr());
  }

  // Run standalone image undistorter.
  StandaloneImageUndistorter undistorter(
      options, UndistortCameraOptions(), image_path, output_path);
  undistorter.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written.
  for (const auto& [image_name, camera] : options.image_names_and_cameras) {
    EXPECT_TRUE(ExistsFile(output_path / image_name));
  }
}

TEST(StandaloneImageUndistorter, TargetCamera) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "pure_output";
  CreateDirIfNotExists(image_path);

  // Already-undistorted images are also undistorted to the target camera
  // instead of being copied.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);
  StandaloneImageUndistorter::Options options;
  for (const auto& [_, image] : reconstruction.Images()) {
    options.image_names_and_cameras.emplace_back(
        image.Name(),
        Camera::CreateFromModelId(
            image.CameraId(), CameraModelId::kPinhole, 100, 100, 100));
  }
  // Spherical images are still copied, since the target camera only applies to
  // perspective cameras. Trailing bytes, which image readers ignore, are only
  // kept by copying.
  auto& [spherical_image_name, spherical_camera] =
      options.image_names_and_cameras.back();
  Bitmap spherical_bitmap;
  ASSERT_TRUE(spherical_bitmap.Read(image_path / spherical_image_name));
  {
    std::ofstream file(image_path / spherical_image_name,
                       std::ios::binary | std::ios::app);
    file << "trailing";
  }
  spherical_camera = Camera::CreateFromModelId(spherical_camera.camera_id,
                                               CameraModelId::kEquirectangular,
                                               /*focal_length=*/0.0,
                                               spherical_bitmap.Width(),
                                               spherical_bitmap.Height());

  UndistortCameraOptions camera_options;
  camera_options.target_camera =
      Camera::CreateFromModelId(1, CameraModelId::kPinhole, 50, 60, 40);
  StandaloneImageUndistorter undistorter(
      options, camera_options, image_path, output_path);
  undistorter.Run();

  for (const auto& [image_name, camera] : options.image_names_and_cameras) {
    if (camera.IsSpherical()) {
      std::vector<char> output_data;
      ReadBinaryBlob(output_path / image_name, &output_data);
      std::vector<char> input_data;
      ReadBinaryBlob(image_path / image_name, &input_data);
      EXPECT_EQ(output_data, input_data);
      continue;
    }
    Bitmap bitmap;
    ASSERT_TRUE(bitmap.Read(output_path / image_name));
    EXPECT_EQ(bitmap.Width(), 60);
    EXPECT_EQ(bitmap.Height(), 40);
  }
}

TEST(StereoImageRectifier, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "stereo_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Create stereo pair from first two images.
  StereoImageRectifier::Options options;
  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  ASSERT_GE(image_ids.size(), 2);
  options.stereo_pairs.emplace_back(image_ids[0], image_ids[1]);

  // Run stereo image rectifier.
  StereoImageRectifier rectifier(options,
                                 UndistortCameraOptions(),
                                 reconstruction,
                                 image_path,
                                 output_path);
  rectifier.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify rectified images were written.
  // StereoImageRectifier creates a subdirectory for each stereo pair.
  const auto& image1 = reconstruction.Image(options.stereo_pairs[0].first);
  const auto& image2 = reconstruction.Image(options.stereo_pairs[0].second);
  const std::string stereo_pair_name =
      StringPrintf("%s-%s", image1.Name().c_str(), image2.Name().c_str());
  EXPECT_TRUE(ExistsDir(output_path / stereo_pair_name));
  EXPECT_TRUE(ExistsFile(output_path / stereo_pair_name / image1.Name()));
  EXPECT_TRUE(ExistsFile(output_path / stereo_pair_name / image2.Name()));
}

}  // namespace
}  // namespace colmap
