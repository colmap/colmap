// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/image.h"

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ImageRegistrator, Nominal) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());
  database.reset();

  ASSERT_EQ(reconstruction.NumRegImages(), 3);
  reconstruction.DeRegisterFrame(reconstruction.RegFrameIds().back());
  ASSERT_EQ(reconstruction.NumRegImages(), 2);

  const auto input_path = test_path / "input";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(input_path);
  CreateDirIfNotExists(output_path);
  reconstruction.Write(input_path);

  std::vector<std::string> args = {
      "image_registrator",
      "--database_path",
      database_path.string(),
      "--input_path",
      input_path.string(),
      "--output_path",
      output_path.string(),
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }

  EXPECT_EQ(RunImageRegistrator(argv.size(), argv.data()), EXIT_SUCCESS);

  Reconstruction output_reconstruction;
  output_reconstruction.Read(output_path);
  EXPECT_EQ(output_reconstruction.NumRegImages(), 3);
}

TEST(ImageRegistrator, RegistersImageWithDisconnectedRegisteredImage) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  ASSERT_EQ(reconstruction.NumRegImages(), 3);
  const auto image_ids = reconstruction.RegImageIds();
  const image_t disconnected_image_id = image_ids.front();
  for (const image_t image_id : image_ids) {
    if (image_id != disconnected_image_id) {
      database->DeleteTwoViewGeometry(disconnected_image_id, image_id);
    }
  }
  database.reset();

  const image_t image_to_register_id = image_ids.back();
  reconstruction.DeRegisterFrame(
      reconstruction.Image(image_to_register_id).FrameId());
  ASSERT_EQ(reconstruction.NumRegImages(), 2);

  const auto input_path = test_path / "input";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(input_path);
  CreateDirIfNotExists(output_path);
  reconstruction.Write(input_path);

  const auto image_list_path = test_path / "image-list.txt";
  {
    std::ofstream image_list_file(image_list_path);
    image_list_file << reconstruction.Image(image_to_register_id).Name()
                    << '\n';
  }

  std::vector<std::string> args = {
      "image_registrator",
      "--database_path",
      database_path.string(),
      "--input_path",
      input_path.string(),
      "--output_path",
      output_path.string(),
      "--Mapper.image_list_path",
      image_list_path.string(),
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }

  EXPECT_EQ(RunImageRegistrator(argv.size(), argv.data()), EXIT_SUCCESS);

  Reconstruction output_reconstruction;
  output_reconstruction.Read(output_path);
  EXPECT_EQ(output_reconstruction.NumRegImages(), 3);
  EXPECT_TRUE(output_reconstruction.Image(image_to_register_id).HasPose());
}

}  // namespace
}  // namespace colmap
