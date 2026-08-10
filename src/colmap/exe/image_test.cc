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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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
