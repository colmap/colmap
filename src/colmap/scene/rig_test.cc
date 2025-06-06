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

#include "colmap/scene/rig.h"

#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

std::string WriteTestConfig(const std::string& config) {
  std::string file_path = CreateTestDir() + "/config.json";
  std::ofstream file(file_path);
  file << config << '\n';
  return file_path;
}

TEST(ReadRigConfig, Empty) {
  EXPECT_THAT(ReadRigConfig(WriteTestConfig("[]")), testing::IsEmpty());
}

TEST(ReadRigConfig, InvalidJson) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig("")));
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig("[{")));
}

TEST(ReadRigConfig, MissingImagePrefix) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
          [
            {
              "cameras": [
                {
                    "ref_sensor": true,
                }
              ]
            }
          ]
          )")));
}

TEST(ReadRigConfig, InvalidRefSensor) {
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
          [
            {
              "cameras": [
                {
                    "image_prefix": "rig1/camera1/",
                    "ref_sensor": true,
                },
                {
                    "image_prefix": "rig1/camera2/",
                    "ref_sensor": true,
                }
              ]
            }
          ]
          )")));
  EXPECT_ANY_THROW(ReadRigConfig(WriteTestConfig(R"(
              [
                {
                  "cameras": [
                    {
                        "image_prefix": "rig1/camera1/",
                    },
                    {
                        "image_prefix": "rig1/camera2/",
                    }
                  ]
                }
              ]
              )")));
}

TEST(ReadRigConfig, Nominal) {
  const std::vector<RigConfig> configs = ReadRigConfig(WriteTestConfig(R"(
[
  {
    "cameras": [
      {
          "image_prefix": "rig1/camera1/",
          "ref_sensor": true,
          "camera_model_name": "OPENCV",
          "camera_params": [640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4]
      },
      {
          "image_prefix": "rig1/camera2/",
          "cam_from_rig_rotation": [0, 1, 0, 0],
          "cam_from_rig_translation": [1, 2, 3]
      }
    ]
  },
  {
    "cameras": [
      {
          "image_prefix": "rig2/camera1/",
          "ref_sensor": true
      },
      {
          "image_prefix": "rig2/camera2/"
      }
    ]
  }
]
)"));
  ASSERT_EQ(configs.size(), 2);
  ASSERT_EQ(configs[0].cameras.size(), 2);
  ASSERT_EQ(configs[1].cameras.size(), 2);

  EXPECT_EQ(configs[0].cameras[0].image_prefix, "rig1/camera1/");
  EXPECT_TRUE(configs[0].cameras[0].ref_sensor);
  ASSERT_FALSE(configs[0].cameras[0].cam_from_rig.has_value());
  ASSERT_TRUE(configs[0].cameras[0].camera.has_value());
  EXPECT_EQ(configs[0].cameras[0].camera->model_id, CameraModelId::kOpenCV);
  EXPECT_TRUE(configs[0].cameras[0].camera->has_prior_focal_length);
  EXPECT_THAT(configs[0].cameras[0].camera->params,
              testing::ElementsAre(640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4));

  EXPECT_EQ(configs[0].cameras[1].image_prefix, "rig1/camera2/");
  EXPECT_FALSE(configs[0].cameras[1].ref_sensor);
  ASSERT_TRUE(configs[0].cameras[1].cam_from_rig.has_value());
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->rotation.coeffs(),
            Eigen::Vector4d(1, 0, 0, 0));
  EXPECT_EQ(configs[0].cameras[1].cam_from_rig->translation,
            Eigen::Vector3d(1, 2, 3));
  ASSERT_FALSE(configs[0].cameras[1].camera.has_value());

  EXPECT_EQ(configs[1].cameras[0].image_prefix, "rig2/camera1/");
  EXPECT_TRUE(configs[1].cameras[0].ref_sensor);
  ASSERT_FALSE(configs[1].cameras[0].cam_from_rig.has_value());
  ASSERT_FALSE(configs[1].cameras[0].camera.has_value());

  EXPECT_EQ(configs[1].cameras[1].image_prefix, "rig2/camera2/");
  EXPECT_FALSE(configs[1].cameras[1].ref_sensor);
  ASSERT_FALSE(configs[1].cameras[1].cam_from_rig.has_value());
  ASSERT_FALSE(configs[1].cameras[1].camera.has_value());
}

TEST(ApplyRigConfig, WithReconstruction) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 2;
  options.num_frames_per_rig = 5;
  SynthesizeDataset(options, &reconstruction, &database);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera000001_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera000002_";

  ApplyRigConfig(configs, database, &reconstruction);
  EXPECT_EQ(database.NumRigs(), 1);
  EXPECT_EQ(database.NumFrames(), options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumRegFrames(), options.num_frames_per_rig);
}

TEST(ApplyRigConfig, WithPartialReconstruction) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 2;
  options.num_frames_per_rig = 5;
  SynthesizeDataset(options, &reconstruction, &database);

  reconstruction.DeRegisterFrame(1);
  reconstruction.DeRegisterFrame(3);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera000001_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera000002_";

  ApplyRigConfig(configs, database, &reconstruction);
  EXPECT_EQ(database.NumRigs(), 1);
  EXPECT_EQ(database.NumFrames(), options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumRegFrames(), options.num_frames_per_rig - 2);
}

TEST(ApplyRigConfig, WithoutReconstruction) {
  Database database(Database::kInMemoryDatabasePath);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 2;
  options.num_frames_per_rig = 5;
  SynthesizeDataset(options, &reconstruction, &database);

  std::vector<RigConfig> configs;
  auto& config = configs.emplace_back();
  auto& camera1 = config.cameras.emplace_back();
  camera1.image_prefix = "camera000001_";
  camera1.ref_sensor = true;
  auto& camera2 = config.cameras.emplace_back();
  camera2.image_prefix = "camera000002_";
  camera2.camera =
      Camera::CreateFromModelId(2, CameraModelId::kOpenCV, 2.0, 1024, 768);
  camera2.cam_from_rig =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());

  ApplyRigConfig(configs, database);
  EXPECT_EQ(database.NumRigs(), 1);
  EXPECT_EQ(database.NumFrames(), options.num_frames_per_rig);
  EXPECT_EQ(reconstruction.NumRigs(), 1);
  EXPECT_EQ(reconstruction.NumFrames(), options.num_frames_per_rig);
  const auto [sensor_id2, sensor2_from_rig] =
      *database.ReadAllRigs().at(0).Sensors().begin();
  EXPECT_EQ(sensor2_from_rig.value().rotation.coeffs(),
            camera2.cam_from_rig.value().rotation.coeffs());
  EXPECT_EQ(sensor2_from_rig.value().translation,
            camera2.cam_from_rig.value().translation);
  EXPECT_EQ(database.ReadCamera(sensor_id2.id), camera2.camera);
}

}  // namespace
}  // namespace colmap
