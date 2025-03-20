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

#include "colmap/exe/database.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {

int RunDatabaseCleaner(int argc, char** argv) {
  std::string type;

  OptionManager options;
  options.AddRequiredOption("type", &type, "{all, images, features, matches}");
  options.AddDatabaseOptions();
  options.Parse(argc, argv);

  StringToLower(&type);
  Database database(*options.database_path);
  PrintHeading1("Clearing database");
  {
    DatabaseTransaction transaction(&database);
    if (type == "all") {
      PrintHeading2("Clearing all tables");
      database.ClearAllTables();
    } else if (type == "images") {
      PrintHeading2("Clearing Images and all dependent tables");
      database.ClearImages();
      database.ClearTwoViewGeometries();
      database.ClearMatches();
    } else if (type == "features") {
      PrintHeading2("Clearing image features and matches");
      database.ClearDescriptors();
      database.ClearKeypoints();
      database.ClearTwoViewGeometries();
      database.ClearMatches();
    } else if (type == "matches") {
      PrintHeading2("Clearing image matches");
      database.ClearTwoViewGeometries();
      database.ClearMatches();
    } else {
      LOG(ERROR) << "Invalid cleanup type; no changes in database";
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int RunDatabaseCreator(int argc, char** argv) {
  OptionManager options;
  options.AddDatabaseOptions();
  options.Parse(argc, argv);

  Database database(*options.database_path);

  return EXIT_SUCCESS;
}

int RunDatabaseMerger(int argc, char** argv) {
  std::string database_path1;
  std::string database_path2;
  std::string merged_database_path;

  OptionManager options;
  options.AddRequiredOption("database_path1", &database_path1);
  options.AddRequiredOption("database_path2", &database_path2);
  options.AddRequiredOption("merged_database_path", &merged_database_path);
  options.Parse(argc, argv);

  if (ExistsFile(merged_database_path)) {
    LOG(ERROR) << "Merged database file must not exist.";
    return EXIT_FAILURE;
  }

  Database database1(database_path1);
  Database database2(database_path2);
  Database merged_database(merged_database_path);
  Database::Merge(database1, database2, &merged_database);

  return EXIT_SUCCESS;
}

// Example for eth3d/delivery_area:
// [
//   {
//       "cameras": [
//           {
//               "image_prefix": "images_rig_cam4_undistorted/",
//               "ref_sensor": true
//           },
//           {
//               "image_prefix": "images_rig_cam5_undistorted/"
//           },
//           {
//               "image_prefix": "images_rig_cam6_undistorted/"
//           },
//           {
//               "image_prefix": "images_rig_cam7_undistorted/"
//           }
//       ]
//   }
// ]
int RunRigConfigurator(int argc, char** argv) {
  std::string database_path;
  std::string rig_config_path;

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("rig_config_path", &rig_config_path);
  options.Parse(argc, argv);

  Database database(database_path);

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.c_str(), pt);

  database.ClearFrames();
  database.ClearRigs();

  const std::vector<Image> images = database.ReadAllImages();

  for (const auto& rig_config : pt) {
    Rig rig;

    int ref_sensor_idx = -1;
    std::vector<std::string> image_prefixes;
    std::vector<Rigid3d> cams_from_rig;
    for (const auto& camera : rig_config.second.get_child("cameras")) {
      image_prefixes.push_back(camera.second.get<std::string>("image_prefix"));
      Rigid3d& cam_from_rig = cams_from_rig.emplace_back();

      auto cam_from_rig_rotation_node =
          camera.second.get_child_optional("cam_from_rig_rotation");
      auto cam_from_rig_translation_node =
          camera.second.get_child_optional("cam_from_rig_translation");
      if (cam_from_rig_rotation_node && cam_from_rig_translation_node) {
        int index = 0;
        Eigen::Vector4d cam_from_rig_wxyz;
        for (const auto& node : cam_from_rig_rotation_node.get()) {
          cam_from_rig_wxyz[index++] = node.second.get_value<double>();
        }
        cam_from_rig.rotation = Eigen::Quaterniond(cam_from_rig_wxyz(0),
                                                   cam_from_rig_wxyz(1),
                                                   cam_from_rig_wxyz(2),
                                                   cam_from_rig_wxyz(3));

        THROW_CHECK(cam_from_rig_translation_node);
        index = 0;
        for (const auto& node : cam_from_rig_translation_node.get()) {
          cam_from_rig.translation(index++) = node.second.get_value<double>();
        }
      }

      auto ref_sensor_node = camera.second.get_child_optional("ref_sensor");
      if (ref_sensor_node && ref_sensor_node.get().get_value<bool>()) {
        THROW_CHECK(!cam_from_rig_rotation_node &&
                    !cam_from_rig_translation_node)
            << "Reference sensor must not have cam_from_rig";
        ref_sensor_idx = image_prefixes.size() - 1;
      }
    }

    const size_t num_cameras = image_prefixes.size();
    THROW_CHECK_GE(ref_sensor_idx, 0) << "No sensor specified as reference";
    THROW_CHECK_LT(ref_sensor_idx, num_cameras);
    THROW_CHECK_EQ(num_cameras, cams_from_rig.size());

    std::vector<std::optional<camera_t>> camera_ids(num_cameras);
    std::unordered_map<std::string, std::vector<const Image*>> frames_to_images;
    for (const Image& image : images) {
      for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        const std::string& image_prefix = image_prefixes[camera_idx];
        if (StringStartsWith(image.Name(), image_prefix)) {
          const std::string image_suffix =
              StringGetAfter(image.Name(), image_prefix);
          frames_to_images[image_suffix].push_back(&image);
          std::optional<camera_t>& camera_id = camera_ids[camera_idx];
          if (camera_id.has_value()) {
            THROW_CHECK_EQ(*camera_id, image.CameraId())
                << "Inconsistent cameras for images with prefix: "
                << image_prefix
                << ". Consider setting --ImageReader.single_camera_per_folder "
                   "during feature extraction or manually assign consistent "
                   "camera_id's.";
          } else {
            camera_id = image.CameraId();
          }
        }
      }
    }

    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const std::optional<camera_t>& camera_id = camera_ids[camera_idx];
      THROW_CHECK(camera_id.has_value())
          << "At least one image must exist for each camera in the rig";
      if (ref_sensor_idx == camera_idx) {
        rig.AddRefSensor(sensor_t(SensorType::CAMERA, *camera_id));
      } else {
        rig.AddSensor(sensor_t(SensorType::CAMERA, *camera_id),
                      cams_from_rig[camera_idx]);
      }
    }

    rig.SetRigId(database.WriteRig(rig));
    LOG(INFO) << "Configured: " << rig;

    for (auto& [_, images] : frames_to_images) {
      Frame frame;
      frame.SetRigId(rig.RigId());
      for (const Image* image : images) {
        const data_t data_id = image->DataId();
        THROW_CHECK(rig.HasSensor(data_id.sensor_id))
            << rig << " must not contain Image(image_id=" << image->ImageId()
            << ", camera_id=" << image->CameraId() << ", name=" << image->Name()
            << ")";
        frame.AddDataId(data_id);
      }
      frame.SetFrameId(database.WriteFrame(frame));
      LOG(INFO) << "Configured: " << frame;
    }
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
