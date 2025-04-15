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
#include "colmap/geometry/pose.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
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

namespace {

// Update the database with extracted rig and calibrations from the given
// reconstruction derived as follows:
//   * Compute the sensor_from_rig poses as the average of the relative
//     poses between registered sensors in the reconstruction.
//   * Set the camera calibration parameters from the first frame with an image
//     of the camera.
void UpdateRigAndCameraCalibFromReconstruction(
    const Reconstruction& reconstruction,
    const std::map<std::string, std::vector<const Image*>>& frames_to_images,
    Rig rig,
    Database& database) {
  std::unordered_map<
      camera_t,
      std::pair<std::vector<Eigen::Quaterniond>, Eigen::Vector3d>>
      rig_from_cams;
  std::set<camera_t> updated_cameras;
  for (auto& [_, images] : frames_to_images) {
    const Image* ref_image = nullptr;
    for (const Image* image : images) {
      if (rig.IsRefSensor(image->DataId().sensor_id)) {
        ref_image = image;
      }
    }

    if (ref_image == nullptr) {
      continue;
    }

    const Image* rig_calib_ref_image =
        reconstruction.FindImageWithName(ref_image->Name());
    if (rig_calib_ref_image == nullptr || !rig_calib_ref_image->HasPose()) {
      continue;
    }

    const Rigid3d ref_cam_from_world = rig_calib_ref_image->CamFromWorld();
    if (updated_cameras.insert(rig_calib_ref_image->CameraId()).second) {
      Camera ref_camera = *rig_calib_ref_image->CameraPtr();
      ref_camera.camera_id = ref_image->CameraId();
      database.UpdateCamera(ref_camera);
    }

    for (const Image* image : images) {
      if (image->CameraId() != ref_image->CameraId()) {
        const Image* rig_calib_image =
            reconstruction.FindImageWithName(image->Name());
        if (rig_calib_image == nullptr || !rig_calib_image->HasPose()) {
          continue;
        }
        const Rigid3d rig_from_cam =
            ref_cam_from_world * Inverse(rig_calib_image->CamFromWorld());
        auto& [rig_from_cam_rotations, rig_from_cam_translation] =
            rig_from_cams[image->CameraId()];
        if (updated_cameras.insert(rig_calib_image->CameraId()).second) {
          Camera camera = *rig_calib_image->CameraPtr();
          camera.camera_id = image->CameraId();
          database.UpdateCamera(camera);
        }
        if (rig_from_cam_rotations.empty()) {
          rig_from_cam_translation = rig_from_cam.translation;
        } else {
          rig_from_cam_translation += rig_from_cam.translation;
        }
        rig_from_cam_rotations.push_back(rig_from_cam.rotation);
      }
    }
  }

  // Compute the average sensor_from_rig poses over all frames.
  for (auto& [sensor_id, sensor_from_rig] : rig.Sensors()) {
    if (sensor_from_rig.has_value()) {
      continue;
    }

    const auto it = rig_from_cams.find(sensor_id.id);
    if (it == rig_from_cams.end()) {
      LOG(WARNING)
          << "Failed to derive sensor_from_rig transformation for camera "
          << sensor_id.id
          << ", because the image was not registered in the given "
             "reconstruction.";
      continue;
    }

    const auto& [rig_from_cam_rotations, rig_from_cam_translation] = it->second;
    const Rigid3d rig_from_cam(
        AverageQuaternions(
            rig_from_cam_rotations,
            std::vector<double>(rig_from_cam_rotations.size(), 1.0)),
        rig_from_cam_translation / rig_from_cam_rotations.size());
    sensor_from_rig = Inverse(rig_from_cam);
  }

  database.UpdateRig(rig);
}

}  // namespace

// Example for eth3d/delivery_area:
// [
//   {
//     "cameras": [
//       {
//           "image_prefix": "images_rig_cam4_undistorted/",
//           "ref_sensor": true
//       },
//       {
//           "image_prefix": "images_rig_cam5_undistorted/"
//       },
//       {
//           "image_prefix": "images_rig_cam6_undistorted/"
//       },
//       {
//           "image_prefix": "images_rig_cam7_undistorted/"
//       }
//     ]
//   }
// ]
// Example for GoPro cubemaps:
// [
//   {
//       "cameras": [
//           {
//               "image_prefix": "0/",
//               "ref_sensor": true
//           },
//           {
//               "image_prefix": "1/",
//               "cam_from_rig_rotation": [
//                   0.7071067811865475,
//                   0.0,
//                   0.7071067811865476,
//                   0.0
//               ],
//               "cam_from_rig_translation": [
//                   0,
//                   0,
//                   0
//               ]
//           },
//           {
//               "image_prefix": "2/",
//               "cam_from_rig_rotation": [
//                   1.0,
//                   0.0,
//                   0.0,
//                   0.0
//               ],
//               "cam_from_rig_translation": [
//                   0,
//                   0,
//                   0
//               ]
//           }
//       ]
//   }
// ]
int RunRigConfigurator(int argc, char** argv) {
  std::string database_path;
  std::string rig_config_path;
  std::string rig_calib_path;

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("rig_config_path", &rig_config_path);
  options.AddDefaultOption(
      "rig_calib_path",
      &rig_calib_path,
      "Compute the rig and camera calibrations from the given reconstruction.");
  options.Parse(argc, argv);

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.c_str(), pt);

  std::optional<Reconstruction> rig_calib_reconstruction;
  if (!rig_calib_path.empty()) {
    rig_calib_reconstruction = std::make_optional<Reconstruction>();
    rig_calib_reconstruction->Read(rig_calib_path);
  }

  Database database(database_path);

  database.ClearFrames();
  database.ClearRigs();

  const std::vector<Image> images = database.ReadAllImages();

  for (const auto& rig_config : pt) {
    Rig rig;

    int ref_sensor_idx = -1;
    std::vector<std::string> image_prefixes;
    std::vector<std::optional<Rigid3d>> cams_from_rig;
    std::vector<std::optional<Camera>> cameras;
    for (const auto& camera : rig_config.second.get_child("cameras")) {
      image_prefixes.push_back(camera.second.get<std::string>("image_prefix"));

      auto cam_from_rig_rotation_node =
          camera.second.get_child_optional("cam_from_rig_rotation");
      auto cam_from_rig_translation_node =
          camera.second.get_child_optional("cam_from_rig_translation");
      if (cam_from_rig_rotation_node && cam_from_rig_translation_node) {
        Rigid3d cam_from_rig;

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
        cams_from_rig.push_back(cam_from_rig);
      } else {
        cams_from_rig.emplace_back();
      }

      auto ref_sensor_node = camera.second.get_child_optional("ref_sensor");
      if (ref_sensor_node && ref_sensor_node.get().get_value<bool>()) {
        THROW_CHECK(!cam_from_rig_rotation_node &&
                    !cam_from_rig_translation_node)
            << "Reference sensor must not have cam_from_rig";
        ref_sensor_idx = image_prefixes.size() - 1;
      }

      auto camera_model_name_node =
          camera.second.get_child_optional("camera_model_name");
      auto camera_params_node =
          camera.second.get_child_optional("camera_params");
      if (camera_model_name_node && camera_params_node) {
        Camera config_camera;
        config_camera.model_id = CameraModelNameToId(
            camera.second.get<std::string>("camera_model_name"));
        for (const auto& node : camera_params_node.get()) {
          config_camera.params.push_back(node.second.get_value<double>());
        }
        cameras.push_back(config_camera);
      } else {
        cameras.emplace_back();
      }
    }

    const size_t num_cameras = image_prefixes.size();
    THROW_CHECK_GE(ref_sensor_idx, 0) << "No sensor specified as reference";
    THROW_CHECK_LT(ref_sensor_idx, num_cameras);
    THROW_CHECK_EQ(num_cameras, cams_from_rig.size());

    std::vector<std::optional<camera_t>> camera_ids(num_cameras);
    std::map<std::string, std::vector<const Image*>> frames_to_images;
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
            auto& config_camera = cameras[camera_idx];
            if (config_camera.has_value()) {
              Camera camera = database.ReadCamera(image.CameraId());
              camera.model_id = config_camera->model_id;
              camera.params = config_camera->params;
              database.UpdateCamera(camera);
              LOG(INFO) << camera;
            }
          }
        }
      }
    }

    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const std::optional<camera_t>& camera_id = camera_ids[camera_idx];
      THROW_CHECK(camera_id.has_value())
          << "At least one image must exist for each camera in the rig";
      if (static_cast<size_t>(ref_sensor_idx) == camera_idx) {
        rig.AddRefSensor(sensor_t(SensorType::CAMERA, *camera_id));
      } else {
        rig.AddSensor(sensor_t(SensorType::CAMERA, *camera_id),
                      cams_from_rig[camera_idx]);
      }
    }

    rig.SetRigId(database.WriteRig(rig));
    LOG(INFO) << "Configured: " << rig;

    std::unordered_map<
        camera_t,
        std::pair<std::vector<Eigen::Quaterniond>, Eigen::Vector3d>>
        rig_from_cams;
    std::set<camera_t> updated_cameras;
    for (auto& [_, images] : frames_to_images) {
      Frame frame;
      frame.SetRigId(rig.RigId());
      for (const Image* image : images) {
        const data_t& data_id = image->DataId();
        THROW_CHECK(rig.HasSensor(data_id.sensor_id))
            << rig << " must not contain Image(image_id=" << image->ImageId()
            << ", camera_id=" << image->CameraId() << ", name=" << image->Name()
            << ")";
        frame.AddDataId(data_id);
      }
      frame.SetFrameId(database.WriteFrame(frame));
      LOG(INFO) << "Configured: " << frame;
    }

    if (rig_calib_reconstruction) {
      UpdateRigAndCameraCalibFromReconstruction(*rig_calib_reconstruction,
                                                frames_to_images,
                                                std::move(rig),
                                                database);
    }
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
