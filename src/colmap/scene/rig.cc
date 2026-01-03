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

#include "colmap/geometry/pose.h"

#include <filesystem>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {
namespace {

// Update the database with extracted rig and calibrations from the given
// reconstruction derived as follows:
//   * Compute the sensor_from_rig poses as the average of the relative
//     poses between registered sensors in the reconstruction.
//   * Set the camera calibration parameters from the first frame with an image
//     of the camera.
void UpdateRigAndCameraCalibsFromReconstruction(
    const Reconstruction& reconstruction,
    const std::map<std::string, std::vector<const Image*>>&
        frame_name_to_images,
    Rig& rig,
    Database& database) {
  std::unordered_map<
      camera_t,
      std::pair<std::vector<Eigen::Quaterniond>, Eigen::Vector3d>>
      rig_from_cams;
  std::set<camera_t> updated_cameras;
  for (auto& [_, images] : frame_name_to_images) {
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
  for (auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
    if (sensor_from_rig.has_value()) {
      // Do not compute it for explicitly provided poses in the config.
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

void UpdateRigsAndFramesFromDatabase(const Database& database,
                                     Reconstruction* reconstruction) {
  const std::vector<Frame> database_frames = database.ReadAllFrames();

  std::unordered_map<rig_t, Rig> database_rigs;
  database_rigs.reserve(database.NumRigs());
  for (auto& rig : database.ReadAllRigs()) {
    database_rigs.emplace(rig.RigId(), std::move(rig));
  }

  std::unordered_map<rig_t, Rig> reconstruction_rigs;
  reconstruction_rigs.reserve(database_rigs.size());

  // Create O(1) lookup table from image names to images.
  std::unordered_map<std::string, const Image*> image_name_to_image;
  image_name_to_image.reserve(reconstruction->NumImages());
  for (const auto& [_, image] : reconstruction->Images()) {
    image_name_to_image.emplace(image.Name(), &image);
  }

  auto visit_frame_data =
      [&database, &image_name_to_image, &database_frames](
          const std::function<void(const Frame&, const Image&, const Image&)>&
              visitor) {
        for (const Frame& database_frame : database_frames) {
          for (const data_t& data_id : database_frame.ImageIds()) {
            const Image database_image = database.ReadImage(data_id.id);
            const auto reconstruction_image =
                image_name_to_image.find(database_image.Name());
            if (reconstruction_image == image_name_to_image.end()) {
              continue;
            }
            visitor(
                database_frame, database_image, *reconstruction_image->second);
          }
        }
      };

  // Update reference sensors in reconstruction rigs.
  // (must be done before updating the non-reference sensors).
  visit_frame_data([&](const Frame& database_frame,
                       const Image& database_image,
                       const Image& reconstruction_image) {
    const Rig& database_rig = database_rigs.at(database_frame.RigId());
    Rig& reconstruction_rig = reconstruction_rigs[database_frame.RigId()];
    reconstruction_rig.SetRigId(database_frame.RigId());

    const sensor_t& database_sensor_id = database_image.DataId().sensor_id;
    const sensor_t& reconstruction_sensor_id =
        reconstruction_image.CameraPtr()->SensorId();

    if (!reconstruction_rig.IsRefSensor(reconstruction_sensor_id) &&
        database_rig.IsRefSensor(database_sensor_id)) {
      reconstruction_rig.AddRefSensor(reconstruction_sensor_id);
    }
  });

  // Update non-reference sensors in reconstruction rigs.
  visit_frame_data([&](const Frame& database_frame,
                       const Image& database_image,
                       const Image& reconstruction_image) {
    const Rig& database_rig = database_rigs.at(database_frame.RigId());
    Rig& reconstruction_rig = reconstruction_rigs[database_frame.RigId()];
    reconstruction_rig.SetRigId(database_frame.RigId());

    const sensor_t& database_sensor_id = database_image.DataId().sensor_id;
    const sensor_t& reconstruction_sensor_id =
        reconstruction_image.CameraPtr()->SensorId();

    if (reconstruction_rig.NonRefSensors().count(reconstruction_sensor_id) ==
            0 &&
        database_rig.NonRefSensors().count(database_sensor_id) != 0) {
      reconstruction_rig.AddSensor(
          reconstruction_sensor_id,
          database_rig.SensorFromRig(database_sensor_id));
    }
  });

  // Update reconstruction frames.
  std::unordered_map<frame_t, Frame> reconstruction_frames;
  reconstruction_frames.reserve(database_frames.size());
  visit_frame_data([&](const Frame& database_frame,
                       const Image& database_image,
                       const Image& reconstruction_image) {
    const Rig& database_rig = database_rigs.at(database_frame.RigId());
    const sensor_t& database_sensor_id = database_image.DataId().sensor_id;
    Frame& reconstruction_frame =
        reconstruction_frames[database_frame.FrameId()];
    reconstruction_frame.SetFrameId(database_frame.FrameId());
    reconstruction_frame.SetRigId(database_frame.RigId());
    reconstruction_frame.AddDataId(reconstruction_image.DataId());
    if (reconstruction_image.HasPose()) {
      if (database_rig.IsRefSensor(database_sensor_id)) {
        reconstruction_frame.SetRigFromWorld(
            reconstruction_image.CamFromWorld());
      } else {
        reconstruction_frame.SetRigFromWorld(
            Inverse(database_rig.SensorFromRig(database_sensor_id)) *
            reconstruction_image.CamFromWorld());
      }
    }
  });

  std::vector<Rig> rigs;
  rigs.reserve(reconstruction_rigs.size());
  for (auto& [_, rig] : reconstruction_rigs) {
    rigs.push_back(std::move(rig));
  }

  std::vector<Frame> frames;
  frames.reserve(reconstruction_frames.size());
  for (auto& [_, frame] : reconstruction_frames) {
    frames.push_back(std::move(frame));
  }

  reconstruction->SetRigsAndFrames(std::move(rigs), std::move(frames));
}

}  // namespace

std::vector<RigConfig> ReadRigConfig(
    const std::filesystem::path& rig_config_path) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.string().c_str(), pt);

  std::vector<RigConfig> configs;
  for (const auto& rig_node : pt) {
    RigConfig& config = configs.emplace_back();
    bool has_ref_sensor = false;
    for (const auto& camera : rig_node.second.get_child("cameras")) {
      RigConfig::RigCamera& config_camera = config.cameras.emplace_back();

      config_camera.image_prefix =
          camera.second.get<std::string>("image_prefix");

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
        config_camera.cam_from_rig = cam_from_rig;
      }

      auto ref_sensor_node = camera.second.get_child_optional("ref_sensor");
      if (ref_sensor_node && ref_sensor_node.get().get_value<bool>()) {
        THROW_CHECK(!cam_from_rig_rotation_node &&
                    !cam_from_rig_translation_node)
            << "Reference sensor must not have cam_from_rig";
        THROW_CHECK(!has_ref_sensor)
            << "Rig must only have one reference sensor";
        config_camera.ref_sensor = true;
        has_ref_sensor = true;
      }

      auto camera_model_name_node =
          camera.second.get_child_optional("camera_model_name");
      auto camera_params_node =
          camera.second.get_child_optional("camera_params");
      if (camera_model_name_node && camera_params_node) {
        config_camera.camera = std::make_optional<Camera>();
        config_camera.camera->model_id = CameraModelNameToId(
            camera.second.get<std::string>("camera_model_name"));
        config_camera.camera->has_prior_focal_length = true;
        for (const auto& node : camera_params_node.get()) {
          config_camera.camera->params.push_back(
              node.second.get_value<double>());
        }
      }
    }

    THROW_CHECK(has_ref_sensor) << "Rig must have one reference sensor";
  }

  return configs;
}

void ApplyRigConfig(const std::vector<RigConfig>& configs,
                    Database& database,
                    Reconstruction* reconstruction) {
  database.ClearFrames();
  database.ClearRigs();

  const std::vector<Image> images = database.ReadAllImages();
  std::set<image_t> configured_image_ids;

  for (const RigConfig& config : configs) {
    Rig rig;

    const size_t num_cameras = config.cameras.size();

    std::vector<std::optional<camera_t>> camera_ids(num_cameras);
    std::map<std::string, std::vector<const Image*>> frame_name_to_images;
    for (const Image& image : images) {
      for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        const auto& config_camera = config.cameras[camera_idx];
        if (StringStartsWith(image.Name(), config_camera.image_prefix)) {
          const std::string frame_name =
              StringGetAfter(image.Name(), config_camera.image_prefix);
          frame_name_to_images[frame_name].push_back(&image);
          std::optional<camera_t>& camera_id = camera_ids[camera_idx];
          if (camera_id.has_value()) {
            THROW_CHECK_EQ(*camera_id, image.CameraId())
                << "Inconsistent cameras for images with prefix: "
                << config_camera.image_prefix
                << ". Consider setting --ImageReader.single_camera_per_folder "
                   "during feature extraction or manually assign consistent "
                   "camera_id's.";
          } else {
            camera_id = image.CameraId();
            if (config_camera.camera.has_value()) {
              Camera database_camera = database.ReadCamera(image.CameraId());
              database_camera.model_id = config_camera.camera->model_id;
              database_camera.params = config_camera.camera->params;
              database.UpdateCamera(database_camera);
              if (reconstruction != nullptr) {
                auto& reconstruction_camera =
                    reconstruction->Camera(image.CameraId());
                reconstruction_camera.model_id = config_camera.camera->model_id;
                reconstruction_camera.params = config_camera.camera->params;
              }
            }
          }
        }
      }
    }

    std::set<camera_t> unique_camera_ids;
    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& config_camera = config.cameras[camera_idx];
      std::optional<camera_t>& camera_id = camera_ids[camera_idx];
      THROW_CHECK(camera_id.has_value())
          << "At least one image must exist for each camera in the rig";
      if (!unique_camera_ids.insert(*camera_id).second) {
        // Clone the camera, if multiple cameras in the rig share a camera.
        *camera_id = database.WriteCamera(database.ReadCamera(*camera_id));
      }
      if (config_camera.ref_sensor) {
        rig.AddRefSensor(sensor_t(SensorType::CAMERA, *camera_id));
      } else {
        rig.AddSensor(sensor_t(SensorType::CAMERA, *camera_id),
                      config_camera.cam_from_rig);
      }
    }

    rig.SetRigId(database.WriteRig(rig));
    LOG(INFO) << "Configured: " << rig;

    for (auto& [frame_name, images] : frame_name_to_images) {
      Frame frame;
      frame.SetRigId(rig.RigId());
      for (const Image* image : images) {
        const data_t& data_id = image->DataId();
        THROW_CHECK(rig.HasSensor(data_id.sensor_id))
            << rig << " must not contain Image(image_id=" << image->ImageId()
            << ", camera_id=" << image->CameraId() << ", name=" << image->Name()
            << ")";
        frame.AddDataId(data_id);
        configured_image_ids.insert(image->ImageId());
      }
      frame.SetFrameId(database.WriteFrame(frame));
      LOG(INFO) << "Configured: " << frame;
    }

    if (reconstruction != nullptr) {
      UpdateRigAndCameraCalibsFromReconstruction(
          *reconstruction, frame_name_to_images, rig, database);
    }
  }

  // Create trivial rigs/frames for images without configuration.
  // This is necessary because we clear rigs/frames above.
  std::unordered_map<camera_t, rig_t> camera_to_rig_id;
  for (const Image& image : images) {
    if (configured_image_ids.count(image.ImageId()) > 0) {
      continue;
    }
    const sensor_t sensor_id(SensorType::CAMERA, image.CameraId());
    auto rig_id_it = camera_to_rig_id.find(image.CameraId());
    if (rig_id_it == camera_to_rig_id.end()) {
      Rig rig;
      rig.AddRefSensor(sensor_id);
      rig_id_it =
          camera_to_rig_id.emplace(image.CameraId(), database.WriteRig(rig))
              .first;
    }
    Frame frame;
    frame.SetRigId(rig_id_it->second);
    frame.AddDataId(data_t(sensor_id, image.ImageId()));
    frame.SetFrameId(database.WriteFrame(frame));
  }

  if (reconstruction != nullptr) {
    UpdateRigsAndFramesFromDatabase(database, reconstruction);
  }
}

}  // namespace colmap
