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
void UpdateRigAndCameraCalibFromReconstruction(
    const Reconstruction& reconstruction,
    const std::map<std::string, std::vector<const Image*>>& frames_to_images,
    Rig& rig,
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

}  // namespace

std::vector<RigConfig> ReadRigConfig(const std::string& rig_config_path) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(rig_config_path.c_str(), pt);

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

  for (const RigConfig& config : configs) {
    Rig rig;

    const size_t num_cameras = config.cameras.size();

    std::vector<std::optional<camera_t>> camera_ids(num_cameras);
    std::map<std::string, std::vector<const Image*>> frames_to_images;
    for (const Image& image : images) {
      for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
        const auto& config_camera = config.cameras[camera_idx];
        if (StringStartsWith(image.Name(), config_camera.image_prefix)) {
          const std::string image_suffix =
              StringGetAfter(image.Name(), config_camera.image_prefix);
          frames_to_images[image_suffix].push_back(&image);
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

    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& config_camera = config.cameras[camera_idx];
      const std::optional<camera_t>& camera_id = camera_ids[camera_idx];
      THROW_CHECK(camera_id.has_value())
          << "At least one image must exist for each camera in the rig";
      if (config_camera.ref_sensor) {
        rig.AddRefSensor(sensor_t(SensorType::CAMERA, *camera_id));
      } else {
        rig.AddSensor(sensor_t(SensorType::CAMERA, *camera_id),
                      config_camera.cam_from_rig);
      }
    }

    rig.SetRigId(database.WriteRig(rig));
    LOG(INFO) << "Configured: " << rig;

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

    if (reconstruction != nullptr) {
      UpdateRigAndCameraCalibFromReconstruction(
          *reconstruction, frames_to_images, rig, database);
      // Set the frame poses from the first image we find in each frame.
      std::vector<Frame> frames = database.ReadAllFrames();
      for (Frame& frame : frames) {
        for (const data_t& data_id : frame.ImageIds()) {
          // Note that the input reconstruction may have a different assignment
          // of images and image_ids, so we associate them uniquely by name.
          // In addition, not all images in the database may be present in the
          // input reconstruction, e.g., when self-calibrating the rigs and
          // cameras from a subset of images.
          const Image* image = reconstruction->FindImageWithName(
              database.ReadImage(data_id.id).Name());
          if (image == nullptr || !image->HasPose()) {
            continue;
          }
          const sensor_t sensor_id = image->CameraPtr()->SensorId();
          if (rig.IsRefSensor(sensor_id)) {
            frame.SetRigFromWorld(image->CamFromWorld());
          } else {
            frame.SetRigFromWorld(Inverse(rig.SensorFromRig(sensor_id)) *
                                  image->CamFromWorld());
          }
          break;
        }
      }
      reconstruction->SetRigsAndFrames(database.ReadAllRigs(),
                                       std::move(frames));
    }
  }
}

}  // namespace colmap
