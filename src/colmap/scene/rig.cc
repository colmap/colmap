// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/rig.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/string.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace colmap {
namespace {

using ImageNameToImage = NodeHashMap<std::string, const Image*>;

// Update the database with extracted rig and calibrations from the given
// reconstruction derived as follows:
//   * Compute the sensor_from_rig poses as the average of the relative
//     poses between registered sensors in the reconstruction.
//   * Set the camera calibration parameters from the first frame with an image
//     of the camera.
void UpdateRigAndCameraCalibsFromReconstruction(
    const ImageNameToImage& image_name_to_image,
    const std::map<std::string, std::vector<const Image*>>&
        frame_name_to_images,
    Rig& rig,
    Database& database,
    const NodeHashMap<camera_t, Camera>& camera_replacements) {
  const auto camera_for = [&](const Image& image) -> const Camera& {
    const auto replacement = camera_replacements.find(image.CameraId());
    if (replacement != camera_replacements.end()) return replacement->second;
    return *image.CameraPtr();
  };
  NodeHashMap<camera_t,
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

    const auto ref_it = image_name_to_image.find(ref_image->Name());
    const Image* rig_calib_ref_image =
        ref_it == image_name_to_image.end() ? nullptr : ref_it->second;
    if (rig_calib_ref_image == nullptr || !rig_calib_ref_image->HasPose()) {
      continue;
    }

    const Rigid3d ref_cam_from_world = rig_calib_ref_image->CamFromWorld();
    if (updated_cameras.insert(rig_calib_ref_image->CameraId()).second) {
      Camera ref_camera = camera_for(*rig_calib_ref_image);
      ref_camera.camera_id = ref_image->CameraId();
      database.UpdateCamera(ref_camera);
    }

    for (const Image* image : images) {
      if (image->CameraId() != ref_image->CameraId()) {
        const auto image_it = image_name_to_image.find(image->Name());
        const Image* rig_calib_image =
            image_it == image_name_to_image.end() ? nullptr : image_it->second;
        if (rig_calib_image == nullptr || !rig_calib_image->HasPose()) {
          continue;
        }
        const Rigid3d rig_from_cam =
            ref_cam_from_world * Inverse(rig_calib_image->CamFromWorld());
        auto& [rig_from_cam_rotations, rig_from_cam_translation] =
            rig_from_cams[image->CameraId()];
        if (updated_cameras.insert(rig_calib_image->CameraId()).second) {
          Camera camera = camera_for(*rig_calib_image);
          camera.camera_id = image->CameraId();
          database.UpdateCamera(camera);
        }
        if (rig_from_cam_rotations.empty()) {
          rig_from_cam_translation = rig_from_cam.translation();
        } else {
          rig_from_cam_translation += rig_from_cam.translation();
        }
        rig_from_cam_rotations.push_back(rig_from_cam.rotation());
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

struct PreparedRigFrames {
  std::vector<Rig> rigs;
  std::vector<Frame> frames;
};

PreparedRigFrames PrepareRigsAndFramesFromDatabase(
    const Database& database,
    const std::vector<Image>& database_images,
    const ImageNameToImage& image_name_to_image) {
  const std::vector<Frame> database_frames = database.ReadAllFrames();

  NodeHashMap<rig_t, Rig> database_rigs;
  database_rigs.reserve(database.NumRigs());
  for (auto& rig : database.ReadAllRigs()) {
    database_rigs.emplace(rig.RigId(), std::move(rig));
  }

  NodeHashMap<rig_t, Rig> reconstruction_rigs;
  reconstruction_rigs.reserve(database_rigs.size());

  // Join the loaded image namespace once; all three owner passes reuse it.
  NodeHashMap<image_t, const Image*> database_image_by_id;
  database_image_by_id.reserve(database_images.size());
  for (const Image& image : database_images) {
    database_image_by_id.emplace(image.ImageId(), &image);
  }
  struct FrameData {
    const Frame* frame;
    const Image* database_image;
    const Image* reconstruction_image;
  };
  std::vector<FrameData> joined_images;
  joined_images.reserve(database_images.size());
  for (const Frame& frame : database_frames) {
    for (const data_t& data_id : frame.ImageIds()) {
      const Image& image = *database_image_by_id.at(data_id.id);
      const auto match = image_name_to_image.find(image.Name());
      if (match != image_name_to_image.end()) {
        joined_images.push_back({&frame, &image, match->second});
      }
    }
  }
  const auto visit_frame_data = [&](const auto& visitor) {
    for (const FrameData& data : joined_images) {
      visitor(*data.frame, *data.database_image, *data.reconstruction_image);
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
  NodeHashMap<frame_t, Frame> reconstruction_frames;
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
      // Images grouped into one rig frame may have independently estimated,
      // slightly inconsistent poses. The reference sensor directly defines
      // the shared rig pose, so prefer it over poses derived from other sensors
      // using the averaged sensor_from_rig calibration. If the reference image
      // has no pose, keep the first available non-reference pose as a fallback.
      if (database_rig.IsRefSensor(database_sensor_id)) {
        reconstruction_frame.SetRigFromWorld(
            reconstruction_image.CamFromWorld());
      } else if (!reconstruction_frame.HasPose()) {
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

  return {std::move(rigs), std::move(frames)};
}

void CopyCameraIntrinsics(const Camera& src, Camera& dst) {
  dst.model_id = src.model_id;
  dst.params = src.params;
  dst.source = src.source;
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
          // Note: get_value<double>() is locale-dependent, so parse the raw
          // token with the locale-independent StringToDouble instead.
          cam_from_rig_wxyz[index++] =
              StringToDouble(node.second.get_value<std::string>());
        }
        cam_from_rig.rotation() = Eigen::Quaterniond(cam_from_rig_wxyz(0),
                                                     cam_from_rig_wxyz(1),
                                                     cam_from_rig_wxyz(2),
                                                     cam_from_rig_wxyz(3));

        THROW_CHECK(cam_from_rig_translation_node);
        index = 0;
        for (const auto& node : cam_from_rig_translation_node.get()) {
          cam_from_rig.translation()(index++) =
              StringToDouble(node.second.get_value<std::string>());
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
        config_camera.camera->source = CameraSource::USER;
        for (const auto& node : camera_params_node.get()) {
          config_camera.camera->params.push_back(
              StringToDouble(node.second.get_value<std::string>()));
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
  auto savepoint = database.CreateSavepoint();
  database.ClearFrames();
  database.ClearRigs();

  const std::vector<Image> images = database.ReadAllImages();
  std::set<image_t> configured_image_ids;
  NodeHashMap<camera_t, Camera> camera_replacements;

  // Match only database images against the reconstruction's image namespace.
  // Reuse the lookup when deriving calibration and rebuilding rig frames.
  ImageNameToImage image_name_to_image;
  if (reconstruction != nullptr) {
    image_name_to_image.reserve(reconstruction->NumImages());
    for (const auto& [_, image] : reconstruction->Images()) {
      image_name_to_image.emplace(image.Name(), &image);
    }
  }

  for (const RigConfig& config : configs) {
    Rig rig;

    const size_t num_cameras = config.cameras.size();

    std::vector<std::optional<camera_t>> camera_ids(num_cameras);
    std::vector<std::set<camera_t>> updated_reconstruction_camera_ids(
        num_cameras);
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
                << ". Consider setting "
                   "--ImageReader.single_camera_per_folder "
                   "during feature extraction or manually assign consistent "
                   "camera_id's.";
          } else {
            camera_id = image.CameraId();
            if (config_camera.camera.has_value()) {
              Camera database_camera = database.ReadCamera(image.CameraId());
              CopyCameraIntrinsics(*config_camera.camera, database_camera);
              database.UpdateCamera(database_camera);
            }
          }
          if (config_camera.camera.has_value() && reconstruction != nullptr) {
            const auto reconstruction_image_it =
                image_name_to_image.find(image.Name());
            if (reconstruction_image_it != image_name_to_image.end()) {
              const Image& reconstruction_image =
                  *reconstruction_image_it->second;
              const camera_t reconstruction_camera_id =
                  reconstruction_image.CameraId();
              THROW_CHECK(
                  reconstruction->ExistsCamera(reconstruction_camera_id));
              Camera& reconstruction_camera =
                  reconstruction->Camera(reconstruction_camera_id);
              THROW_CHECK(reconstruction_image.HasCameraPtr());
              THROW_CHECK_EQ(reconstruction_image.CameraPtr(),
                             &reconstruction_camera);
              if (updated_reconstruction_camera_ids[camera_idx]
                      .insert(reconstruction_camera_id)
                      .second) {
                auto [replacement, inserted] = camera_replacements.try_emplace(
                    reconstruction_camera_id, reconstruction_camera);
                CopyCameraIntrinsics(*config_camera.camera,
                                     replacement->second);
              }
            }
          }
        }
      }
    }

    // Sensors are identified by their camera, so images of different sensors
    // must not share a camera. Otherwise, their images would all be assigned
    // to the same sensor in the frames below.
    std::map<camera_t, size_t> camera_id_to_idx;
    for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
      const auto& config_camera = config.cameras[camera_idx];
      const std::optional<camera_t>& camera_id = camera_ids[camera_idx];
      THROW_CHECK(camera_id.has_value())
          << "At least one image must exist for each camera in the rig";
      const auto [it, inserted] =
          camera_id_to_idx.emplace(*camera_id, camera_idx);
      THROW_CHECK(inserted)
          << "Images with prefixes " << config.cameras[it->second].image_prefix
          << " and " << config_camera.image_prefix << " share camera "
          << *camera_id
          << ". Consider setting --ImageReader.single_camera_per_folder "
             "during feature extraction or manually assign distinct "
             "camera_id's.";
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
      UpdateRigAndCameraCalibsFromReconstruction(image_name_to_image,
                                                 frame_name_to_images,
                                                 rig,
                                                 database,
                                                 camera_replacements);
    }
  }

  // Create trivial rigs/frames for images without configuration.
  // This is necessary because we clear rigs/frames above.
  NodeHashMap<camera_t, rig_t> camera_to_rig_id;
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
    auto prepared =
        PrepareRigsAndFramesFromDatabase(database, images, image_name_to_image);
    reconstruction->SetRigsAndFramesPrepared(std::move(prepared.rigs),
                                             std::move(prepared.frames),
                                             std::move(camera_replacements),
                                             [&] { savepoint->Release(); });
  } else {
    savepoint->Release();
  }
}

}  // namespace colmap
