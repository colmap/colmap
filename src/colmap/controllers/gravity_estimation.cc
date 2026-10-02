// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/gravity_estimation.h"

#include "colmap/calibration/single_view_calibrator.h"
#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <map>

namespace colmap {
namespace {

struct FrameImageGroup {
  frame_t frame_id = kInvalidFrameId;
  std::vector<Image> images;
  std::vector<Eigen::Quaterniond> cam_from_rig;
};

class GravityEstimationController : public Thread {
 public:
  GravityEstimationController(const std::filesystem::path& database_path,
                              const std::filesystem::path& image_path,
                              const GravityEstimationOptions& options,
                              const std::vector<std::string>& image_names,
                              GeoCalibFactory geocalib_factory)
      : image_path_(image_path),
        options_(options),
        image_names_(image_names.begin(), image_names.end()),
        geocalib_factory_(std::move(geocalib_factory)),
        database_(Database::Open(database_path)) {
    THROW_CHECK(options_.Check());
    THROW_CHECK_DIR_EXISTS(image_path_);
  }

 private:
  void Run() override {
    LOG_HEADING1("Gravity estimation (GeoCalib)");
    Timer run_timer;
    run_timer.Start();

    std::vector<Image> all_images = database_->ReadAllImages();
    if (!image_names_.empty()) {
      all_images.erase(
          std::remove_if(all_images.begin(),
                         all_images.end(),
                         [this](const Image& image) {
                           return !image_names_.contains(image.Name());
                         }),
          all_images.end());
    }
    if (all_images.empty()) {
      LOG(WARNING) << "No selected images in database, skipping gravity "
                      "estimation";
      return;
    }

    FlatHashMap<camera_t, Camera> cameras;
    for (const Camera& camera : database_->ReadAllCameras()) {
      cameras[camera.camera_id] = camera;
    }
    FlatHashMap<rig_t, Rig> rigs;
    for (const Rig& rig : database_->ReadAllRigs()) {
      rigs[rig.RigId()] = rig;
    }
    FlatHashMap<frame_t, Frame> frames;
    for (const Frame& frame : database_->ReadAllFrames()) {
      frames[frame.FrameId()] = frame;
    }
    FlatHashMap<image_t, PosePrior> pose_priors;
    for (PosePrior& pose_prior : database_->ReadAllPosePriors()) {
      if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
        const image_t image_id = pose_prior.corr_data_id.id;
        pose_priors.emplace(image_id, std::move(pose_prior));
      }
    }

    // Group selected images by Frame (falling back to single-image groups for
    // frameless images or uncalibrated non-reference rig sensors).
    std::vector<FrameImageGroup> groups;
    std::map<frame_t, std::vector<Image>> images_by_frame;
    for (const Image& image : all_images) {
      if (image.HasFrameId() && frames.contains(image.FrameId())) {
        images_by_frame[image.FrameId()].push_back(image);
      } else {
        FrameImageGroup single_group;
        single_group.images.push_back(image);
        single_group.cam_from_rig.push_back(Eigen::Quaterniond::Identity());
        groups.push_back(std::move(single_group));
      }
    }

    for (auto& [frame_id, frame_images] : images_by_frame) {
      const Frame& frame = frames.at(frame_id);
      const auto rig_it = rigs.find(frame.RigId());
      if (rig_it == rigs.end()) {
        for (const Image& image : frame_images) {
          FrameImageGroup single_group;
          single_group.frame_id = frame_id;
          single_group.images.push_back(image);
          single_group.cam_from_rig.push_back(Eigen::Quaterniond::Identity());
          groups.push_back(std::move(single_group));
        }
        continue;
      }

      const Rig& rig = rig_it->second;
      FrameImageGroup calibrated_group;
      calibrated_group.frame_id = frame_id;
      for (const Image& image : frame_images) {
        const sensor_t sensor_id = image.DataId().sensor_id;
        if (rig.IsRefSensor(sensor_id)) {
          calibrated_group.images.push_back(image);
          calibrated_group.cam_from_rig.push_back(
              Eigen::Quaterniond::Identity());
        } else if (rig.HasSensorFromRig(sensor_id)) {
          calibrated_group.images.push_back(image);
          calibrated_group.cam_from_rig.push_back(
              rig.SensorFromRig(sensor_id).rotation());
        } else {
          // Non-reference sensor without known sensor_from_rig: estimate its
          // sensor-frame gravity independently.
          FrameImageGroup single_group;
          single_group.frame_id = frame_id;
          single_group.images.push_back(image);
          single_group.cam_from_rig.push_back(Eigen::Quaterniond::Identity());
          groups.push_back(std::move(single_group));
        }
      }
      if (!calibrated_group.images.empty()) {
        groups.push_back(std::move(calibrated_group));
      }
    }

    std::unique_ptr<GeoCalib> geocalib;
    try {
      geocalib = geocalib_factory_ ? geocalib_factory_(options_.geocalib)
                                   : GeoCalib::Create(options_.geocalib);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to create GeoCalib: " << e.what();
      return;
    }

    const CameraModelId configured_model_id =
        options_.camera_model.empty()
            ? CameraModelId::kInvalid
            : CameraModelNameToId(options_.camera_model);
    if (!options_.camera_model.empty() &&
        configured_model_id == CameraModelId::kInvalid) {
      LOG(ERROR) << "Invalid camera model: " << options_.camera_model;
      return;
    }

    FlatHashMap<camera_t, std::vector<std::vector<double>>> params_per_camera;
    FlatHashMap<image_t, Eigen::Vector3d> estimated_gravities;
    size_t num_frames_succeeded = 0;
    size_t num_frames_failed = 0;

    for (size_t g_idx = 0; g_idx < groups.size(); ++g_idx) {
      if (IsStopped()) {
        return;
      }
      const FrameImageGroup& group = groups[g_idx];

      // Check if all images in this group already have gravity and no camera
      // needs intrinsic refinement.
      bool needs_gravity = false;
      bool needs_camera_refinement = false;
      for (const Image& image : group.images) {
        const auto prior_it = pose_priors.find(image.ImageId());
        if (options_.overwrite_gravity || prior_it == pose_priors.end() ||
            !prior_it->second.HasGravity()) {
          needs_gravity = true;
        }
        const Camera& cam = cameras.at(image.CameraId());
        if (options_.refine_intrinsics &&
            (options_.force_refine_intrinsics || !cam.has_prior_focal_length)) {
          needs_camera_refinement = true;
        }
      }
      if (!needs_gravity && !needs_camera_refinement) {
        continue;
      }

      LOG(INFO) << StringPrintf(
          "Estimating gravity for frame group [%d/%d] "
          "(%d image(s))",
          g_idx + 1,
          groups.size(),
          group.images.size());

      std::vector<PerspectiveField> fields;
      std::vector<size_t> valid_img_indices;
      fields.reserve(group.images.size());
      valid_img_indices.reserve(group.images.size());

      for (size_t k = 0; k < group.images.size(); ++k) {
        const Image& image = group.images[k];
        const Camera& camera = cameras.at(image.CameraId());

        Bitmap bitmap;
        if (!bitmap.Read(image_path_ / image.Name(), /*as_rgb=*/true)) {
          LOG(WARNING) << "  Failed to read image: " << image.Name();
          continue;
        }
        if (bitmap.Width() != static_cast<int>(camera.width) ||
            bitmap.Height() != static_cast<int>(camera.height)) {
          LOG(WARNING) << StringPrintf(
              "  Image %s dimensions %d x %d do not match camera #%d "
              "dimensions %d x %d, skipping",
              image.Name().c_str(),
              bitmap.Width(),
              bitmap.Height(),
              camera.camera_id,
              static_cast<int>(camera.width),
              static_cast<int>(camera.height));
          continue;
        }

        const auto pose_prior_it = pose_priors.find(image.ImageId());
        const PosePrior pose_prior = pose_prior_it == pose_priors.end()
                                         ? PosePrior()
                                         : pose_prior_it->second;
        try {
          fields.push_back(
              geocalib->PredictPerspectiveField(bitmap, pose_prior));
          valid_img_indices.push_back(k);
        } catch (const std::exception& e) {
          LOG(WARNING) << "  Perspective field prediction failed for "
                       << image.Name() << ": " << e.what();
        }
      }

      if (fields.empty()) {
        ++num_frames_failed;
        continue;
      }

      // Prepare working camera copies per unique camera_id in this group so
      // shared cameras within a frame are optimized jointly.
      std::map<camera_t, Camera> working_cameras;
      std::map<camera_t, bool> refine_camera_flags;
      for (const size_t k : valid_img_indices) {
        const Image& image = group.images[k];
        const camera_t camera_id = image.CameraId();
        if (working_cameras.find(camera_id) != working_cameras.end()) {
          continue;
        }
        const Camera& orig_cam = cameras.at(camera_id);
        const bool refine_cam =
            options_.refine_intrinsics && (options_.force_refine_intrinsics ||
                                           !orig_cam.has_prior_focal_length);
        refine_camera_flags[camera_id] = refine_cam;

        if (refine_cam && configured_model_id != CameraModelId::kInvalid &&
            configured_model_id != orig_cam.model_id) {
          Camera converted =
              Camera::CreateFromModelId(orig_cam.camera_id,
                                        configured_model_id,
                                        orig_cam.MeanFocalLength(),
                                        orig_cam.width,
                                        orig_cam.height);
          converted.has_prior_focal_length = orig_cam.has_prior_focal_length;
          working_cameras[camera_id] = std::move(converted);
        } else {
          working_cameras[camera_id] = orig_cam;
        }
      }

      std::vector<PerspectiveFieldCameraInput> inputs(fields.size());
      for (size_t i = 0; i < fields.size(); ++i) {
        const size_t k = valid_img_indices[i];
        const Image& image = group.images[k];
        const camera_t camera_id = image.CameraId();
        inputs[i].field = &fields[i];
        inputs[i].camera = &working_cameras.at(camera_id);
        inputs[i].cam_from_rig = group.cam_from_rig[k];
        inputs[i].refine_camera = refine_camera_flags.at(camera_id);
      }

      const FittedPerspectiveFields fitted =
          FitPerspectiveFields(options_.geocalib.fitting, inputs);
      if (!fitted.success) {
        ++num_frames_failed;
        LOG(WARNING) << "  Perspective field fitting failed for frame group";
        continue;
      }

      ++num_frames_succeeded;
      LOG(INFO) << StringPrintf("  Gravity (rig):   [%.4f, %.4f, %.4f]",
                                fitted.gravity_in_rig.x(),
                                fitted.gravity_in_rig.y(),
                                fitted.gravity_in_rig.z());

      for (const size_t k : valid_img_indices) {
        const Image& image = group.images[k];
        const Eigen::Vector3d gravity_in_cam =
            (group.cam_from_rig[k] * fitted.gravity_in_rig).normalized();
        estimated_gravities[image.ImageId()] = gravity_in_cam;
      }

      for (auto& [camera_id, working_cam] : working_cameras) {
        if (!refine_camera_flags.at(camera_id)) {
          continue;
        }
        working_cam.has_prior_focal_length = true;
        if (!IsValidCalibration(working_cam) ||
            working_cam.HasBogusParams(options_.min_focal_length_ratio,
                                       options_.max_focal_length_ratio,
                                       options_.max_extra_param)) {
          LOG(WARNING) << "  Rejecting implausible calibrated parameters for "
                       << "camera #" << camera_id << " ("
                       << working_cam.ModelName()
                       << "): " << working_cam.ParamsToString();
          continue;
        }
        params_per_camera[camera_id].push_back(working_cam.params);
        LOG(INFO) << StringPrintf("  Camera #%d (%s): %s",
                                  camera_id,
                                  working_cam.ModelName().c_str(),
                                  working_cam.ParamsToString().c_str());
      }
    }

    if (IsStopped()) {
      return;
    }

    DatabaseTransaction database_transaction(database_.get());
    size_t num_priors_written = 0;
    for (const Image& image : all_images) {
      const auto grav_it = estimated_gravities.find(image.ImageId());
      if (grav_it == estimated_gravities.end()) {
        continue;
      }
      auto prior_it = pose_priors.find(image.ImageId());
      if (prior_it == pose_priors.end()) {
        PosePrior prior;
        prior.corr_data_id = image.DataId();
        prior.gravity = grav_it->second;
        database_->WritePosePrior(prior);
        ++num_priors_written;
      } else if (options_.overwrite_gravity || !prior_it->second.HasGravity()) {
        prior_it->second.gravity = grav_it->second;
        database_->UpdatePosePrior(prior_it->second);
        ++num_priors_written;
      }
    }

    size_t num_cameras_updated = 0;
    for (auto& [camera_id, params_list] : params_per_camera) {
      Camera camera = cameras.at(camera_id);
      const CameraModelId model_id =
          configured_model_id == CameraModelId::kInvalid ? camera.model_id
                                                         : configured_model_id;
      if (AggregateSingleViewCalibrations(model_id, params_list, &camera)) {
        database_->UpdateCamera(camera);
        ++num_cameras_updated;
        LOG(INFO) << "Updated camera #" << camera_id << " ("
                  << camera.ModelName()
                  << ") median parameters: " << camera.ParamsToString();
      }
    }

    LOG(INFO) << StringPrintf(
        "Succeeded on %d/%d frame groups, wrote %d gravity priors, updated "
        "%d/%d cameras",
        num_frames_succeeded,
        groups.size(),
        num_priors_written,
        num_cameras_updated,
        cameras.size());
    if (num_frames_failed > 0) {
      LOG(WARNING) << num_frames_failed << " frame groups failed";
    }
    run_timer.PrintMinutes();
  }

  const std::filesystem::path image_path_;
  const GravityEstimationOptions options_;
  const FlatHashSet<std::string> image_names_;
  const GeoCalibFactory geocalib_factory_;
  const std::shared_ptr<Database> database_;
};

}  // namespace

bool GravityEstimationOptions::Check() const {
  if (!camera_model.empty()) {
    CHECK_OPTION(ExistsCameraModelWithName(camera_model));
    const CameraModelId model_id = CameraModelNameToId(camera_model);
    CHECK_OPTION(CameraModelIsPerspective(model_id));
  }
  CHECK_OPTION_GT(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GT(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GT(max_extra_param, 0.0);
  CHECK_OPTION(geocalib.Check());
  return true;
}

std::unique_ptr<Thread> CreateGravityEstimationController(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const GravityEstimationOptions& options,
    const std::vector<std::string>& image_names,
    GeoCalibFactory geocalib_factory) {
  return std::make_unique<GravityEstimationController>(
      database_path,
      image_path,
      options,
      image_names,
      std::move(geocalib_factory));
}

}  // namespace colmap
