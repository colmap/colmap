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

#include "colmap/controllers/camera_calibration.h"

#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
namespace colmap {
namespace {

class CameraCalibrationController : public Thread {
 public:
  CameraCalibrationController(
      const std::filesystem::path& database_path,
      const std::filesystem::path& image_path,
      const CameraCalibrationOptions& calibration_options,
      const std::vector<std::string>& image_names)
      : image_path_(image_path),
        calibration_options_(calibration_options),
        image_names_(image_names.begin(), image_names.end()),
        database_(Database::Open(database_path)) {
    THROW_CHECK(calibration_options_.Check());
    THROW_CHECK_DIR_EXISTS(image_path_);
  }

 private:
  void Run() override {
    LOG_HEADING1("Camera calibration");
    Timer run_timer;
    run_timer.Start();

    // NOTE: The calibrator is created lazily, because it loads a large network
    // onto the device. Creating it in the constructor would hold that memory
    // for the entire duration of any preceding pipeline stage (e.g. feature
    // extraction in the automatic reconstruction pipeline). Because this runs
    // in a worker thread, an exception must not escape, as it would terminate
    // the whole process rather than fail this stage.
    std::unique_ptr<CameraCalibrator> calibrator;
    try {
      calibrator = CameraCalibrator::Create(calibration_options_);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to create camera calibrator: " << e.what()
                 << ", cameras unchanged";
      return;
    }

    std::vector<Image> images = database_->ReadAllImages();
    if (!image_names_.empty()) {
      images.erase(std::remove_if(images.begin(),
                                  images.end(),
                                  [this](const Image& image) {
                                    return !image_names_.contains(image.Name());
                                  }),
                   images.end());
    }
    if (images.empty()) {
      LOG(WARNING) << "No selected images in database, skipping calibration";
      return;
    }

    FlatHashMap<camera_t, Camera> cameras;
    for (const Camera& camera : database_->ReadAllCameras()) {
      cameras[camera.camera_id] = camera;
    }
    FlatHashMap<image_t, PosePrior> pose_priors;
    for (PosePrior& pose_prior : database_->ReadAllPosePriors()) {
      if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
        const image_t image_id = pose_prior.corr_data_id.id;
        THROW_CHECK(pose_priors.emplace(image_id, std::move(pose_prior)).second)
            << "Duplicate pose prior for image " << image_id;
      }
    }

    // Calibrate each image; group fitted parameters by camera.
    FlatHashMap<camera_t, std::vector<std::vector<double>>> params_per_camera;
    size_t num_succeeded = 0;
    size_t num_failed = 0;
    for (size_t i = 0; i < images.size(); ++i) {
      if (IsStopped()) {
        return;
      }
      const Image& image = images[i];
      LOG(INFO) << StringPrintf(
          "Calibrating image [%d/%d]", i + 1, images.size());
      LOG(INFO) << StringPrintf("  Name:            %s", image.Name().c_str());

      Bitmap bitmap;
      if (!bitmap.Read(image_path_ / image.Name(), /*as_rgb=*/true)) {
        LOG(WARNING) << "  Failed to read image";
        ++num_failed;
        continue;
      }
      LOG(INFO) << StringPrintf(
          "  Dimensions:      %d x %d", bitmap.Width(), bitmap.Height());

      const auto camera_it = cameras.find(image.CameraId());
      THROW_CHECK(camera_it != cameras.end())
          << "Image references missing camera " << image.CameraId();
      const Camera& camera = camera_it->second;
      // The fitted intrinsics live in the pixel frame of the bitmap, while
      // only the parameters (not the dimensions) are written back to the
      // database camera. Mismatching dimensions would therefore silently
      // store focal length and principal point at the wrong scale.
      if (bitmap.Width() != static_cast<int>(camera.width) ||
          bitmap.Height() != static_cast<int>(camera.height)) {
        LOG(WARNING) << StringPrintf(
            "  Image dimensions %d x %d do not match camera #%d dimensions "
            "%d x %d, skipping",
            bitmap.Width(),
            bitmap.Height(),
            camera.camera_id,
            static_cast<int>(camera.width),
            static_cast<int>(camera.height));
        ++num_failed;
        continue;
      }
      Camera calibrated = camera;
      const auto pose_prior_it = pose_priors.find(image.ImageId());
      const PosePrior pose_prior = pose_prior_it == pose_priors.end()
                                       ? PosePrior()
                                       : pose_prior_it->second;
      bool success = false;
      std::string failure_message;
      try {
        success = calibrator->Calibrate(bitmap, &calibrated, pose_prior);
      } catch (const std::exception& e) {
        failure_message = e.what();
      }
      if (success) {
        params_per_camera[image.CameraId()].push_back(calibrated.params);
        ++num_succeeded;
        LOG(INFO) << StringPrintf("  Camera:          #%d - %s",
                                  calibrated.camera_id,
                                  calibrated.ModelName().c_str());
        LOG(INFO) << "  Parameters:      " << calibrated.ParamsToString();
      } else {
        ++num_failed;
        LOG(WARNING) << "  Calibration failed"
                     << (failure_message.empty() ? "" : ": ") << failure_message
                     << ", keeping existing intrinsics";
      }
    }

    if (IsStopped()) {
      return;
    }

    // Aggregate per-camera parameters and update the database.
    const CameraModelId model_id =
        CameraModelNameToId(calibration_options_.camera_model);
    DatabaseTransaction database_transaction(database_.get());
    size_t num_cameras_updated = 0;
    for (auto& [camera_id, params_list] : params_per_camera) {
      auto it = cameras.find(camera_id);
      THROW_CHECK(it != cameras.end())
          << "Image references missing camera " << camera_id;
      Camera camera = it->second;
      if (AggregateCameraCalibrations(model_id, params_list, &camera)) {
        database_->UpdateCamera(camera);
        ++num_cameras_updated;
        VLOG(1) << "Updated camera " << camera_id << ": "
                << camera.ParamsToString();
      }
    }

    LOG(INFO) << StringPrintf("Calibrated %d/%d images, updated %d/%d cameras",
                              num_succeeded,
                              images.size(),
                              num_cameras_updated,
                              cameras.size());
    if (num_succeeded == 0) {
      LOG(ERROR) << "All image calibrations failed, cameras unchanged";
    } else if (num_failed > 0) {
      LOG(WARNING) << num_failed
                   << " image calibrations failed, keeping existing "
                      "intrinsics for those";
    }
    run_timer.PrintMinutes();
  }

  const std::filesystem::path image_path_;
  const CameraCalibrationOptions calibration_options_;
  const FlatHashSet<std::string> image_names_;
  std::shared_ptr<Database> database_;
};

}  // namespace

std::unique_ptr<Thread> CreateCameraCalibrationController(
    const std::filesystem::path& database_path,
    const std::filesystem::path& image_path,
    const CameraCalibrationOptions& calibration_options,
    const std::vector<std::string>& image_names) {
  return std::make_unique<CameraCalibrationController>(
      database_path, image_path, calibration_options, image_names);
}

}  // namespace colmap
