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
    // Create the calibrator eagerly to fail fast on missing models.
    calibrator_ = CameraCalibrator::Create(calibration_options_);
  }

 private:
  void Run() override {
    LOG_HEADING1("Camera calibration");
    Timer run_timer;
    run_timer.Start();

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

    // Calibrate each image; group fitted parameters by camera.
    FlatHashMap<camera_t, std::vector<std::vector<double>>> params_per_camera;
    size_t num_succeeded = 0;
    size_t num_failed = 0;
    for (size_t i = 0; i < images.size(); ++i) {
      if (IsStopped()) {
        return;
      }
      const Image& image = images[i];
      Bitmap bitmap;
      if (!bitmap.Read(image_path_ / image.Name(), /*as_rgb=*/true)) {
        LOG(WARNING) << "Failed to read image: " << image.Name();
        ++num_failed;
        continue;
      }
      Camera calibrated;
      calibrated.camera_id = image.CameraId();
      bool success = false;
      try {
        success = calibrator_->Calibrate(bitmap, &calibrated);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Calibration failed for image " << image.Name() << ": "
                     << e.what();
      }
      if (success) {
        params_per_camera[image.CameraId()].push_back(calibrated.params);
        ++num_succeeded;
        VLOG(1) << "Calibrated image " << image.Name() << ": "
                << calibrated.ParamsToString();
      } else {
        ++num_failed;
        VLOG(1) << "Calibration failed for image " << image.Name()
                << ", keeping existing intrinsics";
      }
      if ((i + 1) % 10 == 0 || i + 1 == images.size()) {
        LOG(INFO) << StringPrintf(
            "Calibrated %d/%d images", i + 1, images.size());
      }
    }

    if (IsStopped()) {
      return;
    }

    // Aggregate per-camera parameters and update the database.
    DatabaseTransaction database_transaction(database_.get());
    size_t num_cameras_updated = 0;
    for (auto& [camera_id, params_list] : params_per_camera) {
      auto it = cameras.find(camera_id);
      THROW_CHECK(it != cameras.end())
          << "Image references missing camera " << camera_id;
      Camera& camera = it->second;
      if (AggregateCameraCalibrations(params_list, &camera)) {
        camera.model_id =
            CameraModelNameToId(calibration_options_.camera_model);
        database_->UpdateCamera(camera);
        ++num_cameras_updated;
        VLOG(1) << "Updated camera " << camera_id << ": "
                << camera.ParamsToString();
      }
    }

    LOG(INFO) << StringPrintf(
        "Calibrated %d/%d images, updated %d/%d cameras in %.3fs",
        num_succeeded,
        images.size(),
        num_cameras_updated,
        cameras.size(),
        run_timer.ElapsedSeconds());
    if (num_succeeded == 0) {
      LOG(ERROR) << "All image calibrations failed, cameras unchanged";
    } else if (num_failed > 0) {
      LOG(WARNING) << num_failed
                   << " image calibrations failed, keeping existing "
                      "intrinsics for those";
    }
  }

  const std::filesystem::path image_path_;
  const CameraCalibrationOptions calibration_options_;
  const FlatHashSet<std::string> image_names_;
  std::shared_ptr<Database> database_;
  std::unique_ptr<CameraCalibrator> calibrator_;
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
