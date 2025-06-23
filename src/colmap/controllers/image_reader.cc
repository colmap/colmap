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

#include "colmap/controllers/image_reader.h"

#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"

namespace colmap {

bool ImageReaderOptions::Check() const {
  CHECK_OPTION_GT(default_focal_length_factor, 0.0);
  CHECK_OPTION(ExistsCameraModelWithName(camera_model));
  const CameraModelId model_id = CameraModelNameToId(camera_model);
  if (!camera_params.empty()) {
    CHECK_OPTION(
        CameraModelVerifyParams(model_id, CSVToVector<double>(camera_params)));
  }
  return true;
}

ImageReader::ImageReader(const ImageReaderOptions& options, Database* database)
    : options_(options), database_(database), image_index_(0) {
  THROW_CHECK(options_.Check());

  // Ensure trailing slash, so that we can build the correct image name.
  options_.image_path =
      EnsureTrailingSlash(StringReplace(options_.image_path, "\\", "/"));
  if (!options_.mask_path.empty()) {
    options_.mask_path =
        EnsureTrailingSlash(StringReplace(options_.mask_path, "\\", "/"));
  }

  // Get a list of all files in the image path, sorted by image name.
  if (options_.image_names.empty()) {
    options_.image_names = GetRecursiveFileList(options_.image_path);
    std::sort(options_.image_names.begin(), options_.image_names.end());
  } else {
    if (!std::is_sorted(options_.image_names.begin(),
                        options_.image_names.end())) {
      std::sort(options_.image_names.begin(), options_.image_names.end());
    }
    for (auto& image_name : options_.image_names) {
      image_name = JoinPaths(options_.image_path, image_name);
    }
  }

  if (static_cast<camera_t>(options_.existing_camera_id) != kInvalidCameraId) {
    THROW_CHECK(database->ExistsCamera(options_.existing_camera_id));
    prev_camera_ = database->ReadCamera(options_.existing_camera_id);
    if (std::optional<Rig> rig =
            database->ReadRigWithSensor(prev_camera_.SensorId());
        rig.has_value()) {
      prev_rig_ = std::move(*rig);
    } else {
      // For backwards compatibility with old databases without rigs.
      prev_rig_.AddRefSensor(prev_camera_.SensorId());
      prev_rig_.SetRigId(database_->WriteRig(prev_rig_));
    }
  } else {
    // Set the manually specified camera parameters.
    prev_camera_.camera_id = kInvalidCameraId;
    THROW_CHECK(ExistsCameraModelWithName(options_.camera_model));
    prev_camera_.model_id = CameraModelNameToId(options_.camera_model);
    prev_camera_.params.resize(CameraModelNumParams(prev_camera_.model_id), 0.);
    if (!options_.camera_params.empty()) {
      THROW_CHECK(prev_camera_.SetParamsFromString(options_.camera_params));
      prev_camera_.has_prior_focal_length = true;
    }
  }
}

ImageReader::Status ImageReader::Next(Rig* rig,
                                      Camera* camera,
                                      Image* image,
                                      PosePrior* pose_prior,
                                      Bitmap* bitmap,
                                      Bitmap* mask) {
  THROW_CHECK_NOTNULL(camera);
  THROW_CHECK_NOTNULL(image);
  THROW_CHECK_NOTNULL(bitmap);

  image_index_ += 1;
  THROW_CHECK_LE(image_index_, options_.image_names.size());

  const std::string image_path = options_.image_names.at(image_index_ - 1);

  DatabaseTransaction database_transaction(database_);

  //////////////////////////////////////////////////////////////////////////////
  // Set the image name.
  //////////////////////////////////////////////////////////////////////////////

  image->SetName(image_path);
  image->SetName(StringReplace(image->Name(), "\\", "/"));
  image->SetName(
      image->Name().substr(options_.image_path.size(),
                           image->Name().size() - options_.image_path.size()));

  const std::string image_folder = GetParentDir(image->Name());

  //////////////////////////////////////////////////////////////////////////////
  // Check if image already read.
  //////////////////////////////////////////////////////////////////////////////

  const bool exists_image = database_->ExistsImageWithName(image->Name());

  if (exists_image) {
    *image = database_->ReadImageWithName(image->Name()).value();
    const bool exists_keypoints = database_->ExistsKeypoints(image->ImageId());
    const bool exists_descriptors =
        database_->ExistsDescriptors(image->ImageId());

    if (exists_keypoints && exists_descriptors) {
      return Status::IMAGE_EXISTS;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Read image.
  //////////////////////////////////////////////////////////////////////////////

  if (!bitmap->Read(image_path, false)) {
    return Status::BITMAP_ERROR;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Read mask.
  //////////////////////////////////////////////////////////////////////////////

  if (mask && !options_.mask_path.empty()) {
    std::string mask_path =
        JoinPaths(options_.mask_path, image->Name() + ".png");
    if (!ExistsFile(mask_path)) {
      bool exists_mask = false;
      if (HasFileExtension(image->Name(), ".png")) {
        std::string alt_mask_path =
            JoinPaths(options_.mask_path, image->Name());
        if (ExistsFile(alt_mask_path)) {
          mask_path = std::move(alt_mask_path);
          exists_mask = true;
        }
      }
      if (!exists_mask) {
        LOG(ERROR) << "Mask at " << mask_path << " does not exist.";
        return Status::MASK_ERROR;
      }
    }
    if (!mask->Read(mask_path, false)) {
      LOG(ERROR) << "Failed to read invalid mask file at: " << mask_path;
      return Status::MASK_ERROR;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Check for well-formed data.
  //////////////////////////////////////////////////////////////////////////////

  if (exists_image) {
    Camera current_camera = database_->ReadCamera(image->CameraId());

    if (options_.single_camera && prev_camera_.camera_id != kInvalidCameraId &&
        (current_camera.width != prev_camera_.width ||
         current_camera.height != prev_camera_.height)) {
      return Status::CAMERA_SINGLE_DIM_ERROR;
    }

    if (static_cast<size_t>(bitmap->Width()) != current_camera.width ||
        static_cast<size_t>(bitmap->Height()) != current_camera.height) {
      return Status::CAMERA_EXIST_DIM_ERROR;
    }

    prev_camera_ = std::move(current_camera);
    if (std::optional<Rig> rig =
            database_->ReadRigWithSensor(prev_camera_.SensorId());
        rig.has_value()) {
      prev_rig_ = std::move(rig.value());
    } else {
      // For backwards compatibility with old databases, we create a rig.
      prev_rig_ = Rig();
      prev_rig_.AddRefSensor(prev_camera_.SensorId());
      prev_rig_.SetRigId(database_->WriteRig(prev_rig_));
    }

  } else {
    //////////////////////////////////////////////////////////////////////////////
    // Check image dimensions.
    //////////////////////////////////////////////////////////////////////////////

    if (prev_camera_.camera_id != kInvalidCameraId &&
        ((options_.single_camera && !options_.single_camera_per_folder) ||
         (options_.single_camera_per_folder &&
          image_folder == prev_image_folder_)) &&
        (prev_camera_.width != static_cast<size_t>(bitmap->Width()) ||
         prev_camera_.height != static_cast<size_t>(bitmap->Height()))) {
      return Status::CAMERA_SINGLE_DIM_ERROR;
    }

    //////////////////////////////////////////////////////////////////////////////
    // Read camera model and check for consistency if it exists
    //////////////////////////////////////////////////////////////////////////////

    std::string camera_model;
    const bool valid_camera_model = bitmap->ExifCameraModel(&camera_model);
    if (camera_model_to_id_.count(camera_model) > 0) {
      Camera camera =
          database_->ReadCamera(camera_model_to_id_.at(camera_model));
      if (camera.width != static_cast<size_t>(bitmap->Width()) ||
          camera.height != static_cast<size_t>(bitmap->Height())) {
        return Status::CAMERA_EXIST_DIM_ERROR;
      }
      prev_camera_ = std::move(camera);
      if (std::optional<Rig> rig =
              database_->ReadRigWithSensor(prev_camera_.SensorId());
          rig.has_value()) {
        prev_rig_ = std::move(rig.value());
      } else {
        // For backwards compatibility with old databases, we create a rig.
        prev_rig_ = Rig();
        prev_rig_.AddRefSensor(prev_camera_.SensorId());
        prev_rig_.SetRigId(database_->WriteRig(prev_rig_));
      }
    }

    //////////////////////////////////////////////////////////////////////////////
    // Extract camera model and focal length
    //////////////////////////////////////////////////////////////////////////////

    if (prev_camera_.camera_id == kInvalidCameraId ||
        options_.single_camera_per_image ||
        (!options_.single_camera && !options_.single_camera_per_folder &&
         static_cast<camera_t>(options_.existing_camera_id) ==
             kInvalidCameraId &&
         camera_model_to_id_.count(camera_model) == 0) ||
        (options_.single_camera_per_folder &&
         image_folders_.count(image_folder) == 0)) {
      if (options_.camera_params.empty()) {
        // Extract focal length.
        double focal_length = 0.0;
        bool has_focal_length = false;
        if (bitmap->ExifFocalLength(&focal_length)) {
          has_focal_length = true;
        } else {
          focal_length = options_.default_focal_length_factor *
                         std::max(bitmap->Width(), bitmap->Height());
        }

        prev_camera_ = Camera::CreateFromModelId(prev_camera_.camera_id,
                                                 prev_camera_.model_id,
                                                 focal_length,
                                                 bitmap->Width(),
                                                 bitmap->Height());
        prev_camera_.has_prior_focal_length = has_focal_length;
      }

      prev_camera_.width = static_cast<size_t>(bitmap->Width());
      prev_camera_.height = static_cast<size_t>(bitmap->Height());

      if (!prev_camera_.VerifyParams()) {
        return Status::CAMERA_PARAM_ERROR;
      }

      prev_camera_.camera_id = database_->WriteCamera(prev_camera_);

      // By default we create a separate rig per camera. Grouping of different
      // cameras into the same rig is expected to be done with the
      // "rig_configurator" after feature extraction.
      if (!database_->ReadRigWithSensor(prev_camera_.SensorId()).has_value()) {
        prev_rig_ = Rig();
        prev_rig_.AddRefSensor(prev_camera_.SensorId());
        prev_rig_.SetRigId(database_->WriteRig(prev_rig_));
      }

      if (valid_camera_model) {
        camera_model_to_id_[camera_model] = prev_camera_.camera_id;
      }
    }

    image->SetCameraId(prev_camera_.camera_id);

    //////////////////////////////////////////////////////////////////////////////
    // Extract GPS data.
    //////////////////////////////////////////////////////////////////////////////

    Eigen::Vector3d position_prior;
    if (bitmap->ExifLatitude(&position_prior.x()) &&
        bitmap->ExifLongitude(&position_prior.y()) &&
        bitmap->ExifAltitude(&position_prior.z())) {
      pose_prior->position = position_prior;
      pose_prior->coordinate_system = PosePrior::CoordinateSystem::WGS84;
    }
  }

  *camera = prev_camera_;
  *rig = prev_rig_;

  image_folders_.insert(image_folder);
  prev_image_folder_ = image_folder;

  return Status::SUCCESS;
}

size_t ImageReader::NextIndex() const { return image_index_; }

size_t ImageReader::NumImages() const { return options_.image_names.size(); }

std::string ImageReader::StatusToString(const ImageReader::Status status) {
  switch (status) {
    case ImageReader::Status::SUCCESS:
      return "SUCCESS";
    case ImageReader::Status::FAILURE:
      return "FAILURE: Failed to process the image.";
    case ImageReader::Status::IMAGE_EXISTS:
      return "IMAGE_EXISTS: Features for image were already extracted.";
    case ImageReader::Status::BITMAP_ERROR:
      return "BITMAP_ERROR: Failed to read the image file format.";
    case ImageReader::Status::MASK_ERROR:
      return "MASK_ERROR: Failed to read the mask file.";
    case ImageReader::Status::CAMERA_SINGLE_DIM_ERROR:
      return "CAMERA_SINGLE_DIM_ERROR: Single camera specified, but images "
             "have different dimensions.";
    case ImageReader::Status::CAMERA_EXIST_DIM_ERROR:
      return "CAMERA_EXIST_DIM_ERROR: Image previously processed, but current "
             "image has different dimensions.";
    case ImageReader::Status::CAMERA_PARAM_ERROR:
      return "CAMERA_PARAM_ERROR: Camera has invalid parameters.";
    default:
      return "Unknown";
  }
}

}  // namespace colmap
