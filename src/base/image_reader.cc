// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "base/image_reader.h"

#include "util/misc.h"

namespace colmap {

bool ImageReaderOptions::Check() const {
  CHECK_OPTION_GT(default_focal_length_factor, 0.0);
  CHECK_OPTION(ExistsCameraModelWithName(camera_model));
  const int model_id = CameraModelNameToId(camera_model);
  if (!camera_params.empty()) {
    CHECK_OPTION(
        CameraModelVerifyParams(model_id, CSVToVector<double>(camera_params)));
  }
  return true;
}

ImageReader::ImageReader(const ImageReaderOptions& options, Database* database)
    : options_(options), database_(database), image_index_(0) {
  CHECK(options_.Check());

  // Ensure trailing slash, so that we can build the correct image name.
  options_.image_path =
      EnsureTrailingSlash(StringReplace(options_.image_path, "\\", "/"));

  // Get a list of all files in the image path, sorted by image name.
  if (options_.image_list.empty()) {
    options_.image_list = GetRecursiveFileList(options_.image_path);
    std::sort(options_.image_list.begin(), options_.image_list.end());
  } else {
    for (auto& image_name : options_.image_list) {
      image_name = JoinPaths(options_.image_path, image_name);
    }
  }

  // Set the manually specified camera parameters.
  prev_camera_.SetCameraId(kInvalidCameraId);
  prev_camera_.SetModelIdFromName(options_.camera_model);
  if (!options_.camera_params.empty()) {
    prev_camera_.SetParamsFromString(options_.camera_params);
  }
}

ImageReader::Status ImageReader::Next(Camera* camera, Image* image,
                                      Bitmap* bitmap) {
  CHECK_NOTNULL(camera);
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(bitmap);

  image_index_ += 1;
  CHECK_LE(image_index_, options_.image_list.size());

  const std::string image_path = options_.image_list.at(image_index_ - 1);

  DatabaseTransaction database_transaction(database_);

  //////////////////////////////////////////////////////////////////////////////
  // Set the image name.
  //////////////////////////////////////////////////////////////////////////////

  image->SetName(image_path);
  image->SetName(StringReplace(image->Name(), "\\", "/"));
  image->SetName(
      image->Name().substr(options_.image_path.size(),
                           image->Name().size() - options_.image_path.size()));

  //////////////////////////////////////////////////////////////////////////////
  // Check if image already read.
  //////////////////////////////////////////////////////////////////////////////

  const bool exists_image = database_->ExistsImageWithName(image->Name());

  if (exists_image) {
    *image = database_->ReadImageWithName(image->Name());
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
  // Check for well-formed data.
  //////////////////////////////////////////////////////////////////////////////

  if (exists_image) {
    const Camera camera = database_->ReadCamera(image->CameraId());

    if (options_.single_camera && prev_camera_.CameraId() != kInvalidCameraId &&
        (camera.Width() != prev_camera_.Width() ||
         camera.Height() != prev_camera_.Height())) {
      return Status::CAMERA_SINGLE_ERROR;
    }

    if (static_cast<size_t>(bitmap->Width()) != camera.Width() ||
        static_cast<size_t>(bitmap->Height()) != camera.Height()) {
      return Status::CAMERA_DIM_ERROR;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Extract image dimensions.
  //////////////////////////////////////////////////////////////////////////////

  if (options_.single_camera && prev_camera_.CameraId() != kInvalidCameraId &&
      (prev_camera_.Width() != static_cast<size_t>(bitmap->Width()) ||
       prev_camera_.Height() != static_cast<size_t>(bitmap->Height()))) {
    return Status::CAMERA_SINGLE_ERROR;
  }

  prev_camera_.SetWidth(static_cast<size_t>(bitmap->Width()));
  prev_camera_.SetHeight(static_cast<size_t>(bitmap->Height()));

  //////////////////////////////////////////////////////////////////////////////
  // Extract camera model and focal length
  //////////////////////////////////////////////////////////////////////////////

  if (!options_.single_camera || prev_camera_.CameraId() == kInvalidCameraId) {
    if (options_.camera_params.empty()) {
      // Extract focal length.
      double focal_length = 0.0;
      if (bitmap->ExifFocalLength(&focal_length)) {
        prev_camera_.SetPriorFocalLength(true);
      } else {
        focal_length = options_.default_focal_length_factor *
                       std::max(bitmap->Width(), bitmap->Height());
        prev_camera_.SetPriorFocalLength(false);
      }

      prev_camera_.InitializeWithId(prev_camera_.ModelId(), focal_length,
                                    prev_camera_.Width(),
                                    prev_camera_.Height());
    }

    if (!prev_camera_.VerifyParams()) {
      return Status::CAMERA_PARAM_ERROR;
    }

    prev_camera_.SetCameraId(database_->WriteCamera(prev_camera_));
  }

  image->SetCameraId(prev_camera_.CameraId());

  //////////////////////////////////////////////////////////////////////////////
  // Extract GPS data.
  //////////////////////////////////////////////////////////////////////////////

  if (!bitmap->ExifLatitude(&image->TvecPrior(0)) ||
      !bitmap->ExifLongitude(&image->TvecPrior(1)) ||
      !bitmap->ExifAltitude(&image->TvecPrior(2))) {
    image->TvecPrior(0) = std::numeric_limits<double>::quiet_NaN();
    image->TvecPrior(1) = std::numeric_limits<double>::quiet_NaN();
    image->TvecPrior(2) = std::numeric_limits<double>::quiet_NaN();
  }

  *camera = prev_camera_;

  return Status::SUCCESS;
}

size_t ImageReader::NextIndex() const { return image_index_; }

size_t ImageReader::NumImages() const { return options_.image_list.size(); }

}  // namespace colmap
