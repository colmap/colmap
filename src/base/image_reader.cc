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

bool ImageReader::Options::Check() const {
  CHECK_OPTION_GT(default_focal_length_factor, 0.0);
  const int model_id = CameraModelNameToId(camera_model);
  CHECK_OPTION_NE(model_id, -1);
  if (!camera_params.empty()) {
    CHECK_OPTION(
        CameraModelVerifyParams(model_id, CSVToVector<double>(camera_params)));
  }
  return true;
}

ImageReader::ImageReader(const Options& options)
    : options_(options), image_index_(0) {
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

bool ImageReader::Next(Image* image, Bitmap* bitmap) {
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(bitmap);

  image_index_ += 1;
  if (image_index_ > options_.image_list.size()) {
    return false;
  }

  const std::string image_path = options_.image_list.at(image_index_ - 1);

  Database database(options_.database_path);

  //////////////////////////////////////////////////////////////////////////////
  // Set the image name.
  //////////////////////////////////////////////////////////////////////////////

  image->SetName(image_path);
  image->SetName(StringReplace(image->Name(), "\\", "/"));
  image->SetName(
      image->Name().substr(options_.image_path.size(),
                           image->Name().size() - options_.image_path.size()));

  std::cout << "  Name:           " << image->Name() << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Check if image already read.
  //////////////////////////////////////////////////////////////////////////////

  const bool exists_image = database.ExistsImageWithName(image->Name());

  if (exists_image) {
    const DatabaseTransaction database_transaction(&database);
    *image = database.ReadImageWithName(image->Name());
    const bool exists_keypoints = database.ExistsKeypoints(image->ImageId());
    const bool exists_descriptors =
        database.ExistsDescriptors(image->ImageId());

    if (exists_keypoints && exists_descriptors) {
      std::cout << "  SKIP: Features already extracted." << std::endl;
      return false;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Read image.
  //////////////////////////////////////////////////////////////////////////////

  if (!bitmap->Read(image_path, false)) {
    std::cout << "  SKIP: Cannot read image at path " << image_path
              << std::endl;
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Check for well-formed data.
  //////////////////////////////////////////////////////////////////////////////

  if (exists_image) {
    const Camera camera = database.ReadCamera(image->CameraId());

    if (options_.single_camera && prev_camera_.CameraId() != kInvalidCameraId &&
        (camera.Width() != prev_camera_.Width() ||
         camera.Height() != prev_camera_.Height())) {
      std::cerr << "  ERROR: Single camera specified, but images have "
                   "different dimensions."
                << std::endl;
      return false;
    }

    if (static_cast<size_t>(bitmap->Width()) != camera.Width() ||
        static_cast<size_t>(bitmap->Height()) != camera.Height()) {
      std::cerr << "  ERROR: Image previously processed, but current version "
                   "has different dimensions."
                << std::endl;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Extract image dimensions.
  //////////////////////////////////////////////////////////////////////////////

  if (options_.single_camera && prev_camera_.CameraId() != kInvalidCameraId &&
      (prev_camera_.Width() != static_cast<size_t>(bitmap->Width()) ||
       prev_camera_.Height() != static_cast<size_t>(bitmap->Height()))) {
    std::cerr << "  ERROR: Single camera specified, but images have "
                 "different dimensions."
              << std::endl;
    return false;
  }

  prev_camera_.SetWidth(static_cast<size_t>(bitmap->Width()));
  prev_camera_.SetHeight(static_cast<size_t>(bitmap->Height()));

  std::cout << "  Width:          " << prev_camera_.Width() << "px"
            << std::endl;
  std::cout << "  Height:         " << prev_camera_.Height() << "px"
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Extract camera model and focal length
  //////////////////////////////////////////////////////////////////////////////

  if (!options_.single_camera || prev_camera_.CameraId() == kInvalidCameraId) {
    if (options_.camera_params.empty()) {
      // Extract focal length.
      double focal_length = 0.0;
      if (bitmap->ExifFocalLength(&focal_length)) {
        prev_camera_.SetPriorFocalLength(true);
        std::cout << "  Focal length:   " << focal_length << "px (EXIF)"
                  << std::endl;
      } else {
        focal_length = options_.default_focal_length_factor *
                       std::max(bitmap->Width(), bitmap->Height());
        prev_camera_.SetPriorFocalLength(false);
        std::cout << "  Focal length:   " << focal_length << "px" << std::endl;
      }

      prev_camera_.InitializeWithId(prev_camera_.ModelId(), focal_length,
                                    prev_camera_.Width(),
                                    prev_camera_.Height());
    }

    if (!prev_camera_.VerifyParams()) {
      std::cerr << "  ERROR: Invalid camera parameters." << std::endl;
      return false;
    }

    prev_camera_.SetCameraId(database.WriteCamera(prev_camera_));
  }

  image->SetCameraId(prev_camera_.CameraId());

  std::cout << "  Camera ID:      " << prev_camera_.CameraId() << std::endl;
  std::cout << "  Camera Model:   " << prev_camera_.ModelName() << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Extract GPS data.
  //////////////////////////////////////////////////////////////////////////////

  if (bitmap->ExifLatitude(&image->TvecPrior(0)) &&
      bitmap->ExifLongitude(&image->TvecPrior(1)) &&
      bitmap->ExifAltitude(&image->TvecPrior(2))) {
    std::cout << StringPrintf("  EXIF GPS:       LAT=%.3f, LON=%.3f, ALT=%.3f",
                              image->TvecPrior(0), image->TvecPrior(1),
                              image->TvecPrior(2))
              << std::endl;
  } else {
    image->TvecPrior(0) = std::numeric_limits<double>::quiet_NaN();
    image->TvecPrior(1) = std::numeric_limits<double>::quiet_NaN();
    image->TvecPrior(2) = std::numeric_limits<double>::quiet_NaN();
  }

  return true;
}

size_t ImageReader::NextIndex() const { return image_index_; }

size_t ImageReader::NumImages() const { return options_.image_list.size(); }

}  // namespace colmap
