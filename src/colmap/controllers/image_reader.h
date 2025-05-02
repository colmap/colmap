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

#pragma once

#include "colmap/geometry/gps.h"
#include "colmap/scene/database.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/threading.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap {

struct ImageReaderOptions {
  // Path to database in which to store the extracted data.
  std::string database_path;

  // Root path to folder which contains the images.
  std::string image_path;

  // Optional root path to folder which contains image masks. For a given image,
  // the corresponding mask must have the same sub-path below this root as the
  // image has below image_path. The filename must be equal, aside from the
  // added extension .png. For example, for an image image_path/abc/012.jpg, the
  // mask would be mask_path/abc/012.jpg.png. No features will be extracted in
  // regions where the mask image is black (pixel intensity value 0 in
  // grayscale).
  std::string mask_path;

  // Optional path to an image file specifying a mask for all images. No
  // features will be extracted in regions where the mask is black (pixel
  // intensity value 0 in grayscale).
  std::string camera_mask_path;

  // Optional list of images to read. The list must contain the relative path
  // of the images with respect to the image_path.
  std::vector<std::string> image_names;
  // Name of the camera model.
  std::string camera_model = "SIMPLE_RADIAL";

  // Manual specification of camera parameters. If empty, camera parameters
  // will be extracted from EXIF, i.e. principal point and focal length.
  std::string camera_params;

  // Whether to use the same camera for all images.
  bool single_camera = false;

  // Whether to use the same camera for all images in the same sub-folder.
  bool single_camera_per_folder = false;
  // Whether to use a different camera for each image.
  bool single_camera_per_image = false;
  // Whether to explicitly use an existing camera for all images. Note that in
  // this case the specified camera model and parameters are ignored.
  int existing_camera_id = kInvalidCameraId;

  // If camera parameters are not specified manually and the image does not
  // have focal length EXIF information, the focal length is set to the
  // value `default_focal_length_factor * max(width, height)`.
  double default_focal_length_factor = 1.2;

  bool Check() const;
};

// Recursively iterate over the images in a directory. Skips an image if it
// already exists in the database. Extracts the camera intrinsics from EXIF and
// writes the camera information to the database.
class ImageReader {
 public:
  enum class Status {
    FAILURE,
    SUCCESS,
    IMAGE_EXISTS,
    BITMAP_ERROR,
    MASK_ERROR,
    CAMERA_SINGLE_DIM_ERROR,
    CAMERA_EXIST_DIM_ERROR,
    CAMERA_PARAM_ERROR
  };

  ImageReader(const ImageReaderOptions& options, Database* database);

  Status Next(Rig* rig,
              Camera* camera,
              Image* image,
              PosePrior* pose_prior,
              Bitmap* bitmap,
              Bitmap* mask);
  size_t NextIndex() const;
  size_t NumImages() const;

  static std::string StatusToString(Status status);

 private:
  // Image reader options.
  ImageReaderOptions options_;
  Database* database_;
  // Index of previously processed image.
  size_t image_index_;
  // Previously processed rig/camera.
  Rig prev_rig_;
  Camera prev_camera_;
  std::unordered_map<std::string, camera_t> camera_model_to_id_;
  // Names of image sub-folders.
  std::string prev_image_folder_;
  std::unordered_set<std::string> image_folders_;
};

}  // namespace colmap
