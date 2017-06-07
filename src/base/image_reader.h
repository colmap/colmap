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

#ifndef COLMAP_SRC_BASE_IMAGE_READER_H_
#define COLMAP_SRC_BASE_IMAGE_READER_H_

#include "base/database.h"
#include "util/bitmap.h"
#include "util/threading.h"

namespace colmap {

// Recursively iterate over the images in a directory. Skips an image if it
// already exists in the database. Extracts the camera intrinsics from EXIF and
// writes the camera information to the database.
class ImageReader {
 public:
  struct Options {
    // Path to database in which to store the extracted data.
    std::string database_path = "";

    // Root path to folder which contains the image.
    std::string image_path = "";

    // Optional list of images to read. The list must contain the relative path
    // of the images with respect to the image_path.
    std::vector<std::string> image_list;

    // Name of the camera model.
    std::string camera_model = "SIMPLE_RADIAL";

    // Whether to use the same camera for all images.
    bool single_camera = false;

    // Specification of manual camera parameters. If empty, camera parameters
    // will be extracted from EXIF, i.e. principal point and focal length.
    std::string camera_params = "";

    // If camera parameters are not specified manually and the image does not
    // have focal length EXIF information, the focal length is set to the
    // value `default_focal_length_factor * max(width, height)`.
    double default_focal_length_factor = 1.2;

    bool Check() const;
  };

  explicit ImageReader(const Options& options);

  bool Next(Image* image, Bitmap* bitmap);
  size_t NextIndex() const;
  size_t NumImages() const;

 private:
  // Image reader options.
  Options options_;
  // Index of previously processed image.
  size_t image_index_;
  // Previously processed camera.
  Camera prev_camera_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_IMAGE_READER_H_
