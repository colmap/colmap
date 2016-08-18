// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "base/undistortion.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

#include "base/camera_models.h"
#include "base/projection.h"
#include "base/warp.h"
#include "util/bitmap.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {
namespace {

// Write camera parameters to file.
void WriteCameraParams(const std::string& path, const Camera& camera) {
  std::ofstream file(path.c_str(), std::ios::trunc);
  CHECK(file.is_open());

  std::ostringstream line;

  file << "# MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;

  line << camera.ModelName() << " ";
  line << camera.Width() << " ";
  line << camera.Height() << " ";

  for (const double param : camera.Params()) {
    line << param << " ";
  }

  std::string line_string = line.str();
  line_string = line_string.substr(0, line_string.size() - 1);

  file << line_string << std::endl;

  file.close();
}

// Write projection matrix P = K * [R t] to file and prepend given header.
void WriteProjectionMatrix(const std::string& path, const Camera& camera,
                           const Image& image, const std::string& header) {
  CHECK_EQ(camera.ModelId(), PinholeCameraModel::model_id);

  std::ofstream file(path.c_str(), std::ios::trunc);
  CHECK(file.is_open());

  Eigen::Matrix3d calib_matrix = Eigen::Matrix3d::Identity();
  calib_matrix(0, 0) = camera.FocalLengthX();
  calib_matrix(1, 1) = camera.FocalLengthY();
  calib_matrix(0, 2) = camera.PrincipalPointX();
  calib_matrix(1, 2) = camera.PrincipalPointY();

  const Eigen::Matrix3x4d proj_matrix = calib_matrix * image.ProjectionMatrix();

  if (!header.empty()) {
    file << header << std::endl;
  }

  file << proj_matrix(0, 0) << " ";
  file << proj_matrix(0, 1) << " ";
  file << proj_matrix(0, 2) << " ";
  file << proj_matrix(0, 3) << std::endl;

  file << proj_matrix(1, 0) << " ";
  file << proj_matrix(1, 1) << " ";
  file << proj_matrix(1, 2) << " ";
  file << proj_matrix(1, 3) << std::endl;

  file << proj_matrix(2, 0) << " ";
  file << proj_matrix(2, 1) << " ";
  file << proj_matrix(2, 2) << " ";
  file << proj_matrix(2, 3) << std::endl;

  file.close();
}

}  // namespace

ImageUndistorter::ImageUndistorter(const UndistortCameraOptions& options,
                                   const Reconstruction& reconstruction,
                                   const std::string& image_path,
                                   const std::string& output_path)
    : options_(options),
      image_path_(EnsureTrailingSlash(image_path)),
      output_path_(EnsureTrailingSlash(output_path)),
      reconstruction_(reconstruction) {}

void ImageUndistorter::Run() {
  PrintHeading1("Image undistortion");

  ThreadPool thread_pool;
  std::vector<std::future<void>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    futures.push_back(
        thread_pool.AddTask(&ImageUndistorter::Undistort, this, i));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (IsStopped()) {
      break;
    }

    std::cout << StringPrintf("Undistorting image [%d/%d]", i + 1,
                              futures.size())
              << std::endl;

    futures[i].get();
  }

  GetTimer().PrintMinutes();
}

void ImageUndistorter::Undistort(const size_t reg_image_idx) const {
  const image_t image_id = reconstruction_.RegImageIds().at(reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = reconstruction_.Camera(image.CameraId());

  Bitmap distorted_bitmap;
  const std::string input_image_path = image_path_ + image.Name();
  if (!distorted_bitmap.Read(input_image_path)) {
    std::cerr << "ERROR: Cannot read image at path " << input_image_path
              << std::endl;
    return;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(options_, distorted_bitmap, camera, &undistorted_bitmap,
                 &undistorted_camera);

  const std::string output_image_path = output_path_ + image.Name();
  undistorted_bitmap.Write(output_image_path);

  const std::string camera_params_path = output_image_path + ".camera.txt";
  WriteCameraParams(camera_params_path.c_str(), undistorted_camera);

  const std::string proj_matrix_path = output_image_path + ".proj_matrix.txt";
  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "");
}

PMVSUndistorter::PMVSUndistorter(const UndistortCameraOptions& options,
                                 const Reconstruction& reconstruction,
                                 const std::string& image_path,
                                 const std::string& output_path)
    : options_(options),
      image_path_(EnsureTrailingSlash(image_path)),
      output_path_(EnsureTrailingSlash(output_path)),
      reconstruction_(reconstruction) {}

void PMVSUndistorter::Run() {
  PrintHeading1("Image undistortion (CMVS/PMVS)");

  CreateDirIfNotExists(output_path_ + "pmvs/");
  CreateDirIfNotExists(output_path_ + "pmvs/txt/");
  CreateDirIfNotExists(output_path_ + "pmvs/visualize/");
  CreateDirIfNotExists(output_path_ + "pmvs/models/");

  ThreadPool thread_pool;
  std::vector<std::future<void>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    futures.push_back(
        thread_pool.AddTask(&PMVSUndistorter::Undistort, this, i));
  }

  // Reconstruction with undistorted cameras, exported to bundle file.
  Reconstruction undistorted_reconstruction = reconstruction_;

  for (size_t i = 0; i < futures.size(); ++i) {
    if (IsStopped()) {
      thread_pool.Stop();
      std::cout << "WARNING: Stopped the undistortion process. Image point "
                   "locations and camera parameters for not yet processed "
                   "images in the Bundler output file is probably wrong."
                << std::endl;
      break;
    }

    std::cout << StringPrintf("Undistorting image [%d/%d]", i + 1,
                              futures.size())
              << std::endl;

    futures[i].get();

    // Undistort the camera and the image points. Note that this operation needs
    // to be done sequentially, otherwise we end up with race conditions when
    // modifying the reconstruction in parallel.
    const image_t image_id = reconstruction_.RegImageIds()[i];
    const Image& image = reconstruction_.Image(image_id);
    const Camera& camera = reconstruction_.Camera(image.CameraId());
    Camera& undistorted_camera =
        undistorted_reconstruction.Camera(image.CameraId());
    undistorted_camera = UndistortCamera(options_, camera);
    Image& undistorted_image =
        undistorted_reconstruction.Image(image.ImageId());
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      Point2D& point2D = undistorted_image.Point2D(point2D_idx);
      const Eigen::Vector2d world_point = camera.ImageToWorld(point2D.XY());
      point2D.SetXY(undistorted_camera.WorldToImage(world_point));
    }
  }

  std::cout << "Writing bundle file" << std::endl;
  const std::string bundle_path = output_path_ + "pmvs/bundle.rd.out";

  undistorted_reconstruction.ExportBundler(bundle_path,
                                           bundle_path + ".list.txt");

  std::cout << "Writing visibility file" << std::endl;
  const std::string vis_path = output_path_ + "pmvs/vis.dat";
  WriteVisibilityData(vis_path);

  std::cout << "Writing option file" << std::endl;
  const std::string option_path = output_path_ + "pmvs/option-all";
  WriteOptionFile(option_path);

  GetTimer().PrintMinutes();
}

void PMVSUndistorter::Undistort(const size_t reg_image_idx) const {
  const image_t image_id = reconstruction_.RegImageIds().at(reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = reconstruction_.Camera(image.CameraId());

  Bitmap distorted_bitmap;
  const std::string input_image_path = image_path_ + image.Name();
  if (!distorted_bitmap.Read(input_image_path)) {
    std::cerr << StringPrintf("ERROR: Cannot read image at path %s",
                              input_image_path.c_str())
              << std::endl;
    return;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(options_, distorted_bitmap, camera, &undistorted_bitmap,
                 &undistorted_camera);

  const std::string output_image_path =
      output_path_ + StringPrintf("pmvs/visualize/%08d.jpg", reg_image_idx);
  undistorted_bitmap.Write(output_image_path);

  const std::string proj_matrix_path =
      output_path_ + StringPrintf("pmvs/txt/%08d.txt", reg_image_idx);
  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "CONTOUR");
}

void PMVSUndistorter::WriteVisibilityData(const std::string& path) const {
  std::ofstream file(path.c_str(), std::ios::trunc);
  CHECK(file.is_open());

  file << "VISDATA" << std::endl;
  file << reconstruction_.NumRegImages() << std::endl;

  const std::vector<image_t>& reg_image_ids = reconstruction_.RegImageIds();

  for (size_t i = 0; i < reg_image_ids.size(); ++i) {
    const image_t image_id = reg_image_ids[i];
    const Image& image = reconstruction_.Image(image_id);
    std::unordered_set<image_t> visible_image_ids;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        const Point3D& point3D = reconstruction_.Point3D(point2D.Point3DId());
        for (const TrackElement track_el : point3D.Track().Elements()) {
          if (track_el.image_id != image_id) {
            visible_image_ids.insert(track_el.image_id);
          }
        }
      }
    }

    std::vector<image_t> sorted_visible_image_ids(visible_image_ids.begin(),
                                                  visible_image_ids.end());
    std::sort(sorted_visible_image_ids.begin(), sorted_visible_image_ids.end());

    file << i << " " << visible_image_ids.size();
    for (const image_t visible_image_id : sorted_visible_image_ids) {
      file << " " << visible_image_id;
    }
    file << std::endl;
  }

  file.close();
}

void PMVSUndistorter::WriteOptionFile(const std::string& path) const {
  std::ofstream file(path.c_str(), std::ios::trunc);
  CHECK(file.is_open());

  file << "# Generated by COLMAP - all images, no clustering." << std::endl;

  file << "level 1" << std::endl;
  file << "csize 2" << std::endl;
  file << "threshold 0.7" << std::endl;
  file << "wsize 7" << std::endl;
  file << "minImageNum 3" << std::endl;
  file << "CPU " << std::thread::hardware_concurrency() << std::endl;
  file << "setEdge 0" << std::endl;
  file << "useBound 0" << std::endl;
  file << "useVisData 1" << std::endl;
  file << "sequence -1" << std::endl;
  file << "maxAngle 10" << std::endl;
  file << "quad 2.0" << std::endl;

  file << "timages " << reconstruction_.NumRegImages();
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    file << " " << i;
  }
  file << std::endl;

  file << "oimages 0" << std::endl;

  file.close();
}

CMPMVSUndistorter::CMPMVSUndistorter(const UndistortCameraOptions& options,
                                     const Reconstruction& reconstruction,
                                     const std::string& image_path,
                                     const std::string& output_path)
    : options_(options),
      image_path_(EnsureTrailingSlash(image_path)),
      output_path_(EnsureTrailingSlash(output_path)),
      reconstruction_(reconstruction) {}

void CMPMVSUndistorter::Run() {
  PrintHeading1("Image undistortion (CMP-MVS)");

  ThreadPool thread_pool;
  std::vector<std::future<void>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    futures.push_back(
        thread_pool.AddTask(&CMPMVSUndistorter::Undistort, this, i));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (IsStopped()) {
      break;
    }

    std::cout << StringPrintf("Undistorting image [%d/%d]", i + 1,
                              futures.size())
              << std::endl;

    futures[i].get();
  }

  GetTimer().PrintMinutes();
}

void CMPMVSUndistorter::Undistort(const size_t reg_image_idx) const {
  const image_t image_id = reconstruction_.RegImageIds().at(reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = reconstruction_.Camera(image.CameraId());

  Bitmap distorted_bitmap;
  const std::string input_image_path = image_path_ + image.Name();
  if (!distorted_bitmap.Read(input_image_path)) {
    std::cerr << "ERROR: Cannot read image at path " << input_image_path
              << std::endl;
    return;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(options_, distorted_bitmap, camera, &undistorted_bitmap,
                 &undistorted_camera);

  const std::string output_image_path =
      output_path_ + StringPrintf("%05d.jpg", reg_image_idx + 1);
  undistorted_bitmap.Write(output_image_path);

  const std::string proj_matrix_path =
      output_path_ + StringPrintf("%05d_P.txt", reg_image_idx + 1);
  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "CONTOUR");
}

Camera UndistortCamera(const UndistortCameraOptions& options,
                       const Camera& camera) {
  CHECK_GE(options.blank_pixels, 0);
  CHECK_LE(options.blank_pixels, 1);
  CHECK_GT(options.min_scale, 0.0);
  CHECK_LE(options.min_scale, options.max_scale);
  CHECK_NE(options.max_image_size, 0);

  Camera undistorted_camera;
  undistorted_camera.SetModelId(PinholeCameraModel::model_id);
  undistorted_camera.Params().resize(PinholeCameraModel::num_params);
  undistorted_camera.SetWidth(camera.Width());
  undistorted_camera.SetHeight(camera.Height());

  // Copy focal length parameters.
  const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
  CHECK_LE(focal_length_idxs.size(), 2)
      << "Not more than two focal length parameters supported.";
  if (focal_length_idxs.size() == 1) {
    undistorted_camera.SetFocalLengthX(camera.FocalLength());
    undistorted_camera.SetFocalLengthY(camera.FocalLength());
  } else if (focal_length_idxs.size() == 2) {
    undistorted_camera.SetFocalLengthX(camera.FocalLengthX());
    undistorted_camera.SetFocalLengthY(camera.FocalLengthY());
  }

  // Copy principal point parameters.
  undistorted_camera.SetPrincipalPointX(camera.PrincipalPointX());
  undistorted_camera.SetPrincipalPointY(camera.PrincipalPointY());

  // Determine min, max coordinates along top / bottom image border.

  double left_min_x = std::numeric_limits<double>::max();
  double left_max_x = std::numeric_limits<double>::lowest();
  double right_min_x = std::numeric_limits<double>::max();
  double right_max_x = std::numeric_limits<double>::lowest();

  for (size_t y = 0; y < camera.Height(); ++y) {
    // Left border.
    const Eigen::Vector2d world_point1 =
        camera.ImageToWorld(Eigen::Vector2d(0.5, y + 0.5));
    const Eigen::Vector2d undistorted_point1 =
        undistorted_camera.WorldToImage(world_point1);
    left_min_x = std::min(left_min_x, undistorted_point1(0));
    left_max_x = std::max(left_max_x, undistorted_point1(0));
    // Right border.
    const Eigen::Vector2d world_point2 =
        camera.ImageToWorld(Eigen::Vector2d(camera.Width() - 0.5, y + 0.5));
    const Eigen::Vector2d undistorted_point2 =
        undistorted_camera.WorldToImage(world_point2);
    right_min_x = std::min(right_min_x, undistorted_point2(0));
    right_max_x = std::max(right_max_x, undistorted_point2(0));
  }

  // Determine min, max coordinates along left / right image border.

  double top_min_y = std::numeric_limits<double>::max();
  double top_max_y = std::numeric_limits<double>::lowest();
  double bottom_min_y = std::numeric_limits<double>::max();
  double bottom_max_y = std::numeric_limits<double>::lowest();

  for (size_t x = 0; x < camera.Width(); ++x) {
    // Top border.
    const Eigen::Vector2d world_point1 =
        camera.ImageToWorld(Eigen::Vector2d(x + 0.5, 0.5));
    const Eigen::Vector2d undistorted_point1 =
        undistorted_camera.WorldToImage(world_point1);
    top_min_y = std::min(top_min_y, undistorted_point1(1));
    top_max_y = std::max(top_max_y, undistorted_point1(1));
    // Bottom border.
    const Eigen::Vector2d world_point2 =
        camera.ImageToWorld(Eigen::Vector2d(x + 0.5, camera.Height() - 0.5));
    const Eigen::Vector2d undistorted_point2 =
        undistorted_camera.WorldToImage(world_point2);
    bottom_min_y = std::min(bottom_min_y, undistorted_point2(1));
    bottom_max_y = std::max(bottom_max_y, undistorted_point2(1));
  }

  const double cx = undistorted_camera.PrincipalPointX();
  const double cy = undistorted_camera.PrincipalPointY();

  // Scale such that undistorted image contains all pixels of distorted image
  const double min_scale_x = std::min(
      cx / (cx - left_min_x), (camera.Width() - 0.5 - cx) / (right_max_x - cx));
  const double min_scale_y =
      std::min(cy / (cy - top_min_y),
               (camera.Height() - 0.5 - cy) / (bottom_max_y - cy));

  // Scale such that there are no blank pixels in undistorted image
  const double max_scale_x = std::max(
      cx / (cx - left_max_x), (camera.Width() - 0.5 - cx) / (right_min_x - cx));
  const double max_scale_y =
      std::max(cy / (cy - top_max_y),
               (camera.Height() - 0.5 - cy) / (bottom_min_y - cy));

  // Interpolate scale according to blank_pixels.
  double scale_x = 1.0 / (min_scale_x * options.blank_pixels +
                          max_scale_x * (1.0 - options.blank_pixels));
  double scale_y = 1.0 / (min_scale_y * options.blank_pixels +
                          max_scale_y * (1.0 - options.blank_pixels));

  // Clip the scaling factors.
  scale_x = Clip(scale_x, options.min_scale, options.max_scale);
  scale_y = Clip(scale_y, options.min_scale, options.max_scale);

  // Scale undistorted camera dimensions.
  undistorted_camera.SetWidth(
      static_cast<size_t>(std::max(1.0, scale_x * undistorted_camera.Width())));
  undistorted_camera.SetHeight(static_cast<size_t>(
      std::max(1.0, scale_y * undistorted_camera.Height())));

  // Scale the principal point according to the new dimensions of the image.
  undistorted_camera.SetPrincipalPointX(
      undistorted_camera.PrincipalPointX() *
      static_cast<double>(undistorted_camera.Width()) / camera.Width());
  undistorted_camera.SetPrincipalPointY(
      undistorted_camera.PrincipalPointY() *
      static_cast<double>(undistorted_camera.Height()) / camera.Height());

  if (options.max_image_size > 0) {
    const double max_image_scale_x =
        options.max_image_size /
        static_cast<double>(undistorted_camera.Width());
    const double max_image_scale_y =
        options.max_image_size /
        static_cast<double>(undistorted_camera.Height());
    const double max_image_scale =
        std::min(max_image_scale_x, max_image_scale_y);
    if (max_image_scale < 1.0) {
      undistorted_camera.Rescale(max_image_scale);
    }
  }

  return undistorted_camera;
}

void UndistortImage(const UndistortCameraOptions& options,
                    const Bitmap& distorted_bitmap,
                    const Camera& distorted_camera, Bitmap* undistorted_bitmap,
                    Camera* undistorted_camera) {
  CHECK_EQ(distorted_camera.Width(), distorted_bitmap.Width());
  CHECK_EQ(distorted_camera.Height(), distorted_bitmap.Height());

  *undistorted_camera = UndistortCamera(options, distorted_camera);
  undistorted_bitmap->Allocate(static_cast<int>(undistorted_camera->Width()),
                               static_cast<int>(undistorted_camera->Height()),
                               distorted_bitmap.IsRGB());
  distorted_bitmap.CloneMetadata(undistorted_bitmap);

  WarpImageBetweenCameras(distorted_camera, *undistorted_camera,
                          distorted_bitmap, undistorted_bitmap);
}

void UndistortReconstruction(const UndistortCameraOptions& options,
                             Reconstruction* reconstruction) {
  const auto distorted_cameras = reconstruction->Cameras();
  for (auto& camera : distorted_cameras) {
    reconstruction->Camera(camera.first) =
        UndistortCamera(options, camera.second);
  }

  for (const auto& distorted_image : reconstruction->Images()) {
    auto& image = reconstruction->Image(distorted_image.first);
    const auto& distorted_camera = distorted_cameras.at(image.CameraId());
    const auto& undistorted_camera = reconstruction->Camera(image.CameraId());
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      auto& point2D = image.Point2D(point2D_idx);
      point2D.SetXY(undistorted_camera.WorldToImage(
          distorted_camera.ImageToWorld(point2D.XY())));
    }
  }
}

}  // namespace colmap
