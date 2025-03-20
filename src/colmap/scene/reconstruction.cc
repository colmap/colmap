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

#include "colmap/scene/reconstruction.h"

#include "colmap/geometry/gps.h"
#include "colmap/geometry/normalization.h"
#include "colmap/geometry/pose.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/reconstruction_io.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"

namespace colmap {

Reconstruction::Reconstruction() : max_point3D_id_(0) {}

Reconstruction::Reconstruction(const Reconstruction& other)
    : cameras_(other.cameras_),
      images_(other.images_),
      points3D_(other.points3D_),
      reg_image_ids_(other.reg_image_ids_),
      max_point3D_id_(other.max_point3D_id_) {
  for (auto& [_, image] : images_) {
    image.ResetCameraPtr();
    image.SetCameraPtr(&Camera(image.CameraId()));
  }
}

Reconstruction& Reconstruction::operator=(const Reconstruction& other) {
  if (this != &other) {
    cameras_ = other.cameras_;
    images_ = other.images_;
    points3D_ = other.points3D_;
    reg_image_ids_ = other.reg_image_ids_;
    max_point3D_id_ = other.max_point3D_id_;
    for (auto& [_, image] : images_) {
      image.ResetCameraPtr();
      image.SetCameraPtr(&Camera(image.CameraId()));
    }
  }
  return *this;
}

std::unordered_set<point3D_t> Reconstruction::Point3DIds() const {
  std::unordered_set<point3D_t> point3D_ids;
  point3D_ids.reserve(points3D_.size());

  for (const auto& point3D : points3D_) {
    point3D_ids.insert(point3D.first);
  }

  return point3D_ids;
}

void Reconstruction::Load(const DatabaseCache& database_cache) {
  // Add cameras.
  cameras_.reserve(database_cache.NumCameras());
  for (const auto& camera : database_cache.Cameras()) {
    if (!ExistsCamera(camera.first)) {
      AddCamera(camera.second);
    }
    // Else: camera was added before, e.g. with `ReadAllCameras`.
  }

  // Add images.
  images_.reserve(database_cache.NumImages());

  for (const auto& image : database_cache.Images()) {
    if (ExistsImage(image.second.ImageId())) {
      class Image& existing_image = Image(image.second.ImageId());
      THROW_CHECK_EQ(existing_image.Name(), image.second.Name());
      if (existing_image.NumPoints2D() == 0) {
        existing_image.SetPoints2D(image.second.Points2D());
      } else {
        THROW_CHECK_EQ(image.second.NumPoints2D(),
                       existing_image.NumPoints2D());
      }
    } else {
      AddImage(image.second);
    }
  }
}

void Reconstruction::TearDown() {
  // Remove all not yet registered images.
  std::unordered_set<camera_t> keep_camera_ids;
  for (auto it = images_.begin(); it != images_.end();) {
    if (IsImageRegistered(it->first)) {
      keep_camera_ids.insert(it->second.CameraId());
      ++it;
    } else {
      it = images_.erase(it);
    }
  }

  // Remove all unused cameras.
  for (auto it = cameras_.begin(); it != cameras_.end();) {
    if (keep_camera_ids.count(it->first) == 0) {
      it = cameras_.erase(it);
    } else {
      ++it;
    }
  }

  // Compress tracks.
  for (auto& point3D : points3D_) {
    point3D.second.track.Compress();
  }
}

void Reconstruction::AddCamera(struct Camera camera) {
  const camera_t camera_id = camera.camera_id;
  THROW_CHECK(camera.VerifyParams());
  THROW_CHECK(cameras_.emplace(camera_id, std::move(camera)).second);
}

void Reconstruction::AddImage(class Image image) {
  THROW_CHECK(image.HasCameraId());
  auto& camera = Camera(image.CameraId());
  if (image.HasCameraPtr()) {
    THROW_CHECK_EQ(image.CameraPtr(), &camera);
  } else {
    image.SetCameraPtr(&camera);
  }
  const image_t image_id = image.ImageId();
  const bool is_registered = image.HasPose();
  THROW_CHECK(images_.emplace(image_id, std::move(image)).second);
  if (is_registered) {
    THROW_CHECK_NE(image_id, kInvalidImageId);
    RegisterImage(image_id);
  }
}

void Reconstruction::AddPoint3D(const point3D_t point3D_id,
                                struct Point3D point3D) {
  max_point3D_id_ = std::max(max_point3D_id_, point3D_id);
  THROW_CHECK(points3D_.emplace(point3D_id, std::move(point3D)).second);
}

point3D_t Reconstruction::AddPoint3D(const Eigen::Vector3d& xyz,
                                     Track track,
                                     const Eigen::Vector3ub& color) {
  const point3D_t point3D_id = ++max_point3D_id_;
  THROW_CHECK(!ExistsPoint3D(point3D_id));

  for (const auto& track_el : track.Elements()) {
    class Image& image = Image(track_el.image_id);
    THROW_CHECK(!image.Point2D(track_el.point2D_idx).HasPoint3D());
    image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
    THROW_CHECK_LE(image.NumPoints3D(), image.NumPoints2D());
  }

  struct Point3D& point3D = points3D_[point3D_id];
  point3D.xyz = xyz;
  point3D.track = std::move(track);
  point3D.color = color;

  return point3D_id;
}

void Reconstruction::AddObservation(const point3D_t point3D_id,
                                    const TrackElement& track_el) {
  class Image& image = Image(track_el.image_id);
  THROW_CHECK(!image.Point2D(track_el.point2D_idx).HasPoint3D());

  image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
  THROW_CHECK_LE(image.NumPoints3D(), image.NumPoints2D());

  struct Point3D& point3D = Point3D(point3D_id);
  point3D.track.AddElement(track_el);
}

point3D_t Reconstruction::MergePoints3D(const point3D_t point3D_id1,
                                        const point3D_t point3D_id2) {
  const struct Point3D& point3D1 = Point3D(point3D_id1);
  const struct Point3D& point3D2 = Point3D(point3D_id2);

  const Eigen::Vector3d merged_xyz =
      (point3D1.track.Length() * point3D1.xyz +
       point3D2.track.Length() * point3D2.xyz) /
      (point3D1.track.Length() + point3D2.track.Length());
  const Eigen::Vector3d merged_rgb =
      (point3D1.track.Length() * point3D1.color.cast<double>() +
       point3D2.track.Length() * point3D2.color.cast<double>()) /
      (point3D1.track.Length() + point3D2.track.Length());

  Track merged_track;
  merged_track.Reserve(point3D1.track.Length() + point3D2.track.Length());
  merged_track.AddElements(point3D1.track.Elements());
  merged_track.AddElements(point3D2.track.Elements());

  DeletePoint3D(point3D_id1);
  DeletePoint3D(point3D_id2);

  const point3D_t merged_point3D_id =
      AddPoint3D(merged_xyz, merged_track, merged_rgb.cast<uint8_t>());

  return merged_point3D_id;
}

void Reconstruction::DeletePoint3D(const point3D_t point3D_id) {
  // Note: Do not change order of these instructions.

  const class Track& track = Point3D(point3D_id).track;
  for (const auto& track_el : track.Elements()) {
    class Image& image = Image(track_el.image_id);
    image.ResetPoint3DForPoint2D(track_el.point2D_idx);
  }

  points3D_.erase(point3D_id);
}

void Reconstruction::DeleteObservation(const image_t image_id,
                                       const point2D_t point2D_idx) {
  // Note: Do not change order of these instructions.

  class Image& image = Image(image_id);
  const point3D_t point3D_id = image.Point2D(point2D_idx).point3D_id;
  struct Point3D& point3D = Point3D(point3D_id);

  if (point3D.track.Length() <= 2) {
    DeletePoint3D(point3D_id);
    return;
  }

  point3D.track.DeleteElement(image_id, point2D_idx);

  image.ResetPoint3DForPoint2D(point2D_idx);
}

void Reconstruction::DeleteAllPoints2DAndPoints3D() {
  points3D_.clear();
  for (auto& image : images_) {
    class Image new_image;
    new_image.SetImageId(image.second.ImageId());
    new_image.SetName(image.second.Name());
    new_image.SetCameraId(image.second.CameraId());
    new_image.SetCameraPtr(image.second.CameraPtr());
    new_image.SetCamFromWorld(image.second.MaybeCamFromWorld());
    image.second = new_image;
  }
}

void Reconstruction::RegisterImage(const image_t image_id) {
  THROW_CHECK(Image(image_id).HasPose());
  reg_image_ids_.insert(image_id);
}

void Reconstruction::DeRegisterImage(const image_t image_id) {
  class Image& image = Image(image_id);

  const auto num_points2D = image.NumPoints2D();
  for (point2D_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
    if (image.Point2D(point2D_idx).HasPoint3D()) {
      DeleteObservation(image_id, point2D_idx);
    }
  }

  image.ResetPose();
  reg_image_ids_.erase(image_id);
}

Sim3d Reconstruction::Normalize(const bool fixed_scale,
                                const double extent,
                                const double min_percentile,
                                const double max_percentile,
                                const bool use_images) {
  THROW_CHECK_GT(extent, 0);

  if ((use_images && NumRegImages() < 2) ||
      (!use_images && points3D_.size() < 2)) {
    return Sim3d();
  }

  const auto [bbox, centroid] =
      ComputeBBBoxAndCentroid(min_percentile, max_percentile, use_images);

  // Calculate scale and translation, such that
  // translation is applied before scaling.
  double scale = 1.;
  if (!fixed_scale) {
    const double old_extent = bbox.diagonal().norm();
    if (old_extent >= std::numeric_limits<double>::epsilon()) {
      scale = extent / old_extent;
    }
  }

  Sim3d tform(scale, Eigen::Quaterniond::Identity(), -scale * centroid);
  Transform(tform);

  return tform;
}

Eigen::Vector3d Reconstruction::ComputeCentroid(const double min_percentile,
                                                const double max_percentile,
                                                bool use_images) const {
  return ComputeBBBoxAndCentroid(min_percentile, max_percentile, use_images)
      .second;
}

Eigen::AlignedBox3d Reconstruction::ComputeBoundingBox(
    const double min_percentile,
    const double max_percentile,
    bool use_images) const {
  return ComputeBBBoxAndCentroid(min_percentile, max_percentile, use_images)
      .first;
}

std::pair<Eigen::AlignedBox3d, Eigen::Vector3d>
Reconstruction::ComputeBBBoxAndCentroid(const double min_percentile,
                                        const double max_percentile,
                                        const bool use_images) const {
  const size_t num_elements = use_images ? NumRegImages() : points3D_.size();
  if (num_elements == 0) {
    return std::make_pair(
        Eigen::AlignedBox3d(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)),
        Eigen::Vector3d(0, 0, 0));
  }

  // Coordinates of image centers or point locations.
  std::vector<double> coords_x;
  std::vector<double> coords_y;
  std::vector<double> coords_z;
  coords_x.reserve(num_elements);
  coords_y.reserve(num_elements);
  coords_z.reserve(num_elements);
  if (use_images) {
    for (const image_t image_id : RegImageIds()) {
      const Eigen::Vector3d proj_center = Image(image_id).ProjectionCenter();
      coords_x.push_back(proj_center(0));
      coords_y.push_back(proj_center(1));
      coords_z.push_back(proj_center(2));
    }
  } else {
    for (const auto& point3D : points3D_) {
      coords_x.push_back(point3D.second.xyz(0));
      coords_y.push_back(point3D.second.xyz(1));
      coords_z.push_back(point3D.second.xyz(2));
    }
  }

  return ComputeBoundingBoxAndCentroid(min_percentile,
                                       max_percentile,
                                       std::move(coords_x),
                                       std::move(coords_y),
                                       std::move(coords_z));
}

void Reconstruction::Transform(const Sim3d& new_from_old_world) {
  for (auto& [_, image] : images_) {
    if (image.HasPose()) {
      image.SetCamFromWorld(
          TransformCameraWorld(new_from_old_world, image.CamFromWorld()));
    }
  }
  for (auto& point3D : points3D_) {
    point3D.second.xyz = new_from_old_world * point3D.second.xyz;
  }
}

Reconstruction Reconstruction::Crop(const Eigen::AlignedBox3d& bbox) const {
  Reconstruction cropped_reconstruction;
  for (const auto& camera : cameras_) {
    cropped_reconstruction.AddCamera(camera.second);
  }
  for (const auto& image : images_) {
    auto new_image = image.second;
    new_image.ResetCameraPtr();
    const auto num_points2D = new_image.NumPoints2D();
    for (point2D_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
      new_image.ResetPoint3DForPoint2D(point2D_idx);
    }
    cropped_reconstruction.AddImage(new_image);
  }
  std::unordered_set<image_t> registered_image_ids;
  for (const auto& point3D : points3D_) {
    if (bbox.contains(point3D.second.xyz)) {
      for (const auto& track_el : point3D.second.track.Elements()) {
        registered_image_ids.insert(track_el.image_id);
      }
      cropped_reconstruction.AddPoint3D(
          point3D.second.xyz, point3D.second.track, point3D.second.color);
    }
  }
  for (const auto& [image_id, _] : cropped_reconstruction.Images()) {
    if (registered_image_ids.count(image_id) == 0) {
      cropped_reconstruction.DeRegisterImage(image_id);
    }
  }
  return cropped_reconstruction;
}

const class Image* Reconstruction::FindImageWithName(
    const std::string& name) const {
  for (const auto& image : images_) {
    if (image.second.Name() == name) {
      return &image.second;
    }
  }
  return nullptr;
}

std::vector<std::pair<image_t, image_t>> Reconstruction::FindCommonRegImageIds(
    const Reconstruction& other) const {
  std::vector<std::pair<image_t, image_t>> common_reg_image_ids;
  for (const auto image_id : RegImageIds()) {
    const auto& image = Image(image_id);
    const auto* other_image = other.FindImageWithName(image.Name());
    if (other_image != nullptr &&
        other.IsImageRegistered(other_image->ImageId())) {
      common_reg_image_ids.emplace_back(image_id, other_image->ImageId());
    }
  }
  return common_reg_image_ids;
}

void Reconstruction::TranscribeImageIdsToDatabase(const Database& database) {
  std::unordered_map<image_t, image_t> old_to_new_image_ids;
  old_to_new_image_ids.reserve(NumImages());

  std::unordered_map<image_t, class Image> new_images;
  new_images.reserve(NumImages());

  for (auto& image : images_) {
    if (!database.ExistsImageWithName(image.second.Name())) {
      LOG(FATAL_THROW) << "Image with name " << image.second.Name()
                       << " does not exist in database";
    }

    const auto database_image = database.ReadImageWithName(image.second.Name());
    old_to_new_image_ids.emplace(image.second.ImageId(),
                                 database_image.ImageId());
    image.second.SetImageId(database_image.ImageId());
    new_images.emplace(database_image.ImageId(), image.second);
  }

  images_ = std::move(new_images);

  std::set<image_t> new_reg_image_ids;
  for (const image_t image_id : RegImageIds()) {
    new_reg_image_ids.insert(old_to_new_image_ids.at(image_id));
  }
  reg_image_ids_ = new_reg_image_ids;

  for (auto& point3D : points3D_) {
    for (auto& track_el : point3D.second.track.Elements()) {
      track_el.image_id = old_to_new_image_ids.at(track_el.image_id);
    }
  }
}

size_t Reconstruction::ComputeNumObservations() const {
  size_t num_obs = 0;
  for (const image_t image_id : RegImageIds()) {
    num_obs += Image(image_id).NumPoints3D();
  }
  return num_obs;
}

double Reconstruction::ComputeMeanTrackLength() const {
  if (points3D_.empty()) {
    return 0.0;
  } else {
    return ComputeNumObservations() / static_cast<double>(points3D_.size());
  }
}

double Reconstruction::ComputeMeanObservationsPerRegImage() const {
  if (NumRegImages() == 0) {
    return 0.0;
  } else {
    return ComputeNumObservations() / static_cast<double>(NumRegImages());
  }
}

double Reconstruction::ComputeMeanReprojectionError() const {
  double error_sum = 0.0;
  size_t num_valid_errors = 0;
  for (const auto& point3D : points3D_) {
    if (point3D.second.HasError()) {
      error_sum += point3D.second.error;
      num_valid_errors += 1;
    }
  }

  if (num_valid_errors == 0) {
    return 0.0;
  } else {
    return error_sum / num_valid_errors;
  }
}

void Reconstruction::UpdatePoint3DErrors() {
  for (auto& point3D : points3D_) {
    if (point3D.second.track.Length() == 0) {
      point3D.second.error = 0;
      continue;
    }
    point3D.second.error = 0;
    for (const auto& track_el : point3D.second.track.Elements()) {
      const auto& image = Image(track_el.image_id);
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const auto& camera = *image.CameraPtr();
      point3D.second.error += std::sqrt(CalculateSquaredReprojectionError(
          point2D.xy, point3D.second.xyz, image.CamFromWorld(), camera));
    }
    point3D.second.error /= point3D.second.track.Length();
  }
}

void Reconstruction::Read(const std::string& path) {
  if (ExistsFile(JoinPaths(path, "cameras.bin")) &&
      ExistsFile(JoinPaths(path, "images.bin")) &&
      ExistsFile(JoinPaths(path, "points3D.bin"))) {
    ReadBinary(path);
  } else if (ExistsFile(JoinPaths(path, "cameras.txt")) &&
             ExistsFile(JoinPaths(path, "images.txt")) &&
             ExistsFile(JoinPaths(path, "points3D.txt"))) {
    ReadText(path);
  } else {
    LOG(FATAL_THROW) << "cameras, images, points3D files do not exist at "
                     << path;
  }
}

void Reconstruction::Write(const std::string& path) const { WriteBinary(path); }

void Reconstruction::ReadText(const std::string& path) {
  cameras_.clear();
  images_.clear();
  points3D_.clear();
  ReadCamerasText(*this, JoinPaths(path, "cameras.txt"));
  ReadImagesText(*this, JoinPaths(path, "images.txt"));
  ReadPoints3DText(*this, JoinPaths(path, "points3D.txt"));
}

void Reconstruction::ReadBinary(const std::string& path) {
  cameras_.clear();
  images_.clear();
  points3D_.clear();
  ReadCamerasBinary(*this, JoinPaths(path, "cameras.bin"));
  ReadImagesBinary(*this, JoinPaths(path, "images.bin"));
  ReadPoints3DBinary(*this, JoinPaths(path, "points3D.bin"));
}

void Reconstruction::WriteText(const std::string& path) const {
  THROW_CHECK_DIR_EXISTS(path);
  WriteCamerasText(*this, JoinPaths(path, "cameras.txt"));
  WriteImagesText(*this, JoinPaths(path, "images.txt"));
  WritePoints3DText(*this, JoinPaths(path, "points3D.txt"));
}

void Reconstruction::WriteBinary(const std::string& path) const {
  THROW_CHECK_DIR_EXISTS(path);
  WriteCamerasBinary(*this, JoinPaths(path, "cameras.bin"));
  WriteImagesBinary(*this, JoinPaths(path, "images.bin"));
  WritePoints3DBinary(*this, JoinPaths(path, "points3D.bin"));
}

std::vector<PlyPoint> Reconstruction::ConvertToPLY() const {
  std::vector<PlyPoint> ply_points;
  ply_points.reserve(points3D_.size());

  for (const auto& point3D : points3D_) {
    PlyPoint ply_point;
    ply_point.x = point3D.second.xyz(0);
    ply_point.y = point3D.second.xyz(1);
    ply_point.z = point3D.second.xyz(2);
    ply_point.r = point3D.second.color(0);
    ply_point.g = point3D.second.color(1);
    ply_point.b = point3D.second.color(2);
    ply_points.push_back(ply_point);
  }

  return ply_points;
}

void Reconstruction::ImportPLY(const std::string& path) {
  points3D_.clear();

  const auto ply_points = ReadPly(path);

  points3D_.reserve(ply_points.size());

  for (const auto& ply_point : ply_points) {
    AddPoint3D(Eigen::Vector3d(ply_point.x, ply_point.y, ply_point.z),
               Track(),
               Eigen::Vector3ub(ply_point.r, ply_point.g, ply_point.b));
  }
}

void Reconstruction::ImportPLY(const std::vector<PlyPoint>& ply_points) {
  points3D_.clear();
  points3D_.reserve(ply_points.size());
  for (const auto& ply_point : ply_points) {
    AddPoint3D(Eigen::Vector3d(ply_point.x, ply_point.y, ply_point.z),
               Track(),
               Eigen::Vector3ub(ply_point.r, ply_point.g, ply_point.b));
  }
}

bool Reconstruction::ExtractColorsForImage(const image_t image_id,
                                           const std::string& path) {
  const class Image& image = Image(image_id);

  Bitmap bitmap;
  if (!bitmap.Read(JoinPaths(path, image.Name()))) {
    return false;
  }

  const Eigen::Vector3ub kBlackColor(0, 0, 0);
  for (const Point2D& point2D : image.Points2D()) {
    if (point2D.HasPoint3D()) {
      struct Point3D& point3D = Point3D(point2D.point3D_id);
      if (point3D.color == kBlackColor) {
        BitmapColor<float> color;
        // COLMAP assumes that the upper left pixel center is (0.5, 0.5).
        if (bitmap.InterpolateBilinear(
                point2D.xy(0) - 0.5, point2D.xy(1) - 0.5, &color)) {
          const BitmapColor<uint8_t> color_ub = color.Cast<uint8_t>();
          point3D.color = Eigen::Vector3ub(color_ub.r, color_ub.g, color_ub.b);
        }
      }
    }
  }

  return true;
}

void Reconstruction::ExtractColorsForAllImages(const std::string& path) {
  std::unordered_map<point3D_t, Eigen::Vector3d> color_sums;
  std::unordered_map<point3D_t, size_t> color_counts;

  for (const auto& image_id : RegImageIds()) {
    const class Image& image = Image(image_id);
    const std::string image_path = JoinPaths(path, image.Name());

    Bitmap bitmap;
    if (!bitmap.Read(image_path)) {
      LOG(WARNING) << StringPrintf("Could not read image %s at path %s.",
                                   image.Name().c_str(),
                                   image_path.c_str())
                   << std::endl;
      continue;
    }

    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        BitmapColor<float> color;
        // COLMAP assumes that the upper left pixel center is (0.5, 0.5).
        if (bitmap.InterpolateBilinear(
                point2D.xy(0) - 0.5, point2D.xy(1) - 0.5, &color)) {
          if (color_sums.count(point2D.point3D_id)) {
            Eigen::Vector3d& color_sum = color_sums[point2D.point3D_id];
            color_sum(0) += color.r;
            color_sum(1) += color.g;
            color_sum(2) += color.b;
            color_counts[point2D.point3D_id] += 1;
          } else {
            color_sums.emplace(point2D.point3D_id,
                               Eigen::Vector3d(color.r, color.g, color.b));
            color_counts.emplace(point2D.point3D_id, 1);
          }
        }
      }
    }
  }

  const Eigen::Vector3ub kBlackColor = Eigen::Vector3ub::Zero();
  for (auto& point3D : points3D_) {
    if (color_sums.count(point3D.first)) {
      Eigen::Vector3d color =
          color_sums[point3D.first] / color_counts[point3D.first];
      for (Eigen::Index i = 0; i < color.size(); ++i) {
        color[i] = std::round(color[i]);
      }
      point3D.second.color = color.cast<uint8_t>();
    } else {
      point3D.second.color = kBlackColor;
    }
  }
}

void Reconstruction::CreateImageDirs(const std::string& path) const {
  std::unordered_set<std::string> image_dirs;
  for (const auto& image : images_) {
    const std::vector<std::string> name_split =
        StringSplit(image.second.Name(), "/");
    if (name_split.size() > 1) {
      std::string dir = path;
      for (size_t i = 0; i < name_split.size() - 1; ++i) {
        dir = JoinPaths(dir, name_split[i]);
        image_dirs.insert(dir);
      }
    }
  }
  for (const auto& dir : image_dirs) {
    CreateDirIfNotExists(dir, /*recursive=*/true);
  }
}

std::ostream& operator<<(std::ostream& stream,
                         const Reconstruction& reconstruction) {
  stream << "Reconstruction(num_cameras=" << reconstruction.NumCameras()
         << ", num_images=" << reconstruction.NumImages()
         << ", num_reg_images=" << reconstruction.NumRegImages()
         << ", num_points3D=" << reconstruction.NumPoints3D() << ")";
  return stream;
}

}  // namespace colmap
