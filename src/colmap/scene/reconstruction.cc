// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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
#include "colmap/geometry/pose.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"

#include <fstream>

namespace colmap {

Reconstruction::Reconstruction()
    : correspondence_graph_(nullptr), num_added_points3D_(0) {}

std::unordered_set<point3D_t> Reconstruction::Point3DIds() const {
  std::unordered_set<point3D_t> point3D_ids;
  point3D_ids.reserve(points3D_.size());

  for (const auto& point3D : points3D_) {
    point3D_ids.insert(point3D.first);
  }

  return point3D_ids;
}

void Reconstruction::Load(const DatabaseCache& database_cache) {
  correspondence_graph_ = nullptr;

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
      existing_image.SetNumObservations(image.second.NumObservations());
      existing_image.SetNumCorrespondences(image.second.NumCorrespondences());
    } else {
      AddImage(image.second);
    }
  }

  // Add image pairs.
  for (const auto& image_pair : database_cache.CorrespondenceGraph()
                                    ->NumCorrespondencesBetweenImages()) {
    ImagePairStat image_pair_stat;
    image_pair_stat.num_total_corrs = image_pair.second;
    image_pair_stats_.emplace(image_pair.first, image_pair_stat);
  }
}

void Reconstruction::SetUp(
    std::shared_ptr<const CorrespondenceGraph> correspondence_graph) {
  correspondence_graph_ = std::move(THROW_CHECK_NOTNULL(correspondence_graph));

  for (auto& image : images_) {
    image.second.SetUp(Camera(image.second.CameraId()));
  }

  // If an existing model was loaded from disk and there were already images
  // registered previously, we need to set observations as triangulated.
  for (const auto image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      if (image.Point2D(point2D_idx).HasPoint3D()) {
        const bool kIsContinuedPoint3D = false;
        SetObservationAsTriangulated(
            image_id, point2D_idx, kIsContinuedPoint3D);
      }
    }
  }
}

void Reconstruction::TearDown() {
  correspondence_graph_ = nullptr;
  image_pair_stats_.clear();

  // Remove all not yet registered images.
  std::unordered_set<camera_t> keep_camera_ids;
  for (auto it = images_.begin(); it != images_.end();) {
    if (it->second.IsRegistered()) {
      keep_camera_ids.insert(it->second.CameraId());
      it->second.TearDown();
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
  const image_t image_id = image.ImageId();
  const bool is_registered = image.IsRegistered();
  THROW_CHECK(images_.emplace(image_id, std::move(image)).second);
  if (is_registered) {
    THROW_CHECK_NE(image_id, kInvalidImageId);
    reg_image_ids_.push_back(image_id);
  }
}

point3D_t Reconstruction::AddPoint3D(const Eigen::Vector3d& xyz,
                                     Track track,
                                     const Eigen::Vector3ub& color) {
  const point3D_t point3D_id = ++num_added_points3D_;
  THROW_CHECK(!ExistsPoint3D(point3D_id));

  for (const auto& track_el : track.Elements()) {
    class Image& image = Image(track_el.image_id);
    THROW_CHECK(!image.Point2D(track_el.point2D_idx).HasPoint3D());
    image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
    THROW_CHECK_LE(image.NumPoints3D(), image.NumPoints2D());
  }

  const bool kIsContinuedPoint3D = false;

  for (const auto& track_el : track.Elements()) {
    SetObservationAsTriangulated(
        track_el.image_id, track_el.point2D_idx, kIsContinuedPoint3D);
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

  const bool kIsContinuedPoint3D = true;
  SetObservationAsTriangulated(
      track_el.image_id, track_el.point2D_idx, kIsContinuedPoint3D);
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
  // Note: Do not change order of these instructions, especially with respect to
  // `Reconstruction::ResetTriObservations`

  const class Track& track = Point3D(point3D_id).track;

  const bool kIsDeletedPoint3D = true;

  for (const auto& track_el : track.Elements()) {
    ResetTriObservations(
        track_el.image_id, track_el.point2D_idx, kIsDeletedPoint3D);
  }

  for (const auto& track_el : track.Elements()) {
    class Image& image = Image(track_el.image_id);
    image.ResetPoint3DForPoint2D(track_el.point2D_idx);
  }

  points3D_.erase(point3D_id);
}

void Reconstruction::DeleteObservation(const image_t image_id,
                                       const point2D_t point2D_idx) {
  // Note: Do not change order of these instructions, especially with respect to
  // `Reconstruction::ResetTriObservations`

  class Image& image = Image(image_id);
  const point3D_t point3D_id = image.Point2D(point2D_idx).point3D_id;
  struct Point3D& point3D = Point3D(point3D_id);

  if (point3D.track.Length() <= 2) {
    DeletePoint3D(point3D_id);
    return;
  }

  point3D.track.DeleteElement(image_id, point2D_idx);

  const bool kIsDeletedPoint3D = false;
  ResetTriObservations(image_id, point2D_idx, kIsDeletedPoint3D);

  image.ResetPoint3DForPoint2D(point2D_idx);
}

void Reconstruction::DeleteAllPoints2DAndPoints3D() {
  points3D_.clear();
  for (auto& image : images_) {
    class Image new_image;
    new_image.SetImageId(image.second.ImageId());
    new_image.SetName(image.second.Name());
    new_image.SetCameraId(image.second.CameraId());
    new_image.SetRegistered(image.second.IsRegistered());
    new_image.SetNumCorrespondences(image.second.NumCorrespondences());
    new_image.CamFromWorld() = image.second.CamFromWorld();
    new_image.CamFromWorldPrior() = image.second.CamFromWorldPrior();
    image.second = std::move(new_image);
  }
}

void Reconstruction::RegisterImage(const image_t image_id) {
  class Image& image = Image(image_id);
  if (!image.IsRegistered()) {
    image.SetRegistered(true);
    reg_image_ids_.push_back(image_id);
  }
}

void Reconstruction::DeRegisterImage(const image_t image_id) {
  class Image& image = Image(image_id);

  const auto num_points2D = image.NumPoints2D();
  for (point2D_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
    if (image.Point2D(point2D_idx).HasPoint3D()) {
      DeleteObservation(image_id, point2D_idx);
    }
  }

  image.SetRegistered(false);

  reg_image_ids_.erase(
      std::remove(reg_image_ids_.begin(), reg_image_ids_.end(), image_id),
      reg_image_ids_.end());
}

void Reconstruction::Normalize(const double extent,
                               const double p0,
                               const double p1,
                               const bool use_images) {
  THROW_CHECK_GT(extent, 0);

  if ((use_images && reg_image_ids_.size() < 2) ||
      (!use_images && points3D_.size() < 2)) {
    return;
  }

  auto bound = ComputeBoundsAndCentroid(p0, p1, use_images);

  // Calculate scale and translation, such that
  // translation is applied before scaling.
  const double old_extent = (std::get<1>(bound) - std::get<0>(bound)).norm();
  double scale;
  if (old_extent < std::numeric_limits<double>::epsilon()) {
    scale = 1;
  } else {
    scale = extent / old_extent;
  }

  Sim3d tform(
      scale, Eigen::Quaterniond::Identity(), -scale * std::get<2>(bound));
  Transform(tform);
}

Eigen::Vector3d Reconstruction::ComputeCentroid(const double p0,
                                                const double p1) const {
  return std::get<2>(ComputeBoundsAndCentroid(p0, p1, false));
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> Reconstruction::ComputeBoundingBox(
    const double p0, const double p1) const {
  auto bound = ComputeBoundsAndCentroid(p0, p1, false);
  return std::make_pair(std::get<0>(bound), std::get<1>(bound));
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
Reconstruction::ComputeBoundsAndCentroid(const double p0,
                                         const double p1,
                                         const bool use_images) const {
  THROW_CHECK_GE(p0, 0);
  THROW_CHECK_LE(p0, 1);
  THROW_CHECK_GE(p1, 0);
  THROW_CHECK_LE(p1, 1);
  THROW_CHECK_LE(p0, p1);

  const size_t num_elements =
      use_images ? reg_image_ids_.size() : points3D_.size();
  if (num_elements == 0) {
    return std::make_tuple(Eigen::Vector3d(0, 0, 0),
                           Eigen::Vector3d(0, 0, 0),
                           Eigen::Vector3d(0, 0, 0));
  }

  // Coordinates of image centers or point locations.
  std::vector<float> coords_x;
  std::vector<float> coords_y;
  std::vector<float> coords_z;
  if (use_images) {
    coords_x.reserve(reg_image_ids_.size());
    coords_y.reserve(reg_image_ids_.size());
    coords_z.reserve(reg_image_ids_.size());
    for (const image_t im_id : reg_image_ids_) {
      const Eigen::Vector3d proj_center = Image(im_id).ProjectionCenter();
      coords_x.push_back(static_cast<float>(proj_center(0)));
      coords_y.push_back(static_cast<float>(proj_center(1)));
      coords_z.push_back(static_cast<float>(proj_center(2)));
    }
  } else {
    coords_x.reserve(points3D_.size());
    coords_y.reserve(points3D_.size());
    coords_z.reserve(points3D_.size());
    for (const auto& point3D : points3D_) {
      coords_x.push_back(static_cast<float>(point3D.second.xyz(0)));
      coords_y.push_back(static_cast<float>(point3D.second.xyz(1)));
      coords_z.push_back(static_cast<float>(point3D.second.xyz(2)));
    }
  }

  // Determine robust bounding box and mean.

  std::sort(coords_x.begin(), coords_x.end());
  std::sort(coords_y.begin(), coords_y.end());
  std::sort(coords_z.begin(), coords_z.end());

  const size_t P0 = static_cast<size_t>(
      (coords_x.size() > 3) ? p0 * (coords_x.size() - 1) : 0);
  const size_t P1 = static_cast<size_t>(
      (coords_x.size() > 3) ? p1 * (coords_x.size() - 1) : coords_x.size() - 1);

  const Eigen::Vector3d bbox_min(coords_x[P0], coords_y[P0], coords_z[P0]);
  const Eigen::Vector3d bbox_max(coords_x[P1], coords_y[P1], coords_z[P1]);

  Eigen::Vector3d mean_coord(0, 0, 0);
  for (size_t i = P0; i <= P1; ++i) {
    mean_coord(0) += coords_x[i];
    mean_coord(1) += coords_y[i];
    mean_coord(2) += coords_z[i];
  }
  mean_coord /= P1 - P0 + 1;

  return std::make_tuple(bbox_min, bbox_max, mean_coord);
}

void Reconstruction::Transform(const Sim3d& new_from_old_world) {
  for (auto& image : images_) {
    image.second.CamFromWorld() =
        TransformCameraWorld(new_from_old_world, image.second.CamFromWorld());
  }
  for (auto& point3D : points3D_) {
    point3D.second.xyz = new_from_old_world * point3D.second.xyz;
  }
}

Reconstruction Reconstruction::Crop(
    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& bbox) const {
  Reconstruction cropped_reconstruction;
  for (const auto& camera : cameras_) {
    cropped_reconstruction.AddCamera(camera.second);
  }
  for (const auto& image : images_) {
    auto new_image = image.second;
    new_image.SetRegistered(false);
    for (auto& point2D : new_image.Points2D()) {
      point2D.point3D_id = kInvalidPoint3DId;
    }
    cropped_reconstruction.AddImage(std::move(new_image));
  }
  std::unordered_set<image_t> registered_image_ids;
  for (const auto& point3D : points3D_) {
    if ((point3D.second.xyz.array() >= bbox.first.array()).all() &&
        (point3D.second.xyz.array() <= bbox.second.array()).all()) {
      for (const auto& track_el : point3D.second.track.Elements()) {
        if (registered_image_ids.count(track_el.image_id) == 0) {
          cropped_reconstruction.RegisterImage(track_el.image_id);
          registered_image_ids.insert(track_el.image_id);
        }
      }
      cropped_reconstruction.AddPoint3D(
          point3D.second.xyz, point3D.second.track, point3D.second.color);
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
  for (const auto image_id : reg_image_ids_) {
    const auto& image = Image(image_id);
    const auto* other_image = other.FindImageWithName(image.Name());
    if (other_image != nullptr && other_image->IsRegistered()) {
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

  for (auto& image_id : reg_image_ids_) {
    image_id = old_to_new_image_ids.at(image_id);
  }

  for (auto& point3D : points3D_) {
    for (auto& track_el : point3D.second.track.Elements()) {
      track_el.image_id = old_to_new_image_ids.at(track_el.image_id);
    }
  }
}

size_t Reconstruction::FilterPoints3D(
    const double max_reproj_error,
    const double min_tri_angle,
    const std::unordered_set<point3D_t>& point3D_ids) {
  size_t num_filtered = 0;
  num_filtered +=
      FilterPoints3DWithLargeReprojectionError(max_reproj_error, point3D_ids);
  num_filtered +=
      FilterPoints3DWithSmallTriangulationAngle(min_tri_angle, point3D_ids);
  return num_filtered;
}

size_t Reconstruction::FilterPoints3DInImages(
    const double max_reproj_error,
    const double min_tri_angle,
    const std::unordered_set<image_t>& image_ids) {
  std::unordered_set<point3D_t> point3D_ids;
  for (const image_t image_id : image_ids) {
    const class Image& image = Image(image_id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids.insert(point2D.point3D_id);
      }
    }
  }
  return FilterPoints3D(max_reproj_error, min_tri_angle, point3D_ids);
}

size_t Reconstruction::FilterAllPoints3D(const double max_reproj_error,
                                         const double min_tri_angle) {
  // Important: First filter observations and points with large reprojection
  // error, so that observations with large reprojection error do not make
  // a point stable through a large triangulation angle.
  const std::unordered_set<point3D_t>& point3D_ids = Point3DIds();
  size_t num_filtered = 0;
  num_filtered +=
      FilterPoints3DWithLargeReprojectionError(max_reproj_error, point3D_ids);
  num_filtered +=
      FilterPoints3DWithSmallTriangulationAngle(min_tri_angle, point3D_ids);
  return num_filtered;
}

size_t Reconstruction::FilterObservationsWithNegativeDepth() {
  size_t num_filtered = 0;
  for (const auto image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    const Eigen::Matrix3x4d cam_from_world = image.CamFromWorld().ToMatrix();
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        const struct Point3D& point3D = Point3D(point2D.point3D_id);
        if (!HasPointPositiveDepth(cam_from_world, point3D.xyz)) {
          DeleteObservation(image_id, point2D_idx);
          num_filtered += 1;
        }
      }
    }
  }
  return num_filtered;
}

std::vector<image_t> Reconstruction::FilterImages(
    const double min_focal_length_ratio,
    const double max_focal_length_ratio,
    const double max_extra_param) {
  std::vector<image_t> filtered_image_ids;
  for (const image_t image_id : RegImageIds()) {
    const class Image& image = Image(image_id);
    if (image.NumPoints3D() == 0 || Camera(image.CameraId())
                                        .HasBogusParams(min_focal_length_ratio,
                                                        max_focal_length_ratio,
                                                        max_extra_param)) {
      filtered_image_ids.push_back(image_id);
    }
  }

  // Only de-register after iterating over reg_image_ids_ to avoid
  // simultaneous iteration and modification of the vector.
  for (const image_t image_id : filtered_image_ids) {
    DeRegisterImage(image_id);
  }

  return filtered_image_ids;
}

size_t Reconstruction::ComputeNumObservations() const {
  size_t num_obs = 0;
  for (const image_t image_id : reg_image_ids_) {
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
  if (reg_image_ids_.empty()) {
    return 0.0;
  } else {
    return ComputeNumObservations() /
           static_cast<double>(reg_image_ids_.size());
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
      const auto& camera = Camera(image.CameraId());
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
  ReadCamerasText(JoinPaths(path, "cameras.txt"));
  ReadImagesText(JoinPaths(path, "images.txt"));
  ReadPoints3DText(JoinPaths(path, "points3D.txt"));
}

void Reconstruction::ReadBinary(const std::string& path) {
  ReadCamerasBinary(JoinPaths(path, "cameras.bin"));
  ReadImagesBinary(JoinPaths(path, "images.bin"));
  ReadPoints3DBinary(JoinPaths(path, "points3D.bin"));
}

void Reconstruction::WriteText(const std::string& path) const {
  THROW_CHECK_DIR_EXISTS(path);
  WriteCamerasText(JoinPaths(path, "cameras.txt"));
  WriteImagesText(JoinPaths(path, "images.txt"));
  WritePoints3DText(JoinPaths(path, "points3D.txt"));
}

void Reconstruction::WriteBinary(const std::string& path) const {
  THROW_CHECK_DIR_EXISTS(path);
  WriteCamerasBinary(JoinPaths(path, "cameras.bin"));
  WriteImagesBinary(JoinPaths(path, "images.bin"));
  WritePoints3DBinary(JoinPaths(path, "points3D.bin"));
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

bool Reconstruction::ExportNVM(const std::string& path,
                               bool skip_distortion) const {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  // Ensure that we don't lose any precision by storing in text.
  file.precision(17);

  // White space added for compatibility with Meshlab.
  file << "NVM_V3 " << std::endl << " " << std::endl;
  file << reg_image_ids_.size() << "  " << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  for (const auto image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    const struct Camera& camera = Camera(image.CameraId());

    double k;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k = -1 * camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
    } else {
      LOG(WARNING) << "NVM only supports `SIMPLE_RADIAL` "
                      "and pinhole camera models."
                   << std::endl;
      return false;
    }

    const Eigen::Vector3d proj_center = image.ProjectionCenter();

    file << image.Name() << " ";
    file << camera.MeanFocalLength() << " ";
    file << image.CamFromWorld().rotation.w() << " ";
    file << image.CamFromWorld().rotation.x() << " ";
    file << image.CamFromWorld().rotation.y() << " ";
    file << image.CamFromWorld().rotation.z() << " ";
    file << proj_center.x() << " ";
    file << proj_center.y() << " ";
    file << proj_center.z() << " ";
    file << k << " ";
    file << 0 << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  file << std::endl << points3D_.size() << std::endl;

  for (const auto& point3D : points3D_) {
    file << point3D.second.xyz(0) << " ";
    file << point3D.second.xyz(1) << " ";
    file << point3D.second.xyz(2) << " ";
    file << static_cast<int>(point3D.second.color(0)) << " ";
    file << static_cast<int>(point3D.second.color(1)) << " ";
    file << static_cast<int>(point3D.second.color(2)) << " ";

    std::ostringstream line;

    std::unordered_set<image_t> image_ids;
    for (const auto& track_el : point3D.second.track.Elements()) {
      // Make sure that each point only has a single observation per image,
      // since VisualSfM does not support with multiple observations.
      if (image_ids.count(track_el.image_id) == 0) {
        const class Image& image = Image(track_el.image_id);
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        line << image_id_to_idx_[track_el.image_id] << " ";
        line << track_el.point2D_idx << " ";
        line << point2D.xy(0) << " ";
        line << point2D.xy(1) << " ";
        image_ids.insert(track_el.image_id);
      }
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << image_ids.size() << " ";
    file << line_string << std::endl;
  }

  return true;
}

bool Reconstruction::ExportCam(const std::string& path,
                               bool skip_distortion) const {
  CreateImageDirs(path);
  for (const auto image_id : reg_image_ids_) {
    std::string name, ext;
    const class Image& image = Image(image_id);
    const struct Camera& camera = Camera(image.CameraId());

    SplitFileExtension(image.Name(), &name, &ext);
    name = JoinPaths(path, name.append(".cam"));
    std::ofstream file(name, std::ios::trunc);

    THROW_CHECK_FILE_OPEN(file, name);

    // Ensure that we don't lose any precision by storing in text.
    file.precision(17);

    double k1, k2;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k1 = 0.0;
      k2 = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k1 = camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
      k2 = 0.0;
    } else if (camera.model_id == RadialCameraModel::model_id) {
      k1 = camera.params[RadialCameraModel::extra_params_idxs[0]];
      k2 = camera.params[RadialCameraModel::extra_params_idxs[1]];
    } else {
      LOG(WARNING) << "CAM only supports `SIMPLE_RADIAL`, `RADIAL`, "
                      "and pinhole camera models."
                   << std::endl;
      return false;
    }

    // If both k1 and k2 values are non-zero, then the CAM format assumes
    // a Bundler-like radial distortion model, which converts well from
    // COLMAP. However, if k2 is zero, then a different model is used
    // that does not translate as well, so we avoid setting k2 to zero.
    if (k1 != 0.0 && k2 == 0.0) {
      k2 = 1e-10;
    }

    const double fx = camera.FocalLengthX();
    const double fy = camera.FocalLengthY();
    double focal_length;
    if (camera.width * fy < camera.height * fx) {
      focal_length = fy / camera.height;
    } else {
      focal_length = fx / camera.width;
    }

    const Eigen::Matrix3d R = image.CamFromWorld().rotation.toRotationMatrix();
    file << image.CamFromWorld().translation.x() << " "
         << image.CamFromWorld().translation.y() << " "
         << image.CamFromWorld().translation.z() << " " << R(0, 0) << " "
         << R(0, 1) << " " << R(0, 2) << " " << R(1, 0) << " " << R(1, 1) << " "
         << R(1, 2) << " " << R(2, 0) << " " << R(2, 1) << " " << R(2, 2)
         << std::endl;
    file << focal_length << " " << k1 << " " << k2 << " " << fy / fx << " "
         << camera.PrincipalPointX() / camera.width << " "
         << camera.PrincipalPointY() / camera.height << std::endl;
  }

  return true;
}

bool Reconstruction::ExportRecon3D(const std::string& path,
                                   bool skip_distortion) const {
  std::string base_path = EnsureTrailingSlash(StringReplace(path, "\\", "/"));
  CreateDirIfNotExists(base_path);
  base_path = base_path.append("Recon/");
  CreateDirIfNotExists(base_path);
  std::string synth_path = base_path + "synth_0.out";
  std::string image_list_path = base_path + "urd-images.txt";
  std::string image_map_path = base_path + "imagemap_0.txt";

  std::ofstream synth_file(synth_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(synth_file, synth_path);
  std::ofstream image_list_file(image_list_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(image_list_file, image_list_path);
  std::ofstream image_map_file(image_map_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(image_map_file, image_map_path);

  // Ensure that we don't lose any precision by storing in text.
  synth_file.precision(17);

  // Write header info
  synth_file << "colmap 1.0" << std::endl;
  synth_file << reg_image_ids_.size() << " " << points3D_.size() << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  // Write image/camera info
  for (const auto image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    const struct Camera& camera = Camera(image.CameraId());

    double k1, k2;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k1 = 0.0;
      k2 = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k1 = -1 * camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
      k2 = 0.0;
    } else if (camera.model_id == RadialCameraModel::model_id) {
      k1 = -1 * camera.params[RadialCameraModel::extra_params_idxs[0]];
      k2 = -1 * camera.params[RadialCameraModel::extra_params_idxs[1]];
    } else {
      LOG(WARNING) << "Recon3D only supports `SIMPLE_RADIAL`, "
                      "`RADIAL`, and pinhole camera models.";
      return false;
    }

    const double scale = 1.0 / (double)std::max(camera.width, camera.height);
    synth_file << scale * camera.MeanFocalLength() << " " << k1 << " " << k2
               << std::endl;
    synth_file << image.CamFromWorld().rotation.toRotationMatrix() << std::endl;
    synth_file << image.CamFromWorld().translation.transpose() << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_list_file << image.Name() << std::endl
                    << camera.width << " " << camera.height << std::endl;
    image_map_file << image_idx << std::endl;

    image_idx += 1;
  }
  image_list_file.close();
  image_map_file.close();

  // Write point info
  for (const auto& point3D : points3D_) {
    auto& p = point3D.second;
    synth_file << p.xyz(0) << " " << p.xyz(1) << " " << p.xyz(2) << std::endl;
    synth_file << static_cast<int>(p.color(0)) << " "
               << static_cast<int>(p.color(1)) << " "
               << static_cast<int>(p.color(2)) << std::endl;

    std::ostringstream line;

    std::unordered_set<image_t> image_ids;
    for (const auto& track_el : p.track.Elements()) {
      // Make sure that each point only has a single observation per image,
      // since VisualSfM does not support with multiple observations.
      if (image_ids.count(track_el.image_id) == 0) {
        const class Image& image = Image(track_el.image_id);
        const struct Camera& camera = Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);

        const double scale =
            1.0 / (double)std::max(camera.width, camera.height);

        line << image_id_to_idx_[track_el.image_id] << " ";
        line << track_el.point2D_idx << " ";
        // Use a scale of -1.0 to mark as invalid as it is not needed currently
        line << "-1.0 ";
        line << (point2D.xy(0) - camera.PrincipalPointX()) * scale << " ";
        line << (point2D.xy(1) - camera.PrincipalPointY()) * scale << " ";
        image_ids.insert(track_el.image_id);
      }
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    synth_file << image_ids.size() << " ";
    synth_file << line_string << std::endl;
  }
  synth_file.close();

  return true;
}

bool Reconstruction::ExportBundler(const std::string& path,
                                   const std::string& list_path,
                                   bool skip_distortion) const {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  std::ofstream list_file(list_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(list_file, list_path);

  // Ensure that we don't lose any precision by storing in text.
  file.precision(17);

  file << "# Bundle file v0.3" << std::endl;

  file << reg_image_ids_.size() << " " << points3D_.size() << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  for (const image_t image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    const struct Camera& camera = Camera(image.CameraId());

    double k1, k2;
    if (skip_distortion ||
        camera.model_id == SimplePinholeCameraModel::model_id ||
        camera.model_id == PinholeCameraModel::model_id) {
      k1 = 0.0;
      k2 = 0.0;
    } else if (camera.model_id == SimpleRadialCameraModel::model_id) {
      k1 = camera.params[SimpleRadialCameraModel::extra_params_idxs[0]];
      k2 = 0.0;
    } else if (camera.model_id == RadialCameraModel::model_id) {
      k1 = camera.params[RadialCameraModel::extra_params_idxs[0]];
      k2 = camera.params[RadialCameraModel::extra_params_idxs[1]];
    } else {
      LOG(WARNING) << "Bundler only supports `SIMPLE_RADIAL`, "
                      "`RADIAL`, and pinhole camera models."
                   << std::endl;
      return false;
    }

    file << camera.MeanFocalLength() << " " << k1 << " " << k2 << std::endl;

    const Eigen::Matrix3d R = image.CamFromWorld().rotation.toRotationMatrix();
    file << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << std::endl;
    file << -R(1, 0) << " " << -R(1, 1) << " " << -R(1, 2) << std::endl;
    file << -R(2, 0) << " " << -R(2, 1) << " " << -R(2, 2) << std::endl;

    file << image.CamFromWorld().translation.x() << " ";
    file << -image.CamFromWorld().translation.y() << " ";
    file << -image.CamFromWorld().translation.z() << std::endl;

    list_file << image.Name() << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  for (const auto& point3D : points3D_) {
    file << point3D.second.xyz(0) << " ";
    file << point3D.second.xyz(1) << " ";
    file << point3D.second.xyz(2) << std::endl;

    file << static_cast<int>(point3D.second.color(0)) << " ";
    file << static_cast<int>(point3D.second.color(1)) << " ";
    file << static_cast<int>(point3D.second.color(2)) << std::endl;

    std::ostringstream line;

    line << point3D.second.track.Length() << " ";

    for (const auto& track_el : point3D.second.track.Elements()) {
      const class Image& image = Image(track_el.image_id);
      const struct Camera& camera = Camera(image.CameraId());

      // Bundler output assumes image coordinate system origin
      // in the lower left corner of the image with the center of
      // the lower left pixel being (0, 0). Our coordinate system
      // starts in the upper left corner with the center of the
      // upper left pixel being (0.5, 0.5).

      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      line << image_id_to_idx_.at(track_el.image_id) << " ";
      line << track_el.point2D_idx << " ";
      line << point2D.xy(0) - camera.PrincipalPointX() << " ";
      line << camera.PrincipalPointY() - point2D.xy(1) << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << line_string << std::endl;
  }

  return true;
}

void Reconstruction::ExportPLY(const std::string& path) const {
  const auto ply_points = ConvertToPLY();

  const bool kWriteNormal = false;
  const bool kWriteRGB = true;
  WriteBinaryPlyPoints(path, ply_points, kWriteNormal, kWriteRGB);
}

void Reconstruction::ExportVRML(const std::string& images_path,
                                const std::string& points3D_path,
                                const double image_scale,
                                const Eigen::Vector3d& image_rgb) const {
  std::ofstream images_file(images_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(images_file, images_path);

  const double six = image_scale * 0.15;
  const double siy = image_scale * 0.1;

  std::vector<Eigen::Vector3d> points;
  points.emplace_back(-six, -siy, six * 1.0 * 2.0);
  points.emplace_back(+six, -siy, six * 1.0 * 2.0);
  points.emplace_back(+six, +siy, six * 1.0 * 2.0);
  points.emplace_back(-six, +siy, six * 1.0 * 2.0);
  points.emplace_back(0, 0, 0);
  points.emplace_back(-six / 3.0, -siy / 3.0, six * 1.0 * 2.0);
  points.emplace_back(+six / 3.0, -siy / 3.0, six * 1.0 * 2.0);
  points.emplace_back(+six / 3.0, +siy / 3.0, six * 1.0 * 2.0);
  points.emplace_back(-six / 3.0, +siy / 3.0, six * 1.0 * 2.0);

  for (const auto& image : images_) {
    if (!image.second.IsRegistered()) {
      continue;
    }

    images_file << "Shape{\n";
    images_file << " appearance Appearance {\n";
    images_file << "  material DEF Default-ffRffGffB Material {\n";
    images_file << "  ambientIntensity 0\n";
    images_file << "  diffuseColor "
                << " " << image_rgb(0) << " " << image_rgb(1) << " "
                << image_rgb(2) << "\n";
    images_file << "  emissiveColor 0.1 0.1 0.1 } }\n";
    images_file << " geometry IndexedFaceSet {\n";
    images_file << " solid FALSE \n";
    images_file << " colorPerVertex TRUE \n";
    images_file << " ccw TRUE \n";

    images_file << " coord Coordinate {\n";
    images_file << " point [\n";

    // Move camera base model to camera pose.
    const Eigen::Matrix3x4d world_from_cam =
        Inverse(image.second.CamFromWorld()).ToMatrix();
    for (size_t i = 0; i < points.size(); i++) {
      const Eigen::Vector3d point = world_from_cam * points[i].homogeneous();
      images_file << point(0) << " " << point(1) << " " << point(2) << "\n";
    }

    images_file << " ] }\n";

    images_file << "color Color {color [\n";
    for (size_t p = 0; p < points.size(); p++) {
      images_file << " " << image_rgb(0) << " " << image_rgb(1) << " "
                  << image_rgb(2) << "\n";
    }

    images_file << "\n] }\n";

    images_file << "coordIndex [\n";
    images_file << " 0, 1, 2, 3, -1\n";
    images_file << " 5, 6, 4, -1\n";
    images_file << " 6, 7, 4, -1\n";
    images_file << " 7, 8, 4, -1\n";
    images_file << " 8, 5, 4, -1\n";
    images_file << " \n] \n";

    images_file << " texCoord TextureCoordinate { point [\n";
    images_file << "  1 1,\n";
    images_file << "  0 1,\n";
    images_file << "  0 0,\n";
    images_file << "  1 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";
    images_file << "  0 0,\n";

    images_file << " ] }\n";
    images_file << "} }\n";
  }

  // Write 3D points

  std::ofstream points3D_file(points3D_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(points3D_file, points3D_path);

  points3D_file << "#VRML V2.0 utf8\n";
  points3D_file << "Background { skyColor [1.0 1.0 1.0] } \n";
  points3D_file << "Shape{ appearance Appearance {\n";
  points3D_file << " material Material {emissiveColor 1 1 1} }\n";
  points3D_file << " geometry PointSet {\n";
  points3D_file << " coord Coordinate {\n";
  points3D_file << "  point [\n";

  for (const auto& point3D : points3D_) {
    points3D_file << point3D.second.xyz(0) << ", ";
    points3D_file << point3D.second.xyz(1) << ", ";
    points3D_file << point3D.second.xyz(2) << std::endl;
  }

  points3D_file << " ] }\n";
  points3D_file << " color Color { color [\n";

  for (const auto& point3D : points3D_) {
    points3D_file << point3D.second.color(0) / 255.0 << ", ";
    points3D_file << point3D.second.color(1) / 255.0 << ", ";
    points3D_file << point3D.second.color(2) / 255.0 << std::endl;
  }

  points3D_file << " ] } } }\n";
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

  for (size_t i = 0; i < reg_image_ids_.size(); ++i) {
    const class Image& image = Image(reg_image_ids_[i]);
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

size_t Reconstruction::FilterPoints3DWithSmallTriangulationAngle(
    const double min_tri_angle,
    const std::unordered_set<point3D_t>& point3D_ids) {
  // Number of filtered points.
  size_t num_filtered = 0;

  // Minimum triangulation angle in radians.
  const double min_tri_angle_rad = DegToRad(min_tri_angle);

  // Cache for image projection centers.
  std::unordered_map<image_t, Eigen::Vector3d> proj_centers;

  for (const auto point3D_id : point3D_ids) {
    if (!ExistsPoint3D(point3D_id)) {
      continue;
    }

    const struct Point3D& point3D = Point3D(point3D_id);

    // Calculate triangulation angle for all pairwise combinations of image
    // poses in the track. Only delete point if none of the combinations
    // has a sufficient triangulation angle.
    bool keep_point = false;
    for (size_t i1 = 0; i1 < point3D.track.Length(); ++i1) {
      const image_t image_id1 = point3D.track.Element(i1).image_id;

      Eigen::Vector3d proj_center1;
      if (proj_centers.count(image_id1) == 0) {
        const class Image& image1 = Image(image_id1);
        proj_center1 = image1.ProjectionCenter();
        proj_centers.emplace(image_id1, proj_center1);
      } else {
        proj_center1 = proj_centers.at(image_id1);
      }

      for (size_t i2 = 0; i2 < i1; ++i2) {
        const image_t image_id2 = point3D.track.Element(i2).image_id;
        const Eigen::Vector3d proj_center2 = proj_centers.at(image_id2);

        const double tri_angle = CalculateTriangulationAngle(
            proj_center1, proj_center2, point3D.xyz);

        if (tri_angle >= min_tri_angle_rad) {
          keep_point = true;
          break;
        }
      }

      if (keep_point) {
        break;
      }
    }

    if (!keep_point) {
      num_filtered += 1;
      DeletePoint3D(point3D_id);
    }
  }

  return num_filtered;
}

size_t Reconstruction::FilterPoints3DWithLargeReprojectionError(
    const double max_reproj_error,
    const std::unordered_set<point3D_t>& point3D_ids) {
  const double max_squared_reproj_error = max_reproj_error * max_reproj_error;

  // Number of filtered points.
  size_t num_filtered = 0;

  for (const auto point3D_id : point3D_ids) {
    if (!ExistsPoint3D(point3D_id)) {
      continue;
    }

    struct Point3D& point3D = Point3D(point3D_id);

    if (point3D.track.Length() < 2) {
      num_filtered += point3D.track.Length();
      DeletePoint3D(point3D_id);
      continue;
    }

    double reproj_error_sum = 0.0;

    std::vector<TrackElement> track_els_to_delete;

    for (const auto& track_el : point3D.track.Elements()) {
      const class Image& image = Image(track_el.image_id);
      const struct Camera& camera = Camera(image.CameraId());
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      const double squared_reproj_error = CalculateSquaredReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      if (squared_reproj_error > max_squared_reproj_error) {
        track_els_to_delete.push_back(track_el);
      } else {
        reproj_error_sum += std::sqrt(squared_reproj_error);
      }
    }

    if (track_els_to_delete.size() >= point3D.track.Length() - 1) {
      num_filtered += point3D.track.Length();
      DeletePoint3D(point3D_id);
    } else {
      num_filtered += track_els_to_delete.size();
      for (const auto& track_el : track_els_to_delete) {
        DeleteObservation(track_el.image_id, track_el.point2D_idx);
      }
      point3D.error = reproj_error_sum / point3D.track.Length();
    }
  }

  return num_filtered;
}

void Reconstruction::ReadCamerasText(const std::string& path) {
  cameras_.clear();

  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  std::string item;

  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    struct Camera camera;

    // ID
    std::getline(line_stream, item, ' ');
    camera.camera_id = std::stoul(item);

    // MODEL
    std::getline(line_stream, item, ' ');
    camera.model_id = CameraModelNameToId(item);

    // WIDTH
    std::getline(line_stream, item, ' ');
    camera.width = std::stoll(item);

    // HEIGHT
    std::getline(line_stream, item, ' ');
    camera.height = std::stoll(item);

    // PARAMS
    camera.params.reserve(CameraModelNumParams(camera.model_id));
    while (!line_stream.eof()) {
      std::getline(line_stream, item, ' ');
      camera.params.push_back(std::stold(item));
    }

    THROW_CHECK(camera.VerifyParams());

    cameras_.emplace(camera.camera_id, std::move(camera));
  }
}

void Reconstruction::ReadImagesText(const std::string& path) {
  images_.clear();

  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  std::string item;

  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream1(line);

    // ID
    std::getline(line_stream1, item, ' ');
    const image_t image_id = std::stoul(item);

    class Image image;
    image.SetImageId(image_id);

    image.SetRegistered(true);
    reg_image_ids_.push_back(image_id);

    Rigid3d& cam_from_world = image.CamFromWorld();

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.w() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.x() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.y() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.rotation.z() = std::stold(item);

    cam_from_world.rotation.normalize();

    std::getline(line_stream1, item, ' ');
    cam_from_world.translation.x() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.translation.y() = std::stold(item);

    std::getline(line_stream1, item, ' ');
    cam_from_world.translation.z() = std::stold(item);

    // CAMERA_ID
    std::getline(line_stream1, item, ' ');
    image.SetCameraId(std::stoul(item));

    // NAME
    std::getline(line_stream1, item, ' ');
    image.SetName(item);

    // POINTS2D
    if (!std::getline(file, line)) {
      break;
    }

    StringTrim(&line);
    std::stringstream line_stream2(line);

    std::vector<Eigen::Vector2d> points2D;
    std::vector<point3D_t> point3D_ids;

    if (!line.empty()) {
      while (!line_stream2.eof()) {
        Eigen::Vector2d point;

        std::getline(line_stream2, item, ' ');
        point.x() = std::stold(item);

        std::getline(line_stream2, item, ' ');
        point.y() = std::stold(item);

        points2D.push_back(point);

        std::getline(line_stream2, item, ' ');
        if (item == "-1") {
          point3D_ids.push_back(kInvalidPoint3DId);
        } else {
          point3D_ids.push_back(std::stoll(item));
        }
      }
    }

    image.SetUp(Camera(image.CameraId()));
    image.SetPoints2D(points2D);

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      if (point3D_ids[point2D_idx] != kInvalidPoint3DId) {
        image.SetPoint3DForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
      }
    }

    images_.emplace(image.ImageId(), image);
  }
}

void Reconstruction::ReadPoints3DText(const std::string& path) {
  points3D_.clear();

  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);

  std::string line;
  std::string item;

  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    // ID
    std::getline(line_stream, item, ' ');
    const point3D_t point3D_id = std::stoll(item);

    // Make sure, that we can add new 3D points after reading 3D points
    // without overwriting existing 3D points.
    num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

    struct Point3D point3D;

    // XYZ
    std::getline(line_stream, item, ' ');
    point3D.xyz(0) = std::stold(item);

    std::getline(line_stream, item, ' ');
    point3D.xyz(1) = std::stold(item);

    std::getline(line_stream, item, ' ');
    point3D.xyz(2) = std::stold(item);

    // Color
    std::getline(line_stream, item, ' ');
    point3D.color(0) = static_cast<uint8_t>(std::stoi(item));

    std::getline(line_stream, item, ' ');
    point3D.color(1) = static_cast<uint8_t>(std::stoi(item));

    std::getline(line_stream, item, ' ');
    point3D.color(2) = static_cast<uint8_t>(std::stoi(item));

    // ERROR
    std::getline(line_stream, item, ' ');
    point3D.error = std::stold(item);

    // TRACK
    while (!line_stream.eof()) {
      TrackElement track_el;

      std::getline(line_stream, item, ' ');
      StringTrim(&item);
      if (item.empty()) {
        break;
      }
      track_el.image_id = std::stoul(item);

      std::getline(line_stream, item, ' ');
      track_el.point2D_idx = std::stoul(item);

      point3D.track.AddElement(track_el);
    }

    point3D.track.Compress();

    points3D_.emplace(point3D_id, point3D);
  }
}

void Reconstruction::ReadCamerasBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
  for (size_t i = 0; i < num_cameras; ++i) {
    struct Camera camera;
    camera.camera_id = ReadBinaryLittleEndian<camera_t>(&file);
    camera.model_id =
        static_cast<CameraModelId>(ReadBinaryLittleEndian<int>(&file));
    camera.width = ReadBinaryLittleEndian<uint64_t>(&file);
    camera.height = ReadBinaryLittleEndian<uint64_t>(&file);
    camera.params.resize(CameraModelNumParams(camera.model_id), 0.);
    ReadBinaryLittleEndian<double>(&file, &camera.params);
    THROW_CHECK(camera.VerifyParams());
    cameras_.emplace(camera.camera_id, std::move(camera));
  }
}

void Reconstruction::ReadImagesBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&file);
  for (size_t i = 0; i < num_reg_images; ++i) {
    class Image image;

    image.SetImageId(ReadBinaryLittleEndian<image_t>(&file));

    Rigid3d& cam_from_world = image.CamFromWorld();
    cam_from_world.rotation.w() = ReadBinaryLittleEndian<double>(&file);
    cam_from_world.rotation.x() = ReadBinaryLittleEndian<double>(&file);
    cam_from_world.rotation.y() = ReadBinaryLittleEndian<double>(&file);
    cam_from_world.rotation.z() = ReadBinaryLittleEndian<double>(&file);
    cam_from_world.rotation.normalize();
    cam_from_world.translation.x() = ReadBinaryLittleEndian<double>(&file);
    cam_from_world.translation.y() = ReadBinaryLittleEndian<double>(&file);
    cam_from_world.translation.z() = ReadBinaryLittleEndian<double>(&file);

    image.SetCameraId(ReadBinaryLittleEndian<camera_t>(&file));

    char name_char;
    do {
      file.read(&name_char, 1);
      if (name_char != '\0') {
        image.Name() += name_char;
      }
    } while (name_char != '\0');

    const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&file);

    std::vector<Eigen::Vector2d> points2D;
    points2D.reserve(num_points2D);
    std::vector<point3D_t> point3D_ids;
    point3D_ids.reserve(num_points2D);
    for (size_t j = 0; j < num_points2D; ++j) {
      const double x = ReadBinaryLittleEndian<double>(&file);
      const double y = ReadBinaryLittleEndian<double>(&file);
      points2D.emplace_back(x, y);
      point3D_ids.push_back(ReadBinaryLittleEndian<point3D_t>(&file));
    }

    image.SetUp(Camera(image.CameraId()));
    image.SetPoints2D(points2D);

    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      if (point3D_ids[point2D_idx] != kInvalidPoint3DId) {
        image.SetPoint3DForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
      }
    }

    image.SetRegistered(true);
    reg_image_ids_.push_back(image.ImageId());

    images_.emplace(image.ImageId(), image);
  }
}

void Reconstruction::ReadPoints3DBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  const size_t num_points3D = ReadBinaryLittleEndian<uint64_t>(&file);
  for (size_t i = 0; i < num_points3D; ++i) {
    struct Point3D point3D;

    const point3D_t point3D_id = ReadBinaryLittleEndian<point3D_t>(&file);
    num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

    point3D.xyz(0) = ReadBinaryLittleEndian<double>(&file);
    point3D.xyz(1) = ReadBinaryLittleEndian<double>(&file);
    point3D.xyz(2) = ReadBinaryLittleEndian<double>(&file);
    point3D.color(0) = ReadBinaryLittleEndian<uint8_t>(&file);
    point3D.color(1) = ReadBinaryLittleEndian<uint8_t>(&file);
    point3D.color(2) = ReadBinaryLittleEndian<uint8_t>(&file);
    point3D.error = ReadBinaryLittleEndian<double>(&file);

    const size_t track_length = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t j = 0; j < track_length; ++j) {
      const image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
      const point2D_t point2D_idx = ReadBinaryLittleEndian<point2D_t>(&file);
      point3D.track.AddElement(image_id, point2D_idx);
    }
    point3D.track.Compress();

    points3D_.emplace(point3D_id, point3D);
  }
}

void Reconstruction::WriteCamerasText(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);

  file << "# Camera list with one line of data per camera:" << std::endl;
  file << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
  file << "# Number of cameras: " << cameras_.size() << std::endl;

  for (const auto& camera : cameras_) {
    std::ostringstream line;
    line.precision(17);

    line << camera.first << " ";
    line << camera.second.ModelName() << " ";
    line << camera.second.width << " ";
    line << camera.second.height << " ";

    for (const double param : camera.second.params) {
      line << param << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << line_string << std::endl;
  }
}

void Reconstruction::WriteImagesText(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);

  file << "# Image list with two lines of data per image:" << std::endl;
  file << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, "
          "NAME"
       << std::endl;
  file << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
  file << "# Number of images: " << reg_image_ids_.size()
       << ", mean observations per image: "
       << ComputeMeanObservationsPerRegImage() << std::endl;

  for (const auto& image : images_) {
    if (!image.second.IsRegistered()) {
      continue;
    }

    std::ostringstream line;
    line.precision(17);

    std::string line_string;

    line << image.first << " ";

    const Rigid3d& cam_from_world = image.second.CamFromWorld();
    line << cam_from_world.rotation.w() << " ";
    line << cam_from_world.rotation.x() << " ";
    line << cam_from_world.rotation.y() << " ";
    line << cam_from_world.rotation.z() << " ";
    line << cam_from_world.translation.x() << " ";
    line << cam_from_world.translation.y() << " ";
    line << cam_from_world.translation.z() << " ";

    line << image.second.CameraId() << " ";

    line << image.second.Name();

    file << line.str() << std::endl;

    line.str("");
    line.clear();

    for (const Point2D& point2D : image.second.Points2D()) {
      line << point2D.xy(0) << " ";
      line << point2D.xy(1) << " ";
      if (point2D.HasPoint3D()) {
        line << point2D.point3D_id << " ";
      } else {
        line << -1 << " ";
      }
    }
    line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);
    file << line_string << std::endl;
  }
}

void Reconstruction::WritePoints3DText(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  // Ensure that we don't loose any precision by storing in text.
  file.precision(17);

  file << "# 3D point list with one line of data per point:" << std::endl;
  file << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
          "TRACK[] as (IMAGE_ID, POINT2D_IDX)"
       << std::endl;
  file << "# Number of points: " << points3D_.size()
       << ", mean track length: " << ComputeMeanTrackLength() << std::endl;

  for (const auto& point3D : points3D_) {
    file << point3D.first << " ";
    file << point3D.second.xyz(0) << " ";
    file << point3D.second.xyz(1) << " ";
    file << point3D.second.xyz(2) << " ";
    file << static_cast<int>(point3D.second.color(0)) << " ";
    file << static_cast<int>(point3D.second.color(1)) << " ";
    file << static_cast<int>(point3D.second.color(2)) << " ";
    file << point3D.second.error << " ";

    std::ostringstream line;
    line.precision(17);

    for (const auto& track_el : point3D.second.track.Elements()) {
      line << track_el.image_id << " ";
      line << track_el.point2D_idx << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << line_string << std::endl;
  }
}

void Reconstruction::WriteCamerasBinary(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  WriteBinaryLittleEndian<uint64_t>(&file, cameras_.size());

  for (const auto& camera : cameras_) {
    WriteBinaryLittleEndian<camera_t>(&file, camera.first);
    WriteBinaryLittleEndian<int>(&file,
                                 static_cast<int>(camera.second.model_id));
    WriteBinaryLittleEndian<uint64_t>(&file, camera.second.width);
    WriteBinaryLittleEndian<uint64_t>(&file, camera.second.height);
    for (const double param : camera.second.params) {
      WriteBinaryLittleEndian<double>(&file, param);
    }
  }
}

void Reconstruction::WriteImagesBinary(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  WriteBinaryLittleEndian<uint64_t>(&file, reg_image_ids_.size());

  for (const auto& image : images_) {
    if (!image.second.IsRegistered()) {
      continue;
    }

    WriteBinaryLittleEndian<image_t>(&file, image.first);

    const Rigid3d& cam_from_world = image.second.CamFromWorld();
    WriteBinaryLittleEndian<double>(&file, cam_from_world.rotation.w());
    WriteBinaryLittleEndian<double>(&file, cam_from_world.rotation.x());
    WriteBinaryLittleEndian<double>(&file, cam_from_world.rotation.y());
    WriteBinaryLittleEndian<double>(&file, cam_from_world.rotation.z());
    WriteBinaryLittleEndian<double>(&file, cam_from_world.translation.x());
    WriteBinaryLittleEndian<double>(&file, cam_from_world.translation.y());
    WriteBinaryLittleEndian<double>(&file, cam_from_world.translation.z());

    WriteBinaryLittleEndian<camera_t>(&file, image.second.CameraId());

    const std::string name = image.second.Name() + '\0';
    file.write(name.c_str(), name.size());

    WriteBinaryLittleEndian<uint64_t>(&file, image.second.NumPoints2D());
    for (const Point2D& point2D : image.second.Points2D()) {
      WriteBinaryLittleEndian<double>(&file, point2D.xy(0));
      WriteBinaryLittleEndian<double>(&file, point2D.xy(1));
      WriteBinaryLittleEndian<point3D_t>(&file, point2D.point3D_id);
    }
  }
}

void Reconstruction::WritePoints3DBinary(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  WriteBinaryLittleEndian<uint64_t>(&file, points3D_.size());

  for (const auto& point3D : points3D_) {
    WriteBinaryLittleEndian<point3D_t>(&file, point3D.first);
    WriteBinaryLittleEndian<double>(&file, point3D.second.xyz(0));
    WriteBinaryLittleEndian<double>(&file, point3D.second.xyz(1));
    WriteBinaryLittleEndian<double>(&file, point3D.second.xyz(2));
    WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.color(0));
    WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.color(1));
    WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.color(2));
    WriteBinaryLittleEndian<double>(&file, point3D.second.error);

    WriteBinaryLittleEndian<uint64_t>(&file, point3D.second.track.Length());
    for (const auto& track_el : point3D.second.track.Elements()) {
      WriteBinaryLittleEndian<image_t>(&file, track_el.image_id);
      WriteBinaryLittleEndian<point2D_t>(&file, track_el.point2D_idx);
    }
  }
}

void Reconstruction::SetObservationAsTriangulated(
    const image_t image_id,
    const point2D_t point2D_idx,
    const bool is_continued_point3D) {
  if (correspondence_graph_ == nullptr) {
    return;
  }

  const class Image& image = Image(image_id);
  THROW_CHECK(image.IsRegistered());

  const Point2D& point2D = image.Point2D(point2D_idx);
  THROW_CHECK(point2D.HasPoint3D());

  const auto corr_range =
      correspondence_graph_->FindCorrespondences(image_id, point2D_idx);
  for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
    class Image& corr_image = Image(corr->image_id);
    const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
    corr_image.IncrementCorrespondenceHasPoint3D(corr->point2D_idx);
    // Update number of shared 3D points between image pairs and make sure to
    // only count the correspondences once (not twice forward and backward).
    if (point2D.point3D_id == corr_point2D.point3D_id &&
        (is_continued_point3D || image_id < corr->image_id)) {
      const image_pair_t pair_id =
          Database::ImagePairToPairId(image_id, corr->image_id);
      auto& stats = image_pair_stats_[pair_id];
      stats.num_tri_corrs += 1;
      THROW_CHECK_LE(stats.num_tri_corrs, stats.num_total_corrs)
          << "The correspondence graph must not contain duplicate matches: "
          << corr->image_id << " " << corr->point2D_idx;
    }
  }
}

void Reconstruction::ResetTriObservations(const image_t image_id,
                                          const point2D_t point2D_idx,
                                          const bool is_deleted_point3D) {
  if (correspondence_graph_ == nullptr) {
    return;
  }

  const class Image& image = Image(image_id);
  THROW_CHECK(image.IsRegistered());
  const Point2D& point2D = image.Point2D(point2D_idx);
  THROW_CHECK(point2D.HasPoint3D());

  const auto corr_range =
      correspondence_graph_->FindCorrespondences(image_id, point2D_idx);
  for (const auto* corr = corr_range.beg; corr < corr_range.end; ++corr) {
    class Image& corr_image = Image(corr->image_id);
    const Point2D& corr_point2D = corr_image.Point2D(corr->point2D_idx);
    corr_image.DecrementCorrespondenceHasPoint3D(corr->point2D_idx);
    // Update number of shared 3D points between image pairs and make sure to
    // only count the correspondences once (not twice forward and backward).
    if (point2D.point3D_id == corr_point2D.point3D_id &&
        (!is_deleted_point3D || image_id < corr->image_id)) {
      const image_pair_t pair_id =
          Database::ImagePairToPairId(image_id, corr->image_id);
      THROW_CHECK_GT(image_pair_stats_[pair_id].num_tri_corrs, 0)
          << "The scene graph graph must not contain duplicate matches";
      image_pair_stats_[pair_id].num_tri_corrs -= 1;
    }
  }
}

}  // namespace colmap
