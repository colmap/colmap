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

#include "base/reconstruction.h"

#include <fstream>

#include <boost/lexical_cast.hpp>

#include "base/pose.h"
#include "base/projection.h"
#include "base/similarity_transform.h"
#include "base/triangulation.h"
#include "estimators/similarity_transform.h"
#include "optim/loransac.h"
#include "util/bitmap.h"
#include "util/misc.h"

namespace colmap {

Reconstruction::Reconstruction()
    : scene_graph_(nullptr), num_added_points3D_(0) {}

std::unordered_set<point3D_t> Reconstruction::Point3DIds() const {
  std::unordered_set<point3D_t> point3D_ids;
  point3D_ids.reserve(points3D_.size());

  for (const auto& point3D : points3D_) {
    point3D_ids.insert(point3D.first);
  }

  return point3D_ids;
}

void Reconstruction::Load(const DatabaseCache& database_cache) {
  scene_graph_ = nullptr;

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
      CHECK_EQ(existing_image.Name(), image.second.Name());
      if (existing_image.NumPoints2D() == 0) {
        existing_image.SetPoints2D(image.second.Points2D());
      } else {
        CHECK_EQ(image.second.NumPoints2D(), existing_image.NumPoints2D());
      }
      existing_image.SetNumObservations(image.second.NumObservations());
      existing_image.SetNumCorrespondences(image.second.NumCorrespondences());
    } else {
      AddImage(image.second);
    }
  }

  // Add image pairs.
  for (const auto& image_pair :
       database_cache.SceneGraph().NumCorrespondencesBetweenImages()) {
    image_pairs_[image_pair.first] = std::make_pair(0, image_pair.second);
  }
}

void Reconstruction::SetUp(const SceneGraph* scene_graph) {
  CHECK_NOTNULL(scene_graph);
  for (auto& image : images_) {
    image.second.SetUp(Camera(image.second.CameraId()));
  }
  scene_graph_ = scene_graph;

  // If an existing model was loaded from disk and there were already images
  // registered previously, we need to set observations as triangulated.
  for (const auto image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      if (image.Point2D(point2D_idx).HasPoint3D()) {
        const bool kIsContinuedPoint3D = false;
        SetObservationAsTriangulated(image_id, point2D_idx,
                                     kIsContinuedPoint3D);
      }
    }
  }
}

void Reconstruction::TearDown() {
  scene_graph_ = nullptr;

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
    point3D.second.Track().Compress();
  }
}

void Reconstruction::AddCamera(const class Camera& camera) {
  CHECK(!ExistsCamera(camera.CameraId()));
  CHECK(camera.VerifyParams());
  cameras_.emplace(camera.CameraId(), camera);
}

void Reconstruction::AddImage(const class Image& image) {
  CHECK(!ExistsImage(image.ImageId()));
  images_[image.ImageId()] = image;
}

point3D_t Reconstruction::AddPoint3D(const Eigen::Vector3d& xyz,
                                     const Track& track) {
  const point3D_t point3D_id = ++num_added_points3D_;
  CHECK(!ExistsPoint3D(point3D_id));

  class Point3D& point3D = points3D_[point3D_id];

  point3D.SetXYZ(xyz);
  point3D.SetTrack(track);

  for (const auto& track_el : track.Elements()) {
    class Image& image = Image(track_el.image_id);
    CHECK(!image.Point2D(track_el.point2D_idx).HasPoint3D());
    image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
    CHECK_LE(image.NumPoints3D(), image.NumPoints2D());
  }

  const bool kIsContinuedPoint3D = false;

  for (const auto& track_el : track.Elements()) {
    SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx,
                                 kIsContinuedPoint3D);
  }

  return point3D_id;
}

void Reconstruction::AddObservation(const point3D_t point3D_id,
                                    const TrackElement& track_el) {
  class Image& image = Image(track_el.image_id);
  CHECK(!image.Point2D(track_el.point2D_idx).HasPoint3D());

  image.SetPoint3DForPoint2D(track_el.point2D_idx, point3D_id);
  CHECK_LE(image.NumPoints3D(), image.NumPoints2D());

  class Point3D& point3D = Point3D(point3D_id);
  point3D.Track().AddElement(track_el);

  const bool kIsContinuedPoint3D = true;
  SetObservationAsTriangulated(track_el.image_id, track_el.point2D_idx,
                               kIsContinuedPoint3D);
}

point3D_t Reconstruction::MergePoints3D(const point3D_t point3D_id1,
                                        const point3D_t point3D_id2) {
  const class Point3D& point3D1 = Point3D(point3D_id1);
  const class Point3D& point3D2 = Point3D(point3D_id2);

  const Eigen::Vector3d merged_xyz =
      (point3D1.Track().Length() * point3D1.XYZ() +
       point3D2.Track().Length() * point3D2.XYZ()) /
      (point3D1.Track().Length() + point3D2.Track().Length());
  const Eigen::Vector3d merged_rgb =
      (point3D1.Track().Length() * point3D1.Color().cast<double>() +
       point3D2.Track().Length() * point3D2.Color().cast<double>()) /
      (point3D1.Track().Length() + point3D2.Track().Length());

  Track merged_track;
  merged_track.Reserve(point3D1.Track().Length() + point3D2.Track().Length());
  merged_track.AddElements(point3D1.Track().Elements());
  merged_track.AddElements(point3D2.Track().Elements());

  DeletePoint3D(point3D_id1);
  DeletePoint3D(point3D_id2);

  const point3D_t merged_point3D_id = AddPoint3D(merged_xyz, merged_track);
  class Point3D& merged_point3D = Point3D(merged_point3D_id);
  merged_point3D.SetColor(merged_rgb.cast<uint8_t>());

  return merged_point3D_id;
}

void Reconstruction::DeletePoint3D(const point3D_t point3D_id) {
  // Note: Do not change order of these instructions, especially with respect to
  // `Reconstruction::ResetTriObservations`

  const class Track& track = Point3D(point3D_id).Track();

  const bool kIsDeletedPoint3D = true;

  for (const auto& track_el : track.Elements()) {
    ResetTriObservations(track_el.image_id, track_el.point2D_idx,
                         kIsDeletedPoint3D);
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
  const point3D_t point3D_id = image.Point2D(point2D_idx).Point3DId();
  class Point3D& point3D = Point3D(point3D_id);

  if (point3D.Track().Length() <= 2) {
    DeletePoint3D(point3D_id);
    return;
  }

  point3D.Track().DeleteElement(image_id, point2D_idx);

  const bool kIsDeletedPoint3D = false;
  ResetTriObservations(image_id, point2D_idx, kIsDeletedPoint3D);

  image.ResetPoint3DForPoint2D(point2D_idx);
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

  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    if (image.Point2D(point2D_idx).HasPoint3D()) {
      DeleteObservation(image_id, point2D_idx);
    }
  }

  image.SetRegistered(false);

  reg_image_ids_.erase(
      std::remove(reg_image_ids_.begin(), reg_image_ids_.end(), image_id),
      reg_image_ids_.end());
}

void Reconstruction::Normalize(const double extent, const double p0,
                               const double p1, const bool use_images) {
  CHECK_GT(extent, 0);
  CHECK_GE(p0, 0);
  CHECK_LE(p0, 1);
  CHECK_GE(p1, 0);
  CHECK_LE(p1, 1);
  CHECK_LE(p0, p1);

  if (use_images && reg_image_ids_.size() < 2) {
    return;
  }

  EIGEN_STL_UMAP(class Image*, Eigen::Vector3d) proj_centers;

  for (size_t i = 0; i < reg_image_ids_.size(); ++i) {
    class Image& image = Image(reg_image_ids_[i]);
    const Eigen::Vector3d proj_center = image.ProjectionCenter();
    proj_centers[&image] = proj_center;
  }

  // Coordinates of image centers or point locations.
  std::vector<float> coords_x;
  std::vector<float> coords_y;
  std::vector<float> coords_z;
  if (use_images) {
    coords_x.reserve(proj_centers.size());
    coords_y.reserve(proj_centers.size());
    coords_z.reserve(proj_centers.size());
    for (const auto& proj_center : proj_centers) {
      coords_x.push_back(static_cast<float>(proj_center.second(0)));
      coords_y.push_back(static_cast<float>(proj_center.second(1)));
      coords_z.push_back(static_cast<float>(proj_center.second(2)));
    }
  } else {
    coords_x.reserve(points3D_.size());
    coords_y.reserve(points3D_.size());
    coords_z.reserve(points3D_.size());
    for (const auto& point3D : points3D_) {
      coords_x.push_back(static_cast<float>(point3D.second.X()));
      coords_y.push_back(static_cast<float>(point3D.second.Y()));
      coords_z.push_back(static_cast<float>(point3D.second.Z()));
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

  // Calculate scale and translation, such that
  // translation is applied before scaling.
  const double old_extent = (bbox_max - bbox_min).norm();
  double scale;
  if (old_extent < std::numeric_limits<double>::epsilon()) {
    scale = 1;
  } else {
    scale = extent / old_extent;
  }

  const Eigen::Vector3d translation = mean_coord;

  // Transform images.
  for (auto& elem : proj_centers) {
    elem.second -= translation;
    elem.second *= scale;
    const Eigen::Quaterniond quat(elem.first->Qvec(0), elem.first->Qvec(1),
                                  elem.first->Qvec(2), elem.first->Qvec(3));
    elem.first->SetTvec(quat * -elem.second);
  }

  // Transform points.
  for (auto& point3D : points3D_) {
    point3D.second.XYZ() -= translation;
    point3D.second.XYZ() *= scale;
  }
}

void Reconstruction::Transform(const double scale, const Eigen::Vector4d& qvec,
                               const Eigen::Vector3d& tvec) {
  CHECK_GT(scale, 0);
  const SimilarityTransform3 tform(scale, qvec, tvec);
  for (auto& image : images_) {
    tform.TransformPose(&image.second.Qvec(), &image.second.Tvec());
  }
  for (auto& point3D : points3D_) {
    tform.TransformPoint(&point3D.second.XYZ());
  }
}

bool Reconstruction::Merge(const Reconstruction& reconstruction,
                           const int min_common_images) {
  CHECK_GE(min_common_images, 3);

  // Find common and missing images in the two reconstructions.

  std::set<image_t> common_image_ids;
  std::set<image_t> missing_image_ids;
  for (const auto& image_id : reconstruction.RegImageIds()) {
    if (ExistsImage(image_id)) {
      CHECK(IsImageRegistered(image_id))
          << "Make sure to tear down the reconstructions before merging";
      common_image_ids.insert(image_id);
    } else {
      missing_image_ids.insert(image_id);
    }
  }

  if (common_image_ids.size() < static_cast<size_t>(min_common_images)) {
    return false;
  }

  // Estimate the similarity transformation between the two reconstructions.

  std::vector<Eigen::Vector3d> src;
  src.reserve(common_image_ids.size());
  std::vector<Eigen::Vector3d> dst;
  dst.reserve(common_image_ids.size());
  for (const auto image_id : common_image_ids) {
    src.push_back(reconstruction.Image(image_id).ProjectionCenter());
    dst.push_back(Image(image_id).ProjectionCenter());
  }

  SimilarityTransform3 tform;
  tform.Estimate(src, dst);

  // Register the missing images in this reconstruction.

  for (const auto image_id : missing_image_ids) {
    auto reg_image = reconstruction.Image(image_id);
    reg_image.SetRegistered(false);
    AddImage(reg_image);
    RegisterImage(image_id);
    if (!ExistsCamera(reg_image.CameraId())) {
      AddCamera(reconstruction.Camera(reg_image.CameraId()));
    }
    auto& image = Image(image_id);
    tform.TransformPose(&image.Qvec(), &image.Tvec());
  }

  // Merge the two point clouds using the following two rules:
  //    - copy points to this reconstruction with non-conflicting tracks,
  //      i.e. points that do not have an already triangulated observation
  //      in this reconstruction.
  //    - merge tracks that are unambiguous, i.e. only merge points in the two
  //      reconstructions if they have a one-to-one mapping.
  // Note that in both cases no cheirality or reprojection test is performed.

  for (const auto& point3D : reconstruction.Points3D()) {
    Track new_track;
    Track old_track;
    std::set<point3D_t> old_point3D_ids;
    for (const auto& track_el : point3D.second.Track().Elements()) {
      if (common_image_ids.count(track_el.image_id) > 0) {
        const auto& point2D =
            Image(track_el.image_id).Point2D(track_el.point2D_idx);
        if (point2D.HasPoint3D()) {
          old_track.AddElement(track_el);
          old_point3D_ids.insert(point2D.Point3DId());
        } else {
          new_track.AddElement(track_el);
        }
      } else if (missing_image_ids.count(track_el.image_id) > 0) {
        Image(track_el.image_id).ResetPoint3DForPoint2D(track_el.point2D_idx);
        new_track.AddElement(track_el);
      }
    }

    const bool create_new_point = new_track.Length() >= 2;
    const bool merge_new_and_old_point =
        (new_track.Length() + old_track.Length()) >= 2 &&
        old_point3D_ids.size() == 1;
    if (create_new_point || merge_new_and_old_point) {
      Eigen::Vector3d xyz = point3D.second.XYZ();
      tform.TransformPoint(&xyz);
      const auto point3D_id = AddPoint3D(xyz, new_track);
      Point3D(point3D_id).SetColor(point3D.second.Color());
      if (old_point3D_ids.size() == 1) {
        MergePoints3D(point3D_id, *old_point3D_ids.begin());
      }
    }
  }

  return true;
}

bool Reconstruction::Align(const std::vector<std::string>& image_names,
                           const std::vector<Eigen::Vector3d>& locations,
                           const int min_common_images) {
  CHECK_GE(min_common_images, 3);
  CHECK_EQ(image_names.size(), locations.size());

  // Find out which images are contained in the reconstruction and get the
  // positions of their camera centers.
  std::set<image_t> common_image_ids;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < image_names.size(); ++i) {
    const class Image* image = FindImageWithName(image_names[i]);
    if (image == nullptr) {
      continue;
    }

    if (!IsImageRegistered(image->ImageId())) {
      continue;
    }

    // Ignore duplicate images.
    if (common_image_ids.count(image->ImageId()) > 0) {
      continue;
    }

    common_image_ids.insert(image->ImageId());
    src.push_back(image->ProjectionCenter());
    dst.push_back(locations[i]);
  }

  // Only compute the alignment if there are enough correspondences.
  if (common_image_ids.size() < static_cast<size_t>(min_common_images)) {
    return false;
  }

  // Estimate the similarity transformation between the two reconstructions.
  SimilarityTransform3 tform;
  tform.Estimate(src, dst);

  // Update the cameras and points using the estimated transform.
  for (auto& image : images_) {
    tform.TransformPose(&image.second.Qvec(), &image.second.Tvec());
  }
  for (auto& point3D : points3D_) {
    tform.TransformPoint(&point3D.second.XYZ());
  }

  return true;
}

bool Reconstruction::AlignRobust(const std::vector<std::string>& image_names,
                                 const std::vector<Eigen::Vector3d>& locations,
                                 const int min_common_images,
                                 const RANSACOptions& ransac_options) {
  CHECK_GE(min_common_images, 3);
  CHECK_EQ(image_names.size(), locations.size());

  // Find out which images are contained in the reconstruction and get the
  // positions of their camera centers.
  std::set<image_t> common_image_ids;
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (size_t i = 0; i < image_names.size(); ++i) {
    const class Image* image = FindImageWithName(image_names[i]);
    if (image == nullptr) {
      continue;
    }

    if (!IsImageRegistered(image->ImageId())) {
      continue;
    }

    // Ignore duplicate images.
    if (common_image_ids.count(image->ImageId()) > 0) {
      continue;
    }

    common_image_ids.insert(image->ImageId());
    src.push_back(image->ProjectionCenter());
    dst.push_back(locations[i]);
  }

  // Only compute the alignment if there are enough correspondences.
  if (common_image_ids.size() < static_cast<size_t>(min_common_images)) {
    return false;
  }

  LORANSAC<SimilarityTransformEstimator<3>, SimilarityTransformEstimator<3>>
      ransac(ransac_options);

  const auto report = ransac.Estimate(src, dst);

  if (report.support.num_inliers < static_cast<size_t>(min_common_images)) {
    return false;
  }

  SimilarityTransform3 tform(report.model);

  // Update the cameras and points using the estimated transform.
  for (auto& image : images_) {
    tform.TransformPose(&image.second.Qvec(), &image.second.Tvec());
  }
  for (auto& point3D : points3D_) {
    tform.TransformPoint(&point3D.second.XYZ());
  }

  return true;
}

const class Image* Reconstruction::FindImageWithName(
    const std::string& name) const {
  for (const auto& elem : images_) {
    if (elem.second.Name() == name) {
      return &elem.second;
    }
  }
  return nullptr;
}

size_t Reconstruction::FilterPoints3D(
    const double max_reproj_error, const double min_tri_angle,
    const std::unordered_set<point3D_t>& point3D_ids) {
  size_t num_filtered = 0;
  num_filtered +=
      FilterPoints3DWithLargeReprojectionError(max_reproj_error, point3D_ids);
  num_filtered +=
      FilterPoints3DWithSmallTriangulationAngle(min_tri_angle, point3D_ids);
  return num_filtered;
}

size_t Reconstruction::FilterPoints3DInImages(
    const double max_reproj_error, const double min_tri_angle,
    const std::unordered_set<image_t>& image_ids) {
  std::unordered_set<point3D_t> point3D_ids;
  for (const image_t image_id : image_ids) {
    const class Image& image = Image(image_id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids.insert(point2D.Point3DId());
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
    const Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        const class Point3D& point3D = Point3D(point2D.Point3DId());
        if (!HasPointPositiveDepth(proj_matrix, point3D.XYZ())) {
          DeleteObservation(image_id, point2D_idx);
          num_filtered += 1;
        }
      }
    }
  }
  return num_filtered;
}

std::vector<image_t> Reconstruction::FilterImages(
    const double min_focal_length_ratio, const double max_focal_length_ratio,
    const double max_extra_param) {
  std::vector<image_t> filtered_image_ids;
  for (const image_t image_id : RegImageIds()) {
    const class Image& image = Image(image_id);
    const class Camera& camera = Camera(image.CameraId());
    if (image.NumPoints3D() == 0) {
      filtered_image_ids.push_back(image_id);
    } else if (camera.HasBogusParams(min_focal_length_ratio,
                                     max_focal_length_ratio, max_extra_param)) {
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
      error_sum += point3D.second.Error();
      num_valid_errors += 1;
    }
  }

  if (num_valid_errors == 0) {
    return 0.0;
  } else {
    return error_sum / num_valid_errors;
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
    LOG(FATAL) << "cameras, images, points3D files do not exist at " << path;
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
  WriteCamerasText(JoinPaths(path, "cameras.txt"));
  WriteImagesText(JoinPaths(path, "images.txt"));
  WritePoints3DText(JoinPaths(path, "points3D.txt"));
}

void Reconstruction::WriteBinary(const std::string& path) const {
  WriteCamerasBinary(JoinPaths(path, "cameras.bin"));
  WriteImagesBinary(JoinPaths(path, "images.bin"));
  WritePoints3DBinary(JoinPaths(path, "points3D.bin"));
}

void Reconstruction::ImportPLY(const std::string& path) {
  points3D_.clear();

  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  std::string line;

  int X_index = -1;
  int Y_index = -1;
  int Z_index = -1;
  int R_index = -1;
  int G_index = -1;
  int B_index = -1;
  int X_byte_pos = -1;
  int Y_byte_pos = -1;
  int Z_byte_pos = -1;
  int R_byte_pos = -1;
  int G_byte_pos = -1;
  int B_byte_pos = -1;

  bool in_vertex_section = false;
  bool is_binary = false;
  bool is_little_endian = false;
  size_t num_bytes_per_line = 0;
  size_t num_vertices = 0;

  int index = 0;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    if (line == "end_header") {
      break;
    }

    if (line.size() >= 6 && line.substr(0, 6) == "format") {
      if (line == "format ascii 1.0") {
        is_binary = false;
      } else if (line == "format binary_little_endian 1.0") {
        is_binary = true;
        is_little_endian = true;
      } else if (line == "format binary_big_endian 1.0") {
        is_binary = true;
        is_little_endian = false;
      }
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 3 && line_elems[0] == "element") {
      in_vertex_section = false;
      if (line_elems[1] == "vertex") {
        num_vertices = boost::lexical_cast<size_t>(line_elems[2]);
        in_vertex_section = true;
      } else if (boost::lexical_cast<size_t>(line_elems[2]) > 0) {
        LOG(FATAL) << "Only vertex elements supported";
      }
    }

    if (!in_vertex_section) {
      continue;
    }

    if (line_elems.size() >= 3 && line_elems[0] == "property") {
      CHECK(line_elems[1] == "float" || line_elems[1] == "uchar")
          << "PLY import only supports the float and uchar data types";
      if (line == "property float x") {
        X_index = index;
        X_byte_pos = num_bytes_per_line;
      } else if (line == "property float y") {
        Y_index = index;
        Y_byte_pos = num_bytes_per_line;
      } else if (line == "property float z") {
        Z_index = index;
        Z_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar r" || line == "property uchar red" ||
                 line == "property uchar diffuse_red") {
        R_index = index;
        R_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar g" || line == "property uchar green" ||
                 line == "property uchar diffuse_green") {
        G_index = index;
        G_byte_pos = num_bytes_per_line;
      } else if (line == "property uchar b" || line == "property uchar blue" ||
                 line == "property uchar diffuse_blue") {
        B_index = index;
        B_byte_pos = num_bytes_per_line;
      }

      index += 1;
      if (line_elems[1] == "float") {
        num_bytes_per_line += 4;
      } else if (line_elems[1] == "uchar") {
        num_bytes_per_line += 1;
      } else {
        LOG(FATAL) << "Invalid data type: " << line_elems[1];
      }
    }
  }

  CHECK(X_index != -1 && Y_index != -1 && Z_index != -1 && R_index != -1 &&
        G_index != -1 && B_index != -1)
      << "Invalid PLY file format: Must specify x, y, z, and color";

  if (is_binary) {
    std::vector<char> buffer(num_bytes_per_line);
    for (size_t i = 0; i < num_vertices; ++i) {
      file.read(buffer.data(), num_bytes_per_line);

      Eigen::Vector3d xyz;
      Eigen::Vector3i rgb;
      if (is_little_endian) {
        xyz(0) = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[X_byte_pos]));
        xyz(1) = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[Y_byte_pos]));
        xyz(2) = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[Z_byte_pos]));

        rgb(0) = LittleEndianToNative(
            *reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
        rgb(1) = LittleEndianToNative(
            *reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
        rgb(2) = LittleEndianToNative(
            *reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
      } else {
        xyz(0) =
            BigEndianToNative(*reinterpret_cast<float*>(&buffer[X_byte_pos]));
        xyz(1) =
            BigEndianToNative(*reinterpret_cast<float*>(&buffer[Y_byte_pos]));
        xyz(2) =
            BigEndianToNative(*reinterpret_cast<float*>(&buffer[Z_byte_pos]));

        rgb(0) =
            BigEndianToNative(*reinterpret_cast<uint8_t*>(&buffer[R_byte_pos]));
        rgb(1) =
            BigEndianToNative(*reinterpret_cast<uint8_t*>(&buffer[G_byte_pos]));
        rgb(2) =
            BigEndianToNative(*reinterpret_cast<uint8_t*>(&buffer[B_byte_pos]));
      }

      const point3D_t point3D_id = AddPoint3D(xyz, Track());
      Point3D(point3D_id).SetColor(rgb.cast<uint8_t>());
    }
  } else {
    while (std::getline(file, line)) {
      StringTrim(&line);
      std::stringstream line_stream(line);

      std::string item;
      std::vector<std::string> items;
      while (!line_stream.eof()) {
        std::getline(line_stream, item, ' ');
        StringTrim(&item);
        items.push_back(item);
      }

      Eigen::Vector3d xyz;
      xyz(0) = boost::lexical_cast<double>(items.at(X_index));
      xyz(1) = boost::lexical_cast<double>(items.at(Y_index));
      xyz(2) = boost::lexical_cast<double>(items.at(Z_index));

      Eigen::Vector3i rgb;
      rgb(0) = boost::lexical_cast<int>(items.at(R_index));
      rgb(1) = boost::lexical_cast<int>(items.at(G_index));
      rgb(2) = boost::lexical_cast<int>(items.at(B_index));

      const point3D_t point3D_id = AddPoint3D(xyz, Track());
      Point3D(point3D_id).SetColor(rgb.cast<uint8_t>());
    }
  }
}

bool Reconstruction::ExportNVM(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.is_open()) << path;

  file << "NVM_V3" << std::endl << std::endl;

  file << reg_image_ids_.size() << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  for (const auto image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    const class Camera& camera = Camera(image.CameraId());

    if (camera.ModelId() != SimpleRadialCameraModel::model_id) {
      std::cout << "WARNING: NVM only supports `SIMPLE_RADIAL` camera model."
                << std::endl;
      return false;
    }

    const double f =
        camera.Params(SimpleRadialCameraModel::focal_length_idxs[0]);
    const double k =
        -1 * camera.Params(SimpleRadialCameraModel::extra_params_idxs[0]);
    const Eigen::Vector3d proj_center = image.ProjectionCenter();

    file << image.Name() << " ";
    file << f << " ";
    file << image.Qvec(0) << " ";
    file << image.Qvec(1) << " ";
    file << image.Qvec(2) << " ";
    file << image.Qvec(3) << " ";
    file << proj_center(0) << " ";
    file << proj_center(1) << " ";
    file << proj_center(2) << " ";
    file << k << " ";
    file << 0 << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  file << std::endl << points3D_.size() << std::endl;

  for (const auto& point3D : points3D_) {
    file << point3D.second.XYZ()(0) << " ";
    file << point3D.second.XYZ()(1) << " ";
    file << point3D.second.XYZ()(2) << " ";
    file << static_cast<int>(point3D.second.Color(0)) << " ";
    file << static_cast<int>(point3D.second.Color(1)) << " ";
    file << static_cast<int>(point3D.second.Color(2)) << " ";

    std::ostringstream line;

    std::unordered_set<image_t> image_ids;
    for (const auto& track_el : point3D.second.Track().Elements()) {
      // Make sure that each point only has a single observation per image,
      // since VisualSfM does not support with multiple observations.
      if (image_ids.count(track_el.image_id) == 0) {
        const class Image& image = Image(track_el.image_id);
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        line << image_id_to_idx_[track_el.image_id] << " ";
        line << track_el.point2D_idx << " ";
        line << point2D.X() << " ";
        line << point2D.Y() << " ";
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

bool Reconstruction::ExportBundler(const std::string& path,
                                   const std::string& list_path) const {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.is_open()) << path;

  std::ofstream list_file(list_path, std::ios::trunc);
  CHECK(list_file.is_open()) << list_path;

  file << "# Bundle file v0.3" << std::endl;

  file << reg_image_ids_.size() << " " << points3D_.size() << std::endl;

  std::unordered_map<image_t, size_t> image_id_to_idx_;
  size_t image_idx = 0;

  for (const image_t image_id : reg_image_ids_) {
    const class Image& image = Image(image_id);
    const class Camera& camera = Camera(image.CameraId());

    double f;
    double k1;
    double k2;
    if (camera.ModelId() == SimplePinholeCameraModel::model_id ||
        camera.ModelId() == PinholeCameraModel::model_id) {
      f = camera.MeanFocalLength();
      k1 = 0.0;
      k2 = 0.0;
    } else if (camera.ModelId() == SimpleRadialCameraModel::model_id) {
      f = camera.Params(SimpleRadialCameraModel::focal_length_idxs[0]);
      k1 = camera.Params(SimpleRadialCameraModel::extra_params_idxs[0]);
      k2 = 0.0;
    } else if (camera.ModelId() == RadialCameraModel::model_id) {
      f = camera.Params(RadialCameraModel::focal_length_idxs[0]);
      k1 = camera.Params(RadialCameraModel::extra_params_idxs[0]);
      k2 = camera.Params(RadialCameraModel::extra_params_idxs[1]);
    } else {
      std::cout << "WARNING: Bundler only supports `SIMPLE_RADIAL` and "
                   "`RADIAL` camera models."
                << std::endl;
      return false;
    }

    file << f << " " << k1 << " " << k2 << std::endl;

    const Eigen::Matrix3d R = image.RotationMatrix();
    file << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << std::endl;
    file << -R(1, 0) << " " << -R(1, 1) << " " << -R(1, 2) << std::endl;
    file << -R(2, 0) << " " << -R(2, 1) << " " << -R(2, 2) << std::endl;

    file << image.Tvec(0) << " ";
    file << -image.Tvec(1) << " ";
    file << -image.Tvec(2) << std::endl;

    list_file << image.Name() << std::endl;

    image_id_to_idx_[image_id] = image_idx;
    image_idx += 1;
  }

  for (const auto& point3D : points3D_) {
    file << point3D.second.XYZ()(0) << " ";
    file << point3D.second.XYZ()(1) << " ";
    file << point3D.second.XYZ()(2) << std::endl;

    file << static_cast<int>(point3D.second.Color(0)) << " ";
    file << static_cast<int>(point3D.second.Color(1)) << " ";
    file << static_cast<int>(point3D.second.Color(2)) << std::endl;

    std::ostringstream line;

    line << point3D.second.Track().Length() << " ";

    for (const auto& track_el : point3D.second.Track().Elements()) {
      const class Image& image = Image(track_el.image_id);
      const class Camera& camera = Camera(image.CameraId());

      // Bundler output assumes image coordinate system origin
      // in the lower left corner of the image with the center of
      // the lower left pixel being (0, 0). Our coordinate system
      // starts in the upper left corner with the center of the
      // upper left pixel being (0.5, 0.5).

      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      line << image_id_to_idx_.at(track_el.image_id) << " ";
      line << track_el.point2D_idx << " ";
      line << point2D.X() - camera.PrincipalPointX() << " ";
      line << camera.PrincipalPointY() - point2D.Y() << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << line_string << std::endl;
  }

  return true;
}

void Reconstruction::ExportPLY(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.is_open()) << path;

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "element vertex " << points3D_.size() << std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  file << "property uchar red" << std::endl;
  file << "property uchar green" << std::endl;
  file << "property uchar blue" << std::endl;
  file << "end_header" << std::endl;

  for (const auto& point3D : points3D_) {
    file << point3D.second.X() << " ";
    file << point3D.second.Y() << " ";
    file << point3D.second.Z() << " ";
    file << static_cast<int>(point3D.second.Color(0)) << " ";
    file << static_cast<int>(point3D.second.Color(1)) << " ";
    file << static_cast<int>(point3D.second.Color(2)) << std::endl;
  }

  file << std::endl;
}

void Reconstruction::ExportVRML(const std::string& images_path,
                                const std::string& points3D_path,
                                const double image_scale,
                                const Eigen::Vector3d& image_rgb) const {
  std::ofstream images_file(images_path, std::ios::trunc);
  CHECK(images_file.is_open()) << images_path;

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

    Eigen::Transform<double, 3, Eigen::Affine> transform;
    transform.matrix().topLeftCorner<3, 4>() =
        image.second.InverseProjectionMatrix();

    // Move camera base model to camera pose.
    for (size_t i = 0; i < points.size(); i++) {
      const Eigen::Vector3d point = transform * points[i];
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
  CHECK(points3D_file.is_open()) << points3D_path;

  points3D_file << "#VRML V2.0 utf8\n";
  points3D_file << "Background { skyColor [1.0 1.0 1.0] } \n";
  points3D_file << "Shape{ appearance Appearance {\n";
  points3D_file << " material Material {emissiveColor 1 1 1} }\n";
  points3D_file << " geometry PointSet {\n";
  points3D_file << " coord Coordinate {\n";
  points3D_file << "  point [\n";

  for (const auto& point3D : points3D_) {
    points3D_file << point3D.second.XYZ()(0) << ", ";
    points3D_file << point3D.second.XYZ()(1) << ", ";
    points3D_file << point3D.second.XYZ()(2) << std::endl;
  }

  points3D_file << " ] }\n";
  points3D_file << " color Color { color [\n";

  for (const auto& point3D : points3D_) {
    points3D_file << point3D.second.Color(0) / 255.0 << ", ";
    points3D_file << point3D.second.Color(1) / 255.0 << ", ";
    points3D_file << point3D.second.Color(2) / 255.0 << std::endl;
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
  for (const Point2D point2D : image.Points2D()) {
    if (point2D.HasPoint3D()) {
      class Point3D& point3D = Point3D(point2D.Point3DId());
      if (point3D.Color() == kBlackColor) {
        BitmapColor<float> color;
        if (bitmap.InterpolateBilinear(point2D.X(), point2D.Y(), &color)) {
          const BitmapColor<uint8_t> color_ub = color.Cast<uint8_t>();
          point3D.SetColor(
              Eigen::Vector3ub(color_ub.r, color_ub.g, color_ub.b));
        }
      }
    }
  }

  return true;
}

void Reconstruction::ExtractColorsForAllImages(const std::string& path) {
  EIGEN_STL_UMAP(point3D_t, Eigen::Vector3d) color_sums;
  std::unordered_map<point3D_t, size_t> color_counts;

  for (size_t i = 0; i < reg_image_ids_.size(); ++i) {
    const class Image& image = Image(reg_image_ids_[i]);
    const std::string image_path = JoinPaths(path, image.Name());

    Bitmap bitmap;
    if (!bitmap.Read(image_path)) {
      std::cout << StringPrintf("Could not read image %s at path %s.",
                                image.Name().c_str(), image_path.c_str())
                << std::endl;
      continue;
    }

    for (const Point2D point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        BitmapColor<float> color;
        if (bitmap.InterpolateBilinear(point2D.X(), point2D.Y(), &color)) {
          if (color_sums.count(point2D.Point3DId())) {
            Eigen::Vector3d& color_sum = color_sums[point2D.Point3DId()];
            color_sum(0) += color.r;
            color_sum(1) += color.g;
            color_sum(2) += color.b;
            color_counts[point2D.Point3DId()] += 1;
          } else {
            color_sums.emplace(point2D.Point3DId(),
                               Eigen::Vector3d(color.r, color.g, color.b));
            color_counts.emplace(point2D.Point3DId(), 1);
          }
        }
      }
    }
  }

  const Eigen::Vector3ub kBlackColor(0, 0, 0);
  for (auto& point3D : points3D_) {
    if (color_sums.count(point3D.first)) {
      Eigen::Vector3d color =
          color_sums[point3D.first] / color_counts[point3D.first];
      color.unaryExpr(std::ptr_fun<double, double>(std::round));
      point3D.second.SetColor(color.cast<uint8_t>());
    } else {
      point3D.second.SetColor(kBlackColor);
    }
  }
}

void Reconstruction::CreateImageDirs(const std::string& path) const {
  std::set<std::string> image_dirs;
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
    CreateDirIfNotExists(dir);
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
  EIGEN_STL_UMAP(image_t, Eigen::Vector3d) proj_centers;

  for (const auto point3D_id : point3D_ids) {
    if (!ExistsPoint3D(point3D_id)) {
      continue;
    }

    const class Point3D& point3D = Point3D(point3D_id);

    // Calculate triangulation angle for all pairwise combinations of image
    // poses in the track. Only delete point if none of the combinations
    // has a sufficient triangulation angle.
    bool keep_point = false;
    for (size_t i1 = 0; i1 < point3D.Track().Length(); ++i1) {
      const image_t image_id1 = point3D.Track().Element(i1).image_id;

      Eigen::Vector3d proj_center1;
      if (proj_centers.count(image_id1) == 0) {
        const class Image& image1 = Image(image_id1);
        proj_center1 = image1.ProjectionCenter();
        proj_centers.emplace(image_id1, proj_center1);
      } else {
        proj_center1 = proj_centers.at(image_id1);
      }

      for (size_t i2 = 0; i2 < i1; ++i2) {
        const image_t image_id2 = point3D.Track().Element(i2).image_id;
        const Eigen::Vector3d proj_center2 = proj_centers.at(image_id2);

        const double tri_angle = CalculateTriangulationAngle(
            proj_center1, proj_center2, point3D.XYZ());

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
  // Number of filtered points.
  size_t num_filtered = 0;

  // Cache for projection matrices.
  EIGEN_STL_UMAP(image_t, Eigen::Matrix3x4d) proj_matrices;

  for (const auto point3D_id : point3D_ids) {
    if (!ExistsPoint3D(point3D_id)) {
      continue;
    }

    class Point3D& point3D = Point3D(point3D_id);

    if (point3D.Track().Length() < 2) {
      DeletePoint3D(point3D_id);
      continue;
    }

    double reproj_error_sum = 0.0;

    std::vector<TrackElement> track_els_to_delete;

    for (const auto& track_el : point3D.Track().Elements()) {
      const class Image& image = Image(track_el.image_id);

      Eigen::Matrix3x4d proj_matrix;
      if (proj_matrices.count(track_el.image_id) == 0) {
        proj_matrix = image.ProjectionMatrix();
        proj_matrices[track_el.image_id] = proj_matrix;
      } else {
        proj_matrix = proj_matrices[track_el.image_id];
      }

      if (HasPointPositiveDepth(proj_matrix, point3D.XYZ())) {
        const class Camera& camera = Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        const double reproj_error = CalculateReprojectionError(
            point2D.XY(), point3D.XYZ(), proj_matrix, camera);
        if (reproj_error > max_reproj_error) {
          track_els_to_delete.push_back(track_el);
        } else {
          reproj_error_sum += reproj_error;
        }
      } else {
        track_els_to_delete.push_back(track_el);
      }
    }

    if (track_els_to_delete.size() == point3D.Track().Length() ||
        track_els_to_delete.size() == point3D.Track().Length() - 1) {
      num_filtered += point3D.Track().Length();
      DeletePoint3D(point3D_id);
    } else {
      num_filtered += track_els_to_delete.size();
      for (const auto& track_el : track_els_to_delete) {
        DeleteObservation(track_el.image_id, track_el.point2D_idx);
      }
      point3D.SetError(reproj_error_sum / point3D.Track().Length());
    }
  }

  return num_filtered;
}

void Reconstruction::ReadCamerasText(const std::string& path) {
  cameras_.clear();

  std::ifstream file(path);
  CHECK(file.is_open()) << path;

  std::string line;
  std::string item;

  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    class Camera camera;

    // ID
    std::getline(line_stream, item, ' ');
    StringTrim(&item);
    camera.SetCameraId(boost::lexical_cast<camera_t>(item));

    // MODEL
    std::getline(line_stream, item, ' ');
    StringTrim(&item);
    camera.SetModelIdFromName(item);

    // WIDTH
    std::getline(line_stream, item, ' ');
    StringTrim(&item);
    camera.SetWidth(boost::lexical_cast<size_t>(item));

    // HEIGHT
    std::getline(line_stream, item, ' ');
    StringTrim(&item);
    camera.SetHeight(boost::lexical_cast<size_t>(item));

    // PARAMS
    camera.Params().clear();
    while (!line_stream.eof()) {
      std::getline(line_stream, item, ' ');
      StringTrim(&item);
      camera.Params().push_back(boost::lexical_cast<double>(item));
    }

    CHECK(camera.VerifyParams());

    cameras_.emplace(camera.CameraId(), camera);
  }
}

void Reconstruction::ReadImagesText(const std::string& path) {
  images_.clear();

  std::ifstream file(path);
  CHECK(file.is_open()) << path;

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
    const image_t image_id = boost::lexical_cast<image_t>(item);

    class Image image;
    image.SetImageId(image_id);

    image.SetRegistered(true);
    reg_image_ids_.push_back(image_id);

    // QVEC (qw, qx, qy, qz)
    std::getline(line_stream1, item, ' ');
    image.Qvec(0) = boost::lexical_cast<double>(item);

    std::getline(line_stream1, item, ' ');
    image.Qvec(1) = boost::lexical_cast<double>(item);

    std::getline(line_stream1, item, ' ');
    image.Qvec(2) = boost::lexical_cast<double>(item);

    std::getline(line_stream1, item, ' ');
    image.Qvec(3) = boost::lexical_cast<double>(item);

    image.NormalizeQvec();

    // TVEC
    std::getline(line_stream1, item, ' ');
    image.Tvec(0) = boost::lexical_cast<double>(item);

    std::getline(line_stream1, item, ' ');
    image.Tvec(1) = boost::lexical_cast<double>(item);

    std::getline(line_stream1, item, ' ');
    image.Tvec(2) = boost::lexical_cast<double>(item);

    // CAMERA_ID
    std::getline(line_stream1, item, ' ');
    image.SetCameraId(boost::lexical_cast<camera_t>(item));

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
        point.x() = boost::lexical_cast<double>(item);

        std::getline(line_stream2, item, ' ');
        point.y() = boost::lexical_cast<double>(item);

        points2D.push_back(point);

        std::getline(line_stream2, item, ' ');
        if (item == "-1") {
          point3D_ids.push_back(kInvalidPoint3DId);
        } else {
          point3D_ids.push_back(boost::lexical_cast<point3D_t>(item));
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
  CHECK(file.is_open()) << path;

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
    const point3D_t point3D_id = boost::lexical_cast<point3D_t>(item);

    // Make sure, that we can add new 3D points after reading 3D points
    // without overwriting existing 3D points.
    num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

    class Point3D point3D;

    // XYZ
    std::getline(line_stream, item, ' ');
    point3D.XYZ(0) = boost::lexical_cast<double>(item);

    std::getline(line_stream, item, ' ');
    point3D.XYZ(1) = boost::lexical_cast<double>(item);

    std::getline(line_stream, item, ' ');
    point3D.XYZ(2) = boost::lexical_cast<double>(item);

    // Color
    std::getline(line_stream, item, ' ');
    point3D.Color(0) = static_cast<uint8_t>(boost::lexical_cast<int>(item));

    std::getline(line_stream, item, ' ');
    point3D.Color(1) = static_cast<uint8_t>(boost::lexical_cast<int>(item));

    std::getline(line_stream, item, ' ');
    point3D.Color(2) = static_cast<uint8_t>(boost::lexical_cast<int>(item));

    // ERROR
    std::getline(line_stream, item, ' ');
    point3D.SetError(boost::lexical_cast<double>(item));

    // TRACK
    while (!line_stream.eof()) {
      TrackElement track_el;

      std::getline(line_stream, item, ' ');
      StringTrim(&item);
      if (item.empty()) {
        break;
      }
      track_el.image_id = boost::lexical_cast<image_t>(item);

      std::getline(line_stream, item, ' ');
      track_el.point2D_idx = boost::lexical_cast<point2D_t>(item);

      point3D.Track().AddElement(track_el);
    }

    point3D.Track().Compress();

    points3D_.emplace(point3D_id, point3D);
  }
}

void Reconstruction::ReadCamerasBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
  for (size_t i = 0; i < num_cameras; ++i) {
    class Camera camera;
    camera.SetCameraId(ReadBinaryLittleEndian<camera_t>(&file));
    camera.SetModelId(ReadBinaryLittleEndian<int>(&file));
    camera.SetWidth(ReadBinaryLittleEndian<uint64_t>(&file));
    camera.SetHeight(ReadBinaryLittleEndian<uint64_t>(&file));
    ReadBinaryLittleEndian<double>(&file, &camera.Params());
    CHECK(camera.VerifyParams());
    cameras_.emplace(camera.CameraId(), camera);
  }
}

void Reconstruction::ReadImagesBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&file);
  for (size_t i = 0; i < num_reg_images; ++i) {
    class Image image;

    image.SetImageId(ReadBinaryLittleEndian<image_t>(&file));

    image.Qvec(0) = ReadBinaryLittleEndian<double>(&file);
    image.Qvec(1) = ReadBinaryLittleEndian<double>(&file);
    image.Qvec(2) = ReadBinaryLittleEndian<double>(&file);
    image.Qvec(3) = ReadBinaryLittleEndian<double>(&file);
    image.NormalizeQvec();

    image.Tvec(0) = ReadBinaryLittleEndian<double>(&file);
    image.Tvec(1) = ReadBinaryLittleEndian<double>(&file);
    image.Tvec(2) = ReadBinaryLittleEndian<double>(&file);

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
  CHECK(file.is_open()) << path;

  const size_t num_points3D = ReadBinaryLittleEndian<uint64_t>(&file);
  for (size_t i = 0; i < num_points3D; ++i) {
    class Point3D point3D;

    const point3D_t point3D_id = ReadBinaryLittleEndian<point3D_t>(&file);
    num_added_points3D_ = std::max(num_added_points3D_, point3D_id);

    point3D.XYZ()(0) = ReadBinaryLittleEndian<double>(&file);
    point3D.XYZ()(1) = ReadBinaryLittleEndian<double>(&file);
    point3D.XYZ()(2) = ReadBinaryLittleEndian<double>(&file);
    point3D.Color(0) = ReadBinaryLittleEndian<uint8_t>(&file);
    point3D.Color(1) = ReadBinaryLittleEndian<uint8_t>(&file);
    point3D.Color(2) = ReadBinaryLittleEndian<uint8_t>(&file);
    point3D.SetError(ReadBinaryLittleEndian<double>(&file));

    const size_t track_length = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t j = 0; j < track_length; ++j) {
      const image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
      const point2D_t point2D_idx = ReadBinaryLittleEndian<point2D_t>(&file);
      point3D.Track().AddElement(image_id, point2D_idx);
    }
    point3D.Track().Compress();

    points3D_.emplace(point3D_id, point3D);
  }
}

void Reconstruction::WriteCamerasText(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.is_open()) << path;

  file << "# Camera list with one line of data per camera:" << std::endl;
  file << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
  file << "# Number of cameras: " << cameras_.size() << std::endl;

  for (const auto& camera : cameras_) {
    std::ostringstream line;

    line << camera.first << " ";
    line << camera.second.ModelName() << " ";
    line << camera.second.Width() << " ";
    line << camera.second.Height() << " ";

    for (const double param : camera.second.Params()) {
      line << param << " ";
    }

    std::string line_string = line.str();
    line_string = line_string.substr(0, line_string.size() - 1);

    file << line_string << std::endl;
  }
}

void Reconstruction::WriteImagesText(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc);
  CHECK(file.is_open()) << path;

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
    std::string line_string;

    line << image.first << " ";

    // QVEC (qw, qx, qy, qz)
    const Eigen::Vector4d normalized_qvec =
        NormalizeQuaternion(image.second.Qvec());
    line << normalized_qvec(0) << " ";
    line << normalized_qvec(1) << " ";
    line << normalized_qvec(2) << " ";
    line << normalized_qvec(3) << " ";

    // TVEC
    line << image.second.Tvec(0) << " ";
    line << image.second.Tvec(1) << " ";
    line << image.second.Tvec(2) << " ";

    line << image.second.CameraId() << " ";

    line << image.second.Name();

    file << line.str() << std::endl;

    line.str("");
    line.clear();

    for (const Point2D& point2D : image.second.Points2D()) {
      line << point2D.X() << " ";
      line << point2D.Y() << " ";
      if (point2D.HasPoint3D()) {
        line << point2D.Point3DId() << " ";
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
  CHECK(file.is_open()) << path;

  file << "# 3D point list with one line of data per point:" << std::endl;
  file << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
          "TRACK[] as (IMAGE_ID, POINT2D_IDX)"
       << std::endl;
  file << "# Number of points: " << points3D_.size()
       << ", mean track length: " << ComputeMeanTrackLength() << std::endl;

  for (const auto& point3D : points3D_) {
    file << point3D.first << " ";
    file << point3D.second.XYZ()(0) << " ";
    file << point3D.second.XYZ()(1) << " ";
    file << point3D.second.XYZ()(2) << " ";
    file << static_cast<int>(point3D.second.Color(0)) << " ";
    file << static_cast<int>(point3D.second.Color(1)) << " ";
    file << static_cast<int>(point3D.second.Color(2)) << " ";
    file << point3D.second.Error() << " ";

    std::ostringstream line;

    for (const auto& track_el : point3D.second.Track().Elements()) {
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
  CHECK(file.is_open()) << path;

  WriteBinaryLittleEndian<uint64_t>(&file, cameras_.size());

  for (const auto& camera : cameras_) {
    WriteBinaryLittleEndian<camera_t>(&file, camera.first);
    WriteBinaryLittleEndian<int>(&file, camera.second.ModelId());
    WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Width());
    WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Height());
    for (const double param : camera.second.Params()) {
      WriteBinaryLittleEndian<double>(&file, param);
    }
  }
}

void Reconstruction::WriteImagesBinary(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  CHECK(file.is_open()) << path;

  WriteBinaryLittleEndian<uint64_t>(&file, reg_image_ids_.size());

  for (const auto& image : images_) {
    if (!image.second.IsRegistered()) {
      continue;
    }

    WriteBinaryLittleEndian<image_t>(&file, image.first);

    const Eigen::Vector4d normalized_qvec =
        NormalizeQuaternion(image.second.Qvec());
    WriteBinaryLittleEndian<double>(&file, normalized_qvec(0));
    WriteBinaryLittleEndian<double>(&file, normalized_qvec(1));
    WriteBinaryLittleEndian<double>(&file, normalized_qvec(2));
    WriteBinaryLittleEndian<double>(&file, normalized_qvec(3));

    WriteBinaryLittleEndian<double>(&file, image.second.Tvec(0));
    WriteBinaryLittleEndian<double>(&file, image.second.Tvec(1));
    WriteBinaryLittleEndian<double>(&file, image.second.Tvec(2));

    WriteBinaryLittleEndian<camera_t>(&file, image.second.CameraId());

    const std::string name = image.second.Name() + '\0';
    file.write(name.c_str(), name.size());

    WriteBinaryLittleEndian<uint64_t>(&file, image.second.NumPoints2D());
    for (const Point2D& point2D : image.second.Points2D()) {
      WriteBinaryLittleEndian<double>(&file, point2D.X());
      WriteBinaryLittleEndian<double>(&file, point2D.Y());
      WriteBinaryLittleEndian<point3D_t>(&file, point2D.Point3DId());
    }
  }
}

void Reconstruction::WritePoints3DBinary(const std::string& path) const {
  std::ofstream file(path, std::ios::trunc | std::ios::binary);
  CHECK(file.is_open()) << path;

  WriteBinaryLittleEndian<uint64_t>(&file, points3D_.size());

  for (const auto& point3D : points3D_) {
    WriteBinaryLittleEndian<point3D_t>(&file, point3D.first);
    WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(0));
    WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(1));
    WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(2));
    WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(0));
    WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(1));
    WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(2));
    WriteBinaryLittleEndian<double>(&file, point3D.second.Error());

    WriteBinaryLittleEndian<uint64_t>(&file, point3D.second.Track().Length());
    for (const auto& track_el : point3D.second.Track().Elements()) {
      WriteBinaryLittleEndian<image_t>(&file, track_el.image_id);
      WriteBinaryLittleEndian<point2D_t>(&file, track_el.point2D_idx);
    }
  }
}

void Reconstruction::SetObservationAsTriangulated(
    const image_t image_id, const point2D_t point2D_idx,
    const bool is_continued_point3D) {
  if (scene_graph_ == nullptr) {
    return;
  }

  const class Image& image = Image(image_id);
  const Point2D& point2D = image.Point2D(point2D_idx);
  const std::vector<SceneGraph::Correspondence>& corrs =
      scene_graph_->FindCorrespondences(image_id, point2D_idx);

  CHECK(image.IsRegistered());
  CHECK(point2D.HasPoint3D());

  for (const auto& corr : corrs) {
    class Image& corr_image = Image(corr.image_id);
    const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
    corr_image.IncrementCorrespondenceHasPoint3D(corr.point2D_idx);
    // Update number of shared 3D points between image pairs and make sure to
    // only count the correspondences once (not twice forward and backward).
    if (point2D.Point3DId() == corr_point2D.Point3DId() &&
        (is_continued_point3D || image_id < corr.image_id)) {
      const image_pair_t pair_id =
          Database::ImagePairToPairId(image_id, corr.image_id);
      image_pairs_[pair_id].first += 1;
      CHECK_LE(image_pairs_[pair_id].first, image_pairs_[pair_id].second)
          << "The scene graph graph must not contain duplicate matches";
    }
  }
}

void Reconstruction::ResetTriObservations(const image_t image_id,
                                          const point2D_t point2D_idx,
                                          const bool is_deleted_point3D) {
  if (scene_graph_ == nullptr) {
    return;
  }

  const class Image& image = Image(image_id);
  const Point2D& point2D = image.Point2D(point2D_idx);
  const std::vector<SceneGraph::Correspondence>& corrs =
      scene_graph_->FindCorrespondences(image_id, point2D_idx);

  CHECK(image.IsRegistered());
  CHECK(point2D.HasPoint3D());

  for (const auto& corr : corrs) {
    class Image& corr_image = Image(corr.image_id);
    const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
    corr_image.DecrementCorrespondenceHasPoint3D(corr.point2D_idx);
    // Update number of shared 3D points between image pairs and make sure to
    // only count the correspondences once (not twice forward and backward).
    if (point2D.Point3DId() == corr_point2D.Point3DId() &&
        (!is_deleted_point3D || image_id < corr.image_id)) {
      const image_pair_t pair_id =
          Database::ImagePairToPairId(image_id, corr.image_id);
      image_pairs_[pair_id].first -= 1;
      CHECK_GE(image_pairs_[pair_id].first, 0)
          << "The scene graph graph must not contain duplicate matches";
    }
  }
}

}  // namespace colmap
