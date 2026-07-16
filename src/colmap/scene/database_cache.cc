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

#include "colmap/scene/database_cache.h"

#include "colmap/geometry/gps.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/string.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>

namespace colmap {
namespace {

// Weiszfeld's algorithm for the L1 geometric median of a point set. Used to
// pick a deterministic, outlier-robust ENU reference instead of an arbitrary
// (e.g. first) row.
Eigen::Vector3d GeometricMedian(const std::vector<Eigen::Vector3d>& points) {
  Eigen::Vector3d median = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& p : points) {
    median += p;
  }
  median /= static_cast<double>(points.size());

  constexpr int kMaxIters = 100;
  constexpr double kConvergenceTol = 1e-9;
  constexpr double kDegenerateDistTol = 1e-9;
  for (int iter = 0; iter < kMaxIters; ++iter) {
    Eigen::Vector3d numerator = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
    for (const Eigen::Vector3d& p : points) {
      const double dist = (p - median).norm();
      if (dist < kDegenerateDistTol) {
        // The current estimate coincides with a sample; skip it to avoid a
        // division by (near-)zero rather than perturbing the estimate.
        continue;
      }
      const double weight = 1.0 / dist;
      numerator += weight * p;
      weight_sum += weight;
    }
    if (weight_sum < kDegenerateDistTol) {
      break;
    }
    const Eigen::Vector3d next = numerator / weight_sum;
    const bool converged = (next - median).norm() < kConvergenceTol;
    median = next;
    if (converged) {
      break;
    }
  }
  return median;
}

bool UseInlierMatchesCheck(const DatabaseCache::Options& options,
                           int two_view_geometry_config,
                           size_t num_matches) {
  return num_matches >= options.min_num_matches &&
         (!options.ignore_watermarks ||
          two_view_geometry_config != TwoViewGeometry::WATERMARK);
};

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
  }
  return points;
}

}  // namespace

DatabaseCache::DatabaseCache()
    : correspondence_graph_(std::make_shared<class CorrespondenceGraph>()) {}

void DatabaseCache::Load(const Database& database, const Options& options) {
  const bool has_rigs = database.NumRigs() > 0;
  const bool has_frames = database.NumFrames() > 0;

  //////////////////////////////////////////////////////////////////////////////
  // Load rigs
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;

  timer.Start();
  LOG(INFO) << "Loading rigs...";

  {
    std::vector<class Rig> rigs = database.ReadAllRigs();
    rigs_.reserve(rigs.size());
    for (auto& rig : rigs) {
      rigs_.emplace(rig.RigId(), std::move(rig));
    }
  }

  LOG(INFO) << StringPrintf(
      " %d in %.3fs", rigs_.size(), timer.ElapsedSeconds());

  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Loading cameras...";

  {
    std::vector<struct Camera> cameras = database.ReadAllCameras();
    cameras_.reserve(cameras.size());
    for (auto& camera : cameras) {
      if (!has_rigs) {
        // For backwards compatibility with old databases from before having
        // support for rigs/frames, we create a rig for each camera.
        class Rig rig;
        rig.SetRigId(camera.camera_id);
        rig.AddRefSensor(camera.SensorId());
        rigs_.emplace(rig.RigId(), std::move(rig));
      }
      cameras_.emplace(camera.camera_id, std::move(camera));
    }
  }

  LOG(INFO) << StringPrintf(
      " %d in %.3fs", cameras_.size(), timer.ElapsedSeconds());

  //////////////////////////////////////////////////////////////////////////////
  // Load frames
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Loading frames...";

  {
    std::vector<class Frame> frames = database.ReadAllFrames();
    frames_.reserve(frames.size());
    for (auto& frame : frames) {
      frames_.emplace(frame.FrameId(), std::move(frame));
    }
  }

  LOG(INFO) << StringPrintf(
      " %d in %.3fs", frames_.size(), timer.ElapsedSeconds());

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Loading matches...";

  std::vector<std::pair<image_pair_t, TwoViewGeometry>> two_view_geometries =
      database.ReadTwoViewGeometries();

  LOG(INFO) << StringPrintf(
      " %d in %.3fs", two_view_geometries.size(), timer.ElapsedSeconds());

  //////////////////////////////////////////////////////////////////////////////
  // Load images
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Loading images...";

  FlatHashSet<frame_t> frame_ids;
  NodeHashMap<image_t, frame_t> image_to_frame_id;

  {
    std::vector<class Image> images = database.ReadAllImages();
    const size_t num_images = images.size();

    for (auto& image : images) {
      // For backwards compatibility with old databases from before having
      // support for rigs/frames, we create a frame for each image.
      if (has_frames) {
        THROW_CHECK(image.HasFrameId());
      } else {
        class Frame frame;
        frame.SetFrameId(image.ImageId());
        frame.SetRigId(image.CameraId());
        frame.AddDataId(image.DataId());
        image.SetFrameId(frame.FrameId());
        frames_.emplace(frame.FrameId(), std::move(frame));
      }

      image_to_frame_id.emplace(image.ImageId(), image.FrameId());
    }

    // Determines for which images data should be loaded.
    if (options.image_names.empty()) {
      for (const auto& image : images) {
        frame_ids.insert(image.FrameId());
      }
    } else {
      for (const auto& image : images) {
        if (options.image_names.count(image.Name()) > 0) {
          frame_ids.insert(image.FrameId());
        }
      }
    }

    // Collect all images that are connected in the correspondence graph.
    FlatHashSet<frame_t> connected_frame_ids;
    if (!options.load_all_images) {
      connected_frame_ids.reserve(frame_ids.size());
      for (const auto& [pair_id, two_view_geometry] : two_view_geometries) {
        if (UseInlierMatchesCheck(options,
                                  two_view_geometry.config,
                                  two_view_geometry.inlier_matches.size())) {
          const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
          const frame_t frame_id1 = image_to_frame_id.at(image_id1);
          const frame_t frame_id2 = image_to_frame_id.at(image_id2);
          if (frame_ids.count(frame_id1) > 0 &&
              frame_ids.count(frame_id2) > 0) {
            connected_frame_ids.insert(frame_id1);
            connected_frame_ids.insert(frame_id2);
          }
        }
      }
    }

    const FlatHashSet<frame_t>& load_frame_ids =
        options.load_all_images ? frame_ids : connected_frame_ids;

    // Remove frames that should not be loaded. Use erase(it++) rather than
    // it = erase(it) so the code is portable across hash map backends (some,
    // e.g. Abseil, return void from erase()); frames_ is node-based, so
    // advancing past the erased element first is safe.
    for (auto it = frames_.begin(); it != frames_.end();) {
      if (load_frame_ids.count(it->first) == 0) {
        frames_.erase(it++);
      } else {
        ++it;
      }
    }

    // Load images and their keypoints. When load_all_images is false, only
    // images with correspondences are loaded, as images without matches are
    // not useful for SfM. When load_all_images is true, all candidate images
    // are loaded so that their keypoints are populated (e.g., for
    // triangulation on an existing reconstruction).
    images_.reserve(load_frame_ids.size());
    for (auto& image : images) {
      if (load_frame_ids.count(image.FrameId()) == 0) {
        continue;
      }

      const image_t image_id = image.ImageId();
      image.SetPoints2D(
          FeatureKeypointsToPointsVector(database.ReadKeypoints(image_id)));
      images_.emplace(image_id, std::move(image));
    }

    if (options.load_all_images) {
      LOG(INFO) << StringPrintf(" %d in %.3fs (loaded all %d)",
                                num_images,
                                timer.ElapsedSeconds(),
                                images_.size());
    } else {
      LOG(INFO) << StringPrintf(" %d in %.3fs (connected %d, loaded %d)",
                                num_images,
                                timer.ElapsedSeconds(),
                                connected_frame_ids.size(),
                                images_.size());
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Load pose priors
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();

  LOG(INFO) << "Loading pose priors...";

  pose_priors_ = database.ReadAllPosePriors();

  if (options.convert_pose_priors_to_enu) {
    ConvertPosePriorsToENU();
  }

  LOG(INFO) << StringPrintf(
      " %d in %.3fs", pose_priors_.size(), timer.ElapsedSeconds());

  //////////////////////////////////////////////////////////////////////////////
  // Build correspondence graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Building correspondence graph...";

  correspondence_graph_ = std::make_shared<class CorrespondenceGraph>();

  for (const auto& [image_id, image] : images_) {
    correspondence_graph_->AddImage(image_id, image.NumPoints2D());
  }

  size_t num_ignored_image_pairs = 0;
  for (auto& [pair_id, two_view_geometry] : two_view_geometries) {
    if (UseInlierMatchesCheck(options,
                              two_view_geometry.config,
                              two_view_geometry.inlier_matches.size())) {
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      const frame_t frame_id1 = image_to_frame_id.at(image_id1);
      const frame_t frame_id2 = image_to_frame_id.at(image_id2);
      if (frame_ids.count(frame_id1) > 0 && frame_ids.count(frame_id2) > 0) {
        correspondence_graph_->AddTwoViewGeometry(
            image_id1, image_id2, std::move(two_view_geometry));
      } else {
        num_ignored_image_pairs += 1;
      }
    } else {
      num_ignored_image_pairs += 1;
    }
  }

  correspondence_graph_->Finalize();

  LOG(INFO) << StringPrintf(" in %.3fs (ignored %d)",
                            timer.ElapsedSeconds(),
                            num_ignored_image_pairs);
}

std::shared_ptr<DatabaseCache> DatabaseCache::Create(const Database& database,
                                                     const Options& options) {
  auto cache = std::make_shared<DatabaseCache>();
  cache->Load(database, options);
  return cache;
}

std::shared_ptr<DatabaseCache> DatabaseCache::CreateFromCache(
    const DatabaseCache& database_cache, const Options& options) {
  auto cache = std::make_shared<DatabaseCache>();

  // Collect candidate image ids matching the name filter.
  // Empty image_names means use all images.
  FlatHashSet<image_t> candidate_image_ids;
  for (const auto& [image_id, image] : database_cache.Images()) {
    if (options.image_names.empty() ||
        options.image_names.count(image.Name()) > 0) {
      candidate_image_ids.insert(image_id);
    }
  }

  const auto& source_graph = database_cache.CorrespondenceGraph();

  FlatHashSet<image_t> connected_image_ids;
  if (!options.load_all_images) {
    for (const auto& [pair_id, num_matches] :
         source_graph->NumMatchesBetweenAllImages()) {
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      if (candidate_image_ids.count(image_id1) == 0 ||
          candidate_image_ids.count(image_id2) == 0) {
        continue;
      }
      const TwoViewGeometry two_view_geometry =
          source_graph->ExtractTwoViewGeometry(
              image_id1, image_id2, /*extract_inlier_matches=*/false);
      if (!UseInlierMatchesCheck(
              options, two_view_geometry.config, num_matches)) {
        continue;
      }
      connected_image_ids.insert(image_id1);
      connected_image_ids.insert(image_id2);
    }
  }

  const FlatHashSet<image_t>& load_image_ids =
      options.load_all_images ? candidate_image_ids : connected_image_ids;

  // Collect frame ids for images to load.
  FlatHashSet<frame_t> filtered_frame_ids;
  for (const image_t image_id : load_image_ids) {
    const auto& image = database_cache.Image(image_id);
    filtered_frame_ids.insert(image.FrameId());
  }

  // Copy all images of filtered frames (not just the images matching the
  // name filter). This is needed for multi-camera rigs where the generalized
  // pose solver needs all images of a frame.
  FlatHashSet<camera_t> filtered_camera_ids;
  for (const auto& [image_id, image] : database_cache.Images()) {
    if (filtered_frame_ids.count(image.FrameId()) > 0) {
      cache->images_.emplace(image_id, image);
      filtered_camera_ids.insert(image.CameraId());
    }
  }

  // Copy filtered frames and collect rig ids.
  FlatHashSet<rig_t> filtered_rig_ids;
  for (const auto& [frame_id, frame] : database_cache.Frames()) {
    if (filtered_frame_ids.count(frame_id) > 0) {
      cache->frames_.emplace(frame_id, frame);
      filtered_rig_ids.insert(frame.RigId());
    }
  }

  // Copy filtered cameras.
  for (const auto& [camera_id, camera] : database_cache.Cameras()) {
    if (filtered_camera_ids.count(camera_id) > 0) {
      cache->cameras_.emplace(camera_id, camera);
    }
  }

  // Copy filtered rigs.
  for (const auto& [rig_id, rig] : database_cache.Rigs()) {
    if (filtered_rig_ids.count(rig_id) > 0) {
      cache->rigs_.emplace(rig_id, rig);
    }
  }

  // Copy pose priors.
  cache->pose_priors_ = database_cache.PosePriors();
  if (options.convert_pose_priors_to_enu) {
    cache->ConvertPosePriorsToENU();
  }

  // Build filtered correspondence graph with all images from connected frames.
  cache->correspondence_graph_ = std::make_shared<class CorrespondenceGraph>();

  for (const auto& [image_id, image] : cache->images_) {
    cache->correspondence_graph_->AddImage(image_id, image.NumPoints2D());
  }

  // Copy correspondences between all image pairs in the cache.
  for (const image_pair_t pair_id : source_graph->ImagePairs()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (cache->images_.count(image_id1) > 0 &&
        cache->images_.count(image_id2) > 0) {
      cache->correspondence_graph_->AddTwoViewGeometry(
          image_id1,
          image_id2,
          source_graph->ExtractTwoViewGeometry(
              image_id1, image_id2, /*extract_inlier_matches=*/true));
    }
  }

  cache->correspondence_graph_->Finalize();

  return cache;
}

void DatabaseCache::AddRig(class Rig rig) {
  const rig_t rig_id = rig.RigId();
  THROW_CHECK(!ExistsRig(rig_id));
  rigs_.emplace(rig_id, std::move(rig));
}

void DatabaseCache::AddCamera(struct Camera camera) {
  const camera_t camera_id = camera.camera_id;
  THROW_CHECK(!ExistsCamera(camera_id));
  cameras_.emplace(camera_id, std::move(camera));
}

void DatabaseCache::AddFrame(class Frame frame) {
  const rig_t frame_id = frame.FrameId();
  THROW_CHECK(!ExistsFrame(frame_id));
  frames_.emplace(frame_id, std::move(frame));
}

void DatabaseCache::AddImage(class Image image) {
  const image_t image_id = image.ImageId();
  THROW_CHECK(!ExistsImage(image_id));
  correspondence_graph_->AddImage(image_id, image.NumPoints2D());
  images_.emplace(image_id, std::move(image));
}

void DatabaseCache::AddPosePrior(struct PosePrior pose_prior) {
  pose_priors_.push_back(std::move(pose_prior));
}

const class Image* DatabaseCache::FindImageWithName(
    const std::string& name) const {
  for (const auto& image : images_) {
    if (image.second.Name() == name) {
      return &image.second;
    }
  }
  return nullptr;
}

void DatabaseCache::ConvertPosePriorsToENU() {
  bool has_wgs84 = false;
  bool has_cartesian = false;
  for (const auto& pose_prior : pose_priors_) {
    if (pose_prior.coordinate_system == PosePrior::CoordinateSystem::WGS84) {
      has_wgs84 = true;
    } else if (pose_prior.coordinate_system ==
              PosePrior::CoordinateSystem::CARTESIAN) {
      has_cartesian = true;
    }
  }

  if (!has_wgs84) {
    // Nothing to convert: either already Cartesian, or no priors at all.
    return;
  }
  THROW_CHECK(!has_cartesian)
      << "Cannot convert a mixture of WGS84 and Cartesian pose priors to a "
        "shared ENU frame";

  const GPSTransform gps_transform(GPSTransform::Ellipsoid::WGS84);

  // Deterministic reference selection: the geometric median (in ECEF) of all
  // rows with a finite full position, so the reference is conditioned by
  // every usable row rather than being the first row (which may be an
  // outlier). If no row has a finite altitude, fall back to the median of
  // whatever finite lat/lon rows exist and use altitude 0 purely as an
  // internal tangent-plane origin, never as a fabricated measurement.
  std::vector<Eigen::Vector3d> full_position_lla;
  std::vector<double> full_altitudes;
  std::vector<Eigen::Vector3d> horizontal_only_lla;
  for (const auto& pose_prior : pose_priors_) {
    const Eigen::Vector3d& p = pose_prior.position;
    const bool has_lat_lon = std::isfinite(p.x()) && std::isfinite(p.y());
    if (!has_lat_lon) {
      continue;
    }
    if (std::isfinite(p.z())) {
      full_position_lla.push_back(p);
      full_altitudes.push_back(p.z());
    } else {
      horizontal_only_lla.emplace_back(p.x(), p.y(), 0.0);
    }
  }

  if (full_position_lla.empty() && horizontal_only_lla.empty()) {
    // No row carries any usable position; leave the archive's Cartesian tag
    // change to still happen below (gravity/rotation groups may still be
    // present and require no reference), but there is no ENU origin to
    // report.
    for (auto& pose_prior : pose_priors_) {
      if (pose_prior.coordinate_system == PosePrior::CoordinateSystem::WGS84) {
        pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
      }
    }
    return;
  }

  double ref_lat;
  double ref_lon;
  double ref_alt;
  const bool ref_alt_is_real = !full_position_lla.empty();
  if (ref_alt_is_real) {
    const std::vector<Eigen::Vector3d> full_position_ecef =
        gps_transform.EllipsoidToECEF(full_position_lla);
    const Eigen::Vector3d median_ecef = GeometricMedian(full_position_ecef);
    const Eigen::Vector3d median_lla =
        gps_transform.ECEFToEllipsoid({median_ecef})[0];
    ref_lat = median_lla.x();
    ref_lon = median_lla.y();
    std::vector<double> sorted_altitudes = full_altitudes;
    std::sort(sorted_altitudes.begin(), sorted_altitudes.end());
    ref_alt = sorted_altitudes[sorted_altitudes.size() / 2];
  } else {
    const std::vector<Eigen::Vector3d> horizontal_ecef =
        gps_transform.EllipsoidToECEF(horizontal_only_lla);
    const Eigen::Vector3d median_ecef = GeometricMedian(horizontal_ecef);
    const Eigen::Vector3d median_lla =
        gps_transform.ECEFToEllipsoid({median_ecef})[0];
    ref_lat = median_lla.x();
    ref_lon = median_lla.y();
    ref_alt = 0.0;
  }

  const Eigen::Vector3d ref_ecef =
      gps_transform.EllipsoidToECEF({Eigen::Vector3d(ref_lat, ref_lon, ref_alt)})[0];
  const Eigen::Matrix3d shared_from_ecef =
      GPSTransform::ENUFromECEF(ref_lat, ref_lon);

  pose_prior_enu_origin_ = Eigen::Vector3d(ref_lat, ref_lon, ref_alt);

  for (auto& pose_prior : pose_priors_) {
    if (pose_prior.coordinate_system != PosePrior::CoordinateSystem::WGS84) {
      continue;
    }

    const Eigen::Vector3d& lla = pose_prior.position;
    const bool has_lat_lon = std::isfinite(lla.x()) && std::isfinite(lla.y());
    const bool has_full_position = has_lat_lon && std::isfinite(lla.z());

    Eigen::Matrix3d local_from_ecef;
    if (has_lat_lon) {
      local_from_ecef = GPSTransform::ENUFromECEF(lla.x(), lla.y());
    }

    if (has_full_position) {
      const Eigen::Vector3d ecef = gps_transform.EllipsoidToECEF({lla})[0];
      pose_prior.position = shared_from_ecef * (ecef - ref_ecef);
    } else if (has_lat_lon) {
      // Horizontal-only row: use the shared reference altitude only to
      // compute East/North, then set Up back to NaN so no altitude is
      // fabricated for this row.
      const Eigen::Vector3d ecef =
          gps_transform.EllipsoidToECEF({Eigen::Vector3d(
              lla.x(), lla.y(), ref_alt)})[0];
      Eigen::Vector3d enu = shared_from_ecef * (ecef - ref_ecef);
      enu.z() = PosePrior::kNaN;
      pose_prior.position = enu;
    }
    // Else: no position for this row (e.g. gravity-only); position stays NaN.

    if (has_lat_lon) {
      // shared_from_local = shared_from_ecef * ecef_from_local, and
      // ecef_from_local = local_from_ecef^T.
      const Eigen::Matrix3d shared_from_local =
          shared_from_ecef * local_from_ecef.transpose();

      if (pose_prior.HasPositionCov()) {
        pose_prior.position_covariance =
            shared_from_local * pose_prior.position_covariance *
            shared_from_local.transpose();
      }

      if (pose_prior.HasRotation()) {
        // local_from_shared = local_from_ecef * ecef_from_shared, and
        // ecef_from_shared = shared_from_ecef^T.
        const Eigen::Matrix3d local_from_shared =
            local_from_ecef * shared_from_ecef.transpose();
        pose_prior.rotation =
            (pose_prior.rotation * Eigen::Quaterniond(local_from_shared))
                .normalized();

        if (pose_prior.HasRotationCov()) {
          pose_prior.rotation_covariance =
              shared_from_local * pose_prior.rotation_covariance *
              shared_from_local.transpose();
        }
      }
    }

    // Gravity is down in sensor coordinates and is unaffected by a world-
    // frame change.

    // A gravity-only row has no coordinate-bearing measurement to transform,
    // but retagging its cache-only coordinate system to CARTESIAN loses no
    // information (its sensor-frame gravity remains unchanged and usable)
    // and avoids leaving a mixed-coordinate-system cache. The database row
    // itself remains WGS84.
    pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
  }
}

}  // namespace colmap
