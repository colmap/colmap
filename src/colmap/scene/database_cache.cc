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

#include "colmap/scene/database_cache.h"

#include "colmap/util/string.h"
#include "colmap/util/timer.h"

#include <set>
#include <unordered_set>

namespace colmap {
namespace {

std::vector<Eigen::Vector2d> FeatureKeypointsToPointsVector(
    const FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points(keypoints.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
  }
  return points;
}

}  // namespace

std::shared_ptr<DatabaseCache> DatabaseCache::Create(
    const Database& database,
    const size_t min_num_matches,
    const bool ignore_watermarks,
    const std::unordered_set<std::string>& image_names) {
  auto cache = std::make_shared<DatabaseCache>();

  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;

  timer.Start();
  LOG(INFO) << "Loading cameras...";

  {
    std::vector<struct Camera> cameras = database.ReadAllCameras();
    cache->cameras_.reserve(cameras.size());
    for (auto& camera : cameras) {
      cache->cameras_.emplace(camera.camera_id, std::move(camera));
    }
  }

  LOG(INFO) << StringPrintf(
      " %d in %.3fs", cache->cameras_.size(), timer.ElapsedSeconds());

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Loading matches...";

  const std::vector<std::pair<image_pair_t, TwoViewGeometry>>
      two_view_geometries = database.ReadTwoViewGeometries();

  LOG(INFO) << StringPrintf(
      " %d in %.3fs", two_view_geometries.size(), timer.ElapsedSeconds());

  auto UseInlierMatchesCheck = [min_num_matches, ignore_watermarks](
                                   const TwoViewGeometry& two_view_geometry) {
    return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >=
               min_num_matches &&
           (!ignore_watermarks ||
            two_view_geometry.config != TwoViewGeometry::WATERMARK);
  };

  //////////////////////////////////////////////////////////////////////////////
  // Load images
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Loading images...";

  std::unordered_set<image_t> image_ids;

  {
    std::vector<class Image> images = database.ReadAllImages();
    const size_t num_images = images.size();

    // Determines for which images data should be loaded.
    if (image_names.empty()) {
      for (const auto& image : images) {
        image_ids.insert(image.ImageId());
      }
    } else {
      for (const auto& image : images) {
        if (image_names.count(image.Name()) > 0) {
          image_ids.insert(image.ImageId());
        }
      }
    }

    // Collect all images that are connected in the correspondence graph.
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(image_ids.size());
    for (const auto& [pair_id, two_view_geometry] : two_view_geometries) {
      if (UseInlierMatchesCheck(two_view_geometry)) {
        const auto [image_id1, image_id2] =
            Database::PairIdToImagePair(pair_id);
        if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
          connected_image_ids.insert(image_id1);
          connected_image_ids.insert(image_id2);
        }
      }
    }

    // Load images with correspondences and discard images without
    // correspondences, as those images are useless for SfM.
    cache->images_.reserve(connected_image_ids.size());
    for (auto& image : images) {
      const image_t image_id = image.ImageId();
      if (image_ids.count(image_id) > 0 &&
          connected_image_ids.count(image_id) > 0) {
        image.SetPoints2D(
            FeatureKeypointsToPointsVector(database.ReadKeypoints(image_id)));
        cache->images_.emplace(image_id, std::move(image));
      }
    }

    LOG(INFO) << StringPrintf(" %d in %.3fs (connected %d)",
                              num_images,
                              timer.ElapsedSeconds(),
                              connected_image_ids.size());
  }

  //////////////////////////////////////////////////////////////////////////////
  // Load pose priors
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Loading pose priors...";

  {
    cache->pose_priors_.reserve(database.NumPosePriors());

    for (const auto& [image_id, _] : cache->images_) {
      if (database.ExistsPosePrior(image_id)) {
        cache->pose_priors_.emplace(image_id, database.ReadPosePrior(image_id));
      }
    }

    LOG(INFO) << StringPrintf(
        " %d in %.3fs", cache->pose_priors_.size(), timer.ElapsedSeconds());
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build correspondence graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  LOG(INFO) << "Building correspondence graph...";

  cache->correspondence_graph_ = std::make_shared<class CorrespondenceGraph>();

  for (const auto& image : cache->images_) {
    cache->correspondence_graph_->AddImage(image.first,
                                           image.second.NumPoints2D());
  }

  size_t num_ignored_image_pairs = 0;
  for (const auto& [pair_id, two_view_geometry] : two_view_geometries) {
    if (UseInlierMatchesCheck(two_view_geometry)) {
      const auto [image_id1, image_id2] = Database::PairIdToImagePair(pair_id);
      if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
        cache->correspondence_graph_->AddCorrespondences(
            image_id1, image_id2, two_view_geometry.inlier_matches);
      } else {
        num_ignored_image_pairs += 1;
      }
    } else {
      num_ignored_image_pairs += 1;
    }
  }

  cache->correspondence_graph_->Finalize();

  LOG(INFO) << StringPrintf(" in %.3fs (ignored %d)",
                            timer.ElapsedSeconds(),
                            num_ignored_image_pairs);

  return cache;
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

bool DatabaseCache::SetupPosePriors() {
  LOG(INFO) << "Setting up prior positions...";

  Timer timer;
  timer.Start();

  if (NumPosePriors() == 0) {
    LOG(ERROR) << "No pose priors in database...";
    return false;
  }

  bool prior_is_gps = true;

  // Get sorted image ids for GPS to cartesian conversion
  std::set<image_t> image_ids_with_prior;
  for (const auto& [image_id, _] : pose_priors_) {
    image_ids_with_prior.insert(image_id);
  }

  // Get GPS priors
  std::vector<Eigen::Vector3d> v_gps_prior;
  v_gps_prior.reserve(NumPosePriors());

  for (const auto& image_id : image_ids_with_prior) {
    const struct PosePrior& pose_prior = PosePrior(image_id);
    if (pose_prior.coordinate_system != PosePrior::CoordinateSystem::WGS84) {
      prior_is_gps = false;
    } else {
      // Image with the lowest id is to be used as the origin for prior
      // position conversion
      v_gps_prior.push_back(pose_prior.position);
    }
  }

  // Convert geographic to cartesian
  if (prior_is_gps) {
    // GPS reference to be used for EllToENU conversion
    const double ref_lat = v_gps_prior[0][0];
    const double ref_lon = v_gps_prior[0][1];

    const GPSTransform gps_transform(GPSTransform::WGS84);
    const std::vector<Eigen::Vector3d> v_xyz_prior =
        gps_transform.EllToENU(v_gps_prior, ref_lat, ref_lon);

    auto xyz_prior_it = v_xyz_prior.begin();
    for (const auto& image_id : image_ids_with_prior) {
      struct PosePrior& pose_prior = PosePrior(image_id);
      pose_prior.position = *xyz_prior_it;
      pose_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
      xyz_prior_it++;
    }
  } else if (!prior_is_gps && !v_gps_prior.empty()) {
    LOG(ERROR)
        << "Database is mixing GPS & non-GPS prior positions... Aborting";
    return false;
  }

  timer.PrintMinutes();

  return true;
}

}  // namespace colmap
