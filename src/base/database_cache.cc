// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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

#include "database_cache.h"

#include <boost/format.hpp>

#include <iostream>

#include "util/timer.h"

namespace colmap {

DatabaseCache::DatabaseCache() {}

void DatabaseCache::AddCamera(const class Camera& camera) {
  CHECK(!ExistsCamera(camera.CameraId()));
  cameras_.emplace(camera.CameraId(), camera);
}

void DatabaseCache::AddImage(const class Image& image) {
  CHECK(!ExistsImage(image.ImageId()));
  images_.emplace(image.ImageId(), image);
  scene_graph_.AddImage(image.ImageId(), image.NumPoints2D());
}

void DatabaseCache::Load(const Database& database, const size_t min_num_matches,
                         const bool ignore_watermarks) {
  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;

  timer.Start();
  std::cout << "Loading cameras..." << std::flush;

  {
    const std::vector<class Camera> cameras = database.ReadAllCameras();
    cameras_.reserve(cameras.size());
    for (const class Camera& camera : cameras) {
      cameras_.emplace(camera.CameraId(), camera);
    }
  }

  std::cout << boost::format(" %d in %.3fs") % cameras_.size() %
                   timer.ElapsedSeconds()
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading matches..." << std::flush;

  const std::vector<std::pair<image_pair_t, TwoViewGeometry>> image_pairs =
      database.ReadAllInlierMatches();

  std::cout << boost::format(" %d in %.3fs") % image_pairs.size() %
                   timer.ElapsedSeconds()
            << std::endl;

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
  std::cout << "Loading images..." << std::flush;

  {
    const std::vector<class Image> images = database.ReadAllImages();

    // Collect all images that are connected in the scene graph.
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(images.size());
    for (const auto& image_pair : image_pairs) {
      if (UseInlierMatchesCheck(image_pair.second)) {
        image_t image_id1;
        image_t image_id2;
        Database::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        connected_image_ids.insert(image_id1);
        connected_image_ids.insert(image_id2);
      }
    }

    // Load images with correspondences and discard images without
    // correspondences, as those images are useless for SfM.
    images_.reserve(connected_image_ids.size());
    for (const class Image& image : images) {
      if (connected_image_ids.count(image.ImageId()) > 0) {
        images_.emplace(image.ImageId(), image);
        const FeatureKeypoints keypoints =
            database.ReadKeypoints(image.ImageId());
        const std::vector<Eigen::Vector2d> points =
            FeatureKeypointsToPointsVector(keypoints);
        images_[image.ImageId()].SetPoints2D(points);
      }
    }

    std::cout << boost::format(" %d in %.3fs (connected %d)") % images.size() %
                     timer.ElapsedSeconds() % connected_image_ids.size()
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build scene graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Building scene graph..." << std::flush;

  for (const auto& image : images_) {
    scene_graph_.AddImage(image.first, image.second.NumPoints2D());
  }

  size_t num_ignored_image_pairs = 0;
  for (const auto& image_pair : image_pairs) {
    if (UseInlierMatchesCheck(image_pair.second)) {
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
      scene_graph_.AddCorrespondences(image_id1, image_id2,
                                      image_pair.second.inlier_matches);
    } else {
      num_ignored_image_pairs += 1;
    }
  }

  scene_graph_.Finalize();

  // Set number of observations and correspondences per image.
  for (auto& image : images_) {
    image.second.SetNumObservations(
        scene_graph_.NumObservationsForImage(image.first));
    image.second.SetNumCorrespondences(
        scene_graph_.NumCorrespondencesForImage(image.first));
  }

  std::cout << boost::format(" in %.3fs (ignored %d)") %
                   timer.ElapsedSeconds() % num_ignored_image_pairs
            << std::endl;
}

}  // namespace colmap
