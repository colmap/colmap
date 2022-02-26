// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "base/database_cache.h"

#include <unordered_set>

#include "feature/utils.h"
#include "util/string.h"
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
  correspondence_graph_.AddImage(image.ImageId(), image.NumPoints2D());
}

void DatabaseCache::Load(const Database& database, const size_t min_num_matches,
                         const bool ignore_watermarks,
                         const std::unordered_set<std::string>& image_names) {
  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;

  timer.Start();
  std::cout << "Loading cameras..." << std::flush;

  {
    const std::vector<class Camera> cameras = database.ReadAllCameras();
    cameras_.reserve(cameras.size());
    for (const auto& camera : cameras) {
      cameras_.emplace(camera.CameraId(), camera);
    }
  }

  std::cout << StringPrintf(" %d in %.3fs", cameras_.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading matches..." << std::flush;

  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  std::cout << StringPrintf(" %d in %.3fs", image_pair_ids.size(),
                            timer.ElapsedSeconds())
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

  std::unordered_set<image_t> image_ids;

  {
    const std::vector<class Image> images = database.ReadAllImages();

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
    for (size_t i = 0; i < image_pair_ids.size(); ++i) {
      if (UseInlierMatchesCheck(two_view_geometries[i])) {
        image_t image_id1;
        image_t image_id2;
        Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
        if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
          connected_image_ids.insert(image_id1);
          connected_image_ids.insert(image_id2);
        }
      }
    }

    // Load images with correspondences and discard images without
    // correspondences, as those images are useless for SfM.
    images_.reserve(connected_image_ids.size());
    for (const auto& image : images) {
      if (image_ids.count(image.ImageId()) > 0 &&
          connected_image_ids.count(image.ImageId()) > 0) {
        images_.emplace(image.ImageId(), image);
        const FeatureKeypoints keypoints =
            database.ReadKeypoints(image.ImageId());
        const std::vector<Eigen::Vector2d> points =
            FeatureKeypointsToPointsVector(keypoints);
        images_[image.ImageId()].SetPoints2D(points);
      }
    }

    std::cout << StringPrintf(" %d in %.3fs (connected %d)", images.size(),
                              timer.ElapsedSeconds(),
                              connected_image_ids.size())
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build correspondence graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Building correspondence graph..." << std::flush;

  for (const auto& image : images_) {
    correspondence_graph_.AddImage(image.first, image.second.NumPoints2D());
  }

  size_t num_ignored_image_pairs = 0;
  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    if (UseInlierMatchesCheck(two_view_geometries[i])) {
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
      if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
        correspondence_graph_.AddCorrespondences(
            image_id1, image_id2, two_view_geometries[i].inlier_matches);
      } else {
        num_ignored_image_pairs += 1;
      }
    } else {
      num_ignored_image_pairs += 1;
    }
  }

  correspondence_graph_.Finalize();

  // Set number of observations and correspondences per image.
  for (auto& image : images_) {
    image.second.SetNumObservations(
        correspondence_graph_.NumObservationsForImage(image.first));
    image.second.SetNumCorrespondences(
        correspondence_graph_.NumCorrespondencesForImage(image.first));
  }

  std::cout << StringPrintf(" in %.3fs (ignored %d)", timer.ElapsedSeconds(),
                            num_ignored_image_pairs)
            << std::endl;
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

}  // namespace colmap
