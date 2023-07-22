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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#pragma once

#include "colmap/base/camera.h"
#include "colmap/base/correspondence_graph.h"
#include "colmap/base/database.h"
#include "colmap/base/image.h"
#include "colmap/camera/models.h"
#include "colmap/util/types.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// A class that caches the contents of the database in memory, used to quickly
// create new reconstruction instances when multiple models are reconstructed.
class DatabaseCache {
 public:
  // Load cameras, images, features, and matches from database.
  //
  // @param database              Source database from which to load data.
  // @param min_num_matches       Only load image pairs with a minimum number
  //                              of matches.
  // @param ignore_watermarks     Whether to ignore watermark image pairs.
  // @param image_names           Whether to use only load the data for a subset
  //                              of the images. All images are used if empty.
  static std::shared_ptr<DatabaseCache> Create(
      const Database& database,
      size_t min_num_matches,
      bool ignore_watermarks,
      const std::unordered_set<std::string>& image_names);

  // Get number of objects.
  inline size_t NumCameras() const;
  inline size_t NumImages() const;

  // Get specific objects.
  inline class Camera& Camera(camera_t camera_id);
  inline const class Camera& Camera(camera_t camera_id) const;
  inline class Image& Image(image_t image_id);
  inline const class Image& Image(image_t image_id) const;

  // Get all objects.
  inline const std::unordered_map<camera_t, class Camera>& Cameras() const;
  inline const std::unordered_map<image_t, class Image>& Images() const;

  // Check whether specific object exists.
  inline bool ExistsCamera(camera_t camera_id) const;
  inline bool ExistsImage(image_t image_id) const;

  // Get reference to const correspondence graph.
  inline std::shared_ptr<const class CorrespondenceGraph> CorrespondenceGraph()
      const;

  // Find specific image by name. Note that this uses linear search.
  const class Image* FindImageWithName(const std::string& name) const;

 private:
  std::shared_ptr<class CorrespondenceGraph> correspondence_graph_;

  std::unordered_map<camera_t, class Camera> cameras_;
  std::unordered_map<image_t, class Image> images_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t DatabaseCache::NumCameras() const { return cameras_.size(); }
size_t DatabaseCache::NumImages() const { return images_.size(); }

class Camera& DatabaseCache::Camera(const camera_t camera_id) {
  return cameras_.at(camera_id);
}

const class Camera& DatabaseCache::Camera(const camera_t camera_id) const {
  return cameras_.at(camera_id);
}

class Image& DatabaseCache::Image(const image_t image_id) {
  return images_.at(image_id);
}

const class Image& DatabaseCache::Image(const image_t image_id) const {
  return images_.at(image_id);
}

const std::unordered_map<camera_t, class Camera>& DatabaseCache::Cameras()
    const {
  return cameras_;
}

const std::unordered_map<image_t, class Image>& DatabaseCache::Images() const {
  return images_;
}

bool DatabaseCache::ExistsCamera(const camera_t camera_id) const {
  return cameras_.find(camera_id) != cameras_.end();
}

bool DatabaseCache::ExistsImage(const image_t image_id) const {
  return images_.find(image_id) != images_.end();
}

std::shared_ptr<const class CorrespondenceGraph>
DatabaseCache::CorrespondenceGraph() const {
  return correspondence_graph_;
}

}  // namespace colmap
