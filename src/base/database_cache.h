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

#ifndef COLMAP_SRC_BASE_DATABASE_CACHE_H_
#define COLMAP_SRC_BASE_DATABASE_CACHE_H_

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/camera_models.h"
#include "base/database.h"
#include "base/image.h"
#include "base/scene_graph.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// A class that caches the contents of the database in memory, used to quickly
// create new reconstruction instances when multiple models are reconstructed.
class DatabaseCache {
 public:
  DatabaseCache();

  // Get number of objects.
  inline size_t NumCameras() const;
  inline size_t NumImages() const;

  // Get specific objects.
  inline class Camera& Camera(const camera_t camera_id);
  inline const class Camera& Camera(const camera_t camera_id) const;
  inline class Image& Image(const image_t image_id);
  inline const class Image& Image(const image_t image_id) const;

  // Get all objects.
  inline const EIGEN_STL_UMAP(camera_t, class Camera) & Cameras() const;
  inline const EIGEN_STL_UMAP(image_t, class Image) & Images() const;

  // Check whether specific object exists.
  inline bool ExistsCamera(const camera_t camera_id) const;
  inline bool ExistsImage(const image_t image_id) const;

  // Get reference to scene graph.
  inline const class SceneGraph& SceneGraph() const;

  // Manually add data to cache.
  void AddCamera(const class Camera& camera);
  void AddImage(const class Image& image);

  // Load cameras, images, features, and matches from database.
  //
  // @param database              Source database from which to load data.
  // @param min_num_matches       Only load image pairs with a minimum number
  //                              of matches.
  // @param ignore_watermarks     Whether to ignore watermark image pairs.
  // @param image_names           Whether to use only load the data for a subset
  //                              of the images. All images are used if empty.
  void Load(const Database& database, const size_t min_num_matches,
            const bool ignore_watermarks,
            const std::set<std::string>& image_names);

 private:
  class SceneGraph scene_graph_;

  EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
  EIGEN_STL_UMAP(image_t, class Image) images_;
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

const EIGEN_STL_UMAP(camera_t, class Camera) & DatabaseCache::Cameras() const {
  return cameras_;
}

const EIGEN_STL_UMAP(image_t, class Image) & DatabaseCache::Images() const {
  return images_;
}

bool DatabaseCache::ExistsCamera(const camera_t camera_id) const {
  return cameras_.find(camera_id) != cameras_.end();
}

bool DatabaseCache::ExistsImage(const image_t image_id) const {
  return images_.find(image_id) != images_.end();
}

inline const class SceneGraph& DatabaseCache::SceneGraph() const {
  return scene_graph_;
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_DATABASE_CACHE_H_
