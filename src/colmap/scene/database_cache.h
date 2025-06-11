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

#pragma once

#include "colmap/geometry/gps.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/database.h"
#include "colmap/scene/image.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <memory>
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
  DatabaseCache();

  // Load cameras, images, features, and matches from database.
  //
  // @param database              Source database from which to load data.
  // @param min_num_matches       Only load image pairs with a minimum number
  //                              of matches.
  // @param ignore_watermarks     Whether to ignore watermark image pairs.
  // @param image_names           Whether to use only load the data for a subset
  //                              of the images. Notice that if one image of a
  //                              frame is included, all other images in the
  //                              same frame will also be included. All images
  //                              are used if empty.
  void Load(const Database& database,
            size_t min_num_matches,
            bool ignore_watermarks,
            const std::unordered_set<std::string>& image_names);

  static std::shared_ptr<DatabaseCache> Create(
      const Database& database,
      size_t min_num_matches,
      bool ignore_watermarks,
      const std::unordered_set<std::string>& image_names);

  // Get number of objects.
  inline size_t NumRigs() const;
  inline size_t NumCameras() const;
  inline size_t NumFrames() const;
  inline size_t NumImages() const;
  inline size_t NumPosePriors() const;

  // Add objects.
  void AddRig(class Rig rig);
  void AddCamera(struct Camera camera);
  void AddFrame(class Frame frame);
  void AddImage(class Image image);
  void AddPosePrior(image_t image_id, struct PosePrior pose_prior);

  // Get specific objects.
  inline class Rig& Rig(rig_t rig_id);
  inline const class Rig& Rig(rig_t rig_id) const;
  inline struct Camera& Camera(camera_t camera_id);
  inline const struct Camera& Camera(camera_t camera_id) const;
  inline class Frame& Frame(frame_t frame_id);
  inline const class Frame& Frame(frame_t frame_id) const;
  inline class Image& Image(image_t image_id);
  inline const class Image& Image(image_t image_id) const;
  inline struct PosePrior& PosePrior(image_t image_id);
  inline const struct PosePrior& PosePrior(image_t image_id) const;

  // Get all objects.
  inline const std::unordered_map<rig_t, class Rig>& Rigs() const;
  inline const std::unordered_map<camera_t, struct Camera>& Cameras() const;
  inline const std::unordered_map<frame_t, class Frame>& Frames() const;
  inline const std::unordered_map<image_t, class Image>& Images() const;
  inline const std::unordered_map<image_t, struct PosePrior>& PosePriors()
      const;

  // Check whether specific object exists.
  inline bool ExistsRig(rig_t rig_id) const;
  inline bool ExistsCamera(camera_t camera_id) const;
  inline bool ExistsFrame(frame_t frame_id) const;
  inline bool ExistsImage(image_t image_id) const;
  inline bool ExistsPosePrior(image_t image_id) const;

  // Get reference to const correspondence graph.
  inline std::shared_ptr<const class CorrespondenceGraph> CorrespondenceGraph()
      const;

  // Find specific image by name. Note that this uses linear search.
  const class Image* FindImageWithName(const std::string& name) const;

  // Setup PosePriors for PosePriorBundleAdjustment
  bool SetupPosePriors();

 private:
  std::shared_ptr<class CorrespondenceGraph> correspondence_graph_;

  std::unordered_map<rig_t, class Rig> rigs_;
  std::unordered_map<camera_t, struct Camera> cameras_;
  std::unordered_map<frame_t, class Frame> frames_;
  std::unordered_map<image_t, class Image> images_;
  std::unordered_map<image_t, struct PosePrior> pose_priors_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t DatabaseCache::NumRigs() const { return rigs_.size(); }

size_t DatabaseCache::NumCameras() const { return cameras_.size(); }

size_t DatabaseCache::NumFrames() const { return frames_.size(); }

size_t DatabaseCache::NumImages() const { return images_.size(); }

size_t DatabaseCache::NumPosePriors() const { return pose_priors_.size(); }

class Rig& DatabaseCache::Rig(const rig_t rig_id) { return rigs_.at(rig_id); }

const class Rig& DatabaseCache::Rig(const rig_t rig_id) const {
  return rigs_.at(rig_id);
}

struct Camera& DatabaseCache::Camera(const camera_t camera_id) {
  return cameras_.at(camera_id);
}

const struct Camera& DatabaseCache::Camera(const camera_t camera_id) const {
  return cameras_.at(camera_id);
}

class Frame& DatabaseCache::Frame(const frame_t frame_id) {
  return frames_.at(frame_id);
}

const class Frame& DatabaseCache::Frame(const frame_t frame_id) const {
  return frames_.at(frame_id);
}

class Image& DatabaseCache::Image(const image_t image_id) {
  return images_.at(image_id);
}

const class Image& DatabaseCache::Image(const image_t image_id) const {
  return images_.at(image_id);
}

struct PosePrior& DatabaseCache::PosePrior(image_t image_id) {
  return pose_priors_.at(image_id);
}

const struct PosePrior& DatabaseCache::PosePrior(image_t image_id) const {
  return pose_priors_.at(image_id);
}

const std::unordered_map<rig_t, class Rig>& DatabaseCache::Rigs() const {
  return rigs_;
}

const std::unordered_map<camera_t, struct Camera>& DatabaseCache::Cameras()
    const {
  return cameras_;
}

const std::unordered_map<frame_t, class Frame>& DatabaseCache::Frames() const {
  return frames_;
}

const std::unordered_map<image_t, class Image>& DatabaseCache::Images() const {
  return images_;
}

const std::unordered_map<image_t, struct PosePrior>& DatabaseCache::PosePriors()
    const {
  return pose_priors_;
}

bool DatabaseCache::ExistsRig(const rig_t rig_id) const {
  return rigs_.find(rig_id) != rigs_.end();
}

bool DatabaseCache::ExistsCamera(const camera_t camera_id) const {
  return cameras_.find(camera_id) != cameras_.end();
}

bool DatabaseCache::ExistsFrame(const frame_t frame_id) const {
  return frames_.find(frame_id) != frames_.end();
}

bool DatabaseCache::ExistsImage(const image_t image_id) const {
  return images_.find(image_id) != images_.end();
}

bool DatabaseCache::ExistsPosePrior(const image_t image_id) const {
  return pose_priors_.find(image_id) != pose_priors_.end();
}

std::shared_ptr<const class CorrespondenceGraph>
DatabaseCache::CorrespondenceGraph() const {
  return correspondence_graph_;
}

}  // namespace colmap
