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

#ifndef COLMAP_SRC_BASE_SCENE_GRAPH_H_
#define COLMAP_SRC_BASE_SCENE_GRAPH_H_

#include <unordered_map>
#include <vector>

#include "base/database.h"
#include "util/types.h"

namespace colmap {

// Scene graph represents the graph of image to image and feature to feature
// correspondences of a dataset. It should be accessed from the DatabaseCache.
class SceneGraph {
 public:
  struct Correspondence {
    Correspondence()
        : image_id(kInvalidImageId), point2D_idx(kInvalidPoint2DIdx) {}
    Correspondence(const image_t image_id, const point2D_t point2D_idx)
        : image_id(image_id), point2D_idx(point2D_idx) {}

    // The identifier of the corresponding image.
    image_t image_id;

    // The index of the corresponding point in the corresponding image.
    point2D_t point2D_idx;
  };

  SceneGraph();

  // Number of added images.
  inline size_t NumImages() const;

  // Check whether image exists.
  inline bool ExistsImage(const image_t image_id) const;

  // Get the number of observations in an image. An observation is an image
  // point that has at least one correspondence.
  inline point2D_t NumObservationsForImage(const image_t image_id) const;

  // Get the number of correspondences per image.
  inline point2D_t NumCorrespondencesForImage(const image_t image_id) const;

  // Get the number of correspondences between a pair of images.
  inline point2D_t NumCorrespondencesBetweenImages(
      const image_t image_id1, const image_t image_id2) const;

  // Get the number of correspondences between all images.
  inline const std::unordered_map<image_pair_t, point2D_t>&
  NumCorrespondencesBetweenImages() const;

  // Finalize the database manager.
  //
  // - Calculates the number of observations per image by counting the number
  //   of image points that have at least one correspondence.
  // - Deletes images without observations, as they are useless for SfM.
  // - Shrinks the correspondence vectors to their size to save memory.
  void Finalize();

  // Add new image to the scene graph.
  void AddImage(const image_t image_id, const size_t num_points2D);

  // Add matches between images. This function ignores invalid correspondences
  // where the point indices are out of bounds or duplicate correspondences
  // between the same image points. Whenever either of the two cases occur
  // this function prints a warning to the standard output.
  void AddCorrespondences(const image_t image_id1, const image_t image_id2,
                          const FeatureMatches& matches);

  // Find the correspondence of an image point to any other image.
  inline const std::vector<Correspondence>& FindCorrespondences(
      const image_t image_id, const point2D_t point2D_idx) const;

  // Find correspondences to the given observation.
  //
  // Transitively collects correspondences to the given observation by first
  // finding correspondences to the given observation, then looking for
  // correspondences to the collected correspondences in the first step, and so
  // forth until the transitivity is exhausted or no more correspondences are
  // found. The returned list does not contain duplicates and contains
  // the given observation.
  std::vector<Correspondence> FindTransitiveCorrespondences(
      const image_t image_id, const point2D_t point2D_idx,
      const size_t transitivity) const;

  // Find all correspondences between two images.
  std::vector<std::pair<point2D_t, point2D_t>> FindCorrespondencesBetweenImages(
      const image_t image_id1, const image_t image_id2) const;

  // Check whether the image point has correspondences.
  inline bool HasCorrespondences(const image_t image_id,
                                 const point2D_t point2D_idx) const;

  // Check whether the given observation is part of a two-view track, i.e.
  // it only has one correspondence and that correspondence has the given
  // observation as its only correspondence.
  bool IsTwoViewObservation(const image_t image_id,
                            const point2D_t point2D_idx) const;

 private:
  struct Image {
    // Number of 2D points with at least one correspondence to another image.
    point2D_t num_observations = 0;

    // Total number of correspondences to other images. This measure is useful
    // to find a good initial pair, that is connected to many images.
    point2D_t num_correspondences = 0;

    // Correspondences to other images per image point.
    std::vector<std::vector<Correspondence>> corrs;
  };

  // The nodes of the scene graph are images.
  EIGEN_STL_UMAP(image_t, Image) images_;

  // The number of correspondences between pairs of images.
  std::unordered_map<image_pair_t, point2D_t> image_pairs_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t SceneGraph::NumImages() const { return images_.size(); }

bool SceneGraph::ExistsImage(const image_t image_id) const {
  return images_.find(image_id) != images_.end();
}

point2D_t SceneGraph::NumObservationsForImage(const image_t image_id) const {
  return images_.at(image_id).num_observations;
}

point2D_t SceneGraph::NumCorrespondencesForImage(const image_t image_id) const {
  return images_.at(image_id).num_correspondences;
}

point2D_t SceneGraph::NumCorrespondencesBetweenImages(
    const image_t image_id1, const image_t image_id2) const {
  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  const auto it = image_pairs_.find(pair_id);
  if (it == image_pairs_.end()) {
    return 0;
  } else {
    return static_cast<point2D_t>(it->second);
  }
}

inline const std::unordered_map<image_pair_t, point2D_t>&
SceneGraph::NumCorrespondencesBetweenImages() const {
  return image_pairs_;
}

const std::vector<SceneGraph::Correspondence>& SceneGraph::FindCorrespondences(
    const image_t image_id, const point2D_t point2D_idx) const {
  return images_.at(image_id).corrs.at(point2D_idx);
}

bool SceneGraph::HasCorrespondences(const image_t image_id,
                                    const point2D_t point2D_idx) const {
  return !images_.at(image_id).corrs.at(point2D_idx).empty();
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_SCENE_GRAPH_H_
