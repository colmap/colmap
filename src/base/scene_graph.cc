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

#include "base/scene_graph.h"

#include <unordered_set>

#include "util/string.h"

namespace colmap {

SceneGraph::SceneGraph() {}

void SceneGraph::Finalize() {
  for (auto it = images_.begin(); it != images_.end();) {
    it->second.num_observations = 0;
    for (auto& corr : it->second.corrs) {
      corr.shrink_to_fit();
      if (corr.size() > 0) {
        it->second.num_observations += 1;
      }
    }
    if (it->second.num_observations == 0) {
      images_.erase(it++);
    } else {
      ++it;
    }
  }
}

void SceneGraph::AddImage(const image_t image_id, const size_t num_points) {
  CHECK(!ExistsImage(image_id));
  images_[image_id].corrs.resize(num_points);
}

void SceneGraph::AddCorrespondences(const image_t image_id1,
                                    const image_t image_id2,
                                    const FeatureMatches& matches) {
  // Avoid self-matches - should only happen, if user provides custom matches.
  if (image_id1 == image_id2) {
    std::cout << "WARNING: Cannot use self-matches for image_id=" << image_id1
              << std::endl;
    return;
  }

  // Corresponding images.
  struct Image& image1 = images_.at(image_id1);
  struct Image& image2 = images_.at(image_id2);

  // Store number of correspondences for each image to find good initial pair.
  image1.num_correspondences += matches.size();
  image2.num_correspondences += matches.size();

  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  point2D_t& num_correspondences = image_pairs_[pair_id];
  num_correspondences += static_cast<point2D_t>(matches.size());

  // Store all matches in correspondence graph data structure. This data-
  // structure uses more memory than storing the raw match matrices, but is
  // significantly more efficient when updating the correspondences in case an
  // observation is triangulated.
  for (size_t i = 0; i < matches.size(); ++i) {
    const point2D_t point2D_idx1 = matches[i].point2D_idx1;
    const point2D_t point2D_idx2 = matches[i].point2D_idx2;

    const bool valid_idx1 = point2D_idx1 < image1.corrs.size();
    const bool valid_idx2 = point2D_idx2 < image2.corrs.size();

    if (valid_idx1 && valid_idx2) {
      auto& corrs1 = image1.corrs[point2D_idx1];
      auto& corrs2 = image2.corrs[point2D_idx2];

      const bool duplicate1 =
          std::find_if(corrs1.begin(), corrs1.end(),
                       [image_id2](const Correspondence& corr) {
                         return corr.image_id == image_id2;
                       }) != corrs1.end();
      const bool duplicate2 =
          std::find_if(corrs2.begin(), corrs2.end(),
                       [image_id1](const Correspondence& corr) {
                         return corr.image_id == image_id1;
                       }) != corrs2.end();

      if (duplicate1 || duplicate2) {
        image1.num_correspondences -= 1;
        image2.num_correspondences -= 1;
        num_correspondences -= 1;
        std::cout << StringPrintf(
                         "WARNING: Duplicate correspondence between "
                         "point2D_idx=%d in image_id=%d and point2D_idx=%d in "
                         "image_id=%d",
                         point2D_idx1, image_id1, point2D_idx2, image_id2)
                  << std::endl;
      } else {
        corrs1.emplace_back(image_id2, point2D_idx2);
        corrs2.emplace_back(image_id1, point2D_idx1);
      }
    } else {
      image1.num_correspondences -= 1;
      image2.num_correspondences -= 1;
      num_correspondences -= 1;
      if (!valid_idx1) {
        std::cout
            << StringPrintf(
                   "WARNING: point2D_idx=%d in image_id=%d does not exist",
                   point2D_idx1, image_id1)
            << std::endl;
      }
      if (!valid_idx2) {
        std::cout
            << StringPrintf(
                   "WARNING: point2D_idx=%d in image_id=%d does not exist",
                   point2D_idx2, image_id2)
            << std::endl;
      }
    }
  }
}

std::vector<SceneGraph::Correspondence>
SceneGraph::FindTransitiveCorrespondences(const image_t image_id,
                                          const point2D_t point2D_idx,
                                          const size_t transitivity) const {
  if (transitivity == 1) {
    return FindCorrespondences(image_id, point2D_idx);
  }

  std::vector<Correspondence> found_corrs;
  if (!HasCorrespondences(image_id, point2D_idx)) {
    return found_corrs;
  }

  found_corrs.emplace_back(image_id, point2D_idx);

  std::unordered_map<image_t, std::unordered_set<point2D_t>> image_corrs;
  image_corrs[image_id].insert(point2D_idx);

  size_t corr_queue_begin = 0;
  size_t corr_queue_end = found_corrs.size();

  for (size_t t = 0; t < transitivity; ++t) {
    // Collect correspondences at transitive level t to all
    // correspondences that were collected at transitive level t - 1.
    for (size_t i = corr_queue_begin; i < corr_queue_end; ++i) {
      const Correspondence ref_corr = found_corrs[i];

      const Image& image = images_.at(ref_corr.image_id);
      const std::vector<Correspondence>& ref_corrs =
          image.corrs[ref_corr.point2D_idx];

      for (const Correspondence corr : ref_corrs) {
        // Check if correspondence already collected, otherwise collect.
        auto& corr_image_corrs = image_corrs[corr.image_id];
        if (corr_image_corrs.count(corr.point2D_idx) == 0) {
          corr_image_corrs.insert(corr.point2D_idx);
          found_corrs.emplace_back(corr.image_id, corr.point2D_idx);
        }
      }
    }

    // Move on to the next block of correspondences at next transitive level.
    corr_queue_begin = corr_queue_end;
    corr_queue_end = found_corrs.size();

    // No new correspondences collected in last transitivity level.
    if (corr_queue_begin == corr_queue_end) {
      break;
    }
  }

  // Remove first element, which is the given observation by swapping it
  // with the last collected correspondence.
  if (found_corrs.size() > 1) {
    found_corrs.front() = found_corrs.back();
  }
  found_corrs.pop_back();

  return found_corrs;
}

std::vector<std::pair<point2D_t, point2D_t>>
SceneGraph::FindCorrespondencesBetweenImages(const image_t image_id1,
                                             const image_t image_id2) const {
  std::vector<std::pair<point2D_t, point2D_t>> found_corrs;
  const struct Image& image1 = images_.at(image_id1);
  for (point2D_t point2D_idx1 = 0; point2D_idx1 < image1.corrs.size();
       ++point2D_idx1) {
    for (const Correspondence& corr1 : image1.corrs[point2D_idx1]) {
      if (corr1.image_id == image_id2) {
        found_corrs.emplace_back(point2D_idx1, corr1.point2D_idx);
      }
    }
  }
  return found_corrs;
}

bool SceneGraph::IsTwoViewObservation(const image_t image_id,
                                      const point2D_t point2D_idx) const {
  const struct Image& image = images_.at(image_id);
  const std::vector<Correspondence>& corrs = image.corrs.at(point2D_idx);
  if (corrs.size() != 1) {
    return false;
  }
  const struct Image& other_image = images_.at(corrs[0].image_id);
  const std::vector<Correspondence>& other_corrs =
      other_image.corrs.at(corrs[0].point2D_idx);
  return other_corrs.size() == 1;
}

}  // namespace colmap
