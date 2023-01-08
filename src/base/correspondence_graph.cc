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

#include "base/correspondence_graph.h"

#include <unordered_set>

#include "base/pose.h"
#include "util/string.h"

namespace colmap {

CorrespondenceGraph::CorrespondenceGraph() {}

std::unordered_map<image_pair_t, point2D_t>
CorrespondenceGraph::NumCorrespondencesBetweenImages() const {
  std::unordered_map<image_pair_t, point2D_t> num_corrs_between_images;
  num_corrs_between_images.reserve(image_pairs_.size());
  for (const auto& image_pair : image_pairs_) {
    num_corrs_between_images.emplace(image_pair.first,
                                     image_pair.second.num_correspondences);
  }
  return num_corrs_between_images;
}

void CorrespondenceGraph::Finalize() {
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

void CorrespondenceGraph::AddImage(const image_t image_id,
                                   const size_t num_points) {
  CHECK(!ExistsImage(image_id));
  images_[image_id].corrs.resize(num_points);
}

void CorrespondenceGraph::AddCorrespondences(const image_t image_id1,
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

  // Set the number of all correspondences for this image pair. Further below,
  // we will make sure that only unique correspondences are counted.
  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  auto& image_pair = image_pairs_[pair_id];
  image_pair.num_correspondences += static_cast<point2D_t>(matches.size());

  // Store all matches in correspondence graph data structure. This data-
  // structure uses more memory than storing the raw match matrices, but is
  // significantly more efficient when updating the correspondences in case an
  // observation is triangulated.

  for (const auto& match : matches) {
    const bool valid_idx1 = match.point2D_idx1 < image1.corrs.size();
    const bool valid_idx2 = match.point2D_idx2 < image2.corrs.size();

    if (valid_idx1 && valid_idx2) {
      auto& corrs1 = image1.corrs[match.point2D_idx1];
      auto& corrs2 = image2.corrs[match.point2D_idx2];

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
        image_pair.num_correspondences -= 1;
        std::cout << StringPrintf(
                         "WARNING: Duplicate correspondence between "
                         "point2D_idx=%d in image_id=%d and point2D_idx=%d in "
                         "image_id=%d",
                         match.point2D_idx1, image_id1, match.point2D_idx2,
                         image_id2)
                  << std::endl;
      } else {
        corrs1.emplace_back(image_id2, match.point2D_idx2);
        corrs2.emplace_back(image_id1, match.point2D_idx1);
      }
    } else {
      image1.num_correspondences -= 1;
      image2.num_correspondences -= 1;
      image_pair.num_correspondences -= 1;
      if (!valid_idx1) {
        std::cout
            << StringPrintf(
                   "WARNING: point2D_idx=%d in image_id=%d does not exist",
                   match.point2D_idx1, image_id1)
            << std::endl;
      }
      if (!valid_idx2) {
        std::cout
            << StringPrintf(
                   "WARNING: point2D_idx=%d in image_id=%d does not exist",
                   match.point2D_idx2, image_id2)
            << std::endl;
      }
    }
  }
}

void CorrespondenceGraph::FindTransitiveCorrespondences(
    const image_t image_id, const point2D_t point2D_idx,
    const size_t transitivity, std::vector<Correspondence>* found_corrs) const {
  CHECK_NE(transitivity, 1) << "Use more efficient FindCorrespondences()";

  found_corrs->clear();

  if (!HasCorrespondences(image_id, point2D_idx)) {
    return;
  }

  found_corrs->emplace_back(image_id, point2D_idx);

  std::unordered_map<image_t, std::unordered_set<point2D_t>> image_corrs;
  image_corrs[image_id].insert(point2D_idx);

  size_t corr_queue_begin = 0;
  size_t corr_queue_end = 1;

  for (size_t t = 0; t < transitivity; ++t) {
    // Collect correspondences at transitive level t to all
    // correspondences that were collected at transitive level t - 1.
    for (size_t i = corr_queue_begin; i < corr_queue_end; ++i) {
      const Correspondence ref_corr = (*found_corrs)[i];

      const Image& image = images_.at(ref_corr.image_id);
      const std::vector<Correspondence>& ref_corrs =
          image.corrs[ref_corr.point2D_idx];

      for (const Correspondence& corr : ref_corrs) {
        // Check if correspondence already collected, otherwise collect.
        auto& corr_image_corrs = image_corrs[corr.image_id];
        if (corr_image_corrs.insert(corr.point2D_idx).second) {
          found_corrs->emplace_back(corr.image_id, corr.point2D_idx);
        }
      }
    }

    // Move on to the next block of correspondences at next transitive level.
    corr_queue_begin = corr_queue_end;
    corr_queue_end = found_corrs->size();

    // No new correspondences collected in last transitivity level.
    if (corr_queue_begin == corr_queue_end) {
      break;
    }
  }

  // Remove first element, which is the given observation by swapping it
  // with the last collected correspondence.
  if (found_corrs->size() > 1) {
    found_corrs->front() = found_corrs->back();
  }
  found_corrs->pop_back();
}

FeatureMatches CorrespondenceGraph::FindCorrespondencesBetweenImages(
    const image_t image_id1, const image_t image_id2) const {
  const auto num_correspondences =
      NumCorrespondencesBetweenImages(image_id1, image_id2);

  if (num_correspondences == 0) {
    return {};
  }

  FeatureMatches found_corrs;
  found_corrs.reserve(num_correspondences);

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

bool CorrespondenceGraph::IsTwoViewObservation(
    const image_t image_id, const point2D_t point2D_idx) const {
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
