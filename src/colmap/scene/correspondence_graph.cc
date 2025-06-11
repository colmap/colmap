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

#include "colmap/scene/correspondence_graph.h"

#include "colmap/geometry/pose.h"
#include "colmap/util/string.h"

#include <map>
#include <set>

namespace colmap {

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
  THROW_CHECK(!finalized_);
  finalized_ = true;

  // Flatten all correspondences, remove images without observations.
  for (auto& [_, image] : images_) {
    // Count number of correspondences and observations.
    image.num_observations = 0;
    size_t num_total_corrs = 0;
    for (auto& corr : image.corrs) {
      num_total_corrs += corr.size();
      if (!corr.empty()) {
        image.num_observations += 1;
      }
    }

    // Reshuffle correspondences into flattened vector.
    const point2D_t num_points2D = image.corrs.size();
    image.flat_corrs.reserve(num_total_corrs);
    image.flat_corr_begs.resize(num_points2D + 1);
    for (point2D_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx) {
      image.flat_corr_begs[point2D_idx] = image.flat_corrs.size();
      std::vector<Correspondence>& corrs = image.corrs[point2D_idx];
      image.flat_corrs.insert(
          image.flat_corrs.end(), corrs.begin(), corrs.end());
    }
    image.flat_corr_begs[num_points2D] = image.flat_corrs.size();

    // Ensure we reserved enough space before insertion.
    THROW_CHECK_EQ(image.flat_corrs.size(), num_total_corrs);

    // Deallocate original data.
    image.corrs.clear();
    image.corrs.shrink_to_fit();
  }
}

void CorrespondenceGraph::AddImage(const image_t image_id,
                                   const size_t num_points) {
  THROW_CHECK(!ExistsImage(image_id));
  images_[image_id].corrs.resize(num_points);
}

void CorrespondenceGraph::AddCorrespondences(const image_t image_id1,
                                             const image_t image_id2,
                                             const FeatureMatches& matches) {
  // Avoid self-matches - should only happen, if user provides custom matches.
  if (image_id1 == image_id2) {
    LOG(WARNING) << "Cannot use self-matches for image_id=" << image_id1;
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
          std::find_if(corrs1.begin(),
                       corrs1.end(),
                       [image_id2](const Correspondence& corr) {
                         return corr.image_id == image_id2;
                       }) != corrs1.end();
      const bool duplicate2 =
          std::find_if(corrs2.begin(),
                       corrs2.end(),
                       [image_id1](const Correspondence& corr) {
                         return corr.image_id == image_id1;
                       }) != corrs2.end();

      if (duplicate1 || duplicate2) {
        image1.num_correspondences -= 1;
        image2.num_correspondences -= 1;
        image_pair.num_correspondences -= 1;
        LOG(WARNING) << StringPrintf(
            "Duplicate correspondence between "
            "point2D_idx=%d in image_id=%d and point2D_idx=%d in "
            "image_id=%d",
            match.point2D_idx1,
            image_id1,
            match.point2D_idx2,
            image_id2);
      } else {
        corrs1.emplace_back(image_id2, match.point2D_idx2);
        corrs2.emplace_back(image_id1, match.point2D_idx1);
      }
    } else {
      image1.num_correspondences -= 1;
      image2.num_correspondences -= 1;
      image_pair.num_correspondences -= 1;
      if (!valid_idx1) {
        LOG(WARNING) << StringPrintf(
            "point2D_idx=%d in image_id=%d does not exist",
            match.point2D_idx1,
            image_id1);
      }
      if (!valid_idx2) {
        LOG(WARNING) << StringPrintf(
            "point2D_idx=%d in image_id=%d does not exist",
            match.point2D_idx2,
            image_id2);
      }
    }
  }
}

CorrespondenceGraph::CorrespondenceRange
CorrespondenceGraph::FindCorrespondences(const image_t image_id,
                                         const point2D_t point2D_idx) const {
  THROW_CHECK(finalized_);
  const point2D_t next_point2D_idx = point2D_idx + 1;
  const Image& image = images_.at(image_id);
  const Correspondence* beg =
      image.flat_corrs.data() + image.flat_corr_begs.at(point2D_idx);
  const Correspondence* end =
      image.flat_corrs.data() + image.flat_corr_begs.at(next_point2D_idx);
  return CorrespondenceRange{beg, end};
}

void CorrespondenceGraph::ExtractCorrespondences(
    const image_t image_id,
    const point2D_t point2D_idx,
    std::vector<Correspondence>* corrs) const {
  const auto range = FindCorrespondences(image_id, point2D_idx);
  corrs->clear();
  corrs->reserve(range.end - range.beg);
  for (const Correspondence* corr = range.beg; corr < range.end; ++corr) {
    corrs->push_back(*corr);
  }
}

void CorrespondenceGraph::ExtractTransitiveCorrespondences(
    const image_t image_id,
    const point2D_t point2D_idx,
    const size_t transitivity,
    std::vector<Correspondence>* corrs) const {
  if (transitivity == 1) {
    ExtractCorrespondences(image_id, point2D_idx, corrs);
    return;
  }

  corrs->clear();
  if (!HasCorrespondences(image_id, point2D_idx)) {
    return;
  }

  // Push requested image point on queue to visit. Will be removed later.
  corrs->emplace_back(image_id, point2D_idx);

  std::map<image_t, std::set<point2D_t>> image_corrs;
  image_corrs[image_id].insert(point2D_idx);

  size_t corr_queue_beg = 0;
  size_t corr_queue_end = 1;

  for (size_t t = 0; t < transitivity; ++t) {
    // Collect correspondences at transitive level t to all
    // correspondences that were collected at transitive level t - 1.
    for (size_t i = corr_queue_beg; i < corr_queue_end; ++i) {
      const Correspondence ref_corr = (*corrs)[i];
      const CorrespondenceRange ref_corr_range =
          FindCorrespondences(ref_corr.image_id, ref_corr.point2D_idx);
      for (const Correspondence* corr = ref_corr_range.beg;
           corr < ref_corr_range.end;
           ++corr) {
        // Check if correspondence already collected, otherwise collect.
        auto& corr_image_corrs = image_corrs[corr->image_id];
        if (corr_image_corrs.insert(corr->point2D_idx).second) {
          corrs->emplace_back(corr->image_id, corr->point2D_idx);
        }
      }
    }

    // Move on to the next block of correspondences at next transitive level.
    corr_queue_beg = corr_queue_end;
    corr_queue_end = corrs->size();

    // No new correspondences collected in last transitivity level.
    if (corr_queue_beg == corr_queue_end) {
      break;
    }
  }

  // Remove first element, which is the given observation by swapping it
  // with the last collected correspondence.
  if (corrs->size() > 1) {
    corrs->front() = corrs->back();
  }
  corrs->pop_back();
}

FeatureMatches CorrespondenceGraph::FindCorrespondencesBetweenImages(
    const image_t image_id1, const image_t image_id2) const {
  const point2D_t num_correspondences =
      NumCorrespondencesBetweenImages(image_id1, image_id2);
  if (num_correspondences == 0) {
    return {};
  }

  FeatureMatches corrs;
  corrs.reserve(num_correspondences);

  const point2D_t num_points2D1 =
      images_.at(image_id1).flat_corr_begs.size() - 1;
  for (point2D_t point2D_idx1 = 0; point2D_idx1 < num_points2D1;
       ++point2D_idx1) {
    const CorrespondenceRange range =
        FindCorrespondences(image_id1, point2D_idx1);
    for (const Correspondence* corr = range.beg; corr < range.end; ++corr) {
      if (corr->image_id == image_id2) {
        corrs.emplace_back(point2D_idx1, corr->point2D_idx);
      }
    }
  }

  return corrs;
}

bool CorrespondenceGraph::IsTwoViewObservation(
    const image_t image_id, const point2D_t point2D_idx) const {
  const CorrespondenceRange range = FindCorrespondences(image_id, point2D_idx);
  if (range.end - range.beg != 1) {
    return false;
  }
  const CorrespondenceRange other_range =
      FindCorrespondences(range.beg->image_id, range.beg->point2D_idx);
  return (other_range.end - other_range.beg) == 1;
}

std::ostream& operator<<(
    std::ostream& stream,
    const CorrespondenceGraph::Correspondence& correspondence) {
  stream << "Correspondence(image_id=" << correspondence.image_id
         << ", point2D_idx=" << correspondence.point2D_idx << ")";
  return stream;
}

std::ostream& operator<<(std::ostream& stream,
                         const CorrespondenceGraph& correspondence_graph) {
  stream << "CorrespondenceGraph(num_images="
         << correspondence_graph.NumImages()
         << ", num_image_pairs=" << correspondence_graph.NumImagePairs() << ")";
  return stream;
}

}  // namespace colmap
