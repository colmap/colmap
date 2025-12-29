#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include "glomap/scene/image_pair.h"
#include "glomap/scene/types.h"

#include <unordered_map>
#include <unordered_set>

namespace glomap {

class ViewGraph {
 public:
  ViewGraph() = default;
  ~ViewGraph() = default;

  // Image pair accessors.
  inline std::unordered_map<image_pair_t, struct ImagePair>& ImagePairs();
  inline const std::unordered_map<image_pair_t, struct ImagePair>& ImagePairs()
      const;
  inline size_t NumImagePairs() const;
  inline size_t NumValidImagePairs() const;
  inline bool Empty() const;
  inline void Clear();

  // Image pair operations.
  inline struct ImagePair& AddImagePair(image_t image_id1,
                                        image_t image_id2,
                                        struct ImagePair image_pair);
  inline bool HasImagePair(image_t image_id1, image_t image_id2) const;
  // Returns a reference to the image pair and whether the IDs were swapped.
  inline std::pair<struct ImagePair&, bool> ImagePair(image_t image_id1,
                                                      image_t image_id2);
  inline std::pair<const struct ImagePair&, bool> ImagePair(
      image_t image_id1, image_t image_id2) const;
  inline struct ImagePair GetImagePair(image_t image_id1,
                                       image_t image_id2) const;
  inline bool DeleteImagePair(image_t image_id1, image_t image_id2);
  inline void UpdateImagePair(image_t image_id1,
                              image_t image_id2,
                              struct ImagePair image_pair);

  // Validity operations.
  inline bool IsValid(image_pair_t pair_id) const;
  inline void SetValidImagePair(image_pair_t pair_id);
  inline void SetInvalidImagePair(image_pair_t pair_id);

  // Image pair weight operations.
  // Weights are stored separately and only allocated when used.
  inline void SetImagePairWeight(image_pair_t pair_id, double weight);
  inline double GetImagePairWeight(image_pair_t pair_id) const;
  inline bool HasImagePairWeights() const;
  inline void DeleteImagePairWeight(image_pair_t pair_id);
  inline void ClearImagePairWeights();

  // Returns a filter view over valid image pairs only.
  auto ValidImagePairs() const {
    return colmap::filter_view(
        [this](const std::pair<const image_pair_t, struct ImagePair>& kv) {
          return IsValid(kv.first);
        },
        image_pairs_.begin(),
        image_pairs_.end());
  }

  // Create the adjacency list for the images in the view graph.
  std::unordered_map<image_t, std::unordered_set<image_t>>
  CreateImageAdjacencyList() const;

  // Mark the images which are not connected to any other images as not
  // registered. Returns the number of images in the largest connected
  // component.
  int KeepLargestConnectedComponents(colmap::Reconstruction& reconstruction);

  // Mark connected clusters of images, where the cluster_id is sorted by the
  // the number of images. Populates `cluster_ids` output parameter.
  int MarkConnectedComponents(const colmap::Reconstruction& reconstruction,
                              std::unordered_map<frame_t, int>& cluster_ids,
                              int min_num_images = -1);

  // Mark image pairs as invalid if their relative rotation differs from the
  // reconstructed rotation by more than max_angle_deg.
  // Keeps existing invalid edges as invalid.
  void FilterByRelativeRotation(const colmap::Reconstruction& reconstruction,
                                double max_angle_deg = 5.0);

  // Mark image pairs as invalid if they have fewer than min_num_inliers.
  // Keeps existing invalid edges as invalid.
  void FilterByNumInliers(int min_num_inliers = 30);

  // Mark image pairs as invalid if their inlier ratio is below
  // min_inlier_ratio.
  // Keeps existing invalid edges as invalid.
  void FilterByInlierRatio(double min_inlier_ratio = 0.25);

 private:
  // Map from pair ID to image pair data. The pair ID is computed from the
  // two image IDs using ImagePairToPairId, with the smaller ID first.
  std::unordered_map<image_pair_t, struct ImagePair> image_pairs_;
  // Set of invalid pair IDs. Pairs not in this set are considered valid.
  std::unordered_set<image_pair_t> invalid_pairs_;
  // Optional weights for image pairs. Only populated when weights are loaded.
  // Returns 1.0 for pairs without explicit weights.
  std::unordered_map<image_pair_t, double> image_pair_weights_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

std::unordered_map<image_pair_t, struct ImagePair>& ViewGraph::ImagePairs() {
  return image_pairs_;
}

const std::unordered_map<image_pair_t, struct ImagePair>&
ViewGraph::ImagePairs() const {
  return image_pairs_;
}

size_t ViewGraph::NumImagePairs() const { return image_pairs_.size(); }

size_t ViewGraph::NumValidImagePairs() const {
  return image_pairs_.size() - invalid_pairs_.size();
}

bool ViewGraph::Empty() const { return image_pairs_.empty(); }

void ViewGraph::Clear() {
  image_pairs_.clear();
  invalid_pairs_.clear();
  image_pair_weights_.clear();
}

struct ImagePair& ViewGraph::AddImagePair(image_t image_id1,
                                          image_t image_id2,
                                          struct ImagePair image_pair) {
  if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
    image_pair.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto [it, inserted] = image_pairs_.emplace(pair_id, std::move(image_pair));
  if (!inserted) {
    throw std::runtime_error(
        "Image pair already exists: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  return it->second;
}

bool ViewGraph::HasImagePair(image_t image_id1, image_t image_id2) const {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return image_pairs_.find(pair_id) != image_pairs_.end();
}

std::pair<struct ImagePair&, bool> ViewGraph::ImagePair(image_t image_id1,
                                                        image_t image_id2) {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {image_pairs_.at(pair_id), swapped};
}

std::pair<const struct ImagePair&, bool> ViewGraph::ImagePair(
    image_t image_id1, image_t image_id2) const {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {image_pairs_.at(pair_id), swapped};
}

bool ViewGraph::DeleteImagePair(image_t image_id1, image_t image_id2) {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return image_pairs_.erase(pair_id) > 0;
}

struct ImagePair ViewGraph::GetImagePair(image_t image_id1,
                                         image_t image_id2) const {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  struct ImagePair result = image_pairs_.at(pair_id);
  if (swapped) {
    result.Invert();
  }
  return result;
}

void ViewGraph::UpdateImagePair(image_t image_id1,
                                image_t image_id2,
                                struct ImagePair image_pair) {
  if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
    image_pair.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto it = image_pairs_.find(pair_id);
  if (it == image_pairs_.end()) {
    throw std::runtime_error(
        "Image pair does not exist: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  it->second = std::move(image_pair);
}

bool ViewGraph::IsValid(image_pair_t pair_id) const {
  return invalid_pairs_.find(pair_id) == invalid_pairs_.end();
}

void ViewGraph::SetValidImagePair(image_pair_t pair_id) {
  invalid_pairs_.erase(pair_id);
}

void ViewGraph::SetInvalidImagePair(image_pair_t pair_id) {
  invalid_pairs_.insert(pair_id);
}

void ViewGraph::SetImagePairWeight(image_pair_t pair_id, double weight) {
  image_pair_weights_[pair_id] = weight;
}

double ViewGraph::GetImagePairWeight(image_pair_t pair_id) const {
  auto it = image_pair_weights_.find(pair_id);
  return it != image_pair_weights_.end() ? it->second : 1.0;
}

bool ViewGraph::HasImagePairWeights() const {
  return !image_pair_weights_.empty();
}

void ViewGraph::DeleteImagePairWeight(image_pair_t pair_id) {
  image_pair_weights_.erase(pair_id);
}

void ViewGraph::ClearImagePairWeights() { image_pair_weights_.clear(); }

}  // namespace glomap
