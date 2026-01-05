#pragma once

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include "glomap/scene/types.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace glomap {

// Minimal relative pose data for pose graph.
struct RelativePoseData {
  RelativePoseData() = default;

  explicit RelativePoseData(Rigid3d cam2_from_cam1)
      : cam2_from_cam1(std::move(cam2_from_cam1)) {}

  // Relative pose from image 1 to image 2.
  std::optional<Rigid3d> cam2_from_cam1;

  // Inlier feature matches between the two images.
  colmap::FeatureMatches inlier_matches;

  // Invert the geometry to match swapped image order.
  void Invert() {
    if (cam2_from_cam1.has_value()) {
      cam2_from_cam1 = colmap::Inverse(*cam2_from_cam1);
    }
    for (auto& match : inlier_matches) {
      std::swap(match.point2D_idx1, match.point2D_idx2);
    }
  }
};

class PoseGraph {
 public:
  PoseGraph() = default;
  ~PoseGraph() = default;

  // Image pair accessors.
  inline std::unordered_map<image_pair_t, RelativePoseData>& ImagePairs();
  inline const std::unordered_map<image_pair_t, RelativePoseData>& ImagePairs()
      const;
  inline size_t NumImagePairs() const;
  inline size_t NumValidImagePairs() const;
  inline bool Empty() const;
  inline void Clear();

  // Read image pairs from the database.
  // If allow_duplicate is false, throws on duplicate pairs. If true, logs a
  // warning and updates the existing pair.
  // Decomposes relative poses for valid pairs that don't have poses yet.
  void LoadFromDatabase(const colmap::Database& database,
                        bool allow_duplicate = false);

  // Image pair operations.
  inline RelativePoseData& AddImagePair(image_t image_id1,
                                        image_t image_id2,
                                        RelativePoseData rel_pose_data);
  inline bool HasImagePair(image_t image_id1, image_t image_id2) const;
  // Returns a reference to the image pair and whether the IDs were swapped.
  inline std::pair<RelativePoseData&, bool> ImagePair(image_t image_id1,
                                                      image_t image_id2);
  inline std::pair<const RelativePoseData&, bool> ImagePair(
      image_t image_id1, image_t image_id2) const;
  inline RelativePoseData GetImagePair(image_t image_id1,
                                       image_t image_id2) const;
  inline bool DeleteImagePair(image_t image_id1, image_t image_id2);
  inline void UpdateImagePair(image_t image_id1,
                              image_t image_id2,
                              RelativePoseData rel_pose_data);

  // Validity operations.
  inline bool IsValid(image_pair_t pair_id) const;
  inline void SetValidImagePair(image_pair_t pair_id);
  inline void SetInvalidImagePair(image_pair_t pair_id);

  // Returns a filter view over valid image pairs only.
  // Pairs are valid if they exist and are not in the invalid set.
  auto ValidImagePairs() const {
    return colmap::filter_view(
        [this](const std::pair<const image_pair_t, RelativePoseData>& kv) {
          return IsValid(kv.first);
        },
        rel_pose_datas_.begin(),
        rel_pose_datas_.end());
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

 private:
  // Map from pair ID to image pair data. The pair ID is computed from the
  // two image IDs using ImagePairToPairId, with the smaller ID first.
  std::unordered_map<image_pair_t, RelativePoseData> rel_pose_datas_;
  // Set of invalid pair IDs. Pairs not in this set are considered valid.
  std::unordered_set<image_pair_t> invalid_pairs_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

std::unordered_map<image_pair_t, RelativePoseData>& PoseGraph::ImagePairs() {
  return rel_pose_datas_;
}

const std::unordered_map<image_pair_t, RelativePoseData>&
PoseGraph::ImagePairs() const {
  return rel_pose_datas_;
}

size_t PoseGraph::NumImagePairs() const { return rel_pose_datas_.size(); }

size_t PoseGraph::NumValidImagePairs() const {
  return rel_pose_datas_.size() - invalid_pairs_.size();
}

bool PoseGraph::Empty() const { return rel_pose_datas_.empty(); }

void PoseGraph::Clear() {
  rel_pose_datas_.clear();
  invalid_pairs_.clear();
}

RelativePoseData& PoseGraph::AddImagePair(image_t image_id1,
                                          image_t image_id2,
                                          RelativePoseData rel_pose_data) {
  THROW_CHECK(rel_pose_data.cam2_from_cam1.has_value());
  if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
    rel_pose_data.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto [it, inserted] = rel_pose_datas_.emplace(pair_id, std::move(rel_pose_data));
  if (!inserted) {
    throw std::runtime_error(
        "Image pair already exists: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  return it->second;
}

bool PoseGraph::HasImagePair(image_t image_id1, image_t image_id2) const {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return rel_pose_datas_.find(pair_id) != rel_pose_datas_.end();
}

std::pair<RelativePoseData&, bool> PoseGraph::ImagePair(image_t image_id1,
                                                        image_t image_id2) {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {rel_pose_datas_.at(pair_id), swapped};
}

std::pair<const RelativePoseData&, bool> PoseGraph::ImagePair(
    image_t image_id1, image_t image_id2) const {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {rel_pose_datas_.at(pair_id), swapped};
}

bool PoseGraph::DeleteImagePair(image_t image_id1, image_t image_id2) {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return rel_pose_datas_.erase(pair_id) > 0;
}

RelativePoseData PoseGraph::GetImagePair(image_t image_id1,
                                         image_t image_id2) const {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  RelativePoseData result = rel_pose_datas_.at(pair_id);
  if (swapped) {
    result.Invert();
  }
  return result;
}

void PoseGraph::UpdateImagePair(image_t image_id1,
                                image_t image_id2,
                                RelativePoseData rel_pose_data) {
  THROW_CHECK(rel_pose_data.cam2_from_cam1.has_value());
  if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
    rel_pose_data.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto it = rel_pose_datas_.find(pair_id);
  if (it == rel_pose_datas_.end()) {
    throw std::runtime_error(
        "Image pair does not exist: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  it->second = std::move(rel_pose_data);
}

bool PoseGraph::IsValid(image_pair_t pair_id) const {
  return rel_pose_datas_.count(pair_id) > 0 &&
         invalid_pairs_.find(pair_id) == invalid_pairs_.end();
}

void PoseGraph::SetValidImagePair(image_pair_t pair_id) {
  THROW_CHECK(rel_pose_datas_.count(pair_id) > 0) << "Image pair does not exist";
  invalid_pairs_.erase(pair_id);
}

void PoseGraph::SetInvalidImagePair(image_pair_t pair_id) {
  THROW_CHECK(rel_pose_datas_.count(pair_id) > 0) << "Image pair does not exist";
  invalid_pairs_.insert(pair_id);
}

}  // namespace glomap
