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

class PoseGraph {
 public:
  // Minimal relative pose data for pose graph edge.
  struct Edge {
    Edge() = default;

    explicit Edge(const Rigid3d& cam2_from_cam1)
        : cam2_from_cam1(cam2_from_cam1) {}

    // Relative pose from image 1 to image 2.
    Rigid3d cam2_from_cam1;

    // Inlier feature matches between the two images.
    colmap::FeatureMatches inlier_matches;

    // Invert the geometry to match swapped image order.
    void Invert() {
      cam2_from_cam1 = colmap::Inverse(cam2_from_cam1);
      for (auto& match : inlier_matches) {
        std::swap(match.point2D_idx1, match.point2D_idx2);
      }
    }
  };

  PoseGraph() = default;
  ~PoseGraph() = default;

  // Edge accessors.
  inline std::unordered_map<image_pair_t, Edge>& Edges();
  inline const std::unordered_map<image_pair_t, Edge>& Edges() const;
  inline size_t NumEdges() const;
  inline size_t NumValidEdges() const;
  inline bool Empty() const;
  inline void Clear();

  // Read edges from the database.
  // If allow_duplicate is false, throws on duplicate pairs. If true, logs a
  // warning and updates the existing pair.
  // Decomposes relative poses for valid pairs that don't have poses yet.
  void LoadFromDatabase(const colmap::Database& database,
                        bool allow_duplicate = false);

  // Edge operations.
  inline Edge& AddEdge(image_t image_id1, image_t image_id2, Edge edge);
  inline bool HasEdge(image_t image_id1, image_t image_id2) const;
  // Returns a reference to the edge and whether the IDs were swapped.
  inline std::pair<Edge&, bool> EdgeRef(image_t image_id1, image_t image_id2);
  inline std::pair<const Edge&, bool> EdgeRef(image_t image_id1,
                                              image_t image_id2) const;
  inline Edge GetEdge(image_t image_id1, image_t image_id2) const;
  inline bool DeleteEdge(image_t image_id1, image_t image_id2);
  inline void UpdateEdge(image_t image_id1, image_t image_id2, Edge edge);

  // Validity operations.
  inline bool IsValid(image_pair_t pair_id) const;
  inline void SetValidEdge(image_pair_t pair_id);
  inline void SetInvalidEdge(image_pair_t pair_id);

  // Returns a filter view over valid edges only.
  // Edges are valid if they exist and are not in the invalid set.
  auto ValidEdges() const {
    return colmap::filter_view(
        [this](const std::pair<const image_pair_t, Edge>& kv) {
          return IsValid(kv.first);
        },
        edges_.begin(),
        edges_.end());
  }

  // Create the adjacency list for the images in the pose graph.
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

  // Mark edges as invalid if their relative rotation differs from the
  // reconstructed rotation by more than max_angle_deg.
  // Keeps existing invalid edges as invalid.
  void FilterByRelativeRotation(const colmap::Reconstruction& reconstruction,
                                double max_angle_deg = 5.0);

 private:
  // Map from pair ID to edge data. The pair ID is computed from the
  // two image IDs using ImagePairToPairId, with the smaller ID first.
  std::unordered_map<image_pair_t, Edge> edges_;
  // Set of invalid pair IDs. Pairs not in this set are considered valid.
  std::unordered_set<image_pair_t> invalid_pairs_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

std::unordered_map<image_pair_t, PoseGraph::Edge>& PoseGraph::Edges() {
  return edges_;
}

const std::unordered_map<image_pair_t, PoseGraph::Edge>& PoseGraph::Edges()
    const {
  return edges_;
}

size_t PoseGraph::NumEdges() const { return edges_.size(); }

size_t PoseGraph::NumValidEdges() const {
  return edges_.size() - invalid_pairs_.size();
}

bool PoseGraph::Empty() const { return edges_.empty(); }

void PoseGraph::Clear() {
  edges_.clear();
  invalid_pairs_.clear();
}

PoseGraph::Edge& PoseGraph::AddEdge(image_t image_id1,
                                    image_t image_id2,
                                    PoseGraph::Edge edge) {
  if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
    edge.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto [it, inserted] = edges_.emplace(pair_id, std::move(edge));
  if (!inserted) {
    throw std::runtime_error(
        "Image pair already exists: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  return it->second;
}

bool PoseGraph::HasEdge(image_t image_id1, image_t image_id2) const {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return edges_.find(pair_id) != edges_.end();
}

std::pair<PoseGraph::Edge&, bool> PoseGraph::EdgeRef(image_t image_id1,
                                                     image_t image_id2) {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {edges_.at(pair_id), swapped};
}

std::pair<const PoseGraph::Edge&, bool> PoseGraph::EdgeRef(
    image_t image_id1, image_t image_id2) const {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return {edges_.at(pair_id), swapped};
}

bool PoseGraph::DeleteEdge(image_t image_id1, image_t image_id2) {
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  return edges_.erase(pair_id) > 0;
}

PoseGraph::Edge PoseGraph::GetEdge(image_t image_id1, image_t image_id2) const {
  const bool swapped = colmap::ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  PoseGraph::Edge result = edges_.at(pair_id);
  if (swapped) {
    result.Invert();
  }
  return result;
}

void PoseGraph::UpdateEdge(image_t image_id1,
                           image_t image_id2,
                           PoseGraph::Edge edge) {
  if (colmap::ShouldSwapImagePair(image_id1, image_id2)) {
    edge.Invert();
  }
  const image_pair_t pair_id = colmap::ImagePairToPairId(image_id1, image_id2);
  auto it = edges_.find(pair_id);
  if (it == edges_.end()) {
    throw std::runtime_error(
        "Image pair does not exist: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  it->second = std::move(edge);
}

bool PoseGraph::IsValid(image_pair_t pair_id) const {
  return edges_.count(pair_id) > 0 &&
         invalid_pairs_.find(pair_id) == invalid_pairs_.end();
}

void PoseGraph::SetValidEdge(image_pair_t pair_id) {
  THROW_CHECK(edges_.count(pair_id) > 0) << "Edge does not exist";
  invalid_pairs_.erase(pair_id);
}

void PoseGraph::SetInvalidEdge(image_pair_t pair_id) {
  THROW_CHECK(edges_.count(pair_id) > 0) << "Edge does not exist";
  invalid_pairs_.insert(pair_id);
}

}  // namespace glomap
