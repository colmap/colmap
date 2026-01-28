#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <unordered_set>

namespace colmap {

class PoseGraph {
 public:
  // Minimal relative pose data for pose graph edge.
  struct Edge {
    Edge() = default;

    explicit Edge(const Rigid3d& cam2_from_cam1)
        : cam2_from_cam1(cam2_from_cam1) {}

    // Relative pose from image 1 to image 2.
    Rigid3d cam2_from_cam1;

    // Number of two-view matches used to compute the relative pose.
    int num_matches = 0;

    // Whether this edge is valid for reconstruction.
    bool valid = true;

    // Invert the geometry to match swapped image order.
    void Invert() { cam2_from_cam1 = Inverse(cam2_from_cam1); }
  };

  PoseGraph() = default;
  ~PoseGraph() = default;

  // Edge accessors.
  inline std::unordered_map<image_pair_t, Edge>& Edges();
  inline const std::unordered_map<image_pair_t, Edge>& Edges() const;
  inline size_t NumEdges() const;
  inline bool Empty() const;
  inline void Clear();

  // Load edges from the correspondence graph.
  void Load(const CorrespondenceGraph& corr_graph);

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
  auto ValidEdges() const {
    return filter_view(
        [](const std::pair<const image_pair_t, Edge>& kv) {
          return kv.second.valid;
        },
        edges_.begin(),
        edges_.end());
  }

  // Compute the largest connected component of frames.
  // If filter_unregistered is true, only considers frames with HasPose().
  // Returns the set of frame_ids in the largest connected component.
  std::unordered_set<frame_t> ComputeLargestConnectedFrameComponent(
      const Reconstruction& reconstruction,
      bool filter_unregistered = true) const;

  // Mark image pairs as invalid if either image is not in the active set.
  void InvalidatePairsOutsideActiveImageIds(
      const std::unordered_set<image_t>& active_image_ids);

  // Mark connected clusters of images, where the cluster_id is sorted by the
  // the number of images. Populates `cluster_ids` output parameter.
  int MarkConnectedComponents(const Reconstruction& reconstruction,
                              std::unordered_map<frame_t, int>& cluster_ids,
                              int min_num_images = -1) const;

 private:
  // Map from pair ID to edge data. The pair ID is computed from the
  // two image IDs using ImagePairToPairId, with the smaller ID first.
  std::unordered_map<image_pair_t, Edge> edges_;
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

bool PoseGraph::Empty() const { return edges_.empty(); }

void PoseGraph::Clear() { edges_.clear(); }

PoseGraph::Edge& PoseGraph::AddEdge(image_t image_id1,
                                    image_t image_id2,
                                    PoseGraph::Edge edge) {
  if (ShouldSwapImagePair(image_id1, image_id2)) {
    edge.Invert();
  }
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  auto [it, inserted] = edges_.emplace(pair_id, std::move(edge));
  if (!inserted) {
    throw std::runtime_error(
        "Image pair already exists: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  return it->second;
}

bool PoseGraph::HasEdge(image_t image_id1, image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return edges_.find(pair_id) != edges_.end();
}

std::pair<PoseGraph::Edge&, bool> PoseGraph::EdgeRef(image_t image_id1,
                                                     image_t image_id2) {
  const bool swapped = ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return {edges_.at(pair_id), swapped};
}

std::pair<const PoseGraph::Edge&, bool> PoseGraph::EdgeRef(
    image_t image_id1, image_t image_id2) const {
  const bool swapped = ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return {edges_.at(pair_id), swapped};
}

bool PoseGraph::DeleteEdge(image_t image_id1, image_t image_id2) {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return edges_.erase(pair_id) > 0;
}

PoseGraph::Edge PoseGraph::GetEdge(image_t image_id1, image_t image_id2) const {
  const bool swapped = ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  PoseGraph::Edge result = edges_.at(pair_id);
  if (swapped) {
    result.Invert();
  }
  return result;
}

void PoseGraph::UpdateEdge(image_t image_id1,
                           image_t image_id2,
                           PoseGraph::Edge edge) {
  if (ShouldSwapImagePair(image_id1, image_id2)) {
    edge.Invert();
  }
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  auto it = edges_.find(pair_id);
  if (it == edges_.end()) {
    throw std::runtime_error(
        "Image pair does not exist: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  it->second = std::move(edge);
}

bool PoseGraph::IsValid(image_pair_t pair_id) const {
  auto it = edges_.find(pair_id);
  return it != edges_.end() && it->second.valid;
}

void PoseGraph::SetValidEdge(image_pair_t pair_id) {
  auto it = edges_.find(pair_id);
  THROW_CHECK(it != edges_.end()) << "Edge does not exist";
  it->second.valid = true;
}

void PoseGraph::SetInvalidEdge(image_pair_t pair_id) {
  auto it = edges_.find(pair_id);
  THROW_CHECK(it != edges_.end()) << "Edge does not exist";
  it->second.valid = false;
}

}  // namespace colmap
