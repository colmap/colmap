#include "glomap/estimators/rotation_averaging.h"

#include "colmap/geometry/pose.h"
#include "colmap/math/spanning_tree.h"

#include "glomap/estimators/rotation_averaging_impl.h"
#include "glomap/estimators/rotation_initializer.h"

#include <queue>

namespace glomap {
namespace {

bool AllSensorsFromRigKnown(
    const std::unordered_map<rig_t, colmap::Rig>& rigs) {
  bool all_known = true;
  for (const auto& [rig_id, rig] : rigs) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (!sensor.has_value()) {
        LOG(ERROR) << "Rig " << rig_id
                   << " with unknown sensor_from_rig for sensor "
                   << sensor_id.id
                   << ", but gravity aligned rotation is "
                      "requested. Please specify the rig calibration.";
        all_known = false;
      }
    }
  }
  return all_known;
}

// Compute maximum spanning tree of the view graph weighted by inlier count.
// Returns the root image_id and populates the parents map.
image_t ComputeMaximumSpanningTree(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<image_t, image_t>& parents) {
  // Build mapping between image_id and contiguous indices.
  std::unordered_map<image_t, int> image_id_to_idx;
  std::vector<image_t> idx_to_image_id;
  image_id_to_idx.reserve(images.size());
  idx_to_image_id.reserve(images.size());

  for (const auto& [image_id, image] : images) {
    if (image.HasPose()) {
      image_id_to_idx[image_id] = static_cast<int>(idx_to_image_id.size());
      idx_to_image_id.push_back(image_id);
    }
  }

  // Build edges and weights from view graph.
  std::vector<std::pair<int, int>> edges;
  std::vector<float> weights;
  edges.reserve(view_graph.image_pairs.size());
  weights.reserve(view_graph.image_pairs.size());

  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) {
      continue;
    }
    const auto it1 = image_id_to_idx.find(image_pair.image_id1);
    const auto it2 = image_id_to_idx.find(image_pair.image_id2);
    if (it1 == image_id_to_idx.end() || it2 == image_id_to_idx.end()) {
      continue;
    }
    edges.emplace_back(it1->second, it2->second);
    weights.push_back(static_cast<float>(image_pair.inliers.size()));
  }

  // Compute spanning tree using generic algorithm.
  const colmap::SpanningTree tree = colmap::ComputeMaximumSpanningTree(
      idx_to_image_id.size(), edges, weights);

  // Convert back to image_id based parent map.
  parents.clear();
  for (size_t i = 0; i < idx_to_image_id.size(); ++i) {
    if (tree.parents[i] >= 0) {
      parents[idx_to_image_id[i]] = idx_to_image_id[tree.parents[i]];
    }
  }

  return idx_to_image_id[tree.root];
}

}  // namespace

bool RotationEstimator::EstimateRotations(
    const ViewGraph& view_graph,
    const std::vector<colmap::PosePrior>& pose_priors,
    colmap::Reconstruction& reconstruction) {
  if (options_.use_gravity && !AllSensorsFromRigKnown(reconstruction.Rigs())) {
    return false;
  }

  // Initialize rotations from maximum spanning tree.
  if (!options_.skip_initialization && !options_.use_gravity) {
    InitializeFromMaximumSpanningTree(view_graph, reconstruction);
  }

  // Build the optimization problem.
  RotationAveragingProblem problem(
      view_graph, reconstruction, pose_priors, options_);

  // Solve and apply results.
  RotationAveragingSolver solver(options_);
  if (!solver.Solve(problem)) {
    return false;
  }

  problem.ApplyResultsToReconstruction(reconstruction);
  return true;
}

void RotationEstimator::InitializeFromMaximumSpanningTree(
    const ViewGraph& view_graph, colmap::Reconstruction& reconstruction) {
  // Here, we assume that largest connected component is already retrieved, so
  // we do not need to do that again. Compute maximum spanning tree.
  std::unordered_map<image_t, image_t> parents;
  const image_t root =
      ComputeMaximumSpanningTree(view_graph, reconstruction.Images(), parents);
  THROW_CHECK(reconstruction.Image(root).HasPose());

  // Iterate through the tree to initialize the rotation.
  // Establish child info.
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.HasPose()) continue;
    children.emplace(image_id, std::vector<image_t>());
  }
  for (auto& [child, parent] : parents) {
    if (root == child) continue;
    children[parent].emplace_back(child);
  }

  std::queue<image_t> indexes;
  indexes.push(root);

  std::unordered_map<image_t, Rigid3d> cams_from_world;
  while (!indexes.empty()) {
    image_t curr = indexes.front();
    indexes.pop();

    // Add all children into the tree.
    for (auto& child : children[curr]) indexes.push(child);
    // If it is root, then fix it to be the original estimation.
    if (curr == root) continue;

    // Directly use the relative pose for estimation rotation.
    // GetImagePair(parent, curr) returns curr_from_parent
    const ImagePair image_pair = view_graph.GetImagePair(parents[curr], curr);
    cams_from_world[curr].rotation =
        (image_pair.cam2_from_cam1 * cams_from_world[parents[curr]]).rotation;
  }

  InitializeRigRotationsFromImages(cams_from_world, reconstruction);
}

}  // namespace glomap
