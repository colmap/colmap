#include "glomap/estimators/rotation_averaging.h"

#include "colmap/geometry/pose.h"

#include "glomap/estimators/rotation_averaging_impl.h"
#include "glomap/estimators/rotation_initializer.h"
#include "glomap/math/tree.h"

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
  const image_t root = MaximumSpanningTree(
      view_graph, reconstruction.Images(), parents, WeightType::INLIER_NUM);
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
    const ImagePair& image_pair = view_graph.image_pairs.at(
        colmap::ImagePairToPairId(curr, parents[curr]));
    if (image_pair.image_id1 == curr) {
      cams_from_world[curr].rotation =
          (Inverse(image_pair.cam2_from_cam1) * cams_from_world[parents[curr]])
              .rotation;
    } else {
      cams_from_world[curr].rotation =
          (image_pair.cam2_from_cam1 * cams_from_world[parents[curr]]).rotation;
    }
  }

  ConvertRotationsFromImageToRig(cams_from_world, reconstruction);
}

}  // namespace glomap
