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
    colmap::Reconstruction& reconstruction,
    const std::vector<colmap::PosePrior>& pose_priors) {
  if (options_.use_gravity && !AllSensorsFromRigKnown(reconstruction.Rigs())) {
    return false;
  }

  // Handle stratified solving for mixed gravity systems.
  if (options_.use_gravity && options_.use_stratified) {
    if (!MaybeSolveGravityAlignedSubset(
            view_graph, reconstruction, pose_priors)) {
      return false;
    }
  }

  // Solve the full system.
  return SolveRotationAveraging(view_graph, reconstruction, pose_priors);
}

bool RotationEstimator::MaybeSolveGravityAlignedSubset(
    const ViewGraph& view_graph,
    colmap::Reconstruction& reconstruction,
    const std::vector<colmap::PosePrior>& pose_priors) {
  // Build map from image to pose prior.
  std::unordered_map<image_t, const colmap::PosePrior*> image_to_pose_prior;
  for (const auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      image_to_pose_prior[pose_prior.corr_data_id.id] = &pose_prior;
    }
  }

  // Separate pairs into gravity-aligned subset.
  ViewGraph gravity_view_graph;
  size_t num_total_pairs = 0;
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    if (!reconstruction.ExistsImage(image_pair.image_id1) ||
        !reconstruction.ExistsImage(image_pair.image_id2)) {
      continue;
    }
    if (!reconstruction.Image(image_pair.image_id1).HasPose() ||
        !reconstruction.Image(image_pair.image_id2).HasPose()) {
      continue;
    }

    num_total_pairs++;

    const auto it1 = image_to_pose_prior.find(image_pair.image_id1);
    const auto it2 = image_to_pose_prior.find(image_pair.image_id2);
    const bool image1_has_gravity =
        it1 != image_to_pose_prior.end() && it1->second->HasGravity();
    const bool image2_has_gravity =
        it2 != image_to_pose_prior.end() && it2->second->HasGravity();

    if (image1_has_gravity && image2_has_gravity) {
      gravity_view_graph.image_pairs.emplace(
          pair_id,
          ImagePair(image_pair.image_id1,
                    image_pair.image_id2,
                    image_pair.cam2_from_cam1));
    }
  }

  const size_t num_gravity_pairs = gravity_view_graph.image_pairs.size();
  LOG(INFO) << "Total image pairs: " << num_total_pairs
            << ", gravity image pairs: " << num_gravity_pairs;

  // Only solve if we have a meaningful subset.
  // Skip if no gravity pairs, or if most pairs (>95%) have gravity since
  // solving the subset separately provides little benefit over the full system.
  const bool should_solve =
      num_gravity_pairs > 0 && num_gravity_pairs <= num_total_pairs * 0.95;

  if (should_solve) {
    LOG(INFO) << "Solving subset 1-DOF rotation averaging problem";
    if (!SolveRotationAveraging(
            gravity_view_graph, reconstruction, pose_priors)) {
      return false;
    }
  }

  return true;
}

bool RotationEstimator::SolveRotationAveraging(
    const ViewGraph& view_graph,
    colmap::Reconstruction& reconstruction,
    const std::vector<colmap::PosePrior>& pose_priors) {
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
      // 1_R_w = 2_R_1^T * 2_R_w
      cams_from_world[curr].rotation =
          (Inverse(image_pair.cam2_from_cam1) * cams_from_world[parents[curr]])
              .rotation;
    } else {
      // 2_R_w = 2_R_1 * 1_R_w
      cams_from_world[curr].rotation =
          (image_pair.cam2_from_cam1 * cams_from_world[parents[curr]]).rotation;
    }
  }

  ConvertRotationsFromImageToRig(cams_from_world, reconstruction);
}

}  // namespace glomap
