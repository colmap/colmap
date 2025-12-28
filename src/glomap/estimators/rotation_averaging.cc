#include "glomap/estimators/rotation_averaging.h"

#include "colmap/geometry/pose.h"
#include "colmap/math/spanning_tree.h"

#include "glomap/estimators/rotation_averaging_impl.h"

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

  // Handle stratified solving for mixed gravity systems.
  if (options_.use_gravity && options_.use_stratified) {
    if (!MaybeSolveGravityAlignedSubset(
            view_graph, pose_priors, reconstruction)) {
      return false;
    }
  }

  // Solve the full system.
  return SolveRotationAveraging(view_graph, pose_priors, reconstruction);
}

bool RotationEstimator::MaybeSolveGravityAlignedSubset(
    const ViewGraph& view_graph,
    const std::vector<colmap::PosePrior>& pose_priors,
    colmap::Reconstruction& reconstruction) {
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
            gravity_view_graph, pose_priors, reconstruction)) {
      return false;
    }
  }

  return true;
}

bool RotationEstimator::SolveRotationAveraging(
    const ViewGraph& view_graph,
    const std::vector<colmap::PosePrior>& pose_priors,
    colmap::Reconstruction& reconstruction) {
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

  InitializeRigRotationsFromImages(cams_from_world, reconstruction);
}

bool InitializeRigRotationsFromImages(
    const std::unordered_map<image_t, Rigid3d>& cams_from_world,
    colmap::Reconstruction& reconstruction) {
  // Step 1: Estimate cam_from_rig for cameras with unknown calibration.
  // Collect samples across frames, then average.
  std::unordered_map<camera_t,
                     std::pair<rig_t, std::vector<Eigen::Quaterniond>>>
      cam_from_rig_samples;

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    // Find the rotation of the reference image.
    const Eigen::Quaterniond* ref_rotation = nullptr;
    for (const auto& data_id : frame.ImageIds()) {
      const auto& image = reconstruction.Image(data_id.id);
      if (image.HasPose() && image.IsRefInFrame()) {
        const auto it = cams_from_world.find(data_id.id);
        if (it != cams_from_world.end()) {
          ref_rotation = &it->second.rotation;
        }
        break;
      }
    }
    if (ref_rotation == nullptr) {
      continue;
    }

    // Collect cam_from_rig samples for non-reference cameras.
    for (const auto& data_id : frame.ImageIds()) {
      const auto& image = reconstruction.Image(data_id.id);
      if (!image.HasPose() || image.IsRefInFrame()) {
        continue;
      }

      const auto it = cams_from_world.find(data_id.id);
      if (it == cams_from_world.end()) {
        continue;
      }

      auto& [rig_id, rotations] = cam_from_rig_samples[image.CameraId()];
      rig_id = frame.RigId();
      rotations.push_back(it->second.rotation * ref_rotation->inverse());
    }
  }

  const Eigen::Vector3d kNaNTranslation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  std::vector<double> weights;
  for (auto& [camera_id, rig_id_and_samples] : cam_from_rig_samples) {
    auto& [rig_id, samples] = rig_id_and_samples;
    weights.resize(samples.size(), 1.0);
    const Eigen::Quaterniond cam_from_rig =
        colmap::AverageQuaternions(samples, weights);
    reconstruction.Rig(rig_id).SetSensorFromRig(
        sensor_t(SensorType::CAMERA, camera_id),
        Rigid3d(cam_from_rig, kNaNTranslation));
  }

  // Step 2: Compute rig_from_world for each frame by averaging across images.
  std::vector<Eigen::Quaterniond> rig_from_world_samples;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    rig_from_world_samples.clear();

    for (const auto& data_id : frame.ImageIds()) {
      if (!reconstruction.ExistsImage(data_id.id)) {
        continue;
      }

      const auto& image = reconstruction.Image(data_id.id);
      if (!image.HasPose()) {
        continue;
      }

      const auto it = cams_from_world.find(data_id.id);
      if (it == cams_from_world.end()) {
        continue;
      }

      if (image.IsRefInFrame()) {
        rig_from_world_samples.push_back(it->second.rotation);
      } else {
        const auto& maybe_cam_from_rig =
            reconstruction.Rig(frame.RigId())
                .MaybeSensorFromRig(
                    sensor_t(SensorType::CAMERA, image.CameraId()));
        if (!maybe_cam_from_rig.has_value()) {
          continue;
        }
        rig_from_world_samples.push_back(
            maybe_cam_from_rig.value().rotation.inverse() *
            it->second.rotation);
      }
    }

    if (!rig_from_world_samples.empty()) {
      weights.resize(rig_from_world_samples.size(), 1.0);
      const Eigen::Quaterniond rig_from_world =
          colmap::AverageQuaternions(rig_from_world_samples, weights);
      reconstruction.Frame(frame_id).SetRigFromWorld(
          Rigid3d(rig_from_world, kNaNTranslation));
    }
  }

  return true;
}

}  // namespace glomap
