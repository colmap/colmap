#include "glomap/estimators/rotation_averaging.h"

#include "colmap/geometry/pose.h"
#include "colmap/math/spanning_tree.h"

#include "glomap/estimators/rotation_averaging_impl.h"

#include <algorithm>
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
    const std::unordered_set<image_t>& image_ids,
    std::unordered_map<image_t, image_t>& parents) {
  // Build mapping between image_id and contiguous indices.
  std::unordered_map<image_t, int> image_id_to_idx;
  std::vector<image_t> idx_to_image_id;
  image_id_to_idx.reserve(image_ids.size());
  idx_to_image_id.reserve(image_ids.size());

  for (const image_t image_id : image_ids) {
    image_id_to_idx[image_id] = static_cast<int>(idx_to_image_id.size());
    idx_to_image_id.push_back(image_id);
  }

  // Build edges and weights from view graph.
  std::vector<std::pair<int, int>> edges;
  std::vector<float> weights;
  edges.reserve(view_graph.NumImagePairs());
  weights.reserve(view_graph.NumImagePairs());

  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const auto it1 = image_id_to_idx.find(image_id1);
    const auto it2 = image_id_to_idx.find(image_id2);
    if (it1 == image_id_to_idx.end() || it2 == image_id_to_idx.end()) {
      continue;
    }
    edges.emplace_back(it1->second, it2->second);
    weights.push_back(static_cast<float>(image_pair.inlier_matches.size()));
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

// Computes the largest connected component and returns image ids.
std::unordered_set<image_t> ComputeLargestConnectedComponentImageIds(
    const ViewGraph& view_graph,
    const colmap::Reconstruction& reconstruction,
    bool filter_unregistered) {
  const std::unordered_set<frame_t> frame_ids =
      view_graph.ComputeLargestConnectedFrameComponent(reconstruction,
                                                       filter_unregistered);

  std::unordered_set<image_t> image_ids;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (frame_ids.count(image.FrameId())) {
      image_ids.insert(image_id);
    }
  }
  return image_ids;
}

// Returns true if any camera in the reconstruction has unknown cam_from_rig.
bool HasUnknownCamsFromRig(const colmap::Reconstruction& reconstruction) {
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      if (!rig.MaybeSensorFromRig(sensor_id).has_value()) {
        return true;
      }
    }
  }
  return false;
}

// Creates an expanded reconstruction where cameras with unknown cam_from_rig
// are split into separate singleton rigs (each such camera becomes its own
// rig). This allows rotation averaging to estimate their orientations
// independently.
colmap::Reconstruction CreateExpandedReconstruction(
    const colmap::Reconstruction& reconstruction) {
  colmap::Reconstruction recon_expanded;

  // Add all cameras first (required before adding rigs).
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    recon_expanded.AddCamera(camera);
  }

  // Create expanded rigs with known sensors only.
  // Cameras with unknown cam_from_rig get their own singleton rigs.
  std::unordered_map<camera_t, rig_t> singleton_rig_ids;

  // First, find the max rig ID to avoid conflicts when creating singleton rigs.
  rig_t next_rig_id = 0;
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    next_rig_id = std::max(next_rig_id, rig_id + 1);
  }

  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    Rig rig_expanded;
    rig_expanded.SetRigId(rig_id);
    rig_expanded.AddRefSensor(rig.RefSensorId());

    for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      if (rig.MaybeSensorFromRig(sensor_id).has_value()) {
        rig_expanded.AddSensor(sensor_id, sensor_from_rig);
      } else {
        // Create singleton rig for this camera.
        const rig_t singleton_rig_id = next_rig_id++;
        Rig rig_singleton;
        rig_singleton.SetRigId(singleton_rig_id);
        rig_singleton.AddRefSensor(sensor_id);
        recon_expanded.AddRig(std::move(rig_singleton));
        singleton_rig_ids[sensor_id.id] = singleton_rig_id;
      }
    }
    recon_expanded.AddRig(std::move(rig_expanded));
  }

  frame_t next_frame_id = 0;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    next_frame_id = std::max(next_frame_id, frame_id + 1);
  }

  const Eigen::Quaterniond kUnknownRotation = Eigen::Quaterniond(
      Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN()));
  const Eigen::Vector3d kUnknownTranslation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const Rigid3d kUnknownPose(kUnknownRotation, kUnknownTranslation);

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    Frame frame_expanded;
    frame_expanded.SetFrameId(frame_id);
    frame_expanded.SetRigId(frame.RigId());
    if (frame.HasPose()) {
      frame_expanded.SetRigFromWorld(frame.RigFromWorld());
    } else {
      frame_expanded.SetRigFromWorld(kUnknownPose);
    }
    recon_expanded.AddFrame(std::move(frame_expanded));
  }

  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    Frame& frame_expanded = recon_expanded.Frame(frame_id);
    const Rig& original_rig = reconstruction.Rig(frame.RigId());

    for (const auto& data_id : frame.ImageIds()) {
      const auto& image = reconstruction.Image(data_id.id);

      Image image_expanded;
      image_expanded.SetImageId(image.ImageId());
      image_expanded.SetCameraId(image.CameraId());
      image_expanded.SetName(image.Name());

      // Check if camera belongs to this frame's rig (ref sensor or known
      // cam_from_rig).
      const bool belongs_to_frame_rig =
          original_rig.RefSensorId() == image.CameraPtr()->SensorId() ||
          original_rig.MaybeSensorFromRig(image.CameraPtr()->SensorId())
              .has_value();

      if (belongs_to_frame_rig) {
        // Camera belongs to this frame's rig.
        frame_expanded.AddDataId(image_expanded.DataId());
        image_expanded.SetFrameId(frame_id);
        recon_expanded.AddImage(std::move(image_expanded));
      } else {
        // Camera has its own singleton rig, create a new frame for it.
        const frame_t new_frame_id = next_frame_id++;
        Frame new_frame;
        new_frame.SetFrameId(new_frame_id);
        new_frame.SetRigId(singleton_rig_ids.at(image.CameraId()));
        new_frame.AddDataId(image_expanded.DataId());
        new_frame.SetRigFromWorld(kUnknownPose);
        recon_expanded.AddFrame(std::move(new_frame));

        image_expanded.SetFrameId(new_frame_id);
        recon_expanded.AddImage(std::move(image_expanded));
      }
    }
  }

  return recon_expanded;
}

}  // namespace

bool RotationEstimator::EstimateRotations(
    const ViewGraph& view_graph,
    const std::vector<colmap::PosePrior>& pose_priors,
    const std::unordered_set<image_t>& active_image_ids,
    colmap::Reconstruction& reconstruction) {
  if (options_.use_gravity && !AllSensorsFromRigKnown(reconstruction.Rigs())) {
    return false;
  }

  // Handle stratified solving for mixed gravity systems.
  if (options_.use_gravity && options_.use_stratified) {
    if (!MaybeSolveGravityAlignedSubset(
            view_graph, pose_priors, active_image_ids, reconstruction)) {
      return false;
    }
  }

  // Solve the full system.
  if (!SolveRotationAveraging(
          view_graph, pose_priors, active_image_ids, reconstruction)) {
    return false;
  }

  // Register frames with computed poses.
  for (const image_t image_id : active_image_ids) {
    const frame_t frame_id = reconstruction.Image(image_id).FrameId();
    THROW_CHECK(reconstruction.Frame(frame_id).HasPose());
    reconstruction.RegisterFrame(frame_id);
  }

  return true;
}

bool RotationEstimator::MaybeSolveGravityAlignedSubset(
    const ViewGraph& view_graph,
    const std::vector<colmap::PosePrior>& pose_priors,
    const std::unordered_set<image_t>& active_image_ids,
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
  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    if (!reconstruction.ExistsImage(image_id1) ||
        !reconstruction.ExistsImage(image_id2)) {
      continue;
    }
    if (!active_image_ids.count(image_id1) ||
        !active_image_ids.count(image_id2)) {
      continue;
    }

    num_total_pairs++;

    const auto it1 = image_to_pose_prior.find(image_id1);
    const auto it2 = image_to_pose_prior.find(image_id2);
    const bool image1_has_gravity =
        it1 != image_to_pose_prior.end() && it1->second->HasGravity();
    const bool image2_has_gravity =
        it2 != image_to_pose_prior.end() && it2->second->HasGravity();

    if (image1_has_gravity && image2_has_gravity) {
      gravity_view_graph.ImagePairs().emplace(
          pair_id, ImagePair(*image_pair.cam2_from_cam1));
    }
  }

  const size_t num_gravity_pairs = gravity_view_graph.NumImagePairs();
  LOG(INFO) << "Total image pairs: " << num_total_pairs
            << ", gravity image pairs: " << num_gravity_pairs;

  // Only solve if we have a meaningful subset.
  // Skip if no gravity pairs, or if most pairs (>95%) have gravity since
  // solving the subset separately provides little benefit over the full system.
  const bool should_solve =
      num_gravity_pairs > 0 && num_gravity_pairs <= num_total_pairs * 0.95;

  if (should_solve) {
    LOG(INFO) << "Solving subset 1-DOF rotation averaging problem";
    colmap::Reconstruction gravity_reconstruction(reconstruction);

    // Compute largest connected component for gravity subset.
    std::unordered_set<image_t> gravity_image_ids =
        ComputeLargestConnectedComponentImageIds(gravity_view_graph,
                                                 gravity_reconstruction,
                                                 /*filter_unregistered=*/false);
    gravity_view_graph.InvalidatePairsOutsideActiveImageIds(gravity_image_ids);

    if (!SolveRotationAveraging(gravity_view_graph,
                                pose_priors,
                                gravity_image_ids,
                                gravity_reconstruction)) {
      return false;
    }

    for (const auto& [gravity_frame_id, gravity_frame] :
         gravity_reconstruction.Frames()) {
      if (!gravity_frame.HasPose()) continue;
      reconstruction.Frame(gravity_frame_id)
          .SetRigFromWorld(gravity_frame.RigFromWorld());
    }

    for (const auto& [gravity_rig_id, gravity_rig] :
         gravity_reconstruction.Rigs()) {
      for (const auto& [sensor_id, sensor_from_rig] :
           gravity_rig.NonRefSensors()) {
        if (!gravity_rig.HasSensorFromRig(sensor_id)) continue;
        reconstruction.Rig(gravity_rig_id)
            .SetSensorFromRig(sensor_id, sensor_from_rig);
      }
    }
  }

  return true;
}

bool RotationEstimator::SolveRotationAveraging(
    const ViewGraph& view_graph,
    const std::vector<colmap::PosePrior>& pose_priors,
    const std::unordered_set<image_t>& active_image_ids,
    colmap::Reconstruction& reconstruction) {
  // Initialize rotations from maximum spanning tree.
  if (!options_.skip_initialization && !options_.use_gravity) {
    InitializeFromMaximumSpanningTree(
        view_graph, active_image_ids, reconstruction);
  }

  // Build the optimization problem.
  RotationAveragingProblem problem(
      view_graph, pose_priors, options_, active_image_ids, reconstruction);

  // Solve and apply results.
  RotationAveragingSolver solver(options_);
  if (!solver.Solve(problem)) {
    return false;
  }

  problem.ApplyResultsToReconstruction(reconstruction);
  return true;
}

void RotationEstimator::InitializeFromMaximumSpanningTree(
    const ViewGraph& view_graph,
    const std::unordered_set<image_t>& active_image_ids,
    colmap::Reconstruction& reconstruction) {
  // Compute maximum spanning tree over active images.
  std::unordered_map<image_t, image_t> parents;
  const image_t root =
      ComputeMaximumSpanningTree(view_graph, active_image_ids, parents);
  THROW_CHECK(active_image_ids.count(root));

  // Iterate through the tree to initialize the rotation.
  // Establish child info.
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!active_image_ids.count(image_id)) continue;
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
        (*image_pair.cam2_from_cam1 * cams_from_world[parents[curr]]).rotation;
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
      if (image.IsRefInFrame()) {
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
      if (image.IsRefInFrame()) {
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

  const Eigen::Vector3d kUnknownTranslation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  std::vector<double> weights;
  for (auto& [camera_id, rig_id_and_samples] : cam_from_rig_samples) {
    auto& [rig_id, samples] = rig_id_and_samples;
    weights.resize(samples.size(), 1.0);
    const Eigen::Quaterniond cam_from_rig =
        colmap::AverageQuaternions(samples, weights);
    reconstruction.Rig(rig_id).SetSensorFromRig(
        sensor_t(SensorType::CAMERA, camera_id),
        Rigid3d(cam_from_rig, kUnknownTranslation));
  }

  // Step 2: Compute rig_from_world for each frame by averaging across images.
  std::vector<Eigen::Quaterniond> rig_from_world_samples;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    rig_from_world_samples.clear();

    for (const auto& data_id : frame.ImageIds()) {
      if (!reconstruction.ExistsImage(data_id.id)) {
        continue;
      }

      const auto it = cams_from_world.find(data_id.id);
      if (it == cams_from_world.end()) {
        continue;
      }
      const auto& image = reconstruction.Image(data_id.id);

      if (image.IsRefInFrame()) {
        rig_from_world_samples.push_back(it->second.rotation);
      } else {
        const auto& maybe_cam_from_rig =
            reconstruction.Rig(frame.RigId())
                .MaybeSensorFromRig(image.CameraPtr()->SensorId());
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
          Rigid3d(rig_from_world, kUnknownTranslation));
    }
  }

  return true;
}

bool SolveRotationAveraging(const RotationEstimatorOptions& options,
                            ViewGraph& view_graph,
                            colmap::Reconstruction& reconstruction,
                            const std::vector<colmap::PosePrior>& pose_priors) {
  std::unordered_set<image_t> active_image_ids;

  // Step 1: Solve rotation averaging on the largest connected component.
  if (!HasUnknownCamsFromRig(reconstruction)) {
    // All cam_from_rig are known, solve directly.
    active_image_ids = ComputeLargestConnectedComponentImageIds(
        view_graph, reconstruction, options.filter_unregistered);

    if (active_image_ids.empty()) {
      LOG(ERROR) << "No connected components found";
      return false;
    }

    view_graph.InvalidatePairsOutsideActiveImageIds(active_image_ids);

    RotationEstimator rotation_estimator(options);
    if (!rotation_estimator.EstimateRotations(
            view_graph, pose_priors, active_image_ids, reconstruction)) {
      return false;
    }
  } else {
    // Some cam_from_rig are unknown. First solve on an expanded reconstruction
    // where each such camera is treated as an independent rig, then use the
    // results to initialize cam_from_rig before the final solve.
    LOG(INFO) << "Detected cameras with unknown cam_from_rig, "
                 "estimating rotations with these cameras as independent";

    // Step 1a: Create expanded reconstruction and solve.
    colmap::Reconstruction recon_expanded =
        CreateExpandedReconstruction(reconstruction);

    std::unordered_set<image_t> expanded_active_image_ids =
        ComputeLargestConnectedComponentImageIds(
            view_graph, recon_expanded, options.filter_unregistered);

    if (expanded_active_image_ids.empty()) {
      LOG(ERROR) << "No connected components found";
      return false;
    }

    view_graph.InvalidatePairsOutsideActiveImageIds(expanded_active_image_ids);

    RotationEstimator rotation_estimator_expanded(options);
    if (!rotation_estimator_expanded.EstimateRotations(
            view_graph,
            pose_priors,
            expanded_active_image_ids,
            recon_expanded)) {
      return false;
    }

    // Step 1b: Initialize cam_from_rig from expanded results.
    std::unordered_map<image_t, Rigid3d> expanded_cams_from_world;
    for (const auto& [image_id, image] : recon_expanded.Images()) {
      if (!image.HasPose()) continue;
      expanded_cams_from_world[image_id] = image.CamFromWorld();
    }

    LOG(INFO)
        << "Initializing cam_from_rig from preliminary rotation estimates";
    InitializeRigRotationsFromImages(expanded_cams_from_world, reconstruction);

    // Step 1c: Solve on original reconstruction with initialized cam_from_rig.
    active_image_ids = ComputeLargestConnectedComponentImageIds(
        view_graph, reconstruction, options.filter_unregistered);

    if (active_image_ids.empty()) {
      LOG(ERROR) << "No connected components found";
      return false;
    }

    view_graph.InvalidatePairsOutsideActiveImageIds(active_image_ids);

    RotationEstimatorOptions options_ra = options;
    options_ra.skip_initialization = true;
    options_ra.use_stratified = false;
    RotationEstimator rotation_estimator(options_ra);
    if (!rotation_estimator.EstimateRotations(
            view_graph, pose_priors, active_image_ids, reconstruction)) {
      return false;
    }
  }

  // Step 2: Filter outlier pairs by rotation error and update the active set.
  if (options.max_rotation_error_deg > 0) {
    view_graph.FilterByRelativeRotation(reconstruction,
                                        options.max_rotation_error_deg);

    // Recompute largest connected component among registered frames.
    active_image_ids = ComputeLargestConnectedComponentImageIds(
        view_graph, reconstruction, /*filter_unregistered=*/true);

    if (active_image_ids.empty()) {
      LOG(ERROR) << "No connected components found after filtering";
      return false;
    }

    view_graph.InvalidatePairsOutsideActiveImageIds(active_image_ids);

    // De-register frames outside the new active set.
    std::unordered_set<frame_t> active_frame_ids;
    for (const image_t image_id : active_image_ids) {
      active_frame_ids.insert(reconstruction.Image(image_id).FrameId());
    }
    for (const image_t image_id : reconstruction.RegImageIds()) {
      const frame_t frame_id = reconstruction.Image(image_id).FrameId();
      THROW_CHECK(reconstruction.Frame(frame_id).HasPose());
      if (!active_frame_ids.count(frame_id)) {
        reconstruction.DeRegisterFrame(frame_id);
      }
    }
  }

  return true;
}

}  // namespace glomap
