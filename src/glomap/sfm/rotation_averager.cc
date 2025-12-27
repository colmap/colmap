#include "glomap/sfm/rotation_averager.h"

#include "colmap/scene/reconstruction_io_utils.h"

#include "glomap/estimators/rotation_initializer.h"
#include "glomap/io/colmap_io.h"

namespace glomap {

bool SolveRotationAveraging(ViewGraph& view_graph,
                            colmap::Reconstruction& reconstruction,
                            std::vector<colmap::PosePrior>& pose_priors,
                            const RotationAveragerOptions& options) {
  view_graph.KeepLargestConnectedComponents(reconstruction);

  bool solve_1dof_system = options.use_gravity && options.use_stratified;

  ViewGraph view_graph_grav;
  image_pair_t total_pairs = 0;
  if (solve_1dof_system) {
    std::unordered_map<image_t, colmap::PosePrior*> image_to_pose_prior;
    for (auto& pose_prior : pose_priors) {
      if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
        const image_t image_id = pose_prior.corr_data_id.id;
        THROW_CHECK(image_to_pose_prior.emplace(image_id, &pose_prior).second)
            << "Duplicate pose prior for image " << image_id;
      }
    }

    // Prepare two sets: ones all with gravity, and one does not have gravity.
    // Solve them separately first, then solve them in a single system
    for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
      if (!image_pair.is_valid) {
        continue;
      }

      if (!reconstruction.Image(image_pair.image_id1).HasPose() ||
          !reconstruction.Image(image_pair.image_id2).HasPose()) {
        continue;
      }

      total_pairs++;

      const auto pose_prior1_it =
          image_to_pose_prior.find(image_pair.image_id1);
      const auto pose_prior2_it =
          image_to_pose_prior.find(image_pair.image_id2);
      const bool has_gravity1 = pose_prior1_it != image_to_pose_prior.end() &&
                                pose_prior1_it->second->HasGravity();
      const bool has_gravity2 = pose_prior2_it != image_to_pose_prior.end() &&
                                pose_prior2_it->second->HasGravity();

      if (has_gravity1 && has_gravity2) {
        view_graph_grav.image_pairs.emplace(
            pair_id,
            ImagePair(image_pair.image_id1,
                      image_pair.image_id2,
                      image_pair.cam2_from_cam1));
      }
    }
  }

  const size_t grav_pairs = view_graph_grav.image_pairs.size();

  LOG(INFO) << "Total image pairs: " << total_pairs
            << ", gravity image pairs: " << grav_pairs;

  // If there is no image pairs with gravity or most image pairs are with
  // gravity, then just run the 3dof version
  const bool status = grav_pairs == 0 || grav_pairs > total_pairs * 0.95;
  solve_1dof_system = solve_1dof_system && !status;

  if (solve_1dof_system) {
    // Run the 1dof optimization
    LOG(INFO) << "Solving subset 1DoF rotation averaging problem in the mixed "
                 "prior system";
    view_graph_grav.KeepLargestConnectedComponents(reconstruction);
    RotationEstimator rotation_estimator_grav(options);
    if (!rotation_estimator_grav.EstimateRotations(
            view_graph_grav, pose_priors, reconstruction)) {
      return false;
    }
    view_graph.KeepLargestConnectedComponents(reconstruction);
  }

  // By default, run trivial rotation averaging for cameras with unknown
  // cam_from_rig.
  std::unordered_set<camera_t> unknown_cams_from_rig;
  rig_t max_rig_id = 0;
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    max_rig_id = std::max(max_rig_id, rig_id);
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor_id.type != SensorType::CAMERA) continue;
      if (!rig.MaybeSensorFromRig(sensor_id).has_value()) {
        unknown_cams_from_rig.insert(sensor_id.id);
      }
    }
  }

  bool status_ra = false;
  // If the trivial rotation averaging is enabled, run it
  if (!unknown_cams_from_rig.empty() && !options.skip_initialization) {
    LOG(INFO) << "Running trivial rotation averaging for rigged cameras";
    // Create a temporary reconstruction for trivial rotation averaging
    colmap::Reconstruction recon_trivial;

    // Add all cameras first (required before adding rigs)
    for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
      recon_trivial.AddCamera(camera);
    }

    // For cameras with known cam_from_rig, create rigs with only those sensors.
    std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      Rig rig_trivial;
      rig_trivial.SetRigId(rig_id);
      rig_trivial.AddRefSensor(rig.RefSensorId());
      for (const auto& [sensor_id, sensor_from_rig] : rig.NonRefSensors()) {
        if (sensor_id.type != SensorType::CAMERA) continue;
        if (rig.MaybeSensorFromRig(sensor_id).has_value()) {
          rig_trivial.AddSensor(sensor_id, sensor_from_rig);
          camera_id_to_rig_id[sensor_id.id] = rig_id;
        }
      }
      camera_id_to_rig_id[rig.RefSensorId().id] = rig_id;
      recon_trivial.AddRig(rig_trivial);
    }

    // For each camera with unknown cam_from_rig, create a separate trivial rig.
    for (const auto& camera_id : unknown_cams_from_rig) {
      Rig rig_trivial;
      rig_trivial.SetRigId(++max_rig_id);
      rig_trivial.AddRefSensor(sensor_t(SensorType::CAMERA, camera_id));
      camera_id_to_rig_id[camera_id] = rig_trivial.RigId();
      recon_trivial.AddRig(rig_trivial);
    }

    frame_t max_frame_id = 0;
    for (const auto& [frame_id, _] : reconstruction.Frames()) {
      THROW_CHECK_NE(frame_id, colmap::kInvalidFrameId);
      max_frame_id = std::max(max_frame_id, frame_id);
    }
    max_frame_id++;

    const Eigen::Quaterniond kUnknownRotation = Eigen::Quaterniond(
        Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN()));
    const Eigen::Vector3d kUnknownTranslation =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    const Rigid3d kUnknownPose(kUnknownRotation, kUnknownTranslation);

    for (const auto& [frame_id, frame] : reconstruction.Frames()) {
      Frame frame_trivial;
      frame_trivial.SetFrameId(frame_id);
      frame_trivial.SetRigId(frame.RigId());
      frame_trivial.SetRigFromWorld(kUnknownPose);
      recon_trivial.AddFrame(frame_trivial);
    }

    for (const auto& [frame_id, frame] : reconstruction.Frames()) {
      Frame& frame_trivial = recon_trivial.Frame(frame_id);
      frame_trivial.SetRigPtr(recon_trivial.ExistsRig(frame.RigId())
                                  ? &recon_trivial.Rig(frame.RigId())
                                  : nullptr);

      for (const auto& data_id : frame.ImageIds()) {
        const auto& image = reconstruction.Image(data_id.id);
        if (!image.HasPose()) {
          continue;
        }

        Image image_trivial;
        image_trivial.SetImageId(image.ImageId());
        image_trivial.SetCameraId(image.CameraId());
        image_trivial.SetName(image.Name());

        if (unknown_cams_from_rig.count(image.CameraId()) == 0) {
          // Image belongs to the existing frame
          frame_trivial.AddDataId(image_trivial.DataId());
          image_trivial.SetFrameId(frame_id);
          recon_trivial.AddImage(std::move(image_trivial));
        } else {
          // If cam_from_rig is unknown, create a trivial frame with the correct
          // rig_id (not camera_id, since we created separate rigs for these)
          const frame_t new_frame_id = ++max_frame_id;
          const rig_t rig_id = camera_id_to_rig_id.at(image.CameraId());
          Frame new_frame;
          new_frame.SetFrameId(new_frame_id);
          new_frame.SetRigId(rig_id);
          new_frame.AddDataId(image_trivial.DataId());
          new_frame.SetRigFromWorld(kUnknownPose);
          recon_trivial.AddFrame(new_frame);

          image_trivial.SetFrameId(new_frame_id);
          recon_trivial.AddImage(std::move(image_trivial));
        }
      }
    }

    view_graph.KeepLargestConnectedComponents(recon_trivial);

    // Run the trivial rotation averaging
    RotationEstimatorOptions options_trivial = options;
    options_trivial.skip_initialization = options.skip_initialization;
    RotationEstimator rotation_estimator_trivial(options_trivial);
    rotation_estimator_trivial.EstimateRotations(
        view_graph, pose_priors, recon_trivial);

    // Collect the results
    std::unordered_map<image_t, Rigid3d> trivial_cams_from_world;
    for (const auto& [image_id, image] : recon_trivial.Images()) {
      if (!image.HasPose()) continue;
      trivial_cams_from_world[image_id] = image.CamFromWorld();
    }

    LOG(INFO) << "Creating trivial rigs";
    InitializeRigRotationsFromImages(trivial_cams_from_world, reconstruction);

    RotationEstimatorOptions options_ra = options;
    options_ra.skip_initialization = true;
    RotationEstimator rotation_estimator(options_ra);
    status_ra = rotation_estimator.EstimateRotations(
        view_graph, pose_priors, reconstruction);
    view_graph.KeepLargestConnectedComponents(reconstruction);
  } else {
    RotationAveragerOptions options_ra = options;
    // For cases where there are some cameras without known cam_from_rig
    // transformation, we need to run the rotation averaging with the
    // skip_initialization flag set to false for convergence
    if (unknown_cams_from_rig.size() > 0) {
      options_ra.skip_initialization = false;
    }

    RotationEstimator rotation_estimator(options_ra);
    status_ra = rotation_estimator.EstimateRotations(
        view_graph, pose_priors, reconstruction);
    view_graph.KeepLargestConnectedComponents(reconstruction);
  }
  return status_ra;
}

}  // namespace glomap
