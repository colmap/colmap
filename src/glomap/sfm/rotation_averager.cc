#include "glomap/sfm/rotation_averager.h"

#include "glomap/estimators/rotation_initializer.h"
#include "glomap/io/colmap_converter.h"

namespace glomap {

bool SolveRotationAveraging(ViewGraph& view_graph,
                            std::unordered_map<rig_t, Rig>& rigs,
                            std::unordered_map<frame_t, Frame>& frames,
                            std::unordered_map<image_t, Image>& images,
                            std::vector<colmap::PosePrior>& pose_priors,
                            const RotationAveragerOptions& options) {
  view_graph.KeepLargestConnectedComponents(frames, images);

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

      if (!images[image_pair.image_id1].HasPose() ||
          !images[image_pair.image_id2].HasPose()) {
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
    view_graph_grav.KeepLargestConnectedComponents(frames, images);
    RotationEstimator rotation_estimator_grav(options);
    if (!rotation_estimator_grav.EstimateRotations(
            view_graph_grav, rigs, frames, images, pose_priors)) {
      return false;
    }
    view_graph.KeepLargestConnectedComponents(frames, images);
  }

  // By default, run trivial rotation averaging for cameras with unknown
  // cam_from_rig.
  std::unordered_set<camera_t> unknown_cams_from_rig;
  rig_t max_rig_id = 0;
  for (const auto& [rig_id, rig] : rigs) {
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
    // Create a rig for each camera
    std::unordered_map<rig_t, Rig> rigs_trivial;
    std::unordered_map<frame_t, Frame> frames_trivial;
    std::unordered_map<image_t, Image> images_trivial;

    // For cameras with known cam_from_rig, create rigs with only those sensors.
    std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
    for (const auto& [rig_id, rig] : rigs) {
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
      rigs_trivial[rig_trivial.RigId()] = std::move(rig_trivial);
    }

    // For each camera with unknown cam_from_rig, create a separate trivial rig.
    for (const auto& camera_id : unknown_cams_from_rig) {
      Rig rig_trivial;
      rig_trivial.SetRigId(++max_rig_id);
      rig_trivial.AddRefSensor(sensor_t(SensorType::CAMERA, camera_id));
      camera_id_to_rig_id[camera_id] = rig_trivial.RigId();
      rigs_trivial[rig_trivial.RigId()] = std::move(rig_trivial);
    }

    frame_t max_frame_id = 0;
    for (const auto& [frame_id, _] : frames) {
      THROW_CHECK_NE(frame_id, colmap::kInvalidFrameId);
      max_frame_id = std::max(max_frame_id, frame_id);
    }
    max_frame_id++;

    const Eigen::Quaterniond kUnknownRotation = Eigen::Quaterniond(
        Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN()));
    const Eigen::Vector3d kUnknownTranslation =
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    const Rigid3d kUnknownPose(kUnknownRotation, kUnknownTranslation);

    for (auto& [frame_id, frame] : frames) {
      Frame& frame_trivial = frames_trivial[frame_id];
      frame_trivial.SetFrameId(frame_id);
      frame_trivial.SetRigId(frame.RigId());
      frame_trivial.SetRigPtr(rigs_trivial.find(frame.RigId()) !=
                                      rigs_trivial.end()
                                  ? &rigs_trivial[frame.RigId()]
                                  : nullptr);
      frame_trivial.SetRigFromWorld(kUnknownPose);

      for (const auto& data_id : frame.ImageIds()) {
        const auto& image = images.at(data_id.id);
        if (!image.HasPose()) {
          continue;
        }

        auto& image_trivial = images_trivial[image.ImageId()];
        image_trivial.SetImageId(image.ImageId());
        image_trivial.SetCameraId(image.CameraId());
        image_trivial.SetName(image.Name());

        if (unknown_cams_from_rig.count(image_trivial.CameraId()) == 0) {
          frame_trivial.AddDataId(image_trivial.DataId());
          image_trivial.SetFrameId(frame_id);
          image_trivial.SetFramePtr(&frame_trivial);
        } else {
          // If cam_from_rig is unknown, then create a trivial frame.
          CreateFrameForImage(kUnknownPose,
                              image_trivial,
                              rigs_trivial,
                              frames_trivial,
                              camera_id_to_rig_id[image.CameraId()],
                              max_frame_id);
          max_frame_id++;
        }
      }
    }

    view_graph.KeepLargestConnectedComponents(frames_trivial, images_trivial);

    // Run the trivial rotation averaging
    RotationEstimatorOptions options_trivial = options;
    options_trivial.skip_initialization = options.skip_initialization;
    RotationEstimator rotation_estimator_trivial(options_trivial);
    rotation_estimator_trivial.EstimateRotations(
        view_graph, rigs_trivial, frames_trivial, images_trivial, pose_priors);

    // Collect the results
    std::unordered_map<image_t, Rigid3d> trivial_cams_from_world;
    for (const auto& [image_id, image] : images_trivial) {
      if (!image.HasPose()) continue;
      trivial_cams_from_world[image_id] = image.CamFromWorld();
    }

    LOG(INFO) << "Creating trivial rigs";
    ConvertRotationsFromImageToRig(
        trivial_cams_from_world, images, rigs, frames);

    RotationEstimatorOptions options_ra = options;
    options_ra.skip_initialization = true;
    RotationEstimator rotation_estimator(options_ra);
    status_ra = rotation_estimator.EstimateRotations(
        view_graph, rigs, frames, images, pose_priors);
    view_graph.KeepLargestConnectedComponents(frames, images);
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
        view_graph, rigs, frames, images, pose_priors);
    view_graph.KeepLargestConnectedComponents(frames, images);
  }
  return status_ra;
}

}  // namespace glomap
