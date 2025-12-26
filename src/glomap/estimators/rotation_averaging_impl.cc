#include "glomap/estimators/rotation_averaging_impl.h"

#include "colmap/geometry/pose.h"
#include "colmap/optim/least_absolute_deviations.h"

#include <limits>
#include <random>

#include <Eigen/CholmodSupport>

namespace glomap {
namespace {

// Computes the 1-DOF residual for gravity-aligned rotation constraints.
// Returns (angle_2 - angle_1) - angle_12, wrapped to [-π, π] with jitter
// near boundaries to avoid local minima.
double ComputeGravityAligned1DOFResidual(std::mt19937& rng,
                                         double angle_12,
                                         double angle_1,
                                         double angle_2) {
  double residual =
      std::remainder((angle_2 - angle_1) - angle_12, 2 * EIGEN_PI);

  // Inject random noise if the angle is too close to the boundary to break the
  // possible balance at the local minima.
  constexpr double kEps = 0.01;
  if (std::abs(residual) > EIGEN_PI - kEps) {
    std::uniform_real_distribution<double> dist(0.0, kEps);
    const double jitter = dist(rng);
    if (residual < 0) {
      residual += jitter;
    } else {
      residual -= jitter;
    }
  }

  return residual;
}

std::unordered_map<frame_t, const colmap::PosePrior*> ExtractFrameToPosePrior(
    const std::unordered_map<image_t, Image>& images,
    const std::vector<colmap::PosePrior>& pose_priors) {
  std::unordered_map<image_t, frame_t> image_to_frame;
  image_to_frame.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    image_to_frame[image_id] = image.FrameId();
  }

  std::unordered_map<frame_t, const colmap::PosePrior*> frame_to_pose_prior;
  for (const auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      const image_t image_id = pose_prior.corr_data_id.id;
      const auto it = images.find(image_id);
      if (it == images.end()) continue;
      const Image& image = it->second;
      if (image.IsRefInFrame()) {
        const frame_t frame_id = image_to_frame.at(pose_prior.corr_data_id.id);
        THROW_CHECK(frame_to_pose_prior.emplace(frame_id, &pose_prior).second)
            << "Duplicate pose prior for frame" << frame_id;
      }
    }
  }

  return frame_to_pose_prior;
}

const Eigen::Vector3d* GetFrameGravityOrNull(
    const std::unordered_map<frame_t, const colmap::PosePrior*>&
        frame_to_pose_prior,
    frame_t frame_id) {
  auto it = frame_to_pose_prior.find(frame_id);
  if (it == frame_to_pose_prior.end() || !it->second->HasGravity()) {
    return nullptr;
  }
  return &it->second->gravity;
}

}  // namespace

RotationAveragingProblem::RotationAveragingProblem(
    const ViewGraph& view_graph,
    colmap::Reconstruction& reconstruction,
    const std::vector<colmap::PosePrior>& pose_priors,
    const RotationEstimatorOptions& options)
    : options_(options) {
  frame_to_pose_prior_ =
      ExtractFrameToPosePrior(reconstruction.Images(), pose_priors);

  const size_t num_params = AllocateParameters(reconstruction);
  BuildPairConstraints(view_graph, reconstruction);
  BuildConstraintMatrix(num_params, view_graph, reconstruction);
}

bool RotationAveragingProblem::HasFrameGravity(frame_t frame_id) const {
  const auto it = frame_to_pose_prior_.find(frame_id);
  return options_.use_gravity && it != frame_to_pose_prior_.end() &&
         it->second->HasGravity();
}

size_t RotationAveragingProblem::AllocateParameters(
    const colmap::Reconstruction& reconstruction) {
  // Build mapping from camera to rig for all registered images.
  camera_id_to_param_idx_.reserve(reconstruction.NumCameras());
  estimated_rotations_.resize(6 * reconstruction.NumImages());

  std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    for (const auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (!reconstruction.ExistsImage(image_id)) continue;
      const auto& image = reconstruction.Image(image_id);
      if (!image.HasPose()) continue;
      camera_id_to_rig_id[image.CameraId()] = frame.RigId();
      // Cache image_id to frame_id mapping.
      image_id_to_frame_id_[image_id] = frame_id;
    }
  }

  // Identify cameras that need cam_from_rig estimation
  // (non-reference cameras without calibrated extrinsics).
  std::unordered_map<camera_t, Eigen::AngleAxisd> cam_from_rig_rotations;
  for (auto& [camera_id, rig_id] : camera_id_to_rig_id) {
    sensor_t sensor_id(SensorType::CAMERA, camera_id);
    if (reconstruction.Rig(rig_id).IsRefSensor(sensor_id)) continue;

    auto cam_from_rig =
        reconstruction.Rig(rig_id).MaybeSensorFromRig(sensor_id);
    if (!cam_from_rig.has_value() ||
        cam_from_rig.value().translation.hasNaN()) {
      if (camera_id_to_param_idx_.find(camera_id) ==
          camera_id_to_param_idx_.end()) {
        // Mark for later allocation (actual index set below).
        camera_id_to_param_idx_[camera_id] = -1;
        if (cam_from_rig.has_value()) {
          cam_from_rig_rotations[camera_id] =
              Eigen::AngleAxisd(cam_from_rig->rotation);
        }
      }
    }
  }

  // Allocate frame parameters and cache frame info.
  size_t num_params = 0;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (!frame.HasPose()) continue;
    frame_id_to_param_idx_[frame_id] = num_params;

    const Eigen::Vector3d* frame_gravity =
        GetFrameGravityOrNull(frame_to_pose_prior_, frame_id);
    const bool has_gravity = HasFrameGravity(frame_id);

    // Cache camera_id -> frame_id mapping for UpdateState cam_from_rig
    // averaging.
    for (const auto& data_id : frame.ImageIds()) {
      if (!reconstruction.ExistsImage(data_id.id)) continue;
      const auto& image = reconstruction.Image(data_id.id);
      if (camera_id_to_param_idx_.find(image.CameraId()) !=
          camera_id_to_param_idx_.end()) {
        camera_to_frame_ids_[image.CameraId()].push_back(frame_id);
      }
    }

    if (has_gravity) {
      // Gravity-aligned frame: 1-DOF (Y-axis rotation only).
      estimated_rotations_[num_params] = colmap::YAxisAngleFromRotation(
          colmap::GravityAlignedRotation(*frame_gravity).transpose() *
          frame.RigFromWorld().rotation.toRotationMatrix());
      num_params++;

      // Use first gravity-aligned frame as fixed frame.
      if (fixed_frame_id_ == colmap::kInvalidFrameId) {
        fixed_frame_rotation_ =
            Eigen::Vector3d(0, estimated_rotations_[num_params - 1], 0);
        fixed_frame_id_ = frame_id;
        num_gauge_fixing_residuals_ = 1;
      }
    } else {
      // General frame: 3-DOF.
      Eigen::AngleAxisd rig_from_world;
      if (frame.MaybeRigFromWorld().has_value()) {
        rig_from_world = Eigen::AngleAxisd(frame.RigFromWorld().rotation);
      } else {
        rig_from_world = Eigen::AngleAxisd::Identity();
      }
      estimated_rotations_.segment(num_params, 3) =
          rig_from_world.angle() * rig_from_world.axis();
      num_params += 3;
    }
  }

  // Allocate camera parameters (for unknown cam_from_rig rotations).
  for (auto& [camera_id, camera_param_idx] : camera_id_to_param_idx_) {
    camera_id_to_param_idx_[camera_id] = num_params;
    if (const auto it = cam_from_rig_rotations.find(camera_id);
        it != cam_from_rig_rotations.end()) {
      estimated_rotations_.segment(num_params, 3) =
          it->second.angle() * it->second.axis();
    } else {
      estimated_rotations_.segment(num_params, 3) = Eigen::Vector3d::Zero();
    }
    num_params += 3;
  }

  // If no gravity-aligned frame found, use first registered frame as fixed.
  if (fixed_frame_id_ == colmap::kInvalidFrameId) {
    for (const auto& [frame_id, frame] : reconstruction.Frames()) {
      if (!frame.HasPose()) continue;
      fixed_frame_id_ = frame_id;
      const Eigen::AngleAxisd rig_from_world(frame.RigFromWorld().rotation);
      fixed_frame_rotation_ = rig_from_world.angle() * rig_from_world.axis();
      num_gauge_fixing_residuals_ = 3;
      break;
    }
  }

  estimated_rotations_.conservativeResize(num_params);
  return num_params;
}

void RotationAveragingProblem::BuildPairConstraints(
    const ViewGraph& view_graph, const colmap::Reconstruction& reconstruction) {
  int gravity_aligned_count = 0;

  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    const auto& image1 = reconstruction.Image(image_pair.image_id1);
    const auto& image2 = reconstruction.Image(image_pair.image_id2);
    const auto& frame1 = *image1.FramePtr();
    const auto& frame2 = *image2.FramePtr();

    const int frame_param_idx1 = frame_id_to_param_idx_[frame1.FrameId()];
    const int frame_param_idx2 = frame_id_to_param_idx_[frame2.FrameId()];

    // Get known cam_from_rig transforms (nullopt for reference cameras or
    // cameras with unknown cam_from_rig that need to be estimated).
    std::optional<Rigid3d> cam1_from_rig1, cam2_from_rig2;

    if (!image1.IsRefInFrame()) {
      if (camera_id_to_param_idx_.find(image1.CameraId()) ==
          camera_id_to_param_idx_.end()) {
        cam1_from_rig1 =
            reconstruction.Rig(frame1.RigId())
                .SensorFromRig(sensor_t(SensorType::CAMERA, image1.CameraId()));
      }
    }
    if (!image2.IsRefInFrame()) {
      if (camera_id_to_param_idx_.find(image2.CameraId()) ==
          camera_id_to_param_idx_.end()) {
        cam2_from_rig2 =
            reconstruction.Rig(frame2.RigId())
                .SensorFromRig(sensor_t(SensorType::CAMERA, image2.CameraId()));
      }
    }

    // Skip self-loops within the same frame when both cam_from_rig are known.
    if (cam1_from_rig1.has_value() && cam2_from_rig2.has_value() &&
        frame_param_idx1 == frame_param_idx2) {
      continue;
    }

    // Compute relative rotation between rigs.
    Eigen::Matrix3d R_cam2_from_cam1 =
        (cam2_from_rig2.value_or(Rigid3d()).rotation.inverse() *
         image_pair.cam2_from_cam1.rotation *
         cam1_from_rig1.value_or(Rigid3d()).rotation)
            .toRotationMatrix();

    const Eigen::Vector3d* frame_gravity1 =
        GetFrameGravityOrNull(frame_to_pose_prior_, frame1.FrameId());
    const Eigen::Vector3d* frame_gravity2 =
        GetFrameGravityOrNull(frame_to_pose_prior_, frame2.FrameId());

    // Apply gravity alignment transformations if available.
    if (options_.use_gravity) {
      if (frame_gravity1 != nullptr) {
        R_cam2_from_cam1 =
            R_cam2_from_cam1 * colmap::GravityAlignedRotation(*frame_gravity1);
      }
      if (frame_gravity2 != nullptr) {
        R_cam2_from_cam1 =
            colmap::GravityAlignedRotation(*frame_gravity2).transpose() *
            R_cam2_from_cam1;
      }
    }

    // Create constraint based on gravity availability.
    PairConstraint& constraint = pair_constraints_[pair_id];
    constraint.image_id1 = image_pair.image_id1;
    constraint.image_id2 = image_pair.image_id2;
    if (options_.use_gravity && frame_gravity1 != nullptr &&
        frame_gravity2 != nullptr) {
      // Both frames have gravity: use 1-DOF constraint.
      gravity_aligned_count++;
      const Eigen::Vector3d aa =
          colmap::RotationMatrixToAngleAxis(R_cam2_from_cam1);
      constraint.constraint =
          GravityAligned1DOF{aa[1], aa[0] * aa[0] + aa[2] * aa[2]};
    } else {
      // General case: use 3-DOF constraint.
      constraint.constraint = Full3DOF{R_cam2_from_cam1};
    }
  }

  VLOG(2) << gravity_aligned_count << " image pairs are gravity aligned";
}

void RotationAveragingProblem::BuildConstraintMatrix(
    size_t num_params,
    const ViewGraph& view_graph,
    const colmap::Reconstruction& reconstruction) {
  std::vector<Eigen::Triplet<double>> coeffs;
  coeffs.reserve(pair_constraints_.size() * 6 + 3);

  std::vector<double> weights;
  if (options_.use_weight) {
    weights.reserve(3 * view_graph.image_pairs.size());
  }

  size_t curr_row = 0;

  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    if (pair_constraints_.find(pair_id) == pair_constraints_.end()) continue;

    const auto& image1 = reconstruction.Image(image_pair.image_id1);
    const auto& image2 = reconstruction.Image(image_pair.image_id2);
    const auto& frame1 = *image1.FramePtr();
    const auto& frame2 = *image2.FramePtr();

    if (!frame1.HasPose() || !frame2.HasPose()) continue;

    const int frame_param_idx1 = frame_id_to_param_idx_[frame1.FrameId()];
    const int frame_param_idx2 = frame_id_to_param_idx_[frame2.FrameId()];

    // Look up camera parameter indices (-1 if cam_from_rig is known).
    int cam1_param_idx = -1;
    int cam2_param_idx = -1;
    if (auto it = camera_id_to_param_idx_.find(image1.CameraId());
        it != camera_id_to_param_idx_.end()) {
      cam1_param_idx = it->second;
    }
    if (auto it = camera_id_to_param_idx_.find(image2.CameraId());
        it != camera_id_to_param_idx_.end()) {
      cam2_param_idx = it->second;
    }

    PairConstraint& constraint = pair_constraints_[pair_id];
    constraint.row_index = curr_row;
    constraint.cam1_from_rig_param_idx = cam1_param_idx;
    constraint.cam2_from_rig_param_idx = cam2_param_idx;

    const double pair_weight = image_pair.weight >= 0 ? image_pair.weight : 1.0;

    if (std::holds_alternative<GravityAligned1DOF>(constraint.constraint)) {
      // 1-DOF constraint: single row.
      coeffs.emplace_back(curr_row, frame_param_idx1, -1);
      coeffs.emplace_back(curr_row, frame_param_idx2, 1);
      if (options_.use_weight) {
        weights.push_back(pair_weight);
      }
      curr_row++;
    } else {
      // 3-DOF constraint: three rows.
      const Eigen::Vector3d* frame_gravity1 =
          GetFrameGravityOrNull(frame_to_pose_prior_, frame1.FrameId());
      const Eigen::Vector3d* frame_gravity2 =
          GetFrameGravityOrNull(frame_to_pose_prior_, frame2.FrameId());

      if (!options_.use_gravity || frame_gravity1 == nullptr) {
        for (int i = 0; i < 3; i++) {
          coeffs.emplace_back(curr_row + i, frame_param_idx1 + i, -1);
        }
      } else {
        // Gravity-aligned frame1: only Y-axis contributes.
        coeffs.emplace_back(curr_row + 1, frame_param_idx1, -1);
      }

      if (!options_.use_gravity || frame_gravity2 == nullptr) {
        for (int i = 0; i < 3; i++) {
          coeffs.emplace_back(curr_row + i, frame_param_idx2 + i, 1);
        }
      } else {
        // Gravity-aligned frame2: only Y-axis contributes.
        coeffs.emplace_back(curr_row + 1, frame_param_idx2, 1);
      }

      // Add cam_from_rig terms if being estimated.
      if (cam1_param_idx != -1 || cam2_param_idx != -1) {
        if (cam1_param_idx != -1) {
          for (int i = 0; i < 3; i++) {
            coeffs.emplace_back(curr_row + i, cam1_param_idx + i, -1);
          }
        }
        if (cam2_param_idx != -1) {
          for (int i = 0; i < 3; i++) {
            coeffs.emplace_back(curr_row + i, cam2_param_idx + i, 1);
          }
        }
      }

      if (options_.use_weight) {
        for (int i = 0; i < 3; i++) {
          weights.push_back(pair_weight);
        }
      }
      curr_row += 3;
    }
  }

  // Add gauge-fixing constraint for the fixed frame.
  const int fixed_frame_param_idx = frame_id_to_param_idx_[fixed_frame_id_];
  if (num_gauge_fixing_residuals_ == 1) {
    // 1-DOF gauge fix.
    coeffs.emplace_back(curr_row, fixed_frame_param_idx, 1);
    if (options_.use_weight) {
      weights.push_back(1);
    }
    curr_row++;
  } else {
    // 3-DOF gauge fix.
    for (int i = 0; i < 3; i++) {
      coeffs.emplace_back(curr_row + i, fixed_frame_param_idx + i, 1);
      if (options_.use_weight) {
        weights.push_back(1);
      }
    }
    curr_row += 3;
  }

  // Build sparse matrix.
  constraint_matrix_.resize(curr_row, num_params);
  constraint_matrix_.setFromTriplets(coeffs.begin(), coeffs.end());

  // Set up weight vector.
  if (!options_.use_weight) {
    edge_weights_ = Eigen::VectorXd::Ones(curr_row);
  } else {
    edge_weights_ = Eigen::Map<Eigen::VectorXd>(weights.data(), weights.size());
  }

  // Initialize residual vector.
  residuals_.resize(curr_row);
}

void RotationAveragingProblem::ComputeResiduals() {
  std::mt19937 rng(std::random_device{}());

  for (const auto& [pair_id, constraint] : pair_constraints_) {
    const frame_t frame_id1 = image_id_to_frame_id_.at(constraint.image_id1);
    const frame_t frame_id2 = image_id_to_frame_id_.at(constraint.image_id2);
    const int frame_param_idx1 = frame_id_to_param_idx_.at(frame_id1);
    const int frame_param_idx2 = frame_id_to_param_idx_.at(frame_id2);

    if (const auto* constraint_1dof =
            std::get_if<GravityAligned1DOF>(&constraint.constraint)) {
      // 1-DOF case: compute Y-axis angle residual.
      residuals_[constraint.row_index] = ComputeGravityAligned1DOFResidual(
          rng,
          constraint_1dof->angle_cam2_from_cam1,
          estimated_rotations_[frame_param_idx1],
          estimated_rotations_[frame_param_idx2]);
    } else if (const auto* full =
                   std::get_if<Full3DOF>(&constraint.constraint)) {
      // 3-DOF case: compute full rotation error.

      Eigen::Matrix3d estimated_cam1_from_world, estimated_cam2_from_world;

      const Eigen::Vector3d* frame_gravity1 =
          GetFrameGravityOrNull(frame_to_pose_prior_, frame_id1);
      const Eigen::Vector3d* frame_gravity2 =
          GetFrameGravityOrNull(frame_to_pose_prior_, frame_id2);

      if (options_.use_gravity && frame_gravity1 != nullptr) {
        estimated_cam1_from_world = colmap::RotationFromYAxisAngle(
            estimated_rotations_[frame_param_idx1]);
      } else {
        estimated_cam1_from_world = colmap::AngleAxisToRotationMatrix(
            estimated_rotations_.segment(frame_param_idx1, 3));
      }

      if (options_.use_gravity && frame_gravity2 != nullptr) {
        estimated_cam2_from_world = colmap::RotationFromYAxisAngle(
            estimated_rotations_[frame_param_idx2]);
      } else {
        estimated_cam2_from_world = colmap::AngleAxisToRotationMatrix(
            estimated_rotations_.segment(frame_param_idx2, 3));
      }

      if (constraint.cam1_from_rig_param_idx != -1) {
        estimated_cam1_from_world =
            colmap::AngleAxisToRotationMatrix(estimated_rotations_.segment(
                constraint.cam1_from_rig_param_idx, 3)) *
            estimated_cam1_from_world;
      }
      if (constraint.cam2_from_rig_param_idx != -1) {
        estimated_cam2_from_world =
            colmap::AngleAxisToRotationMatrix(estimated_rotations_.segment(
                constraint.cam2_from_rig_param_idx, 3)) *
            estimated_cam2_from_world;
      }

      residuals_.segment(constraint.row_index, 3) =
          -colmap::RotationMatrixToAngleAxis(
              estimated_cam2_from_world.transpose() * full->R_cam2_from_cam1 *
              estimated_cam1_from_world);
    } else {
      LOG(FATAL) << "Unknown constraint type";
    }
  }

  // Fixed frame residual.
  const int fixed_frame_param_idx = frame_id_to_param_idx_.at(fixed_frame_id_);
  if (num_gauge_fixing_residuals_ == 1) {
    residuals_[residuals_.size() - 1] =
        estimated_rotations_[fixed_frame_param_idx] - fixed_frame_rotation_[1];
  } else {
    residuals_
        .segment(residuals_.size() - 3, 3) = colmap::RotationMatrixToAngleAxis(
        colmap::AngleAxisToRotationMatrix(fixed_frame_rotation_).transpose() *
        colmap::AngleAxisToRotationMatrix(
            estimated_rotations_.segment(fixed_frame_param_idx, 3)));
  }
}

void RotationAveragingProblem::UpdateState(const Eigen::VectorXd& step) {
  // Update frame rotations.
  for (const auto& [frame_id, frame_param_idx] : frame_id_to_param_idx_) {
    if (!HasFrameGravity(frame_id)) {
      const Eigen::Matrix3d estimated_rig_from_world =
          colmap::AngleAxisToRotationMatrix(
              estimated_rotations_.segment(frame_param_idx, 3));
      estimated_rotations_.segment(frame_param_idx, 3) =
          colmap::RotationMatrixToAngleAxis(
              estimated_rig_from_world * colmap::AngleAxisToRotationMatrix(
                                             -step.segment(frame_param_idx, 3)));
    } else {
      estimated_rotations_[frame_param_idx] -= step[frame_param_idx];
    }
  }

  // Compute current frame rotations for cam_from_rig averaging.
  std::unordered_map<frame_t, Eigen::Matrix3d> frame_rotations;
  for (const auto& [frame_id, frame_param_idx] : frame_id_to_param_idx_) {
    if (!HasFrameGravity(frame_id)) {
      frame_rotations[frame_id] = colmap::AngleAxisToRotationMatrix(
          estimated_rotations_.segment(frame_param_idx, 3));
    } else {
      frame_rotations[frame_id] =
          colmap::RotationFromYAxisAngle(estimated_rotations_[frame_param_idx]);
    }
  }

  // Update the global rotations for cam_from_rig cameras.
  // Note: the update is non-trivial, and we need to average the rotations from
  // all the frames.
  for (const auto& [camera_id, camera_param_idx] : camera_id_to_param_idx_) {
    const Eigen::Matrix3d estimated_cam_from_rig =
        colmap::AngleAxisToRotationMatrix(
            estimated_rotations_.segment(camera_param_idx, 3));
    const Eigen::Matrix3d R_update =
        colmap::AngleAxisToRotationMatrix(-step.segment(camera_param_idx, 3));

    std::vector<Eigen::Quaterniond> rig_rotations;
    for (const frame_t frame_id : camera_to_frame_ids_[camera_id]) {
      const Eigen::Matrix3d& R = frame_rotations[frame_id];
      rig_rotations.push_back(
          Eigen::Quaterniond(estimated_cam_from_rig * R * R_update * R.transpose()));
    }

    // Average the rotations for the rig.
    const Eigen::Quaterniond R_ave = colmap::AverageQuaternions(
        rig_rotations, std::vector<double>(rig_rotations.size(), 1));
    estimated_rotations_.segment(camera_param_idx, 3) =
        colmap::RotationMatrixToAngleAxis(R_ave.toRotationMatrix());
  }
}

double RotationAveragingProblem::AverageStepSize(
    const Eigen::VectorXd& step) const {
  double total_update = 0;
  for (const auto& [frame_id, frame_param_idx] : frame_id_to_param_idx_) {
    if (HasFrameGravity(frame_id)) {
      total_update += std::abs(step[frame_param_idx]);
    } else {
      total_update += step.segment(frame_param_idx, 3).norm();
    }
  }
  return total_update / frame_id_to_param_idx_.size();
}

void RotationAveragingProblem::ApplyResultsToReconstruction(
    colmap::Reconstruction& reconstruction) {
  const Eigen::Vector3d kUnknownTranslation =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());

  for (const auto& [frame_id, frame_param_idx] : frame_id_to_param_idx_) {
    const Eigen::Vector3d* frame_gravity =
        GetFrameGravityOrNull(frame_to_pose_prior_, frame_id);

    if (HasFrameGravity(frame_id)) {
      reconstruction.Frame(frame_id).SetRigFromWorld(Rigid3d(
          Eigen::Quaterniond(colmap::GravityAlignedRotation(*frame_gravity) *
                             colmap::RotationFromYAxisAngle(
                                 estimated_rotations_[frame_param_idx])),
          kUnknownTranslation));
    } else {
      reconstruction.Frame(frame_id).SetRigFromWorld(
          Rigid3d(Eigen::Quaterniond(colmap::AngleAxisToRotationMatrix(
                      estimated_rotations_.segment(frame_param_idx, 3))),
                  kUnknownTranslation));
    }
  }

  // Add the estimated cam_from_rig rotations.
  for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (camera_id_to_param_idx_.find(sensor_id.id) ==
          camera_id_to_param_idx_.end()) {
        continue;  // Skip cameras that are not estimated.
      }
      Rigid3d cam_from_rig;
      cam_from_rig.rotation =
          colmap::AngleAxisToRotationMatrix(estimated_rotations_.segment(
              camera_id_to_param_idx_.at(sensor_id.id), 3));
      cam_from_rig.translation.setConstant(
          std::numeric_limits<double>::quiet_NaN());  // No translation yet.
      reconstruction.Rig(rig_id).SetSensorFromRig(sensor_id, cam_from_rig);
    }
  }
}

bool RotationAveragingSolver::Solve(RotationAveragingProblem& problem) {
  if (options_.max_num_l1_iterations > 0) {
    VLOG(2) << "Solving L1 regression problem";
    if (!SolveL1Regression(problem)) {
      return false;
    }
  }

  if (options_.max_num_irls_iterations > 0) {
    VLOG(2) << "Solving IRLS problem";
    if (!SolveIRLS(problem)) {
      return false;
    }
  }

  return true;
}

bool RotationAveragingSolver::SolveL1Regression(
    RotationAveragingProblem& problem) {
  colmap::LeastAbsoluteDeviationSolver::Options l1_solver_options;
  l1_solver_options.max_num_iterations = 10;
  l1_solver_options.solver_type = colmap::LeastAbsoluteDeviationSolver::
      Options::SolverType::SupernodalCholmodLLT;

  const Eigen::SparseMatrix<double> A =
      problem.EdgeWeights().asDiagonal() * problem.ConstraintMatrix();

  colmap::LeastAbsoluteDeviationSolver l1_solver(l1_solver_options, A);
  double prev_norm = 0;
  double curr_norm = 0;

  Eigen::VectorXd step(problem.NumParameters());

  int iteration = 0;
  for (iteration = 0; iteration < options_.max_num_l1_iterations; iteration++) {
    VLOG(2) << "L1 ADMM iteration: " << iteration;

    problem.ComputeResiduals();

    prev_norm = curr_norm;

    step.setZero();
    l1_solver.Solve(problem.EdgeWeights().asDiagonal() * problem.Residuals(),
                    &step);
    if (step.array().isNaN().any()) {
      LOG(ERROR) << "nan error";
      return false;
    }

    if (VLOG_IS_ON(2))
      LOG(INFO) << "residual:"
                << (problem.ConstraintMatrix() * step - problem.Residuals())
                       .array()
                       .abs()
                       .sum();

    curr_norm = step.norm();
    problem.UpdateState(step);

    // Check convergence.
    constexpr double kEps = 1e-12;
    if (problem.AverageStepSize(step) <
            options_.l1_step_convergence_threshold ||
        std::abs(prev_norm - curr_norm) < kEps) {
      if (std::abs(prev_norm - curr_norm) < kEps)
        LOG(INFO) << "std::abs(prev_norm - curr_norm) < " << kEps;
      iteration++;
      break;
    }
    l1_solver_options.max_num_iterations =
        std::min(l1_solver_options.max_num_iterations * 2, 100);
  }
  VLOG(2) << "L1 ADMM total iteration: " << iteration;
  return true;
}

std::optional<Eigen::VectorXd> RotationAveragingSolver::ComputeIRLSWeights(
    const RotationAveragingProblem& problem, double sigma) const {
  Eigen::VectorXd weights(problem.NumResiduals());

  for (const auto& [pair_id, constraint] : problem.PairConstraints()) {
    double err_squared = 0;
    bool is_1dof = false;

    if (const auto* c1 =
            std::get_if<RotationAveragingProblem::GravityAligned1DOF>(
                &constraint.constraint)) {
      // 1-DOF: Y-axis error plus xz_error.
      const double residual = problem.Residuals()[constraint.row_index];
      err_squared = residual * residual + c1->xz_error;
      is_1dof = true;
    } else {
      // 3-DOF: full rotation error.
      err_squared =
          problem.Residuals().segment<3>(constraint.row_index).squaredNorm();
    }

    // Compute the weight.
    double w = 0;
    if (options_.weight_type == RotationEstimatorOptions::GEMAN_MCCLURE) {
      double tmp = err_squared + sigma * sigma;
      w = sigma * sigma / (tmp * tmp);
    } else if (options_.weight_type == RotationEstimatorOptions::HALF_NORM) {
      // Exponent for half-norm weight: (p - 2) / 2 where p = 0.5.
      constexpr double kHalfNormExponent = (0.5 - 2) / 2;
      w = std::pow(err_squared, kHalfNormExponent);
    }

    if (std::isnan(w)) {
      LOG(ERROR) << "nan weight!";
      return std::nullopt;
    }

    // Set weights for appropriate number of equations.
    if (is_1dof) {
      weights[constraint.row_index] = w;
    } else {
      weights.segment<3>(constraint.row_index).setConstant(w);
    }
  }

  // Set gauge-fixing weights to 1.
  const int gauge_rows = problem.NumGaugeFixingResiduals();
  if (gauge_rows == 1) {
    weights[problem.NumResiduals() - 1] = 1;
  } else {
    weights.segment(problem.NumResiduals() - 3, 3).setConstant(1);
  }

  return weights;
}

bool RotationAveragingSolver::SolveIRLS(RotationAveragingProblem& problem) {
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> llt;

  llt.analyzePattern(problem.ConstraintMatrix().transpose() *
                     problem.ConstraintMatrix());

  const double sigma = colmap::DegToRad(options_.irls_loss_parameter_sigma);

  Eigen::SparseMatrix<double> at_weight;
  Eigen::VectorXd step(problem.NumParameters());

  int iteration = 0;
  for (iteration = 0; iteration < options_.max_num_irls_iterations;
       iteration++) {
    problem.ComputeResiduals();

    // Compute the weights for IRLS.
    auto weights_irls = ComputeIRLSWeights(problem, sigma);
    if (!weights_irls) {
      return false;
    }

    // Update the factorization for the weighted values.
    at_weight = problem.ConstraintMatrix().transpose() *
                weights_irls->asDiagonal() * problem.EdgeWeights().asDiagonal();

    llt.factorize(at_weight * problem.ConstraintMatrix());

    // Solve the least squares problem.
    step.setZero();
    step = llt.solve(at_weight * problem.Residuals());
    problem.UpdateState(step);

    const double avg_step = problem.AverageStepSize(step);
    VLOG(2) << "IRLS iteration " << iteration << ", average step: " << avg_step;

    // Check convergence.
    if (avg_step < options_.irls_step_convergence_threshold) {
      iteration++;
      break;
    }
  }
  VLOG(2) << "IRLS total iteration: " << iteration;

  return true;
}

}  // namespace glomap
