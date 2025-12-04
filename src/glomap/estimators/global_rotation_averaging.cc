#include "global_rotation_averaging.h"

#include "colmap/geometry/pose.h"
#include "colmap/optim/least_absolute_deviations.h"

#include "glomap/estimators/rotation_initializer.h"
#include "glomap/math/gravity.h"
#include "glomap/math/tree.h"

#include <iostream>
#include <queue>

#include <Eigen/CholmodSupport>
#include <Eigen/SparseCholesky>

namespace glomap {
namespace {

double RelAngleError(std::mt19937& rng,
                     double angle_12,
                     double angle_1,
                     double angle_2) {
  double est = (angle_2 - angle_1) - angle_12;

  while (est >= EIGEN_PI) {
    est -= 2 * EIGEN_PI;
  }
  while (est < -EIGEN_PI) {
    est += 2 * EIGEN_PI;
  }

  // Inject random noise if the angle is too close to the boundary to break the
  // possible balance at the local minima
  constexpr double kEps = 0.01;
  if (std::abs(est) > EIGEN_PI - kEps) {
    std::uniform_real_distribution<double> dist(0.0, kEps);
    const double jitter = dist(rng);
    if (est < 0) {
      est += jitter;
    } else {
      est -= jitter;
    }
  }

  return est;
}

bool AllSensorsFromRigKnown(const std::unordered_map<rig_t, Rig>& rigs) {
  bool all_known = true;
  for (auto& [rig_id, rig] : rigs) {
    for (auto& [sensor_id, sensor] : rig.NonRefSensors()) {
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

std::unordered_map<frame_t, colmap::PosePrior*> ExtractFrameToPosePrior(
    const std::unordered_map<image_t, Image>& images,
    std::vector<colmap::PosePrior>& pose_priors) {
  std::unordered_map<image_t, frame_t> image_to_frame;
  for (const auto& [image_id, image] : images) {
    image_to_frame[image_id] = image.frame_id;
  }

  std::unordered_map<frame_t, colmap::PosePrior*> frame_to_pose_prior;
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      const image_t image_id = pose_prior.corr_data_id.id;
      const Image& image = images.at(image_id);
      if (image.HasTrivialFrame()) {
        const frame_t frame_id = image_to_frame.at(pose_prior.corr_data_id.id);
        THROW_CHECK(frame_to_pose_prior.emplace(frame_id, &pose_prior).second)
            << "Duplicate pose prior for frame" << frame_id;
      }
    }
  }

  return frame_to_pose_prior;
}

Eigen::Vector3d* GetFrameGravityOrNull(
    const std::unordered_map<frame_t, colmap::PosePrior*>& frame_to_pose_prior,
    frame_t frame_id) {
  auto it = frame_to_pose_prior.find(frame_id);
  if (it == frame_to_pose_prior.end() || !it->second->HasGravity()) {
    return nullptr;
  }
  return &it->second->gravity;
}

}  // namespace

bool RotationEstimator::EstimateRotations(
    const ViewGraph& view_graph,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::vector<colmap::PosePrior>& pose_priors) {
  // Now, for the gravity aligned case, we only support trivial rigs only or
  // non-trivial rigs with known sensor_from_rig.
  if (options_.use_gravity && !AllSensorsFromRigKnown(rigs)) {
    return false;
  }

  std::unordered_map<frame_t, colmap::PosePrior*> frame_to_pose_prior =
      ExtractFrameToPosePrior(images, pose_priors);

  // Initialize the rotation from maximum spanning tree
  if (!options_.skip_initialization && !options_.use_gravity) {
    InitializeFromMaximumSpanningTree(view_graph, rigs, frames, images);
  }

  // Set up the linear system
  SetupLinearSystem(view_graph, rigs, frames, images, frame_to_pose_prior);

  // Solve the linear system for L1 norm optimization
  if (options_.max_num_l1_iterations > 0) {
    if (!SolveL1Regression(view_graph, frames, images, frame_to_pose_prior)) {
      return false;
    }
  }

  // Solve the linear system for IRLS optimization
  if (options_.max_num_irls_iterations > 0) {
    if (!SolveIRLS(view_graph, frames, images, frame_to_pose_prior)) {
      return false;
    }
  }

  ConvertResults(rigs, frames, frame_to_pose_prior);

  return true;
}

void RotationEstimator::InitializeFromMaximumSpanningTree(
    const ViewGraph& view_graph,
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images) {
  // Here, we assume that largest connected component is already retrieved, so
  // we do not need to do that again compute maximum spanning tree.
  std::unordered_map<image_t, image_t> parents;
  const image_t root =
      MaximumSpanningTree(view_graph, images, parents, WeightType::INLIER_NUM);

  // Iterate through the tree to initialize the rotation
  // Establish child info
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& [image_id, image] : images) {
    if (!image.IsRegistered()) continue;
    children.insert(std::make_pair(image_id, std::vector<image_t>()));
  }
  for (auto& [child, parent] : parents) {
    if (root == child) continue;
    children[parent].emplace_back(child);
  }

  std::queue<image_t> indexes;
  indexes.push(root);

  std::unordered_map<image_t, Rigid3d> cam_from_worlds;
  while (!indexes.empty()) {
    image_t curr = indexes.front();
    indexes.pop();

    // Add all children into the tree
    for (auto& child : children[curr]) indexes.push(child);
    // If it is root, then fix it to be the original estimation
    if (curr == root) continue;

    // Directly use the relative pose for estimation rotation
    const ImagePair& image_pair = view_graph.image_pairs.at(
        colmap::ImagePairToPairId(curr, parents[curr]));
    if (image_pair.image_id1 == curr) {
      // 1_R_w = 2_R_1^T * 2_R_w
      cam_from_worlds[curr].rotation =
          (Inverse(image_pair.cam2_from_cam1) * cam_from_worlds[parents[curr]])
              .rotation;
    } else {
      // 2_R_w = 2_R_1 * 1_R_w
      cam_from_worlds[curr].rotation =
          (image_pair.cam2_from_cam1 * cam_from_worlds[parents[curr]]).rotation;
    }
  }

  ConvertRotationsFromImageToRig(cam_from_worlds, images, rigs, frames);
}

// TODO: refine the code
void RotationEstimator::SetupLinearSystem(
    const ViewGraph& view_graph,
    const std::unordered_map<rig_t, Rig>& rigs,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<frame_t, colmap::PosePrior*>&
        frame_to_pose_prior) {
  // Clear all the structures
  sparse_matrix_.resize(0, 0);
  tangent_space_step_.resize(0);
  tangent_space_residual_.resize(0);
  rotation_estimated_.resize(0);
  image_id_to_idx_.clear();
  camera_id_to_idx_.clear();
  rel_temp_info_.clear();

  // Initialize the structures for estimated rotation
  image_id_to_idx_.reserve(images.size());
  camera_id_to_idx_.reserve(images.size());
  rotation_estimated_.resize(6 * images.size());
  image_t num_dof = 0;
  std::unordered_map<camera_t, rig_t> camera_id_to_rig_id;
  for (auto& [frame_id, frame] : frames) {
    for (const auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (images.find(image_id) == images.end()) continue;
      const auto& image = images.at(image_id);
      if (!image.IsRegistered()) continue;
      camera_id_to_rig_id[image.camera_id] = frame.RigId();
    }
  }

  // First, we need to determine which cameras need to be estimated
  std::unordered_map<camera_t, Eigen::AngleAxisd> cam_from_rig_rotations;
  for (auto& [camera_id, rig_id] : camera_id_to_rig_id) {
    sensor_t sensor_id(SensorType::CAMERA, camera_id);
    if (rigs.at(rig_id).IsRefSensor(sensor_id)) continue;

    auto cam_from_rig = rigs.at(rig_id).MaybeSensorFromRig(sensor_id);
    if (!cam_from_rig.has_value() ||
        cam_from_rig.value().translation.hasNaN()) {
      if (camera_id_to_idx_.find(camera_id) == camera_id_to_idx_.end()) {
        camera_id_to_idx_[camera_id] = -1;
        if (cam_from_rig.has_value()) {
          // If the camera is not part of a rig, then we can use the first image
          // to initialize the rotation
          cam_from_rig_rotations[camera_id] =
              Eigen::AngleAxisd(cam_from_rig->rotation);
        }
      }
    }
  }

  for (auto& [frame_id, frame] : frames) {
    // Skip the unregistered frames
    if (!frames.at(frame_id).is_registered) continue;
    frame_id_to_idx_[frame_id] = num_dof;
    image_t image_id_ref = -1;
    for (auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (images.find(image_id) == images.end()) continue;
      image_id_to_idx_[image_id] = num_dof;  // point to the first element
      if (images.at(image_id).HasTrivialFrame()) {
        image_id_ref = image_id;
      }
    }

    Eigen::Vector3d* frame_gravity =
        GetFrameGravityOrNull(frame_to_pose_prior, frame_id);
    if (options_.use_gravity && frame_gravity != nullptr) {
      rotation_estimated_[num_dof] =
          RotUpToAngle(GetAlignRot(*frame_gravity).transpose() *
                       frame.RigFromWorld().rotation.toRotationMatrix());
      num_dof++;

      if (fixed_image_id_ == -1) {
        fixed_camera_rotation_ =
            Eigen::Vector3d(0, rotation_estimated_[num_dof - 1], 0);
        fixed_image_id_ = image_id_ref;
      }
    } else {
      Eigen::AngleAxisd rig_from_world;
      if (frame.MaybeRigFromWorld().has_value()) {
        rig_from_world = Eigen::AngleAxisd(frame.RigFromWorld().rotation);
      } else {
        rig_from_world = Eigen::AngleAxisd::Identity();
      }
      rotation_estimated_.segment(num_dof, 3) =
          rig_from_world.angle() * rig_from_world.axis();
      num_dof += 3;
    }
  }

  // Set the camera id to index mapping for cameras that need to be
  // estimated.
  for (auto& [camera_id, camera_idx] : camera_id_to_idx_) {
    // If the camera is not part of a rig, then we can use the first image
    // to initialize the rotation
    camera_id_to_idx_[camera_id] = num_dof;
    if (const auto it = cam_from_rig_rotations.find(camera_id);
        it != cam_from_rig_rotations.end()) {
      rotation_estimated_.segment(num_dof, 3) =
          it->second.angle() * it->second.axis();
    } else {
      // If the camera is part of a rig, then we can use the rig rotation
      // to initialize the rotation
      rotation_estimated_.segment(num_dof, 3) = Eigen::Vector3d::Zero();
    }
    num_dof += 3;
  }

  // If no cameras are set to be fixed, then take the first camera
  if (fixed_image_id_ == -1) {
    for (auto& [frame_id, frame] : frames) {
      if (!frames.at(frame_id).is_registered) continue;

      fixed_image_id_ = frame.DataIds().begin()->id;
      const Eigen::AngleAxisd rig_from_world(frame.RigFromWorld().rotation);
      fixed_camera_rotation_ = rig_from_world.angle() * rig_from_world.axis();
      break;
    }
  }

  rotation_estimated_.conservativeResize(num_dof);

  // Prepare the relative information
  int counter = 0;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    const auto& image1 = images.at(image_pair.image_id1);
    const auto& image2 = images.at(image_pair.image_id2);
    const auto& frame1 = frames.at(image1.frame_id);
    const auto& frame2 = frames.at(image2.frame_id);

    const int image_idx1 = image_id_to_idx_[image_pair.image_id1];
    const int image_idx2 = image_id_to_idx_[image_pair.image_id2];

    Rigid3d cam1_from_rig1, cam2_from_rig2;

    bool has_sensor_from_rig1 = false;
    bool has_sensor_from_rig2 = false;
    if (!image1.HasTrivialFrame()) {
      if (camera_id_to_idx_.find(image1.camera_id) == camera_id_to_idx_.end()) {
        cam1_from_rig1 =
            rigs.at(frame1.RigId())
                .SensorFromRig(sensor_t(SensorType::CAMERA, image1.camera_id));
        has_sensor_from_rig1 = true;
      }
    }
    if (!image2.HasTrivialFrame()) {
      if (camera_id_to_idx_.find(image2.camera_id) == camera_id_to_idx_.end()) {
        cam2_from_rig2 =
            rigs.at(frame2.RigId())
                .SensorFromRig(sensor_t(SensorType::CAMERA, image2.camera_id));
        has_sensor_from_rig2 = true;
      }
    }

    // If both images are from the same rig and there is no need to estimate
    // the cam_from_rig, skip the estimation
    if (has_sensor_from_rig1 && has_sensor_from_rig2 &&
        image_idx1 == image_idx2) {
      continue;  // Skip the self loop
    }

    rel_temp_info_[pair_id].R_rel =
        (cam2_from_rig2.rotation.inverse() *
         image_pair.cam2_from_cam1.rotation * cam1_from_rig1.rotation)
            .toRotationMatrix();

    Eigen::Vector3d* frame_gravity1 =
        GetFrameGravityOrNull(frame_to_pose_prior, image1.frame_id);
    Eigen::Vector3d* frame_gravity2 =
        GetFrameGravityOrNull(frame_to_pose_prior, image2.frame_id);
    if (options_.use_gravity) {
      if (frame_gravity1 != nullptr) {
        rel_temp_info_[pair_id].R_rel =
            rel_temp_info_[pair_id].R_rel * GetAlignRot(*frame_gravity1);
      }

      if (frame_gravity2 != nullptr) {
        rel_temp_info_[pair_id].R_rel =
            GetAlignRot(*frame_gravity2).transpose() *
            rel_temp_info_[pair_id].R_rel;
      }
    }

    if (options_.use_gravity && frame_gravity1 != nullptr &&
        frame_gravity2 != nullptr) {
      counter++;
      Eigen::Vector3d aa =
          colmap::RotationMatrixToAngleAxis(rel_temp_info_[pair_id].R_rel);
      double error = aa[0] * aa[0] + aa[2] * aa[2];

      // Keep track of the error for x and z axis for gravity-aligned relative
      // pose
      rel_temp_info_[pair_id].xz_error = error;
      rel_temp_info_[pair_id].has_gravity = true;
      rel_temp_info_[pair_id].angle_rel = aa[1];
    } else {
      rel_temp_info_[pair_id].has_gravity = false;
    }
  }

  VLOG(2) << counter << " image pairs are gravity aligned" << '\n';

  std::vector<Eigen::Triplet<double>> coeffs;
  coeffs.reserve(rel_temp_info_.size() * 6 + 3);

  // Establish linear systems
  size_t curr_pos = 0;
  std::vector<double> weights;
  weights.reserve(3 * view_graph.image_pairs.size());
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    if (rel_temp_info_.find(pair_id) == rel_temp_info_.end()) continue;

    const auto& image1 = images.at(image_pair.image_id1);
    const auto& image2 = images.at(image_pair.image_id2);
    const auto& frame1 = frames.at(image1.frame_id);
    const auto& frame2 = frames.at(image2.frame_id);

    const int image_idx1 = image_id_to_idx_[image_pair.image_id1];
    const int image_idx2 = image_id_to_idx_[image_pair.image_id2];

    if (!frame1.is_registered || !frame2.is_registered) {
      continue;  // skip unregistered frames
    }

    int vector_idx_cam1 = -1;
    int vector_idx_cam2 = -1;
    if (camera_id_to_idx_.find(image1.camera_id) != camera_id_to_idx_.end()) {
      vector_idx_cam1 = camera_id_to_idx_[image1.camera_id];
    }
    if (camera_id_to_idx_.find(image2.camera_id) != camera_id_to_idx_.end()) {
      vector_idx_cam2 = camera_id_to_idx_[image2.camera_id];
    }

    rel_temp_info_[pair_id].index = curr_pos;
    rel_temp_info_[pair_id].idx_cam1 = vector_idx_cam1;
    rel_temp_info_[pair_id].idx_cam2 = vector_idx_cam2;

    // TODO: figure out the logic for the gravity aligned case
    if (rel_temp_info_[pair_id].has_gravity) {
      coeffs.emplace_back(Eigen::Triplet<double>(curr_pos, image_idx1, -1));
      coeffs.emplace_back(Eigen::Triplet<double>(curr_pos, image_idx2, 1));
      if (image_pair.weight >= 0)
        weights.emplace_back(image_pair.weight);
      else
        weights.emplace_back(1);
      curr_pos++;
    } else {
      // If it is not gravity aligned, then we need to consider 3 dof
      if (!options_.use_gravity ||
          GetFrameGravityOrNull(frame_to_pose_prior, image1.frame_id) ==
              nullptr) {
        for (int i = 0; i < 3; i++) {
          coeffs.emplace_back(
              Eigen::Triplet<double>(curr_pos + i, image_idx1 + i, -1));
        }
      } else
        // else, other components are zero, and can be safely ignored
        coeffs.emplace_back(
            Eigen::Triplet<double>(curr_pos + 1, image_idx1, -1));

      // Similarly for the second componenet
      if (!options_.use_gravity ||
          GetFrameGravityOrNull(frame_to_pose_prior, image2.frame_id) ==
              nullptr) {
        for (int i = 0; i < 3; i++) {
          coeffs.emplace_back(
              Eigen::Triplet<double>(curr_pos + i, image_idx2 + i, 1));
        }
      } else
        coeffs.emplace_back(
            Eigen::Triplet<double>(curr_pos + 1, image_idx2, 1));
      for (int i = 0; i < 3; i++) {
        if (image_pair.weight >= 0)
          weights.emplace_back(image_pair.weight);
        else
          weights.emplace_back(1);
      }

      // If both camera share the same rig, the terms in the linear system would
      // be cancelled
      if (!(vector_idx_cam1 == -1 && vector_idx_cam2 == -1)) {
        if (vector_idx_cam1 != -1) {
          // If the camera is not part of a rig, then we can use the first image
          // to initialize the rotation
          for (int i = 0; i < 3; i++) {
            coeffs.emplace_back(
                Eigen::Triplet<double>(curr_pos + i, vector_idx_cam1 + i, -1));
          }
        }
        if (vector_idx_cam2 != -1) {
          for (int i = 0; i < 3; i++) {
            coeffs.emplace_back(
                Eigen::Triplet<double>(curr_pos + i, vector_idx_cam2 + i, 1));
          }
        }
      }

      curr_pos += 3;
    }
  }

  const Image& fixed_image = images.at(fixed_image_id_);
  const auto fixed_pose_prior_it =
      frame_to_pose_prior.find(fixed_image.frame_id);
  const bool fixed_image_has_gravity =
      fixed_pose_prior_it != frame_to_pose_prior.end() &&
      fixed_pose_prior_it->second->HasGravity();

  // Set some cameras to be fixed
  // if some cameras have gravity, then add a single term constraint
  // Else, change to 3 constriants
  if (options_.use_gravity && fixed_image_has_gravity) {
    coeffs.emplace_back(
        Eigen::Triplet<double>(curr_pos, image_id_to_idx_[fixed_image_id_], 1));
    weights.emplace_back(1);
    curr_pos++;
  } else {
    for (int i = 0; i < 3; i++) {
      coeffs.emplace_back(Eigen::Triplet<double>(
          curr_pos + i, image_id_to_idx_[fixed_image_id_] + i, 1));
      weights.emplace_back(1);
    }
    curr_pos += 3;
  }

  sparse_matrix_.resize(curr_pos, num_dof);
  sparse_matrix_.setFromTriplets(coeffs.begin(), coeffs.end());

  // Set up the weight matrix for the linear system
  if (!options_.use_weight) {
    weights_ = Eigen::ArrayXd::Ones(curr_pos);
  } else {
    weights_ = Eigen::ArrayXd(weights.size());
    for (size_t i = 0; i < weights.size(); i++) weights_[i] = weights[i];
  }

  // Initialize x and b
  tangent_space_step_.resize(num_dof);
  tangent_space_residual_.resize(curr_pos);
}

bool RotationEstimator::SolveL1Regression(
    const ViewGraph& view_graph,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<frame_t, colmap::PosePrior*>&
        frame_to_pose_prior) {
  colmap::LeastAbsoluteDeviationSolver::Options l1_solver_options;
  l1_solver_options.max_num_iterations = 10;
  l1_solver_options.solver_type = colmap::LeastAbsoluteDeviationSolver::
      Options::SolverType::SupernodalCholmodLLT;

  const Eigen::SparseMatrix<double> A =
      weights_.matrix().asDiagonal() * sparse_matrix_;

  colmap::LeastAbsoluteDeviationSolver l1_solver(l1_solver_options, A);
  double last_norm = 0;
  double curr_norm = 0;

  ComputeResiduals(view_graph, images, frame_to_pose_prior);
  VLOG(2) << "ComputeResiduals done";

  int iteration = 0;
  for (iteration = 0; iteration < options_.max_num_l1_iterations; iteration++) {
    VLOG(2) << "L1 ADMM iteration: " << iteration;

    last_norm = curr_norm;
    // use the current residual as b (Ax - b)

    tangent_space_step_.setZero();
    l1_solver.Solve(weights_.matrix().asDiagonal() * tangent_space_residual_,
                    &tangent_space_step_);
    if (tangent_space_step_.array().isNaN().any()) {
      LOG(ERROR) << "nan error";
      iteration++;
      return false;
    }

    if (VLOG_IS_ON(2))
      LOG(INFO) << "residual:"
                << (sparse_matrix_ * tangent_space_step_ -
                    tangent_space_residual_)
                       .array()
                       .abs()
                       .sum();

    curr_norm = tangent_space_step_.norm();
    UpdateGlobalRotations(view_graph, frames, images, frame_to_pose_prior);
    ComputeResiduals(view_graph, images, frame_to_pose_prior);

    // Check the residual. If it is small, stop
    // TODO: strange bug for the L1 solver: update norm state constant
    constexpr double kEps = 1e-12;
    if (ComputeAverageStepSize(frames, frame_to_pose_prior) <
            options_.l1_step_convergence_threshold ||
        std::abs(last_norm - curr_norm) < kEps) {
      if (std::abs(last_norm - curr_norm) < kEps)
        LOG(INFO) << "std::abs(last_norm - curr_norm) < " << kEps;
      iteration++;
      break;
    }
    l1_solver_options.max_num_iterations =
        std::min(l1_solver_options.max_num_iterations * 2, 100);
  }
  VLOG(2) << "L1 ADMM total iteration: " << iteration;
  return true;
}

bool RotationEstimator::SolveIRLS(
    const ViewGraph& view_graph,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    const std::unordered_map<frame_t, colmap::PosePrior*>&
        frame_to_pose_prior) {
  // TODO: Determine what is the best solver for this part
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> llt;

  llt.analyzePattern(sparse_matrix_.transpose() * sparse_matrix_);

  const double sigma = colmap::DegToRad(options_.irls_loss_parameter_sigma);
  VLOG(2) << "sigma: " << options_.irls_loss_parameter_sigma;

  Eigen::ArrayXd weights_irls(sparse_matrix_.rows());
  Eigen::SparseMatrix<double> at_weight;

  const Image& fixed_image = images.at(fixed_image_id_);
  const auto fixed_pose_prior_it =
      frame_to_pose_prior.find(fixed_image.frame_id);
  const bool fixed_image_has_gravity =
      fixed_pose_prior_it != frame_to_pose_prior.end() &&
      fixed_pose_prior_it->second->HasGravity();

  if (options_.use_gravity && fixed_image_has_gravity) {
    weights_irls[sparse_matrix_.rows() - 1] = 1;
  } else {
    weights_irls.segment(sparse_matrix_.rows() - 3, 3).setConstant(1);
  }

  ComputeResiduals(view_graph, images, frame_to_pose_prior);
  int iteration = 0;
  for (iteration = 0; iteration < options_.max_num_irls_iterations;
       iteration++) {
    VLOG(2) << "IRLS iteration: " << iteration;

    // Compute the weights for IRLS
    for (auto& [pair_id, pair_info] : rel_temp_info_) {
      image_pair_t image_pair_pos = pair_info.index;
      double err_squared = 0;
      double w = 0;
      // If both cameras have gravity, then we only consider the y-axis
      if (pair_info.has_gravity)
        err_squared = std::pow(tangent_space_residual_[image_pair_pos], 2) +
                      pair_info.xz_error;
      // Otherwise, we consider all 3 dof
      else
        err_squared =
            tangent_space_residual_.segment<3>(image_pair_pos).squaredNorm();

      // Compute the weight
      if (options_.weight_type == RotationEstimatorOptions::GEMAN_MCCLURE) {
        double tmp = err_squared + sigma * sigma;
        w = sigma * sigma / (tmp * tmp);
      } else if (options_.weight_type == RotationEstimatorOptions::HALF_NORM) {
        w = std::pow(err_squared, (0.5 - 2) / 2);
      }

      if (std::isnan(w)) {
        LOG(ERROR) << "nan weight!";
        return false;
      }

      // If both cameras have gravity, then only 1 equation
      if (pair_info.has_gravity) weights_irls[image_pair_pos] = w;
      // Otherwise, 3 equations
      else
        weights_irls.segment<3>(image_pair_pos).setConstant(w);
    }

    // Update the factorization for the weighted values.
    at_weight = sparse_matrix_.transpose() *
                weights_irls.matrix().asDiagonal() *
                weights_.matrix().asDiagonal();

    llt.factorize(at_weight * sparse_matrix_);

    // Solve the least squares problem..
    tangent_space_step_.setZero();
    tangent_space_step_ = llt.solve(at_weight * tangent_space_residual_);
    UpdateGlobalRotations(view_graph, frames, images, frame_to_pose_prior);
    ComputeResiduals(view_graph, images, frame_to_pose_prior);

    // Check the residual. If it is small, stop
    if (ComputeAverageStepSize(frames, frame_to_pose_prior) <
        options_.irls_step_convergence_threshold) {
      iteration++;
      break;
    }
  }
  VLOG(2) << "IRLS total iteration: " << iteration;

  return true;
}

void RotationEstimator::UpdateGlobalRotations(
    const ViewGraph& view_graph,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<frame_t, colmap::PosePrior*>&
        frame_to_pose_prior) {
  for (const auto& [frame_id, frame] : frames) {
    if (!frame.is_registered) continue;
    const frame_t frame_idx = frame_id_to_idx_[frame_id];
    const auto pose_prior_it = frame_to_pose_prior.find(frame_id);
    const bool has_gravity = pose_prior_it != frame_to_pose_prior.end() &&
                             pose_prior_it->second->HasGravity();
    if (!(options_.use_gravity && has_gravity)) {
      const Eigen::Matrix3d R_ori = colmap::AngleAxisToRotationMatrix(
          rotation_estimated_.segment(frame_idx, 3));
      rotation_estimated_.segment(frame_idx, 3) =
          colmap::RotationMatrixToAngleAxis(
              R_ori * colmap::AngleAxisToRotationMatrix(
                          -tangent_space_step_.segment(frame_idx, 3)));
    } else {
      rotation_estimated_[frame_idx] -= tangent_space_step_[frame_idx];
    }
  }

  std::unordered_map<camera_t, std::vector<Eigen::Matrix3d>> cam_from_rigs;
  for (const auto& [camera_id, camera_idx] : camera_id_to_idx_) {
    cam_from_rigs[camera_id] = std::vector<Eigen::Matrix3d>();
  }
  for (const auto& [frame_id, frame] : frames) {
    if (!frames.at(frame_id).is_registered) continue;
    const auto pose_prior_it = frame_to_pose_prior.find(frame_id);
    const bool has_gravity = pose_prior_it != frame_to_pose_prior.end() &&
                             pose_prior_it->second->HasGravity();
    // Update the rig from world for the frame
    Eigen::Matrix3d R_ori;
    if (!options_.use_gravity || !has_gravity) {
      R_ori = colmap::AngleAxisToRotationMatrix(
          rotation_estimated_.segment(frame_id_to_idx_[frame_id], 3));
    } else {
      R_ori = AngleToRotUp(rotation_estimated_[frame_id_to_idx_[frame_id]]);
    }

    // Update the cam_from_rig for the cameras in the frame
    for (const auto& data_id : frame.ImageIds()) {
      if (images.find(data_id.id) == images.end()) continue;
      const auto& image = images.at(data_id.id);
      if (camera_id_to_idx_.find(image.camera_id) != camera_id_to_idx_.end()) {
        cam_from_rigs[image.camera_id].push_back(R_ori);
      }
    }
  }

  // Update the global rotations for cam_from_rig cameras
  // Note: the update is non trivial, and we need to average the rotations from
  // all the frames
  for (auto& [camera_id, camera_idx] : camera_id_to_idx_) {
    const Eigen::Matrix3d R_ori = colmap::AngleAxisToRotationMatrix(
        rotation_estimated_.segment(camera_idx, 3));
    std::vector<Eigen::Quaterniond> rig_rotations;
    const Eigen::Matrix3d R_update = colmap::AngleAxisToRotationMatrix(
        -tangent_space_step_.segment(camera_idx, 3));
    for (const auto& R : cam_from_rigs[camera_id]) {
      // Update the rotation for the camera
      rig_rotations.push_back(
          Eigen::Quaterniond(R_ori * R * R_update * R.transpose()));
    }
    // Average the rotations for the rig
    const Eigen::Quaterniond R_ave = colmap::AverageQuaternions(
        rig_rotations, std::vector<double>(rig_rotations.size(), 1));
    rotation_estimated_.segment(camera_idx, 3) =
        colmap::RotationMatrixToAngleAxis(R_ave.toRotationMatrix());
  }
}

void RotationEstimator::ComputeResiduals(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<frame_t, colmap::PosePrior*>&
        frame_to_pose_prior) {
  std::mt19937 rng(std::random_device{}());

  int curr_pos = 0;
  for (const auto& [pair_id, pair_info] : rel_temp_info_) {
    const image_t image_id1 = view_graph.image_pairs.at(pair_id).image_id1;
    const image_t image_id2 = view_graph.image_pairs.at(pair_id).image_id2;
    const int idx_cam1 = pair_info.idx_cam1;
    const int idx_cam2 = pair_info.idx_cam2;

    if (pair_info.has_gravity) {
      tangent_space_residual_[pair_info.index] =
          (RelAngleError(rng,
                         pair_info.angle_rel,
                         rotation_estimated_[image_id_to_idx_[image_id1]],
                         rotation_estimated_[image_id_to_idx_[image_id2]]));
    } else {
      Eigen::Matrix3d R_1, R_2;
      const Image& image1 = images.at(image_id1);
      const Image& image2 = images.at(image_id2);

      Eigen::Vector3d* frame_gravity1 =
          GetFrameGravityOrNull(frame_to_pose_prior, image1.frame_id);
      Eigen::Vector3d* frame_gravity2 =
          GetFrameGravityOrNull(frame_to_pose_prior, image2.frame_id);

      if (options_.use_gravity && frame_gravity1 != nullptr) {
        R_1 = AngleToRotUp(rotation_estimated_[image_id_to_idx_[image_id1]]);
      } else {
        R_1 = colmap::AngleAxisToRotationMatrix(
            rotation_estimated_.segment(image_id_to_idx_[image_id1], 3));
      }

      if (options_.use_gravity && frame_gravity2 != nullptr) {
        R_2 = AngleToRotUp(rotation_estimated_[image_id_to_idx_[image_id2]]);
      } else {
        R_2 = colmap::AngleAxisToRotationMatrix(
            rotation_estimated_.segment(image_id_to_idx_[image_id2], 3));
      }

      if (idx_cam1 != -1) {
        // If the camera is not part of a rig, then we can use the first image
        // to initialize the rotation
        R_1 = colmap::AngleAxisToRotationMatrix(
                  rotation_estimated_.segment(idx_cam1, 3)) *
              R_1;
      }
      if (idx_cam2 != -1) {
        R_2 = colmap::AngleAxisToRotationMatrix(
                  rotation_estimated_.segment(idx_cam2, 3)) *
              R_2;
      }

      tangent_space_residual_.segment(pair_info.index, 3) =
          -colmap::RotationMatrixToAngleAxis(R_2.transpose() * pair_info.R_rel *
                                             R_1);
    }
  }

  const Image& fixed_image = images.at(fixed_image_id_);
  const auto fixed_pose_prior_it =
      frame_to_pose_prior.find(fixed_image.frame_id);
  const bool fixed_image_has_gravity =
      fixed_pose_prior_it != frame_to_pose_prior.end() &&
      fixed_pose_prior_it->second->HasGravity();
  if (options_.use_gravity && fixed_image_has_gravity) {
    tangent_space_residual_[tangent_space_residual_.size() - 1] =
        rotation_estimated_[image_id_to_idx_[fixed_image_id_]] -
        fixed_camera_rotation_[1];
  } else {
    tangent_space_residual_.segment(tangent_space_residual_.size() - 3, 3) =
        colmap::RotationMatrixToAngleAxis(
            colmap::AngleAxisToRotationMatrix(fixed_camera_rotation_)
                .transpose() *
            colmap::AngleAxisToRotationMatrix(rotation_estimated_.segment(
                image_id_to_idx_[fixed_image_id_], 3)));
  }
}

double RotationEstimator::ComputeAverageStepSize(
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<frame_t, colmap::PosePrior*>&
        frame_to_pose_prior) {
  double total_update = 0;
  for (const auto& [frame_id, frame] : frames) {
    if (frames.at(frame_id).is_registered) continue;
    const auto pose_prior_it = frame_to_pose_prior.find(frame_id);
    const bool has_gravity = pose_prior_it != frame_to_pose_prior.end() &&
                             pose_prior_it->second->HasGravity();
    if (options_.use_gravity && has_gravity) {
      total_update += std::abs(tangent_space_step_[frame_id_to_idx_[frame_id]]);
    } else {
      total_update +=
          tangent_space_step_.segment(frame_id_to_idx_[frame_id], 3).norm();
    }
  }
  return total_update / frame_id_to_idx_.size();
}

void RotationEstimator::ConvertResults(
    std::unordered_map<rig_t, Rig>& rigs,
    std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<frame_t, colmap::PosePrior*>&
        frame_to_pose_prior) {
  for (auto& [frame_id, frame] : frames) {
    if (!frames[frame_id].is_registered) continue;

    const image_t image_id_begin = frame.DataIds().begin()->id;

    // Set the rig from world rotation
    // If the frame has gravity, then use the first image's gravity
    const auto pose_prior_it = frame_to_pose_prior.find(frame_id);
    const bool has_gravity = pose_prior_it != frame_to_pose_prior.end() &&
                             pose_prior_it->second->HasGravity();

    if (options_.use_gravity && has_gravity) {
      frame.SetRigFromWorld(Rigid3d(
          Eigen::Quaterniond(
              GetAlignRot(pose_prior_it->second->gravity) *
              AngleToRotUp(
                  rotation_estimated_[image_id_to_idx_[image_id_begin]])),
          Eigen::Vector3d::Zero()));
    } else {
      frame.SetRigFromWorld(Rigid3d(
          Eigen::Quaterniond(
              colmap::AngleAxisToRotationMatrix(rotation_estimated_.segment(
                  image_id_to_idx_[image_id_begin], 3))),
          Eigen::Vector3d::Zero()));
    }
  }

  // add the estimated
  for (auto& [rig_id, rig] : rigs) {
    for (auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (camera_id_to_idx_.find(sensor_id.id) == camera_id_to_idx_.end()) {
        continue;  // Skip cameras that are not estimated
      }
      Rigid3d cam_from_rig;
      cam_from_rig.rotation = colmap::AngleAxisToRotationMatrix(
          rotation_estimated_.segment(camera_id_to_idx_[sensor_id.id], 3));
      cam_from_rig.translation.setConstant(
          std::numeric_limits<double>::quiet_NaN());  // No translation yet
      rig.SetSensorFromRig(sensor_id, cam_from_rig);
    }
  }
}

}  // namespace glomap
