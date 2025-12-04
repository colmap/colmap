#include "gravity_refinement.h"

#include "colmap/estimators/manifold.h"

#include "glomap/estimators/cost_function.h"
#include "glomap/math/gravity.h"

namespace glomap {
void GravityRefiner::RefineGravity(
    const ViewGraph& view_graph,
    std::unordered_map<frame_t, Frame>& frames,
    std::unordered_map<image_t, Image>& images,
    std::vector<colmap::PosePrior>& pose_priors) {
  const std::unordered_map<image_t, std::unordered_set<image_t>>&
      adjacency_list = view_graph.CreateImageAdjacencyList();
  if (adjacency_list.empty()) {
    LOG(INFO) << "Adjacency list not established";
    return;
  }

  std::unordered_map<image_t, frame_t> image_to_frame;
  for (const auto& [image_id, image] : images) {
    image_to_frame[image_id] = image.frame_id;
  }

  std::unordered_map<image_t, colmap::PosePrior*> image_to_pose_prior;
  std::unordered_map<frame_t, colmap::PosePrior*> frame_to_pose_prior;
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      THROW_CHECK(
          image_to_pose_prior.emplace(pose_prior.corr_data_id.id, &pose_prior)
              .second)
          << "Duplicate pose prior for image " << pose_prior.corr_data_id.id;
      const frame_t frame_id = image_to_frame.at(pose_prior.corr_data_id.id);
      THROW_CHECK(frame_to_pose_prior.emplace(frame_id, &pose_prior).second)
          << "Duplicate pose prior for frame" << frame_id;
    }
  }

  // Identify the images that are error prone
  int counter_rect = 0;
  std::unordered_set<frame_t> error_prone_frames;
  IdentifyErrorProneGravity(
      view_graph, frames, images, image_to_pose_prior, error_prone_frames);

  if (error_prone_frames.empty()) {
    LOG(INFO) << "No error prone frames found";
    return;
  }

  // Get the relevant pair ids for frames
  std::unordered_map<frame_t, std::unordered_set<image_pair_t>>
      adjacency_list_frames_to_pair_id;
  for (auto& [image_id, neighbors] : adjacency_list) {
    for (const auto& neighbor : neighbors) {
      adjacency_list_frames_to_pair_id[images[image_id].frame_id].insert(
          colmap::ImagePairToPairId(image_id, neighbor));
    }
  }

  loss_function_ = options_.CreateLossFunction();

  int counter_progress = 0;
  // Iterate through the error prone images
  for (const frame_t frame_id : error_prone_frames) {
    if ((counter_progress + 1) % 10 == 0 ||
        counter_progress == error_prone_frames.size() - 1) {
      std::cout << "\r Refining frame " << counter_progress + 1 << " / "
                << error_prone_frames.size() << std::flush;
    }
    counter_progress++;
    const std::unordered_set<image_pair_t>& neighbors =
        adjacency_list_frames_to_pair_id.at(frame_id);
    std::vector<Eigen::Vector3d> gravities;
    gravities.reserve(neighbors.size());

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    int counter = 0;
    Eigen::Vector3d gravity = frame_to_pose_prior.at(frame_id)->gravity;
    for (const auto& pair_id : neighbors) {
      const image_t image_id1 = view_graph.image_pairs.at(pair_id).image_id1;
      const image_t image_id2 = view_graph.image_pairs.at(pair_id).image_id2;

      const auto pose_prior1_it = image_to_pose_prior.find(image_id1);
      const auto pose_prior2_it = image_to_pose_prior.find(image_id2);
      const bool has_gravity1 = pose_prior1_it != image_to_pose_prior.end() &&
                                pose_prior1_it->second->HasGravity();
      const bool has_gravity2 = pose_prior2_it != image_to_pose_prior.end() &&
                                pose_prior2_it->second->HasGravity();
      if (!has_gravity1 || !has_gravity2) {
        continue;
      }

      // Get the cam_from_rig
      Rigid3d cam1_from_rig1, cam2_from_rig2;
      if (!images.at(image_id1).HasTrivialFrame()) {
        cam1_from_rig1 =
            images.at(image_id1).frame_ptr->RigPtr()->SensorFromRig(
                sensor_t(SensorType::CAMERA, images.at(image_id1).camera_id));
      }
      if (!images.at(image_id2).HasTrivialFrame()) {
        cam2_from_rig2 =
            images.at(image_id2).frame_ptr->RigPtr()->SensorFromRig(
                sensor_t(SensorType::CAMERA, images.at(image_id2).camera_id));
      }

      // Note: for the case where both cameras are from the same frames, we only
      // consider a single cost term
      if (images.at(image_id1).frame_id == frame_id) {
        gravities.emplace_back(
            (colmap::Inverse(view_graph.image_pairs.at(pair_id).cam2_from_cam1 *
                             cam1_from_rig1)
                 .rotation.toRotationMatrix() *
             GetAlignRot(pose_prior2_it->second->gravity))
                .col(1));
      } else if (images.at(image_id2).frame_id == frame_id) {
        gravities.emplace_back(
            ((colmap::Inverse(cam2_from_rig2) *
              view_graph.image_pairs.at(pair_id).cam2_from_cam1)
                 .rotation.toRotationMatrix() *
             GetAlignRot(pose_prior1_it->second->gravity))
                .col(1));
      }

      ceres::CostFunction* coor_cost =
          GravError::CreateCost(gravities[counter]);
      problem.AddResidualBlock(coor_cost, loss_function_.get(), gravity.data());
      counter++;
    }

    if (gravities.size() < options_.min_num_neighbors) continue;

    // Then, run refinment
    gravity = AverageGravity(gravities);
    colmap::SetSphereManifold<3>(&problem, gravity.data());
    ceres::Solver::Summary summary_solver;
    ceres::Solve(options_.solver_options, &problem, &summary_solver);

    // Check the error with respect to the neighbors
    int counter_outlier = 0;
    for (int i = 0; i < gravities.size(); i++) {
      const double error = colmap::RadToDeg(
          std::acos(std::max(std::min(gravities[i].dot(gravity), 1.), -1.)));
      if (error > options_.max_gravity_error * 2) counter_outlier++;
    }
    // If the refined gravity now consistent with more images, then accept it
    if (static_cast<double>(counter_outlier) /
            static_cast<double>(gravities.size()) <
        options_.max_outlier_ratio) {
      counter_rect++;
      frame_to_pose_prior.at(frame_id)->gravity = gravity;
    }
  }

  LOG(INFO) << "Number of refined gravities: " << counter_rect << " / "
            << error_prone_frames.size();
}

void GravityRefiner::IdentifyErrorProneGravity(
    const ViewGraph& view_graph,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<image_t, colmap::PosePrior*>& image_to_pose_prior,
    std::unordered_set<frame_t>& error_prone_frames) {
  error_prone_frames.clear();

  const double max_gravity_error_rad =
      colmap::DegToRad(options_.max_gravity_error);

  // image_id: (mistake, total)
  std::unordered_map<frame_t, std::pair<int, int>> frame_counter;
  frame_counter.reserve(frames.size());
  // Set the counter of all images to 0
  for (const auto& [frame_id, frame] : frames) {
    frame_counter[frame_id] = std::make_pair(0, 0);
  }

  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    const auto pose_prior1_it = image_to_pose_prior.find(image_pair.image_id1);
    const auto pose_prior2_it = image_to_pose_prior.find(image_pair.image_id2);
    const bool has_gravity1 = pose_prior1_it != image_to_pose_prior.end() &&
                              pose_prior1_it->second->HasGravity();
    const bool has_gravity2 = pose_prior2_it != image_to_pose_prior.end() &&
                              pose_prior2_it->second->HasGravity();

    if (has_gravity1 && has_gravity2) {
      // Calculate the gravity aligned relative rotation
      const Eigen::Matrix3d R_rel =
          GetAlignRot(pose_prior2_it->second->gravity).transpose() *
          image_pair.cam2_from_cam1.rotation.toRotationMatrix() *
          GetAlignRot(pose_prior1_it->second->gravity);
      // Convert it to the closest upright rotation
      const Eigen::Matrix3d R_rel_up = AngleToRotUp(RotUpToAngle(R_rel));

      const auto& image1 = images.at(image_pair.image_id1);
      const auto& image2 = images.at(image_pair.image_id2);

      // increment the total count
      frame_counter[image1.frame_id].second++;
      frame_counter[image2.frame_id].second++;

      // increment the mistake count
      if (Eigen::Quaterniond(R_rel).angularDistance(
              Eigen::Quaterniond(R_rel_up)) > max_gravity_error_rad) {
        frame_counter[image1.frame_id].first++;
        frame_counter[image2.frame_id].first++;
      }
    }
  }

  // Filter the images with too many mistakes
  for (const auto& [frame_id, counter] : frame_counter) {
    if (counter.second < options_.min_num_neighbors) continue;
    if (static_cast<double>(counter.first) /
            static_cast<double>(counter.second) >=
        options_.max_outlier_ratio) {
      error_prone_frames.insert(frame_id);
    }
  }
  LOG(INFO) << "Number of error prone frames: " << error_prone_frames.size();
}

}  // namespace glomap
