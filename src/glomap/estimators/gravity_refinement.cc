#include "gravity_refinement.h"

#include "colmap/estimators/manifold.h"

#include "glomap/estimators/cost_function.h"
#include "glomap/math/gravity.h"

namespace glomap {
void GravityRefiner::RefineGravity(const ViewGraph& view_graph,
                                   std::unordered_map<frame_t, Frame>& frames,
                                   std::unordered_map<image_t, Image>& images) {
  const std::unordered_map<image_t, std::unordered_set<image_t>>&
      adjacency_list = view_graph.CreateImageAdjacencyList();
  if (adjacency_list.empty()) {
    LOG(INFO) << "Adjacency list not established";
    return;
  }

  // Identify the images that are error prone
  int counter_rect = 0;
  std::unordered_set<frame_t> error_prone_frames;
  IdentifyErrorProneGravity(view_graph, frames, images, error_prone_frames);

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
  for (auto frame_id : error_prone_frames) {
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
    Eigen::Vector3d gravity = frames[frame_id].gravity_info.GetGravity();
    for (const auto& pair_id : neighbors) {
      const image_t image_id1 = view_graph.image_pairs.at(pair_id).image_id1;
      const image_t image_id2 = view_graph.image_pairs.at(pair_id).image_id2;
      if (!images.at(image_id1).HasGravity() ||
          !images.at(image_id2).HasGravity())
        continue;

      // Get the cam_from_rig
      Rigid3d cam1_from_rig1, cam2_from_rig2;
      if (!images.at(image_id1).IsReferenceInFrame()) {
        cam1_from_rig1 =
            images.at(image_id1).frame_ptr->RigPtr()->SensorFromRig(
                sensor_t(SensorType::CAMERA, images.at(image_id1).camera_id));
      }
      if (!images.at(image_id2).IsReferenceInFrame()) {
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
             images[image_id2].GetRAlign())
                .col(1));
      } else if (images.at(image_id2).frame_id == frame_id) {
        gravities.emplace_back(
            ((colmap::Inverse(cam2_from_rig2) *
              view_graph.image_pairs.at(pair_id).cam2_from_cam1)
                 .rotation.toRotationMatrix() *
             images[image_id1].GetRAlign())
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
      double error = colmap::RadToDeg(
          std::acos(std::max(std::min(gravities[i].dot(gravity), 1.), -1.)));
      if (error > options_.max_gravity_error * 2) counter_outlier++;
    }
    // If the refined gravity now consistent with more images, then accept it
    if (double(counter_outlier) / double(gravities.size()) <
        options_.max_outlier_ratio) {
      counter_rect++;
      frames[frame_id].gravity_info.SetGravity(gravity);
    }
  }
  LOG(INFO) << "Number of rectified frames: " << counter_rect << " / "
            << error_prone_frames.size();
}

void GravityRefiner::IdentifyErrorProneGravity(
    const ViewGraph& view_graph,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
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
    const auto& image1 = images.at(image_pair.image_id1);
    const auto& image2 = images.at(image_pair.image_id2);

    if (image1.HasGravity() && image2.HasGravity()) {
      // Calculate the gravity aligned relative rotation
      const Eigen::Matrix3d R_rel =
          image2.GetRAlign().transpose() *
          image_pair.cam2_from_cam1.rotation.toRotationMatrix() *
          image1.GetRAlign();
      // Convert it to the closest upright rotation
      const Eigen::Matrix3d R_rel_up = AngleToRotUp(RotUpToAngle(R_rel));

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
    if (double(counter.first) / double(counter.second) >=
        options_.max_outlier_ratio) {
      error_prone_frames.insert(frame_id);
    }
  }
  LOG(INFO) << "Number of error prone frames: " << error_prone_frames.size();
}
}  // namespace glomap
