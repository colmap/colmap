#include "gravity_refinement.h"

#include "colmap/estimators/manifold.h"

#include "glomap/estimators/cost_function.h"
#include "glomap/math/gravity.h"

namespace glomap {
namespace {

Eigen::Matrix3d GetImageAlignRot(const Image& image,
                                 const Eigen::Vector3d& gravity) {
  if (image.IsRefInFrame()) {
    return GetAlignRot(gravity);
  } else {
    return image.FramePtr()
               ->RigPtr()
               ->SensorFromRig(sensor_t(SensorType::CAMERA, image.CameraId()))
               .rotation *
           GetAlignRot(gravity);
  }
}

Eigen::Vector3d* GetImageGravityOrNull(
    const std::unordered_map<image_t, colmap::PosePrior*>& image_to_pose_prior,
    image_t image_id) {
  auto it = image_to_pose_prior.find(image_id);
  if (it == image_to_pose_prior.end() || !it->second->HasGravity()) {
    return nullptr;
  }
  return &it->second->gravity;
}

}  // namespace

void GravityRefiner::RefineGravity(
    const ViewGraph& view_graph,
    const std::unordered_map<frame_t, Frame>& frames,
    const std::unordered_map<image_t, Image>& images,
    std::vector<colmap::PosePrior>& pose_priors) {
  const std::unordered_map<image_t, std::unordered_set<image_t>>&
      adjacency_list = view_graph.CreateImageAdjacencyList();
  if (adjacency_list.empty()) {
    LOG(INFO) << "Adjacency list not established";
    return;
  }

  std::unordered_map<image_t, frame_t> image_to_frame;
  image_to_frame.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    image_to_frame[image_id] = image.FrameId();
  }

  std::unordered_map<image_t, colmap::PosePrior*> image_to_pose_prior;
  std::unordered_map<frame_t, colmap::PosePrior*> frame_to_pose_prior;
  for (auto& pose_prior : pose_priors) {
    if (pose_prior.corr_data_id.sensor_id.type == SensorType::CAMERA) {
      const image_t image_id = pose_prior.corr_data_id.id;
      const Image& image = images.at(image_id);
      // TODO(jsch): Can only handle trivial frames.
      if (image.IsRefInFrame()) {
        THROW_CHECK(image_to_pose_prior.emplace(image_id, &pose_prior).second)
            << "Duplicate pose prior for image " << image_id;
        const frame_t frame_id = image_to_frame.at(image_id);
        THROW_CHECK(frame_to_pose_prior.emplace(frame_id, &pose_prior).second)
            << "Duplicate pose prior for frame" << frame_id;
      }
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
      adjacency_list_frames_to_pair_id[images.at(image_id).FrameId()].insert(
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
      const auto& pair = view_graph.image_pairs.at(pair_id);
      const image_t image_id1 = pair.image_id1;
      const image_t image_id2 = pair.image_id2;

      Eigen::Vector3d* image_gravity1 =
          GetImageGravityOrNull(image_to_pose_prior, image_id1);
      Eigen::Vector3d* image_gravity2 =
          GetImageGravityOrNull(image_to_pose_prior, image_id2);
      if (image_gravity1 == nullptr || image_gravity2 == nullptr) {
        continue;
      }

      const auto& image1 = images.at(image_id1);
      const auto& image2 = images.at(image_id2);

      // Get the cam_from_rig
      Rigid3d cam1_from_rig1, cam2_from_rig2;
      if (!image1.IsRefInFrame()) {
        cam1_from_rig1 = image1.FramePtr()->RigPtr()->SensorFromRig(
            sensor_t(SensorType::CAMERA, image1.CameraId()));
      }
      if (!image2.IsRefInFrame()) {
        cam2_from_rig2 = image2.FramePtr()->RigPtr()->SensorFromRig(
            sensor_t(SensorType::CAMERA, image2.CameraId()));
      }

      // Note: for the case where both cameras are from the same frames, we only
      // consider a single cost term
      if (image1.FrameId() == frame_id) {
        gravities.emplace_back(
            (colmap::Inverse(pair.cam2_from_cam1 * cam1_from_rig1)
                 .rotation.toRotationMatrix() *
             GetImageAlignRot(image2, *image_gravity2))
                .col(1));
      } else if (image2.FrameId() == frame_id) {
        gravities.emplace_back(
            ((colmap::Inverse(cam2_from_rig2) * pair.cam2_from_cam1)
                 .rotation.toRotationMatrix() *
             GetImageAlignRot(image1, *image_gravity1))
                .col(1));
      }

      problem.AddResidualBlock(GravityCostFunctor::Create(gravities[counter]),
                               loss_function_.get(),
                               gravity.data());
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

    Eigen::Vector3d* image_gravity1 =
        GetImageGravityOrNull(image_to_pose_prior, image_pair.image_id1);
    Eigen::Vector3d* image_gravity2 =
        GetImageGravityOrNull(image_to_pose_prior, image_pair.image_id2);
    if (image_gravity1 == nullptr || image_gravity2 == nullptr) {
      continue;
    }

    const auto& image1 = images.at(image_pair.image_id1);
    const auto& image2 = images.at(image_pair.image_id2);
    // Calculate the gravity aligned relative rotation
    const Eigen::Matrix3d R_rel =
        GetImageAlignRot(image2, *image_gravity2).transpose() *
        image_pair.cam2_from_cam1.rotation.toRotationMatrix() *
        GetImageAlignRot(image1, *image_gravity1);
    // Convert it to the closest upright rotation
    const Eigen::Matrix3d R_rel_up = AngleToRotUp(RotUpToAngle(R_rel));

    // increment the total count
    frame_counter[image1.FrameId()].second++;
    frame_counter[image2.FrameId()].second++;

    // increment the mistake count
    if (Eigen::Quaterniond(R_rel).angularDistance(
            Eigen::Quaterniond(R_rel_up)) > max_gravity_error_rad) {
      frame_counter[image1.FrameId()].first++;
      frame_counter[image2.FrameId()].first++;
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
