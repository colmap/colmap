#include "glomap/estimators/view_graph_calibration.h"

#include "colmap/scene/two_view_geometry.h"

#include "glomap/estimators/cost_functions.h"

#include <thread>

namespace glomap {

bool ViewGraphCalibrator::Solve(ViewGraph& view_graph,
                                colmap::Reconstruction& reconstruction) {
  // Reset the problem
  LOG(INFO) << "Start ViewGraphCalibrator";

  Reset(reconstruction);

  // Set the solver options.
  if (reconstruction.NumCameras() < 50)
    options_.solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  else
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  // Add the image pairs into the problem
  AddImagePairsToProblem(view_graph, reconstruction);

  // Set the cameras to be constant if they have prior intrinsics
  const size_t num_cameras = ParameterizeCameras(reconstruction);

  if (num_cameras == 0) {
    LOG(INFO) << "No cameras to optimize";
    return true;
  }

  // Solve the problem
  ceres::Solver::Summary summary;
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  VLOG(2) << summary.FullReport();

  // Convert the results back to the camera
  CopyBackResults(reconstruction);
  FilterImagePairs(view_graph);

  return summary.IsSolutionUsable();
}

void ViewGraphCalibrator::Reset(const colmap::Reconstruction& reconstruction) {
  // Initialize the problem
  focals_.clear();
  focals_.reserve(reconstruction.NumCameras());
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    focals_[camera_id] = camera.MeanFocalLength();
  }

  // Set up the problem
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();
}

void ViewGraphCalibrator::AddImagePairsToProblem(
    const ViewGraph& view_graph, const colmap::Reconstruction& reconstruction) {
  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    if (image_pair.config != colmap::TwoViewGeometry::CALIBRATED &&
        image_pair.config != colmap::TwoViewGeometry::UNCALIBRATED)
      continue;

    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    AddImagePair(image_id1, image_id2, image_pair, reconstruction);
  }
}

void ViewGraphCalibrator::AddImagePair(
    image_t image_id1,
    image_t image_id2,
    const ImagePair& image_pair,
    const colmap::Reconstruction& reconstruction) {
  const camera_t camera_id1 = reconstruction.Image(image_id1).CameraId();
  const camera_t camera_id2 = reconstruction.Image(image_id2).CameraId();

  if (camera_id1 == camera_id2) {
    problem_->AddResidualBlock(
        FetzerFocalLengthSameCameraCostFunctor::Create(
            image_pair.F, reconstruction.Camera(camera_id1).PrincipalPoint()),
        loss_function_.get(),
        &(focals_[camera_id1]));
  } else {
    problem_->AddResidualBlock(
        FetzerFocalLengthCostFunctor::Create(
            image_pair.F,
            reconstruction.Camera(camera_id1).PrincipalPoint(),
            reconstruction.Camera(camera_id2).PrincipalPoint()),
        loss_function_.get(),
        &(focals_[camera_id1]),
        &(focals_[camera_id2]));
  }
}

size_t ViewGraphCalibrator::ParameterizeCameras(
    const colmap::Reconstruction& reconstruction) {
  size_t num_cameras = 0;
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    if (!problem_->HasParameterBlock(&(focals_[camera_id]))) continue;

    num_cameras++;
    problem_->SetParameterLowerBound(&(focals_[camera_id]), 0, 1e-3);
    if (camera.has_prior_focal_length) {
      problem_->SetParameterBlockConstant(&(focals_[camera_id]));
      num_cameras--;
    }
  }

  return num_cameras;
}

void ViewGraphCalibrator::CopyBackResults(
    colmap::Reconstruction& reconstruction) {
  size_t counter = 0;
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    if (!problem_->HasParameterBlock(&(focals_[camera_id]))) continue;

    // if the estimated parameter is too crazy, reject it
    if (focals_[camera_id] / camera.MeanFocalLength() >
            options_.thres_higher_ratio ||
        focals_[camera_id] / camera.MeanFocalLength() <
            options_.thres_lower_ratio) {
      VLOG(2) << "Ignoring degenerate camera camera " << camera_id
              << " focal: " << focals_[camera_id]
              << " original focal: " << camera.MeanFocalLength();
      counter++;

      continue;
    }

    // Update the focal length
    colmap::Camera& cam_ref = reconstruction.Camera(camera_id);
    for (const size_t idx : cam_ref.FocalLengthIdxs()) {
      cam_ref.params[idx] = focals_[camera_id];
    }
  }
  LOG(INFO) << counter << " cameras are rejected in view graph calibration";
}

size_t ViewGraphCalibrator::FilterImagePairs(ViewGraph& view_graph) const {
  ceres::Problem::EvaluateOptions eval_options;
  eval_options.num_threads = options_.solver_options.num_threads;
  eval_options.apply_loss_function = false;
  std::vector<double> residuals;
  problem_->Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr);

  // Dump the residuals into the original data structure
  size_t counter = 0;
  size_t invalid_counter = 0;

  const double thres_two_view_error_sq =
      options_.thres_two_view_error * options_.thres_two_view_error;

  for (const auto& [image_pair_id, image_pair] : view_graph.ImagePairs()) {
    if (image_pair.config != colmap::TwoViewGeometry::CALIBRATED &&
        image_pair.config != colmap::TwoViewGeometry::UNCALIBRATED)
      continue;
    if (!view_graph.IsValid(image_pair_id)) continue;

    const Eigen::Vector2d error(residuals[counter], residuals[counter + 1]);

    // Set the two view geometry to be invalid if the error is too high
    if (error.squaredNorm() > thres_two_view_error_sq) {
      invalid_counter++;
      view_graph.SetInvalidImagePair(image_pair_id);
    }

    counter += 2;
  }

  LOG(INFO) << "invalid / total number of two view geometry: "
            << invalid_counter << " / " << (counter / 2);

  return invalid_counter;
}

}  // namespace glomap
