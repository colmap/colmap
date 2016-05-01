// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "optim/bundle_adjustment.h"

#include <iomanip>
#include <stdexcept>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/pose.h"
#include "base/projection.h"
#include "ann_float/ANN.h"
#include "pba/pba.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/timer.h"

namespace colmap {

BundleAdjustmentConfiguration::BundleAdjustmentConfiguration() {}

size_t BundleAdjustmentConfiguration::NumImages() const {
  return image_ids_.size();
}

size_t BundleAdjustmentConfiguration::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfiguration::NumConstantCameras() const {
  return constant_camera_ids_.size();
}

size_t BundleAdjustmentConfiguration::NumConstantPoses() const {
  return constant_poses_.size();
}

size_t BundleAdjustmentConfiguration::NumConstantTvecs() const {
  return constant_tvecs_.size();
}

size_t BundleAdjustmentConfiguration::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfiguration::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

void BundleAdjustmentConfiguration::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool BundleAdjustmentConfiguration::HasImage(const image_t image_id) const {
  return image_ids_.count(image_id) > 0;
}

void BundleAdjustmentConfiguration::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentConfiguration::SetConstantCamera(
    const camera_t camera_id) {
  constant_camera_ids_.insert(camera_id);
}

void BundleAdjustmentConfiguration::SetVariableCamera(
    const camera_t camera_id) {
  constant_camera_ids_.erase(camera_id);
}

bool BundleAdjustmentConfiguration::IsConstantCamera(
    const camera_t camera_id) const {
  return constant_camera_ids_.count(camera_id) > 0;
}

void BundleAdjustmentConfiguration::SetConstantPose(const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantTvec(image_id));
  constant_poses_.insert(image_id);
}

void BundleAdjustmentConfiguration::SetVariablePose(const image_t image_id) {
  constant_poses_.erase(image_id);
}

bool BundleAdjustmentConfiguration::HasConstantPose(
    const image_t image_id) const {
  return constant_poses_.count(image_id) > 0;
}

void BundleAdjustmentConfiguration::SetConstantTvec(
    const image_t image_id, const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantPose(image_id));
  std::vector<int> unique_idxs = idxs;
  CHECK(std::unique(unique_idxs.begin(), unique_idxs.end()) ==
        unique_idxs.end())
      << "Tvec indices must not contain duplicates";
  constant_tvecs_.emplace(image_id, idxs);
}

void BundleAdjustmentConfiguration::RemoveConstantTvec(const image_t image_id) {
  constant_tvecs_.erase(image_id);
}

bool BundleAdjustmentConfiguration::HasConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.count(image_id) > 0;
}

const std::unordered_set<image_t>& BundleAdjustmentConfiguration::Images()
    const {
  return image_ids_;
}

const std::unordered_set<point3D_t>&
BundleAdjustmentConfiguration::VariablePoints() const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>&
BundleAdjustmentConfiguration::ConstantPoints() const {
  return constant_point3D_ids_;
}

const std::vector<int>& BundleAdjustmentConfiguration::ConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.at(image_id);
}

void BundleAdjustmentConfiguration::AddVariablePoint(
    const point3D_t point3D_id) {
  CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfiguration::AddConstantPoint(
    const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfiguration::HasPoint(const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfiguration::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.count(point3D_id) > 0;
}

bool BundleAdjustmentConfiguration::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.count(point3D_id) > 0;
}

void BundleAdjustmentConfiguration::RemoveVariablePoint(
    const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfiguration::RemoveConstantPoint(
    const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

ceres::LossFunction* BundleAdjuster::Options::CreateLossFunction() const {
  ceres::LossFunction* loss_function = nullptr;
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      loss_function = new ceres::TrivialLoss();
      break;
    case LossFunctionType::CAUCHY:
      loss_function = new ceres::CauchyLoss(loss_function_scale);
      break;
  }
  return loss_function;
}

void BundleAdjuster::Options::Check() const {
  CHECK_GE(min_observations_per_image, 0);
}

BundleAdjuster::BundleAdjuster(const Options& options,
                               const BundleAdjustmentConfiguration& config)
    : options_(options), config_(config) {
  options_.Check();
}

bool BundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  point3D_num_images_.clear();

  problem_.reset(new ceres::Problem());

  ceres::LossFunction* loss_function = options_.CreateLossFunction();
  SetUp(reconstruction, loss_function);

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

#ifdef OPENMP_ENABLED
  if (solver_options.num_threads <= 0) {
    solver_options.num_threads = omp_get_max_threads();
  }
  if (solver_options.num_linear_solver_threads <= 0) {
    solver_options.num_linear_solver_threads = omp_get_max_threads();
  }
#else
  solver_options.num_threads = 1;
  solver_options.num_linear_solver_threads = 1;
#endif

  std::string error;
  CHECK(solver_options.IsValid(&error)) << error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Bundle Adjustment Report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction);

  return true;
}

ceres::Solver::Summary BundleAdjuster::Summary() const { return summary_; }

void BundleAdjuster::SetUp(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
  // Warning: FillPoints assumes that FillImages is called first.
  // Do not change order of instructions!

  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, loss_function);
  }

  FillPoints(config_.VariablePoints(), reconstruction, loss_function);
  FillPoints(config_.ConstantPoints(), reconstruction, loss_function);
}

void BundleAdjuster::TearDown(Reconstruction*) {
  // Nothing to do
}

void BundleAdjuster::AddImageToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);

  if (image.NumPoints3D() <
      static_cast<size_t>(options_.min_observations_per_image)) {
    return;
  }

  Camera& camera = reconstruction->Camera(image.CameraId());

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();

  double* qvec_data = image.Qvec().data();
  double* tvec_data = image.Tvec().data();
  double* camera_params_data = camera.ParamsData();

  // Collect cameras for final parameterization.
  CHECK(image.HasCamera());
  camera_ids_.insert(image.CameraId());

  const bool constant_pose = config_.HasConstantPose(image_id);

  // Add residuals to bundle adjustment problem.
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    point3D_num_images_[point2D.Point3DId()] += 1;

    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);

    ceres::CostFunction* cost_function = nullptr;

    if (constant_pose) {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::model_id:                                             \
    cost_function =                                                       \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(    \
            image.Qvec(), image.Tvec(), point2D.XY());                    \
    problem_->AddResidualBlock(cost_function, loss_function,              \
                               point3D.XYZ().data(), camera_params_data); \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::model_id:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    problem_->AddResidualBlock(cost_function, loss_function, qvec_data,  \
                               tvec_data, point3D.XYZ().data(),          \
                               camera_params_data);                      \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    }
  }

  // Set pose parameterization.
  if (!constant_pose) {
    ceres::LocalParameterization* quaternion_parameterization =
        new ceres::QuaternionParameterization;
    problem_->SetParameterization(qvec_data, quaternion_parameterization);
    if (config_.HasConstantTvec(image_id)) {
      const std::vector<int>& constant_tvec_idxs =
          config_.ConstantTvec(image_id);
      ceres::SubsetParameterization* tvec_parameterization =
          new ceres::SubsetParameterization(3, constant_tvec_idxs);
      problem_->SetParameterization(tvec_data, tvec_parameterization);
    }
  }
}

void BundleAdjuster::FillPoints(
    const std::unordered_set<point3D_t>& point3D_ids,
    Reconstruction* reconstruction, ceres::LossFunction* loss_function) {
  for (const point3D_t point3D_id : point3D_ids) {
    Point3D& point3D = reconstruction->Point3D(point3D_id);

    // Is 3D point already fully contained in the problem? I.e. its entire track
    // is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.
    if (point3D_num_images_[point3D_id] == point3D.Track().Length()) {
      continue;
    }

    for (const auto& track_el : point3D.Track().Elements()) {
      // Skip observations that were already added in `FillImages`.
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }

      point3D_num_images_[point3D_id] += 1;

      Image& image = reconstruction->Image(track_el.image_id);
      Camera& camera = reconstruction->Camera(image.CameraId());
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);

      // We do not want to refine the camera of images that are not
      // part of `constant_image_ids_`, `constant_image_ids_`,
      // `constant_x_image_ids_`.
      if (camera_ids_.count(image.CameraId()) == 0) {
        camera_ids_.insert(image.CameraId());
        config_.SetConstantCamera(image.CameraId());
      }

      ceres::CostFunction* cost_function = nullptr;

      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::model_id:                                              \
    cost_function =                                                        \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(     \
            image.Qvec(), image.Tvec(), point2D.XY());                     \
    problem_->AddResidualBlock(cost_function, loss_function,               \
                               point3D.XYZ().data(), camera.ParamsData()); \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    }
  }
}

void BundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;
  for (const camera_t camera_id : camera_ids_) {
    Camera& camera = reconstruction->Camera(camera_id);

    if (constant_camera || config_.IsConstantCamera(camera_id)) {
      problem_->SetParameterBlockConstant(camera.ParamsData());
      continue;
    } else {
      std::vector<int> const_camera_params;

      if (!options_.refine_focal_length) {
        const std::vector<size_t>& params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_principal_point) {
        const std::vector<size_t>& params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_extra_params) {
        const std::vector<size_t>& params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        ceres::SubsetParameterization* camera_params_parameterization =
            new ceres::SubsetParameterization(
                static_cast<int>(camera.NumParams()), const_camera_params);
        problem_->SetParameterization(camera.ParamsData(),
                                      camera_params_parameterization);
      }
    }
  }
}

void BundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
  for (const auto num_images : point3D_num_images_) {
    if (!config_.HasVariablePoint(num_images.first) &&
        !config_.HasConstantPoint(num_images.first)) {
      Point3D& point3D = reconstruction->Point3D(num_images.first);
      if (point3D.Track().Length() > point3D_num_images_[num_images.first]) {
        problem_->SetParameterBlockConstant(point3D.XYZ().data());
      }
    }
  }

  for (const point3D_t point3D_id : config_.ConstantPoints()) {
    Point3D& point3D = reconstruction->Point3D(point3D_id);
    problem_->SetParameterBlockConstant(point3D.XYZ().data());
  }
}

void ParallelBundleAdjuster::Options::Check() const {
  CHECK_GE(min_observations_per_image, 0);
  CHECK_GE(max_num_iterations, 0);
}

ParallelBundleAdjuster::ParallelBundleAdjuster(
    const Options& options, const BundleAdjustmentConfiguration& config)
    : config_(config), options_(options), num_measurements_(0) {
  options_.Check();
  CHECK(config_.NumConstantTvecs() == 0)
      << "PBA does not allow to set individual translational elements constant";
  CHECK(config_.NumVariablePoints() == 0 && config_.NumConstantPoints() == 0)
      << "PBA does not allow to parameterize individual 3D points";
}

bool ParallelBundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK_EQ(num_measurements_, 0)
      << "Cannot use the same ParallelBundleAdjuster multiple times";

  SetUp(reconstruction);

  pba::ParallelBA::DeviceT device;
  if (options_.gpu_index < 0) {
    device = pba::ParallelBA::PBA_CUDA_DEVICE_DEFAULT;
  } else {
    device = static_cast<pba::ParallelBA::DeviceT>(
        pba::ParallelBA::PBA_CUDA_DEVICE0 + options_.gpu_index);
  }

  pba::ParallelBA pba(device);
  pba.SetNextBundleMode(pba::ParallelBA::BUNDLE_FULL);
  pba.EnableRadialDistortion(pba::ParallelBA::PBA_PROJECTION_DISTORTION);

  pba::ConfigBA* pba_config = pba.GetInternalConfig();
  pba_config->__lm_mse_threshold = 0.0f;
  pba_config->__cg_min_iteration = 10;
  pba_config->__verbose_level = 2;
  pba_config->__lm_max_iteration = options_.max_num_iterations;

  pba.SetCameraData(cameras_.size(), cameras_.data());
  pba.SetPointData(points3D_.size(), points3D_.data());
  pba.SetProjection(measurements_.size(), measurements_.data(),
                    point3D_idxs_.data(), camera_idxs_.data());

  Timer timer;
  timer.Start();
  pba.RunBundleAdjustment();
  timer.Pause();

  // Compose Ceres solver summary from PBA options.
  ceres::Solver::Summary summary;
  summary.num_residuals_reduced = static_cast<int>(2 * measurements_.size());
  summary.num_effective_parameters_reduced =
      static_cast<int>(8 * config_.NumImages() -
                       2 * config_.NumConstantCameras() + 3 * points3D_.size());
  summary.num_successful_steps = pba_config->GetIterationsLM();
  summary.termination_type = ceres::TerminationType::USER_SUCCESS;
  summary.initial_cost =
      pba_config->GetInitialMSE() * summary.num_residuals_reduced / 4;
  summary.final_cost =
      pba_config->GetFinalMSE() * summary.num_residuals_reduced / 4;
  summary.total_time_in_seconds = timer.ElapsedSeconds();

  TearDown(reconstruction);

  if (options_.print_summary) {
    PrintHeading2("Bundle Adjustment Report");
    PrintSolverSummary(summary);
  }

  return true;
}

void ParallelBundleAdjuster::SetUp(Reconstruction* reconstruction) {
  // Important: PBA requires the track of 3D points to be stored
  // contiguously, i.e. the point3D_idxs_ vector contains consecutive indices.
  cameras_.reserve(config_.NumImages());
  camera_ids_.reserve(config_.NumImages());
  ordered_image_ids_.reserve(config_.NumImages());
  image_id_to_camera_idx_.reserve(config_.NumImages());
  FillImages(reconstruction);
  FillPoints(reconstruction);
}

void ParallelBundleAdjuster::TearDown(Reconstruction* reconstruction) {
  for (size_t i = 0; i < cameras_.size(); ++i) {
    const image_t image_id = ordered_image_ids_[i];
    pba::CameraT& pba_camera = cameras_[i];

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    Image& image = reconstruction->Image(image_id);
    Eigen::Matrix3d rotation_matrix;
    pba_camera.GetMatrixRotation(rotation_matrix.data());
    pba_camera.GetTranslation(image.Tvec().data());
    image.Qvec() = RotationMatrixToQuaternion(rotation_matrix.transpose());

    Camera& camera = reconstruction->Camera(image.CameraId());
    camera.Params(0) = pba_camera.GetFocalLength();
    camera.Params(3) = pba_camera.GetProjectionDistortion();
  }

  for (size_t i = 0; i < points3D_.size(); ++i) {
    Point3D& point3D = reconstruction->Point3D(ordered_point3D_ids_[i]);
    points3D_[i].GetPoint(point3D.XYZ().data());
  }
}

void ParallelBundleAdjuster::FillImages(Reconstruction* reconstruction) {
  for (const image_t image_id : config_.Images()) {
    const Image& image = reconstruction->Image(image_id);
    CHECK_EQ(camera_ids_.count(image.CameraId()), 0)
        << "PBA does not support shared intrinsics";

    const Camera& camera = reconstruction->Camera(image.CameraId());
    CHECK_EQ(camera.ModelId(), SimpleRadialCameraModel::model_id)
        << "PBA only supports SimpleRadial camera model";

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    const Eigen::Matrix3d rotation_matrix =
        QuaternionToRotationMatrix(image.Qvec()).transpose();

    pba::CameraT pba_camera;
    pba_camera.SetFocalLength(camera.Params(0));
    pba_camera.SetProjectionDistortion(camera.Params(3));
    pba_camera.SetMatrixRotation(rotation_matrix.data());
    pba_camera.SetTranslation(image.Tvec().data());

    if (config_.HasConstantPose(image_id)) {
      CHECK(config_.IsConstantCamera(image.CameraId()))
          << "PBA cannot fix extrinsics only";
      pba_camera.SetConstantCamera();
    } else if (config_.IsConstantCamera(image.CameraId())) {
      pba_camera.SetFixedIntrinsic();
    } else {
      pba_camera.SetVariableCamera();
    }

    num_measurements_ += image.NumPoints3D();
    cameras_.push_back(pba_camera);
    camera_ids_.insert(image.CameraId());
    ordered_image_ids_.push_back(image_id);
    image_id_to_camera_idx_.emplace(image_id,
                                    static_cast<int>(cameras_.size()) - 1);

    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids_.insert(point2D.Point3DId());
      }
    }
  }
}

void ParallelBundleAdjuster::FillPoints(Reconstruction* reconstruction) {
  points3D_.resize(point3D_ids_.size());
  ordered_point3D_ids_.resize(point3D_ids_.size());
  measurements_.resize(num_measurements_);
  camera_idxs_.resize(num_measurements_);
  point3D_idxs_.resize(num_measurements_);

  int point3D_idx = 0;
  size_t measurement_idx = 0;

  for (const auto point3D_id : point3D_ids_) {
    const Point3D& point3D = reconstruction->Point3D(point3D_id);
    points3D_[point3D_idx].SetPoint(point3D.XYZ().data());
    ordered_point3D_ids_[point3D_idx] = point3D_id;

    for (const auto track_el : point3D.Track().Elements()) {
      if (image_id_to_camera_idx_.count(track_el.image_id) > 0) {
        const Image& image = reconstruction->Image(track_el.image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        measurements_[measurement_idx].SetPoint2D(
            point2D.X() - camera.Params(1), point2D.Y() - camera.Params(2));
        camera_idxs_[measurement_idx] =
            image_id_to_camera_idx_.at(track_el.image_id);
        point3D_idxs_[measurement_idx] = point3D_idx;
        measurement_idx += 1;
      }
    }
    point3D_idx += 1;
  }

  CHECK_EQ(point3D_idx, points3D_.size());
  CHECK_EQ(measurement_idx, measurements_.size());
}

void PrintSolverSummary(const ceres::Solver::Summary& summary) {
  std::cout << std::right << std::setw(16) << "Residuals : ";
  std::cout << std::left << summary.num_residuals_reduced << std::endl;

  std::cout << std::right << std::setw(16) << "Parameters : ";
  std::cout << std::left << summary.num_effective_parameters_reduced
            << std::endl;

  std::cout << std::right << std::setw(16) << "Iterations : ";
  std::cout << std::left
            << summary.num_successful_steps + summary.num_unsuccessful_steps
            << std::endl;

  std::cout << std::right << std::setw(16) << "Time : ";
  std::cout << std::left << summary.total_time_in_seconds << " [s]"
            << std::endl;

  std::cout << std::right << std::setw(16) << "Initial cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Final cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  std::cout << std::right << termination << std::endl;
  std::cout << std::endl;
}

}  // namespace colmap
