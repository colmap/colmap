// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "base/cost_functions.h"
#include "base/reconstruction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

using namespace colmap;

namespace config = boost::program_options;

const double kLossFunctionScale = 4.0;

////////////////////////////////////////////////////////////////////////////////
// LiftedBundleAdjustmentCostFunction
////////////////////////////////////////////////////////////////////////////////

template <typename CameraModel>
class LiftedBundleAdjustmentCostFunction {
 public:
  LiftedBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : point2D_(point2D) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            LiftedBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
            CameraModel::num_params, 1>(
        new LiftedBundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  const T* lift_weight, T* residuals) const {
    // Rotate and translate.
    T point3D_local[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, point3D_local);
    point3D_local[0] += tvec[0];
    point3D_local[1] += tvec[1];
    point3D_local[2] += tvec[2];

    // Normalize to image plane.
    point3D_local[0] /= point3D_local[2];
    point3D_local[1] /= point3D_local[2];

    // Distort and transform to pixel space.
    T x, y;
    CameraModel::WorldToImage(camera_params, point3D_local[0], point3D_local[1],
                              &x, &y);

    // Re-projection error.
    residuals[0] = x - T(point2D_(0));
    residuals[1] = y - T(point2D_(1));

    // Weighted residuals.
    residuals[0] *= lift_weight[0];
    residuals[1] *= lift_weight[0];

    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
};

////////////////////////////////////////////////////////////////////////////////
// LiftedBundleAdjustmentConstantPoseCostFunction
////////////////////////////////////////////////////////////////////////////////

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class LiftedBundleAdjustmentConstantPoseCostFunction {
 public:
  LiftedBundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                                 const Eigen::Vector3d& tvec,
                                                 const Eigen::Vector2d& point2D)
      : qvec_(qvec), tvec_(tvec), point2D_(point2D) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            LiftedBundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
            CameraModel::num_params, 1>(
        new LiftedBundleAdjustmentConstantPoseCostFunction(qvec, tvec,
                                                           point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  const T* lift_weight, T* residuals) const {
    T qvec[4] = {T(qvec_(0)), T(qvec_(1)), T(qvec_(2)), T(qvec_(3))};

    // Rotate and translate.
    T point3D_local[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, point3D_local);
    point3D_local[0] += T(tvec_(0));
    point3D_local[1] += T(tvec_(1));
    point3D_local[2] += T(tvec_(2));

    // Normalize to image plane.
    point3D_local[0] /= point3D_local[2];
    point3D_local[1] /= point3D_local[2];

    // Distort and transform to pixel space.
    T x, y;
    CameraModel::WorldToImage(camera_params, point3D_local[0], point3D_local[1],
                              &x, &y);

    // Re-projection error.
    residuals[0] = x - T(point2D_(0));
    residuals[1] = y - T(point2D_(1));

    // Weighted residuals.
    residuals[0] *= lift_weight[0];
    residuals[1] *= lift_weight[0];

    return true;
  }

 private:
  const Eigen::Vector4d qvec_;
  const Eigen::Vector3d tvec_;
  const Eigen::Vector2d point2D_;
};

////////////////////////////////////////////////////////////////////////////////
// LiftedResidualWeightRegularizer
////////////////////////////////////////////////////////////////////////////////

// The regularizer for the quadratic truncated cost, i.e. p = 2.
class LiftedResidualWeightRegularizer {
 public:
  LiftedResidualWeightRegularizer(const double tau)
      : factor_(tau / std::sqrt(2)) {}

  static ceres::CostFunction* Create(const double tau) {
    return (
        new ceres::AutoDiffCostFunction<LiftedResidualWeightRegularizer, 1, 1>(
            new LiftedResidualWeightRegularizer(tau)));
  }

  template <typename T>
  bool operator()(const T* const lift_weight, T* residuals) const {
    residuals[0] = T(factor_) * (lift_weight[0] * lift_weight[0] - T(1.0));
    return true;
  }

 private:
  const double factor_;
};

////////////////////////////////////////////////////////////////////////////////
// LiftedBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

class LiftedBundleAdjuster {
 public:
  LiftedBundleAdjuster(const BundleAdjuster::Options& options,
                       const BundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction);

  // Get the Ceres solver summary for the last call to `Solve`.
  ceres::Solver::Summary Summary() const;

 private:
  void SetUp(Reconstruction* reconstruction,
             ceres::LossFunction* loss_function);
  void TearDown(Reconstruction* reconstruction);

  void AddImageToProblem(const image_t image_id, Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

  void AddPointToProblem(const point3D_t point3D_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

 protected:
  void ParameterizeCameras(Reconstruction* reconstruction);
  void ParameterizePoints(Reconstruction* reconstruction);

  const BundleAdjuster::Options options_;
  BundleAdjustmentConfig config_;
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_images_;
  std::list<double> lift_weights_;
};

LiftedBundleAdjuster::LiftedBundleAdjuster(
    const BundleAdjuster::Options& options,
    const BundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  options_.Check();
}

bool LiftedBundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK(!problem_) << "Cannot use the same LiftedBundleAdjuster multiple times";

  const double reproj_error_before =
      reconstruction->ComputeMeanReprojectionError();

  point3D_num_images_.clear();

  problem_.reset(new ceres::Problem());

  ceres::LossFunction* loss_function = options_.CreateLossFunction();
  SetUp(reconstruction, loss_function);

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
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction);

  std::cout << "Lift weights:" << std::endl;
  for (const auto& lift_weight : lift_weights_) {
    std::cout << lift_weight << " ";
  }
  std::cout << std::endl;

  std::cout << "Mean reprojection error (before):" << reproj_error_before
            << std::endl;
  std::cout << "Mean reprojection error (after):"
            << reconstruction->ComputeMeanReprojectionError() << std::endl;

  return true;
}

ceres::Solver::Summary LiftedBundleAdjuster::Summary() const {
  return summary_;
}

void LiftedBundleAdjuster::SetUp(Reconstruction* reconstruction,
                                 ceres::LossFunction* loss_function) {
  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
  // Do not change order of instructions!
  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void LiftedBundleAdjuster::TearDown(Reconstruction*) {
  // Nothing to do
}

void LiftedBundleAdjuster::AddImageToProblem(
    const image_t image_id, Reconstruction* reconstruction,
    ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();

  double* qvec_data = image.Qvec().data();
  double* tvec_data = image.Tvec().data();
  double* camera_params_data = camera.ParamsData();

  // Collect cameras for final parameterization.
  CHECK(image.HasCamera());

  const bool constant_pose = config_.HasConstantPose(image_id);

  // Add residuals to bundle adjustment problem.
  size_t num_observations = 0;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    num_observations += 1;
    point3D_num_images_[point2D.Point3DId()] += 1;

    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);

    lift_weights_.push_back(1.0);
    problem_->AddResidualBlock(
        LiftedResidualWeightRegularizer::Create(kLossFunctionScale), nullptr,
        &lift_weights_.back());

    ceres::CostFunction* cost_function = nullptr;

    if (constant_pose) {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                       \
  case CameraModel::model_id:                                                \
    cost_function =                                                          \
        LiftedBundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                       \
    problem_->AddResidualBlock(cost_function, nullptr, point3D.XYZ().data(), \
                               camera_params_data, &lift_weights_.back());   \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    cost_function =                                                            \
        LiftedBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    problem_->AddResidualBlock(cost_function, nullptr, qvec_data, tvec_data,   \
                               point3D.XYZ().data(), camera_params_data,       \
                               &lift_weights_.back());                         \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    }
  }

  if (num_observations > 0) {
    camera_ids_.insert(image.CameraId());

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
}

void LiftedBundleAdjuster::AddPointToProblem(
    const point3D_t point3D_id, Reconstruction* reconstruction,
    ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_images_[point3D_id] == point3D.Track().Length()) {
    return;
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

    lift_weights_.push_back(1.0);
    problem_->AddResidualBlock(
        LiftedResidualWeightRegularizer::Create(kLossFunctionScale), nullptr,
        &lift_weights_.back());

    ceres::CostFunction* cost_function = nullptr;

    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                       \
  case CameraModel::model_id:                                                \
    cost_function =                                                          \
        LiftedBundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY());                       \
    problem_->AddResidualBlock(cost_function, loss_function,                 \
                               point3D.XYZ().data(), camera.ParamsData());   \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
  }
}

void LiftedBundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
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

void LiftedBundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
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

////////////////////////////////////////////////////////////////////////////////
// LiftedBundleAdjustmentController
////////////////////////////////////////////////////////////////////////////////

class LiftedBundleAdjustmentController : public Thread {
 public:
  LiftedBundleAdjustmentController(const OptionManager& options,
                                   Reconstruction* reconstruction);

 private:
  void Run();

  const OptionManager options_;
  Reconstruction* reconstruction_;
};

LiftedBundleAdjustmentController::LiftedBundleAdjustmentController(
    const OptionManager& options, Reconstruction* reconstruction)
    : options_(options), reconstruction_(reconstruction) {}

void LiftedBundleAdjustmentController::Run() {
  CHECK_NOTNULL(reconstruction_);

  PrintHeading1("Global bundle adjustment");

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  if (reg_image_ids.size() < 2) {
    std::cout << "ERROR: Need at least two views." << std::endl;
    reconstruction_ = nullptr;
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  BundleAdjuster::Options ba_options = options_.ba_options->Options();
  ba_options.solver_options.minimizer_progress_to_stdout = true;

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }
  ba_config.SetConstantPose(reg_image_ids[0]);
  ba_config.SetConstantTvec(reg_image_ids[1], {0});

  // Run bundle adjustment.
  LiftedBundleAdjuster bundle_adjuster(ba_options, ba_config);
  bundle_adjuster.Solve(reconstruction_);

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction_->Normalize();

  GetTimer().PrintMinutes();
}

////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddBundleAdjustmentOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);

  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (options.ParseHelp(argc, argv)) {
    return EXIT_SUCCESS;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  LiftedBundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}
