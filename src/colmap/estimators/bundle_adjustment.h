// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"

#include <memory>
#include <unordered_set>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    BundleAdjustmentGauge, -1, UNSPECIFIED, TWO_CAMS_FROM_WORLD, THREE_POINTS);

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
 public:
  BundleAdjustmentConfig() = default;

  void FixGauge(BundleAdjustmentGauge gauge);
  BundleAdjustmentGauge FixedGauge() const;

  size_t NumImages() const;

  size_t NumPoints() const;
  size_t NumVariablePoints() const;
  size_t NumConstantPoints() const;

  size_t NumConstantCamIntrinsics() const;

  size_t NumConstantSensorFromRigPoses() const;
  size_t NumConstantRigFromWorldPoses() const;

  // Determine the number of residuals for the given reconstruction. The number
  // of residuals equals the number of observations times two.
  size_t NumResiduals(const Reconstruction& reconstruction) const;

  // Add / remove images from the configuration.
  void AddImage(image_t image_id);
  bool HasImage(image_t image_id) const;
  void RemoveImage(image_t image_id);

  // Set cameras of added images as constant or variable. By default all
  // cameras of added images are variable. Note that the corresponding images
  // have to be added prior to calling these methods.
  void SetConstantCamIntrinsics(camera_t camera_id);
  void SetVariableCamIntrinsics(camera_t camera_id);
  bool HasConstantCamIntrinsics(camera_t camera_id) const;

  // Set the pose of added images as constant. The pose is defined as the
  // rotational and translational part of the projection matrix.
  void SetConstantSensorFromRigPose(sensor_t sensor_id);
  void SetVariableSensorFromRigPose(sensor_t sensor_id);
  bool HasConstantSensorFromRigPose(sensor_t sensor_id) const;

  // Set the rig from world pose as constant.
  void SetConstantRigFromWorldPose(frame_t frame_id);
  void SetVariableRigFromWorldPose(frame_t frame_id);
  bool HasConstantRigFromWorldPose(frame_t frame_id) const;

  // Add / remove points from the configuration. Note that points can either
  // be variable or constant but not both at the same time.
  void AddVariablePoint(point3D_t point3D_id);
  void AddConstantPoint(point3D_t point3D_id);
  bool HasPoint(point3D_t point3D_id) const;
  bool HasVariablePoint(point3D_t point3D_id) const;
  bool HasConstantPoint(point3D_t point3D_id) const;
  void RemoveVariablePoint(point3D_t point3D_id);
  void RemoveConstantPoint(point3D_t point3D_id);

  // Access configuration data.
  const std::unordered_set<image_t>& Images() const;
  const std::unordered_set<point3D_t>& VariablePoints() const;
  const std::unordered_set<point3D_t>& ConstantPoints() const;
  const std::unordered_set<camera_t> ConstantCamIntrinsics() const;
  const std::unordered_set<sensor_t>& ConstantSensorFromRigPoses() const;
  const std::unordered_set<frame_t>& ConstantRigFromWorldPoses() const;

 private:
  BundleAdjustmentGauge fixed_gauge_ = BundleAdjustmentGauge::UNSPECIFIED;
  std::unordered_set<camera_t> constant_cam_intrinsics_;
  std::unordered_set<image_t> image_ids_;
  std::unordered_set<point3D_t> variable_point3D_ids_;
  std::unordered_set<point3D_t> constant_point3D_ids_;
  std::unordered_set<sensor_t> constant_sensor_from_rig_poses_;
  std::unordered_set<frame_t> constant_rig_from_world_poses_;
};

struct BundleAdjustmentOptions {
  // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
  enum class LossFunctionType { TRIVIAL, SOFT_L1, CAUCHY };
  LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

  // Scaling factor determines residual at which robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the focal length parameter group.
  bool refine_focal_length = true;

  // Whether to refine the principal point parameter group.
  bool refine_principal_point = false;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = true;

  // Whether to refine the extrinsic parameter group.
  bool refine_sensor_from_rig = true;
  bool refine_rig_from_world = true;

  // Whether to print a final summary.
  bool print_summary = true;

  // Whether to use Ceres' CUDA linear algebra library, if available.
  bool use_gpu = false;
  std::string gpu_index = "-1";

  // Heuristic threshold to switch from CPU to GPU based solvers.
  // Typically, the GPU is faster for large problems but the overhead of
  // transferring memory from the CPU to the GPU leads to better CPU performance
  // for small problems. This depends on the specific problem and hardware.
  int min_num_images_gpu_solver = 50;

  // Heuristic threshold on the minimum number of residuals to enable
  // multi-threading. Note that single-threaded is typically better for small
  // bundle adjustment problems due to the overhead of threading.
  int min_num_residuals_for_cpu_multi_threading = 50000;

  // Heuristic thresholds to switch between direct, sparse, and iterative
  // solvers. These thresholds may not be optimal for all types of problems.
  int max_num_images_direct_dense_cpu_solver = 50;
  int max_num_images_direct_sparse_cpu_solver = 1000;
  int max_num_images_direct_dense_gpu_solver = 200;
  int max_num_images_direct_sparse_gpu_solver = 4000;

  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  BundleAdjustmentOptions() {
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 1e-4;
    solver_options.parameter_tolerance = 0.0;
    solver_options.logging_type = ceres::LoggingType::SILENT;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
  }

  // Create a new loss function based on the specified options. The caller
  // takes ownership of the loss function.
  ceres::LossFunction* CreateLossFunction() const;

  // Create options tailored for given bundle adjustment config and problem.
  ceres::Solver::Options CreateSolverOptions(
      const BundleAdjustmentConfig& config,
      const ceres::Problem& problem) const;

  bool Check() const;
};

struct PosePriorBundleAdjustmentOptions {
  // Whether to use a robust loss on prior locations.
  bool use_robust_loss_on_prior_position = false;

  // Threshold on the residual for the robust loss
  // (chi2 for 3DOF at 95% = 7.815).
  double prior_position_loss_scale = 7.815;

  // Maximum RANSAC error for Sim3 alignment.
  double ransac_max_error = 0.;
};

class BundleAdjuster {
 public:
  BundleAdjuster(BundleAdjustmentOptions options,
                 BundleAdjustmentConfig config);
  virtual ~BundleAdjuster() = default;

  virtual ceres::Solver::Summary Solve() = 0;
  virtual std::shared_ptr<ceres::Problem>& Problem() = 0;

  const BundleAdjustmentOptions& Options() const;
  const BundleAdjustmentConfig& Config() const;

 protected:
  BundleAdjustmentOptions options_;
  BundleAdjustmentConfig config_;
};

std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
    BundleAdjustmentOptions options,
    BundleAdjustmentConfig config,
    Reconstruction& reconstruction);

std::unique_ptr<BundleAdjuster> CreatePosePriorBundleAdjuster(
    BundleAdjustmentOptions options,
    PosePriorBundleAdjustmentOptions prior_options,
    BundleAdjustmentConfig config,
    std::unordered_map<image_t, PosePrior> pose_priors,
    Reconstruction& reconstruction);

void PrintSolverSummary(const ceres::Solver::Summary& summary,
                        const std::string& header);

}  // namespace colmap
