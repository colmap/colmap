// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/scene/camera_rig.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/eigen_alignment.h"

#include <memory>
#include <unordered_set>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

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
  bool refine_extrinsics = true;

  // Whether to print a final summary.
  bool print_summary = true;

  // Minimum number of residuals to enable multi-threading. Note that
  // single-threaded is typically better for small bundle adjustment problems
  // due to the overhead of threading.
  int min_num_residuals_for_multi_threading = 50000;

  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  BundleAdjustmentOptions() {
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
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

  bool Check() const;
};

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
 public:
  BundleAdjustmentConfig();

  size_t NumImages() const;
  size_t NumPoints() const;
  size_t NumConstantCamIntrinsics() const;
  size_t NumConstantCamPoses() const;
  size_t NumConstantCamPositions() const;
  size_t NumVariablePoints() const;
  size_t NumConstantPoints() const;

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
  void SetConstantCamPose(image_t image_id);
  void SetVariableCamPose(image_t image_id);
  bool HasConstantCamPose(image_t image_id) const;

  // Set the translational part of the pose, hence the constant pose
  // indices may be in [0, 1, 2] and must be unique. Note that the
  // corresponding images have to be added prior to calling these methods.
  void SetConstantCamPositions(image_t image_id, const std::vector<int>& idxs);
  void RemoveConstantCamPositions(image_t image_id);
  bool HasConstantCamPositions(image_t image_id) const;

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
  const std::unordered_set<camera_t> ConstantIntrinsics() const;
  const std::unordered_set<image_t>& Images() const;
  const std::unordered_set<point3D_t>& VariablePoints() const;
  const std::unordered_set<point3D_t>& ConstantPoints() const;
  const std::unordered_set<image_t>& ConstantCamPoses() const;
  const std::vector<int>& ConstantCamPositions(image_t image_id) const;

 private:
  std::unordered_set<camera_t> constant_intrinsics_;
  std::unordered_set<image_t> image_ids_;
  std::unordered_set<point3D_t> variable_point3D_ids_;
  std::unordered_set<point3D_t> constant_point3D_ids_;
  std::unordered_set<image_t> constant_cam_poses_;
  std::unordered_map<image_t, std::vector<int>> constant_cam_positions_;
};

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class BundleAdjuster {
 public:
  BundleAdjuster(const BundleAdjustmentOptions& options,
                 const BundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction);

  // Set up the problem
  void SetUpProblem(Reconstruction* reconstruction,
                    ceres::LossFunction* loss_function);
  ceres::Solver::Options SetUpSolverOptions(
      const ceres::Problem& problem,
      const ceres::Solver::Options& input_solver_options) const;

  // Getter functions below
  const BundleAdjustmentOptions& Options() const;
  const BundleAdjustmentConfig& Config() const;
  // Get the Ceres problem after the last call to "set_up"
  std::shared_ptr<ceres::Problem> Problem();
  // Get the Ceres solver summary after the last call to `Solve`.
  const ceres::Solver::Summary& Summary() const;

 private:
  void AddImageToProblem(image_t image_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

  void AddPointToProblem(point3D_t point3D_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

 protected:
  void ParameterizeCameras(Reconstruction* reconstruction);
  void ParameterizePoints(Reconstruction* reconstruction);

  const BundleAdjustmentOptions options_;
  BundleAdjustmentConfig config_;
  std::shared_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_observations_;

  // Hold the life of loss function for Solve()
  std::unique_ptr<ceres::LossFunction> loss_function_;
};

class RigBundleAdjuster : public BundleAdjuster {
 public:
  struct Options {
    // Whether to optimize the relative poses of the camera rigs.
    bool refine_relative_poses = true;

    // The maximum allowed reprojection error for an observation to be
    // considered in the bundle adjustment. Some observations might have large
    // reprojection errors due to the concatenation of the absolute and relative
    // rig poses, which might be different from the absolute pose of the image
    // in the reconstruction.
    double max_reproj_error = 1000.0;
  };

  RigBundleAdjuster(const BundleAdjustmentOptions& options,
                    const Options& rig_options,
                    const BundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction,
             std::vector<CameraRig>* camera_rigs);

  void SetUpProblem(Reconstruction* reconstruction,
                    std::vector<CameraRig>* camera_rigs,
                    ceres::LossFunction* loss_function);

  void TearDown(Reconstruction* reconstruction,
                const std::vector<CameraRig>& camera_rigs);

 private:
  void AddImageToProblem(image_t image_id,
                         Reconstruction* reconstruction,
                         std::vector<CameraRig>* camera_rigs,
                         ceres::LossFunction* loss_function);

  void AddPointToProblem(point3D_t point3D_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

  void ComputeCameraRigPoses(const Reconstruction& reconstruction,
                             const std::vector<CameraRig>& camera_rigs);

  void ParameterizeCameraRigs(Reconstruction* reconstruction);

  const Options rig_options_;

  // Mapping from images to camera rigs.
  std::unordered_map<image_t, CameraRig*> image_id_to_camera_rig_;

  // Mapping from images to the absolute camera rig poses.
  std::unordered_map<image_t, Rigid3d*> image_id_to_rig_from_world_;

  // For each camera rig, the absolute camera rig poses for all snapshots.
  std::vector<std::vector<Rigid3d>> rigs_from_world_;

  // The Quaternions added to the problem, used to set the local
  // parameterization once after setting up the problem.
  std::unordered_set<double*> parameterized_quats_;
};

void PrintSolverSummary(const ceres::Solver::Summary& summary,
                        const std::string& header);

}  // namespace colmap
