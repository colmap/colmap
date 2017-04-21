// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#ifndef COLMAP_SRC_OPTIM_BUNDLE_ADJUSTMENT_H_
#define COLMAP_SRC_OPTIM_BUNDLE_ADJUSTMENT_H_

#include <memory>
#include <unordered_set>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "base/camera_rig.h"
#include "base/reconstruction.h"
#include "ext/PBA/pba.h"
#include "util/alignment.h"

namespace colmap {

// Configuration container to setup bundle adjustment problems.
class BundleAdjustmentConfig {
 public:
  BundleAdjustmentConfig();

  size_t NumImages() const;
  size_t NumPoints() const;
  size_t NumConstantCameras() const;
  size_t NumConstantPoses() const;
  size_t NumConstantTvecs() const;
  size_t NumVariablePoints() const;
  size_t NumConstantPoints() const;

  // Determine the number of residuals for the given reconstruction. The number
  // of residuals equals the number of observations times two.
  size_t NumResiduals(const Reconstruction& reconstruction) const;

  // Add / remove images from the configuration.
  void AddImage(const image_t image_id);
  bool HasImage(const image_t image_id) const;
  void RemoveImage(const image_t image_id);

  // Set cameras of added images as constant or variable. By default all
  // cameras of added images are variable. Note that the corresponding images
  // have to be added prior to calling these methods.
  void SetConstantCamera(const camera_t camera_id);
  void SetVariableCamera(const camera_t camera_id);
  bool IsConstantCamera(const camera_t camera_id) const;

  // Set the pose of added images as constant. The pose is defined as the
  // rotational and translational part of the projection matrix.
  void SetConstantPose(const image_t image_id);
  void SetVariablePose(const image_t image_id);
  bool HasConstantPose(const image_t image_id) const;

  // Set the translational part of the pose, hence the constant pose
  // indices may be in [0, 1, 2] and must be unique. Note that the
  // corresponding images have to be added prior to calling these methods.
  void SetConstantTvec(const image_t image_id, const std::vector<int>& idxs);
  void RemoveConstantTvec(const image_t image_id);
  bool HasConstantTvec(const image_t image_id) const;

  // Add / remove points from the configuration. Note that points can either
  // be variable or constant but not both at the same time.
  void AddVariablePoint(const point3D_t point3D_id);
  void AddConstantPoint(const point3D_t point3D_id);
  bool HasPoint(const point3D_t point3D_id) const;
  bool HasVariablePoint(const point3D_t point3D_id) const;
  bool HasConstantPoint(const point3D_t point3D_id) const;
  void RemoveVariablePoint(const point3D_t point3D_id);
  void RemoveConstantPoint(const point3D_t point3D_id);

  // Access configuration data.
  const std::unordered_set<image_t>& Images() const;
  const std::unordered_set<point3D_t>& VariablePoints() const;
  const std::unordered_set<point3D_t>& ConstantPoints() const;
  const std::vector<int>& ConstantTvec(const image_t image_id) const;

 private:
  std::unordered_set<camera_t> constant_camera_ids_;
  std::unordered_set<image_t> image_ids_;
  std::unordered_set<point3D_t> variable_point3D_ids_;
  std::unordered_set<point3D_t> constant_point3D_ids_;
  std::unordered_set<image_t> constant_poses_;
  std::unordered_map<image_t, std::vector<int>> constant_tvecs_;
};

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class BundleAdjuster {
 public:
  struct Options {
    // Loss function types: Trivial (non-robust) and Cauchy (robust) loss.
    enum class LossFunctionType { TRIVIAL, CAUCHY };
    LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

    // Scaling factor determines residual at which robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to refine the focal length parameter group.
    bool refine_focal_length = true;

    // Whether to refine the principal point parameter group.
    bool refine_principal_point = false;

    // Whether to refine the extra parameter group.
    bool refine_extra_params = true;

    // Whether to print a final summary.
    bool print_summary = true;

    // Ceres-Solver options.
    ceres::Solver::Options solver_options;

    Options() {
      solver_options.function_tolerance = 0.0;
      solver_options.gradient_tolerance = 0.0;
      solver_options.parameter_tolerance = 0.0;
      solver_options.minimizer_progress_to_stdout = false;
      solver_options.max_num_iterations = 100;
      solver_options.max_linear_solver_iterations = 200;
      solver_options.max_num_consecutive_invalid_steps = 10;
      solver_options.max_consecutive_nonmonotonic_steps = 10;
      solver_options.num_threads = -1;
      solver_options.num_linear_solver_threads = -1;
    }

    // Create a new loss function based on the specified options. The caller
    // takes ownership of the loss function.
    ceres::LossFunction* CreateLossFunction() const;

    bool Check() const;
  };

  BundleAdjuster(const Options& options, const BundleAdjustmentConfig& config);

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

  const Options options_;
  BundleAdjustmentConfig config_;
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_map<point3D_t, size_t> point3D_num_images_;
};

// Bundle adjustment using PBA (GPU or CPU). Less flexible and accurate than
// Ceres-Solver bundle adjustment but much faster. Only supports SimpleRadial
// camera model.
class ParallelBundleAdjuster {
 public:
  struct Options {
    // Whether to print a final summary.
    bool print_summary = true;

    // Maximum number of iterations.
    int max_num_iterations = 50;

    // Index of the GPU used for bundle adjustment.
    int gpu_index = -1;

    // Number of threads for CPU based bundle adjustment.
    int num_threads = -1;

    bool Check() const;
  };

  ParallelBundleAdjuster(const Options& options,
                         const BundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction);

  // Get the Ceres solver summary for the last call to `Solve`.
  ceres::Solver::Summary Summary() const;

  // Check whether PBA is supported for the given reconstruction. If the
  // reconstruction is not supported, the PBA solver will exit ungracefully.
  static bool IsReconstructionSupported(const Reconstruction& reconstruction);

 private:
  void SetUp(Reconstruction* reconstruction);
  void TearDown(Reconstruction* reconstruction);

  void AddImagesToProblem(Reconstruction* reconstruction);
  void AddPointsToProblem(Reconstruction* reconstruction);

  const Options options_;
  BundleAdjustmentConfig config_;
  ceres::Solver::Summary summary_;

  size_t num_measurements_;
  std::vector<pba::CameraT> cameras_;
  std::vector<pba::Point3D> points3D_;
  std::vector<pba::Point2D> measurements_;
  std::unordered_set<camera_t> camera_ids_;
  std::unordered_set<point3D_t> point3D_ids_;
  std::vector<int> camera_idxs_;
  std::vector<int> point3D_idxs_;
  std::vector<image_t> ordered_image_ids_;
  std::vector<point3D_t> ordered_point3D_ids_;
  std::unordered_map<image_t, int> image_id_to_camera_idx_;
};

class RigBundleAdjuster : public BundleAdjuster {
 public:
  struct RigOptions {
    // Whether to optimize the relative poses of the camera rigs.
    bool refine_relative_poses = true;

    // The maximum allowed reprojection error for an observation to be
    // considered in the bundle adjustment. Some observations might have large
    // reprojection errors due to the concatenation of the absolute and relative
    // rig poses, which might be different from the absolute pose of the image
    // in the reconstruction.
    double max_reproj_error = 1000.0;
  };

  RigBundleAdjuster(const Options& options, const RigOptions& rig_options,
                    const BundleAdjustmentConfig& config);

  bool Solve(Reconstruction* reconstruction,
             std::vector<CameraRig>* camera_rigs);

 private:
  void SetUp(Reconstruction* reconstruction,
             std::vector<CameraRig>* camera_rigs,
             ceres::LossFunction* loss_function);
  void TearDown(Reconstruction* reconstruction,
                const std::vector<CameraRig>& camera_rigs);

  void AddImageToProblem(const image_t image_id, Reconstruction* reconstruction,
                         std::vector<CameraRig>* camera_rigs,
                         ceres::LossFunction* loss_function);

  void AddPointToProblem(const point3D_t point3D_id,
                         Reconstruction* reconstruction,
                         ceres::LossFunction* loss_function);

  void ComputeCameraRigPoses(const Reconstruction& reconstruction,
                             const std::vector<CameraRig>& camera_rigs);

  void ParameterizeCameraRigs(Reconstruction* reconstruction);

  const RigOptions rig_options_;

  // Mapping from images to camera rigs.
  std::unordered_map<image_t, CameraRig*> image_id_to_camera_rig_;

  // Mapping from images to the absolute camera rig poses.
  std::unordered_map<image_t, Eigen::Vector4d*> image_id_to_rig_qvec_;
  std::unordered_map<image_t, Eigen::Vector3d*> image_id_to_rig_tvec_;

  // For each camera rig, the absolute camera rig poses.
  std::vector<std::vector<Eigen::Vector4d>> camera_rig_qvecs_;
  std::vector<std::vector<Eigen::Vector3d>> camera_rig_tvecs_;

  // The Quaternions added to the problem, used to set the local
  // parameterization once after setting up the problem.
  std::unordered_set<double*> parameterized_qvec_data_;
};

void PrintSolverSummary(const ceres::Solver::Summary& summary);

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_BUNDLE_ADJUSTMENT_H_
