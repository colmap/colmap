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

#include "colmap/optim/ransac.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/enum_utils.h"

#include <memory>
#include <unordered_set>

#include <Eigen/Core>

namespace colmap {

struct CeresBundleAdjustmentOptions;
struct CeresPosePriorBundleAdjustmentOptions;

MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    BundleAdjustmentGauge, -1, UNSPECIFIED, TWO_CAMS_FROM_WORLD, THREE_POINTS);

// Termination type for bundle adjustment, independent of solver backend.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(BundleAdjustmentTerminationType,
                                0,
                                CONVERGENCE,
                                NO_CONVERGENCE,
                                FAILURE,
                                USER_SUCCESS,
                                USER_FAILURE);

// Backend for bundle adjustment solver.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(BundleAdjustmentBackend, 0, CERES);

// Summary of bundle adjustment results, independent of solver backend.
struct BundleAdjustmentSummary {
  BundleAdjustmentTerminationType termination_type =
      BundleAdjustmentTerminationType::FAILURE;
  // Number of residuals connected to at least one variable parameter block.
  // Excludes residuals where all connected parameters are constant.
  int num_residuals = 0;

  bool IsSolutionUsable() const;
  virtual std::string BriefReport() const;

  virtual ~BundleAdjustmentSummary() = default;
};

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
  void IgnorePoint(point3D_t point3D_id);
  bool HasPoint(point3D_t point3D_id) const;
  bool HasVariablePoint(point3D_t point3D_id) const;
  bool HasConstantPoint(point3D_t point3D_id) const;
  bool IsIgnoredPoint(point3D_t point3D_id) const;
  void RemoveVariablePoint(point3D_t point3D_id);
  void RemoveConstantPoint(point3D_t point3D_id);

  // Access configuration data.
  const std::unordered_set<image_t>& Images() const;
  const std::unordered_set<point3D_t>& VariablePoints() const;
  const std::unordered_set<point3D_t>& ConstantPoints() const;
  const std::unordered_set<camera_t>& ConstantCamIntrinsics() const;
  const std::unordered_set<sensor_t>& ConstantSensorFromRigPoses() const;
  const std::unordered_set<frame_t>& ConstantRigFromWorldPoses() const;

 private:
  BundleAdjustmentGauge fixed_gauge_ = BundleAdjustmentGauge::UNSPECIFIED;
  std::unordered_set<camera_t> constant_cam_intrinsics_;
  std::unordered_set<image_t> image_ids_;
  std::unordered_set<point3D_t> variable_point3D_ids_;
  std::unordered_set<point3D_t> constant_point3D_ids_;
  std::unordered_set<point3D_t> ignored_point3D_ids_;
  std::unordered_set<sensor_t> constant_sensor_from_rig_poses_;
  std::unordered_set<frame_t> constant_rig_from_world_poses_;
};

struct BundleAdjustmentBackendOptions {
  // Ceres-specific options (only used when backend == CERES).
  std::shared_ptr<CeresBundleAdjustmentOptions> ceres;

  BundleAdjustmentBackendOptions();
  BundleAdjustmentBackendOptions(const BundleAdjustmentBackendOptions& other);
  BundleAdjustmentBackendOptions& operator=(
      const BundleAdjustmentBackendOptions& other);
  BundleAdjustmentBackendOptions(BundleAdjustmentBackendOptions&& other) =
      default;
  BundleAdjustmentBackendOptions& operator=(
      BundleAdjustmentBackendOptions&& other) = default;
};

// Solver-agnostic bundle adjustment options.
struct BundleAdjustmentOptions : public BundleAdjustmentBackendOptions {
  // Whether to refine the focal length parameter group.
  bool refine_focal_length = true;

  // Whether to refine the principal point parameter group.
  bool refine_principal_point = false;

  // Whether to refine the extra parameter group.
  bool refine_extra_params = true;

  // Whether to refine the extrinsic parameter group.
  bool refine_sensor_from_rig = true;
  bool refine_rig_from_world = true;

  // Whether to refine the 3D point positions. When false, all 3D points are
  // treated as constant, enabling refinement of only camera intrinsics and
  // poses. This is useful when 3D points come from a reference model and
  // should not be modified.
  bool refine_points3D = true;

  // Minimum track length for a 3D point to be included in bundle adjustment.
  // Points with fewer observations are ignored.
  int min_track_length = 0;

  // Whether to keep the rotation component of rig_from_world constant.
  // Only takes effect when refine_rig_from_world is true.
  // When true, only translation is refined.
  bool constant_rig_from_world_rotation = false;

  // Whether to print a final summary.
  bool print_summary = true;

  // Solver backend to use for bundle adjustment.
  BundleAdjustmentBackend backend = BundleAdjustmentBackend::CERES;

  bool Check() const;
};

// Abstract base class for bundle adjustment, independent of solver backend.
class BundleAdjuster {
 public:
  BundleAdjuster(const BundleAdjustmentOptions& options,
                 const BundleAdjustmentConfig& config);
  virtual ~BundleAdjuster() = default;

  virtual std::shared_ptr<BundleAdjustmentSummary> Solve() = 0;

  const BundleAdjustmentOptions& Options() const;
  const BundleAdjustmentConfig& Config() const;

 protected:
  BundleAdjustmentOptions options_;
  BundleAdjustmentConfig config_;
};

// Factory function to create bundle adjusters.
// Currently uses Ceres as the backend, but can be extended to support
// other backends (e.g., Caspar) in the future.
std::unique_ptr<BundleAdjuster> CreateDefaultBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction);

struct PosePriorBundleAdjustmentBackendOptions {
  // Ceres-specific options (only used when backend == CERES).
  std::shared_ptr<CeresPosePriorBundleAdjustmentOptions> ceres;

  PosePriorBundleAdjustmentBackendOptions();
  PosePriorBundleAdjustmentBackendOptions(
      const PosePriorBundleAdjustmentBackendOptions& other);
  PosePriorBundleAdjustmentBackendOptions& operator=(
      const PosePriorBundleAdjustmentBackendOptions& other);
  PosePriorBundleAdjustmentBackendOptions(
      PosePriorBundleAdjustmentBackendOptions&& other) = default;
  PosePriorBundleAdjustmentBackendOptions& operator=(
      PosePriorBundleAdjustmentBackendOptions&& other) = default;
};

// Solver-agnostic pose prior bundle adjustment options.
struct PosePriorBundleAdjustmentOptions
    : public PosePriorBundleAdjustmentBackendOptions {
  // Fallback if no prior position covariance is provided.
  double prior_position_fallback_stddev = 1.0;

  // Sim3 alignment options.
  RANSACOptions alignment_ransac_options;

  bool Check() const;
};

// Factory function to create pose prior bundle adjusters.
std::unique_ptr<BundleAdjuster> CreatePosePriorBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const PosePriorBundleAdjustmentOptions& prior_options,
    const BundleAdjustmentConfig& config,
    std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction);

}  // namespace colmap
