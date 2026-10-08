// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/enum_utils.h"
#include "colmap/util/hash_containers.h"

#include <memory>
#include <vector>

#include <Eigen/Core>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refer
// to the paper "Gravity Aligned Rotation Averaging"
namespace colmap {

// Reweighting scheme applied to the relative-rotation constraints.
//   UNIFORM: all constraints are weighted equally.
//   INLIER_MATCH_COUNT: weight each constraint by the number of inlier
//     two-view matches (PoseGraph::Edge::num_matches) of the corresponding
//     edge, normalized to (0, 1].
//   COVARIANCE: whiten each constraint with the covariance of its relative
//     rotation (PoseGraph::Edge::rot_cov), which must be set for all edges.
//     The global mapper estimates it from the two-view correspondences (see
//     EstimatePoseGraphCovariances). Only supported by the CERES backend,
//     which then uses CeresRotationAveragerOptions::covariance_loss_scale.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    RotationAveragingReweighting, 0, UNIFORM, INLIER_MATCH_COUNT, COVARIANCE);

// Solver backend for rotation averaging.
//   L1_IRLS: L1 regression (ADMM) followed by IRLS using CHOLMOD.
//   CERES: Nonlinear least squares on SO(3) using Ceres Solver.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(RotationAveragingBackend, 0, L1_IRLS, CERES);

struct L1IrlsRotationAveragerOptions;
struct CeresRotationAveragerOptions;

struct RotationEstimatorBackendOptions {
  // L1-IRLS-specific options (used when backend == L1_IRLS and when CERES
  // falls back to L1_IRLS for gravity priors).
  // Type defined in rotation_averaging_l1_irls.h.
  std::shared_ptr<L1IrlsRotationAveragerOptions> l1_irls;

  // Ceres-specific options (only used when backend == CERES).
  // Type defined in rotation_averaging_ceres.h.
  std::shared_ptr<CeresRotationAveragerOptions> ceres;

  RotationEstimatorBackendOptions();
  RotationEstimatorBackendOptions(const RotationEstimatorBackendOptions& other);
  RotationEstimatorBackendOptions& operator=(
      const RotationEstimatorBackendOptions& other);
  RotationEstimatorBackendOptions(RotationEstimatorBackendOptions&& other) =
      default;
  RotationEstimatorBackendOptions& operator=(
      RotationEstimatorBackendOptions&& other) = default;
};

struct RotationEstimatorOptions : public RotationEstimatorBackendOptions {
  // PRNG seed for stochastic methods during rotation averaging.
  // If -1 (default), the seed is derived from the current time
  // (non-deterministic). If >= 0, the rotation averaging is deterministic with
  // the given seed.
  int random_seed = -1;

  // Solver backend to use for rotation averaging. Backend-specific options are
  // in the corresponding member of RotationEstimatorBackendOptions. The CERES
  // backend does not support gravity priors and falls back to L1_IRLS when
  // use_gravity is true and gravity priors are given.
  RotationAveragingBackend backend = RotationAveragingBackend::L1_IRLS;

  // Gravity direction.
  Eigen::Vector3d gravity_dir = Eigen::Vector3d::UnitY();

  // Flag to skip maximum spanning tree initialization.
  bool skip_initialization = false;

  // Flag to use gravity priors for rotation averaging.
  bool use_gravity = false;

  // Flag to use stratified solving for mixed gravity systems.
  // If true and use_gravity is true, first solves the 1-DOF system with
  // gravity-only pairs, then solves the full 3-DOF system.
  bool use_stratified = true;

  // If true, only consider frames with existing poses when computing
  // connected components. Set to true for refinement passes.
  bool filter_unregistered = false;

  // If > 0, filter image pairs with rotation error exceeding this threshold
  // after solving, then recompute active set.
  double max_rotation_error_deg = 10.0;

  // When false, treat each non-ref sensor's cam_from_rig rotation as a
  // pre-calibrated constant
  bool refine_sensor_from_rig = true;

  // Reweighting scheme for the relative-rotation constraints. Weights are
  // applied as a block-diagonal scaling of the linear system, so a noise-free
  // (consistent) system yields an identical solution regardless of reweighting.
  RotationAveragingReweighting reweighting =
      RotationAveragingReweighting::UNIFORM;

  // With COVARIANCE reweighting, isotropic standard deviation (in degrees)
  // added in quadrature to the estimated relative rotation covariances. It
  // accounts for unmodeled errors, e.g., in the intrinsics, and prevents
  // pairs with many correspondences from dominating.
  double covariance_sigma_floor_deg = 0.02;

  // With COVARIANCE reweighting, number of threads for estimating the relative
  // rotation covariances (-1 = auto-select).
  int num_threads = -1;
};

// High-level interface for rotation averaging.
// Combines problem setup and solving into a single call.
// TODO: Refactor this class into free functions (e.g., EstimateGlobalRotations)
// since it holds no state other than options.
class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}

  // Estimates the global orientations of all views.
  // Solves rotation averaging and registers frames with computed poses.
  // active_image_ids defines which images to include.
  // Returns true on successful estimation.
  bool EstimateRotations(const PoseGraph& pose_graph,
                         const std::vector<PosePrior>& pose_priors,
                         const FlatHashSet<image_t>& active_image_ids,
                         Reconstruction& reconstruction);

 private:
  // Maybe solves 1-DOF rotation averaging on the gravity-aligned subset.
  // This is the first phase of stratified solving for mixed gravity systems.
  bool MaybeSolveGravityAlignedSubset(
      const PoseGraph& pose_graph,
      const std::vector<PosePrior>& pose_priors,
      const FlatHashSet<image_t>& active_image_ids,
      Reconstruction& reconstruction);

  // Core rotation averaging solver.
  bool SolveRotationAveraging(const PoseGraph& pose_graph,
                              const std::vector<PosePrior>& pose_priors,
                              const FlatHashSet<image_t>& active_image_ids,
                              Reconstruction& reconstruction);

  bool SolveRotationAveragingWithCeres(
      const PoseGraph& pose_graph,
      const FlatHashSet<image_t>& active_image_ids,
      Reconstruction& reconstruction);

  const RotationEstimatorOptions options_;
};

// Initializes rotations from maximum spanning tree.
void InitializeFromMaximumSpanningTree(
    const PoseGraph& pose_graph,
    const FlatHashSet<image_t>& active_image_ids,
    Reconstruction& reconstruction,
    bool refine_sensor_from_rig);

// Initialize rig rotations by averaging per-image rotations.
// Estimates cam_from_rig for cameras with unknown calibration,
// then computes rig_from_world for each frame.
// When refine_sensor_from_rig is false, the per-sensor cam_from_rig
// values are left untouched.
bool InitializeRigRotationsFromImages(
    const NodeHashMap<image_t, Rigid3d>& cams_from_world,
    Reconstruction& reconstruction,
    bool refine_sensor_from_rig = true);

// High-level rotation averaging solver that handles rig expansion.
// For cameras with unknown cam_from_rig, first estimates their orientations
// independently using an expanded reconstruction, then initializes the
// cam_from_rig and runs rotation averaging on the original reconstruction.
bool RunRotationAveraging(const RotationEstimatorOptions& options,
                          PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          const std::vector<PosePrior>& pose_priors);

// Estimates rotations for the given connected component `active_image_ids`
// and registers its frames. Handles rigs with unknown cam_from_rig by first
// solving on an expanded reconstruction. Does not perform outlier filtering or
// de-registration; callers compose these primitives to implement the desired
// post-processing.
bool RunRotationAveragingOnComponent(
    const RotationEstimatorOptions& options,
    PoseGraph& pose_graph,
    const FlatHashSet<image_t>& active_image_ids,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors);

// Marks image pairs as invalid whose relative rotation disagrees with the
// reconstructed rotations by more than `max_angle_deg`. Pairs whose images do
// not both have a pose are left untouched.
void FilterEdgesByRelativeRotation(PoseGraph& pose_graph,
                                   const Reconstruction& reconstruction,
                                   double max_angle_deg);

}  // namespace colmap
