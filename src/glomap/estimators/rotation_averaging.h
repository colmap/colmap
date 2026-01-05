#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/reconstruction.h"

#include "glomap/scene/pose_graph.h"
#include "glomap/scene/types.h"

#include <vector>

#include <Eigen/Core>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refer
// to the paper "Gravity Aligned Rotation Averaging"
namespace glomap {

struct RotationEstimatorOptions {
  // PRNG seed for stochastic methods during rotation averaging.
  // If -1 (default), the seed is derived from the current time
  // (non-deterministic). If >= 0, the rotation averaging is deterministic with
  // the given seed.
  int random_seed = -1;

  // Maximum number of times to run L1 minimization.
  int max_num_l1_iterations = 5;

  // Average step size threshold to terminate the L1 minimization.
  double l1_step_convergence_threshold = 0.001;

  // The number of iterative reweighted least squares iterations to perform.
  int max_num_irls_iterations = 100;

  // Average step size threshold to terminate the IRLS minimization.
  double irls_step_convergence_threshold = 0.001;

  // Gravity direction.
  Eigen::Vector3d gravity_dir = Eigen::Vector3d(0, 1, 0);

  // The point where the Huber-like cost function switches from L1 to L2.
  double irls_loss_parameter_sigma = 5.0;  // in degrees

  enum WeightType {
    // Geman-McClure weight from "Efficient and robust large-scale rotation
    // averaging" (Chatterjee et al., 2013)
    GEMAN_MCCLURE,
    // Half norm from "Robust Relative Rotation Averaging"
    // (Chatterjee et al., 2017)
    HALF_NORM,
  } weight_type = GEMAN_MCCLURE;

  // Flag to skip maximum spanning tree initialization.
  bool skip_initialization = false;

  // Flag to use gravity priors for rotation averaging.
  bool use_gravity = false;

  // Flag to use stratified solving for mixed gravity systems.
  // If true and use_gravity is true, first solves the 1-DOF system with
  // gravity-only pairs, then solves the full 3-DOF system.
  bool use_stratified = true;
};

// High-level interface for rotation averaging.
// Combines problem setup and solving into a single call.
class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}

  // Estimates the global orientations of all views.
  // Returns true on successful estimation.
  bool EstimateRotations(const PoseGraph& pose_graph,
                         const std::vector<colmap::PosePrior>& pose_priors,
                         colmap::Reconstruction& reconstruction);

 private:
  // Maybe solves 1-DOF rotation averaging on the gravity-aligned subset.
  // This is the first phase of stratified solving for mixed gravity systems.
  bool MaybeSolveGravityAlignedSubset(
      const PoseGraph& pose_graph,
      const std::vector<colmap::PosePrior>& pose_priors,
      colmap::Reconstruction& reconstruction);

  // Core rotation averaging solver.
  bool SolveRotationAveraging(const PoseGraph& pose_graph,
                              const std::vector<colmap::PosePrior>& pose_priors,
                              colmap::Reconstruction& reconstruction);

  // Initializes rotations from maximum spanning tree.
  void InitializeFromMaximumSpanningTree(
      const PoseGraph& pose_graph, colmap::Reconstruction& reconstruction);

  const RotationEstimatorOptions options_;
};

// Initialize rig rotations by averaging per-image rotations.
// Estimates cam_from_rig for cameras with unknown calibration,
// then computes rig_from_world for each frame.
bool InitializeRigRotationsFromImages(
    const std::unordered_map<image_t, Rigid3d>& cams_from_world,
    colmap::Reconstruction& reconstruction);

// High-level rotation averaging solver that handles rig expansion.
// For cameras with unknown cam_from_rig, first estimates their orientations
// independently using an expanded reconstruction, then initializes the
// cam_from_rig and runs rotation averaging on the original reconstruction.
bool SolveRotationAveraging(const RotationEstimatorOptions& options,
                            PoseGraph& pose_graph,
                            colmap::Reconstruction& reconstruction,
                            const std::vector<colmap::PosePrior>& pose_priors);

}  // namespace glomap
