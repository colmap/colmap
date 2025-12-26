#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"

#include "glomap/scene/view_graph.h"

#include <vector>

#include <Eigen/Core>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refer
// to the paper "Gravity Aligned Rotation Averaging"
namespace glomap {

struct RotationEstimatorOptions {
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

  // Flag to use edge weights for rotation averaging.
  bool use_weight = false;

  // Flag to use gravity priors for rotation averaging.
  bool use_gravity = false;
};

// High-level interface for rotation averaging.
// Combines problem setup and solving into a single call.
class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}

  // Estimates the global orientations of all views.
  // Returns true on successful estimation.
  bool EstimateRotations(const ViewGraph& view_graph,
                         const std::vector<colmap::PosePrior>& pose_priors,
                         colmap::Reconstruction& reconstruction);

 private:
  // Initializes rotations from maximum spanning tree.
  void InitializeFromMaximumSpanningTree(
      const ViewGraph& view_graph, colmap::Reconstruction& reconstruction);

  const RotationEstimatorOptions options_;
};

}  // namespace glomap
