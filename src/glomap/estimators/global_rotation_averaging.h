#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/rig.h"

#include "glomap/scene/view_graph.h"

#include <string>
#include <variant>
#include <vector>

#include <Eigen/Sparse>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refere
// to the paper "Gravity Aligned Rotation Averaging"
namespace glomap {

// The linear system being solved is: A * x = b
// where:
//   x = [rig_from_world_rotations..., unknown_cam_from_rig_rotations...]
//   b = residuals from relative rotation constraints
//   A = sparse matrix encoding the constraint equations
//
// For each image pair (i, j), the constraint is (in tangent space):
//   cam2_from_cam1 ≈ cam2_from_rig * rig2_from_world * world_from_rig1 *
//   rig1_from_cam
//
// where cam_from_rig is identity for the reference camera, a known value if
// calibrated, or an additional unknown to be optimized.

// Preprocessed constraint information for an image pair.
// Built once in SetupLinearSystem(), used during L1/IRLS solving.
// The constraint geometry is fixed; only residuals change during optimization.
struct PairConstraint {
  // 1-DOF constraint for pairs where both frames have gravity priors.
  // Gravity alignment reduces full 3D rotation to Y-axis rotation only.
  struct GravityAligned1DOF {
    double angle_cam2_from_cam1;  // Relative Y-axis rotation
    double xz_error;  // Squared error in x,z axes (for IRLS weighting)
  };

  // 3-DOF constraint for the general case (no gravity or partial gravity).
  struct Full3DOF {
    Eigen::Matrix3d R_cam2_from_cam1;  // Relative rotation
  };

  // Starting row in sparse matrix A where this pair's equations begin.
  // - Gravity1DOF: occupies 1 row
  // - Full3DOF: occupies 3 consecutive rows
  int row_index = -1;

  // Column indices in x for unknown cam_from_rig rotations.
  // -1 means the camera's cam_from_rig is known (or it's the reference camera).
  // If >= 0, points to 3 consecutive entries in x (always 3D, not reduced by
  // gravity).
  int cam1_from_rig_param_idx = -1;
  int cam2_from_rig_param_idx = -1;

  std::variant<GravityAligned1DOF, Full3DOF> constraint;
};

struct RotationEstimatorOptions {
  // Maximum number of times to run L1 minimization.
  int max_num_l1_iterations = 5;

  // Average step size threshold to terminate the L1 minimization
  double l1_step_convergence_threshold = 0.001;

  // The number of iterative reweighted least squares iterations to perform.
  int max_num_irls_iterations = 100;

  // Average step size threshold to termininate the IRLS minimization
  double irls_step_convergence_threshold = 0.001;

  Eigen::Vector3d axis = Eigen::Vector3d(0, 1, 0);

  // This is the point where the Huber-like cost function switches from L1 to
  // L2.
  double irls_loss_parameter_sigma = 5.0;  // in degree

  enum WeightType {
    // For Geman-McClure weight, refer to the paper "Efficient and robust
    // large-scale rotation averaging" (Chatterjee et. al, 2013)
    GEMAN_MCCLURE,
    // For Half Norm, refer to the paper "Robust Relative Rotation Averaging"
    // (Chatterjee et. al, 2017)
    HALF_NORM,
  } weight_type = GEMAN_MCCLURE;

  // Flg to use maximum spanning tree for initialization
  bool skip_initialization = false;

  // Flag to use weighting for rotation averaging
  bool use_weight = false;

  // Flag to use gravity for rotation averaging
  bool use_gravity = false;
};

class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}

  // Estimates the global orientations of all views based on an initial
  // guess. Returns true on successful estimation and false otherwise.
  // In the gravity aligned case, currently only gravity measurements
  // for the reference sensor are supported
  bool EstimateRotations(const ViewGraph& view_graph,
                         colmap::Reconstruction& reconstruction,
                         const std::vector<colmap::PosePrior>& pose_priors);

 protected:
  // Initialize the rotation from the maximum spanning tree
  // Number of inliers serve as weights
  void InitializeFromMaximumSpanningTree(
      const ViewGraph& view_graph, colmap::Reconstruction& reconstruction);

  // Sets up the sparse linear system such that dR_ij = dR_j - dR_i. This is the
  // first-order approximation of the angle-axis rotations. This should only be
  // called once.
  void SetupLinearSystem(
      const ViewGraph& view_graph,
      const colmap::Reconstruction& reconstruction,
      const std::unordered_map<frame_t, const colmap::PosePrior*>&
          frame_to_pose_prior);

  // Performs the L1 robust loss minimization.
  bool SolveL1Regression(
      const ViewGraph& view_graph,
      const colmap::Reconstruction& reconstruction,
      const std::unordered_map<frame_t, const colmap::PosePrior*>&
          frame_to_pose_prior);

  // Performs the iteratively reweighted least squares.
  bool SolveIRLS(const ViewGraph& view_graph,
                 const colmap::Reconstruction& reconstruction,
                 const std::unordered_map<frame_t, const colmap::PosePrior*>&
                     frame_to_pose_prior);

  // Updates the global rotations based on the current rotation change.
  void UpdateGlobalRotations(
      const ViewGraph& view_graph,
      const colmap::Reconstruction& reconstruction,
      const std::unordered_map<frame_t, const colmap::PosePrior*>&
          frame_to_pose_prior);

  // Computes the relative rotation (tangent space) residuals based on the
  // current global orientation estimates.
  void ComputeResiduals(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, colmap::Image>& images,
      const std::unordered_map<frame_t, const colmap::PosePrior*>&
          frame_to_pose_prior);

  // Computes the average size of the most recent step of the algorithm.
  // The is the average over all non-fixed global_orientations_ of their
  // rotation magnitudes.
  double ComputeAverageStepSize(
      const std::unordered_map<frame_t, colmap::Frame>& frames,
      const std::unordered_map<frame_t, const colmap::PosePrior*>&
          frame_to_pose_prior);

  // Converts the results from the tangent space to the global rotations and
  // updates the frames and images with the new rotations.
  void ConvertResults(
      colmap::Reconstruction& reconstruction,
      const std::unordered_map<frame_t, const colmap::PosePrior*>&
          frame_to_pose_prior);

  // Data
  // Options for the solver.
  const RotationEstimatorOptions& options_;

  // The sparse matrix used to maintain the linear system. This is matrix A in
  // Ax = b.
  Eigen::SparseMatrix<double> sparse_matrix_;

  // x in the linear system Ax = b.
  Eigen::VectorXd tangent_space_step_;

  // b in the linear system Ax = b.
  Eigen::VectorXd tangent_space_residual_;

  Eigen::VectorXd rotation_estimated_;

  // Parameter index mappings into solution vector x.
  // Maps frame ID to rig_from_world rotation params.
  std::unordered_map<frame_t, int> frame_id_to_param_idx_;
  // Maps camera ID to cam_from_rig rotation params (unknown only).
  std::unordered_map<camera_t, int> camera_id_to_param_idx_;

  // Preprocessed constraints for each image pair.
  std::unordered_map<image_pair_t, PairConstraint> pair_constraints_;

  // The fixed frame id. This is used to remove gauge ambiguity.
  frame_t fixed_frame_id_ = colmap::kInvalidFrameId;

  // The fixed frame's rotation (non-identity if initialization was used).
  Eigen::Vector3d fixed_frame_rotation_;

  // The weights for the edges
  Eigen::ArrayXd weights_;
};

}  // namespace glomap
