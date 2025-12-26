#pragma once

#include "glomap/estimators/rotation_averaging.h"

#include <optional>
#include <variant>

#include <Eigen/Sparse>

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
// Built once during problem setup, used during L1/IRLS solving.
// The constraint geometry is fixed; only residuals change during optimization.
struct PairConstraint {
  // 1-DOF constraint for pairs where both frames have gravity priors.
  // Gravity alignment reduces full 3D rotation to Y-axis rotation only.
  struct GravityAligned1DOF {
    double angle_cam2_from_cam1;  // Relative Y-axis rotation between frames
    double xz_error;  // Squared error in x,z axes (for IRLS weighting)
  };

  // 3-DOF constraint for the general case (no gravity or partial gravity).
  struct Full3DOF {
    Eigen::Matrix3d
        R_cam2_from_cam1;  // Relative rotation (gravity-aligned if applicable)
  };

  // Image IDs for this pair (cached from view graph).
  image_t image_id1 = colmap::kInvalidImageId;
  image_t image_id2 = colmap::kInvalidImageId;

  // Starting row in sparse matrix A where this pair's equations begin.
  // - GravityAligned1DOF: occupies 1 row
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

// Holds the rotation averaging problem data and provides methods to update
// the solution state. The problem is built from a view graph and reconstruction
// and maintains the sparse linear system A*x = b.
class RotationAveragingProblem {
 public:
  RotationAveragingProblem(const ViewGraph& view_graph,
                           colmap::Reconstruction& reconstruction,
                           const std::vector<colmap::PosePrior>& pose_priors,
                           const RotationEstimatorOptions& options);

  // Computes the residual vector b from current rotation estimates.
  void ComputeResiduals();

  // Updates internal rotation estimates by applying the solution step.
  void UpdateState(const Eigen::VectorXd& step);

  // Computes the average rotation step size for convergence checking.
  double AverageStepSize(const Eigen::VectorXd& step) const;

  // Writes the optimized rotations back to the reconstruction.
  void ApplyResultsToReconstruction(colmap::Reconstruction& reconstruction);

  // Read-only access to the linear system for the solver.
  const Eigen::SparseMatrix<double>& ConstraintMatrix() const {
    return constraint_matrix_;
  }
  const Eigen::VectorXd& Residuals() const { return residuals_; }
  const Eigen::ArrayXd& EdgeWeights() const { return edge_weights_; }
  int NumParameters() const { return constraint_matrix_.cols(); }
  int NumResiduals() const { return constraint_matrix_.rows(); }
  int GaugeFixingRows() const { return gauge_fixing_rows_; }
  const std::unordered_map<image_pair_t, PairConstraint>& PairConstraints()
      const {
    return pair_constraints_;
  }

 private:
  // Allocates parameter indices for frames and cameras, initializes rotations.
  int AllocateParameters(const colmap::Reconstruction& reconstruction);

  // Builds PairConstraint objects for each valid image pair.
  void BuildPairConstraints(const ViewGraph& view_graph,
                            const colmap::Reconstruction& reconstruction);

  // Builds the sparse matrix A and weight vector.
  void BuildConstraintMatrix(int num_params,
                             const ViewGraph& view_graph,
                             const colmap::Reconstruction& reconstruction);

  // Cached options.
  const RotationEstimatorOptions& options_;

  // Processed pose priors indexed by frame ID.
  std::unordered_map<frame_t, const colmap::PosePrior*> frame_to_pose_prior_;

  // The sparse matrix A in the linear system A*x = b.
  Eigen::SparseMatrix<double> constraint_matrix_;

  // Residual vector b in A*x = b.
  Eigen::VectorXd residuals_;

  // Current rotation estimates in tangent space.
  Eigen::VectorXd rotation_estimated_;

  // Maps frame ID to parameter index for rig_from_world rotation.
  std::unordered_map<frame_t, int> frame_id_to_param_idx_;

  // Maps camera ID to parameter index for unknown cam_from_rig rotation.
  std::unordered_map<camera_t, int> camera_id_to_param_idx_;

  // Preprocessed constraints for each image pair.
  std::unordered_map<image_pair_t, PairConstraint> pair_constraints_;

  // Fixed frame for gauge fixing.
  frame_t fixed_frame_id_ = colmap::kInvalidFrameId;
  Eigen::Vector3d fixed_frame_rotation_;
  int gauge_fixing_rows_ = 3;

  // Edge weights for the linear system.
  Eigen::ArrayXd edge_weights_;

  // Cached data for ComputeResiduals.
  // Maps image_id to frame_id.
  std::unordered_map<image_t, frame_t> image_id_to_frame_id_;

  // Cached data for UpdateState cam_from_rig averaging.
  // Maps camera_id to list of frame_ids it appears in.
  std::unordered_map<camera_t, std::vector<frame_t>> camera_to_frame_ids_;
};

// Solves the rotation averaging problem using L1 regression followed by IRLS.
class RotationAveragingSolver {
 public:
  explicit RotationAveragingSolver(const RotationEstimatorOptions& options)
      : options_(options) {}

  // Solves the rotation averaging problem.
  bool Solve(RotationAveragingProblem& problem);

 private:
  // L1 robust loss minimization phase.
  bool SolveL1Regression(RotationAveragingProblem& problem);

  // Iteratively reweighted least squares phase.
  bool SolveIRLS(RotationAveragingProblem& problem);

  // Computes IRLS weights for all constraints.
  // Returns nullopt if any weight is NaN.
  std::optional<Eigen::ArrayXd> ComputeIRLSWeights(
      const RotationAveragingProblem& problem, double sigma) const;

  const RotationEstimatorOptions& options_;
};

}  // namespace glomap
