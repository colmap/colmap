#pragma once

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/util/hash_containers.h"

#include <optional>
#include <variant>

#include <Eigen/Sparse>

namespace colmap {

// Rotation averaging problem formulated as linear system A*x = b where:
//   x = [rig_from_world rotations, unknown cam_from_rig rotations]
//   b = residuals from relative rotation constraints
//   A = sparse matrix encoding constraint equations
class RotationAveragingProblem {
 public:
  // 1-DOF constraint when both frames have gravity priors.
  struct GravityAligned1DOF {
    double angle_cam2_from_cam1;  // Relative Y-axis rotation.
    double xz_error;              // Squared error in x,z axes for IRLS.
  };

  // 3-DOF constraint for general case (no gravity or partial gravity).
  struct Full3DOF {
    Eigen::Matrix3d R_cam2_from_cam1;
  };

  // Preprocessed constraint for an image pair, built once during setup.
  struct PairConstraint {
    image_t image_id1 = kInvalidImageId;
    image_t image_id2 = kInvalidImageId;
    // Starting row in matrix A (1 row for 1-DOF, 3 rows for 3-DOF).
    int row_index = -1;
    // Column indices for unknown cam_from_rig rotations (-1 if known).
    int cam1_from_rig_param_idx = -1;
    int cam2_from_rig_param_idx = -1;
    std::variant<GravityAligned1DOF, Full3DOF> constraint;
  };

  RotationAveragingProblem(const PoseGraph& pose_graph,
                           const std::vector<PosePrior>& pose_priors,
                           const RotationEstimatorOptions& options,
                           const FlatHashSet<image_t>& active_image_ids,
                           Reconstruction& reconstruction);

  // Computes residual vector b from current rotation estimates.
  void ComputeResiduals();

  // Updates rotation estimates by applying the solution step.
  void UpdateState(const Eigen::VectorXd& step);

  // Returns average rotation step size for convergence checking.
  double AverageStepSize(const Eigen::VectorXd& step) const;

  // Writes optimized rotations back to reconstruction.
  void ApplyResultsToReconstruction(Reconstruction& reconstruction);

  const Eigen::SparseMatrix<double>& ConstraintMatrix() const {
    return constraint_matrix_;
  }
  const Eigen::VectorXd& Residuals() const { return residuals_; }
  // Diagonal of the residual-space reweighting operator W (one weight per
  // residual row), or nullopt when no reweighting is configured.
  const std::optional<Eigen::VectorXd>& ResidualReweighting() const {
    return residual_reweighting_;
  }
  // Constraint matrix A with the reweighting operator applied to its rows
  // (W * A), or the plain constraint matrix when no reweighting is configured.
  Eigen::SparseMatrix<double> WeightedConstraintMatrix() const;
  // Residual vector b with the reweighting operator applied (W * b), or the
  // plain residuals when no reweighting is configured.
  Eigen::VectorXd WeightedResiduals() const;
  int NumParameters() const { return constraint_matrix_.cols(); }
  int NumResiduals() const { return constraint_matrix_.rows(); }
  int NumGaugeFixingResiduals() const { return num_gauge_fixing_residuals_; }
  const NodeHashMap<image_pair_t, PairConstraint>& PairConstraints() const {
    return pair_constraints_;
  }

 private:
  // Returns true if frame has gravity prior and gravity mode is enabled.
  bool HasFrameGravity(frame_t frame_id) const;

  // Allocates parameter indices for frames and cameras, initializes rotations.
  size_t AllocateParameters(const Reconstruction& reconstruction);

  // Builds PairConstraint for each valid image pair.
  void BuildPairConstraints(const PoseGraph& pose_graph,
                            const Reconstruction& reconstruction);

  // Builds sparse matrix A and the residual-space reweighting operator W.
  void BuildConstraintMatrix(size_t num_params,
                             const PoseGraph& pose_graph,
                             const Reconstruction& reconstruction);

  const RotationEstimatorOptions options_;

  // Pose priors indexed by frame ID.
  NodeHashMap<frame_t, const PosePrior*> frame_to_pose_prior_;

  // Linear system components.
  Eigen::SparseMatrix<double> constraint_matrix_;  // Matrix A.
  Eigen::VectorXd residuals_;                      // Vector b.

  // Optional reweighting operator W applied to the residual space (rows of A
  // and b); the solver works on the reweighted system min ||W (A x - b)||.
  // Populated when the reweighting scheme is not UNIFORM. Stored as the
  // diagonal of W (one weight per residual row).
  std::optional<Eigen::VectorXd> residual_reweighting_;

  // Current rotation estimates in tangent space (angle-axis).
  Eigen::VectorXd estimated_rotations_;

  // Parameter index mappings.
  NodeHashMap<frame_t, int> frame_id_to_param_idx_;
  NodeHashMap<camera_t, int> camera_id_to_param_idx_;

  // Preprocessed constraints for each image pair.
  NodeHashMap<image_pair_t, PairConstraint> pair_constraints_;

  // Gauge fixing (removes rotational ambiguity).
  frame_t fixed_frame_id_ = kInvalidFrameId;
  Eigen::Vector3d fixed_frame_rotation_;
  int num_gauge_fixing_residuals_ = 3;  // 1 for gravity-aligned, 3 otherwise.

  // Cached lookups for ComputeResiduals and UpdateState.
  NodeHashMap<image_t, frame_t> image_id_to_frame_id_;
  NodeHashMap<camera_t, rig_t> camera_id_to_rig_id_;
  NodeHashMap<camera_t, std::vector<frame_t>> camera_to_frame_ids_;

  // Active frames for the current solve.
  FlatHashSet<frame_t> active_frame_ids_;
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
  std::optional<Eigen::VectorXd> ComputeIRLSWeights(
      const RotationAveragingProblem& problem, double sigma) const;

  const RotationEstimatorOptions options_;
};

}  // namespace colmap
