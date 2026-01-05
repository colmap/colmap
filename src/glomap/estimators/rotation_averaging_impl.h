#pragma once

#include "glomap/estimators/rotation_averaging.h"

#include <optional>
#include <variant>

#include <Eigen/Sparse>

namespace glomap {

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
    image_t image_id1 = colmap::kInvalidImageId;
    image_t image_id2 = colmap::kInvalidImageId;
    // Starting row in matrix A (1 row for 1-DOF, 3 rows for 3-DOF).
    int row_index = -1;
    // Column indices for unknown cam_from_rig rotations (-1 if known).
    int cam1_from_rig_param_idx = -1;
    int cam2_from_rig_param_idx = -1;
    std::variant<GravityAligned1DOF, Full3DOF> constraint;
  };

  RotationAveragingProblem(const ViewGraph& view_graph,
                           const std::vector<colmap::PosePrior>& pose_priors,
                           const RotationEstimatorOptions& options,
                           const std::unordered_set<image_t>& active_image_ids,
                           colmap::Reconstruction& reconstruction);

  // Computes residual vector b from current rotation estimates.
  void ComputeResiduals();

  // Updates rotation estimates by applying the solution step.
  void UpdateState(const Eigen::VectorXd& step);

  // Returns average rotation step size for convergence checking.
  double AverageStepSize(const Eigen::VectorXd& step) const;

  // Writes optimized rotations back to reconstruction.
  void ApplyResultsToReconstruction(colmap::Reconstruction& reconstruction);

  const Eigen::SparseMatrix<double>& ConstraintMatrix() const {
    return constraint_matrix_;
  }
  const Eigen::VectorXd& Residuals() const { return residuals_; }
  int NumParameters() const { return constraint_matrix_.cols(); }
  int NumResiduals() const { return constraint_matrix_.rows(); }
  int NumGaugeFixingResiduals() const { return num_gauge_fixing_residuals_; }
  const std::unordered_map<image_pair_t, PairConstraint>& PairConstraints()
      const {
    return pair_constraints_;
  }

 private:
  // Returns true if frame has gravity prior and gravity mode is enabled.
  bool HasFrameGravity(frame_t frame_id) const;

  // Allocates parameter indices for frames and cameras, initializes rotations.
  size_t AllocateParameters(const colmap::Reconstruction& reconstruction);

  // Builds PairConstraint for each valid image pair.
  void BuildPairConstraints(const ViewGraph& view_graph,
                            const colmap::Reconstruction& reconstruction);

  // Builds sparse matrix A and edge weight vector.
  void BuildConstraintMatrix(size_t num_params,
                             const ViewGraph& view_graph,
                             const colmap::Reconstruction& reconstruction);

  const RotationEstimatorOptions options_;

  // Pose priors indexed by frame ID.
  std::unordered_map<frame_t, const colmap::PosePrior*> frame_to_pose_prior_;

  // Linear system components.
  Eigen::SparseMatrix<double> constraint_matrix_;  // Matrix A.
  Eigen::VectorXd residuals_;                      // Vector b.

  // Current rotation estimates in tangent space (angle-axis).
  Eigen::VectorXd estimated_rotations_;

  // Parameter index mappings.
  std::unordered_map<frame_t, int> frame_id_to_param_idx_;
  std::unordered_map<camera_t, int> camera_id_to_param_idx_;

  // Preprocessed constraints for each image pair.
  std::unordered_map<image_pair_t, PairConstraint> pair_constraints_;

  // Gauge fixing (removes rotational ambiguity).
  frame_t fixed_frame_id_ = colmap::kInvalidFrameId;
  Eigen::Vector3d fixed_frame_rotation_;
  int num_gauge_fixing_residuals_ = 3;  // 1 for gravity-aligned, 3 otherwise.

  // Cached lookups for ComputeResiduals and UpdateState.
  std::unordered_map<image_t, frame_t> image_id_to_frame_id_;
  std::unordered_map<camera_t, rig_t> camera_id_to_rig_id_;
  std::unordered_map<camera_t, std::vector<frame_t>> camera_to_frame_ids_;

  // Active frames for the current solve.
  std::unordered_set<frame_t> active_frame_ids_;
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

}  // namespace glomap
