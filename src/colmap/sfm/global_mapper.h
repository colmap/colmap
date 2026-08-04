#pragma once

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/global_positioning.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_triangulator.h"
#include "colmap/util/enum_utils.h"

#include <filesystem>
#include <functional>
#include <limits>

namespace colmap {

// off: existing visual rotation averaging only.
// initialize: run a robust consensus gauge rotation from full-orientation
//   pose priors before global positioning (see GlobalMapper::Solve).
// Lowercase enumerators so MAKE_ENUM_CLASS's generated ToString/FromString

// off: no gravity residual in bundle adjustment (the existing hard
//   ra_use_gravity 1-DoF rotation-averaging reduction, if enabled, is
//   unaffected by this setting -- it is a separate mechanism).
// optimize: add a soft, yaw-free gravity residual to every BA stage that
//   runs once pose_prior_position_mode == optimize has engaged (there is no
//   other point in the solve where "down" has a known physical direction to
//   compare against). No "initialize" state: unlike position gauging or the
//   one-time rotation-gauge snap (pose_prior_rotation_mode), a BA residual
//   is inherently a continuous-optimization concept only.
MAKE_ENUM_CLASS(PosePriorGravityBAMode, 0, off, optimize);

struct GlobalMapperOptions {
  // Number of threads.
  int num_threads = -1;

  // PRNG seed for all stochastic methods during reconstruction.
  // If -1 (default), the seed is derived from the current time
  // (non-deterministic). If >= 0, the pipeline is deterministic with the given
  // seed.
  int random_seed = -1;

  // The image path at which to find the images to extract point colors.
  // If not specified, all point colors will be black.
  std::filesystem::path image_path;

  // When false, treat each non-ref sensor's cam_from_rig as a pre-calibrated.
  bool refine_sensor_from_rig = true;

  // Whether/how to initialize the global rotation gauge from full-orientation

  // Options for each component.
  RotationEstimatorOptions rotation_averaging;
  GlobalPositionerOptions global_positioning;
  BundleAdjustmentOptions bundle_adjustment = [] {
    BundleAdjustmentOptions options;
    options.min_track_length = 3;
    options.print_summary = false;
    if (options.ceres) {
      options.ceres->loss_function_type =
          CeresBundleAdjustmentOptions::LossFunctionType::HUBER;
      options.ceres->use_gpu = true;
      // TODO: Investigate whether disabling auto solver selection and using
      // explicit SPARSE_SCHUR is necessary for global SfM, or if we can just
      // rely on COLMAP's auto selection.
      options.ceres->auto_select_solver_type = false;
      options.ceres->solver_options.function_tolerance = 1e-5;
      options.ceres->solver_options.max_num_iterations = 200;
      options.ceres->solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    return options;
  }();
  IncrementalTriangulator::Options retriangulation = [] {
    IncrementalTriangulator::Options opts;
    opts.complete_max_reproj_error = 15.0;
    opts.merge_max_reproj_error = 15.0;
    opts.min_angle = 1.0;
    return opts;
  }();

  // Track establishment options.
  // Max pixel distance between observations of the same track within one image.
  double track_intra_image_consistency_threshold = 10.;
  // Required number of tracks per view before early stopping.
  int track_required_tracks_per_view = std::numeric_limits<int>::max();
  // Minimum number of views per track.
  int track_min_num_views_per_track = 3;
  // Maximum total number of tracks to establish. Tracks are selected in order
  // of decreasing length, so the longest tracks are kept. Use this to bound
  // memory usage on large datasets. By default, there is no limit.
  int keep_max_num_tracks = std::numeric_limits<int>::max();

  // Thresholds for each component.
  double max_angular_reproj_error_deg = 1.;   // for global positioning
  double max_normalized_reproj_error = 1e-2;  // for bundle adjustment
  double min_tri_angle_deg = 1.;              // for triangulation

  // GPU device index for bundle adjustment, shared by the Ceres and Caspar
  // backends (-1 = auto-select).
  std::string ba_gpu_index = "-1";

  // Control the number of iterations for bundle adjustment.
  int ba_num_iterations = 3;

  // Whether to use a robust loss on the prior-position residual for
  // GlobalMapper's own bundle-adjustment stages (iterative BA and the tail
  // of retriangulation) when pose_prior_position_mode == optimize is
  // engaged. Independent of gp_*, which governs the one-shot global
  // positioning stage only.
  //
  // Defaults to true: in optimize mode, every valid registered GPS prior is
  // offered to these later BA stages (not just the initial RANSAC inlier
  // set used to establish the metric gauge -- see AlignReconstruction), so
  // any prior rejected by that initial RANSAC pass must not be reintroduced
  // with a TRIVIAL (unrobustified) loss, or a single bad fix can pull the
  // whole solve. This is a behavior change from the previous false default;
  // there is no safe reason to run optimize mode without it.
  bool ba_use_robust_loss_on_prior_position = true;

  // Threshold on the *standardized* (covariance-whitened) residual norm for
  // that robust loss, in "number of standard deviations" -- NOT chi-square
  // units and NOT degrees/radians/metres. Ceres scales a robust loss as
  // rho(s, a) = a^2 * rho(s / a^2), where `a` is in residual-norm units and
  // `s` is squared residual norm, so a 95%-confidence-radius threshold for a
  // 3-DOF whitened residual is sqrt(chi-square_3dof_95), not the raw
  // chi-square quantile itself. Mirrors bundle_adjustment_ceres.h's
  // prior_position_loss_scale default.
  double ba_prior_position_loss_scale = std::sqrt(kChiSquare95ThreeDof);

  // Whether/how to add a soft, yaw-free gravity residual to GlobalMapper's
  // own bundle-adjustment stages. See PosePriorGravityBAMode. Off by
  // default -- preserves existing behavior byte-for-byte until explicitly
  // requested; the existing hard ra_use_gravity rotation-averaging
  // reduction remains available unchanged as the legacy mechanism.
  PosePriorGravityBAMode pose_prior_gravity_ba_mode =
      PosePriorGravityBAMode::off;

  // Global sensor-class angular uncertainty for the gravity residual, in
  // degrees. PosePrior has no per-row gravity covariance field (deferred;
  // see doc/pose_priors.rst), so this single scalar is applied uniformly.
  double pose_prior_gravity_stddev_deg = 5.0;

  // Threshold on the *standardized* (covariance-whitened) residual norm for
  // the gravity robust loss, in "number of standard deviations" -- NOT
  // radians/degrees, even though the gravity covariance itself is built from
  // pose_prior_gravity_stddev_deg. CovarianceWeightedCostFunctor whitens the
  // raw angular residual by dividing by sigma_rad before Ceres ever sees it,
  // so the robust loss operates on a dimensionless standardized residual;
  // the correct 95%-confidence-radius threshold for a 2-DOF whitened
  // residual is sqrt(chi-square_2dof_95), not the raw chi-square quantile
  // (see AbsoluteGravityPriorCostFunctor's local-rank-2 comment in
  // pose_prior.h for why 2 DOF applies to a 3-component chordal residual).
  // The loss function itself defaults to Cauchy (see
  // CeresPosePriorBundleAdjustmentOptions::prior_gravity_loss_function_type)
  // -- unlike the position prior, gravity's robust loss is not separately
  // gated by a use_robust_loss_on_prior_gravity flag, since a single
  // corrupted gravity reading is a known real failure mode for this sensor
  // class and robustness is wanted by default whenever gravity mode is on.
  double pose_prior_gravity_loss_scale = std::sqrt(kChiSquare95TwoDof);

  // Whether to skip the fixed-rotation stage in bundle adjustment.
  // By default, BA runs in two stages: first with fixed rotations (position
  // only), then with full optimization. Setting this to true skips the first
  // stage and runs full optimization directly.
  bool ba_skip_fixed_rotation_stage = false;

  // Whether to skip the joint optimization stage in bundle adjustment.
  // When set to true, only the fixed-rotation stage is run (optimizing
  // positions only). This is mutually exclusive with
  // ba_skip_fixed_rotation_stage.
  bool ba_skip_joint_optimization_stage = false;

  // Control the flow of the global sfm
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_global_positioning = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = false;

  RotationEstimatorOptions RotationAveraging() const;
  GlobalPositionerOptions GlobalPositioning() const;
  BundleAdjustmentOptions BundleAdjustment() const;
  IncrementalTriangulator::Options Retriangulation() const;
};

class GlobalMapper {
 public:
  explicit GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache);

  // Prepare the mapper for a new reconstruction. This will initialize the
  // reconstruction and view graph from the database.
  void BeginReconstruction(
      const std::shared_ptr<Reconstruction>& reconstruction);

  // Run the global SfM pipeline. The optional `on_progress` callback is invoked
  // after global positioning, after each bundle-adjustment iteration, and after
  // retriangulation/refinement; it returns true if a stop has been requested,
  // in which case the pipeline terminates early and keeps the current result.
  bool Solve(const GlobalMapperOptions& options,
             const std::function<bool()>& on_progress = {});

  // Run rotation averaging to estimate global rotations.
  bool RotationAveraging(const RotationEstimatorOptions& options);

  // Selects one global rotation gauge from full-orientation pose priors via
  // a small deterministic consensus (not a new rotation-averaging
  // framework) and applies it once to the whole reconstruction. Run after
  // RotationAveraging() and before GlobalPositioning(). Always returns true;
  // "no consensus" is requested-but-not-engaged, not a hard failure.

  // Establish tracks from feature matches.
  void EstablishTracks(const GlobalMapperOptions& options);

  // Estimate global camera positions.
  bool GlobalPositioning(const GlobalPositionerOptions& options,
                         double max_angular_reproj_error_deg,
                         double max_normalized_reproj_error,
                         double min_tri_angle_deg);

  // Run iterative bundle adjustment to refine poses and structure. The optional
  // `on_progress` callback is invoked after each iteration and returns true if
  // a stop has been requested, in which case the iteration terminates early.
  bool IterativeBundleAdjustment(const BundleAdjustmentOptions& options,
                                 double max_normalized_reproj_error,
                                 double min_tri_angle_deg,
                                 int num_iterations,
                                 bool skip_fixed_rotation_stage = false,
                                 bool skip_joint_optimization_stage = false,
                                 const std::function<bool()>& on_progress = {});

  // Iteratively retriangulate tracks and refine to improve structure.
  bool IterativeRetriangulateAndRefine(
      const IncrementalTriangulator::Options& options,
      const BundleAdjustmentOptions& ba_options,
      double max_normalized_reproj_error,
      double min_tri_angle_deg);

  // Getter functions.
  std::shared_ptr<class Reconstruction> Reconstruction() const;

 private:
  // Runs bundle adjustment on the current reconstruction. Uses
  // CreatePosePriorBundleAdjuster (instead of the default gauge-fixing
  // adjuster) whenever an engaged `optimize` pose-prior position solve has
  // already established the metric gauge -- see pose_prior_position_engaged_
  // -- so GPS position-prior residuals stay active through every BA stage,
  // not just the one-shot GlobalPositioner. Otherwise behaves exactly like
  // the previous free-function CreateDefaultBundleAdjuster path.
  bool RunBundleAdjustment(const BundleAdjustmentOptions& options);

  std::shared_ptr<const DatabaseCache> database_cache_;
  std::shared_ptr<class PoseGraph> pose_graph_;
  std::shared_ptr<class Reconstruction> reconstruction_;

  // Set by GlobalPositioning() when pose_prior_position_mode == optimize and
  // the robust prior-constrained solve actually engaged. Ordinary post-
  // positioning/BA/retriangulation normalization is gauge-destroying, so it
  // is skipped while this is true (it would silently discard the metric
  // gauge the prior optimization just established).
  bool pose_prior_position_engaged_ = false;

  // Cached scalar knobs for RunBundleAdjustment()'s pose-prior branch.
  // pose_prior_ba_use_robust_loss_/pose_prior_ba_loss_scale_ are read once
  // from GlobalMapperOptions at the top of Solve(); pose_prior_position_
  // fallback_stddev_ is read from GlobalPositionerOptions inside
  // GlobalPositioning() (so it stays correct even if a caller invokes the
  // BA stages directly after GlobalPositioning(), without going through
  // Solve()).
  bool pose_prior_ba_use_robust_loss_ = true;
  double pose_prior_ba_loss_scale_ = std::sqrt(kChiSquare95ThreeDof);
  double pose_prior_position_fallback_stddev_ = 1.0;

  // Cached from GlobalMapperOptions at the top of Solve(); used by
  // RunBundleAdjustment()'s pose-prior branch, combined there with
  // pose_prior_position_engaged_ (gravity mode only takes effect once
  // position priors have established the metric/ENU gauge).
  bool pose_prior_gravity_requested_ = false;
  double pose_prior_gravity_stddev_deg_ = 5.0;
  double pose_prior_gravity_loss_scale_ = std::sqrt(kChiSquare95TwoDof);
};

}  // namespace colmap
