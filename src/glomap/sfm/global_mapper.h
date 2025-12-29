#pragma once

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"

#include "glomap/estimators/bundle_adjustment.h"
#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/relpose_estimation.h"
#include "glomap/estimators/rotation_averaging.h"
#include "glomap/estimators/view_graph_calibration.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/scene/view_graph.h"
#include "glomap/sfm/track_establishment.h"
#include "glomap/sfm/track_retriangulation.h"

namespace glomap {

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
  std::string image_path;

  // Options for each component
  ViewGraphCalibratorOptions view_graph_calibration;
  RelativePoseEstimationOptions relative_pose_estimation;
  RotationEstimatorOptions rotation_averaging;
  TrackEstablishmentOptions track_establishment;
  GlobalPositionerOptions global_positioning;
  BundleAdjusterOptions bundle_adjustment;
  TriangulatorOptions retriangulation;

  // Inlier thresholds for each component
  InlierThresholdOptions inlier_thresholds;

  // Control the number of iterations for each component
  int num_iterations_ba = 3;
  int num_iterations_retriangulation = 1;

  // Control the flow of the global sfm
  bool skip_preprocessing = false;
  bool skip_view_graph_calibration = false;
  bool skip_relative_pose_estimation = false;
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_global_positioning = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = false;
  bool skip_pruning = true;
};

// TODO: Refactor the code to reuse the pipeline code more
class GlobalMapper {
 public:
  explicit GlobalMapper(const GlobalMapperOptions& options)
      : options_(options) {}

  // database can be nullptr if skip_retriangulation is true
  bool Solve(const colmap::Database* database,
             ViewGraph& view_graph,
             colmap::Reconstruction& reconstruction,
             std::vector<colmap::PosePrior>& pose_priors,
             std::unordered_map<frame_t, int>& cluster_ids);

 private:
  const GlobalMapperOptions options_;
};

}  // namespace glomap
