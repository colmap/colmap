#pragma once

#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_triangulator.h"

#include "glomap/estimators/bundle_adjustment.h"
#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/rotation_averaging.h"
#include "glomap/estimators/view_graph_calibration.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/scene/view_graph.h"
#include "glomap/sfm/track_establishment.h"

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
  RotationEstimatorOptions rotation_averaging;
  TrackEstablishmentOptions track_establishment;
  GlobalPositionerOptions global_positioning;
  BundleAdjusterOptions bundle_adjustment;
  colmap::IncrementalTriangulator::Options retriangulation = [] {
    colmap::IncrementalTriangulator::Options opts;
    opts.complete_max_reproj_error = 15.0;
    opts.merge_max_reproj_error = 15.0;
    opts.min_angle = 1.0;
    return opts;
  }();

  // Inlier thresholds for each component
  InlierThresholdOptions inlier_thresholds;

  // Control the number of iterations for bundle adjustment.
  int num_iterations_ba = 3;

  // Control the flow of the global sfm
  bool skip_view_graph_calibration = false;
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_global_positioning = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = false;
  bool skip_pruning = true;
};

class GlobalMapper {
 public:
  explicit GlobalMapper(std::shared_ptr<const colmap::Database> database);

  // Prepare the mapper for a new reconstruction. This will initialize the
  // reconstruction and view graph from the database.
  void BeginReconstruction(
      const std::shared_ptr<colmap::Reconstruction>& reconstruction);

  // Run the global SfM pipeline.
  bool Solve(const GlobalMapperOptions& options,
             std::unordered_map<frame_t, int>& cluster_ids);

  // Run rotation averaging to estimate global rotations.
  bool RotationAveraging(const RotationEstimatorOptions& options,
                         double max_rotation_error);

  // Establish tracks from feature matches.
  void EstablishTracks(const TrackEstablishmentOptions& options);

  // Estimate global camera positions.
  bool GlobalPositioning(const GlobalPositionerOptions& options,
                         double max_angle_error,
                         double max_reprojection_error,
                         double min_triangulation_angle);

  // Run iterative bundle adjustment to refine poses and structure.
  bool IterativeBundleAdjustment(const BundleAdjusterOptions& options,
                                 double max_reprojection_error,
                                 double min_triangulation_angle,
                                 int num_iterations);

  // Iteratively retriangulate tracks and refine to improve structure.
  bool IterativeRetriangulateAndRefine(
      const colmap::IncrementalTriangulator::Options& options,
      const BundleAdjusterOptions& ba_options,
      double max_reprojection_error,
      double min_triangulation_angle);

  // Getter functions.
  std::shared_ptr<colmap::Reconstruction> Reconstruction() const;
  std::shared_ptr<class ViewGraph> ViewGraph() const;

 private:
  // Class that provides access to the database.
  const std::shared_ptr<const colmap::Database> database_;

  // Class that caches data loaded from the database.
  std::shared_ptr<const colmap::DatabaseCache> database_cache_;

  // Class that holds data of the reconstruction.
  std::shared_ptr<colmap::Reconstruction> reconstruction_;

  // Class that holds the view graph.
  std::shared_ptr<class ViewGraph> view_graph_;
};

}  // namespace glomap
