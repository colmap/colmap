#pragma once

#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_triangulator.h"

#include "glomap/estimators/bundle_adjustment.h"
#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/rotation_averaging.h"
#include "glomap/scene/pose_graph.h"

#include <filesystem>
#include <limits>

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
  std::filesystem::path image_path;

  // Options for each component
  RotationEstimatorOptions rotation_averaging;
  GlobalPositionerOptions global_positioning;
  BundleAdjusterOptions bundle_adjustment;
  colmap::IncrementalTriangulator::Options retriangulation = [] {
    colmap::IncrementalTriangulator::Options opts;
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

  // Thresholds for each component.
  double max_angular_reproj_error_deg = 1.;   // for global positioning
  double max_normalized_reproj_error = 1e-2;  // for bundle adjustment
  double min_tri_angle_deg = 1.;              // for triangulation

  // Control the number of iterations for bundle adjustment.
  int ba_num_iterations = 3;

  // Control the flow of the global sfm
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_global_positioning = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = false;
};

class GlobalMapper {
 public:
  explicit GlobalMapper(
      std::shared_ptr<const colmap::DatabaseCache> database_cache);

  // Prepare the mapper for a new reconstruction. This will initialize the
  // reconstruction and view graph from the database.
  void BeginReconstruction(
      const std::shared_ptr<colmap::Reconstruction>& reconstruction);

  // Run the global SfM pipeline.
  bool Solve(const GlobalMapperOptions& options,
             std::unordered_map<frame_t, int>& cluster_ids);

  // Run rotation averaging to estimate global rotations.
  bool RotationAveraging(const RotationEstimatorOptions& options);

  // Establish tracks from feature matches.
  void EstablishTracks(const GlobalMapperOptions& options);

  // Estimate global camera positions.
  bool GlobalPositioning(const GlobalPositionerOptions& options,
                         double max_angular_reproj_error_deg,
                         double max_normalized_reproj_error,
                         double min_tri_angle_deg);

  // Run iterative bundle adjustment to refine poses and structure.
  bool IterativeBundleAdjustment(const BundleAdjusterOptions& options,
                                 double max_normalized_reproj_error,
                                 double min_tri_angle_deg,
                                 int num_iterations);

  // Iteratively retriangulate tracks and refine to improve structure.
  bool IterativeRetriangulateAndRefine(
      const colmap::IncrementalTriangulator::Options& options,
      const BundleAdjusterOptions& ba_options,
      double max_normalized_reproj_error,
      double min_tri_angle_deg);

  // Getter functions.
  std::shared_ptr<colmap::Reconstruction> Reconstruction() const;

 private:
  std::shared_ptr<const colmap::DatabaseCache> database_cache_;
  std::shared_ptr<class PoseGraph> pose_graph_;
  std::shared_ptr<colmap::Reconstruction> reconstruction_;
};

}  // namespace glomap
