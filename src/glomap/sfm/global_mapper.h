#pragma once

#include "colmap/scene/database.h"

#include "glomap/estimators/bundle_adjustment.h"
#include "glomap/estimators/global_positioning.h"
#include "glomap/estimators/global_rotation_averaging.h"
#include "glomap/estimators/relpose_estimation.h"
#include "glomap/estimators/view_graph_calibration.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/sfm/track_establishment.h"
#include "glomap/sfm/track_retriangulation.h"

namespace glomap {

struct GlobalMapperOptions {
  // Options for each component
  ViewGraphCalibratorOptions opt_vgcalib;
  RelativePoseEstimationOptions opt_relpose;
  RotationEstimatorOptions opt_ra;
  TrackEstablishmentOptions opt_track;
  GlobalPositionerOptions opt_gp;
  BundleAdjusterOptions opt_ba;
  TriangulatorOptions opt_triangulator;

  // Inlier thresholds for each component
  InlierThresholdOptions inlier_thresholds;

  // Control the number of iterations for each component
  int num_iteration_bundle_adjustment = 3;
  int num_iteration_retriangulation = 1;

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

  bool Solve(const colmap::Database& database,
             ViewGraph& view_graph,
             std::unordered_map<rig_t, Rig>& rigs,
             std::unordered_map<camera_t, colmap::Camera>& cameras,
             std::unordered_map<frame_t, Frame>& frames,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<point3D_t, Point3D>& tracks,
             std::vector<colmap::PosePrior>& pose_priors);

 private:
  const GlobalMapperOptions options_;
};

}  // namespace glomap
