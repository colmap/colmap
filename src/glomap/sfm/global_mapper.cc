#include "glomap/sfm/global_mapper.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include "glomap/estimators/rotation_averaging.h"
#include "glomap/io/colmap_io.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/processors/reconstruction_pruning.h"
#include "glomap/processors/view_graph_manipulation.h"

namespace glomap {

GlobalMapper::GlobalMapper(std::shared_ptr<const colmap::Database> database)
    : database_(std::move(THROW_CHECK_NOTNULL(database))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<colmap::Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);

  reconstruction_ = reconstruction;
  view_graph_ = std::make_shared<class ViewGraph>();

  InitializeEmptyReconstructionFromDatabase(*database_, *reconstruction_);
  InitializeViewGraphFromDatabase(*database_, *reconstruction_, *view_graph_);
}

std::shared_ptr<colmap::Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

std::shared_ptr<class ViewGraph> GlobalMapper::ViewGraph() const {
  return view_graph_;
}

bool GlobalMapper::ReestimateRelativePoses(
    const RelativePoseEstimationOptions& options,
    const InlierThresholdOptions& inlier_thresholds) {
  // Relative pose relies on the undistorted images
  EstimateRelativePoses(*view_graph_, *reconstruction_, options);

  // Undistort the images and filter edges by inlier number
  ImagePairsInlierCount(
      *view_graph_, *reconstruction_, inlier_thresholds, true);

  view_graph_->FilterByNumInliers(inlier_thresholds.min_inlier_num);
  view_graph_->FilterByInlierRatio(inlier_thresholds.min_inlier_ratio);

  if (view_graph_->KeepLargestConnectedComponents(*reconstruction_) == 0) {
    LOG(ERROR) << "no connected components are found";
    return false;
  }

  return true;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options,
                                     double max_rotation_error) {
  // Read pose priors from the database.
  std::vector<colmap::PosePrior> pose_priors = database_->ReadAllPosePriors();

  // The first run is for filtering
  SolveRotationAveraging(options, *view_graph_, *reconstruction_, pose_priors);

  view_graph_->FilterByRelativeRotation(*reconstruction_, max_rotation_error);
  if (view_graph_->KeepLargestConnectedComponents(*reconstruction_) == 0) {
    LOG(ERROR) << "no connected components are found";
    return false;
  }

  // The second run is for final estimation
  if (!SolveRotationAveraging(
          options, *view_graph_, *reconstruction_, pose_priors)) {
    return false;
  }
  view_graph_->FilterByRelativeRotation(*reconstruction_, max_rotation_error);
  image_t num_img =
      view_graph_->KeepLargestConnectedComponents(*reconstruction_);
  if (num_img == 0) {
    LOG(ERROR) << "no connected components are found";
    return false;
  }
  LOG(INFO) << num_img << " / " << reconstruction_->NumImages()
            << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks(const TrackEstablishmentOptions& options) {
  // TrackEngine reads images, writes unfiltered points3D to a temporary map,
  // then filters into the main reconstruction
  std::unordered_map<point3D_t, Point3D> unfiltered_points3D;
  TrackEngine track_engine(*view_graph_, reconstruction_->Images(), options);
  track_engine.EstablishFullTracks(unfiltered_points3D);

  // Filter the points3D into a selected subset
  std::unordered_map<point3D_t, Point3D> selected_points3D;
  point3D_t num_points3D =
      track_engine.FindTracksForProblem(unfiltered_points3D, selected_points3D);

  // Add selected points3D to reconstruction
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);
  for (auto& [point3D_id, point3D] : selected_points3D) {
    reconstruction_->AddPoint3D(point3D_id, std::move(point3D));
  }
  LOG(INFO) << "Before filtering: " << unfiltered_points3D.size()
            << ", after filtering: " << num_points3D;
}

bool GlobalMapper::GlobalPositioning(const GlobalPositionerOptions& options,
                                     double max_angle_error,
                                     double max_reprojection_error,
                                     double min_triangulation_angle) {
  if (options.constraint_type !=
      GlobalPositionerOptions::ConstraintType::ONLY_POINTS) {
    LOG(ERROR) << "Only points are used for solving camera positions";
    return false;
  }

  GlobalPositioner gp_engine(options);

  // TODO: consider to support other modes as well
  if (!gp_engine.Solve(*view_graph_, *reconstruction_)) {
    return false;
  }

  // Filter tracks based on the estimation
  colmap::ObservationManager obs_manager(*reconstruction_);

  // First pass: use relaxed threshold (2x) for cameras without prior focal.
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      2.0 * max_angle_error,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::ANGULAR);

  // Second pass: apply strict threshold for cameras with prior focal length.
  const double max_angle_error_rad = colmap::DegToRad(max_angle_error);
  std::vector<std::pair<colmap::image_t, colmap::point2D_t>> obs_to_delete;
  for (const auto point3D_id : reconstruction_->Point3DIds()) {
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    const auto& point3D = reconstruction_->Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      const auto& image = reconstruction_->Image(track_el.image_id);
      const auto& camera = *image.CameraPtr();
      if (!camera.has_prior_focal_length) {
        continue;
      }
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const double error = colmap::CalculateAngularReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      if (error > max_angle_error_rad) {
        obs_to_delete.emplace_back(track_el.image_id, track_el.point2D_idx);
      }
    }
  }
  for (const auto& [image_id, point2D_idx] : obs_to_delete) {
    if (reconstruction_->Image(image_id).Point2D(point2D_idx).HasPoint3D()) {
      obs_manager.DeleteObservation(image_id, point2D_idx);
    }
  }

  // Filter tracks based on triangulation angle and reprojection error
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_triangulation_angle, reconstruction_->Point3DIds());
  // Set the threshold to be larger to avoid removing too many tracks
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10 * max_reprojection_error,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);

  // Normalize the structure
  // If the camera rig is used, the structure do not need to be normalized
  reconstruction_->Normalize();

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment(
    const BundleAdjusterOptions& options,
    double max_reprojection_error,
    double min_triangulation_angle,
    int num_iterations) {
  for (int ite = 0; ite < num_iterations; ite++) {
    // First stage: optimize positions only (rotation constant)
    if (!RunBundleAdjustment(options,
                             /*constant_rotation=*/true,
                             *reconstruction_)) {
      return false;
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << ", stage 1 finished (position only)";

    // Second stage: optimize rotations if desired
    if (options.optimize_rotations &&
        !RunBundleAdjustment(options,
                             /*constant_rotation=*/false,
                             *reconstruction_)) {
      return false;
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << ", stage 2 finished";

    // Normalize the structure
    reconstruction_->Normalize();

    // Filter tracks based on the estimation
    // For the filtering, in each round, the criteria for outlier is
    // tightened. If only few tracks are changed, no need to start bundle
    // adjustment right away. Instead, use a more strict criteria to filter
    LOG(INFO) << "Filtering tracks by reprojection ...";

    colmap::ObservationManager obs_manager(*reconstruction_);
    bool status = true;
    size_t filtered_num = 0;
    while (status && ite < num_iterations) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * max_reprojection_error,
          reconstruction_->Point3DIds(),
          colmap::ReprojectionErrorType::NORMALIZED);

      if (filtered_num > 1e-3 * reconstruction_->NumPoints3D()) {
        status = false;
      } else {
        ite++;
      }
    }
    if (status) {
      LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
      break;
    }
  }

  // Filter tracks based on the estimation
  LOG(INFO) << "Filtering tracks by reprojection ...";
  {
    colmap::ObservationManager obs_manager(*reconstruction_);
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        max_reprojection_error,
        reconstruction_->Point3DIds(),
        colmap::ReprojectionErrorType::NORMALIZED);
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        min_triangulation_angle, reconstruction_->Point3DIds());
  }

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine(
    const colmap::IncrementalTriangulator::Options& options,
    const BundleAdjusterOptions& ba_options,
    double max_reprojection_error,
    double min_triangulation_angle) {
  // Create database cache for retriangulation.
  constexpr int kMinNumMatches = 15;
  auto database_cache =
      colmap::DatabaseCache::Create(*database_,
                                    kMinNumMatches,
                                    /*ignore_watermarks=*/false,
                                    /*image_names=*/{});

  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();
  reconstruction_->TranscribeImageIdsToDatabase(*database_);

  // Initialize mapper.
  colmap::IncrementalMapper mapper(database_cache);
  mapper.BeginReconstruction(reconstruction_);

  // Triangulate all registered images.
  for (const auto image_id : reconstruction_->RegImageIds()) {
    mapper.TriangulateImage(options, image_id);
  }

  // Set up bundle adjustment options for colmap's incremental mapper.
  colmap::BundleAdjustmentOptions colmap_ba_options;
  colmap_ba_options.solver_options.num_threads =
      ba_options.solver_options.num_threads;
  colmap_ba_options.solver_options.max_num_iterations = 50;
  colmap_ba_options.solver_options.max_linear_solver_iterations = 100;
  colmap_ba_options.print_summary = false;

  // Iterative global refinement.
  colmap::IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options.random_seed;
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   colmap_ba_options,
                                   options,
                                   /*normalize_reconstruction=*/true);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  colmap::ObservationManager obs_manager(*reconstruction_);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_reprojection_error,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);

  if (!RunBundleAdjustment(ba_options,
                           /*constant_rotation=*/false,
                           *reconstruction_)) {
    return false;
  }

  reconstruction_->Normalize();

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_reprojection_error,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_triangulation_angle, reconstruction_->Point3DIds());

  return true;
}

// TODO: Rig normalizaiton has not be done
bool GlobalMapper::Solve(const GlobalMapperOptions& options,
                         std::unordered_map<frame_t, int>& cluster_ids) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(view_graph_);

  // Propagate random seed and num_threads to component options.
  GlobalMapperOptions opts = options;
  if (opts.random_seed >= 0) {
    opts.relative_pose_estimation.random_seed = opts.random_seed;
    opts.rotation_averaging.random_seed = opts.random_seed;
    opts.global_positioning.random_seed = opts.random_seed;
    opts.global_positioning.use_parameter_block_ordering = false;
    opts.retriangulation.random_seed = opts.random_seed;
  }
  opts.view_graph_calibration.solver_options.num_threads = opts.num_threads;
  opts.relative_pose_estimation.num_threads = opts.num_threads;
  opts.global_positioning.solver_options.num_threads = opts.num_threads;
  opts.bundle_adjustment.solver_options.num_threads = opts.num_threads;

  // 0. Preprocessing
  if (!opts.skip_preprocessing) {
    LOG(INFO) << "----- Running preprocessing -----";
    colmap::Timer run_timer;
    run_timer.Start();
    ViewGraphManipulator::DecomposeRelPose(
        *view_graph_, *reconstruction_, opts.num_threads);
    LOG(INFO) << "Preprocessing done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // 1. Run view graph calibration
  if (!opts.skip_view_graph_calibration) {
    LOG(INFO) << "----- Running view graph calibration -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!CalibrateViewGraph(
            opts.view_graph_calibration, *view_graph_, *reconstruction_)) {
      return false;
    }
    LOG(INFO) << "View graph calibration done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // 2. Run relative pose re-estimation
  if (!opts.skip_relative_pose_estimation) {
    LOG(INFO) << "----- Running relative pose re-estimation -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!ReestimateRelativePoses(opts.relative_pose_estimation,
                                 opts.inlier_thresholds)) {
      return false;
    }
    LOG(INFO) << "Relative pose re-estimation done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // 3. Run rotation averaging
  if (!opts.skip_rotation_averaging) {
    LOG(INFO) << "----- Running rotation averaging -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(opts.rotation_averaging,
                           opts.inlier_thresholds.max_rotation_error)) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // 4. Track establishment and selection
  if (!opts.skip_track_establishment) {
    LOG(INFO) << "----- Running track establishment -----";
    colmap::Timer run_timer;
    run_timer.Start();
    EstablishTracks(opts.track_establishment);
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // 5. Global positioning
  if (!opts.skip_global_positioning) {
    LOG(INFO) << "----- Running global positioning -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning(opts.global_positioning,
                           opts.inlier_thresholds.max_angle_error,
                           opts.inlier_thresholds.max_reprojection_error,
                           opts.inlier_thresholds.min_triangulation_angle)) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // 6. Bundle adjustment
  if (!opts.skip_bundle_adjustment) {
    LOG(INFO) << "----- Running iterative bundle adjustment -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment(
            opts.bundle_adjustment,
            opts.inlier_thresholds.max_reprojection_error,
            opts.inlier_thresholds.min_triangulation_angle,
            opts.num_iterations_ba)) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // 7. Retriangulation
  if (!opts.skip_retriangulation) {
    LOG(INFO) << "----- Running iterative retriangulation and refinement -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine(
            opts.retriangulation,
            opts.bundle_adjustment,
            opts.inlier_thresholds.max_reprojection_error,
            opts.inlier_thresholds.min_triangulation_angle)) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // 8. Reconstruction pruning
  if (!opts.skip_pruning) {
    LOG(INFO) << "----- Running postprocessing -----";
    colmap::Timer run_timer;
    run_timer.Start();
    cluster_ids = PruneWeaklyConnectedFrames(*reconstruction_);
    LOG(INFO) << "Postprocessing done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  return true;
}

}  // namespace glomap
