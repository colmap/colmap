#include "glomap/sfm/global_mapper.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include "glomap/estimators/rotation_averaging.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/processors/reconstruction_pruning.h"
#include "glomap/processors/view_graph_manipulation.h"

namespace glomap {

// TODO: Rig normalizaiton has not be done
bool GlobalMapper::Solve(const colmap::Database* database,
                         ViewGraph& view_graph,
                         colmap::Reconstruction& reconstruction,
                         std::vector<colmap::PosePrior>& pose_priors,
                         std::unordered_map<frame_t, int>& cluster_ids) {
  // Propagate random seed to component options for deterministic behavior.
  GlobalMapperOptions options = options_;
  if (options.random_seed >= 0) {
    options.relative_pose_estimation.random_seed = options.random_seed;
    options.rotation_averaging.random_seed = options.random_seed;
    options.global_positioning.random_seed = options.random_seed;
  }

  // 0. Preprocessing
  if (!options.skip_preprocessing) {
    LOG(INFO) << "----- Running preprocessing -----";

    colmap::Timer run_timer;
    run_timer.Start();
    // If camera intrinsics seem to be good, force the pair to use essential
    // matrix
    ViewGraphManipulator::UpdateImagePairsConfig(view_graph, reconstruction);
    ViewGraphManipulator::DecomposeRelPose(view_graph, reconstruction);
    run_timer.PrintSeconds();
  }

  // 1. Run view graph calibration
  if (!options.skip_view_graph_calibration) {
    LOG(INFO) << "----- Running view graph calibration -----";
    ViewGraphCalibrator vgcalib_engine(options.view_graph_calibration);
    if (!vgcalib_engine.Solve(view_graph, reconstruction)) {
      return false;
    }
  }

  // 2. Run relative pose estimation
  //   TODO: Use generalized relative pose estimation for rigs.
  if (!options.skip_relative_pose_estimation) {
    LOG(INFO) << "----- Running relative pose estimation -----";

    colmap::Timer run_timer;
    run_timer.Start();
    // Relative pose relies on the undistorted images
    EstimateRelativePoses(
        view_graph, reconstruction, options.relative_pose_estimation);

    InlierThresholdOptions inlier_thresholds = options.inlier_thresholds;
    // Undistort the images and filter edges by inlier number
    ImagePairsInlierCount(view_graph, reconstruction, inlier_thresholds, true);

    view_graph.FilterByNumInliers(options.inlier_thresholds.min_inlier_num);
    view_graph.FilterByInlierRatio(options.inlier_thresholds.min_inlier_ratio);

    if (view_graph.KeepLargestConnectedComponents(reconstruction) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    run_timer.PrintSeconds();
  }

  // 3. Run rotation averaging for three times
  if (!options.skip_rotation_averaging) {
    LOG(INFO) << "----- Running rotation averaging -----";

    colmap::Timer run_timer;
    run_timer.Start();

    // The first run is for filtering
    SolveRotationAveraging(
        options.rotation_averaging, view_graph, reconstruction, pose_priors);

    view_graph.FilterByRelativeRotation(
        reconstruction, options.inlier_thresholds.max_rotation_error);
    if (view_graph.KeepLargestConnectedComponents(reconstruction) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    // The second run is for final estimation
    if (!SolveRotationAveraging(options.rotation_averaging,
                                view_graph,
                                reconstruction,
                                pose_priors)) {
      return false;
    }
    view_graph.FilterByRelativeRotation(
        reconstruction, options.inlier_thresholds.max_rotation_error);
    image_t num_img = view_graph.KeepLargestConnectedComponents(reconstruction);
    if (num_img == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }
    LOG(INFO) << num_img << " / " << reconstruction.NumImages()
              << " images are within the connected component.";

    run_timer.PrintSeconds();
  }

  // 4. Track establishment and selection
  if (!options.skip_track_establishment) {
    LOG(INFO) << "----- Running track establishment -----";
    colmap::Timer run_timer;
    run_timer.Start();

    // TrackEngine reads images, writes unfiltered points3D to a temporary map,
    // then filters into the main reconstruction
    std::unordered_map<point3D_t, Point3D> unfiltered_points3D;
    TrackEngine track_engine(
        view_graph, reconstruction.Images(), options.track_establishment);
    track_engine.EstablishFullTracks(unfiltered_points3D);

    // Filter the points3D into a selected subset
    std::unordered_map<point3D_t, Point3D> selected_points3D;
    point3D_t num_points3D = track_engine.FindTracksForProblem(
        unfiltered_points3D, selected_points3D);

    // Add selected points3D to reconstruction
    THROW_CHECK_EQ(reconstruction.NumPoints3D(), 0);
    for (auto& [point3D_id, point3D] : selected_points3D) {
      reconstruction.AddPoint3D(point3D_id, std::move(point3D));
    }
    LOG(INFO) << "Before filtering: " << unfiltered_points3D.size()
              << ", after filtering: " << num_points3D;
    run_timer.PrintSeconds();
  }

  // 5. Global positioning
  if (!options.skip_global_positioning) {
    LOG(INFO) << "----- Running global positioning -----";

    if (options.global_positioning.constraint_type !=
        GlobalPositionerOptions::ConstraintType::ONLY_POINTS) {
      LOG(ERROR) << "Only points are used for solving camera positions";
      return false;
    }

    colmap::Timer run_timer;
    run_timer.Start();

    GlobalPositioner gp_engine(options.global_positioning);

    // TODO: consider to support other modes as well
    if (!gp_engine.Solve(view_graph, reconstruction)) {
      return false;
    }
    // Filter tracks based on the estimation
    colmap::ObservationManager obs_manager(reconstruction);
    // First pass: use relaxed threshold (2x) for cameras without prior focal.
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        2.0 * options.inlier_thresholds.max_angle_error,
        reconstruction.Point3DIds(),
        colmap::ReprojectionErrorType::ANGULAR);
    // Second pass: apply strict threshold for cameras with prior focal length.
    const double max_angle_error_rad =
        colmap::DegToRad(options.inlier_thresholds.max_angle_error);
    std::vector<std::pair<colmap::image_t, colmap::point2D_t>> obs_to_delete;
    for (const auto point3D_id : reconstruction.Point3DIds()) {
      if (!reconstruction.ExistsPoint3D(point3D_id)) {
        continue;
      }
      const auto& point3D = reconstruction.Point3D(point3D_id);
      for (const auto& track_el : point3D.track.Elements()) {
        const auto& image = reconstruction.Image(track_el.image_id);
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
      if (reconstruction.Image(image_id).Point2D(point2D_idx).HasPoint3D()) {
        obs_manager.DeleteObservation(image_id, point2D_idx);
      }
    }

    // Filter tracks based on triangulation angle and reprojection error
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        options.inlier_thresholds.min_triangulation_angle,
        reconstruction.Point3DIds());
    // Set the threshold to be larger to avoid removing too many tracks
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        10 * options.inlier_thresholds.max_reprojection_error,
        reconstruction.Point3DIds(),
        colmap::ReprojectionErrorType::NORMALIZED);
    // Normalize the structure
    // If the camera rig is used, the structure do not need to be normalized
    reconstruction.Normalize();

    run_timer.PrintSeconds();
  }

  // 6. Bundle adjustment
  if (!options.skip_bundle_adjustment) {
    LOG(INFO) << "----- Running bundle adjustment -----";

    colmap::Timer run_timer;
    run_timer.Start();

    for (int ite = 0; ite < options.num_iterations_ba; ite++) {
      // 6.1. First stage: optimize positions only (rotation constant)
      if (!RunBundleAdjustment(options.bundle_adjustment,
                               /*constant_rotation=*/true,
                               reconstruction)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options.num_iterations_ba
                << ", stage 1 finished (position only)";
      run_timer.PrintSeconds();

      // 6.2. Second stage: optimize rotations if desired
      if (options.bundle_adjustment.optimize_rotations &&
          !RunBundleAdjustment(options.bundle_adjustment,
                               /*constant_rotation=*/false,
                               reconstruction)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options.num_iterations_ba << ", stage 2 finished";
      if (ite != options.num_iterations_ba - 1) run_timer.PrintSeconds();

      // Normalize the structure
      reconstruction.Normalize();

      // 6.3. Filter tracks based on the estimation
      // For the filtering, in each round, the criteria for outlier is
      // tightened. If only few tracks are changed, no need to start bundle
      // adjustment right away. Instead, use a more strict criteria to filter
      LOG(INFO) << "Filtering tracks by reprojection ...";

      colmap::ObservationManager obs_manager(reconstruction);
      bool status = true;
      size_t filtered_num = 0;
      while (status && ite < options.num_iterations_ba) {
        double scaling = std::max(3 - ite, 1);
        filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
            scaling * options.inlier_thresholds.max_reprojection_error,
            reconstruction.Point3DIds(),
            colmap::ReprojectionErrorType::NORMALIZED);

        if (filtered_num > 1e-3 * reconstruction.NumPoints3D()) {
          status = false;
        } else
          ite++;
      }
      if (status) {
        LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
        break;
      }
    }

    // Filter tracks based on the estimation
    LOG(INFO) << "Filtering tracks by reprojection ...";
    {
      colmap::ObservationManager obs_manager(reconstruction);
      obs_manager.FilterPoints3DWithLargeReprojectionError(
          options.inlier_thresholds.max_reprojection_error,
          reconstruction.Point3DIds(),
          colmap::ReprojectionErrorType::NORMALIZED);
      obs_manager.FilterPoints3DWithSmallTriangulationAngle(
          options.inlier_thresholds.min_triangulation_angle,
          reconstruction.Point3DIds());
    }

    run_timer.PrintSeconds();
  }

  // 7. Retriangulation and refinement
  if (!options.skip_retriangulation) {
    LOG(INFO) << "----- Running retriangulation and refinement -----";
    THROW_CHECK_NOTNULL(database);

    colmap::Timer run_timer;
    run_timer.Start();
    if (!RetriangulateAndRefine(*database, reconstruction)) {
      return false;
    }
    run_timer.PrintSeconds();
  }

  // 8. Reconstruction pruning
  if (!options.skip_pruning) {
    LOG(INFO) << "----- Running postprocessing -----";

    colmap::Timer run_timer;
    run_timer.Start();
    cluster_ids = PruneWeaklyConnectedFrames(reconstruction);
    run_timer.PrintSeconds();
  }

  return true;
}

bool GlobalMapper::RetriangulateAndRefine(
    const colmap::Database& database, colmap::Reconstruction& reconstruction) {
  // Create database cache for retriangulation.
  constexpr int kMinNumMatches = 15;
  auto database_cache =
      colmap::DatabaseCache::Create(database,
                                    kMinNumMatches,
                                    /*ignore_watermarks=*/false,
                                    /*camera_rig_config=*/{});

  // Wrap reconstruction in shared_ptr for IncrementalMapper.
  auto reconstruction_ptr =
      std::shared_ptr<colmap::Reconstruction>(&reconstruction, [](auto) {});

  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction.DeleteAllPoints2DAndPoints3D();
  reconstruction.TranscribeImageIdsToDatabase(database);

  // Initialize mapper.
  colmap::IncrementalMapper mapper(database_cache);
  mapper.BeginReconstruction(reconstruction_ptr);

  // Triangulate all registered images.
  for (const auto image_id : reconstruction.RegImageIds()) {
    mapper.TriangulateImage(options_.retriangulation, image_id);
  }

  // Set up bundle adjustment options.
  colmap::BundleAdjustmentOptions ba_options;
  ba_options.solver_options.max_num_iterations = 50;
  ba_options.solver_options.max_linear_solver_iterations = 100;
  ba_options.print_summary = false;

  // Iterative global refinement.
  colmap::IncrementalMapper::Options mapper_options;
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   ba_options,
                                   options_.retriangulation,
                                   /*normalize_reconstruction=*/true);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  colmap::ObservationManager obs_manager(reconstruction);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      options_.inlier_thresholds.max_reprojection_error,
      reconstruction.Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);

  if (!RunBundleAdjustment(options_.bundle_adjustment,
                           /*constant_rotation=*/false,
                           reconstruction)) {
    return false;
  }

  reconstruction.Normalize();

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      options_.inlier_thresholds.max_reprojection_error,
      reconstruction.Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      options_.inlier_thresholds.min_triangulation_angle,
      reconstruction.Point3DIds());

  return true;
}

}  // namespace glomap
