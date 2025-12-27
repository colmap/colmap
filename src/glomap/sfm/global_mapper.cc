#include "glomap/sfm/global_mapper.h"

#include "colmap/sfm/observation_manager.h"
#include "colmap/util/timer.h"

#include "glomap/processors/image_pair_inliers.h"
#include "glomap/processors/reconstruction_pruning.h"
#include "glomap/processors/view_graph_manipulation.h"
#include "glomap/sfm/rotation_averager.h"

namespace glomap {

// TODO: Rig normalizaiton has not be done
bool GlobalMapper::Solve(const colmap::Database* database,
                         ViewGraph& view_graph,
                         colmap::Reconstruction& reconstruction,
                         std::vector<colmap::PosePrior>& pose_priors,
                         std::unordered_map<frame_t, int>& cluster_ids) {
  // 0. Preprocessing
  if (!options_.skip_preprocessing) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running preprocessing ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    colmap::Timer run_timer;
    run_timer.Start();
    // If camera intrinsics seem to be good, force the pair to use essential
    // matrix
    ViewGraphManipulater::UpdateImagePairsConfig(view_graph, reconstruction);
    ViewGraphManipulater::DecomposeRelPose(view_graph, reconstruction);
    run_timer.PrintSeconds();
  }

  // 1. Run view graph calibration
  if (!options_.skip_view_graph_calibration) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running view graph calibration ..." << '\n';
    std::cout << "-------------------------------------" << '\n';
    ViewGraphCalibrator vgcalib_engine(options_.opt_vgcalib);
    if (!vgcalib_engine.Solve(view_graph, reconstruction)) {
      return false;
    }
  }

  // 2. Run relative pose estimation
  //   TODO: Use generalized relative pose estimation for rigs.
  if (!options_.skip_relative_pose_estimation) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running relative pose estimation ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    colmap::Timer run_timer;
    run_timer.Start();
    // Relative pose relies on the undistorted images
    EstimateRelativePoses(view_graph, reconstruction, options_.opt_relpose);

    InlierThresholdOptions inlier_thresholds = options_.inlier_thresholds;
    // Undistort the images and filter edges by inlier number
    ImagePairsInlierCount(view_graph, reconstruction, inlier_thresholds, true);

    view_graph.FilterByNumInliers(options_.inlier_thresholds.min_inlier_num);
    view_graph.FilterByInlierRatio(options_.inlier_thresholds.min_inlier_ratio);

    if (view_graph.KeepLargestConnectedComponents(reconstruction) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    run_timer.PrintSeconds();
  }

  // 3. Run rotation averaging for three times
  if (!options_.skip_rotation_averaging) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running rotation averaging ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    colmap::Timer run_timer;
    run_timer.Start();

    // The first run is for filtering
    SolveRotationAveraging(view_graph,
                           reconstruction,
                           pose_priors,
                           RotationAveragerOptions(options_.opt_ra));

    view_graph.FilterByRelativeRotation(
        reconstruction, options_.inlier_thresholds.max_rotation_error);
    if (view_graph.KeepLargestConnectedComponents(reconstruction) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    // The second run is for final estimation
    if (!SolveRotationAveraging(view_graph,
                                reconstruction,
                                pose_priors,
                                RotationAveragerOptions(options_.opt_ra))) {
      return false;
    }
    view_graph.FilterByRelativeRotation(
        reconstruction, options_.inlier_thresholds.max_rotation_error);
    image_t num_img = view_graph.KeepLargestConnectedComponents(reconstruction);
    if (num_img == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }
    LOG(INFO) << num_img << " / " << reconstruction.NumImages()
              << " images are within the connected component." << '\n';

    run_timer.PrintSeconds();
  }

  // 4. Track establishment and selection
  if (!options_.skip_track_establishment) {
    colmap::Timer run_timer;
    run_timer.Start();

    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running track establishment ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    // TrackEngine reads images, writes unfiltered tracks to a temporary map,
    // then filters into the main reconstruction
    std::unordered_map<point3D_t, Point3D> unfiltered_tracks;
    TrackEngine track_engine(
        view_graph, reconstruction.Images(), options_.opt_track);
    track_engine.EstablishFullTracks(unfiltered_tracks);

    // Filter the tracks into a selected subset
    std::unordered_map<point3D_t, Point3D> selected_tracks;
    point3D_t num_tracks =
        track_engine.FindTracksForProblem(unfiltered_tracks, selected_tracks);

    // Add selected tracks to reconstruction
    for (auto& [track_id, track] : selected_tracks) {
      reconstruction.AddPoint3D(track_id, std::move(track));
    }
    LOG(INFO) << "Before filtering: " << unfiltered_tracks.size()
              << ", after filtering: " << num_tracks << '\n';

    run_timer.PrintSeconds();
  }

  // 5. Global positioning
  if (!options_.skip_global_positioning) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running global positioning ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    if (options_.opt_gp.constraint_type !=
        GlobalPositionerOptions::ConstraintType::ONLY_POINTS) {
      LOG(ERROR) << "Only points are used for solving camera positions";
      return false;
    }

    colmap::Timer run_timer;
    run_timer.Start();

    GlobalPositioner gp_engine(options_.opt_gp);

    // TODO: consider to support other modes as well
    if (!gp_engine.Solve(view_graph, reconstruction)) {
      return false;
    }
    // Filter tracks based on the estimation
    colmap::ObservationManager obs_manager(reconstruction);
    obs_manager.FilterPoints3DWithLargeAngularError(
        options_.inlier_thresholds.max_angle_error,
        reconstruction.Point3DIds());

    // Filter tracks based on triangulation angle and reprojection error
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        options_.inlier_thresholds.min_triangulation_angle,
        reconstruction.Point3DIds());
    // Set the threshold to be larger to avoid removing too many tracks
    obs_manager.FilterPoints3DWithLargeReprojectionError(
        10 * options_.inlier_thresholds.max_reprojection_error,
        reconstruction.Point3DIds(),
        /*use_normalized_error=*/true);
    // Normalize the structure
    // If the camera rig is used, the structure do not need to be normalized
    reconstruction.Normalize();

    run_timer.PrintSeconds();
  }

  // 6. Bundle adjustment
  if (!options_.skip_bundle_adjustment) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running bundle adjustment ..." << '\n';
    std::cout << "-------------------------------------" << '\n';
    LOG(INFO) << "Bundle adjustment start" << '\n';

    colmap::Timer run_timer;
    run_timer.Start();

    for (int ite = 0; ite < options_.num_iteration_bundle_adjustment; ite++) {
      BundleAdjuster ba_engine(options_.opt_ba);

      BundleAdjusterOptions& ba_engine_options_inner = ba_engine.GetOptions();

      // Staged bundle adjustment
      // 6.1. First stage: optimize positions only
      ba_engine_options_inner.optimize_rotations = false;
      if (!ba_engine.Solve(reconstruction)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 1 finished (position only)";
      run_timer.PrintSeconds();

      // 6.2. Second stage: optimize rotations if desired
      ba_engine_options_inner.optimize_rotations =
          options_.opt_ba.optimize_rotations;
      if (ba_engine_options_inner.optimize_rotations &&
          !ba_engine.Solve(reconstruction)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 2 finished";
      if (ite != options_.num_iteration_bundle_adjustment - 1)
        run_timer.PrintSeconds();

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
      while (status && ite < options_.num_iteration_bundle_adjustment) {
        double scaling = std::max(3 - ite, 1);
        filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
            scaling * options_.inlier_thresholds.max_reprojection_error,
            reconstruction.Point3DIds(),
            /*use_normalized_error=*/true);

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
          options_.inlier_thresholds.max_reprojection_error,
          reconstruction.Point3DIds(),
          /*use_normalized_error=*/true);
      obs_manager.FilterPoints3DWithSmallTriangulationAngle(
          options_.inlier_thresholds.min_triangulation_angle,
          reconstruction.Point3DIds());
    }

    run_timer.PrintSeconds();
  }

  // 7. Retriangulation
  if (!options_.skip_retriangulation) {
    THROW_CHECK_NOTNULL(database);
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running retriangulation ..." << '\n';
    std::cout << "-------------------------------------" << '\n';
    for (int ite = 0; ite < options_.num_iteration_retriangulation; ite++) {
      colmap::Timer run_timer;
      run_timer.Start();
      RetriangulateTracks(options_.opt_triangulator, *database, reconstruction);
      run_timer.PrintSeconds();

      std::cout << "-------------------------------------" << '\n';
      std::cout << "Running bundle adjustment ..." << '\n';
      std::cout << "-------------------------------------" << '\n';
      LOG(INFO) << "Bundle adjustment start" << '\n';
      BundleAdjuster ba_engine(options_.opt_ba);
      if (!ba_engine.Solve(reconstruction)) {
        return false;
      }

      // Filter tracks based on the estimation
      LOG(INFO) << "Filtering tracks by reprojection ...";
      colmap::ObservationManager(reconstruction)
          .FilterPoints3DWithLargeReprojectionError(
              options_.inlier_thresholds.max_reprojection_error,
              reconstruction.Point3DIds(),
              /*use_normalized_error=*/true);
      if (!ba_engine.Solve(reconstruction)) {
        return false;
      }
      run_timer.PrintSeconds();
    }

    // Normalize the structure
    reconstruction.Normalize();

    // Filter tracks based on the estimation
    LOG(INFO) << "Filtering tracks by reprojection ...";
    {
      colmap::ObservationManager obs_manager(reconstruction);
      obs_manager.FilterPoints3DWithLargeReprojectionError(
          options_.inlier_thresholds.max_reprojection_error,
          reconstruction.Point3DIds(),
          /*use_normalized_error=*/true);
      obs_manager.FilterPoints3DWithSmallTriangulationAngle(
          options_.inlier_thresholds.min_triangulation_angle,
          reconstruction.Point3DIds());
    }
  }

  // 8. Reconstruction pruning
  if (!options_.skip_pruning) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running postprocessing ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    colmap::Timer run_timer;
    run_timer.Start();

    // Prune weakly connected images
    PruneWeaklyConnectedImages(reconstruction, cluster_ids);

    run_timer.PrintSeconds();
  }

  return true;
}

}  // namespace glomap
