#include "global_mapper.h"

#include "colmap/util/file.h"
#include "colmap/util/timer.h"

#include "glomap/controllers/rotation_averager.h"
#include "glomap/io/colmap_converter.h"
#include "glomap/processors/image_pair_inliers.h"
#include "glomap/processors/image_undistorter.h"
#include "glomap/processors/reconstruction_normalizer.h"
#include "glomap/processors/reconstruction_pruning.h"
#include "glomap/processors/relpose_filter.h"
#include "glomap/processors/track_filter.h"
#include "glomap/processors/view_graph_manipulation.h"

namespace glomap {

// TODO: Rig normalizaiton has not be done
bool GlobalMapper::Solve(const colmap::Database& database,
                         ViewGraph& view_graph,
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<camera_t, colmap::Camera>& cameras,
                         std::unordered_map<frame_t, Frame>& frames,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks) {
  // 0. Preprocessing
  if (!options_.skip_preprocessing) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running preprocessing ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    colmap::Timer run_timer;
    run_timer.Start();
    // If camera intrinsics seem to be good, force the pair to use essential
    // matrix
    ViewGraphManipulater::UpdateImagePairsConfig(view_graph, cameras, images);
    ViewGraphManipulater::DecomposeRelPose(view_graph, cameras, images);
    run_timer.PrintSeconds();
  }

  // 1. Run view graph calibration
  if (!options_.skip_view_graph_calibration) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running view graph calibration ..." << '\n';
    std::cout << "-------------------------------------" << '\n';
    ViewGraphCalibrator vgcalib_engine(options_.opt_vgcalib);
    if (!vgcalib_engine.Solve(view_graph, cameras, images)) {
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
    UndistortImages(cameras, images, true);
    EstimateRelativePoses(view_graph, cameras, images, options_.opt_relpose);

    InlierThresholdOptions inlier_thresholds = options_.inlier_thresholds;
    // Undistort the images and filter edges by inlier number
    ImagePairsInlierCount(view_graph, cameras, images, inlier_thresholds, true);

    RelPoseFilter::FilterInlierNum(view_graph,
                                   options_.inlier_thresholds.min_inlier_num);
    RelPoseFilter::FilterInlierRatio(
        view_graph, options_.inlier_thresholds.min_inlier_ratio);

    if (view_graph.KeepLargestConnectedComponents(frames, images) == 0) {
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
    SolveRotationAveraging(view_graph, rigs, frames, images, options_.opt_ra);

    RelPoseFilter::FilterRotations(
        view_graph, images, options_.inlier_thresholds.max_rotation_error);
    if (view_graph.KeepLargestConnectedComponents(frames, images) == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }

    // The second run is for final estimation
    if (!SolveRotationAveraging(
            view_graph, rigs, frames, images, options_.opt_ra)) {
      return false;
    }
    RelPoseFilter::FilterRotations(
        view_graph, images, options_.inlier_thresholds.max_rotation_error);
    image_t num_img = view_graph.KeepLargestConnectedComponents(frames, images);
    if (num_img == 0) {
      LOG(ERROR) << "no connected components are found";
      return false;
    }
    LOG(INFO) << num_img << " / " << images.size()
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
    TrackEngine track_engine(view_graph, images, options_.opt_track);
    std::unordered_map<track_t, Track> tracks_full;
    track_engine.EstablishFullTracks(tracks_full);

    // Filter the tracks
    track_t num_tracks = track_engine.FindTracksForProblem(tracks_full, tracks);
    LOG(INFO) << "Before filtering: " << tracks_full.size()
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
    // Undistort images in case all previous steps are skipped
    // Skip images where an undistortion already been done
    UndistortImages(cameras, images, false);

    GlobalPositioner gp_engine(options_.opt_gp);

    // TODO: consider to support other modes as well
    if (!gp_engine.Solve(view_graph, rigs, cameras, frames, images, tracks)) {
      return false;
    }
    // Filter tracks based on the estimation
    TrackFilter::FilterTracksByAngle(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_angle_error);

    // Filter tracks based on triangulation angle and reprojection error
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);
    // Set the threshold to be larger to avoid removing too many tracks
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        10 * options_.inlier_thresholds.max_reprojection_error);
    // Normalize the structure
    // If the camera rig is used, the structure do not need to be normalized
    NormalizeReconstruction(rigs, cameras, frames, images, tracks);

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
      if (!ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
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
          !ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << options_.num_iteration_bundle_adjustment
                << ", stage 2 finished";
      if (ite != options_.num_iteration_bundle_adjustment - 1)
        run_timer.PrintSeconds();

      // Normalize the structure
      NormalizeReconstruction(rigs, cameras, frames, images, tracks);

      // 6.3. Filter tracks based on the estimation
      // For the filtering, in each round, the criteria for outlier is
      // tightened. If only few tracks are changed, no need to start bundle
      // adjustment right away. Instead, use a more strict criteria to filter
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";

      bool status = true;
      size_t filtered_num = 0;
      while (status && ite < options_.num_iteration_bundle_adjustment) {
        double scaling = std::max(3 - ite, 1);
        filtered_num += TrackFilter::FilterTracksByReprojection(
            view_graph,
            cameras,
            images,
            tracks,
            scaling * options_.inlier_thresholds.max_reprojection_error);

        if (filtered_num > 1e-3 * tracks.size()) {
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
    UndistortImages(cameras, images, true);
    LOG(INFO) << "Filtering tracks by reprojection ...";
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);

    run_timer.PrintSeconds();
  }

  // 7. Retriangulation
  if (!options_.skip_retriangulation) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running retriangulation ..." << '\n';
    std::cout << "-------------------------------------" << '\n';
    for (int ite = 0; ite < options_.num_iteration_retriangulation; ite++) {
      colmap::Timer run_timer;
      run_timer.Start();
      RetriangulateTracks(options_.opt_triangulator,
                          database,
                          rigs,
                          cameras,
                          frames,
                          images,
                          tracks);
      run_timer.PrintSeconds();

      std::cout << "-------------------------------------" << '\n';
      std::cout << "Running bundle adjustment ..." << '\n';
      std::cout << "-------------------------------------" << '\n';
      LOG(INFO) << "Bundle adjustment start" << '\n';
      BundleAdjuster ba_engine(options_.opt_ba);
      if (!ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
        return false;
      }

      // Filter tracks based on the estimation
      UndistortImages(cameras, images, true);
      LOG(INFO) << "Filtering tracks by reprojection ...";
      TrackFilter::FilterTracksByReprojection(
          view_graph,
          cameras,
          images,
          tracks,
          options_.inlier_thresholds.max_reprojection_error);
      if (!ba_engine.Solve(rigs, cameras, frames, images, tracks)) {
        return false;
      }
      run_timer.PrintSeconds();
    }

    // Normalize the structure
    NormalizeReconstruction(rigs, cameras, frames, images, tracks);

    // Filter tracks based on the estimation
    UndistortImages(cameras, images, true);
    LOG(INFO) << "Filtering tracks by reprojection ...";
    TrackFilter::FilterTracksByReprojection(
        view_graph,
        cameras,
        images,
        tracks,
        options_.inlier_thresholds.max_reprojection_error);
    TrackFilter::FilterTrackTriangulationAngle(
        view_graph,
        images,
        tracks,
        options_.inlier_thresholds.min_triangulation_angle);
  }

  // 8. Reconstruction pruning
  if (!options_.skip_pruning) {
    std::cout << "-------------------------------------" << '\n';
    std::cout << "Running postprocessing ..." << '\n';
    std::cout << "-------------------------------------" << '\n';

    colmap::Timer run_timer;
    run_timer.Start();

    // Prune weakly connected images
    PruneWeaklyConnectedImages(frames, images, tracks);

    run_timer.PrintSeconds();
  }

  return true;
}

}  // namespace glomap
