#include "glomap/sfm/global_mapper.h"

#include "colmap/math/union_find.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include "glomap/estimators/rotation_averaging.h"
#include "glomap/processors/reconstruction_pruning.h"

#include <algorithm>

namespace glomap {
namespace {

GlobalMapperOptions InitializeOptions(const GlobalMapperOptions& options) {
  // Propagate random seed and num_threads to component options.
  GlobalMapperOptions custom = options;
  if (custom.random_seed >= 0) {
    custom.rotation_averaging.random_seed = custom.random_seed;
    custom.global_positioning.random_seed = custom.random_seed;
    custom.global_positioning.use_parameter_block_ordering = false;
    custom.retriangulation.random_seed = custom.random_seed;
  }
  custom.global_positioning.solver_options.num_threads = custom.num_threads;
  custom.bundle_adjustment.solver_options.num_threads = custom.num_threads;
  return custom;
}

}  // namespace

GlobalMapper::GlobalMapper(
    const GlobalMapperOptions& options,
    std::shared_ptr<const colmap::DatabaseCache> database_cache)
    : options_(InitializeOptions(options)),
      database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<colmap::Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  pose_graph_ = std::make_shared<class PoseGraph>();
  pose_graph_->Load(*database_cache_->CorrespondenceGraph());
}

std::shared_ptr<colmap::Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Read pose priors from the database cache.
  const std::vector<colmap::PosePrior>& pose_priors =
      database_cache_->PosePriors();

  // First pass: solve rotation averaging on all frames, then filter outlier
  // pairs by rotation error and de-register frames outside the largest
  // connected component.
  RotationEstimatorOptions custom_options = options;
  custom_options.filter_unregistered = false;
  if (!SolveRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  // Second pass: re-solve on registered frames only to refine rotations
  // after outlier removal.
  custom_options.filter_unregistered = true;
  if (!SolveRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  VLOG(1) << reconstruction_->NumRegImages() << " / "
          << reconstruction_->NumImages()
          << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks() {
  using Observation = std::pair<image_t, colmap::point2D_t>;
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);

  // Build keypoints map from registered images.
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
      image_id_to_keypoints;
  for (const auto image_id : reconstruction_->RegImageIds()) {
    const auto& image = reconstruction_->Image(image_id);
    std::vector<Eigen::Vector2d> points;
    points.reserve(image.NumPoints2D());
    for (const auto& point2D : image.Points2D()) {
      points.push_back(point2D.xy);
    }
    image_id_to_keypoints.emplace(image_id, std::move(points));
  }

  auto corr_graph = database_cache_->CorrespondenceGraph();

  // Union all matching observations.
  colmap::UnionFind<Observation> uf;
  colmap::FeatureMatches matches;
  for (const auto& [pair_id, edge] : pose_graph_->ValidEdges()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    THROW_CHECK(image_id_to_keypoints.count(image_id1))
        << "Missing keypoints for image " << image_id1;
    THROW_CHECK(image_id_to_keypoints.count(image_id2))
        << "Missing keypoints for image " << image_id2;
    corr_graph->ExtractMatchesBetweenImages(image_id1, image_id2, matches);
    for (const auto& match : matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    }
  }

  // Group observations by their root.
  uf.Compress();
  std::unordered_map<Observation, std::vector<Observation>> track_map;
  for (const auto& [obs, root] : uf.Parents()) {
    track_map[root].push_back(obs);
  }
  LOG(INFO) << "Established " << track_map.size() << " tracks from "
            << uf.Parents().size() << " observations";

  // Validate tracks, check consistency, and collect valid ones with lengths.
  std::unordered_map<point3D_t, Point3D> candidate_points3D;
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;

  for (const auto& [track_id, observations] : track_map) {
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : observations) {
      const Eigen::Vector2d& xy =
          image_id_to_keypoints.at(image_id).at(feature_id);

      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          const double sq_threshold =
              options_.track_intra_image_consistency_threshold *
              options_.track_intra_image_consistency_threshold;
          if ((existing_xy - xy).squaredNorm() > sq_threshold) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          ++discarded_counter;
          break;
        }
        it->second.push_back(xy);
      } else {
        image_id_set[image_id].push_back(xy);
      }
      point3D.track.AddElement(image_id, feature_id);
    }

    if (!is_consistent) continue;

    const size_t num_images = image_id_set.size();
    if (num_images <
        static_cast<size_t>(options_.track_min_num_views_per_track))
      continue;

    const point3D_t point3D_id = next_point3D_id++;
    track_lengths.emplace_back(point3D.track.Length(), point3D_id);
    candidate_points3D.emplace(point3D_id, std::move(point3D));
  }

  LOG(INFO) << "Kept " << candidate_points3D.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";

  // Sort tracks by length (descending) and select for problem.
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  std::unordered_map<image_t, size_t> tracks_per_image;
  size_t images_left = image_id_to_keypoints.size();
  for (const auto& [track_length, point3D_id] : track_lengths) {
    auto& point3D = candidate_points3D.at(point3D_id);

    // Check if any image in this track still needs more observations.
    const bool should_add = std::any_of(
        point3D.track.Elements().begin(),
        point3D.track.Elements().end(),
        [&](const auto& obs) {
          return tracks_per_image[obs.image_id] <=
                 static_cast<size_t>(options_.track_required_tracks_per_view);
        });
    if (!should_add) continue;

    // Update image counts.
    for (const auto& obs : point3D.track.Elements()) {
      auto& count = tracks_per_image[obs.image_id];
      if (count == static_cast<size_t>(options_.track_required_tracks_per_view))
        --images_left;
      ++count;
    }

    // Add track after updating counts so we can move.
    reconstruction_->AddPoint3D(point3D_id, std::move(point3D));

    if (images_left == 0) break;
  }

  LOG(INFO) << "Before filtering: " << candidate_points3D.size()
            << ", after filtering: " << reconstruction_->NumPoints3D();
}

bool GlobalMapper::GlobalPositioning() {
  if (options_.global_positioning.constraint_type !=
      GlobalPositioningConstraintType::ONLY_POINTS) {
    LOG(ERROR) << "Only points are used for solving camera positions";
    return false;
  }

  GlobalPositioner gp_engine(options_.global_positioning);

  // TODO: consider to support other modes as well
  if (!gp_engine.Solve(*pose_graph_, *reconstruction_)) {
    return false;
  }

  // Filter tracks based on the estimation
  colmap::ObservationManager obs_manager(*reconstruction_);

  // First pass: use relaxed threshold (2x) for cameras without prior focal.
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      2.0 * options_.max_angular_reproj_error_deg,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::ANGULAR);

  // Second pass: apply strict threshold for cameras with prior focal length.
  const double max_angular_error_rad =
      colmap::DegToRad(options_.max_angular_reproj_error_deg);
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
      if (error > max_angular_error_rad) {
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
      options_.min_tri_angle_deg, reconstruction_->Point3DIds());
  // Set the threshold to be larger to avoid removing too many tracks
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10 * options_.max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);

  // Normalize the structure
  // If the camera rig is used, the structure do not need to be normalized
  reconstruction_->Normalize();

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment() {
  const BundleAdjusterOptions& ba_options = options_.bundle_adjustment;
  for (int ite = 0; ite < options_.num_iterations_ba; ite++) {
    // First stage: optimize positions only (rotation constant)
    if (!RunBundleAdjustment(options_.bundle_adjustment,
                             /*constant_rotation=*/true,
                             *reconstruction_)) {
      return false;
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << options_.num_iterations_ba
              << ", stage 1 finished (position only)";

    // Second stage: optimize rotations if desired
    if (options_.bundle_adjustment.optimize_rotations &&
        !RunBundleAdjustment(options_.bundle_adjustment,
                             /*constant_rotation=*/false,
                             *reconstruction_)) {
      return false;
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << options_.num_iterations_ba << ", stage 2 finished";

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
    while (status && ite < options_.num_iterations_ba) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * options_.max_normalized_reproj_error,
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
        options_.max_normalized_reproj_error,
        reconstruction_->Point3DIds(),
        colmap::ReprojectionErrorType::NORMALIZED);
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(
        options_.min_tri_angle_deg, reconstruction_->Point3DIds());
  }

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine() {
  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();

  // Initialize mapper.
  colmap::IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  // Triangulate all registered images.
  for (const auto image_id : reconstruction_->RegImageIds()) {
    mapper.TriangulateImage(options_.retriangulation, image_id);
  }

  // Set up bundle adjustment options for colmap's incremental mapper.
  colmap::BundleAdjustmentOptions colmap_ba_options;
  colmap_ba_options.solver_options.num_threads =
      options_.bundle_adjustment.solver_options.num_threads;
  colmap_ba_options.solver_options.max_num_iterations = 50;
  colmap_ba_options.solver_options.max_linear_solver_iterations = 100;
  colmap_ba_options.print_summary = false;

  // Iterative global refinement.
  colmap::IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options_.random_seed;
  mapper.IterativeGlobalRefinement(/*max_num_refinements=*/5,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   colmap_ba_options,
                                   options_.retriangulation,
                                   /*normalize_reconstruction=*/true);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  colmap::ObservationManager obs_manager(*reconstruction_);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      options_.max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);

  if (!RunBundleAdjustment(options_.bundle_adjustment,
                           /*constant_rotation=*/false,
                           *reconstruction_)) {
    return false;
  }

  reconstruction_->Normalize();

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      options_.max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      colmap::ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      options_.min_tri_angle_deg, reconstruction_->Point3DIds());

  return true;
}

// TODO: Rig normalizaiton has not been done
bool GlobalMapper::Solve(std::unordered_map<frame_t, int>& cluster_ids) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Run rotation averaging
  if (!options_.skip_rotation_averaging) {
    LOG(INFO) << "----- Running rotation averaging -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(options_.rotation_averaging)) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Track establishment and selection
  if (!options_.skip_track_establishment) {
    LOG(INFO) << "----- Running track establishment -----";
    colmap::Timer run_timer;
    run_timer.Start();
    EstablishTracks();
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Global positioning
  if (!options_.skip_global_positioning) {
    LOG(INFO) << "----- Running global positioning -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning()) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Bundle adjustment
  if (!options_.skip_bundle_adjustment) {
    LOG(INFO) << "----- Running iterative bundle adjustment -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment()) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Retriangulation
  if (!options_.skip_retriangulation) {
    LOG(INFO) << "----- Running iterative retriangulation and refinement -----";
    colmap::Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine()) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Reconstruction pruning
  if (!options_.skip_pruning) {
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
