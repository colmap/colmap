#include "glomap/sfm/track_retriangulation.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/database_cache.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/util/logging.h"

#include "glomap/scene/types.h"

namespace glomap {
namespace {

colmap::BundleAdjustmentOptions GetBundleAdjustmentOptions() {
  colmap::BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = 0.0;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = 50;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.logging_type = ceres::LoggingType::SILENT;
  if (VLOG_IS_ON(2)) {
    options.solver_options.minimizer_progress_to_stdout = true;
    options.solver_options.logging_type =
        ceres::LoggingType::PER_MINIMIZER_ITERATION;
  }
  options.solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = false;
  return options;
}

}  // namespace

bool RetriangulateTracks(const TriangulatorOptions& options,
                         const colmap::Database& database,
                         colmap::Reconstruction& reconstruction) {
  // Following code adapted from COLMAP
  auto database_cache =
      colmap::DatabaseCache::Create(database,
                                    options.min_num_matches,
                                    false,  // ignore_watermarks
                                    {}      // reconstruct all possible images
      );

  // Check whether the image is in the database cache. If not, set the image
  // as not registered to avoid memory error.
  std::vector<image_t> image_ids_notconnected;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!database_cache->ExistsImage(image_id) && image.HasPose()) {
      image_ids_notconnected.push_back(image_id);
      reconstruction.DeRegisterFrame(image.FrameId());
    }
  }

  // Create a shared_ptr copy for IncrementalMapper which requires shared_ptr
  std::shared_ptr<colmap::Reconstruction> recon_ptr =
      std::make_shared<colmap::Reconstruction>(reconstruction);

  colmap::IncrementalTriangulator::Options tri_options;
  tri_options.complete_max_reproj_error = options.tri_complete_max_reproj_error;
  tri_options.merge_max_reproj_error = options.tri_merge_max_reproj_error;
  tri_options.min_angle = options.tri_min_angle;

  recon_ptr->DeleteAllPoints2DAndPoints3D();
  recon_ptr->TranscribeImageIdsToDatabase(database);

  colmap::IncrementalMapper mapper(database_cache);
  mapper.BeginReconstruction(recon_ptr);

  // Triangulate all images.

  const std::vector<image_t> reg_image_ids = recon_ptr->RegImageIds();

  for (const image_t image_id : reg_image_ids) {
    mapper.TriangulateImage(tri_options, image_id);
  }
  LOG(INFO) << "Triangulated " << reg_image_ids.size() << " images";

  // Merge and complete tracks.
  mapper.CompleteAndMergeTracks(tri_options);
  const auto ba_options = GetBundleAdjustmentOptions();

  // Configure bundle adjustment.
  colmap::BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  colmap::ObservationManager observation_manager(*recon_ptr);

  const int kNumRefinements = 5;
  const double kMaxRefinementChange = 0.0005;
  for (int i = 0; i < kNumRefinements; ++i) {
    VLOG(1) << "Global bundle adjustment iteration " << i + 1 << " / "
            << kNumRefinements;
    // Avoid degeneracies in bundle adjustment.
    observation_manager.FilterObservationsWithNegativeDepth();

    const size_t num_observations = recon_ptr->ComputeNumObservations();

    std::unique_ptr<colmap::BundleAdjuster> bundle_adjuster;
    bundle_adjuster =
        CreateDefaultBundleAdjuster(ba_options, ba_config, *recon_ptr);
    if (!bundle_adjuster->Solve().IsSolutionUsable()) {
      return false;
    }

    size_t num_changed_observations = 0;
    num_changed_observations += mapper.CompleteAndMergeTracks(tri_options);
    num_changed_observations +=
        mapper.FilterPoints(colmap::IncrementalMapper::Options());
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    if (changed < kMaxRefinementChange) {
      LOG(INFO) << "Converged after " << i + 1 << " iterations";
      break;
    }
  }

  // Add the removed images back to the reconstruction
  for (const auto& image_id : image_ids_notconnected) {
    reconstruction.Image(image_id).FramePtr()->SetRigFromWorld(
        colmap::Rigid3d());
    recon_ptr->AddImage(reconstruction.Image(image_id));
  }

  // Copy the results back to the original reconstruction
  reconstruction = *recon_ptr;

  return true;
}

}  // namespace glomap
