#include "glomap/sfm/track_retriangulation.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/database_cache.h"
#include "colmap/sfm/incremental_mapper.h"

#include "glomap/io/colmap_converter.h"

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
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<camera_t, colmap::Camera>& cameras,
                         std::unordered_map<frame_t, Frame>& frames,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<point3D_t, Point3D>& tracks) {
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
  for (auto& [image_id, image] : images) {
    if (!database_cache->ExistsImage(image_id) && image.HasPose()) {
      image_ids_notconnected.push_back(image_id);
      image.FramePtr()->ResetPose();
    }
  }

  // Convert the glomap data structures to colmap data structures
  std::shared_ptr<colmap::Reconstruction> reconstruction =
      std::make_shared<colmap::Reconstruction>();
  ConvertGlomapToColmap(rigs,
                        cameras,
                        frames,
                        images,
                        std::unordered_map<point3D_t, Point3D>(),
                        *reconstruction);

  colmap::IncrementalTriangulator::Options tri_options;
  tri_options.complete_max_reproj_error = options.tri_complete_max_reproj_error;
  tri_options.merge_max_reproj_error = options.tri_merge_max_reproj_error;
  tri_options.min_angle = options.tri_min_angle;

  reconstruction->DeleteAllPoints2DAndPoints3D();
  reconstruction->TranscribeImageIdsToDatabase(database);

  colmap::IncrementalMapper mapper(database_cache);
  mapper.BeginReconstruction(reconstruction);

  // Triangulate all images.

  const std::vector<image_t> reg_image_ids = reconstruction->RegImageIds();

  size_t image_idx = 0;
  for (const image_t image_id : reg_image_ids) {
    std::cout << "\r Triangulating image " << image_idx++ + 1 << " / "
              << reg_image_ids.size() << std::flush;
    mapper.TriangulateImage(tri_options, image_id);
  }
  std::cout << '\n';

  // Merge and complete tracks.
  mapper.CompleteAndMergeTracks(tri_options);
  const auto ba_options = GetBundleAdjustmentOptions();

  // Configure bundle adjustment.
  colmap::BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  colmap::ObservationManager observation_manager(*reconstruction);

  const int kNumRefinements = 5;
  const double kMaxRefinementChange = 0.0005;
  for (int i = 0; i < kNumRefinements; ++i) {
    std::cout << "\r Global bundle adjustment iteration " << i + 1 << " / "
              << kNumRefinements << std::flush;
    // Avoid degeneracies in bundle adjustment.
    observation_manager.FilterObservationsWithNegativeDepth();

    const size_t num_observations = reconstruction->ComputeNumObservations();

    std::unique_ptr<colmap::BundleAdjuster> bundle_adjuster;
    bundle_adjuster =
        CreateDefaultBundleAdjuster(ba_options, ba_config, *reconstruction);
    if (bundle_adjuster->Solve().termination_type == ceres::FAILURE) {
      return false;
    }

    size_t num_changed_observations = 0;
    num_changed_observations += mapper.CompleteAndMergeTracks(tri_options);
    num_changed_observations +=
        mapper.FilterPoints(colmap::IncrementalMapper::Options());
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    if (changed < kMaxRefinementChange) {
      break;
    }
  }
  std::cout << '\n';

  // Add the removed images to the reconstruction
  for (const auto& image_id : image_ids_notconnected) {
    const auto& image = images[image_id];
    image.FramePtr()->SetRigFromWorld(Rigid3d());
    colmap::Image image_colmap;
    ConvertGlomapToColmapImage(images[image_id], image_colmap, true);
    reconstruction->AddImage(std::move(image_colmap));
  }

  // Convert the colmap data structures back to glomap data structures
  ConvertColmapToGlomap(*reconstruction, rigs, cameras, frames, images, tracks);

  return true;
}

}  // namespace glomap
