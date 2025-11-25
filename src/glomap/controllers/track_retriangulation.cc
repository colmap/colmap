#include "glomap/controllers/track_retriangulation.h"

#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/scene/database_cache.h"

#include "glomap/io/colmap_converter.h"

#include <set>

namespace glomap {

bool RetriangulateTracks(const TriangulatorOptions& options,
                         const colmap::Database& database,
                         std::unordered_map<rig_t, Rig>& rigs,
                         std::unordered_map<camera_t, colmap::Camera>& cameras,
                         std::unordered_map<frame_t, Frame>& frames,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks) {
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
  for (auto& image : images) {
    if (!database_cache->ExistsImage(image.first) &&
        image.second.IsRegistered()) {
      image_ids_notconnected.push_back(image.first);
      image.second.frame_ptr->is_registered = false;
    }
  }

  // Convert the glomap data structures to colmap data structures
  std::shared_ptr<colmap::Reconstruction> reconstruction_ptr =
      std::make_shared<colmap::Reconstruction>();
  ConvertGlomapToColmap(rigs,
                        cameras,
                        frames,
                        images,
                        std::unordered_map<track_t, Track>(),
                        *reconstruction_ptr);

  colmap::IncrementalPipelineOptions options_colmap;
  options_colmap.triangulation.complete_max_reproj_error =
      options.tri_complete_max_reproj_error;
  options_colmap.triangulation.merge_max_reproj_error =
      options.tri_merge_max_reproj_error;
  options_colmap.triangulation.min_angle = options.tri_min_angle;

  reconstruction_ptr->DeleteAllPoints2DAndPoints3D();
  reconstruction_ptr->TranscribeImageIdsToDatabase(database);

  colmap::IncrementalMapper mapper(database_cache);
  mapper.BeginReconstruction(reconstruction_ptr);

  // Triangulate all images.
  const auto tri_options = options_colmap.Triangulation();
  const auto mapper_options = options_colmap.Mapper();

  const std::vector<image_t> reg_image_ids = reconstruction_ptr->RegImageIds();

  size_t image_idx = 0;
  for (const image_t image_id : reg_image_ids) {
    std::cout << "\r Triangulating image " << image_idx++ + 1 << " / "
              << reg_image_ids.size() << std::flush;
    mapper.TriangulateImage(tri_options, image_id);
  }
  std::cout << '\n';

  // Merge and complete tracks.
  mapper.CompleteAndMergeTracks(tri_options);

  auto ba_options = options_colmap.GlobalBundleAdjustment();
  ba_options.refine_focal_length = false;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = false;
  ba_options.refine_sensor_from_rig = false;
  ba_options.refine_rig_from_world = false;

  // Configure bundle adjustment.
  colmap::BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  colmap::ObservationManager observation_manager(*reconstruction_ptr);

  for (int i = 0; i < options_colmap.ba_global_max_refinements; ++i) {
    std::cout << "\r Global bundle adjustment iteration " << i + 1 << " / "
              << options_colmap.ba_global_max_refinements << std::flush;
    // Avoid degeneracies in bundle adjustment.
    observation_manager.FilterObservationsWithNegativeDepth();

    const size_t num_observations =
        reconstruction_ptr->ComputeNumObservations();

    std::unique_ptr<colmap::BundleAdjuster> bundle_adjuster;
    bundle_adjuster =
        CreateDefaultBundleAdjuster(ba_options, ba_config, *reconstruction_ptr);
    if (bundle_adjuster->Solve().termination_type == ceres::FAILURE) {
      return false;
    }

    size_t num_changed_observations = 0;
    num_changed_observations += mapper.CompleteAndMergeTracks(tri_options);
    num_changed_observations += mapper.FilterPoints(mapper_options);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    if (changed < options_colmap.ba_global_max_refinement_change) {
      break;
    }
  }
  std::cout << '\n';

  // Add the removed images to the reconstruction
  for (const auto& image_id : image_ids_notconnected) {
    images[image_id].frame_ptr->is_registered = true;
    colmap::Image image_colmap;
    ConvertGlomapToColmapImage(images[image_id], image_colmap, true);
    reconstruction_ptr->AddImage(std::move(image_colmap));
  }

  // Convert the colmap data structures back to glomap data structures
  ConvertColmapToGlomap(
      *reconstruction_ptr, rigs, cameras, frames, images, tracks);

  return true;
}

}  // namespace glomap
