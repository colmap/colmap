// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "sfm/controllers.h"

#include <boost/filesystem.hpp>

#include "util/misc.h"

namespace colmap {
namespace {

size_t TriangulateImage(const SparseMapperOptions& options, const Image& image,
                        IncrementalMapper* mapper) {
  std::cout << "  => Continued observations: " << image.NumPoints3D()
            << std::endl;
  const size_t num_tris =
      mapper->TriangulateImage(options.TriangulationOptions(), image.ImageId());
  std::cout << "  => Added observations: " << num_tris << std::endl;
  return num_tris;
}

size_t CompleteAndMergeTracks(const SparseMapperOptions& options,
                              IncrementalMapper* mapper) {
  const size_t num_completed_observations =
      mapper->CompleteTracks(options.TriangulationOptions());
  std::cout << "  => Merged observations: " << num_completed_observations
            << std::endl;
  const size_t num_merged_observations =
      mapper->MergeTracks(options.TriangulationOptions());
  std::cout << "  => Completed observations: " << num_merged_observations
            << std::endl;
  return num_completed_observations + num_merged_observations;
}

size_t FilterPoints(const SparseMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.IncrementalMapperOptions());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;
  return num_filtered_observations;
}

size_t FilterImages(const SparseMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_images =
      mapper->FilterImages(options.IncrementalMapperOptions());
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
  return num_filtered_images;
}

void AdjustGlobalBundle(const SparseMapperOptions& options,
                        const Reconstruction& reconstruction,
                        IncrementalMapper* mapper) {
  BundleAdjuster::Options custom_options =
      options.GlobalBundleAdjustmentOptions();

  const size_t num_reg_images = reconstruction.NumRegImages();

  // Use stricter convergence criteria for first registered images.
  const size_t kMinNumRegImages = 10;
  if (num_reg_images < kMinNumRegImages) {
    custom_options.solver_options.function_tolerance /= 10;
    custom_options.solver_options.gradient_tolerance /= 10;
    custom_options.solver_options.parameter_tolerance /= 10;
    custom_options.solver_options.max_num_iterations *= 2;
    custom_options.solver_options.max_linear_solver_iterations = 200;
  }

  PrintHeading1("Global bundle adjustment");
  if (options.ba_global_use_pba && num_reg_images >= kMinNumRegImages &&
      ParallelBundleAdjuster::IsReconstructionSupported(reconstruction)) {
    mapper->AdjustParallelGlobalBundle(
        options.ParallelGlobalBundleAdjustmentOptions());
  } else {
    mapper->AdjustGlobalBundle(custom_options);
  }
}

void IterativeLocalRefinement(const SparseMapperOptions& options,
                              const image_t image_id,
                              IncrementalMapper* mapper) {
  auto ba_options = options.LocalBundleAdjustmentOptions();
  for (int i = 0; i < options.ba_local_max_refinements; ++i) {
    const auto report = mapper->AdjustLocalBundle(
        options.IncrementalMapperOptions(), ba_options,
        options.TriangulationOptions(), image_id);
    std::cout << "  => Merged observations: " << report.num_merged_observations
              << std::endl;
    std::cout << "  => Completed observations: "
              << report.num_completed_observations << std::endl;
    std::cout << "  => Filtered observations: "
              << report.num_filtered_observations << std::endl;
    const double changed =
        (report.num_merged_observations + report.num_completed_observations +
         report.num_filtered_observations) /
        static_cast<double>(report.num_adjusted_observations);
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < options.ba_local_max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration.
    ba_options.loss_function_type =
        BundleAdjuster::Options::LossFunctionType::TRIVIAL;
  }
}

void IterativeGlobalRefinement(const SparseMapperOptions& options,
                               const Reconstruction& reconstruction,
                               IncrementalMapper* mapper) {
  PrintHeading1("Retriangulation");
  CompleteAndMergeTracks(options, mapper);
  std::cout << "  => Retriangulated observations: "
            << mapper->Retriangulate(options.TriangulationOptions())
            << std::endl;

  for (int i = 0; i < options.ba_global_max_refinements; ++i) {
    const size_t num_observations = reconstruction.ComputeNumObservations();
    size_t num_changed_observations = 0;
    AdjustGlobalBundle(options, reconstruction, mapper);
    num_changed_observations += CompleteAndMergeTracks(options, mapper);
    num_changed_observations += FilterPoints(options, mapper);
    const double changed =
        static_cast<double>(num_changed_observations) / num_observations;
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  FilterImages(options, mapper);
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
    std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                              reconstruction->Image(image_id).Name().c_str(),
                              image_path.c_str())
              << std::endl;
  }
}

}  // namespace

IncrementalMapperController::IncrementalMapperController(
    const OptionManager& options, ReconstructionManager* reconstruction_manager)
    : options_(options), reconstruction_manager_(reconstruction_manager) {
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

void IncrementalMapperController::Run() {
  const SparseMapperOptions& mapper_options = *options_.sparse_mapper_options;

  //////////////////////////////////////////////////////////////////////////////
  // Load data from database
  //////////////////////////////////////////////////////////////////////////////

  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    Database database(*options_.database_path);
    Timer timer;
    timer.Start();
    const size_t min_num_matches =
        static_cast<size_t>(mapper_options.min_num_matches);
    database_cache.Load(database, min_num_matches,
                        mapper_options.ignore_watermarks);
    std::cout << std::endl;
    timer.PrintMinutes();
  }

  std::cout << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Main loop
  //////////////////////////////////////////////////////////////////////////////

  IncrementalMapper mapper(&database_cache);

  // Is there a sub-model before we start the reconstruction? I.e. the user
  // has imported an existing reconstruction.
  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                  "single reconstruction, but "
                                                  "multiple are given.";

  for (int num_trials = 0; num_trials < mapper_options.init_num_trials;
       ++num_trials) {
    WaitIfPaused();
    if (IsStopped()) {
      break;
    }

    size_t reconstruction_idx;
    if (!initial_reconstruction_given || num_trials > 0) {
      reconstruction_idx = reconstruction_manager_->Add();
    } else {
      reconstruction_idx = 0;
    }

    Reconstruction& reconstruction =
        reconstruction_manager_->Get(reconstruction_idx);

    mapper.BeginReconstruction(&reconstruction);

    ////////////////////////////////////////////////////////////////////////////
    // Register initial pair
    ////////////////////////////////////////////////////////////////////////////

    if (reconstruction.NumRegImages() == 0) {
      image_t image_id1, image_id2;

      image_id1 = static_cast<image_t>(mapper_options.init_image_id1);
      image_id2 = static_cast<image_t>(mapper_options.init_image_id2);

      // Try to find good initial pair
      if (mapper_options.init_image_id1 == -1 ||
          mapper_options.init_image_id2 == -1) {
        const bool find_init_success = mapper.FindInitialImagePair(
            mapper_options.IncrementalMapperOptions(), &image_id1, &image_id2);

        if (!find_init_success) {
          std::cerr << "  => Could not find good initial pair." << std::endl;
          const bool kDiscardReconstruction = true;
          mapper.EndReconstruction(kDiscardReconstruction);
          reconstruction_manager_->Delete(reconstruction_idx);
          break;
        }
      }

      PrintHeading1(StringPrintf("Initializing with images #%d and #%d",
                                 image_id1, image_id2));
      const bool reg_init_success = mapper.RegisterInitialImagePair(
          mapper_options.IncrementalMapperOptions(), image_id1, image_id2);

      if (!reg_init_success) {
        std::cout << "  => Initialization failed." << std::endl;
        break;
      }

      AdjustGlobalBundle(mapper_options, reconstruction, &mapper);
      FilterPoints(mapper_options, &mapper);
      FilterImages(mapper_options, &mapper);

      // Initial image pair failed to register.
      if (reconstruction.NumRegImages() == 0 ||
          reconstruction.NumPoints3D() == 0) {
        const bool kDiscardReconstruction = true;
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        continue;
      }

      if (mapper_options.extract_colors) {
        ExtractColors(*options_.image_path, image_id1, &reconstruction);
      }
    }

    Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

    ////////////////////////////////////////////////////////////////////////////
    // Incremental mapping
    ////////////////////////////////////////////////////////////////////////////

    size_t prev_num_reg_images = reconstruction.NumRegImages();
    size_t prev_num_points = reconstruction.NumPoints3D();
    int num_global_bas = 1;

    bool reg_next_success = true;
    while (reg_next_success) {
      WaitIfPaused();
      if (IsStopped()) {
        break;
      }

      reg_next_success = false;

      const std::vector<image_t> next_images =
          mapper.FindNextImages(mapper_options.IncrementalMapperOptions());

      if (next_images.empty()) {
        break;
      }

      for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
        const image_t next_image_id = next_images[reg_trial];
        const Image& next_image = reconstruction.Image(next_image_id);

        PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                                   reconstruction.NumRegImages() + 1));

        std::cout << StringPrintf("  => Image sees %d / %d points",
                                  next_image.NumVisiblePoints3D(),
                                  next_image.NumObservations())
                  << std::endl;

        reg_next_success = mapper.RegisterNextImage(
            mapper_options.IncrementalMapperOptions(), next_image_id);

        if (reg_next_success) {
          TriangulateImage(mapper_options, next_image, &mapper);
          IterativeLocalRefinement(mapper_options, next_image_id, &mapper);

          if (reconstruction.NumRegImages() >=
                  mapper_options.ba_global_images_ratio * prev_num_reg_images ||
              reconstruction.NumRegImages() >=
                  mapper_options.ba_global_images_freq + prev_num_reg_images ||
              reconstruction.NumPoints3D() >=
                  mapper_options.ba_global_points_ratio * prev_num_points ||
              reconstruction.NumPoints3D() >=
                  mapper_options.ba_global_points_freq + prev_num_points) {
            IterativeGlobalRefinement(mapper_options, reconstruction, &mapper);
            prev_num_points = reconstruction.NumPoints3D();
            prev_num_reg_images = reconstruction.NumRegImages();
            num_global_bas += 1;
          }

          if (mapper_options.extract_colors) {
            ExtractColors(*options_.image_path, next_image_id, &reconstruction);
          }

          Callback(NEXT_IMAGE_REG_CALLBACK);

          break;
        } else {
          std::cout << "  => Could not register, trying another image."
                    << std::endl;

          // If initial pair fails to continue for some time,
          // abort and try different initial pair.
          const size_t kMinNumInitialRegTrials = 30;
          if (reg_trial >= kMinNumInitialRegTrials &&
              reconstruction.NumRegImages() <
                  static_cast<size_t>(mapper_options.min_model_size)) {
            break;
          }
        }
      }

      const size_t max_model_overlap =
          static_cast<size_t>(mapper_options.max_model_overlap);
      if (mapper.NumSharedRegImages() >= max_model_overlap) {
        break;
      }
    }

    if (IsStopped()) {
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
      break;
    }

    // Only run final global BA, if last incremental BA was not global.
    if (reconstruction.NumRegImages() >= 2 &&
        reconstruction.NumRegImages() != prev_num_reg_images &&
        reconstruction.NumPoints3D() != prev_num_points) {
      IterativeGlobalRefinement(mapper_options, reconstruction, &mapper);
    }

    // If the total number of images is small then do not enforce the minimum
    // model size so that we can reconstruct small image collections.
    const size_t min_model_size =
        std::min(database_cache.NumImages(),
                 static_cast<size_t>(mapper_options.min_model_size));
    if ((mapper_options.multiple_models &&
         reconstruction.NumRegImages() < min_model_size) ||
        reconstruction.NumRegImages() == 0) {
      const bool kDiscardReconstruction = true;
      mapper.EndReconstruction(kDiscardReconstruction);
      reconstruction_manager_->Delete(reconstruction_idx);
    } else {
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
    }

    Callback(LAST_IMAGE_REG_CALLBACK);

    const size_t max_num_models =
        static_cast<size_t>(mapper_options.max_num_models);
    if (initial_reconstruction_given || !mapper_options.multiple_models ||
        reconstruction_manager_->Size() >= max_num_models ||
        mapper.NumTotalRegImages() >= database_cache.NumImages() - 1) {
      break;
    }
  }

  std::cout << std::endl;
  GetTimer().PrintMinutes();
}

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options)
    : reconstruction(nullptr), options_(options) {}

void BundleAdjustmentController::Run() {
  CHECK_NOTNULL(reconstruction);

  PrintHeading1("Global bundle adjustment");

  const std::vector<image_t>& reg_image_ids = reconstruction->RegImageIds();

  if (reg_image_ids.size() < 2) {
    std::cout << "ERROR: Need at least two views." << std::endl;
    reconstruction = nullptr;
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  reconstruction->FilterObservationsWithNegativeDepth();

  BundleAdjuster::Options ba_options = options_.ba_options->Options();
  ba_options.solver_options.minimizer_progress_to_stdout = true;

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }
  ba_config.SetConstantPose(reg_image_ids[0]);
  ba_config.SetConstantTvec(reg_image_ids[1], {0});

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  bundle_adjuster.Solve(reconstruction);

  // Normalize scene for numerical stability and
  // to avoid large scale changes in viewer.
  reconstruction->Normalize();

  reconstruction = nullptr;

  GetTimer().PrintMinutes();
}

}  // namespace colmap
