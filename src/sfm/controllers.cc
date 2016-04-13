// COLMAP - Structure-from-Motion.
// Copyright (C) 2016  Johannes L. Schoenberger <jsch at cs.unc.edu>
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
#include <boost/format.hpp>

#include "util/misc.h"

namespace colmap {
namespace {

size_t TriangulateImage(const MapperOptions& options, const Image& image,
                        IncrementalMapper* mapper) {
  std::cout << "  => Continued observations: " << image.NumPoints3D()
            << std::endl;
  const size_t num_tris =
      mapper->TriangulateImage(options.TriangulationOptions(), image.ImageId());
  std::cout << "  => Added observations: " << num_tris << std::endl;
  return num_tris;
}

size_t CompleteAndMergeTracks(const MapperOptions& options,
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

size_t FilterPoints(const MapperOptions& options, IncrementalMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.IncrementalMapperOptions());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;
  return num_filtered_observations;
}

size_t FilterImages(const MapperOptions& options, IncrementalMapper* mapper) {
  const size_t num_filtered_images =
      mapper->FilterImages(options.IncrementalMapperOptions());
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
  return num_filtered_images;
}

void AdjustGlobalBundle(const MapperOptions& options,
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
  if (options.ba_global_use_pba && num_reg_images >= kMinNumRegImages) {
    mapper->AdjustParallelGlobalBundle(
        options.ParallelGlobalBundleAdjustmentOptions());
  } else {
    mapper->AdjustGlobalBundle(custom_options);
  }
}

void IterativeLocalRefinement(const MapperOptions& options,
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
    std::cout << "  => Changed observations: " << changed << std::endl;
    if (changed < options.ba_local_max_refinement_change) {
      break;
    }
    // Only use robust cost function for first iteration.
    ba_options.loss_function_type =
        BundleAdjuster::Options::LossFunctionType::TRIVIAL;
  }
}

void IterativeGlobalRefinement(const MapperOptions& options,
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
    std::cout << "  => Changed observations: " << changed << std::endl;
    if (changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  FilterImages(options, mapper);
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColors(image_id, image_path)) {
    std::cout << boost::format("WARNING: Could not read image %s at path %s.") %
                     reconstruction->Image(image_id).Name() % image_path
              << std::endl;
  }
}

}  // namespace

IncrementalMapperController::IncrementalMapperController(
    const OptionManager& options)
    : action_render(nullptr),
      action_render_now(nullptr),
      action_finish(nullptr),
      terminate_(false),
      pause_(false),
      running_(false),
      started_(false),
      finished_(false),
      options_(options) {}

IncrementalMapperController::IncrementalMapperController(
    const OptionManager& options, class Reconstruction* initial_model)
    : IncrementalMapperController(options) {
  models_.emplace_back(initial_model);
}

void IncrementalMapperController::Stop() {
  {
    QMutexLocker control_locker(&control_mutex_);
    terminate_ = true;
    running_ = false;
    finished_ = true;
  }
  Resume();
}

void IncrementalMapperController::Pause() {
  QMutexLocker control_locker(&control_mutex_);
  if (pause_) {
    return;
  }
  pause_ = true;
  running_ = false;
}

void IncrementalMapperController::Resume() {
  QMutexLocker control_locker(&control_mutex_);
  if (!pause_) {
    return;
  }
  pause_ = false;
  running_ = true;
  pause_condition_.wakeAll();
}

bool IncrementalMapperController::IsRunning() {
  QMutexLocker control_locker(&control_mutex_);
  return running_;
}

bool IncrementalMapperController::IsStarted() {
  QMutexLocker control_locker(&control_mutex_);
  return started_;
}

bool IncrementalMapperController::IsPaused() {
  QMutexLocker control_locker(&control_mutex_);
  return pause_;
}

bool IncrementalMapperController::IsFinished() { return finished_; }

size_t IncrementalMapperController::AddModel() {
  const size_t model_idx = models_.size();
  models_.emplace_back(new class Reconstruction());
  return model_idx;
}

void IncrementalMapperController::Render() {
  {
    QMutexLocker control_locker(&control_mutex_);
    if (terminate_) {
      return;
    }
  }

  if (action_render != nullptr) {
    action_render->trigger();
  }
}

void IncrementalMapperController::RenderNow() {
  {
    QMutexLocker control_locker(&control_mutex_);
    if (terminate_) {
      return;
    }
  }

  if (action_render_now != nullptr) {
    action_render_now->trigger();
  }
}

void IncrementalMapperController::Finish() {
  {
    QMutexLocker control_locker(&control_mutex_);
    running_ = false;
    finished_ = true;
    if (terminate_) {
      return;
    }
  }

  if (action_finish != nullptr) {
    action_finish->trigger();
  }
}

void IncrementalMapperController::run() {
  if (IsRunning()) {
    exit(0);
  }

  {
    QMutexLocker control_locker(&control_mutex_);
    terminate_ = false;
    pause_ = false;
    running_ = true;
    started_ = true;
    finished_ = false;
  }

  const MapperOptions& mapper_options = *options_.mapper_options;

  //////////////////////////////////////////////////////////////////////////////
  // Load data from database
  //////////////////////////////////////////////////////////////////////////////

  Timer total_timer;
  total_timer.Start();

  PrintHeading1("Loading database");

  DatabaseCache database_cache;

  {
    Database database;
    database.Open(*options_.database_path);
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
  const bool initial_model_given = !models_.empty();

  for (int num_trials = 0; num_trials < mapper_options.init_num_trials;
       ++num_trials) {
    {
      QMutexLocker control_locker(&control_mutex_);
      if (pause_ && !terminate_) {
        total_timer.Pause();
        pause_condition_.wait(&control_mutex_);
        total_timer.Resume();
      } else if (terminate_) {
        break;
      }
    }

    if (!initial_model_given || num_trials > 0) {
      AddModel();
    }

    const size_t model_idx = initial_model_given ? 0 : NumModels() - 1;
    Reconstruction& reconstruction = Model(model_idx);
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
          models_.pop_back();
          break;
        }
      }

      PrintHeading1("Initializing with images #" + std::to_string(image_id1) +
                    " and #" + std::to_string(image_id2));
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
        models_.pop_back();
        continue;
      }

      if (mapper_options.extract_colors) {
        ExtractColors(*options_.image_path, image_id1, &reconstruction);
      }
    }

    RenderNow();

    ////////////////////////////////////////////////////////////////////////////
    // Incremental mapping
    ////////////////////////////////////////////////////////////////////////////

    size_t prev_num_reg_images = reconstruction.NumRegImages();
    size_t prev_num_points = reconstruction.NumPoints3D();
    int num_global_bas = 1;

    bool reg_next_success = true;

    while (reg_next_success) {
      {
        QMutexLocker control_locker(&control_mutex_);
        if (pause_) {
          total_timer.Pause();
          pause_condition_.wait(&control_mutex_);
          total_timer.Resume();
        }
        if (terminate_) {
          break;
        }
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

        PrintHeading1("Processing image #" + std::to_string(next_image_id) +
                      " (" + std::to_string(reconstruction.NumRegImages() + 1) +
                      ")");

        std::cout << "  => Image sees " << next_image.NumVisiblePoints3D()
                  << " / " << next_image.NumObservations() << " points."
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

          Render();

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

    {
      QMutexLocker control_locker(&control_mutex_);
      if (terminate_) {
        const bool kDiscardReconstruction = false;
        mapper.EndReconstruction(kDiscardReconstruction);
        break;
      }
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
      models_.pop_back();
    } else {
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
      RenderNow();
    }

    const size_t max_num_models =
        static_cast<size_t>(mapper_options.max_num_models);
    if (initial_model_given || !mapper_options.multiple_models ||
        models_.size() >= max_num_models ||
        mapper.NumTotalRegImages() >= database_cache.NumImages() - 1) {
      break;
    }
  }

  std::cout << std::endl;

  total_timer.PrintMinutes();

  RenderNow();
  Finish();

  exit(0);
}

InteractiveMapperController::InteractiveMapperController(
    const OptionManager& options)
    : options_(options) {}

void InteractiveMapperController::run() {
  Timer total_timer;
  total_timer.Start();

  std::string input;

  PrintCommandPrompt();

  while (std::getline(std::cin, input)) {
    auto commands = StringSplit(input, " ");

    for (auto& command : commands) {
      boost::trim(command);
    }

    if (commands.empty()) {
      continue;
    }

    bool handler_success = false;

    if (commands[0].empty()) {
      PrintCommandPrompt();
      continue;
    } else if (commands[0] == "help") {
      handler_success = HandleHelp(commands);
    } else if (commands[0] == "exit") {
      break;
    } else if (commands[0] == "stats") {
      handler_success = HandleStats(commands);
    } else if (commands[0] == "export") {
      handler_success = HandleExport(commands);
    } else if (commands[0] == "register") {
      handler_success = HandleRegister(commands);
    } else if (commands[0] == "adjust_bundle") {
      handler_success = HandleBundleAdjustment(commands);
    } else {
      std::cerr << "ERROR: Unkown command." << std::endl;
    }

    if (handler_success) {
      std::cout << "Command exit status: SUCCESS" << std::endl;
    } else {
      std::cerr << "Command exit status: FAILURE" << std::endl;
    }

    PrintCommandPrompt();
  }
}

void InteractiveMapperController::PrintCommandPrompt() {
  std::cout << ">> " << std::flush;
}

bool InteractiveMapperController::HandleHelp(
    const std::vector<std::string>& commands) {
  std::cout << "help" << std::endl;
  std::cout << "  Show this text" << std::endl;

  std::cout << std::endl;

  std::cout << "exit" << std::endl;
  std::cout << "  Exit the application" << std::endl;

  std::cout << std::endl;

  std::cout << "stats" << std::endl;
  std::cout << "  Print statistics" << std::endl;

  std::cout << std::endl;

  std::cout << "export <path>" << std::endl;
  std::cout << "  Export current model" << std::endl;
  std::cout << "export all <path>" << std::endl;
  std::cout << "  Export all models" << std::endl;

  std::cout << std::endl;

  std::cout << "register initial" << std::endl;
  std::cout << "  Automatically choose initial image pair" << std::endl;
  std::cout << "register initial <image_id1> <image_id2>" << std::endl;
  std::cout << "  Specific initial image pair" << std::endl;
  std::cout << "register next" << std::endl;
  std::cout << "  Automatically find next image" << std::endl;
  std::cout << "register <image_id>" << std::endl;
  std::cout << "  Specific next image" << std::endl;

  std::cout << std::endl;

  std::cout << "adjust_bundle local <image_id>" << std::endl;
  std::cout << "  Local bundle adjustment" << std::endl;
  std::cout << "adjust_bundle global" << std::endl;
  std::cout << "  Global bundle adjustment" << std::endl;

  std::cout << std::endl;

  return true;
}

bool InteractiveMapperController::HandleStats(
    const std::vector<std::string>& commands) {
  return true;
}

bool InteractiveMapperController::HandleExport(
    const std::vector<std::string>& commands) {
  return false;
}

bool InteractiveMapperController::HandleRegister(
    const std::vector<std::string>& commands) {
  return false;
}

bool InteractiveMapperController::HandleBundleAdjustment(
    const std::vector<std::string>& commands) {
  return false;
}

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options)
    : reconstruction(nullptr), options_(options), running_(false) {}

bool BundleAdjustmentController::IsRunning() {
  QMutexLocker locker(&mutex_);
  return running_;
}

void BundleAdjustmentController::run() {
  CHECK_NOTNULL(reconstruction);

  if (IsRunning()) {
    exit(0);
  }

  {
    QMutexLocker locker(&mutex_);
    running_ = true;
  }

  Timer timer;
  timer.Start();

  PrintHeading1("Global bundle adjustment");

  const std::vector<image_t>& reg_image_ids = reconstruction->RegImageIds();

  if (reg_image_ids.size() < 2) {
    exit(0);
  }

  // Avoid degeneracies in bundle adjustment.
  reconstruction->FilterObservationsWithNegativeDepth();

  BundleAdjuster::Options ba_options = options_.ba_options->Options();
  ba_options.solver_options.minimizer_progress_to_stdout = true;

  // Configure bundle adjustment.
  BundleAdjustmentConfiguration ba_config;
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

  timer.PrintMinutes();

  if (action_finish != nullptr) {
    action_finish->trigger();
  }

  {
    QMutexLocker locker(&mutex_);
    running_ = false;
  }

  exit(0);
}

}  // namespace colmap
