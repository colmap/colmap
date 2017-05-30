// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
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

#include "controllers/automatic_reconstruction.h"

#include "base/feature_extraction.h"
#include "base/feature_matching.h"
#include "base/undistortion.h"
#include "controllers/incremental_mapper.h"
#include "mvs/fusion.h"
#include "mvs/patch_match.h"
#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

AutomaticReconstructionController::AutomaticReconstructionController(
    const Options& options, ReconstructionManager* reconstruction_manager)
    : options_(options),
      reconstruction_manager_(reconstruction_manager),
      active_thread_(nullptr) {
  CHECK(ExistsDir(options_.workspace_path));
  CHECK(ExistsDir(options_.image_path));
  CHECK_NOTNULL(reconstruction_manager_);

  option_manager_.AddAllOptions();

  *option_manager_.image_path = options_.image_path;
  *option_manager_.database_path =
      JoinPaths(options_.workspace_path, "database.db");

  if (options_.data_type == DataType::VIDEO) {
    option_manager_.InitForVideoData();
  } else if (options_.data_type == DataType::INDIVIDUAL) {
    option_manager_.InitForIndividualData();
  } else if (options_.data_type == DataType::INTERNET) {
    option_manager_.InitForInternetData();
  } else {
    LOG(FATAL) << "Data type not supported";
  }

  if (options_.quality == Quality::LOW) {
    option_manager_.sift_extraction->max_image_size = 1000;
    option_manager_.sequential_matching->loop_detection_num_images /= 2;
    option_manager_.vocab_tree_matching->num_images /= 2;
    option_manager_.mapper->ba_local_max_num_iterations /= 2;
    option_manager_.mapper->ba_global_max_num_iterations /= 2;
    option_manager_.mapper->ba_global_images_ratio *= 1.2;
    option_manager_.mapper->ba_global_points_ratio *= 1.2;
    option_manager_.mapper->ba_global_max_refinements = 2;
    option_manager_.dense_stereo->max_image_size = 1000;
    option_manager_.dense_stereo->window_radius = 4;
    option_manager_.dense_stereo->num_samples /= 2;
    option_manager_.dense_stereo->num_iterations = 3;
    option_manager_.dense_stereo->geom_consistency = false;
    option_manager_.dense_fusion->check_num_images /= 2;
    option_manager_.dense_fusion->max_image_size = 1000;
  } else if (options_.quality == Quality::MEDIUM) {
    option_manager_.sift_extraction->max_image_size = 1600;
    option_manager_.sequential_matching->loop_detection_num_images /= 1.5;
    option_manager_.vocab_tree_matching->num_images /= 1.5;
    option_manager_.mapper->ba_local_max_num_iterations /= 1.5;
    option_manager_.mapper->ba_global_max_num_iterations /= 1.5;
    option_manager_.mapper->ba_global_images_ratio *= 1.1;
    option_manager_.mapper->ba_global_points_ratio *= 1.1;
    option_manager_.mapper->ba_global_max_refinements = 2;
    option_manager_.dense_stereo->max_image_size = 1600;
    option_manager_.dense_stereo->window_radius = 5;
    option_manager_.dense_stereo->num_samples /= 1.5;
    option_manager_.dense_stereo->num_iterations = 5;
    option_manager_.dense_stereo->geom_consistency = false;
    option_manager_.dense_fusion->check_num_images /= 1.5;
    option_manager_.dense_fusion->max_image_size = 1600;
  }  // else: high quality is the default.

  ImageReader::Options reader_options = *option_manager_.image_reader;
  reader_options.database_path = *option_manager_.database_path;
  reader_options.image_path = *option_manager_.image_path;
  reader_options.single_camera = options_.single_camera;

  option_manager_.sift_matching->use_gpu = options_.use_gpu;

  if (options_.use_gpu) {
    if (!options_.use_opengl) {
      option_manager_.sift_gpu_extraction->index = 0;
    }

    feature_extractor_.reset(new SiftGPUFeatureExtractor(
        reader_options, *option_manager_.sift_extraction,
        *option_manager_.sift_gpu_extraction));
  } else {
    feature_extractor_.reset(new SiftCPUFeatureExtractor(
        reader_options, *option_manager_.sift_extraction,
        *option_manager_.sift_cpu_extraction));
  }

  exhaustive_matcher_.reset(new ExhaustiveFeatureMatcher(
      *option_manager_.exhaustive_matching, *option_manager_.sift_matching,
      *option_manager_.database_path));

  if (!options_.vocab_tree_path.empty()) {
    option_manager_.sequential_matching->loop_detection = true;
    option_manager_.sequential_matching->vocab_tree_path =
        options_.vocab_tree_path;
  }

  sequential_matcher_.reset(new SequentialFeatureMatcher(
      *option_manager_.sequential_matching, *option_manager_.sift_matching,
      *option_manager_.database_path));

  if (!options_.vocab_tree_path.empty()) {
    option_manager_.vocab_tree_matching->vocab_tree_path =
        options_.vocab_tree_path;
    vocab_tree_matcher_.reset(new VocabTreeFeatureMatcher(
        *option_manager_.vocab_tree_matching, *option_manager_.sift_matching,
        *option_manager_.database_path));
  }
}

void AutomaticReconstructionController::Stop() {
  if (active_thread_ != nullptr) {
    active_thread_->Stop();
  }
  Thread::Stop();
}

void AutomaticReconstructionController::Run() {
  if (IsStopped()) {
    return;
  }

  RunFeatureExtraction();

  if (IsStopped()) {
    return;
  }

  RunFeatureMatching();

  if (IsStopped()) {
    return;
  }

  if (options_.sparse) {
    RunSparseMapper();
  }

  if (IsStopped()) {
    return;
  }

  if (options_.dense) {
    RunDenseMapper();
  }
}

void AutomaticReconstructionController::RunFeatureExtraction() {
  CHECK(feature_extractor_);
  active_thread_ = feature_extractor_.get();
  feature_extractor_->Start();
  feature_extractor_->Wait();
  feature_extractor_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunFeatureMatching() {
  Thread* matcher = nullptr;
  if (options_.data_type == DataType::VIDEO) {
    matcher = sequential_matcher_.get();
  } else if (options_.data_type == DataType::INDIVIDUAL ||
             options_.data_type == DataType::INTERNET) {
    Database database(*option_manager_.database_path);
    const size_t num_images = database.NumImages();
    if (options_.vocab_tree_path.empty() || num_images < 200) {
      matcher = exhaustive_matcher_.get();
    } else {
      matcher = vocab_tree_matcher_.get();
    }
  }

  CHECK(matcher);
  active_thread_ = matcher;
  matcher->Start();
  matcher->Wait();
  exhaustive_matcher_.reset();
  sequential_matcher_.reset();
  vocab_tree_matcher_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunSparseMapper() {
  const auto sparse_path = JoinPaths(options_.workspace_path, "sparse");
  if (ExistsDir(sparse_path)) {
    auto dir_list = GetDirList(sparse_path);
    std::sort(dir_list.begin(), dir_list.end());
    if (dir_list.size() > 0) {
      std::cout << std::endl
                << "WARNING: Skipping sparse reconstruction because it is "
                   "already computed"
                << std::endl;
      for (const auto& dir : dir_list) {
        reconstruction_manager_->Read(dir);
      }
      return;
    }
  }

  IncrementalMapperController mapper(
      option_manager_.mapper.get(), *option_manager_.image_path,
      *option_manager_.database_path, reconstruction_manager_);
  active_thread_ = &mapper;
  mapper.Start();
  mapper.Wait();
  active_thread_ = nullptr;

  CreateDirIfNotExists(sparse_path);
  reconstruction_manager_->Write(sparse_path, &option_manager_);
}

void AutomaticReconstructionController::RunDenseMapper() {
#ifndef CUDA_ENABLED
  std::cout
      << std::endl
      << "WARNING: Skipping dense reconstruction because CUDA is not available"
      << std::endl;
  return;
#endif

  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "dense"));

  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    if (IsStopped()) {
      return;
    }

    const std::string dense_path =
        JoinPaths(options_.workspace_path, "dense", std::to_string(i));
    const std::string fused_path = JoinPaths(dense_path, "fused.ply");
    const std::string meshed_path = JoinPaths(dense_path, "meshed.ply");

    if (ExistsFile(fused_path) && ExistsFile(meshed_path)) {
      continue;
    }

    // Image undistortion

    if (!ExistsDir(dense_path)) {
      CreateDirIfNotExists(dense_path);

      UndistortCameraOptions undistortion_options;
      undistortion_options.max_image_size =
          option_manager_.dense_stereo->max_image_size;
      COLMAPUndistorter undistorter(undistortion_options,
                                    reconstruction_manager_->Get(i),
                                    *option_manager_.image_path, dense_path);
      active_thread_ = &undistorter;
      undistorter.Start();
      undistorter.Wait();
      active_thread_ = nullptr;
    }

    if (IsStopped()) {
      return;
    }

    // Dense stereo

    {
      mvs::PatchMatchController patch_match_controller(
          *option_manager_.dense_stereo, dense_path, "COLMAP", "");
      active_thread_ = &patch_match_controller;
      patch_match_controller.Start();
      patch_match_controller.Wait();
      active_thread_ = nullptr;
    }

    if (IsStopped()) {
      return;
    }

    // Dense fusion

    if (!ExistsFile(fused_path)) {
      mvs::StereoFusion fuser(
          *option_manager_.dense_fusion, dense_path, "COLMAP",
          options_.quality == Quality::HIGH ? "geometric" : "photometric");
      active_thread_ = &fuser;
      fuser.Start();
      fuser.Wait();
      active_thread_ = nullptr;

      std::cout << "Writing output: " << fused_path << std::endl;
      WritePlyBinary(fused_path, fuser.GetFusedPoints());
    }

    if (IsStopped()) {
      return;
    }

    // Dense meshing

    if (!ExistsFile(meshed_path)) {
      mvs::PoissonReconstruction(*option_manager_.dense_meshing, fused_path,
                                 meshed_path);
    }
  }
}

}  // namespace colmap
