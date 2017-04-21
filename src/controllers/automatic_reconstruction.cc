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
    const Options& options)
    : options_(options) {
  CHECK(ExistsDir(options_.workspace_path));
  CHECK(ExistsDir(options_.image_path));

  option_manager_.AddAllOptions();

  *option_manager_.image_path = options_.image_path;
  *option_manager_.database_path =
      JoinPaths(options_.workspace_path, "database.db");

  if (options_.data_type == DataType::VIDEO) {
    option_manager_.InitForVideoData();
  } else if (options_.data_type == DataType::DSLR) {
    option_manager_.InitForDSLRData();
  } else if (options_.data_type == DataType::INTERNET) {
    option_manager_.InitForInternetData();
  } else {
    LOG(FATAL) << "Data type not supported";
  }

  if (!options_.high_quality) {
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
  }

  ImageReader::Options reader_options = *option_manager_.image_reader;
  reader_options.database_path = *option_manager_.database_path;
  reader_options.image_path = *option_manager_.image_path;

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

void AutomaticReconstructionController::Run() {
  RunFeatureExtraction();
  RunFeatureMatching();
  if (options_.sparse) {
    RunSparseMapper();
  }
  if (options_.dense) {
    RunDenseMapper();
  }
}

void AutomaticReconstructionController::RunFeatureExtraction() {
  CHECK(feature_extractor_);
  feature_extractor_->Start();
  feature_extractor_->Wait();
}

void AutomaticReconstructionController::RunFeatureMatching() {
  Thread* matcher = nullptr;
  if (options_.data_type == DataType::VIDEO) {
    matcher = sequential_matcher_.get();
  } else if (options_.data_type == DataType::DSLR ||
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
  matcher->Start();
  matcher->Wait();
}

void AutomaticReconstructionController::RunSparseMapper() {
  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "sparse"));

  IncrementalMapperController mapper(
      option_manager_.mapper.get(), *option_manager_.image_path,
      *option_manager_.database_path, &reconstruction_manager_);
  mapper.Start();
  mapper.Wait();

  reconstruction_manager_.Write(JoinPaths(options_.workspace_path, "sparse"),
                                &option_manager_);
}

void AutomaticReconstructionController::RunDenseMapper() {
#ifndef CUDA_ENABLED
  std::cout
      << "WARNING: Skipping dense reconstruction because CUDA is not available"
      << std::endl;
  return;
#endif

  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "dense"));

  for (size_t i = 0; i < reconstruction_manager_.Size(); ++i) {
    const std::string dense_path =
        JoinPaths(options_.workspace_path, "dense", std::to_string(i));
    CreateDirIfNotExists(dense_path);

    // Image undistortion.

    UndistortCameraOptions undistortion_options;
    undistortion_options.max_image_size =
        option_manager_.dense_stereo->max_image_size;
    COLMAPUndistorter undistorter(undistortion_options,
                                  reconstruction_manager_.Get(i),
                                  *option_manager_.image_path, dense_path);
    undistorter.Start();
    undistorter.Wait();

    // Dense stereo

    if (options_.high_quality) {
      // Photometric stereo.
      option_manager_.dense_stereo->filter = false;
      option_manager_.dense_stereo->geom_consistency = false;
      mvs::PatchMatchController photometric_patch_match_controller(
          *option_manager_.dense_stereo, dense_path, "COLMAP", "");
      photometric_patch_match_controller.Start();
      photometric_patch_match_controller.Wait();
      // Geometric stereo.
      option_manager_.dense_stereo->filter = true;
      option_manager_.dense_stereo->geom_consistency = true;
      mvs::PatchMatchController geometric_patch_match_controller(
          *option_manager_.dense_stereo, dense_path, "COLMAP", "");
      geometric_patch_match_controller.Start();
      geometric_patch_match_controller.Wait();
    } else {
      option_manager_.dense_stereo->filter = true;
      option_manager_.dense_stereo->geom_consistency = false;
      mvs::PatchMatchController patch_match_controller(
          *option_manager_.dense_stereo, dense_path, "COLMAP", "");
      patch_match_controller.Start();
      patch_match_controller.Wait();
    }

    // Dense fusion.

    std::string fusion_input_type;
    if (options_.high_quality) {
      fusion_input_type = "geometric";
    } else {
      fusion_input_type = "photometric";
    }

    mvs::StereoFusion fuser(*option_manager_.dense_fusion, dense_path, "COLMAP",
                            fusion_input_type);
    fuser.Start();
    fuser.Wait();
    std::cout << "Writing output: " << JoinPaths(dense_path, "fused.ply")
              << std::endl;
    WritePlyBinary(JoinPaths(dense_path, "fused.ply"), fuser.GetFusedPoints());

    // Dense meshing.

    CHECK(mvs::PoissonReconstruction(*option_manager_.dense_meshing,
                                     JoinPaths(dense_path, "fused.ply"),
                                     JoinPaths(dense_path, "meshed.ply")));
  }
}

}  // namespace colmap
