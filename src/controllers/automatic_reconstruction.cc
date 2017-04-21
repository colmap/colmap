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

#include <QApplication>

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
  CHECK(ExistsFile(options_.vocab_tree_path));

  option_manager_.AddAllOptions();

  *option_manager_.image_path = JoinPaths(options_.image_path);
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
  ImageReader::Options reader_options = *option_manager_.image_reader;
  reader_options.database_path = *option_manager_.database_path;
  reader_options.image_path = *option_manager_.image_path;

  std::unique_ptr<Thread> feature_extractor;
  if (options_.use_gpu) {
    feature_extractor.reset(new SiftGPUFeatureExtractor(
        reader_options, *option_manager_.sift_extraction,
        *option_manager_.sift_gpu_extraction));
  } else {
    feature_extractor.reset(new SiftCPUFeatureExtractor(
        reader_options, *option_manager_.sift_extraction,
        *option_manager_.sift_cpu_extraction));
  }

  CHECK(feature_extractor);

  if (options_.use_gpu) {
    RunThreadWithOpenGLContext(feature_extractor.get());
  } else {
    feature_extractor->Start();
    feature_extractor->Wait();
  }
}

void AutomaticReconstructionController::RunFeatureMatching() {
  std::unique_ptr<Thread> feature_matcher;
  if (options_.data_type == DataType::VIDEO) {
    option_manager_.sequential_matching->loop_detection = true;
    option_manager_.sequential_matching->vocab_tree_path =
        options_.vocab_tree_path;
    feature_matcher.reset(new SequentialFeatureMatcher(
        *option_manager_.sequential_matching, *option_manager_.sift_matching,
        *option_manager_.database_path));
  } else if (options_.data_type == DataType::DSLR ||
             options_.data_type == DataType::INTERNET) {
    Database database(*option_manager_.database_path);
    const size_t num_images = database.NumImages();
    if (num_images < 200) {
      feature_matcher.reset(new ExhaustiveFeatureMatcher(
          *option_manager_.exhaustive_matching, *option_manager_.sift_matching,
          *option_manager_.database_path));
    } else {
      option_manager_.vocab_tree_matching->vocab_tree_path =
          options_.vocab_tree_path;
      feature_matcher.reset(new VocabTreeFeatureMatcher(
          *option_manager_.vocab_tree_matching, *option_manager_.sift_matching,
          *option_manager_.database_path));
    }
  }

  CHECK(feature_matcher);

  if (options_.use_gpu && options_.use_opengl) {
    RunThreadWithOpenGLContext(feature_matcher.get());
  } else {
    feature_matcher->Start();
    feature_matcher->Wait();
  }
}

void AutomaticReconstructionController::RunSparseMapper() {
  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "sparse"));

  IncrementalMapperController mapper(
      option_manager_.mapper.get(), *option_manager_.image_path,
      *option_manager_.database_path, &reconstruction_manager_);
  mapper.Start();
  mapper.Wait();

  for (size_t i = 0; i < reconstruction_manager_.Size(); ++i) {
    const std::string sparse_path =
        JoinPaths(options_.workspace_path, "sparse", std::to_string(i));
    CreateDirIfNotExists(sparse_path);
    reconstruction_manager_.Get(i).Write(sparse_path);
  }
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
    COLMAPUndistorter undistorter(UndistortCameraOptions(),
                                  reconstruction_manager_.Get(i),
                                  *option_manager_.image_path, dense_path);
    mvs::PatchMatchController patch_match_controller(
        *option_manager_.dense_stereo, dense_path, "COLMAP", "");
    patch_match_controller.Start();
    patch_match_controller.Wait();

    mvs::StereoFusion fuser(*option_manager_.dense_fusion, dense_path, "COLMAP",
                            "photometric");
    fuser.Start();
    fuser.Wait();
    std::cout << "Writing output: " << JoinPaths(dense_path, "fused.ply")
              << std::endl;
    WritePlyBinary(JoinPaths(dense_path, "fused.ply"), fuser.GetFusedPoints());

    CHECK(mvs::PoissonReconstruction(*option_manager_.dense_meshing,
                                     JoinPaths(dense_path, "fused.ply"),
                                     JoinPaths(dense_path, "meshed.ply")));
  }
}

}  // namespace colmap
