// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/option_manager.h"

#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/incremental_mapper.h"
#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/pairing.h"
#include "colmap/feature/sift.h"
#include "colmap/math/random.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/ui/render_options.h"
#include "colmap/util/misc.h"
#include "colmap/util/version.h"

#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ini_parser.hpp>

namespace config = boost::program_options;

namespace colmap {

OptionManager::OptionManager(bool add_project_options) {
  project_path = std::make_shared<std::string>();
  database_path = std::make_shared<std::string>();
  image_path = std::make_shared<std::string>();

  image_reader = std::make_shared<ImageReaderOptions>();
  sift_extraction = std::make_shared<SiftExtractionOptions>();
  sift_matching = std::make_shared<SiftMatchingOptions>();
  two_view_geometry = std::make_shared<TwoViewGeometryOptions>();
  exhaustive_matching = std::make_shared<ExhaustiveMatchingOptions>();
  sequential_matching = std::make_shared<SequentialMatchingOptions>();
  vocab_tree_matching = std::make_shared<VocabTreeMatchingOptions>();
  spatial_matching = std::make_shared<SpatialMatchingOptions>();
  transitive_matching = std::make_shared<TransitiveMatchingOptions>();
  image_pairs_matching = std::make_shared<ImagePairsMatchingOptions>();
  bundle_adjustment = std::make_shared<BundleAdjustmentOptions>();
  mapper = std::make_shared<IncrementalPipelineOptions>();
  patch_match_stereo = std::make_shared<mvs::PatchMatchOptions>();
  stereo_fusion = std::make_shared<mvs::StereoFusionOptions>();
  poisson_meshing = std::make_shared<mvs::PoissonMeshingOptions>();
  delaunay_meshing = std::make_shared<mvs::DelaunayMeshingOptions>();
  render = std::make_shared<RenderOptions>();

  Reset();

  desc_->add_options()("help,h", "");

  AddRandomOptions();
  AddLogOptions();

  if (add_project_options) {
    desc_->add_options()("project_path", config::value<std::string>());
  }
}

void OptionManager::ModifyForIndividualData() {
  mapper->min_focal_length_ratio = 0.1;
  mapper->max_focal_length_ratio = 10;
  mapper->max_extra_param = std::numeric_limits<double>::max();
}

void OptionManager::ModifyForVideoData() {
  const bool kResetPaths = false;
  ResetOptions(kResetPaths);
  mapper->mapper.init_min_tri_angle /= 2;
  mapper->ba_global_images_ratio = 1.4;
  mapper->ba_global_points_ratio = 1.4;
  mapper->min_focal_length_ratio = 0.1;
  mapper->max_focal_length_ratio = 10;
  mapper->max_extra_param = std::numeric_limits<double>::max();
  stereo_fusion->min_num_pixels = 15;
}

void OptionManager::ModifyForInternetData() {
  stereo_fusion->min_num_pixels = 10;
}

void OptionManager::ModifyForLowQuality() {
  sift_extraction->max_image_size = 1000;
  sift_extraction->max_num_features = 2048;
  sequential_matching->loop_detection_num_images /= 2;
  vocab_tree_matching->num_images /= 2;
  mapper->ba_local_max_num_iterations /= 2;
  mapper->ba_global_max_num_iterations /= 2;
  mapper->ba_global_images_ratio *= 1.2;
  mapper->ba_global_points_ratio *= 1.2;
  mapper->ba_global_max_refinements = 2;
  patch_match_stereo->max_image_size = 1000;
  patch_match_stereo->window_radius = 4;
  patch_match_stereo->window_step = 2;
  patch_match_stereo->num_samples /= 2;
  patch_match_stereo->num_iterations = 3;
  patch_match_stereo->geom_consistency = false;
  stereo_fusion->check_num_images /= 2;
  stereo_fusion->max_image_size = 1000;
}

void OptionManager::ModifyForMediumQuality() {
  sift_extraction->max_image_size = 1600;
  sift_extraction->max_num_features = 4096;
  sequential_matching->loop_detection_num_images /= 1.5;
  vocab_tree_matching->num_images /= 1.5;
  mapper->ba_local_max_num_iterations /= 1.5;
  mapper->ba_global_max_num_iterations /= 1.5;
  mapper->ba_global_images_ratio *= 1.1;
  mapper->ba_global_points_ratio *= 1.1;
  mapper->ba_global_max_refinements = 2;
  patch_match_stereo->max_image_size = 1600;
  patch_match_stereo->window_radius = 4;
  patch_match_stereo->window_step = 2;
  patch_match_stereo->num_samples /= 1.5;
  patch_match_stereo->num_iterations = 5;
  patch_match_stereo->geom_consistency = false;
  stereo_fusion->check_num_images /= 1.5;
  stereo_fusion->max_image_size = 1600;
}

void OptionManager::ModifyForHighQuality() {
  sift_extraction->estimate_affine_shape = true;
  sift_extraction->max_image_size = 2400;
  sift_extraction->max_num_features = 8192;
  sift_matching->guided_matching = true;
  mapper->ba_local_max_num_iterations = 30;
  mapper->ba_local_max_refinements = 3;
  mapper->ba_global_max_num_iterations = 75;
  patch_match_stereo->max_image_size = 2400;
  stereo_fusion->max_image_size = 2400;
}

void OptionManager::ModifyForExtremeQuality() {
  // Most of the options are set to extreme quality by default.
  sift_extraction->estimate_affine_shape = true;
  sift_extraction->domain_size_pooling = true;
  sift_matching->guided_matching = true;
  mapper->ba_local_max_num_iterations = 40;
  mapper->ba_local_max_refinements = 3;
  mapper->ba_global_max_num_iterations = 100;
}

void OptionManager::AddAllOptions() {
  AddLogOptions();
  AddRandomOptions();
  AddDatabaseOptions();
  AddImageOptions();
  AddExtractionOptions();
  AddMatchingOptions();
  AddExhaustiveMatchingOptions();
  AddSequentialMatchingOptions();
  AddVocabTreeMatchingOptions();
  AddSpatialMatchingOptions();
  AddTransitiveMatchingOptions();
  AddImagePairsMatchingOptions();
  AddBundleAdjustmentOptions();
  AddMapperOptions();
  AddPatchMatchStereoOptions();
  AddStereoFusionOptions();
  AddPoissonMeshingOptions();
  AddDelaunayMeshingOptions();
  AddRenderOptions();
}

void OptionManager::AddLogOptions() {
  if (added_log_options_) {
    return;
  }
  added_log_options_ = true;

  AddAndRegisterDefaultOption("log_to_stderr", &FLAGS_logtostderr);
  AddAndRegisterDefaultOption("log_level", &FLAGS_v);
}

void OptionManager::AddRandomOptions() {
  if (added_random_options_) {
    return;
  }
  added_random_options_ = true;

  AddAndRegisterDefaultOption("random_seed", &kDefaultPRNGSeed);
}

void OptionManager::AddDatabaseOptions() {
  if (added_database_options_) {
    return;
  }
  added_database_options_ = true;

  AddAndRegisterRequiredOption("database_path", database_path.get());
}

void OptionManager::AddImageOptions() {
  if (added_image_options_) {
    return;
  }
  added_image_options_ = true;

  AddAndRegisterRequiredOption("image_path", image_path.get());
}

void OptionManager::AddExtractionOptions() {
  if (added_extraction_options_) {
    return;
  }
  added_extraction_options_ = true;

  AddAndRegisterDefaultOption("ImageReader.mask_path",
                              &image_reader->mask_path);
  AddAndRegisterDefaultOption("ImageReader.camera_model",
                              &image_reader->camera_model);
  AddAndRegisterDefaultOption("ImageReader.single_camera",
                              &image_reader->single_camera);
  AddAndRegisterDefaultOption("ImageReader.single_camera_per_folder",
                              &image_reader->single_camera_per_folder);
  AddAndRegisterDefaultOption("ImageReader.single_camera_per_image",
                              &image_reader->single_camera_per_image);
  AddAndRegisterDefaultOption("ImageReader.existing_camera_id",
                              &image_reader->existing_camera_id);
  AddAndRegisterDefaultOption("ImageReader.camera_params",
                              &image_reader->camera_params);
  AddAndRegisterDefaultOption("ImageReader.default_focal_length_factor",
                              &image_reader->default_focal_length_factor);
  AddAndRegisterDefaultOption("ImageReader.camera_mask_path",
                              &image_reader->camera_mask_path);

  AddAndRegisterDefaultOption("SiftExtraction.num_threads",
                              &sift_extraction->num_threads);
  AddAndRegisterDefaultOption("SiftExtraction.use_gpu",
                              &sift_extraction->use_gpu);
  AddAndRegisterDefaultOption("SiftExtraction.gpu_index",
                              &sift_extraction->gpu_index);
  AddAndRegisterDefaultOption("SiftExtraction.max_image_size",
                              &sift_extraction->max_image_size);
  AddAndRegisterDefaultOption("SiftExtraction.max_num_features",
                              &sift_extraction->max_num_features);
  AddAndRegisterDefaultOption("SiftExtraction.first_octave",
                              &sift_extraction->first_octave);
  AddAndRegisterDefaultOption("SiftExtraction.num_octaves",
                              &sift_extraction->num_octaves);
  AddAndRegisterDefaultOption("SiftExtraction.octave_resolution",
                              &sift_extraction->octave_resolution);
  AddAndRegisterDefaultOption("SiftExtraction.peak_threshold",
                              &sift_extraction->peak_threshold);
  AddAndRegisterDefaultOption("SiftExtraction.edge_threshold",
                              &sift_extraction->edge_threshold);
  AddAndRegisterDefaultOption("SiftExtraction.estimate_affine_shape",
                              &sift_extraction->estimate_affine_shape);
  AddAndRegisterDefaultOption("SiftExtraction.max_num_orientations",
                              &sift_extraction->max_num_orientations);
  AddAndRegisterDefaultOption("SiftExtraction.upright",
                              &sift_extraction->upright);
  AddAndRegisterDefaultOption("SiftExtraction.domain_size_pooling",
                              &sift_extraction->domain_size_pooling);
  AddAndRegisterDefaultOption("SiftExtraction.dsp_min_scale",
                              &sift_extraction->dsp_min_scale);
  AddAndRegisterDefaultOption("SiftExtraction.dsp_max_scale",
                              &sift_extraction->dsp_max_scale);
  AddAndRegisterDefaultOption("SiftExtraction.dsp_num_scales",
                              &sift_extraction->dsp_num_scales);
}

void OptionManager::AddMatchingOptions() {
  if (added_match_options_) {
    return;
  }
  added_match_options_ = true;

  AddAndRegisterDefaultOption("SiftMatching.num_threads",
                              &sift_matching->num_threads);
  AddAndRegisterDefaultOption("SiftMatching.use_gpu", &sift_matching->use_gpu);
  AddAndRegisterDefaultOption("SiftMatching.gpu_index",
                              &sift_matching->gpu_index);
  AddAndRegisterDefaultOption("SiftMatching.max_ratio",
                              &sift_matching->max_ratio);
  AddAndRegisterDefaultOption("SiftMatching.max_distance",
                              &sift_matching->max_distance);
  AddAndRegisterDefaultOption("SiftMatching.cross_check",
                              &sift_matching->cross_check);
  AddAndRegisterDefaultOption("SiftMatching.guided_matching",
                              &sift_matching->guided_matching);
  AddAndRegisterDefaultOption("SiftMatching.max_num_matches",
                              &sift_matching->max_num_matches);
  AddAndRegisterDefaultOption("TwoViewGeometry.min_num_inliers",
                              &two_view_geometry->min_num_inliers);
  AddAndRegisterDefaultOption("TwoViewGeometry.multiple_models",
                              &two_view_geometry->multiple_models);
  AddAndRegisterDefaultOption("TwoViewGeometry.compute_relative_pose",
                              &two_view_geometry->compute_relative_pose);
  AddAndRegisterDefaultOption("TwoViewGeometry.max_error",
                              &two_view_geometry->ransac_options.max_error);
  AddAndRegisterDefaultOption("TwoViewGeometry.confidence",
                              &two_view_geometry->ransac_options.confidence);
  AddAndRegisterDefaultOption(
      "TwoViewGeometry.max_num_trials",
      &two_view_geometry->ransac_options.max_num_trials);
  AddAndRegisterDefaultOption(
      "TwoViewGeometry.min_inlier_ratio",
      &two_view_geometry->ransac_options.min_inlier_ratio);
}

void OptionManager::AddExhaustiveMatchingOptions() {
  if (added_exhaustive_match_options_) {
    return;
  }
  added_exhaustive_match_options_ = true;

  AddMatchingOptions();

  AddAndRegisterDefaultOption("ExhaustiveMatching.block_size",
                              &exhaustive_matching->block_size);
}

void OptionManager::AddSequentialMatchingOptions() {
  if (added_sequential_match_options_) {
    return;
  }
  added_sequential_match_options_ = true;

  AddMatchingOptions();

  AddAndRegisterDefaultOption("SequentialMatching.overlap",
                              &sequential_matching->overlap);
  AddAndRegisterDefaultOption("SequentialMatching.quadratic_overlap",
                              &sequential_matching->quadratic_overlap);
  AddAndRegisterDefaultOption("SequentialMatching.loop_detection",
                              &sequential_matching->loop_detection);
  AddAndRegisterDefaultOption("SequentialMatching.loop_detection_period",
                              &sequential_matching->loop_detection_period);
  AddAndRegisterDefaultOption("SequentialMatching.loop_detection_num_images",
                              &sequential_matching->loop_detection_num_images);
  AddAndRegisterDefaultOption(
      "SequentialMatching.loop_detection_num_nearest_neighbors",
      &sequential_matching->loop_detection_num_nearest_neighbors);
  AddAndRegisterDefaultOption("SequentialMatching.loop_detection_num_checks",
                              &sequential_matching->loop_detection_num_checks);
  AddAndRegisterDefaultOption(
      "SequentialMatching.loop_detection_num_images_after_verification",
      &sequential_matching->loop_detection_num_images_after_verification);
  AddAndRegisterDefaultOption(
      "SequentialMatching.loop_detection_max_num_features",
      &sequential_matching->loop_detection_max_num_features);
  AddAndRegisterDefaultOption("SequentialMatching.vocab_tree_path",
                              &sequential_matching->vocab_tree_path);
}

void OptionManager::AddVocabTreeMatchingOptions() {
  if (added_vocab_tree_match_options_) {
    return;
  }
  added_vocab_tree_match_options_ = true;

  AddMatchingOptions();

  AddAndRegisterDefaultOption("VocabTreeMatching.num_images",
                              &vocab_tree_matching->num_images);
  AddAndRegisterDefaultOption("VocabTreeMatching.num_nearest_neighbors",
                              &vocab_tree_matching->num_nearest_neighbors);
  AddAndRegisterDefaultOption("VocabTreeMatching.num_checks",
                              &vocab_tree_matching->num_checks);
  AddAndRegisterDefaultOption(
      "VocabTreeMatching.num_images_after_verification",
      &vocab_tree_matching->num_images_after_verification);
  AddAndRegisterDefaultOption("VocabTreeMatching.max_num_features",
                              &vocab_tree_matching->max_num_features);
  AddAndRegisterDefaultOption("VocabTreeMatching.vocab_tree_path",
                              &vocab_tree_matching->vocab_tree_path);
  AddAndRegisterDefaultOption("VocabTreeMatching.match_list_path",
                              &vocab_tree_matching->match_list_path);
}

void OptionManager::AddSpatialMatchingOptions() {
  if (added_spatial_match_options_) {
    return;
  }
  added_spatial_match_options_ = true;

  AddMatchingOptions();

  AddAndRegisterDefaultOption("SpatialMatching.ignore_z",
                              &spatial_matching->ignore_z);
  AddAndRegisterDefaultOption("SpatialMatching.max_num_neighbors",
                              &spatial_matching->max_num_neighbors);
  AddAndRegisterDefaultOption("SpatialMatching.max_distance",
                              &spatial_matching->max_distance);
}

void OptionManager::AddTransitiveMatchingOptions() {
  if (added_transitive_match_options_) {
    return;
  }
  added_transitive_match_options_ = true;

  AddMatchingOptions();

  AddAndRegisterDefaultOption("TransitiveMatching.batch_size",
                              &transitive_matching->batch_size);
  AddAndRegisterDefaultOption("TransitiveMatching.num_iterations",
                              &transitive_matching->num_iterations);
}

void OptionManager::AddImagePairsMatchingOptions() {
  if (added_image_pairs_match_options_) {
    return;
  }
  added_image_pairs_match_options_ = true;

  AddMatchingOptions();

  AddAndRegisterDefaultOption("ImagePairsMatching.block_size",
                              &image_pairs_matching->block_size);
}

void OptionManager::AddBundleAdjustmentOptions() {
  if (added_ba_options_) {
    return;
  }
  added_ba_options_ = true;

  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_num_iterations",
      &bundle_adjustment->solver_options.max_num_iterations);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_linear_solver_iterations",
      &bundle_adjustment->solver_options.max_linear_solver_iterations);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.function_tolerance",
      &bundle_adjustment->solver_options.function_tolerance);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.gradient_tolerance",
      &bundle_adjustment->solver_options.gradient_tolerance);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.parameter_tolerance",
      &bundle_adjustment->solver_options.parameter_tolerance);
  AddAndRegisterDefaultOption("BundleAdjustment.refine_focal_length",
                              &bundle_adjustment->refine_focal_length);
  AddAndRegisterDefaultOption("BundleAdjustment.refine_principal_point",
                              &bundle_adjustment->refine_principal_point);
  AddAndRegisterDefaultOption("BundleAdjustment.refine_extra_params",
                              &bundle_adjustment->refine_extra_params);
  AddAndRegisterDefaultOption("BundleAdjustment.refine_extrinsics",
                              &bundle_adjustment->refine_extrinsics);
  AddAndRegisterDefaultOption("BundleAdjustment.use_gpu",
                              &bundle_adjustment->use_gpu);
  AddAndRegisterDefaultOption("BundleAdjustment.gpu_index",
                              &bundle_adjustment->gpu_index);
  AddAndRegisterDefaultOption("BundleAdjustment.min_num_images_gpu_solver",
                              &bundle_adjustment->min_num_images_gpu_solver);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.min_num_residuals_for_cpu_multi_threading",
      &bundle_adjustment->min_num_residuals_for_cpu_multi_threading);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_num_images_direct_dense_cpu_solver",
      &bundle_adjustment->max_num_images_direct_dense_cpu_solver);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_num_images_direct_sparse_cpu_solver",
      &bundle_adjustment->max_num_images_direct_sparse_cpu_solver);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_num_images_direct_dense_gpu_solver",
      &bundle_adjustment->max_num_images_direct_dense_gpu_solver);
  AddAndRegisterDefaultOption(
      "BundleAdjustment.max_num_images_direct_sparse_gpu_solver",
      &bundle_adjustment->max_num_images_direct_sparse_gpu_solver);
}

void OptionManager::AddMapperOptions() {
  if (added_mapper_options_) {
    return;
  }
  added_mapper_options_ = true;

  AddAndRegisterDefaultOption("Mapper.min_num_matches",
                              &mapper->min_num_matches);
  AddAndRegisterDefaultOption("Mapper.ignore_watermarks",
                              &mapper->ignore_watermarks);
  AddAndRegisterDefaultOption("Mapper.multiple_models",
                              &mapper->multiple_models);
  AddAndRegisterDefaultOption("Mapper.max_num_models", &mapper->max_num_models);
  AddAndRegisterDefaultOption("Mapper.max_model_overlap",
                              &mapper->max_model_overlap);
  AddAndRegisterDefaultOption("Mapper.min_model_size", &mapper->min_model_size);
  AddAndRegisterDefaultOption("Mapper.init_image_id1", &mapper->init_image_id1);
  AddAndRegisterDefaultOption("Mapper.init_image_id2", &mapper->init_image_id2);
  AddAndRegisterDefaultOption("Mapper.init_num_trials",
                              &mapper->init_num_trials);
  AddAndRegisterDefaultOption("Mapper.extract_colors", &mapper->extract_colors);
  AddAndRegisterDefaultOption("Mapper.num_threads", &mapper->num_threads);
  AddAndRegisterDefaultOption("Mapper.min_focal_length_ratio",
                              &mapper->min_focal_length_ratio);
  AddAndRegisterDefaultOption("Mapper.max_focal_length_ratio",
                              &mapper->max_focal_length_ratio);
  AddAndRegisterDefaultOption("Mapper.max_extra_param",
                              &mapper->max_extra_param);
  AddAndRegisterDefaultOption("Mapper.ba_refine_focal_length",
                              &mapper->ba_refine_focal_length);
  AddAndRegisterDefaultOption("Mapper.ba_refine_principal_point",
                              &mapper->ba_refine_principal_point);
  AddAndRegisterDefaultOption("Mapper.ba_refine_extra_params",
                              &mapper->ba_refine_extra_params);
  AddAndRegisterDefaultOption("Mapper.ba_local_num_images",
                              &mapper->ba_local_num_images);
  AddAndRegisterDefaultOption("Mapper.ba_local_function_tolerance",
                              &mapper->ba_local_function_tolerance);
  AddAndRegisterDefaultOption("Mapper.ba_local_max_num_iterations",
                              &mapper->ba_local_max_num_iterations);
  AddAndRegisterDefaultOption("Mapper.ba_global_images_ratio",
                              &mapper->ba_global_images_ratio);
  AddAndRegisterDefaultOption("Mapper.ba_global_points_ratio",
                              &mapper->ba_global_points_ratio);
  AddAndRegisterDefaultOption("Mapper.ba_global_images_freq",
                              &mapper->ba_global_images_freq);
  AddAndRegisterDefaultOption("Mapper.ba_global_points_freq",
                              &mapper->ba_global_points_freq);
  AddAndRegisterDefaultOption("Mapper.ba_global_function_tolerance",
                              &mapper->ba_global_function_tolerance);
  AddAndRegisterDefaultOption("Mapper.ba_global_max_num_iterations",
                              &mapper->ba_global_max_num_iterations);
  AddAndRegisterDefaultOption("Mapper.ba_global_max_refinements",
                              &mapper->ba_global_max_refinements);
  AddAndRegisterDefaultOption("Mapper.ba_global_max_refinement_change",
                              &mapper->ba_global_max_refinement_change);
  AddAndRegisterDefaultOption("Mapper.ba_local_max_refinements",
                              &mapper->ba_local_max_refinements);
  AddAndRegisterDefaultOption("Mapper.ba_local_max_refinement_change",
                              &mapper->ba_local_max_refinement_change);
  AddAndRegisterDefaultOption("Mapper.ba_use_gpu", &mapper->ba_use_gpu);
  AddAndRegisterDefaultOption("Mapper.ba_gpu_index", &mapper->ba_gpu_index);
  AddAndRegisterDefaultOption(
      "Mapper.ba_min_num_residuals_for_cpu_multi_threading",
      &mapper->ba_min_num_residuals_for_cpu_multi_threading);
  AddAndRegisterDefaultOption("Mapper.snapshot_path", &mapper->snapshot_path);
  AddAndRegisterDefaultOption("Mapper.snapshot_images_freq",
                              &mapper->snapshot_images_freq);
  AddAndRegisterDefaultOption("Mapper.fix_existing_images",
                              &mapper->fix_existing_images);

  // IncrementalMapper.
  AddAndRegisterDefaultOption("Mapper.init_min_num_inliers",
                              &mapper->mapper.init_min_num_inliers);
  AddAndRegisterDefaultOption("Mapper.init_max_error",
                              &mapper->mapper.init_max_error);
  AddAndRegisterDefaultOption("Mapper.init_max_forward_motion",
                              &mapper->mapper.init_max_forward_motion);
  AddAndRegisterDefaultOption("Mapper.init_min_tri_angle",
                              &mapper->mapper.init_min_tri_angle);
  AddAndRegisterDefaultOption("Mapper.init_max_reg_trials",
                              &mapper->mapper.init_max_reg_trials);
  AddAndRegisterDefaultOption("Mapper.abs_pose_max_error",
                              &mapper->mapper.abs_pose_max_error);
  AddAndRegisterDefaultOption("Mapper.abs_pose_min_num_inliers",
                              &mapper->mapper.abs_pose_min_num_inliers);
  AddAndRegisterDefaultOption("Mapper.abs_pose_min_inlier_ratio",
                              &mapper->mapper.abs_pose_min_inlier_ratio);
  AddAndRegisterDefaultOption("Mapper.filter_max_reproj_error",
                              &mapper->mapper.filter_max_reproj_error);
  AddAndRegisterDefaultOption("Mapper.filter_min_tri_angle",
                              &mapper->mapper.filter_min_tri_angle);
  AddAndRegisterDefaultOption("Mapper.max_reg_trials",
                              &mapper->mapper.max_reg_trials);
  AddAndRegisterDefaultOption("Mapper.local_ba_min_tri_angle",
                              &mapper->mapper.local_ba_min_tri_angle);

  // IncrementalTriangulator.
  AddAndRegisterDefaultOption("Mapper.tri_max_transitivity",
                              &mapper->triangulation.max_transitivity);
  AddAndRegisterDefaultOption("Mapper.tri_create_max_angle_error",
                              &mapper->triangulation.create_max_angle_error);
  AddAndRegisterDefaultOption("Mapper.tri_continue_max_angle_error",
                              &mapper->triangulation.continue_max_angle_error);
  AddAndRegisterDefaultOption("Mapper.tri_merge_max_reproj_error",
                              &mapper->triangulation.merge_max_reproj_error);
  AddAndRegisterDefaultOption("Mapper.tri_complete_max_reproj_error",
                              &mapper->triangulation.complete_max_reproj_error);
  AddAndRegisterDefaultOption("Mapper.tri_complete_max_transitivity",
                              &mapper->triangulation.complete_max_transitivity);
  AddAndRegisterDefaultOption("Mapper.tri_re_max_angle_error",
                              &mapper->triangulation.re_max_angle_error);
  AddAndRegisterDefaultOption("Mapper.tri_re_min_ratio",
                              &mapper->triangulation.re_min_ratio);
  AddAndRegisterDefaultOption("Mapper.tri_re_max_trials",
                              &mapper->triangulation.re_max_trials);
  AddAndRegisterDefaultOption("Mapper.tri_min_angle",
                              &mapper->triangulation.min_angle);
  AddAndRegisterDefaultOption("Mapper.tri_ignore_two_view_tracks",
                              &mapper->triangulation.ignore_two_view_tracks);
}

void OptionManager::AddPatchMatchStereoOptions() {
  if (added_patch_match_stereo_options_) {
    return;
  }
  added_patch_match_stereo_options_ = true;

  AddAndRegisterDefaultOption("PatchMatchStereo.max_image_size",
                              &patch_match_stereo->max_image_size);
  AddAndRegisterDefaultOption("PatchMatchStereo.gpu_index",
                              &patch_match_stereo->gpu_index);
  AddAndRegisterDefaultOption("PatchMatchStereo.depth_min",
                              &patch_match_stereo->depth_min);
  AddAndRegisterDefaultOption("PatchMatchStereo.depth_max",
                              &patch_match_stereo->depth_max);
  AddAndRegisterDefaultOption("PatchMatchStereo.window_radius",
                              &patch_match_stereo->window_radius);
  AddAndRegisterDefaultOption("PatchMatchStereo.window_step",
                              &patch_match_stereo->window_step);
  AddAndRegisterDefaultOption("PatchMatchStereo.sigma_spatial",
                              &patch_match_stereo->sigma_spatial);
  AddAndRegisterDefaultOption("PatchMatchStereo.sigma_color",
                              &patch_match_stereo->sigma_color);
  AddAndRegisterDefaultOption("PatchMatchStereo.num_samples",
                              &patch_match_stereo->num_samples);
  AddAndRegisterDefaultOption("PatchMatchStereo.ncc_sigma",
                              &patch_match_stereo->ncc_sigma);
  AddAndRegisterDefaultOption("PatchMatchStereo.min_triangulation_angle",
                              &patch_match_stereo->min_triangulation_angle);
  AddAndRegisterDefaultOption("PatchMatchStereo.incident_angle_sigma",
                              &patch_match_stereo->incident_angle_sigma);
  AddAndRegisterDefaultOption("PatchMatchStereo.num_iterations",
                              &patch_match_stereo->num_iterations);
  AddAndRegisterDefaultOption("PatchMatchStereo.geom_consistency",
                              &patch_match_stereo->geom_consistency);
  AddAndRegisterDefaultOption(
      "PatchMatchStereo.geom_consistency_regularizer",
      &patch_match_stereo->geom_consistency_regularizer);
  AddAndRegisterDefaultOption("PatchMatchStereo.geom_consistency_max_cost",
                              &patch_match_stereo->geom_consistency_max_cost);
  AddAndRegisterDefaultOption("PatchMatchStereo.filter",
                              &patch_match_stereo->filter);
  AddAndRegisterDefaultOption("PatchMatchStereo.filter_min_ncc",
                              &patch_match_stereo->filter_min_ncc);
  AddAndRegisterDefaultOption(
      "PatchMatchStereo.filter_min_triangulation_angle",
      &patch_match_stereo->filter_min_triangulation_angle);
  AddAndRegisterDefaultOption("PatchMatchStereo.filter_min_num_consistent",
                              &patch_match_stereo->filter_min_num_consistent);
  AddAndRegisterDefaultOption(
      "PatchMatchStereo.filter_geom_consistency_max_cost",
      &patch_match_stereo->filter_geom_consistency_max_cost);
  AddAndRegisterDefaultOption("PatchMatchStereo.cache_size",
                              &patch_match_stereo->cache_size);
  AddAndRegisterDefaultOption("PatchMatchStereo.allow_missing_files",
                              &patch_match_stereo->allow_missing_files);
  AddAndRegisterDefaultOption("PatchMatchStereo.write_consistency_graph",
                              &patch_match_stereo->write_consistency_graph);
}

void OptionManager::AddStereoFusionOptions() {
  if (added_stereo_fusion_options_) {
    return;
  }
  added_stereo_fusion_options_ = true;

  AddAndRegisterDefaultOption("StereoFusion.mask_path",
                              &stereo_fusion->mask_path);
  AddAndRegisterDefaultOption("StereoFusion.num_threads",
                              &stereo_fusion->num_threads);
  AddAndRegisterDefaultOption("StereoFusion.max_image_size",
                              &stereo_fusion->max_image_size);
  AddAndRegisterDefaultOption("StereoFusion.min_num_pixels",
                              &stereo_fusion->min_num_pixels);
  AddAndRegisterDefaultOption("StereoFusion.max_num_pixels",
                              &stereo_fusion->max_num_pixels);
  AddAndRegisterDefaultOption("StereoFusion.max_traversal_depth",
                              &stereo_fusion->max_traversal_depth);
  AddAndRegisterDefaultOption("StereoFusion.max_reproj_error",
                              &stereo_fusion->max_reproj_error);
  AddAndRegisterDefaultOption("StereoFusion.max_depth_error",
                              &stereo_fusion->max_depth_error);
  AddAndRegisterDefaultOption("StereoFusion.max_normal_error",
                              &stereo_fusion->max_normal_error);
  AddAndRegisterDefaultOption("StereoFusion.check_num_images",
                              &stereo_fusion->check_num_images);
  AddAndRegisterDefaultOption("StereoFusion.cache_size",
                              &stereo_fusion->cache_size);
  AddAndRegisterDefaultOption("StereoFusion.use_cache",
                              &stereo_fusion->use_cache);
}

void OptionManager::AddPoissonMeshingOptions() {
  if (added_poisson_meshing_options_) {
    return;
  }
  added_poisson_meshing_options_ = true;

  AddAndRegisterDefaultOption("PoissonMeshing.point_weight",
                              &poisson_meshing->point_weight);
  AddAndRegisterDefaultOption("PoissonMeshing.depth", &poisson_meshing->depth);
  AddAndRegisterDefaultOption("PoissonMeshing.color", &poisson_meshing->color);
  AddAndRegisterDefaultOption("PoissonMeshing.trim", &poisson_meshing->trim);
  AddAndRegisterDefaultOption("PoissonMeshing.num_threads",
                              &poisson_meshing->num_threads);
}

void OptionManager::AddDelaunayMeshingOptions() {
  if (added_delaunay_meshing_options_) {
    return;
  }
  added_delaunay_meshing_options_ = true;

  AddAndRegisterDefaultOption("DelaunayMeshing.max_proj_dist",
                              &delaunay_meshing->max_proj_dist);
  AddAndRegisterDefaultOption("DelaunayMeshing.max_depth_dist",
                              &delaunay_meshing->max_depth_dist);
  AddAndRegisterDefaultOption("DelaunayMeshing.visibility_sigma",
                              &delaunay_meshing->visibility_sigma);
  AddAndRegisterDefaultOption("DelaunayMeshing.distance_sigma_factor",
                              &delaunay_meshing->distance_sigma_factor);
  AddAndRegisterDefaultOption("DelaunayMeshing.quality_regularization",
                              &delaunay_meshing->quality_regularization);
  AddAndRegisterDefaultOption("DelaunayMeshing.max_side_length_factor",
                              &delaunay_meshing->max_side_length_factor);
  AddAndRegisterDefaultOption("DelaunayMeshing.max_side_length_percentile",
                              &delaunay_meshing->max_side_length_percentile);
  AddAndRegisterDefaultOption("DelaunayMeshing.num_threads",
                              &delaunay_meshing->num_threads);
}

void OptionManager::AddRenderOptions() {
  if (added_render_options_) {
    return;
  }
  added_render_options_ = true;

  AddAndRegisterDefaultOption("Render.min_track_len", &render->min_track_len);
  AddAndRegisterDefaultOption("Render.max_error", &render->max_error);
  AddAndRegisterDefaultOption("Render.refresh_rate", &render->refresh_rate);
  AddAndRegisterDefaultOption("Render.adapt_refresh_rate",
                              &render->adapt_refresh_rate);
  AddAndRegisterDefaultOption("Render.image_connections",
                              &render->image_connections);
  AddAndRegisterDefaultOption("Render.projection_type",
                              &render->projection_type);
}

void OptionManager::Reset() {
  FLAGS_logtostderr = true;

  const bool kResetPaths = true;
  ResetOptions(kResetPaths);

  desc_ = std::make_shared<boost::program_options::options_description>();

  options_bool_.clear();
  options_int_.clear();
  options_double_.clear();
  options_string_.clear();

  added_log_options_ = false;
  added_random_options_ = false;
  added_database_options_ = false;
  added_image_options_ = false;
  added_extraction_options_ = false;
  added_match_options_ = false;
  added_exhaustive_match_options_ = false;
  added_sequential_match_options_ = false;
  added_vocab_tree_match_options_ = false;
  added_spatial_match_options_ = false;
  added_transitive_match_options_ = false;
  added_image_pairs_match_options_ = false;
  added_ba_options_ = false;
  added_mapper_options_ = false;
  added_patch_match_stereo_options_ = false;
  added_stereo_fusion_options_ = false;
  added_poisson_meshing_options_ = false;
  added_delaunay_meshing_options_ = false;
  added_render_options_ = false;
}

void OptionManager::ResetOptions(const bool reset_paths) {
  if (reset_paths) {
    *project_path = "";
    *database_path = "";
    *image_path = "";
  }
  *image_reader = ImageReaderOptions();
  *sift_extraction = SiftExtractionOptions();
  *sift_matching = SiftMatchingOptions();
  *exhaustive_matching = ExhaustiveMatchingOptions();
  *sequential_matching = SequentialMatchingOptions();
  *vocab_tree_matching = VocabTreeMatchingOptions();
  *spatial_matching = SpatialMatchingOptions();
  *transitive_matching = TransitiveMatchingOptions();
  *image_pairs_matching = ImagePairsMatchingOptions();
  *bundle_adjustment = BundleAdjustmentOptions();
  *mapper = IncrementalPipelineOptions();
  *patch_match_stereo = mvs::PatchMatchOptions();
  *stereo_fusion = mvs::StereoFusionOptions();
  *poisson_meshing = mvs::PoissonMeshingOptions();
  *delaunay_meshing = mvs::DelaunayMeshingOptions();
  *render = RenderOptions();
}

bool OptionManager::Check() {
  bool success = true;

  if (added_database_options_) {
    const auto database_parent_path = GetParentDir(*database_path);
    success = success && CHECK_OPTION_IMPL(!ExistsDir(*database_path)) &&
              CHECK_OPTION_IMPL(database_parent_path == "" ||
                                ExistsDir(database_parent_path));
  }

  if (added_image_options_)
    success = success && CHECK_OPTION_IMPL(ExistsDir(*image_path));

  if (image_reader) success = success && image_reader->Check();
  if (sift_extraction) success = success && sift_extraction->Check();

  if (sift_matching) success = success && sift_matching->Check();
  if (two_view_geometry) success = success && two_view_geometry->Check();
  if (exhaustive_matching) success = success && exhaustive_matching->Check();
  if (sequential_matching) success = success && sequential_matching->Check();
  if (vocab_tree_matching) success = success && vocab_tree_matching->Check();
  if (spatial_matching) success = success && spatial_matching->Check();
  if (transitive_matching) success = success && transitive_matching->Check();
  if (image_pairs_matching) success = success && image_pairs_matching->Check();

  if (bundle_adjustment) success = success && bundle_adjustment->Check();
  if (mapper) success = success && mapper->Check();

  if (patch_match_stereo) success = success && patch_match_stereo->Check();
  if (stereo_fusion) success = success && stereo_fusion->Check();
  if (poisson_meshing) success = success && poisson_meshing->Check();
  if (delaunay_meshing) success = success && delaunay_meshing->Check();

#if defined(COLMAP_GUI_ENABLED)
  if (render) success = success && render->Check();
#endif

  return success;
}

void OptionManager::Parse(const int argc, char** argv) {
  config::variables_map vmap;

  try {
    config::store(config::parse_command_line(argc, argv, *desc_), vmap);

    if (vmap.count("help")) {
      LOG(INFO) << StringPrintf(
          "%s (%s)", GetVersionInfo().c_str(), GetBuildInfo().c_str());
      LOG(INFO)
          << "Options can either be specified via command-line or by defining "
             "them in a .ini project file passed to `--project_path`.\n"
          << *desc_;
      // NOLINTNEXTLINE(concurrency-mt-unsafe)
      exit(EXIT_SUCCESS);
    }

    if (vmap.count("project_path")) {
      *project_path = vmap["project_path"].as<std::string>();
      if (!Read(*project_path)) {
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        exit(EXIT_FAILURE);
      }
    } else {
      vmap.notify();
    }
  } catch (std::exception& exc) {
    LOG(ERROR) << "Failed to parse options - " << exc.what() << ".";
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    exit(EXIT_FAILURE);
  } catch (...) {
    LOG(ERROR) << "Failed to parse options for unknown reason.";
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    exit(EXIT_FAILURE);
  }

  if (!Check()) {
    LOG(ERROR) << "Invalid options provided.";
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    exit(EXIT_FAILURE);
  }
}

bool OptionManager::Read(const std::string& path) {
  config::variables_map vmap;

  if (!ExistsFile(path)) {
    LOG(ERROR) << "Configuration file does not exist.";
    return false;
  }

  try {
    std::ifstream file(path);
    THROW_CHECK_FILE_OPEN(file, path);
    config::store(config::parse_config_file(file, *desc_), vmap);
    vmap.notify();
  } catch (std::exception& e) {
    LOG(ERROR) << "Failed to parse options " << e.what() << ".";
    return false;
  } catch (...) {
    LOG(ERROR) << "Failed to parse options for unknown reason.";
    return false;
  }

  return Check();
}

bool OptionManager::ReRead(const std::string& path) {
  Reset();
  AddAllOptions();
  return Read(path);
}

void OptionManager::Write(const std::string& path) const {
  boost::property_tree::ptree pt;

  // First, put all options without a section and then those with a section.
  // This is necessary as otherwise older Boost versions will write the
  // options without a section in between other sections and therefore
  // the errors will be assigned to the wrong section if read later.

  for (const auto& option : options_bool_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_int_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_double_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_string_) {
    if (!StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_bool_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_int_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_double_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  for (const auto& option : options_string_) {
    if (StringContains(option.first, ".")) {
      pt.put(option.first, *option.second);
    }
  }

  std::ofstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  // Ensure that we don't lose any precision by storing in text.
  file.precision(17);
  boost::property_tree::write_ini(file, pt);
  file.close();
}

}  // namespace colmap
