// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/controllers/global_pipeline.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/controllers/pairing.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/global_positioning.h"
#include "colmap/estimators/gravity_refinement.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/aliked.h"
#include "colmap/feature/sift.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/scene/reconstruction_clustering.h"
#include "colmap/ui/render_options.h"
#include "colmap/util/file.h"
#include "colmap/util/version.h"

namespace config = boost::program_options;

namespace colmap {

OptionManager::OptionManager(bool add_project_options)
    : BaseOptionManager(add_project_options) {
  image_reader = std::make_shared<ImageReaderOptions>();
  feature_extraction = std::make_shared<FeatureExtractionOptions>();
  feature_matching = std::make_shared<FeatureMatchingOptions>();
  two_view_geometry = std::make_shared<TwoViewGeometryOptions>();
  exhaustive_pairing = std::make_shared<ExhaustivePairingOptions>();
  sequential_pairing = std::make_shared<SequentialPairingOptions>();
  vocab_tree_pairing = std::make_shared<VocabTreePairingOptions>();
  spatial_pairing = std::make_shared<SpatialPairingOptions>();
  transitive_pairing = std::make_shared<TransitivePairingOptions>();
  imported_pairing = std::make_shared<ImportedPairingOptions>();
  bundle_adjustment = std::make_shared<BundleAdjustmentOptions>();
  mapper = std::make_shared<IncrementalPipelineOptions>();
  global_mapper = std::make_shared<GlobalPipelineOptions>();
  gravity_refiner = std::make_shared<GravityRefinerOptions>();
  reconstruction_clusterer =
      std::make_shared<ReconstructionClusteringOptions>();
  patch_match_stereo = std::make_shared<mvs::PatchMatchOptions>();
  stereo_fusion = std::make_shared<mvs::StereoFusionOptions>();
  poisson_meshing = std::make_shared<mvs::PoissonMeshingOptions>();
  delaunay_meshing = std::make_shared<mvs::DelaunayMeshingOptions>();
  render = std::make_shared<RenderOptions>();
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
  mapper->ba_global_frames_ratio = 1.4;
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
  feature_extraction->max_image_size = 1000;
  feature_extraction->sift->max_num_features = 2048;
  sequential_pairing->loop_detection_num_images /= 2;
  vocab_tree_pairing->max_num_features = 256;
  vocab_tree_pairing->num_images /= 2;
  mapper->ba_local_max_num_iterations /= 2;
  mapper->ba_global_max_num_iterations /= 2;
  mapper->ba_global_frames_ratio *= 1.2;
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
  feature_extraction->max_image_size = 1600;
  feature_extraction->sift->max_num_features = 4096;
  sequential_pairing->loop_detection_num_images /= 1.5;
  vocab_tree_pairing->max_num_features = 1024;
  vocab_tree_pairing->num_images /= 1.5;
  mapper->ba_local_max_num_iterations /= 1.5;
  mapper->ba_global_max_num_iterations /= 1.5;
  mapper->ba_global_frames_ratio *= 1.1;
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
  feature_extraction->sift->estimate_affine_shape = true;
  feature_extraction->max_image_size = 2400;
  feature_extraction->sift->max_num_features = 8192;
  feature_matching->guided_matching = true;
  vocab_tree_pairing->max_num_features = 4096;
  mapper->ba_local_max_num_iterations = 30;
  mapper->ba_local_max_refinements = 3;
  mapper->ba_global_max_num_iterations = 75;
  patch_match_stereo->max_image_size = 2400;
  stereo_fusion->max_image_size = 2400;
}

void OptionManager::ModifyForExtremeQuality() {
  // Most of the options are set to extreme quality by default.
  feature_extraction->sift->estimate_affine_shape = true;
  feature_extraction->sift->domain_size_pooling = true;
  feature_matching->guided_matching = true;
  mapper->ba_local_max_num_iterations = 40;
  mapper->ba_local_max_refinements = 3;
  mapper->ba_global_max_num_iterations = 100;
}

void OptionManager::AddAllOptions() {
  BaseOptionManager::AddAllOptions();
  AddFeatureExtractionOptions();
  AddFeatureMatchingOptions();
  AddTwoViewGeometryOptions();
  AddExhaustivePairingOptions();
  AddSequentialPairingOptions();
  AddVocabTreePairingOptions();
  AddSpatialPairingOptions();
  AddTransitivePairingOptions();
  AddImportedPairingOptions();
  AddBundleAdjustmentOptions();
  AddMapperOptions();
  AddPatchMatchStereoOptions();
  AddStereoFusionOptions();
  AddPoissonMeshingOptions();
  AddDelaunayMeshingOptions();
  AddRenderOptions();
}

void OptionManager::AddFeatureExtractionOptions() {
  if (added_feature_extraction_options_) {
    return;
  }
  added_feature_extraction_options_ = true;

  AddDefaultOption("ImageReader.mask_path", &image_reader->mask_path);
  AddDefaultOption("ImageReader.camera_model", &image_reader->camera_model);
  AddDefaultOption("ImageReader.single_camera", &image_reader->single_camera);
  AddDefaultOption("ImageReader.single_camera_per_folder",
                   &image_reader->single_camera_per_folder);
  AddDefaultOption("ImageReader.single_camera_per_image",
                   &image_reader->single_camera_per_image);
  AddDefaultOption("ImageReader.existing_camera_id",
                   &image_reader->existing_camera_id);
  AddDefaultOption("ImageReader.camera_params", &image_reader->camera_params);
  AddDefaultOption("ImageReader.default_focal_length_factor",
                   &image_reader->default_focal_length_factor);
  AddDefaultOption("ImageReader.camera_mask_path",
                   &image_reader->camera_mask_path);

  AddDefaultEnumOption("FeatureExtraction.type",
                       &feature_extraction->type,
                       FeatureExtractorTypeToString,
                       FeatureExtractorTypeFromString);
  AddDefaultOption("FeatureExtraction.num_threads",
                   &feature_extraction->num_threads);
  AddDefaultOption("FeatureExtraction.use_gpu", &feature_extraction->use_gpu);
  AddDefaultOption("FeatureExtraction.gpu_index",
                   &feature_extraction->gpu_index);
  AddDefaultOption("FeatureExtraction.max_image_size",
                   &feature_extraction->max_image_size);

  AddDefaultOption("SiftExtraction.max_num_features",
                   &feature_extraction->sift->max_num_features);
  AddDefaultOption("SiftExtraction.first_octave",
                   &feature_extraction->sift->first_octave);
  AddDefaultOption("SiftExtraction.num_octaves",
                   &feature_extraction->sift->num_octaves);
  AddDefaultOption("SiftExtraction.octave_resolution",
                   &feature_extraction->sift->octave_resolution);
  AddDefaultOption("SiftExtraction.peak_threshold",
                   &feature_extraction->sift->peak_threshold);
  AddDefaultOption("SiftExtraction.edge_threshold",
                   &feature_extraction->sift->edge_threshold);
  AddDefaultOption("SiftExtraction.estimate_affine_shape",
                   &feature_extraction->sift->estimate_affine_shape);
  AddDefaultOption("SiftExtraction.max_num_orientations",
                   &feature_extraction->sift->max_num_orientations);
  AddDefaultOption("SiftExtraction.upright",
                   &feature_extraction->sift->upright);
  AddDefaultOption("SiftExtraction.domain_size_pooling",
                   &feature_extraction->sift->domain_size_pooling);
  AddDefaultOption("SiftExtraction.dsp_min_scale",
                   &feature_extraction->sift->dsp_min_scale);
  AddDefaultOption("SiftExtraction.dsp_max_scale",
                   &feature_extraction->sift->dsp_max_scale);
  AddDefaultOption("SiftExtraction.dsp_num_scales",
                   &feature_extraction->sift->dsp_num_scales);

  AddDefaultOption("AlikedExtraction.max_num_features",
                   &feature_extraction->aliked->max_num_features);
  AddDefaultOption("AlikedExtraction.min_score",
                   &feature_extraction->aliked->min_score);
  AddDefaultOption("AlikedExtraction.n16rot_model_path",
                   &feature_extraction->aliked->n16rot_model_path);
  AddDefaultOption("AlikedExtraction.n32_model_path",
                   &feature_extraction->aliked->n32_model_path);
}

void OptionManager::AddFeatureMatchingOptions() {
  if (added_feature_matching_options_) {
    return;
  }
  added_feature_matching_options_ = true;

  AddDefaultEnumOption("FeatureMatching.type",
                       &feature_matching->type,
                       FeatureMatcherTypeToString,
                       FeatureMatcherTypeFromString);
  AddDefaultOption("FeatureMatching.num_threads",
                   &feature_matching->num_threads);
  AddDefaultOption("FeatureMatching.use_gpu", &feature_matching->use_gpu);
  AddDefaultOption("FeatureMatching.gpu_index", &feature_matching->gpu_index);
  AddDefaultOption("FeatureMatching.guided_matching",
                   &feature_matching->guided_matching);
  AddDefaultOption("FeatureMatching.skip_geometric_verification",
                   &feature_matching->skip_geometric_verification);
  AddDefaultOption("FeatureMatching.rig_verification",
                   &feature_matching->rig_verification);
  AddDefaultOption("FeatureMatching.skip_image_pairs_in_same_frame",
                   &feature_matching->skip_image_pairs_in_same_frame);
  AddDefaultOption("FeatureMatching.max_num_matches",
                   &feature_matching->max_num_matches);

  AddDefaultOption("SiftMatching.max_ratio",
                   &feature_matching->sift->max_ratio);
  AddDefaultOption("SiftMatching.max_distance",
                   &feature_matching->sift->max_distance);
  AddDefaultOption("SiftMatching.cross_check",
                   &feature_matching->sift->cross_check);
  AddDefaultOption("SiftMatching.cpu_brute_force_matcher",
                   &feature_matching->sift->cpu_brute_force_matcher);
  AddDefaultOption("SiftMatching.lightglue_min_score",
                   &feature_matching->sift->lightglue.min_score);
  AddDefaultOption("SiftMatching.lightglue_model_path",
                   &feature_matching->sift->lightglue.model_path);

  AddDefaultOption("AlikedMatching.brute_force_min_cossim",
                   &feature_matching->aliked->brute_force.min_cossim);
  AddDefaultOption("AlikedMatching.brute_force_max_ratio",
                   &feature_matching->aliked->brute_force.max_ratio);
  AddDefaultOption("AlikedMatching.brute_force_cross_check",
                   &feature_matching->aliked->brute_force.cross_check);
  AddDefaultOption("AlikedMatching.bruteforce_model_path",
                   &feature_matching->aliked->brute_force.model_path);
  AddDefaultOption("AlikedMatching.lightglue_min_score",
                   &feature_matching->aliked->lightglue.min_score);
  AddDefaultOption("AlikedMatching.lightglue_model_path",
                   &feature_matching->aliked->lightglue.model_path);
}

void OptionManager::AddTwoViewGeometryOptions() {
  if (added_two_view_geometry_options_) {
    return;
  }
  added_two_view_geometry_options_ = true;
  AddDefaultOption("TwoViewGeometry.min_num_inliers",
                   &two_view_geometry->min_num_inliers);
  AddDefaultOption("TwoViewGeometry.multiple_models",
                   &two_view_geometry->multiple_models);
  AddDefaultOption("TwoViewGeometry.compute_relative_pose",
                   &two_view_geometry->compute_relative_pose);
  AddDefaultOption("TwoViewGeometry.detect_watermark",
                   &two_view_geometry->detect_watermark);
  AddDefaultOption("TwoViewGeometry.multiple_ignore_watermark",
                   &two_view_geometry->multiple_ignore_watermark);
  AddDefaultOption("TwoViewGeometry.watermark_detection_max_error",
                   &two_view_geometry->watermark_detection_max_error);
  AddDefaultOption("TwoViewGeometry.filter_stationary_matches",
                   &two_view_geometry->filter_stationary_matches);
  AddDefaultOption("TwoViewGeometry.stationary_matches_max_error",
                   &two_view_geometry->stationary_matches_max_error);
  AddDefaultOption("TwoViewGeometry.max_error",
                   &two_view_geometry->ransac_options.max_error);
  AddDefaultOption("TwoViewGeometry.confidence",
                   &two_view_geometry->ransac_options.confidence);
  AddDefaultOption("TwoViewGeometry.max_num_trials",
                   &two_view_geometry->ransac_options.max_num_trials);
  AddDefaultOption("TwoViewGeometry.min_inlier_ratio",
                   &two_view_geometry->ransac_options.min_inlier_ratio);
  AddDefaultOption("TwoViewGeometry.random_seed",
                   &two_view_geometry->ransac_options.random_seed);
}

void OptionManager::AddExhaustivePairingOptions() {
  if (added_exhaustive_pairing_options_) {
    return;
  }
  added_exhaustive_pairing_options_ = true;

  AddFeatureMatchingOptions();
  AddTwoViewGeometryOptions();

  AddDefaultOption("ExhaustiveMatching.block_size",
                   &exhaustive_pairing->block_size);
}

void OptionManager::AddSequentialPairingOptions() {
  if (added_sequential_pairing_options_) {
    return;
  }
  added_sequential_pairing_options_ = true;

  AddFeatureMatchingOptions();
  AddTwoViewGeometryOptions();

  AddDefaultOption("SequentialMatching.overlap", &sequential_pairing->overlap);
  AddDefaultOption("SequentialMatching.quadratic_overlap",
                   &sequential_pairing->quadratic_overlap);
  AddDefaultOption("SequentialMatching.expand_rig_images",
                   &sequential_pairing->expand_rig_images);
  AddDefaultOption("SequentialMatching.loop_detection",
                   &sequential_pairing->loop_detection);
  AddDefaultOption("SequentialMatching.loop_detection_period",
                   &sequential_pairing->loop_detection_period);
  AddDefaultOption("SequentialMatching.loop_detection_num_images",
                   &sequential_pairing->loop_detection_num_images);
  AddDefaultOption("SequentialMatching.loop_detection_num_nearest_neighbors",
                   &sequential_pairing->loop_detection_num_nearest_neighbors);
  AddDefaultOption("SequentialMatching.loop_detection_num_checks",
                   &sequential_pairing->loop_detection_num_checks);
  AddDefaultOption(
      "SequentialMatching.loop_detection_num_images_after_verification",
      &sequential_pairing->loop_detection_num_images_after_verification);
  AddDefaultOption("SequentialMatching.loop_detection_max_num_features",
                   &sequential_pairing->loop_detection_max_num_features);
  AddDefaultOption("SequentialMatching.vocab_tree_path",
                   &sequential_pairing->vocab_tree_path);
  AddDefaultOption("SequentialMatching.num_threads",
                   &sequential_pairing->num_threads);
}

void OptionManager::AddVocabTreePairingOptions() {
  if (added_vocab_tree_pairing_options_) {
    return;
  }
  added_vocab_tree_pairing_options_ = true;

  AddFeatureMatchingOptions();
  AddTwoViewGeometryOptions();

  AddDefaultOption("VocabTreeMatching.num_images",
                   &vocab_tree_pairing->num_images);
  AddDefaultOption("VocabTreeMatching.num_nearest_neighbors",
                   &vocab_tree_pairing->num_nearest_neighbors);
  AddDefaultOption("VocabTreeMatching.num_checks",
                   &vocab_tree_pairing->num_checks);
  AddDefaultOption("VocabTreeMatching.num_images_after_verification",
                   &vocab_tree_pairing->num_images_after_verification);
  AddDefaultOption("VocabTreeMatching.max_num_features",
                   &vocab_tree_pairing->max_num_features);
  AddDefaultOption("VocabTreeMatching.vocab_tree_path",
                   &vocab_tree_pairing->vocab_tree_path);
  AddDefaultOption("VocabTreeMatching.match_list_path",
                   &vocab_tree_pairing->match_list_path);
  AddDefaultOption("VocabTreeMatching.num_threads",
                   &vocab_tree_pairing->num_threads);
}

void OptionManager::AddSpatialPairingOptions() {
  if (added_spatial_pairing_options_) {
    return;
  }
  added_spatial_pairing_options_ = true;

  AddFeatureMatchingOptions();
  AddTwoViewGeometryOptions();

  AddDefaultOption("SpatialMatching.ignore_z", &spatial_pairing->ignore_z);
  AddDefaultOption("SpatialMatching.max_num_neighbors",
                   &spatial_pairing->max_num_neighbors);
  AddDefaultOption("SpatialMatching.min_num_neighbors",
                   &spatial_pairing->min_num_neighbors);
  AddDefaultOption("SpatialMatching.max_distance",
                   &spatial_pairing->max_distance);
}

void OptionManager::AddTransitivePairingOptions() {
  if (added_transitive_pairing_options_) {
    return;
  }
  added_transitive_pairing_options_ = true;

  AddFeatureMatchingOptions();
  AddTwoViewGeometryOptions();

  AddDefaultOption("TransitiveMatching.batch_size",
                   &transitive_pairing->batch_size);
  AddDefaultOption("TransitiveMatching.num_iterations",
                   &transitive_pairing->num_iterations);
}

void OptionManager::AddImportedPairingOptions() {
  if (added_image_pairs_pairing_options_) {
    return;
  }
  added_image_pairs_pairing_options_ = true;

  AddFeatureMatchingOptions();
  AddTwoViewGeometryOptions();

  AddDefaultOption("ImagePairsMatching.block_size",
                   &imported_pairing->block_size);
}

void OptionManager::AddBundleAdjustmentOptions() {
  if (added_ba_options_) {
    return;
  }
  added_ba_options_ = true;

  // Solver-agnostic options
  AddDefaultOption("BundleAdjustment.refine_focal_length",
                   &bundle_adjustment->refine_focal_length);
  AddDefaultOption("BundleAdjustment.refine_principal_point",
                   &bundle_adjustment->refine_principal_point);
  AddDefaultOption("BundleAdjustment.refine_extra_params",
                   &bundle_adjustment->refine_extra_params);
  AddDefaultOption("BundleAdjustment.refine_rig_from_world",
                   &bundle_adjustment->refine_rig_from_world);
  AddDefaultOption("BundleAdjustment.refine_sensor_from_rig",
                   &bundle_adjustment->refine_sensor_from_rig);
  AddDefaultOption("BundleAdjustment.refine_points3D",
                   &bundle_adjustment->refine_points3D);
  AddDefaultOption("BundleAdjustment.constant_rig_from_world_rotation",
                   &bundle_adjustment->constant_rig_from_world_rotation);
  AddDefaultOption("BundleAdjustment.min_track_length",
                   &bundle_adjustment->min_track_length);

  // Ceres-specific options
  AddDefaultOption(
      "BundleAdjustmentCeres.max_num_iterations",
      &bundle_adjustment->ceres->solver_options.max_num_iterations);
  AddDefaultOption(
      "BundleAdjustmentCeres.max_linear_solver_iterations",
      &bundle_adjustment->ceres->solver_options.max_linear_solver_iterations);
  AddDefaultOption(
      "BundleAdjustmentCeres.function_tolerance",
      &bundle_adjustment->ceres->solver_options.function_tolerance);
  AddDefaultOption(
      "BundleAdjustmentCeres.gradient_tolerance",
      &bundle_adjustment->ceres->solver_options.gradient_tolerance);
  AddDefaultOption(
      "BundleAdjustmentCeres.parameter_tolerance",
      &bundle_adjustment->ceres->solver_options.parameter_tolerance);
  AddDefaultOption("BundleAdjustmentCeres.use_gpu",
                   &bundle_adjustment->ceres->use_gpu);
  AddDefaultOption("BundleAdjustmentCeres.gpu_index",
                   &bundle_adjustment->ceres->gpu_index);
  AddDefaultOption("BundleAdjustmentCeres.min_num_images_gpu_solver",
                   &bundle_adjustment->ceres->min_num_images_gpu_solver);
  AddDefaultOption(
      "BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading",
      &bundle_adjustment->ceres->min_num_residuals_for_cpu_multi_threading);
  AddDefaultOption(
      "BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver",
      &bundle_adjustment->ceres->max_num_images_direct_dense_cpu_solver);
  AddDefaultOption(
      "BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver",
      &bundle_adjustment->ceres->max_num_images_direct_sparse_cpu_solver);
  AddDefaultOption(
      "BundleAdjustmentCeres.max_num_images_direct_dense_gpu_solver",
      &bundle_adjustment->ceres->max_num_images_direct_dense_gpu_solver);
  AddDefaultOption(
      "BundleAdjustmentCeres.max_num_images_direct_sparse_gpu_solver",
      &bundle_adjustment->ceres->max_num_images_direct_sparse_gpu_solver);
}

void OptionManager::AddMapperOptions() {
  if (added_mapper_options_) {
    return;
  }
  added_mapper_options_ = true;

  AddDefaultOption("Mapper.min_num_matches", &mapper->min_num_matches);
  AddDefaultOption("Mapper.ignore_watermarks", &mapper->ignore_watermarks);
  AddDefaultOption("Mapper.multiple_models", &mapper->multiple_models);
  AddDefaultOption("Mapper.max_num_models", &mapper->max_num_models);
  AddDefaultOption("Mapper.max_model_overlap", &mapper->max_model_overlap);
  AddDefaultOption("Mapper.min_model_size", &mapper->min_model_size);
  AddDefaultOption("Mapper.init_image_id1", &mapper->init_image_id1);
  AddDefaultOption("Mapper.init_image_id2", &mapper->init_image_id2);
  AddDefaultOption("Mapper.init_num_trials", &mapper->init_num_trials);
  AddDefaultOption("Mapper.structure_less_registration_fallback",
                   &mapper->structure_less_registration_fallback);
  AddDefaultOption("Mapper.structure_less_registration_only",
                   &mapper->structure_less_registration_only);
  AddDefaultOption("Mapper.extract_colors", &mapper->extract_colors);
  AddDefaultOption("Mapper.num_threads", &mapper->num_threads);
  AddDefaultOption("Mapper.random_seed", &mapper->random_seed);
  AddDefaultOption("Mapper.min_focal_length_ratio",
                   &mapper->min_focal_length_ratio);
  AddDefaultOption("Mapper.max_focal_length_ratio",
                   &mapper->max_focal_length_ratio);
  AddDefaultOption("Mapper.max_extra_param", &mapper->max_extra_param);
  AddDefaultOption("Mapper.ba_refine_focal_length",
                   &mapper->ba_refine_focal_length);
  AddDefaultOption("Mapper.ba_refine_principal_point",
                   &mapper->ba_refine_principal_point);
  AddDefaultOption("Mapper.ba_refine_extra_params",
                   &mapper->ba_refine_extra_params);
  AddDefaultOption("Mapper.ba_refine_sensor_from_rig",
                   &mapper->ba_refine_sensor_from_rig);
  AddDefaultOption("Mapper.ba_local_function_tolerance",
                   &mapper->ba_local_function_tolerance);
  AddDefaultOption("Mapper.ba_local_max_num_iterations",
                   &mapper->ba_local_max_num_iterations);
  AddDefaultOption("Mapper.ba_global_frames_ratio",
                   &mapper->ba_global_frames_ratio);
  AddDefaultOption("Mapper.ba_global_points_ratio",
                   &mapper->ba_global_points_ratio);
  AddDefaultOption("Mapper.ba_global_frames_freq",
                   &mapper->ba_global_frames_freq);
  AddDefaultOption("Mapper.ba_global_points_freq",
                   &mapper->ba_global_points_freq);
  AddDefaultOption("Mapper.ba_global_function_tolerance",
                   &mapper->ba_global_function_tolerance);
  AddDefaultOption("Mapper.ba_global_max_num_iterations",
                   &mapper->ba_global_max_num_iterations);
  AddDefaultOption("Mapper.ba_global_max_refinements",
                   &mapper->ba_global_max_refinements);
  AddDefaultOption("Mapper.ba_global_max_refinement_change",
                   &mapper->ba_global_max_refinement_change);
  AddDefaultOption("Mapper.ba_local_max_refinements",
                   &mapper->ba_local_max_refinements);
  AddDefaultOption("Mapper.ba_local_max_refinement_change",
                   &mapper->ba_local_max_refinement_change);
  AddDefaultOption("Mapper.ba_use_gpu", &mapper->ba_use_gpu);
  AddDefaultOption("Mapper.ba_gpu_index", &mapper->ba_gpu_index);
  AddDefaultOption("Mapper.ba_min_num_residuals_for_cpu_multi_threading",
                   &mapper->ba_min_num_residuals_for_cpu_multi_threading);
  AddDefaultOption("Mapper.snapshot_path", &mapper->snapshot_path);
  AddDefaultOption("Mapper.snapshot_frames_freq",
                   &mapper->snapshot_frames_freq);
  AddDefaultOption("Mapper.fix_existing_frames", &mapper->fix_existing_frames);

  // IncrementalMapper.
  AddDefaultOption("Mapper.init_min_num_inliers",
                   &mapper->mapper.init_min_num_inliers);
  AddDefaultOption("Mapper.init_max_error", &mapper->mapper.init_max_error);
  AddDefaultOption("Mapper.init_max_forward_motion",
                   &mapper->mapper.init_max_forward_motion);
  AddDefaultOption("Mapper.init_min_tri_angle",
                   &mapper->mapper.init_min_tri_angle);
  AddDefaultOption("Mapper.init_max_reg_trials",
                   &mapper->mapper.init_max_reg_trials);
  AddDefaultOption("Mapper.abs_pose_max_error",
                   &mapper->mapper.abs_pose_max_error);
  AddDefaultOption("Mapper.abs_pose_min_num_inliers",
                   &mapper->mapper.abs_pose_min_num_inliers);
  AddDefaultOption("Mapper.abs_pose_min_inlier_ratio",
                   &mapper->mapper.abs_pose_min_inlier_ratio);
  AddDefaultOption("Mapper.filter_max_reproj_error",
                   &mapper->mapper.filter_max_reproj_error);
  AddDefaultOption("Mapper.filter_min_tri_angle",
                   &mapper->mapper.filter_min_tri_angle);
  AddDefaultOption("Mapper.max_reg_trials", &mapper->mapper.max_reg_trials);
  AddDefaultOption("Mapper.ba_local_num_images",
                   &mapper->mapper.ba_local_num_images);
  AddDefaultOption("Mapper.ba_local_min_tri_angle",
                   &mapper->mapper.ba_local_min_tri_angle);
  AddDefaultOption("Mapper.ba_global_ignore_redundant_points3D",
                   &mapper->mapper.ba_global_ignore_redundant_points3D);
  AddDefaultOption(
      "Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain",
      &mapper->mapper.ba_global_ignore_redundant_points3D_min_coverage_gain);

  AddDefaultOption("Mapper.image_list_path", &mapper_image_list_path_);
  AddDefaultOption("Mapper.constant_rig_list_path",
                   &mapper_constant_rig_list_path_);
  AddDefaultOption("Mapper.constant_camera_list_path",
                   &mapper_constant_camera_list_path_);
  AddDefaultOption("Mapper.max_runtime_seconds", &mapper->max_runtime_seconds);

  // IncrementalTriangulator.
  AddDefaultOption("Mapper.tri_max_transitivity",
                   &mapper->triangulation.max_transitivity);
  AddDefaultOption("Mapper.tri_create_max_angle_error",
                   &mapper->triangulation.create_max_angle_error);
  AddDefaultOption("Mapper.tri_continue_max_angle_error",
                   &mapper->triangulation.continue_max_angle_error);
  AddDefaultOption("Mapper.tri_merge_max_reproj_error",
                   &mapper->triangulation.merge_max_reproj_error);
  AddDefaultOption("Mapper.tri_complete_max_reproj_error",
                   &mapper->triangulation.complete_max_reproj_error);
  AddDefaultOption("Mapper.tri_complete_max_transitivity",
                   &mapper->triangulation.complete_max_transitivity);
  AddDefaultOption("Mapper.tri_re_max_angle_error",
                   &mapper->triangulation.re_max_angle_error);
  AddDefaultOption("Mapper.tri_re_min_ratio",
                   &mapper->triangulation.re_min_ratio);
  AddDefaultOption("Mapper.tri_re_max_trials",
                   &mapper->triangulation.re_max_trials);
  AddDefaultOption("Mapper.tri_min_angle", &mapper->triangulation.min_angle);
  AddDefaultOption("Mapper.tri_ignore_two_view_tracks",
                   &mapper->triangulation.ignore_two_view_tracks);
}

void OptionManager::AddGlobalMapperOptions() {
  if (added_global_mapper_options_) {
    return;
  }
  added_global_mapper_options_ = true;

  // Global mapper options.
  AddDefaultOption("GlobalMapper.image_list_path",
                   &global_mapper_image_list_path_);
  AddDefaultOption("GlobalMapper.min_num_matches",
                   &global_mapper->min_num_matches);
  AddDefaultOption("GlobalMapper.ignore_watermarks",
                   &global_mapper->ignore_watermarks);
  AddDefaultOption("GlobalMapper.num_threads", &global_mapper->num_threads);
  AddDefaultOption("GlobalMapper.random_seed", &global_mapper->random_seed);
  AddDefaultOption("GlobalMapper.decompose_relative_pose",
                   &global_mapper->decompose_relative_pose);
  AddDefaultOption("GlobalMapper.ba_num_iterations",
                   &global_mapper->mapper.ba_num_iterations);
  AddDefaultOption("GlobalMapper.skip_rotation_averaging",
                   &global_mapper->mapper.skip_rotation_averaging);
  AddDefaultOption("GlobalMapper.skip_track_establishment",
                   &global_mapper->mapper.skip_track_establishment);
  AddDefaultOption("GlobalMapper.skip_global_positioning",
                   &global_mapper->mapper.skip_global_positioning);
  AddDefaultOption("GlobalMapper.skip_bundle_adjustment",
                   &global_mapper->mapper.skip_bundle_adjustment);
  AddDefaultOption("GlobalMapper.skip_retriangulation",
                   &global_mapper->mapper.skip_retriangulation);

  // Track establishment options.
  AddDefaultOption(
      "GlobalMapper.track_intra_image_consistency_threshold",
      &global_mapper->mapper.track_intra_image_consistency_threshold);
  AddDefaultOption("GlobalMapper.track_required_tracks_per_view",
                   &global_mapper->mapper.track_required_tracks_per_view);
  AddDefaultOption("GlobalMapper.track_min_num_views_per_track",
                   &global_mapper->mapper.track_min_num_views_per_track);

  // Global positioning options.
  AddDefaultOption("GlobalMapper.gp_use_gpu",
                   &global_mapper->mapper.global_positioning.use_gpu);
  AddDefaultOption("GlobalMapper.gp_gpu_index",
                   &global_mapper->mapper.global_positioning.gpu_index);
  AddDefaultOption(
      "GlobalMapper.gp_optimize_positions",
      &global_mapper->mapper.global_positioning.optimize_positions);
  AddDefaultOption("GlobalMapper.gp_optimize_points",
                   &global_mapper->mapper.global_positioning.optimize_points);
  AddDefaultOption("GlobalMapper.gp_optimize_scales",
                   &global_mapper->mapper.global_positioning.optimize_scales);
  AddDefaultOption(
      "GlobalMapper.gp_loss_function_scale",
      &global_mapper->mapper.global_positioning.loss_function_scale);
  AddDefaultOption("GlobalMapper.gp_max_num_iterations",
                   &global_mapper->mapper.global_positioning.solver_options
                        .max_num_iterations);

  // Bundle adjustment options (solver-agnostic).
  AddDefaultOption(
      "GlobalMapper.ba_refine_focal_length",
      &global_mapper->mapper.bundle_adjustment.refine_focal_length);
  AddDefaultOption(
      "GlobalMapper.ba_refine_principal_point",
      &global_mapper->mapper.bundle_adjustment.refine_principal_point);
  AddDefaultOption(
      "GlobalMapper.ba_refine_extra_params",
      &global_mapper->mapper.bundle_adjustment.refine_extra_params);
  AddDefaultOption(
      "GlobalMapper.ba_refine_sensor_from_rig",
      &global_mapper->mapper.bundle_adjustment.refine_sensor_from_rig);
  AddDefaultOption(
      "GlobalMapper.ba_refine_rig_from_world",
      &global_mapper->mapper.bundle_adjustment.refine_rig_from_world);
  AddDefaultOption("GlobalMapper.ba_refine_points3D",
                   &global_mapper->mapper.bundle_adjustment.refine_points3D);
  AddDefaultOption("GlobalMapper.ba_min_track_length",
                   &global_mapper->mapper.bundle_adjustment.min_track_length);
  // Bundle adjustment options (Ceres-specific).
  AddDefaultOption("GlobalMapper.ba_ceres_use_gpu",
                   &global_mapper->mapper.bundle_adjustment.ceres->use_gpu);
  AddDefaultOption("GlobalMapper.ba_ceres_gpu_index",
                   &global_mapper->mapper.bundle_adjustment.ceres->gpu_index);
  AddDefaultOption(
      "GlobalMapper.ba_ceres_loss_function_scale",
      &global_mapper->mapper.bundle_adjustment.ceres->loss_function_scale);
  AddDefaultOption("GlobalMapper.ba_ceres_max_num_iterations",
                   &global_mapper->mapper.bundle_adjustment.ceres
                        ->solver_options.max_num_iterations);
  AddDefaultOption("GlobalMapper.ba_skip_fixed_rotation_stage",
                   &global_mapper->mapper.ba_skip_fixed_rotation_stage);
  AddDefaultOption("GlobalMapper.ba_skip_joint_optimization_stage",
                   &global_mapper->mapper.ba_skip_joint_optimization_stage);

  // Retriangulation options.
  AddDefaultOption(
      "GlobalMapper.tri_complete_max_reproj_error",
      &global_mapper->mapper.retriangulation.complete_max_reproj_error);
  AddDefaultOption(
      "GlobalMapper.tri_merge_max_reproj_error",
      &global_mapper->mapper.retriangulation.merge_max_reproj_error);
  AddDefaultOption("GlobalMapper.tri_min_angle",
                   &global_mapper->mapper.retriangulation.min_angle);

  // Rotation averaging options.
  AddDefaultOption(
      "GlobalMapper.ra_max_rotation_error_deg",
      &global_mapper->mapper.rotation_averaging.max_rotation_error_deg);

  // Threshold options.
  AddDefaultOption("GlobalMapper.max_angular_reproj_error_deg",
                   &global_mapper->mapper.max_angular_reproj_error_deg);
  AddDefaultOption("GlobalMapper.max_normalized_reproj_error",
                   &global_mapper->mapper.max_normalized_reproj_error);
  AddDefaultOption("GlobalMapper.min_tri_angle_deg",
                   &global_mapper->mapper.min_tri_angle_deg);
}

void OptionManager::AddGravityRefinerOptions() {
  if (added_gravity_refiner_options_) {
    return;
  }
  added_gravity_refiner_options_ = true;

  AddDefaultOption("GravityRefiner.max_outlier_ratio",
                   &gravity_refiner->max_outlier_ratio);
  AddDefaultOption("GravityRefiner.max_gravity_error",
                   &gravity_refiner->max_gravity_error);
  AddDefaultOption("GravityRefiner.min_num_neighbors",
                   &gravity_refiner->min_num_neighbors);
}

void OptionManager::AddReconstructionClustererOptions() {
  if (added_reconstruction_clusterer_options_) {
    return;
  }
  added_reconstruction_clusterer_options_ = true;

  AddDefaultOption("ReconstructionClusterer.min_covisibility_count",
                   &reconstruction_clusterer->min_covisibility_count);
  AddDefaultOption("ReconstructionClusterer.min_edge_weight_threshold",
                   &reconstruction_clusterer->min_edge_weight_threshold);
  AddDefaultOption("ReconstructionClusterer.min_num_reg_frames",
                   &reconstruction_clusterer->min_num_reg_frames);
}

void OptionManager::AddPatchMatchStereoOptions() {
  if (added_patch_match_stereo_options_) {
    return;
  }
  added_patch_match_stereo_options_ = true;

  AddDefaultOption("PatchMatchStereo.max_image_size",
                   &patch_match_stereo->max_image_size);
  AddDefaultOption("PatchMatchStereo.gpu_index",
                   &patch_match_stereo->gpu_index);
  AddDefaultOption("PatchMatchStereo.depth_min",
                   &patch_match_stereo->depth_min);
  AddDefaultOption("PatchMatchStereo.depth_max",
                   &patch_match_stereo->depth_max);
  AddDefaultOption("PatchMatchStereo.window_radius",
                   &patch_match_stereo->window_radius);
  AddDefaultOption("PatchMatchStereo.window_step",
                   &patch_match_stereo->window_step);
  AddDefaultOption("PatchMatchStereo.sigma_spatial",
                   &patch_match_stereo->sigma_spatial);
  AddDefaultOption("PatchMatchStereo.sigma_color",
                   &patch_match_stereo->sigma_color);
  AddDefaultOption("PatchMatchStereo.num_samples",
                   &patch_match_stereo->num_samples);
  AddDefaultOption("PatchMatchStereo.ncc_sigma",
                   &patch_match_stereo->ncc_sigma);
  AddDefaultOption("PatchMatchStereo.min_triangulation_angle",
                   &patch_match_stereo->min_triangulation_angle);
  AddDefaultOption("PatchMatchStereo.incident_angle_sigma",
                   &patch_match_stereo->incident_angle_sigma);
  AddDefaultOption("PatchMatchStereo.num_iterations",
                   &patch_match_stereo->num_iterations);
  AddDefaultOption("PatchMatchStereo.geom_consistency",
                   &patch_match_stereo->geom_consistency);
  AddDefaultOption("PatchMatchStereo.geom_consistency_regularizer",
                   &patch_match_stereo->geom_consistency_regularizer);
  AddDefaultOption("PatchMatchStereo.geom_consistency_max_cost",
                   &patch_match_stereo->geom_consistency_max_cost);
  AddDefaultOption("PatchMatchStereo.filter", &patch_match_stereo->filter);
  AddDefaultOption("PatchMatchStereo.filter_min_ncc",
                   &patch_match_stereo->filter_min_ncc);
  AddDefaultOption("PatchMatchStereo.filter_min_triangulation_angle",
                   &patch_match_stereo->filter_min_triangulation_angle);
  AddDefaultOption("PatchMatchStereo.filter_min_num_consistent",
                   &patch_match_stereo->filter_min_num_consistent);
  AddDefaultOption("PatchMatchStereo.filter_geom_consistency_max_cost",
                   &patch_match_stereo->filter_geom_consistency_max_cost);
  AddDefaultOption("PatchMatchStereo.cache_size",
                   &patch_match_stereo->cache_size);
  AddDefaultOption("PatchMatchStereo.allow_missing_files",
                   &patch_match_stereo->allow_missing_files);
  AddDefaultOption("PatchMatchStereo.write_consistency_graph",
                   &patch_match_stereo->write_consistency_graph);
}

void OptionManager::AddStereoFusionOptions() {
  if (added_stereo_fusion_options_) {
    return;
  }
  added_stereo_fusion_options_ = true;

  AddDefaultOption("StereoFusion.mask_path", &stereo_fusion->mask_path);
  AddDefaultOption("StereoFusion.num_threads", &stereo_fusion->num_threads);
  AddDefaultOption("StereoFusion.max_image_size",
                   &stereo_fusion->max_image_size);
  AddDefaultOption("StereoFusion.min_num_pixels",
                   &stereo_fusion->min_num_pixels);
  AddDefaultOption("StereoFusion.max_num_pixels",
                   &stereo_fusion->max_num_pixels);
  AddDefaultOption("StereoFusion.max_traversal_depth",
                   &stereo_fusion->max_traversal_depth);
  AddDefaultOption("StereoFusion.max_reproj_error",
                   &stereo_fusion->max_reproj_error);
  AddDefaultOption("StereoFusion.max_depth_error",
                   &stereo_fusion->max_depth_error);
  AddDefaultOption("StereoFusion.max_normal_error",
                   &stereo_fusion->max_normal_error);
  AddDefaultOption("StereoFusion.check_num_images",
                   &stereo_fusion->check_num_images);
  AddDefaultOption("StereoFusion.cache_size", &stereo_fusion->cache_size);
  AddDefaultOption("StereoFusion.use_cache", &stereo_fusion->use_cache);
}

void OptionManager::AddPoissonMeshingOptions() {
  if (added_poisson_meshing_options_) {
    return;
  }
  added_poisson_meshing_options_ = true;

  AddDefaultOption("PoissonMeshing.point_weight",
                   &poisson_meshing->point_weight);
  AddDefaultOption("PoissonMeshing.depth", &poisson_meshing->depth);
  AddDefaultOption("PoissonMeshing.color", &poisson_meshing->color);
  AddDefaultOption("PoissonMeshing.trim", &poisson_meshing->trim);
  AddDefaultOption("PoissonMeshing.num_threads", &poisson_meshing->num_threads);
}

void OptionManager::AddDelaunayMeshingOptions() {
  if (added_delaunay_meshing_options_) {
    return;
  }
  added_delaunay_meshing_options_ = true;

  AddDefaultOption("DelaunayMeshing.max_proj_dist",
                   &delaunay_meshing->max_proj_dist);
  AddDefaultOption("DelaunayMeshing.max_depth_dist",
                   &delaunay_meshing->max_depth_dist);
  AddDefaultOption("DelaunayMeshing.visibility_sigma",
                   &delaunay_meshing->visibility_sigma);
  AddDefaultOption("DelaunayMeshing.distance_sigma_factor",
                   &delaunay_meshing->distance_sigma_factor);
  AddDefaultOption("DelaunayMeshing.quality_regularization",
                   &delaunay_meshing->quality_regularization);
  AddDefaultOption("DelaunayMeshing.max_side_length_factor",
                   &delaunay_meshing->max_side_length_factor);
  AddDefaultOption("DelaunayMeshing.max_side_length_percentile",
                   &delaunay_meshing->max_side_length_percentile);
  AddDefaultOption("DelaunayMeshing.num_threads",
                   &delaunay_meshing->num_threads);
}

void OptionManager::AddRenderOptions() {
  if (added_render_options_) {
    return;
  }
  added_render_options_ = true;

  AddDefaultOption("Render.min_track_len", &render->min_track_len);
  AddDefaultOption("Render.max_error", &render->max_error);
  AddDefaultOption("Render.refresh_rate", &render->refresh_rate);
  AddDefaultOption("Render.adapt_refresh_rate", &render->adapt_refresh_rate);
  AddDefaultOption("Render.image_connections", &render->image_connections);
  AddDefaultOption("Render.projection_type", &render->projection_type);
}

void OptionManager::Reset(bool reset_logging) {
  BaseOptionManager::Reset(reset_logging);

  added_feature_extraction_options_ = false;
  added_feature_matching_options_ = false;
  added_two_view_geometry_options_ = false;
  added_exhaustive_pairing_options_ = false;
  added_sequential_pairing_options_ = false;
  added_vocab_tree_pairing_options_ = false;
  added_spatial_pairing_options_ = false;
  added_transitive_pairing_options_ = false;
  added_image_pairs_pairing_options_ = false;
  added_ba_options_ = false;
  added_mapper_options_ = false;
  added_global_mapper_options_ = false;
  added_gravity_refiner_options_ = false;
  added_reconstruction_clusterer_options_ = false;
  added_patch_match_stereo_options_ = false;
  added_stereo_fusion_options_ = false;
  added_poisson_meshing_options_ = false;
  added_delaunay_meshing_options_ = false;
  added_render_options_ = false;
}

void OptionManager::ResetOptions(const bool reset_paths) {
  BaseOptionManager::ResetOptions(reset_paths);

  *image_reader = ImageReaderOptions();
  *feature_extraction = FeatureExtractionOptions();
  *feature_matching = FeatureMatchingOptions();
  *exhaustive_pairing = ExhaustivePairingOptions();
  *sequential_pairing = SequentialPairingOptions();
  *vocab_tree_pairing = VocabTreePairingOptions();
  *spatial_pairing = SpatialPairingOptions();
  *transitive_pairing = TransitivePairingOptions();
  *imported_pairing = ImportedPairingOptions();
  *bundle_adjustment = BundleAdjustmentOptions();
  *mapper = IncrementalPipelineOptions();
  *global_mapper = GlobalPipelineOptions();
  *gravity_refiner = GravityRefinerOptions();
  *reconstruction_clusterer = ReconstructionClusteringOptions();
  *patch_match_stereo = mvs::PatchMatchOptions();
  *stereo_fusion = mvs::StereoFusionOptions();
  *poisson_meshing = mvs::PoissonMeshingOptions();
  *delaunay_meshing = mvs::DelaunayMeshingOptions();
  *render = RenderOptions();
}

bool OptionManager::Check() {
  if (!BaseOptionManager::Check()) {
    return false;
  }

  bool success = true;

  if (image_reader) success = success && image_reader->Check();
  if (feature_extraction) success = success && feature_extraction->Check();

  if (feature_matching) success = success && feature_matching->Check();
  if (two_view_geometry) success = success && two_view_geometry->Check();
  if (exhaustive_pairing) success = success && exhaustive_pairing->Check();
  if (sequential_pairing) success = success && sequential_pairing->Check();
  if (vocab_tree_pairing) success = success && vocab_tree_pairing->Check();
  if (spatial_pairing) success = success && spatial_pairing->Check();
  if (transitive_pairing) success = success && transitive_pairing->Check();
  if (imported_pairing) success = success && imported_pairing->Check();

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

bool OptionManager::Read(const std::filesystem::path& path,
                         bool allow_unregistered) {
  if (!BaseOptionManager::Read(path, allow_unregistered)) {
    return false;
  }
  return Check();
}

void OptionManager::PostParse() {
  if (!mapper_image_list_path_.empty()) {
    mapper->image_names = ReadTextFileLines(mapper_image_list_path_);
  }
  if (!global_mapper_image_list_path_.empty()) {
    global_mapper->image_names =
        ReadTextFileLines(global_mapper_image_list_path_);
  }
  if (!mapper_constant_rig_list_path_.empty()) {
    for (const std::string& line :
         ReadTextFileLines(mapper_constant_rig_list_path_)) {
      mapper->constant_rigs.insert(std::stoi(line));
    }
  }
  if (!mapper_constant_camera_list_path_.empty()) {
    for (const std::string& line :
         ReadTextFileLines(mapper_constant_camera_list_path_)) {
      mapper->constant_cameras.insert(std::stoi(line));
    }
  }
}

void OptionManager::PrintHelp() const {
  LOG(INFO) << StringPrintf(
      "%s (%s)", GetVersionInfo().c_str(), GetBuildInfo().c_str());
  LOG(INFO) << "Options can either be specified via command-line or by "
               "defining them in a .ini project file passed to "
               "`--project_path`.\n"
            << *desc_;
}

}  // namespace colmap
