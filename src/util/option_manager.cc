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

#include "util/option_manager.h"

#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "util/misc.h"
#include "util/version.h"

namespace config = boost::program_options;

namespace colmap {

OptionManager::OptionManager() {
  project_path.reset(new std::string());
  database_path.reset(new std::string());
  image_path.reset(new std::string());

  image_reader.reset(new ImageReader::Options());
  sift_extraction.reset(new SiftExtractionOptions());
  sift_cpu_extraction.reset(new SiftCPUFeatureExtractor::Options());
  sift_gpu_extraction.reset(new SiftGPUFeatureExtractor::Options());
  sift_matching.reset(new SiftMatchingOptions());
  exhaustive_matching.reset(new ExhaustiveFeatureMatcher::Options());
  sequential_matching.reset(new SequentialFeatureMatcher::Options());
  vocab_tree_matching.reset(new VocabTreeFeatureMatcher::Options());
  spatial_matching.reset(new SpatialFeatureMatcher::Options());
  transitive_matching.reset(new TransitiveFeatureMatcher::Options());
  bundle_adjustment.reset(new BundleAdjuster::Options());
  mapper.reset(new IncrementalMapperController::Options());
  dense_stereo.reset(new mvs::PatchMatch::Options());
  dense_fusion.reset(new mvs::StereoFusion::Options());
  dense_meshing.reset(new mvs::PoissonReconstructionOptions());
  render.reset(new RenderOptions());

  Reset();

  desc_->add_options()("help,h", "");
  desc_->add_options()("project_path", config::value<std::string>());
}

void OptionManager::InitForIndividualData() {
  mapper->max_extra_param = std::numeric_limits<double>::max();
}

void OptionManager::InitForVideoData() {
  mapper->mapper.init_min_tri_angle /= 2;
  mapper->ba_global_images_ratio = 1.4;
  mapper->ba_global_points_ratio = 1.4;
  mapper->max_extra_param = std::numeric_limits<double>::max();
  dense_fusion->min_num_pixels = 15;
}

void OptionManager::InitForInternetData() { dense_fusion->min_num_pixels = 10; }

void OptionManager::AddAllOptions() {
  AddLogOptions();
  AddDatabaseOptions();
  AddImageOptions();
  AddExtractionOptions();
  AddMatchingOptions();
  AddExhaustiveMatchingOptions();
  AddSequentialMatchingOptions();
  AddVocabTreeMatchingOptions();
  AddSpatialMatchingOptions();
  AddTransitiveMatchingOptions();
  AddBundleAdjustmentOptions();
  AddMapperOptions();
  AddDenseStereoOptions();
  AddDenseFusionOptions();
  AddDenseMeshingOptions();
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

  AddAndRegisterDefaultOption("ImageReader.camera_model",
                              &image_reader->camera_model);
  AddAndRegisterDefaultOption("ImageReader.single_camera",
                              &image_reader->single_camera);
  AddAndRegisterDefaultOption("ImageReader.camera_params",
                              &image_reader->camera_params);
  AddAndRegisterDefaultOption("ImageReader.default_focal_length_factor",
                              &image_reader->default_focal_length_factor);

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
  AddAndRegisterDefaultOption("SiftExtraction.max_num_orientations",
                              &sift_extraction->max_num_orientations);
  AddAndRegisterDefaultOption("SiftExtraction.upright",
                              &sift_extraction->upright);

  AddAndRegisterDefaultOption("SiftCPUExtraction.batch_size_factor",
                              &sift_cpu_extraction->batch_size_factor);
  AddAndRegisterDefaultOption("SiftCPUExtraction.num_threads",
                              &sift_cpu_extraction->num_threads);

  AddAndRegisterDefaultOption("SiftGPUExtraction.index",
                              &sift_gpu_extraction->index);
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
  AddAndRegisterDefaultOption("SiftMatching.max_error",
                              &sift_matching->max_error);
  AddAndRegisterDefaultOption("SiftMatching.max_num_matches",
                              &sift_matching->max_num_matches);
  AddAndRegisterDefaultOption("SiftMatching.confidence",
                              &sift_matching->confidence);
  AddAndRegisterDefaultOption("SiftMatching.max_num_trials",
                              &sift_matching->max_num_trials);
  AddAndRegisterDefaultOption("SiftMatching.min_inlier_ratio",
                              &sift_matching->min_inlier_ratio);
  AddAndRegisterDefaultOption("SiftMatching.min_num_inliers",
                              &sift_matching->min_num_inliers);
  AddAndRegisterDefaultOption("SiftMatching.multiple_models",
                              &sift_matching->multiple_models);
  AddAndRegisterDefaultOption("SiftMatching.guided_matching",
                              &sift_matching->guided_matching);
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
  AddAndRegisterDefaultOption("SequentialMatching.loop_detection",
                              &sequential_matching->loop_detection);
  AddAndRegisterDefaultOption("SequentialMatching.loop_detection_period",
                              &sequential_matching->loop_detection_period);
  AddAndRegisterDefaultOption("SequentialMatching.loop_detection_num_images",
                              &sequential_matching->loop_detection_num_images);
  AddAndRegisterDefaultOption(
      "SequentialMatching.loop_detection_num_verifications",
      &sequential_matching->loop_detection_num_verifications);
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
  AddAndRegisterDefaultOption("VocabTreeMatching.num_verifications",
                              &vocab_tree_matching->num_verifications);
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

  AddAndRegisterDefaultOption("SpatialMatching.is_gps",
                              &spatial_matching->is_gps);
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
  AddAndRegisterDefaultOption("Mapper.ba_local_max_num_iterations",
                              &mapper->ba_local_max_num_iterations);
  AddAndRegisterDefaultOption("Mapper.ba_global_use_pba",
                              &mapper->ba_global_use_pba);
  AddAndRegisterDefaultOption("Mapper.ba_global_pba_gpu_index",
                              &mapper->ba_global_pba_gpu_index);
  AddAndRegisterDefaultOption("Mapper.ba_global_images_ratio",
                              &mapper->ba_global_images_ratio);
  AddAndRegisterDefaultOption("Mapper.ba_global_points_ratio",
                              &mapper->ba_global_points_ratio);
  AddAndRegisterDefaultOption("Mapper.ba_global_images_freq",
                              &mapper->ba_global_images_freq);
  AddAndRegisterDefaultOption("Mapper.ba_global_points_freq",
                              &mapper->ba_global_points_freq);
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
  AddAndRegisterDefaultOption("Mapper.snapshot_path", &mapper->snapshot_path);
  AddAndRegisterDefaultOption("Mapper.snapshot_images_freq",
                              &mapper->snapshot_images_freq);

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

void OptionManager::AddDenseStereoOptions() {
  if (added_dense_stereo_options_) {
    return;
  }
  added_dense_stereo_options_ = true;

  AddAndRegisterDefaultOption("DenseStereo.max_image_size",
                              &dense_stereo->max_image_size);
  AddAndRegisterDefaultOption("DenseStereo.gpu_index",
                              &dense_stereo->gpu_index);
  AddAndRegisterDefaultOption("DenseStereo.window_radius",
                              &dense_stereo->window_radius);
  AddAndRegisterDefaultOption("DenseStereo.sigma_spatial",
                              &dense_stereo->sigma_spatial);
  AddAndRegisterDefaultOption("DenseStereo.sigma_color",
                              &dense_stereo->sigma_color);
  AddAndRegisterDefaultOption("DenseStereo.num_samples",
                              &dense_stereo->num_samples);
  AddAndRegisterDefaultOption("DenseStereo.ncc_sigma",
                              &dense_stereo->ncc_sigma);
  AddAndRegisterDefaultOption("DenseStereo.min_triangulation_angle",
                              &dense_stereo->min_triangulation_angle);
  AddAndRegisterDefaultOption("DenseStereo.incident_angle_sigma",
                              &dense_stereo->incident_angle_sigma);
  AddAndRegisterDefaultOption("DenseStereo.num_iterations",
                              &dense_stereo->num_iterations);
  AddAndRegisterDefaultOption("DenseStereo.geom_consistency",
                              &dense_stereo->geom_consistency);
  AddAndRegisterDefaultOption("DenseStereo.geom_consistency_regularizer",
                              &dense_stereo->geom_consistency_regularizer);
  AddAndRegisterDefaultOption("DenseStereo.geom_consistency_max_cost",
                              &dense_stereo->geom_consistency_max_cost);
  AddAndRegisterDefaultOption("DenseStereo.filter", &dense_stereo->filter);
  AddAndRegisterDefaultOption("DenseStereo.filter_min_ncc",
                              &dense_stereo->filter_min_ncc);
  AddAndRegisterDefaultOption("DenseStereo.filter_min_triangulation_angle",
                              &dense_stereo->filter_min_triangulation_angle);
  AddAndRegisterDefaultOption("DenseStereo.filter_min_num_consistent",
                              &dense_stereo->filter_min_num_consistent);
  AddAndRegisterDefaultOption("DenseStereo.filter_geom_consistency_max_cost",
                              &dense_stereo->filter_geom_consistency_max_cost);
  AddAndRegisterDefaultOption("DenseStereo.cache_size",
                              &dense_stereo->cache_size);
  AddAndRegisterDefaultOption("DenseStereo.write_consistency_graph",
                              &dense_stereo->write_consistency_graph);
}

void OptionManager::AddDenseFusionOptions() {
  if (added_dense_fusion_options_) {
    return;
  }
  added_dense_fusion_options_ = true;

  AddAndRegisterDefaultOption("DenseFusion.max_image_size",
                              &dense_fusion->max_image_size);
  AddAndRegisterDefaultOption("DenseFusion.min_num_pixels",
                              &dense_fusion->min_num_pixels);
  AddAndRegisterDefaultOption("DenseFusion.max_num_pixels",
                              &dense_fusion->max_num_pixels);
  AddAndRegisterDefaultOption("DenseFusion.max_traversal_depth",
                              &dense_fusion->max_traversal_depth);
  AddAndRegisterDefaultOption("DenseFusion.max_reproj_error",
                              &dense_fusion->max_reproj_error);
  AddAndRegisterDefaultOption("DenseFusion.max_depth_error",
                              &dense_fusion->max_depth_error);
  AddAndRegisterDefaultOption("DenseFusion.max_normal_error",
                              &dense_fusion->max_normal_error);
  AddAndRegisterDefaultOption("DenseFusion.check_num_images",
                              &dense_fusion->check_num_images);
  AddAndRegisterDefaultOption("DenseFusion.cache_size",
                              &dense_fusion->cache_size);
}

void OptionManager::AddDenseMeshingOptions() {
  if (added_dense_meshing_options_) {
    return;
  }
  added_dense_meshing_options_ = true;

  AddAndRegisterDefaultOption("DenseMeshing.point_weight",
                              &dense_meshing->point_weight);
  AddAndRegisterDefaultOption("DenseMeshing.depth", &dense_meshing->depth);
  AddAndRegisterDefaultOption("DenseMeshing.color", &dense_meshing->color);
  AddAndRegisterDefaultOption("DenseMeshing.trim", &dense_meshing->trim);
  AddAndRegisterDefaultOption("DenseMeshing.num_threads",
                              &dense_meshing->num_threads);
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
  FLAGS_logtostderr = false;
  FLAGS_v = 2;

  *project_path = "";
  *database_path = "";
  *image_path = "";

  *image_reader = ImageReader::Options();
  *sift_extraction = SiftExtractionOptions();
  *sift_cpu_extraction = SiftCPUFeatureExtractor::Options();
  *sift_gpu_extraction = SiftGPUFeatureExtractor::Options();
  *sift_matching = SiftMatchingOptions();
  *exhaustive_matching = ExhaustiveFeatureMatcher::Options();
  *sequential_matching = SequentialFeatureMatcher::Options();
  *vocab_tree_matching = VocabTreeFeatureMatcher::Options();
  *spatial_matching = SpatialFeatureMatcher::Options();
  *transitive_matching = TransitiveFeatureMatcher::Options();
  *bundle_adjustment = BundleAdjuster::Options();
  *mapper = IncrementalMapperController::Options();
  *dense_stereo = mvs::PatchMatch::Options();
  *dense_fusion = mvs::StereoFusion::Options();
  *dense_meshing = mvs::PoissonReconstructionOptions();
  *render = RenderOptions();

  desc_.reset(new boost::program_options::options_description());

  options_bool_.clear();
  options_int_.clear();
  options_double_.clear();
  options_string_.clear();

  added_log_options_ = false;
  added_database_options_ = false;
  added_image_options_ = false;
  added_extraction_options_ = false;
  added_match_options_ = false;
  added_exhaustive_match_options_ = false;
  added_sequential_match_options_ = false;
  added_vocab_tree_match_options_ = false;
  added_spatial_match_options_ = false;
  added_transitive_match_options_ = false;
  added_ba_options_ = false;
  added_mapper_options_ = false;
  added_dense_stereo_options_ = false;
  added_dense_fusion_options_ = false;
  added_dense_meshing_options_ = false;
  added_render_options_ = false;
}

bool OptionManager::Check() {
  bool success = true;

  if (added_database_options_) {
    const auto database_parent_path = GetParentDir(*database_path);
    success = success && !ExistsDir(*database_path) &&
              (database_parent_path == "" || ExistsDir(database_parent_path));
  }

  if (added_image_options_) success = success && ExistsDir(*image_path);

  if (image_reader) success = success && image_reader->Check();
  if (sift_extraction) success = success && sift_extraction->Check();
  if (sift_cpu_extraction) success = success && sift_cpu_extraction->Check();
  if (sift_gpu_extraction) success = success && sift_gpu_extraction->Check();

  if (sift_matching) success = success && sift_matching->Check();
  if (exhaustive_matching) success = success && exhaustive_matching->Check();
  if (sequential_matching) success = success && sequential_matching->Check();
  if (vocab_tree_matching) success = success && vocab_tree_matching->Check();
  if (spatial_matching) success = success && spatial_matching->Check();

  if (bundle_adjustment) success = success && bundle_adjustment->Check();
  if (mapper) success = success && mapper->Check();

  if (dense_stereo) success = success && dense_stereo->Check();
  if (dense_fusion) success = success && dense_fusion->Check();
  if (dense_meshing) success = success && dense_meshing->Check();

  if (render) success = success && render->Check();

  return success;
}

void OptionManager::Parse(const int argc, char** argv) {
  config::variables_map vmap;

  try {
    config::store(config::parse_command_line(argc, argv, *desc_), vmap);

    if (vmap.count("help")) {
      std::cout << StringPrintf("%s (%s)", GetVersionInfo().c_str(),
                                GetBuildInfo().c_str())
                << std::endl
                << std::endl;
      std::cout
          << "Options can either be specified via command-line or by defining"
          << std::endl
          << "them in a .ini project file passed to `--project_path`."
          << std::endl
          << std::endl;
      std::cout << *desc_ << std::endl;
      exit(EXIT_SUCCESS);
    }

    if (vmap.count("project_path")) {
      *project_path = vmap["project_path"].as<std::string>();
      if (!Read(*project_path)) {
        exit(EXIT_FAILURE);
      }
    } else {
      vmap.notify();
    }
  } catch (std::exception& e) {
    std::cerr << "ERROR: Failed to parse options: " << e.what() << "."
              << std::endl;
    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "ERROR: Failed to parse options for unknown reason."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!Check()) {
    std::cerr << "ERROR: Invalid options provided." << std::endl;
    exit(EXIT_FAILURE);
  }
}

bool OptionManager::Read(const std::string& path) {
  config::variables_map vmap;

  if (!ExistsFile(path)) {
    std::cout << "ERROR: Configuration file does not exist." << std::endl;
    return false;
  }

  try {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;
    config::store(config::parse_config_file(file, *desc_), vmap);
    vmap.notify();
  } catch (std::exception& e) {
    std::cout << "ERROR: Failed to parse options " << e.what() << "."
              << std::endl;
    return false;
  } catch (...) {
    std::cout << "ERROR: Failed to parse options for unknown reason."
              << std::endl;
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

  for (const auto& option : options_bool_) {
    pt.put(option.first, *option.second);
  }

  for (const auto& option : options_int_) {
    pt.put(option.first, *option.second);
  }

  for (const auto& option : options_double_) {
    pt.put(option.first, *option.second);
  }

  for (const auto& option : options_string_) {
    pt.put(option.first, *option.second);
  }

  boost::property_tree::write_ini(path, pt);
}

}  // namespace colmap
