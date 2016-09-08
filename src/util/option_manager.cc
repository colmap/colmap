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

#include "util/option_manager.h"

#include <iostream>

#include <glog/logging.h>

#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "base/camera_models.h"
#include "util/misc.h"

namespace config = boost::program_options;

#ifndef CHECK_OPTION
#define CHECK_OPTION(option_class, option, expression)                   \
  verified = verified && CheckOption((option expression), #option_class, \
                                     #option, #expression)
#endif

#ifndef VERIFY_OPTION_MSG
#define VERIFY_OPTION_MSG(option_class, option, expression, error) \
  verified =                                                       \
      verified && CheckOption((expression), #option_class, #option, error)
#endif

#ifndef ADD_OPTION_REQUIRED
#define ADD_OPTION_REQUIRED(option_class, option_attr, option)             \
  {                                                                        \
    std::string option_str = #option;                                      \
    option_str = StringReplace(option_str, ".", "_");                      \
    const std::string option_name =                                        \
        std::string(#option_class) + "." + option_str;                     \
    desc_->add_options()(                                                  \
        option_name.c_str(),                                               \
        config::value<decltype(option_attr->option)>(&option_attr->option) \
            ->required());                                                 \
    RegisterOption(option_name, &option_attr->option);                     \
  }
#endif

#ifndef ADD_OPTION_DEFAULT
#define ADD_OPTION_DEFAULT(option_class, option_attr, option)              \
  {                                                                        \
    std::string option_str = #option;                                      \
    option_str = StringReplace(option_str, ".", "_");                      \
    const std::string option_name =                                        \
        std::string(#option_class) + "." + option_str;                     \
    desc_->add_options()(                                                  \
        option_name.c_str(),                                               \
        config::value<decltype(option_attr->option)>(&option_attr->option) \
            ->default_value(option_class().option));                       \
    RegisterOption(option_name, &option_attr->option);                     \
  }
#endif

namespace colmap {
namespace {

bool CheckOption(const bool value, const std::string& option_class,
                 const std::string& option, const std::string& expression) {
  if (!value) {
    std::cerr << StringPrintf("ERROR: Option %s.%s failed check - %s",
                              option_class.c_str(), option.c_str(),
                              expression.c_str())
              << std::endl;
  }
  return value;
}

}  // namespace

void BaseOptions::Reset() {}

bool BaseOptions::Check() { return false; }

ExtractionOptions::ExtractionOptions() { Reset(); }

void ExtractionOptions::Reset() {
  reader = ImageReader::Options();
  sift = SiftOptions();
  cpu = SiftCPUFeatureExtractor::Options();
}

bool ExtractionOptions::Check() {
  bool verified = true;

  CHECK_OPTION(ExtractionOptions, reader.default_focal_length_factor, > 0);

  if (!reader.camera_model.empty()) {
    const auto model_id = CameraModelNameToId(reader.camera_model);
    VERIFY_OPTION_MSG(ExtractionOptions, reader.camera_model, model_id != -1,
                      "Camera model does not exist");

    if (!reader.camera_params.empty()) {
      const auto camera_params_vector =
          CSVToVector<double>(reader.camera_params);
      VERIFY_OPTION_MSG(ExtractionOptions, reader.camera_params,
                        CameraModelVerifyParams(model_id, camera_params_vector),
                        "Invalid camera parameters");
    }
  }

  CHECK_OPTION(ExtractionOptions, sift.max_image_size, > 0);
  CHECK_OPTION(ExtractionOptions, sift.max_num_features, > 0);
  CHECK_OPTION(ExtractionOptions, sift.octave_resolution, > 0);
  CHECK_OPTION(ExtractionOptions, sift.peak_threshold, > 0);
  CHECK_OPTION(ExtractionOptions, sift.edge_threshold, > 0);
  CHECK_OPTION(ExtractionOptions, sift.max_num_orientations, > 0);
  CHECK_OPTION(ExtractionOptions, cpu.batch_size_factor, > 0);

  return verified;
}

MatchOptions::MatchOptions() { Reset(); }

void MatchOptions::Reset() {
  SiftMatchOptions options;
  num_threads = options.num_threads;
  gpu_index = options.gpu_index;
  max_ratio = options.max_ratio;
  max_distance = options.max_distance;
  cross_check = options.cross_check;
  max_num_matches = options.max_num_matches;
  max_error = options.max_error;
  confidence = options.confidence;
  max_num_trials = options.max_num_trials;
  min_inlier_ratio = options.min_inlier_ratio;
  min_num_inliers = options.min_num_inliers;
  multiple_models = options.multiple_models;
  guided_matching = options.guided_matching;
}

bool MatchOptions::Check() {
  bool verified = true;

  CHECK_OPTION(MatchOptions, num_threads, >= -1);
  CHECK_OPTION(MatchOptions, gpu_index, >= -1);
  CHECK_OPTION(MatchOptions, max_ratio, >= 0);
  CHECK_OPTION(MatchOptions, max_ratio, <= 1);
  CHECK_OPTION(MatchOptions, max_distance, >= 0);
  CHECK_OPTION(MatchOptions, max_error, >= 0);
  CHECK_OPTION(MatchOptions, max_num_matches, >= 0);
  CHECK_OPTION(MatchOptions, confidence, <= 1);
  CHECK_OPTION(MatchOptions, confidence, >= 0);
  CHECK_OPTION(MatchOptions, max_num_trials, > 0);
  CHECK_OPTION(MatchOptions, min_inlier_ratio, >= 0);
  CHECK_OPTION(MatchOptions, min_inlier_ratio, <= 1);
  CHECK_OPTION(MatchOptions, min_num_inliers, >= 0);

  return verified;
}

SiftMatchOptions MatchOptions::Options() const {
  SiftMatchOptions options;
  options.num_threads = num_threads;
  options.gpu_index = gpu_index;
  options.max_ratio = max_ratio;
  options.max_distance = max_distance;
  options.cross_check = cross_check;
  options.max_num_matches = max_num_matches;
  options.max_error = max_error;
  options.confidence = confidence;
  options.max_num_trials = max_num_trials;
  options.min_inlier_ratio = min_inlier_ratio;
  options.min_num_inliers = min_num_inliers;
  options.multiple_models = multiple_models;
  options.guided_matching = guided_matching;
  return options;
}

ExhaustiveMatchOptions::ExhaustiveMatchOptions() { Reset(); }

void ExhaustiveMatchOptions::Reset() {
  ExhaustiveFeatureMatcher::Options options;
  block_size = options.block_size;
  preemptive = options.preemptive;
  preemptive_num_features = options.preemptive_num_features;
  preemptive_min_num_matches = options.preemptive_min_num_matches;
}

bool ExhaustiveMatchOptions::Check() {
  bool verified = true;

  CHECK_OPTION(ExhaustiveMatchOptions, block_size, > 1);
  CHECK_OPTION(ExhaustiveMatchOptions, preemptive_num_features, > 0);
  CHECK_OPTION(ExhaustiveMatchOptions, preemptive_min_num_matches, > 0);
  CHECK_OPTION(ExhaustiveMatchOptions, preemptive_min_num_matches,
               <= preemptive_num_features);

  return verified;
}

ExhaustiveFeatureMatcher::Options ExhaustiveMatchOptions::Options() const {
  ExhaustiveFeatureMatcher::Options options;
  options.block_size = block_size;
  options.preemptive = preemptive;
  options.preemptive_num_features = preemptive_num_features;
  options.preemptive_min_num_matches = preemptive_min_num_matches;
  return options;
}

SequentialMatchOptions::SequentialMatchOptions() { Reset(); }

void SequentialMatchOptions::Reset() {
  SequentialFeatureMatcher::Options options;
  overlap = options.overlap;
  loop_detection = options.loop_detection;
  loop_detection_period = options.loop_detection_period;
  loop_detection_num_images = options.loop_detection_num_images;
  vocab_tree_path = options.vocab_tree_path;
}

bool SequentialMatchOptions::Check() {
  bool verified = true;

  CHECK_OPTION(SequentialMatchOptions, loop_detection_period, > 0);
  CHECK_OPTION(SequentialMatchOptions, loop_detection_num_images, > 0);

  return verified;
}

SequentialFeatureMatcher::Options SequentialMatchOptions::Options() const {
  SequentialFeatureMatcher::Options options;
  options.overlap = overlap;
  options.loop_detection = loop_detection;
  options.loop_detection_period = loop_detection_period;
  options.loop_detection_num_images = loop_detection_num_images;
  options.vocab_tree_path = vocab_tree_path;
  return options;
}

VocabTreeMatchOptions::VocabTreeMatchOptions() { Reset(); }

void VocabTreeMatchOptions::Reset() {
  VocabTreeFeatureMatcher::Options options;
  num_images = options.num_images;
  vocab_tree_path = options.vocab_tree_path;
  match_list_path = options.match_list_path;
}

bool VocabTreeMatchOptions::Check() {
  bool verified = true;

  CHECK_OPTION(VocabTreeMatchOptions, num_images, > 0);

  return verified;
}

VocabTreeFeatureMatcher::Options VocabTreeMatchOptions::Options() const {
  VocabTreeFeatureMatcher::Options options;
  options.num_images = num_images;
  options.vocab_tree_path = vocab_tree_path;
  options.match_list_path = match_list_path;
  return options;
}

SpatialMatchOptions::SpatialMatchOptions() { Reset(); }

void SpatialMatchOptions::Reset() {
  SpatialFeatureMatcher::Options options;
  is_gps = options.is_gps;
  ignore_z = options.ignore_z;
  max_num_neighbors = options.max_num_neighbors;
  max_distance = options.max_distance;
}

bool SpatialMatchOptions::Check() {
  bool verified = true;

  CHECK_OPTION(SpatialMatchOptions, max_num_neighbors, > 0);
  CHECK_OPTION(SpatialMatchOptions, max_distance, > 0);

  return verified;
}

SpatialFeatureMatcher::Options SpatialMatchOptions::Options() const {
  SpatialFeatureMatcher::Options options;
  options.is_gps = is_gps;
  options.ignore_z = ignore_z;
  options.max_num_neighbors = max_num_neighbors;
  options.max_distance = max_distance;
  return options;
}

BundleAdjustmentOptions::BundleAdjustmentOptions() { Reset(); }

void BundleAdjustmentOptions::Reset() {
  max_num_iterations = 100;
  max_linear_solver_iterations = 200;

  function_tolerance = 0;
  gradient_tolerance = 0;
  parameter_tolerance = 0;

  max_num_consecutive_invalid_steps = 10;
  max_consecutive_nonmonotonic_steps = 10;
  use_nonmonotonic_steps = false;

  minimizer_progress_to_stdout = false;

  BundleAdjuster::Options options;
  loss_function_scale = options.loss_function_scale;
  refine_focal_length = options.refine_focal_length;
  refine_principal_point = options.refine_principal_point;
  refine_extra_params = options.refine_extra_params;
  print_summary = options.print_summary;
}

bool BundleAdjustmentOptions::Check() {
  bool verified = true;

  CHECK_OPTION(BundleAdjustmentOptions, max_num_iterations, >= 0);
  CHECK_OPTION(BundleAdjustmentOptions, max_linear_solver_iterations, >= 0);
  CHECK_OPTION(BundleAdjustmentOptions, function_tolerance, >= 0);
  CHECK_OPTION(BundleAdjustmentOptions, gradient_tolerance, >= 0);
  CHECK_OPTION(BundleAdjustmentOptions, parameter_tolerance, >= 0);

  return verified;
}

BundleAdjuster::Options BundleAdjustmentOptions::Options() const {
  BundleAdjuster::Options options;
  options.solver_options.max_num_iterations = max_num_iterations;
  options.solver_options.max_linear_solver_iterations =
      max_linear_solver_iterations;
  options.solver_options.function_tolerance = function_tolerance;
  options.solver_options.gradient_tolerance = gradient_tolerance;
  options.solver_options.parameter_tolerance = parameter_tolerance;
  options.solver_options.max_num_consecutive_invalid_steps =
      max_num_consecutive_invalid_steps;
  options.solver_options.max_consecutive_nonmonotonic_steps =
      max_consecutive_nonmonotonic_steps;
  options.solver_options.use_nonmonotonic_steps = use_nonmonotonic_steps;
  options.solver_options.minimizer_progress_to_stdout =
      minimizer_progress_to_stdout;
  options.loss_function_scale = loss_function_scale;
  options.refine_focal_length = refine_focal_length;
  options.refine_principal_point = refine_principal_point;
  options.refine_extra_params = refine_extra_params;
  options.print_summary = print_summary;
  return options;
}

TriangulationOptions::TriangulationOptions() { Reset(); }

void TriangulationOptions::Reset() {
  IncrementalTriangulator::Options options;
  max_transitivity = options.max_transitivity;
  create_max_angle_error = options.create_max_angle_error;
  continue_max_angle_error = options.continue_max_angle_error;
  merge_max_reproj_error = options.merge_max_reproj_error;
  re_max_angle_error = options.re_max_angle_error;
  re_min_ratio = options.re_min_ratio;
  re_max_trials = options.re_max_trials;
  complete_max_reproj_error = options.complete_max_reproj_error;
  complete_max_transitivity = options.complete_max_transitivity;
  min_angle = options.min_angle;
  ignore_two_view_tracks = options.ignore_two_view_tracks;
}

bool TriangulationOptions::Check() {
  bool verified = true;

  CHECK_OPTION(TriangulationOptions, max_transitivity, >= 0);
  CHECK_OPTION(TriangulationOptions, create_max_angle_error, > 0);
  CHECK_OPTION(TriangulationOptions, continue_max_angle_error, > 0);
  CHECK_OPTION(TriangulationOptions, merge_max_reproj_error, > 0);
  CHECK_OPTION(TriangulationOptions, complete_max_reproj_error, > 0);
  CHECK_OPTION(TriangulationOptions, complete_max_transitivity, >= 0);
  CHECK_OPTION(TriangulationOptions, re_max_angle_error, > 0);
  CHECK_OPTION(TriangulationOptions, re_min_ratio, >= 0);
  CHECK_OPTION(TriangulationOptions, re_min_ratio, <= 1);
  CHECK_OPTION(TriangulationOptions, re_max_trials, >= 0);
  CHECK_OPTION(TriangulationOptions, min_angle, > 0);

  return verified;
}

IncrementalTriangulator::Options TriangulationOptions::Options() const {
  IncrementalTriangulator::Options options;
  options.max_transitivity = max_transitivity;
  options.create_max_angle_error = create_max_angle_error;
  options.continue_max_angle_error = continue_max_angle_error;
  options.merge_max_reproj_error = merge_max_reproj_error;
  options.complete_max_reproj_error = complete_max_reproj_error;
  options.complete_max_transitivity = complete_max_transitivity;
  options.re_max_angle_error = re_max_angle_error;
  options.re_min_ratio = re_min_ratio;
  options.re_max_trials = re_max_trials;
  options.min_angle = min_angle;
  options.ignore_two_view_tracks = ignore_two_view_tracks;
  return options;
}

IncrementalMapperOptions::IncrementalMapperOptions() { Reset(); }

void IncrementalMapperOptions::Reset() {
  IncrementalMapper::Options options;
  init_min_num_inliers = options.init_min_num_inliers;
  init_max_error = options.init_max_error;
  init_max_forward_motion = options.init_max_forward_motion;
  init_min_tri_angle = options.init_min_tri_angle;
  abs_pose_max_error = options.abs_pose_max_error;
  abs_pose_min_num_inliers = options.abs_pose_min_num_inliers;
  abs_pose_min_inlier_ratio = options.abs_pose_min_inlier_ratio;
  filter_max_reproj_error = options.filter_max_reproj_error;
  filter_min_tri_angle = options.filter_min_tri_angle;
  max_reg_trials = options.max_reg_trials;
}

bool IncrementalMapperOptions::Check() {
  bool verified = true;

  CHECK_OPTION(MapperOptions, init_min_num_inliers, > 0);
  CHECK_OPTION(MapperOptions, init_max_error, > 0);
  CHECK_OPTION(MapperOptions, init_max_forward_motion, >= 0);
  CHECK_OPTION(MapperOptions, init_max_forward_motion, <= 1);
  CHECK_OPTION(MapperOptions, init_min_tri_angle, > 0);

  CHECK_OPTION(MapperOptions, abs_pose_max_error, > 0);
  CHECK_OPTION(MapperOptions, abs_pose_min_num_inliers, > 0);
  CHECK_OPTION(MapperOptions, abs_pose_min_inlier_ratio, >= 0);
  CHECK_OPTION(MapperOptions, abs_pose_min_inlier_ratio, <= 1);

  CHECK_OPTION(MapperOptions, filter_max_reproj_error, > 0);
  CHECK_OPTION(MapperOptions, filter_min_tri_angle, > 0);

  CHECK_OPTION(MapperOptions, max_reg_trials, > 0);

  return verified;
}

IncrementalMapper::Options IncrementalMapperOptions::Options() const {
  IncrementalMapper::Options options;
  options.init_min_num_inliers = init_min_num_inliers;
  options.init_max_error = init_max_error;
  options.init_max_forward_motion = init_max_forward_motion;
  options.init_min_tri_angle = init_min_tri_angle;
  options.abs_pose_max_error = abs_pose_max_error;
  options.abs_pose_min_num_inliers = abs_pose_min_num_inliers;
  options.abs_pose_min_inlier_ratio = abs_pose_min_inlier_ratio;
  options.filter_max_reproj_error = filter_max_reproj_error;
  options.filter_min_tri_angle = filter_min_tri_angle;
  options.max_reg_trials = max_reg_trials;
  return options;
}

MapperOptions::MapperOptions() { Reset(); }

void MapperOptions::Reset() {
  min_num_matches = 15;
  ignore_watermarks = false;

  multiple_models = true;
  max_num_models = 50;
  max_model_overlap = 20;
  min_model_size = 10;

  init_image_id1 = -1;
  init_image_id2 = -1;
  init_num_trials = 200;

  extract_colors = true;

  num_threads = -1;

  min_focal_length_ratio = 0.1;   // Opening angle of ~130deg
  max_focal_length_ratio = 10.0;  // Opening angle of ~5deg
  max_extra_param = 1.0;

  ba_local_num_images = 6;
  ba_local_max_num_iterations = 25;

  ba_global_use_pba = true;
  ba_global_images_ratio = 1.05;
  ba_global_points_ratio = 1.05;
  ba_global_images_freq = 500;
  ba_global_points_freq = 50000;
  ba_global_max_num_iterations = 50;
  ba_global_pba_gpu_index = -1;

  ba_refine_focal_length = true;
  ba_refine_principal_point = false;
  ba_refine_extra_params = true;

  ba_local_max_refinements = 3;
  ba_local_max_refinement_change = 0.001;
  ba_global_max_refinements = 5;
  ba_global_max_refinement_change = 0.0005;

  incremental_mapper.Reset();
  triangulation.Reset();
}

bool MapperOptions::Check() {
  bool verified = true;

  CHECK_OPTION(MapperOptions, min_num_matches, > 0);
  CHECK_OPTION(MapperOptions, max_num_models, > 0);
  CHECK_OPTION(MapperOptions, max_model_overlap, > 0);
  CHECK_OPTION(MapperOptions, min_model_size, >= 0);

  CHECK_OPTION(MapperOptions, init_num_trials, > 0);

  CHECK_OPTION(MapperOptions, min_focal_length_ratio, > 0);
  CHECK_OPTION(MapperOptions, max_focal_length_ratio, > 0);
  CHECK_OPTION(MapperOptions, max_extra_param, >= 0);

  CHECK_OPTION(MapperOptions, ba_local_num_images, >= 2);
  CHECK_OPTION(MapperOptions, ba_local_max_num_iterations, >= 0);

  CHECK_OPTION(MapperOptions, ba_global_images_ratio, > 1.0);
  CHECK_OPTION(MapperOptions, ba_global_points_ratio, > 1.0);
  CHECK_OPTION(MapperOptions, ba_global_images_freq, > 0);
  CHECK_OPTION(MapperOptions, ba_global_points_freq, > 0);
  CHECK_OPTION(MapperOptions, ba_global_max_num_iterations, > 0);

  CHECK_OPTION(MapperOptions, ba_local_max_refinements, > 0);
  CHECK_OPTION(MapperOptions, ba_local_max_refinement_change, >= 0);
  CHECK_OPTION(MapperOptions, ba_global_max_refinements, > 0);
  CHECK_OPTION(MapperOptions, ba_global_max_refinement_change, >= 0);

  verified = verified && incremental_mapper.Check();
  verified = verified && triangulation.Check();

  return verified;
}

IncrementalMapper::Options MapperOptions::IncrementalMapperOptions() const {
  IncrementalMapper::Options options = incremental_mapper.Options();
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  return options;
}

IncrementalTriangulator::Options MapperOptions::TriangulationOptions() const {
  IncrementalTriangulator::Options options = triangulation.Options();
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjuster::Options MapperOptions::LocalBundleAdjustmentOptions() const {
  BundleAdjuster::Options options;
  options.solver_options.function_tolerance = 0.0;
  options.solver_options.gradient_tolerance = 10.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_local_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.solver_options.num_threads = num_threads;
  options.solver_options.num_linear_solver_threads = num_threads;
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.loss_function_scale = 1.0;
  options.loss_function_type =
      BundleAdjuster::Options::LossFunctionType::CAUCHY;
  return options;
}

BundleAdjuster::Options MapperOptions::GlobalBundleAdjustmentOptions() const {
  BundleAdjuster::Options options;
  options.solver_options.function_tolerance = 0.0;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = true;
  options.solver_options.num_threads = num_threads;
  options.solver_options.num_linear_solver_threads = num_threads;
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.loss_function_type =
      BundleAdjuster::Options::LossFunctionType::TRIVIAL;
  return options;
}

ParallelBundleAdjuster::Options
MapperOptions::ParallelGlobalBundleAdjustmentOptions() const {
  ParallelBundleAdjuster::Options options;
  options.max_num_iterations = ba_global_max_num_iterations;
  options.print_summary = true;
  options.gpu_index = ba_global_pba_gpu_index;
  options.num_threads = num_threads;
  return options;
}

DenseMapperOptions::DenseMapperOptions() { Reset(); }

void DenseMapperOptions::Reset() {
  max_image_size = 0;
  patch_match = mvs::PatchMatch::Options();
  fusion = mvs::StereoFusion::Options();
  poisson = mvs::PoissonReconstructionOptions();
}

bool DenseMapperOptions::Check() {
  const int kMaxWindowRadius = mvs::PatchMatch::kMaxWindowRadius;

  bool verified = true;

  CHECK_OPTION(DenseMapperOptions, max_image_size, >= 0);

  CHECK_OPTION(DenseMapperOptions, patch_match.gpu_index, >= -1);
  CHECK_OPTION(DenseMapperOptions, patch_match.window_radius, > 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.window_radius,
               <= kMaxWindowRadius);
  CHECK_OPTION(DenseMapperOptions, patch_match.sigma_spatial, > 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.sigma_color, > 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.num_samples, > 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.ncc_sigma, > 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.min_triangulation_angle, >= 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.min_triangulation_angle, < 180);
  CHECK_OPTION(DenseMapperOptions, patch_match.incident_angle_sigma, > 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.num_iterations, > 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.geom_consistency_regularizer,
               >= 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.geom_consistency_max_cost, >= 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.filter_min_ncc, >= -1);
  CHECK_OPTION(DenseMapperOptions, patch_match.filter_min_ncc, <= 1);
  CHECK_OPTION(DenseMapperOptions, patch_match.filter_min_triangulation_angle,
               >= 0);
  CHECK_OPTION(DenseMapperOptions, patch_match.filter_min_num_consistent,
               <= 180);
  CHECK_OPTION(DenseMapperOptions, patch_match.filter_geom_consistency_max_cost,
               >= 0);

  CHECK_OPTION(DenseMapperOptions, fusion.min_num_pixels, >= 0);
  CHECK_OPTION(DenseMapperOptions, fusion.max_num_pixels,
               >= fusion.min_num_pixels);
  CHECK_OPTION(DenseMapperOptions, fusion.max_traversal_depth, > 0);
  CHECK_OPTION(DenseMapperOptions, fusion.max_reproj_error, >= 0);
  CHECK_OPTION(DenseMapperOptions, fusion.max_depth_error, >= 0);
  CHECK_OPTION(DenseMapperOptions, fusion.max_normal_error, >= 0);

  CHECK_OPTION(DenseMapperOptions, poisson.point_weight, >= 0);
  CHECK_OPTION(DenseMapperOptions, poisson.depth, > 0);
  CHECK_OPTION(DenseMapperOptions, poisson.trim, >= 0);

  return verified;
}

RenderOptions::RenderOptions() { Reset(); }

void RenderOptions::Reset() {
  min_track_len = 3;
  max_error = 2;
  refresh_rate = 1;
  adapt_refresh_rate = true;
  image_connections = false;
}

bool RenderOptions::Check() {
  bool verified = true;

  CHECK_OPTION(RenderOptions, min_track_len, >= 2);
  CHECK_OPTION(RenderOptions, max_error, >= 0);
  CHECK_OPTION(RenderOptions, refresh_rate, > 0);

  return verified;
}

OptionManager::OptionManager() {
  project_path.reset(new std::string());
  log_path.reset(new std::string());
  database_path.reset(new std::string());
  image_path.reset(new std::string());

  extraction_options.reset(new ExtractionOptions());
  match_options.reset(new MatchOptions());
  exhaustive_match_options.reset(new ExhaustiveMatchOptions());
  sequential_match_options.reset(new SequentialMatchOptions());
  vocab_tree_match_options.reset(new VocabTreeMatchOptions());
  spatial_match_options.reset(new SpatialMatchOptions());
  ba_options.reset(new BundleAdjustmentOptions());
  sparse_mapper_options.reset(new MapperOptions());
  dense_mapper_options.reset(new DenseMapperOptions());
  render_options.reset(new RenderOptions());

  Reset();

  desc_->add_options()(
      "help,h",
      "Configuration can either be specified via command-line or by defining "
      "the options in a .ini project file provided as `--project_path`.")(
      "project_path", config::value<std::string>());

  AddDebugOptions();
}

void OptionManager::AddAllOptions() {
  AddDebugOptions();
  AddDatabaseOptions();
  AddImageOptions();
  AddLogOptions();
  AddExtractionOptions();
  AddMatchOptions();
  AddExhaustiveMatchOptions();
  AddSequentialMatchOptions();
  AddVocabTreeMatchOptions();
  AddSpatialMatchOptions();
  AddBundleAdjustmentOptions();
  AddMapperOptions();
  AddDenseMapperOptions();
  AddRenderOptions();
}

void OptionManager::AddDebugOptions() {
  if (added_debug_options_) {
    return;
  }
  added_debug_options_ = true;

  const int kDefaultDebugLog = false;
  desc_->add_options()(
      "General.debug_log_to_stderr",
      config::value<bool>(&FLAGS_logtostderr)->default_value(kDefaultDebugLog));
  RegisterOption("General.debug_log_to_stderr", &FLAGS_logtostderr);

  const int kDefaultDebugLevel = 2;
  desc_->add_options()(
      "General.debug_log_level",
      config::value<int>(&FLAGS_v)->default_value(kDefaultDebugLevel));
  RegisterOption("General.debug_log_level", &FLAGS_v);
}

void OptionManager::AddDatabaseOptions() {
  if (added_database_options_) {
    return;
  }
  added_database_options_ = true;

  desc_->add_options()(
      "General.database_path",
      config::value<std::string>(database_path.get())->required());
  RegisterOption("General.database_path", database_path.get());
}

void OptionManager::AddImageOptions() {
  if (added_image_options_) {
    return;
  }
  added_image_options_ = true;

  desc_->add_options()(
      "General.image_path",
      config::value<std::string>(image_path.get())->required());
  RegisterOption("General.image_path", image_path.get());
}

void OptionManager::AddLogOptions() {
  if (added_log_options_) {
    return;
  }
  added_log_options_ = true;

  desc_->add_options()(
      "General.log_path",
      config::value<std::string>(log_path.get())->default_value(""));
  RegisterOption("General.log_path", log_path.get());
}

void OptionManager::AddExtractionOptions() {
  if (added_extraction_options_) {
    return;
  }
  added_extraction_options_ = true;

  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     reader.camera_model);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     reader.single_camera);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     reader.camera_params);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     reader.default_focal_length_factor);

  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     sift.max_image_size);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     sift.max_num_features);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options, sift.first_octave);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     sift.octave_resolution);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     sift.peak_threshold);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     sift.edge_threshold);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     sift.max_num_orientations);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options, sift.upright);

  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options,
                     cpu.batch_size_factor);
  ADD_OPTION_DEFAULT(ExtractionOptions, extraction_options, cpu.num_threads);
}

void OptionManager::AddMatchOptions() {
  if (added_match_options_) {
    return;
  }
  added_match_options_ = true;

  ADD_OPTION_DEFAULT(MatchOptions, match_options, num_threads);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, gpu_index);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, max_ratio);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, max_distance);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, cross_check);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, max_error);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, max_num_matches);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, confidence);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, max_num_trials);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, min_inlier_ratio);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, min_num_inliers);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, multiple_models);
  ADD_OPTION_DEFAULT(MatchOptions, match_options, guided_matching);
}

void OptionManager::AddExhaustiveMatchOptions() {
  if (added_exhaustive_match_options_) {
    return;
  }
  added_exhaustive_match_options_ = true;

  ADD_OPTION_DEFAULT(ExhaustiveMatchOptions, exhaustive_match_options,
                     block_size);
  ADD_OPTION_DEFAULT(ExhaustiveMatchOptions, exhaustive_match_options,
                     preemptive);
  ADD_OPTION_DEFAULT(ExhaustiveMatchOptions, exhaustive_match_options,
                     preemptive_num_features);
  ADD_OPTION_DEFAULT(ExhaustiveMatchOptions, exhaustive_match_options,
                     preemptive_min_num_matches);
}

void OptionManager::AddSequentialMatchOptions() {
  if (added_sequential_match_options_) {
    return;
  }
  added_sequential_match_options_ = true;

  ADD_OPTION_DEFAULT(SequentialMatchOptions, sequential_match_options, overlap);
  ADD_OPTION_DEFAULT(SequentialMatchOptions, sequential_match_options,
                     loop_detection);
  ADD_OPTION_DEFAULT(SequentialMatchOptions, sequential_match_options,
                     loop_detection_period);
  ADD_OPTION_DEFAULT(SequentialMatchOptions, sequential_match_options,
                     loop_detection_num_images);
  ADD_OPTION_DEFAULT(SequentialMatchOptions, sequential_match_options,
                     vocab_tree_path);
}

void OptionManager::AddVocabTreeMatchOptions() {
  if (added_vocab_tree_match_options_) {
    return;
  }
  added_vocab_tree_match_options_ = true;

  ADD_OPTION_DEFAULT(VocabTreeMatchOptions, vocab_tree_match_options,
                     num_images);
  ADD_OPTION_DEFAULT(VocabTreeMatchOptions, vocab_tree_match_options,
                     vocab_tree_path);
  ADD_OPTION_DEFAULT(VocabTreeMatchOptions, vocab_tree_match_options,
                     match_list_path);
}

void OptionManager::AddSpatialMatchOptions() {
  if (added_spatial_match_options_) {
    return;
  }
  added_spatial_match_options_ = true;

  ADD_OPTION_DEFAULT(SpatialMatchOptions, spatial_match_options, is_gps);
  ADD_OPTION_DEFAULT(SpatialMatchOptions, spatial_match_options, ignore_z);
  ADD_OPTION_DEFAULT(SpatialMatchOptions, spatial_match_options,
                     max_num_neighbors);
  ADD_OPTION_DEFAULT(SpatialMatchOptions, spatial_match_options, max_distance);
}

void OptionManager::AddBundleAdjustmentOptions() {
  if (added_ba_options_) {
    return;
  }
  added_ba_options_ = true;

  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options, max_num_iterations);
  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options,
                     max_linear_solver_iterations);
  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options, function_tolerance);
  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options, gradient_tolerance);
  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options, parameter_tolerance);
  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options, refine_focal_length);
  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options,
                     refine_principal_point);
  ADD_OPTION_DEFAULT(BundleAdjustmentOptions, ba_options, refine_extra_params);
}

void OptionManager::AddMapperOptions() {
  if (added_sparse_mapper_options_) {
    return;
  }
  added_sparse_mapper_options_ = true;

  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, min_num_matches);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, multiple_models);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, max_num_models);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, max_model_overlap);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, min_model_size);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, init_image_id1);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, init_image_id2);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, init_num_trials);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, extract_colors);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, num_threads);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     min_focal_length_ratio);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     max_focal_length_ratio);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, max_extra_param);

  // IncrementalMapper.
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.init_min_num_inliers);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.init_max_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.init_max_forward_motion);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.init_min_tri_angle);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.abs_pose_max_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.abs_pose_min_num_inliers);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.abs_pose_min_inlier_ratio);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.filter_max_reproj_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.filter_min_tri_angle);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     incremental_mapper.max_reg_trials);

  // IncrementalTriangulator.
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.max_transitivity);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.create_max_angle_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.continue_max_angle_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.merge_max_reproj_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.complete_max_reproj_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.complete_max_transitivity);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.re_max_angle_error);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.re_min_ratio);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.re_max_trials);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.min_angle);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     triangulation.ignore_two_view_tracks);

  // General bundle adjustment.
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_refine_focal_length);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_refine_principal_point);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_refine_extra_params);

  // Local bundle adjustment.
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, ba_local_num_images);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_local_max_num_iterations);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_local_max_refinements);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_local_max_refinement_change);

  // Global bundle adjustment.
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options, ba_global_use_pba);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_images_ratio);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_images_freq);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_points_ratio);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_points_freq);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_max_num_iterations);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_pba_gpu_index);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_max_refinements);
  ADD_OPTION_DEFAULT(MapperOptions, sparse_mapper_options,
                     ba_global_max_refinement_change);
}

void OptionManager::AddDenseMapperOptions() {
  if (added_dense_mapper_options_) {
    return;
  }
  added_dense_mapper_options_ = true;

  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options, max_image_size);

  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.gpu_index);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.window_radius);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.sigma_spatial);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.sigma_color);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.num_samples);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.ncc_sigma);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.min_triangulation_angle);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.incident_angle_sigma);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.num_iterations);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.geom_consistency);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.geom_consistency_regularizer);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.geom_consistency_max_cost);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.filter);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.filter_min_ncc);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.filter_min_triangulation_angle);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.filter_min_num_consistent);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     patch_match.filter_geom_consistency_max_cost);

  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     fusion.min_num_pixels);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     fusion.max_num_pixels);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     fusion.max_traversal_depth);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     fusion.max_reproj_error);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     fusion.max_depth_error);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     fusion.max_normal_error);

  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options,
                     poisson.point_weight);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options, poisson.depth);
  ADD_OPTION_DEFAULT(DenseMapperOptions, dense_mapper_options, poisson.trim);
}

void OptionManager::AddRenderOptions() {
  if (added_render_options_) {
    return;
  }
  added_render_options_ = true;

  ADD_OPTION_DEFAULT(RenderOptions, render_options, min_track_len);
  ADD_OPTION_DEFAULT(RenderOptions, render_options, max_error);
  ADD_OPTION_DEFAULT(RenderOptions, render_options, refresh_rate);
  ADD_OPTION_DEFAULT(RenderOptions, render_options, adapt_refresh_rate);
  ADD_OPTION_DEFAULT(RenderOptions, render_options, image_connections);
}

void OptionManager::Reset() {
  project_path->clear();
  log_path->clear();
  database_path->clear();
  image_path->clear();

  extraction_options->Reset();
  match_options->Reset();
  exhaustive_match_options->Reset();
  sequential_match_options->Reset();
  vocab_tree_match_options->Reset();
  spatial_match_options->Reset();
  ba_options->Reset();
  sparse_mapper_options->Reset();
  render_options->Reset();

  desc_.reset(new boost::program_options::options_description());

  options_bool_.clear();
  options_int_.clear();
  options_double_.clear();
  options_string_.clear();

  added_debug_options_ = false;
  added_database_options_ = false;
  added_image_options_ = false;
  added_log_options_ = false;
  added_extraction_options_ = false;
  added_match_options_ = false;
  added_exhaustive_match_options_ = false;
  added_sequential_match_options_ = false;
  added_vocab_tree_match_options_ = false;
  added_spatial_match_options_ = false;
  added_ba_options_ = false;
  added_sparse_mapper_options_ = false;
  added_dense_mapper_options_ = false;
  added_render_options_ = false;
}

bool OptionManager::Parse(const int argc, char** argv) {
  config::variables_map vmap;

  try {
    config::store(config::parse_command_line(argc, argv, *desc_), vmap);

    if (vmap.count("help")) {
      std::cout << *desc_ << std::endl;
      return true;
    }

    if (vmap.count("project_path")) {
      *project_path = vmap["project_path"].as<std::string>();
      Read(*project_path);
    } else {
      vmap.notify();
    }
  } catch (std::exception& e) {
    std::cerr << "Error occurred while parsing options: " << e.what() << "."
              << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Unknown error occurred while parsing options." << std::endl;
    return false;
  }

  return true;
}

bool OptionManager::ParseHelp(const int argc, char** argv) {
  return argc == 2 &&
         (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}

bool OptionManager::Read(const std::string& path) {
  config::variables_map vmap;

  try {
    std::ifstream file(path.c_str());
    CHECK(file.is_open());
    config::store(config::parse_config_file(file, *desc_), vmap);
    vmap.notify();
  } catch (std::exception& e) {
    std::cerr << "Error occurred while parsing options: " << e.what() << "."
              << std::endl;
    return false;
  } catch (...) {
    std::cerr << "Unknown error occurred while parsing options." << std::endl;
    return false;
  }

  return true;
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

bool OptionManager::Check() {
  bool verified = true;

  verified =
      verified && boost::filesystem::is_directory(
                      boost::filesystem::path(*database_path).parent_path());
  verified = verified && boost::filesystem::is_directory(*image_path);

  verified = verified && extraction_options->Check();
  verified = verified && match_options->Check();
  verified = verified && exhaustive_match_options->Check();
  verified = verified && sequential_match_options->Check();
  verified = verified && vocab_tree_match_options->Check();
  verified = verified && spatial_match_options->Check();
  verified = verified && ba_options->Check();
  verified = verified && sparse_mapper_options->Check();
  verified = verified && dense_mapper_options->Check();
  verified = verified && render_options->Check();

  return verified;
}

}  // namespace colmap
