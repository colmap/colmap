// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mvs_estimator_controller.h"

#include "colmap/math/math.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
#include "colmap/util/cuda.h"
#endif

#include <algorithm>
#include <map>
#include <numeric>
#include <utility>

namespace colmap {
namespace mvs {
namespace {

const PatchMatchStereo::Options& PatchMatchOptions(
    const MVSEstimator::Options& options) {
  return *THROW_CHECK_NOTNULL(options.patch_match);
}

const MVSFormerPlusPlus::Options& MVSFormerOptions(
    const MVSEstimator::Options& options) {
  return *THROW_CHECK_NOTNULL(options.mvsformer_pp);
}

int MaxImageSize(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).max_image_size
             : MVSFormerOptions(options).max_image_size;
}

double CacheSize(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).cache_size
             : MVSFormerOptions(options).cache_size;
}

int NumThreads(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).num_threads
             : MVSFormerOptions(options).num_threads;
}

double MinTriangulationAngle(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).min_triangulation_angle
             : 1.0;
}

double DepthMin(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).depth_min
             : MVSFormerOptions(options).depth_min;
}

double DepthMax(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).depth_max
             : MVSFormerOptions(options).depth_max;
}

bool GeometricConsistency(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).geom_consistency
             : MVSFormerOptions(options).geom_consistency;
}

bool AllowMissingFiles(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).allow_missing_files
             : MVSFormerOptions(options).allow_missing_files;
}

bool WriteConsistencyGraph(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).write_consistency_graph
             : MVSFormerOptions(options).write_consistency_graph;
}

std::string GpuIndex(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH
             ? PatchMatchOptions(options).gpu_index
             : MVSFormerOptions(options).gpu_index;
}

bool UseGpu(const MVSEstimator::Options& options) {
  return options.type == MVSEstimator::Type::PATCH_MATCH ||
         MVSFormerOptions(options).use_gpu;
}

void AdjustSourceImages(const MVSEstimator::Capabilities& capabilities,
                        std::vector<int>* src_image_idxs) {
  THROW_CHECK_NOTNULL(src_image_idxs);
  if (capabilities.max_num_source_images.has_value() &&
      src_image_idxs->size() > *capabilities.max_num_source_images) {
    src_image_idxs->resize(*capabilities.max_num_source_images);
  }
  if (capabilities.supports_repeated_source_images &&
      !src_image_idxs->empty() &&
      src_image_idxs->size() < capabilities.min_num_source_images) {
    src_image_idxs->resize(capabilities.min_num_source_images,
                           src_image_idxs->front());
  }
}

void PrintOptions(const MVSEstimator::Options& options) {
  if (options.type == MVSEstimator::Type::PATCH_MATCH) {
    PatchMatchOptions(options).Print();
  } else {
    MVSFormerOptions(options).Print();
  }
}

}  // namespace

MVSEstimatorController::MVSEstimatorController(
    const MVSEstimator::Options& options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name),
      config_path_(config_path) {
  THROW_CHECK(options_.Check());
}

MVSEstimatorController::~MVSEstimatorController() = default;

std::string MVSEstimatorController::OutputType(
    const MVSEstimator::Pass pass) const {
  const std::string pass_name =
      pass == MVSEstimator::Pass::GEOMETRIC ? "geometric" : "photometric";
  if (options_.type == MVSEstimator::Type::PATCH_MATCH) {
    return pass_name;
  }
  return StringPrintf("mvsformer_pp_%d.%s",
                      MVSFormerOptions(options_).num_views,
                      pass_name.c_str());
}

void MVSEstimatorController::Run() {
  Timer run_timer;
  run_timer.Start();
  ReadWorkspace();
  ReadProblems();
  ReadDeviceIndices();

  thread_pool_ = std::make_unique<ThreadPool>(device_indices_.size());
  estimators_.reserve(device_indices_.size());
  for (const int device_index : device_indices_) {
    estimators_.push_back(MVSEstimator::Create(options_, device_index));
  }

  if (GeometricConsistency(options_)) {
    for (size_t problem_idx = 0; problem_idx < problems_.size();
         ++problem_idx) {
      thread_pool_->AddTask(&MVSEstimatorController::ProcessProblem,
                            this,
                            MVSEstimator::Pass::PHOTOMETRIC,
                            problem_idx);
    }
    thread_pool_->Wait();
  }

  const MVSEstimator::Pass final_pass = GeometricConsistency(options_)
                                            ? MVSEstimator::Pass::GEOMETRIC
                                            : MVSEstimator::Pass::PHOTOMETRIC;
  for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
    thread_pool_->AddTask(
        &MVSEstimatorController::ProcessProblem, this, final_pass, problem_idx);
  }
  thread_pool_->Wait();
  run_timer.PrintMinutes();
}

void MVSEstimatorController::ReadWorkspace() {
  LOG(INFO) << "Reading workspace...";
  Workspace::Options workspace_options;
  std::string workspace_format = workspace_format_;
  StringToLower(&workspace_format);
  if (workspace_format == "pmvs") {
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }
  workspace_options.max_image_size = MaxImageSize(options_);
  workspace_options.num_threads = NumThreads(options_);
  workspace_options.image_as_rgb =
      options_.type == MVSEstimator::Type::MVSFORMER_PP;
  workspace_options.cache_size = CacheSize(options_);
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type =
      GeometricConsistency(options_)
          ? OutputType(MVSEstimator::Pass::PHOTOMETRIC)
          : "";
  workspace_ = std::make_unique<CachedWorkspace>(workspace_options);
  if (workspace_format == "pmvs") {
    ImportPMVSWorkspace(*workspace_, pmvs_option_name_);
  }
  const auto confidence_maps_path =
      workspace_path_ / workspace_options.stereo_folder / "confidence_maps";
  CreateDirIfNotExists(confidence_maps_path);
  for (size_t image_idx = 0; image_idx < workspace_->GetModel().images.size();
       ++image_idx) {
    const auto parent_path =
        std::filesystem::path(workspace_->GetModel().GetImageName(image_idx))
            .parent_path();
    if (!parent_path.empty()) {
      CreateDirIfNotExists(confidence_maps_path / parent_path, true);
    }
  }
  depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();
}

void MVSEstimatorController::ReadProblems() {
  LOG(INFO) << "Reading configuration...";
  problems_.clear();
  const auto& model = workspace_->GetModel();
  const auto config_path = config_path_.empty()
                               ? workspace_path_ /
                                     workspace_->GetOptions().stereo_folder /
                                     "patch-match.cfg"
                               : config_path_;
  std::vector<std::string> config = ReadTextFileLines(config_path);

  std::vector<std::map<int, int>> shared_num_points;
  std::vector<std::map<int, float>> triangulation_angles;
  const float min_triangulation_angle =
      DegToRad(MinTriangulationAngle(options_));

  std::string ref_image_name;
  for (std::string config_line : config) {
    StringTrim(&config_line);
    if (config_line.empty() || config_line[0] == '#') {
      continue;
    }
    if (ref_image_name.empty()) {
      ref_image_name = config_line;
      continue;
    }

    MVSEstimator::Problem problem;
    problem.ref_image_idx = model.GetImageIdx(ref_image_name);
    const std::vector<std::string> src_image_names =
        CSVToVector<std::string>(config_line);
    if (src_image_names.size() == 1 && src_image_names[0] == "__all__") {
      for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
        if (static_cast<int>(image_idx) != problem.ref_image_idx) {
          problem.src_image_idxs.push_back(image_idx);
        }
      }
    } else if (src_image_names.size() == 2 &&
               src_image_names[0] == "__auto__") {
      if (shared_num_points.empty()) {
        shared_num_points = model.ComputeSharedPoints();
        triangulation_angles = model.ComputeTriangulationAngles(75);
      }
      std::vector<std::pair<int, int>> src_images;
      for (const auto& image : shared_num_points.at(problem.ref_image_idx)) {
        if (triangulation_angles.at(problem.ref_image_idx).at(image.first) >=
            min_triangulation_angle) {
          src_images.emplace_back(image.first, image.second);
        }
      }
      const size_t max_num_src_images = std::stoull(src_image_names.at(1));
      const size_t num_src_images =
          std::min(src_images.size(), max_num_src_images);
      std::partial_sort(src_images.begin(),
                        src_images.begin() + num_src_images,
                        src_images.end(),
                        [](const auto& lhs, const auto& rhs) {
                          return lhs.second > rhs.second;
                        });
      for (size_t i = 0; i < num_src_images; ++i) {
        problem.src_image_idxs.push_back(src_images[i].first);
      }
    } else {
      for (const std::string& src_image_name : src_image_names) {
        problem.src_image_idxs.push_back(model.GetImageIdx(src_image_name));
      }
    }
    if (problem.src_image_idxs.empty()) {
      LOG(WARNING) << "Ignoring reference image " << ref_image_name
                   << " because it has no source images";
    } else {
      problems_.push_back(std::move(problem));
    }
    ref_image_name.clear();
  }
  LOG(INFO) << StringPrintf("Configuration has %d problems...",
                            problems_.size());
}

void MVSEstimatorController::ReadDeviceIndices() {
  if (!UseGpu(options_)) {
    device_indices_ = {-1};
    return;
  }
  device_indices_ = CSVToVector<int>(GpuIndex(options_));
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  if (device_indices_.size() == 1 && device_indices_[0] == -1) {
    const int num_devices = GetNumCudaDevices();
    THROW_CHECK_GT(num_devices, 0);
    device_indices_.resize(num_devices);
    std::iota(device_indices_.begin(), device_indices_.end(), 0);
  }
#else
  device_indices_ = {-1};
#endif
  THROW_CHECK(!device_indices_.empty());
}

void MVSEstimatorController::ProcessProblem(const MVSEstimator::Pass pass,
                                            const size_t problem_idx) {
  if (CheckIfStopped()) {
    return;
  }
  const auto& model = workspace_->GetModel();
  MVSEstimator::Problem problem = problems_.at(problem_idx);
  MVSEstimator* estimator =
      estimators_.at(thread_pool_->GetThreadIndex()).get();
  const MVSEstimator::Capabilities capabilities = estimator->GetCapabilities();
  AdjustSourceImages(capabilities, &problem.src_image_idxs);
  if (problem.src_image_idxs.size() < capabilities.min_num_source_images) {
    const std::string& image_name = model.GetImageName(problem.ref_image_idx);
    if (AllowMissingFiles(options_)) {
      LOG(WARNING) << "Skipping " << image_name << " because the estimator "
                   << "requires at least " << capabilities.min_num_source_images
                   << " source images";
      return;
    }
    LOG(FATAL_THROW) << "Estimator requires at least "
                     << capabilities.min_num_source_images
                     << " source images for " << image_name;
  }

  const std::string output_type = OutputType(pass);
  const std::string image_name = model.GetImageName(problem.ref_image_idx);
  const std::string file_name =
      StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
  const auto stereo_path =
      workspace_path_ / workspace_->GetOptions().stereo_folder;
  const auto depth_map_path = stereo_path / "depth_maps" / file_name;
  const auto normal_map_path = stereo_path / "normal_maps" / file_name;
  const auto confidence_map_path = stereo_path / "confidence_maps" / file_name;
  const auto consistency_graph_path =
      stereo_path / "consistency_graphs" / file_name;
  if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
      (!capabilities.produces_confidence ||
       pass == MVSEstimator::Pass::GEOMETRIC ||
       ExistsFile(confidence_map_path)) &&
      (!WriteConsistencyGraph(options_) ||
       ExistsFile(consistency_graph_path))) {
    return;
  }

  LOG_HEADING1(StringPrintf("Processing view %d / %d for %s",
                            problem_idx + 1,
                            problems_.size(),
                            image_name.c_str()));
  problem.depth_min = DepthMin(options_);
  problem.depth_max = DepthMax(options_);
  if (problem.depth_min < 0 || problem.depth_max < 0) {
    problem.depth_min = depth_ranges_.at(problem.ref_image_idx).first;
    problem.depth_max = depth_ranges_.at(problem.ref_image_idx).second;
    THROW_CHECK(problem.depth_min > 0 && problem.depth_max > 0)
        << "Depth bounds must be set when no sparse model is available";
  }

  std::vector<Image> images = model.images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  std::vector<Mat<float>> confidence_maps;
  if (pass == MVSEstimator::Pass::GEOMETRIC) {
    depth_maps.resize(model.images.size());
    normal_maps.resize(model.images.size());
    confidence_maps.resize(model.images.size());
  }
  problem.images = &images;
  problem.depth_maps =
      pass == MVSEstimator::Pass::GEOMETRIC ? &depth_maps : nullptr;
  problem.normal_maps =
      pass == MVSEstimator::Pass::GEOMETRIC ? &normal_maps : nullptr;
  problem.confidence_maps =
      pass == MVSEstimator::Pass::GEOMETRIC ? &confidence_maps : nullptr;

  {
    std::unique_lock<std::mutex> lock(workspace_mutex_);
    std::vector<int> valid_src_image_idxs;
    bool valid_ref_image = false;
    std::vector<int> used_image_idxs = problem.src_image_idxs;
    used_image_idxs.push_back(problem.ref_image_idx);
    for (const int image_idx : used_image_idxs) {
      const bool missing = !workspace_->HasBitmap(image_idx) ||
                           (pass == MVSEstimator::Pass::GEOMETRIC &&
                            (!workspace_->HasDepthMap(image_idx) ||
                             !workspace_->HasNormalMap(image_idx))) ||
                           (pass == MVSEstimator::Pass::GEOMETRIC &&
                            capabilities.produces_confidence &&
                            !workspace_->HasConfidenceMap(image_idx));
      if (missing) {
        if (!AllowMissingFiles(options_)) {
          LOG(FATAL_THROW) << "Missing MVS input for "
                           << model.GetImageName(image_idx);
        }
        LOG(WARNING) << "Skipping missing MVS input "
                     << model.GetImageName(image_idx);
        continue;
      }
      images.at(image_idx).SetBitmap(workspace_->GetBitmap(image_idx));
      if (pass == MVSEstimator::Pass::GEOMETRIC) {
        depth_maps.at(image_idx) = workspace_->GetDepthMap(image_idx);
        normal_maps.at(image_idx) = workspace_->GetNormalMap(image_idx);
        if (capabilities.produces_confidence) {
          confidence_maps.at(image_idx) =
              workspace_->GetConfidenceMap(image_idx);
        }
      }
      if (image_idx != problem.ref_image_idx) {
        valid_src_image_idxs.push_back(image_idx);
      } else {
        valid_ref_image = true;
      }
    }
    problem.src_image_idxs = std::move(valid_src_image_idxs);
    AdjustSourceImages(capabilities, &problem.src_image_idxs);
    if (!valid_ref_image ||
        problem.src_image_idxs.size() < capabilities.min_num_source_images) {
      LOG(WARNING) << "Skipping " << image_name
                   << " because too many MVS inputs are missing";
      return;
    }
  }

  problem.Print();
  PrintOptions(options_);
  MVSEstimator::Result result = estimator->Estimate(problem, pass);
  result.depth_map.Write(depth_map_path);
  result.normal_map.Write(normal_map_path);
  if (result.confidence_map.has_value()) {
    result.confidence_map->Write(confidence_map_path);
  }
  if (WriteConsistencyGraph(options_) && result.consistency_graph.has_value()) {
    result.consistency_graph->Write(consistency_graph_path);
  }
}

}  // namespace mvs
}  // namespace colmap
