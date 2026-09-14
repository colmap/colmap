// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include "colmap/math/math.h"
#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>

#define PrintOption(option) LOG(INFO) << #option ": " << option << std::endl

namespace colmap {
namespace mvs {

namespace {

void CheckProblem(const PatchMatchOptions& options,
                  const MVSEstimator::Problem& problem) {
  THROW_CHECK(options.Check());

  THROW_CHECK(!options.gpu_index.empty());
  const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  THROW_CHECK_EQ(gpu_indices.size(), 1);
  THROW_CHECK_GE(gpu_indices[0], -1);

  THROW_CHECK_NOTNULL(problem.images);
  if (options.geom_consistency) {
    THROW_CHECK_NOTNULL(problem.depth_maps);
    THROW_CHECK_NOTNULL(problem.normal_maps);
    THROW_CHECK_EQ(problem.depth_maps->size(), problem.images->size());
    THROW_CHECK_EQ(problem.normal_maps->size(), problem.images->size());
  }

  THROW_CHECK_GT(problem.src_image_idxs.size(), 0);

  // Check that there are no duplicate images and that the reference image
  // is not defined as a source image.
  std::set<int> unique_image_idxs(problem.src_image_idxs.begin(),
                                  problem.src_image_idxs.end());
  unique_image_idxs.insert(problem.ref_image_idx);
  THROW_CHECK_EQ(problem.src_image_idxs.size() + 1, unique_image_idxs.size());

  // Check that input data is well-formed.
  for (const int image_idx : unique_image_idxs) {
    THROW_CHECK_GE(image_idx, 0) << image_idx;
    THROW_CHECK_LT(image_idx, problem.images->size()) << image_idx;

    const Image& image = problem.images->at(image_idx);
    THROW_CHECK_GT(image.GetBitmap().Width(), 0) << image_idx;
    THROW_CHECK_GT(image.GetBitmap().Height(), 0) << image_idx;
    THROW_CHECK(image.GetBitmap().IsGrey()) << image_idx;
    THROW_CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_idx;
    THROW_CHECK_EQ(image.GetHeight(), image.GetBitmap().Height()) << image_idx;

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    THROW_CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;

    if (options.geom_consistency) {
      THROW_CHECK_LT(image_idx, problem.depth_maps->size()) << image_idx;
      const DepthMap& depth_map = problem.depth_maps->at(image_idx);
      THROW_CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
      THROW_CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
    }
  }

  if (options.geom_consistency) {
    const Image& ref_image = problem.images->at(problem.ref_image_idx);
    const NormalMap& ref_normal_map =
        problem.normal_maps->at(problem.ref_image_idx);
    THROW_CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
    THROW_CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
  }
}

}  // namespace

PatchMatch::PatchMatch(Options options) : options_(std::move(options)) {}

MVSEstimator::Capabilities PatchMatch::GetCapabilities() const {
  Capabilities capabilities;
  capabilities.requires_rgb = false;
  capabilities.supports_geometric_pass = true;
  capabilities.min_num_source_images = 1;
  return capabilities;
}

MVSEstimator::Result PatchMatch::Estimate(const Problem& problem,
                                          const Pass pass) {
  THROW_CHECK_GT(problem.depth_min, 0);
  THROW_CHECK_LE(problem.depth_min, problem.depth_max);

  Options options = options_;
  options.depth_min = problem.depth_min;
  options.depth_max = problem.depth_max;
  if (options.sigma_spatial <= 0) {
    options.sigma_spatial = options.window_radius;
  }
  options.geom_consistency = pass == Pass::GEOMETRIC;
  // If geometric consistency is enabled, the photometric output initializes
  // the geometric pass and must not be filtered.
  if (pass == Pass::PHOTOMETRIC && options_.geom_consistency) {
    options.filter = false;
  }
  options.filter_min_num_consistent =
      std::min(options.filter_min_num_consistent,
               static_cast<int>(problem.src_image_idxs.size()));
  options.Print();

  CheckProblem(options, problem);

  PatchMatchCuda::Problem cuda_problem;
  cuda_problem.ref_image_idx = problem.ref_image_idx;
  cuda_problem.src_image_idxs = problem.src_image_idxs;
  cuda_problem.images = problem.images;
  cuda_problem.depth_maps = problem.depth_maps;
  cuda_problem.normal_maps = problem.normal_maps;

  PatchMatchCuda patch_match_cuda(options, cuda_problem);
  patch_match_cuda.Run();

  Result result;
  result.depth_map = patch_match_cuda.GetDepthMap();
  result.normal_map = patch_match_cuda.GetNormalMap();
  if (options.write_consistency_graph) {
    const Image& ref_image = problem.images->at(problem.ref_image_idx);
    result.consistency_graph =
        ConsistencyGraph(ref_image.GetWidth(),
                         ref_image.GetHeight(),
                         patch_match_cuda.GetConsistentImageIdxs());
  }
  return result;
}

PatchMatchController::PatchMatchController(
    const PatchMatchOptions& options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name),
      config_path_(config_path) {
  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
}

void PatchMatchController::Run() {
  Timer run_timer;
  run_timer.Start();
  ReadWorkspace();
  ReadProblems();
  ReadGpuIndices();

  thread_pool_ = std::make_unique<ThreadPool>(gpu_indices_.size());
  io_thread_pool_ = std::make_unique<ThreadPool>(
      GetEffectiveNumThreads(options_.num_threads));

  // If geometric consistency is enabled, then photometric output must be
  // computed first for all images.
  if (options_.geom_consistency) {
    for (size_t problem_idx = 0; problem_idx < problems_.size();
         ++problem_idx) {
      thread_pool_->AddTask(&PatchMatchController::ProcessProblem,
                            this,
                            MVSEstimator::Pass::PHOTOMETRIC,
                            problem_idx);
    }

    thread_pool_->Wait();
  }

  const MVSEstimator::Pass pass = options_.geom_consistency
                                      ? MVSEstimator::Pass::GEOMETRIC
                                      : MVSEstimator::Pass::PHOTOMETRIC;
  for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
    thread_pool_->AddTask(
        &PatchMatchController::ProcessProblem, this, pass, problem_idx);
  }

  thread_pool_->Wait();

  run_timer.PrintMinutes();
}

void PatchMatchController::ReadWorkspace() {
  LOG(INFO) << "Reading workspace...";

  Workspace::Options workspace_options;

  auto workspace_format_lower_case = workspace_format_;
  StringToLower(&workspace_format_lower_case);
  if (workspace_format_lower_case == "pmvs") {
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }

  workspace_options.max_image_size = options_.max_image_size;
  workspace_options.image_as_rgb = false;
  workspace_options.cache_size = options_.cache_size;
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type = options_.geom_consistency ? "photometric" : "";

  workspace_ = std::make_unique<CachedWorkspace>(workspace_options);

  if (workspace_format_lower_case == "pmvs") {
    LOG(INFO) << StringPrintf("Importing PMVS workspace (option %s)...",
                              pmvs_option_name_.c_str());
    ImportPMVSWorkspace(*workspace_, pmvs_option_name_);
  }

  depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();
}

void PatchMatchController::ReadProblems() {
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

  const float min_triangulation_angle_rad =
      DegToRad(options_.min_triangulation_angle);

  std::string ref_image_name;
  FlatHashSet<int> ref_image_idxs;

  struct ProblemConfig {
    std::string ref_image_name;
    std::vector<std::string> src_image_names;
  };
  std::vector<ProblemConfig> problem_configs;

  for (size_t i = 0; i < config.size(); ++i) {
    std::string& config_line = config[i];
    StringTrim(&config_line);

    if (config_line.empty() || config_line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = config_line;
      continue;
    }

    ref_image_idxs.insert(model.GetImageIdx(ref_image_name));

    ProblemConfig problem_config;
    problem_config.ref_image_name = ref_image_name;
    problem_config.src_image_names = CSVToVector<std::string>(config_line);
    problem_configs.push_back(problem_config);

    ref_image_name.clear();
  }

  for (const auto& problem_config : problem_configs) {
    PatchMatch::Problem problem;

    problem.ref_image_idx = model.GetImageIdx(problem_config.ref_image_name);

    if (problem_config.src_image_names.size() == 1 &&
        problem_config.src_image_names[0] == "__all__") {
      // Use all images as source images.
      problem.src_image_idxs.clear();
      problem.src_image_idxs.reserve(model.images.size() - 1);
      for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
        if (static_cast<int>(image_idx) != problem.ref_image_idx) {
          problem.src_image_idxs.push_back(image_idx);
        }
      }
    } else if (problem_config.src_image_names.size() == 2 &&
               problem_config.src_image_names[0] == "__auto__") {
      // Use maximum number of overlapping images as source images. Overlapping
      // will be sorted based on the number of shared points to the reference
      // image and the top ranked images are selected. Note that images are only
      // selected if some points have a sufficient triangulation angle.

      if (shared_num_points.empty()) {
        shared_num_points = model.ComputeSharedPoints();
      }
      if (triangulation_angles.empty()) {
        const float kTriangulationAnglePercentile = 75;
        triangulation_angles =
            model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
      }

      const size_t max_num_src_images =
          std::stoll(problem_config.src_image_names[1]);

      const auto& overlapping_images =
          shared_num_points.at(problem.ref_image_idx);
      const auto& overlapping_triangulation_angles =
          triangulation_angles.at(problem.ref_image_idx);

      std::vector<std::pair<int, int>> src_images;
      src_images.reserve(overlapping_images.size());
      for (const auto& image : overlapping_images) {
        if (overlapping_triangulation_angles.at(image.first) >=
            min_triangulation_angle_rad) {
          src_images.emplace_back(image.first, image.second);
        }
      }

      const size_t eff_max_num_src_images =
          std::min(src_images.size(), max_num_src_images);

      std::partial_sort(src_images.begin(),
                        src_images.begin() + eff_max_num_src_images,
                        src_images.end(),
                        [](const std::pair<int, int>& image1,
                           const std::pair<int, int>& image2) {
                          return image1.second > image2.second;
                        });

      problem.src_image_idxs.reserve(eff_max_num_src_images);
      for (size_t i = 0; i < eff_max_num_src_images; ++i) {
        problem.src_image_idxs.push_back(src_images[i].first);
      }
    } else {
      problem.src_image_idxs.reserve(problem_config.src_image_names.size());
      for (const auto& src_image_name : problem_config.src_image_names) {
        problem.src_image_idxs.push_back(model.GetImageIdx(src_image_name));
      }
    }

    if (problem.src_image_idxs.empty()) {
      LOG(WARNING) << StringPrintf(
          "Ignoring reference image %s, because it has no "
          "source images.",
          problem_config.ref_image_name.c_str());
    } else {
      problems_.push_back(problem);
    }
  }

  LOG(INFO) << StringPrintf("Configuration has %d problems...",
                            problems_.size());
}

void PatchMatchController::ReadGpuIndices() {
  gpu_indices_ = CSVToVector<int>(options_.gpu_index);
  if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    THROW_CHECK_GT(num_cuda_devices, 0);
    gpu_indices_.resize(num_cuda_devices);
    std::iota(gpu_indices_.begin(), gpu_indices_.end(), 0);
  }
}

void PatchMatchController::ProcessProblem(const MVSEstimator::Pass pass,
                                          const size_t problem_idx) {
  if (CheckIfStopped()) {
    return;
  }

  const auto& model = workspace_->GetModel();

  auto& problem = problems_.at(problem_idx);
  const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
  THROW_CHECK_GE(gpu_index, -1);

  const bool geom_consistency = pass == MVSEstimator::Pass::GEOMETRIC;

  const std::string& stereo_folder = workspace_->GetOptions().stereo_folder;
  const std::string output_type =
      geom_consistency ? "geometric" : "photometric";
  const std::string image_name = model.GetImageName(problem.ref_image_idx);
  const std::string file_name =
      StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
  const auto depth_map_path =
      workspace_path_ / stereo_folder / "depth_maps" / file_name;
  const auto normal_map_path =
      workspace_path_ / stereo_folder / "normal_maps" / file_name;
  const auto consistency_graph_path =
      workspace_path_ / stereo_folder / "consistency_graphs" / file_name;

  if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
      (!options_.write_consistency_graph ||
       ExistsFile(consistency_graph_path))) {
    return;
  }

  LOG_HEADING1(StringPrintf("Processing view %d / %d for %s",
                            problem_idx + 1,
                            problems_.size(),
                            image_name.c_str()));

  if (options_.depth_min < 0 || options_.depth_max < 0) {
    problem.depth_min = depth_ranges_.at(problem.ref_image_idx).first;
    problem.depth_max = depth_ranges_.at(problem.ref_image_idx).second;
    THROW_CHECK(problem.depth_min > 0 && problem.depth_max > 0)
        << " - You must manually set the minimum and maximum depth, since no "
           "sparse model is provided in the workspace.";
  } else {
    problem.depth_min = static_cast<float>(options_.depth_min);
    problem.depth_max = static_cast<float>(options_.depth_max);
  }

  std::vector<Image> images = model.images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  if (geom_consistency) {
    depth_maps.resize(model.images.size());
    normal_maps.resize(model.images.size());
  }

  problem.images = &images;
  problem.depth_maps = &depth_maps;
  problem.normal_maps = &normal_maps;

  {
    // Collect all used images in current problem.
    FlatHashSet<int> used_image_idxs(problem.src_image_idxs.begin(),
                                     problem.src_image_idxs.end());
    used_image_idxs.insert(problem.ref_image_idx);

    // Only access workspace from one thread at a time and only spawn resample
    // threads from one master thread at a time.
    std::unique_lock<std::mutex> lock(workspace_mutex_);

    LOG(INFO) << "Reading inputs...";

    // Filter by file existence first (fast, no heavy I/O).
    std::vector<int> src_image_idxs;
    std::vector<int> valid_image_idxs;
    for (const auto image_idx : used_image_idxs) {
      const auto image_path = workspace_->GetBitmapPath(image_idx);
      const auto depth_path = workspace_->GetDepthMapPath(image_idx);
      const auto normal_path = workspace_->GetNormalMapPath(image_idx);

      if (!ExistsFile(image_path) ||
          (geom_consistency && !ExistsFile(depth_path)) ||
          (geom_consistency && !ExistsFile(normal_path))) {
        if (options_.allow_missing_files) {
          LOG(WARNING) << StringPrintf(
              "Skipping source image %d: %s for missing "
              "image or depth/normal map",
              image_idx,
              model.GetImageName(image_idx).c_str());
          continue;
        } else {
          LOG(ERROR) << StringPrintf(
              "Missing image or map dependency for image %d: %s",
              image_idx,
              model.GetImageName(image_idx).c_str());
        }
      }

      valid_image_idxs.push_back(image_idx);
      if (image_idx != problem.ref_image_idx) {
        src_image_idxs.push_back(image_idx);
      }
    }

    // Read images in parallel using the shared I/O thread pool.
    std::vector<std::shared_future<void>> futures;
    futures.reserve(valid_image_idxs.size());
    for (const auto image_idx : valid_image_idxs) {
      futures.push_back(io_thread_pool_->AddTask([&, image_idx]() {
        images.at(image_idx).SetBitmap(workspace_->GetBitmap(image_idx));
        if (geom_consistency) {
          depth_maps.at(image_idx) = workspace_->GetDepthMap(image_idx);
          normal_maps.at(image_idx) = workspace_->GetNormalMap(image_idx);
        }
      }));
    }
    for (auto& future : futures) {
      future.get();
    }

    problem.src_image_idxs = src_image_idxs;
  }

  problem.Print();

  PatchMatchOptions patch_match_options = options_;
  patch_match_options.gpu_index = std::to_string(gpu_index);
  PatchMatch patch_match(std::move(patch_match_options));
  const MVSEstimator::Result result = patch_match.Estimate(problem, pass);

  LOG(INFO) << std::endl
            << StringPrintf("Writing %s output for %s",
                            output_type.c_str(),
                            image_name.c_str());

  result.depth_map.Write(depth_map_path);
  result.normal_map.Write(normal_map_path);
  if (result.consistency_graph.has_value()) {
    result.consistency_graph->Write(consistency_graph_path);
  }
}

}  // namespace mvs
}  // namespace colmap
