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

#include "mvs/patch_match.h"

#include <numeric>
#include <unordered_set>

#include "mvs/consistency_graph.h"
#include "mvs/patch_match_cuda.h"
#include "mvs/workspace.h"
#include "util/math.h"
#include "util/misc.h"

#define PrintOption(option) std::cout << #option ": " << option << std::endl

namespace colmap {
namespace mvs {
namespace {

void ImportPMVSOption(const Model& model, const std::string& path,
                      const std::string& option_name) {
  CreateDirIfNotExists(JoinPaths(path, "stereo"));
  CreateDirIfNotExists(JoinPaths(path, "stereo/depth_maps"));
  CreateDirIfNotExists(JoinPaths(path, "stereo/normal_maps"));
  CreateDirIfNotExists(JoinPaths(path, "stereo/consistency_graphs"));

  const auto option_lines = ReadTextFileLines(JoinPaths(path, option_name));
  for (const auto& line : option_lines) {
    if (StringStartsWith(line, "timages")) {
      const auto elems = StringSplit(line, " ");
      const int num_images = std::stoi(elems[1]);
      CHECK_EQ(num_images + 2, elems.size());
      std::vector<std::string> image_names;
      image_names.reserve(num_images);
      for (size_t i = 2; i < elems.size(); ++i) {
        const int image_id = std::stoi(elems[i]);
        const std::string image_name = model.GetImageName(image_id);
        image_names.push_back(image_name);
      }

      const auto patch_match_path = JoinPaths(path, "stereo/patch-match.cfg");
      const auto fusion_path = JoinPaths(path, "stereo/fusion.cfg");
      std::ofstream patch_match_file(patch_match_path, std::ios::trunc);
      std::ofstream fusion_file(fusion_path, std::ios::trunc);
      CHECK(patch_match_file.is_open()) << patch_match_path;
      CHECK(fusion_file.is_open()) << fusion_path;
      for (const auto ref_image_name : image_names) {
        patch_match_file << ref_image_name << std::endl;
        fusion_file << ref_image_name << std::endl;
        std::ostringstream line;
        for (const auto& image_name : image_names) {
          if (ref_image_name != image_name) {
            line << image_name << ",";
          }
        }
        const auto line_string = line.str();
        patch_match_file << line_string.substr(0, line_string.size() - 1)
                         << std::endl;
      }
    }
  }
}

}  // namespace

PatchMatch::PatchMatch(const Options& options, const Problem& problem)
    : options_(options), problem_(problem) {}

PatchMatch::~PatchMatch() {}

void PatchMatch::Options::Print() const {
  PrintHeading2("PatchMatch::Options");
  PrintOption(max_image_size);
  PrintOption(gpu_index);
  PrintOption(depth_min);
  PrintOption(depth_max);
  PrintOption(window_radius);
  PrintOption(sigma_spatial);
  PrintOption(sigma_color);
  PrintOption(num_samples);
  PrintOption(ncc_sigma);
  PrintOption(min_triangulation_angle);
  PrintOption(incident_angle_sigma);
  PrintOption(num_iterations);
  PrintOption(geom_consistency);
  PrintOption(geom_consistency_regularizer);
  PrintOption(geom_consistency_max_cost);
  PrintOption(filter);
  PrintOption(filter_min_ncc);
  PrintOption(filter_min_triangulation_angle);
  PrintOption(filter_min_num_consistent);
  PrintOption(filter_geom_consistency_max_cost);
  PrintOption(write_consistency_graph);
}

void PatchMatch::Problem::Print() const {
  PrintHeading2("PatchMatch::Problem");

  PrintOption(ref_image_id);

  std::cout << "src_image_ids: ";
  if (!src_image_ids.empty()) {
    for (size_t i = 0; i < src_image_ids.size() - 1; ++i) {
      std::cout << src_image_ids[i] << " ";
    }
    std::cout << src_image_ids.back() << std::endl;
  } else {
    std::cout << std::endl;
  }
}

void PatchMatch::Check() const {
  CHECK(options_.Check());

  CHECK(!options_.gpu_index.empty());
  const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1);
  CHECK_GE(gpu_indices[0], -1);

  CHECK_NOTNULL(problem_.images);
  if (options_.geom_consistency) {
    CHECK_NOTNULL(problem_.depth_maps);
    CHECK_NOTNULL(problem_.normal_maps);
    CHECK_EQ(problem_.depth_maps->size(), problem_.images->size());
    CHECK_EQ(problem_.normal_maps->size(), problem_.images->size());
  }

  CHECK_GT(problem_.src_image_ids.size(), 0);

  // Check that there are no duplicate images and that the reference image
  // is not defined as a source image.
  std::set<int> unique_image_ids(problem_.src_image_ids.begin(),
                                 problem_.src_image_ids.end());
  unique_image_ids.insert(problem_.ref_image_id);
  CHECK_EQ(problem_.src_image_ids.size() + 1, unique_image_ids.size());

  // Check that input data is well-formed.
  for (const int image_id : unique_image_ids) {
    CHECK_GE(image_id, 0) << image_id;
    CHECK_LT(image_id, problem_.images->size()) << image_id;

    const Image& image = problem_.images->at(image_id);
    CHECK_GT(image.GetBitmap().Width(), 0) << image_id;
    CHECK_GT(image.GetBitmap().Height(), 0) << image_id;
    CHECK(image.GetBitmap().IsGrey()) << image_id;
    CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_id;
    CHECK_EQ(image.GetHeight(), image.GetBitmap().Height()) << image_id;

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_id;
    CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_id;

    if (options_.geom_consistency) {
      CHECK_LT(image_id, problem_.depth_maps->size()) << image_id;
      const DepthMap& depth_map = problem_.depth_maps->at(image_id);
      CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_id;
      CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_id;
    }
  }

  if (options_.geom_consistency) {
    const Image& ref_image = problem_.images->at(problem_.ref_image_id);
    const NormalMap& ref_normal_map =
        problem_.normal_maps->at(problem_.ref_image_id);
    CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
    CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
  }
}

void PatchMatch::Run() {
  PrintHeading2("PatchMatch::Run");

  Check();

  patch_match_cuda_.reset(new PatchMatchCuda(options_, problem_));
  patch_match_cuda_->Run();
}

DepthMap PatchMatch::GetDepthMap() const {
  return patch_match_cuda_->GetDepthMap();
}

NormalMap PatchMatch::GetNormalMap() const {
  return patch_match_cuda_->GetNormalMap();
}

Mat<float> PatchMatch::GetSelProbMap() const {
  return patch_match_cuda_->GetSelProbMap();
}

ConsistencyGraph PatchMatch::GetConsistencyGraph() const {
  const auto& ref_image = problem_.images->at(problem_.ref_image_id);
  return ConsistencyGraph(ref_image.GetWidth(), ref_image.GetHeight(),
                          patch_match_cuda_->GetConsistentImageIds());
}

PatchMatchController::PatchMatchController(const PatchMatch::Options& options,
                                           const std::string& workspace_path,
                                           const std::string& workspace_format,
                                           const std::string& pmvs_option_name)
    : options_(options),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name) {
  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
}

void PatchMatchController::Run() {
  ReadWorkspace();
  ReadProblems();
  ReadGpuIndices();

  thread_pool_.reset(new ThreadPool(gpu_indices_.size()));

  // If geometric consistency is enabled, then photometric output must be
  // computed first for all images without filtering.
  if (options_.geom_consistency) {
    auto photometric_options = options_;
    photometric_options.geom_consistency = false;
    photometric_options.filter = false;

    for (size_t problem_idx = 0; problem_idx < problems_.size();
         ++problem_idx) {
      thread_pool_->AddTask(&PatchMatchController::ProcessProblem, this,
                            photometric_options, problem_idx);
    }

    thread_pool_->Wait();
  }

  for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
    thread_pool_->AddTask(&PatchMatchController::ProcessProblem, this, options_,
                          problem_idx);
  }

  thread_pool_->Wait();

  GetTimer().PrintMinutes();
}

void PatchMatchController::ReadWorkspace() {
  std::cout << "Reading workspace..." << std::endl;
  Workspace::Options workspace_options;
  workspace_options.max_image_size = options_.max_image_size;
  workspace_options.image_as_rgb = false;
  workspace_options.cache_size = options_.cache_size;
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type = options_.geom_consistency ? "photometric" : "";
  workspace_.reset(new Workspace(workspace_options));
  depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();
}

void PatchMatchController::ReadProblems() {
  problems_.clear();

  const auto model = workspace_->GetModel();

  if (workspace_format_ == "PMVS") {
    std::cout << "Importing PMVS options..." << std::endl;
    ImportPMVSOption(model, workspace_path_, pmvs_option_name_);
  }

  std::cout << "Reading configuration..." << std::endl;

  const auto config_path = JoinPaths(workspace_path_, "stereo/patch-match.cfg");
  std::ifstream file(config_path);
  CHECK(file.is_open()) << config_path;

  std::vector<std::map<int, int>> shared_num_points;
  std::vector<std::map<int, float>> triangulation_angles;

  const float min_triangulation_angle_rad =
      DegToRad(options_.min_triangulation_angle);

  std::string line;
  std::string ref_image_name;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = line;
      continue;
    }

    PatchMatch::Problem problem;

    problem.ref_image_id = model.GetImageId(ref_image_name);

    const std::vector<std::string> src_image_names =
        CSVToVector<std::string>(line);

    if (src_image_names.size() == 1 && src_image_names[0] == "__all__") {
      // Use all images as source images.
      problem.src_image_ids.clear();
      problem.src_image_ids.reserve(model.images.size() - 1);
      for (size_t image_id = 0; image_id < model.images.size(); ++image_id) {
        if (static_cast<int>(image_id) != problem.ref_image_id) {
          problem.src_image_ids.push_back(image_id);
        }
      }
    } else if (src_image_names.size() == 2 &&
               src_image_names[0] == "__auto__") {
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
          boost::lexical_cast<int>(src_image_names[1]);

      const auto& overlapping_images =
          shared_num_points.at(problem.ref_image_id);
      const auto& overlapping_triangulation_angles =
          triangulation_angles.at(problem.ref_image_id);

      if (max_num_src_images >= overlapping_images.size()) {
        problem.src_image_ids.reserve(overlapping_images.size());
        for (const auto& image : overlapping_images) {
          if (overlapping_triangulation_angles.at(image.first) >=
              min_triangulation_angle_rad) {
            problem.src_image_ids.push_back(image.first);
          }
        }
      } else {
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
                          [](const std::pair<int, int> image1,
                             const std::pair<int, int> image2) {
                            return image1.second > image2.second;
                          });

        problem.src_image_ids.reserve(eff_max_num_src_images);
        for (size_t i = 0; i < eff_max_num_src_images; ++i) {
          problem.src_image_ids.push_back(src_images[i].first);
        }
      }
    } else {
      problem.src_image_ids.reserve(src_image_names.size());
      for (const auto& src_image_name : src_image_names) {
        problem.src_image_ids.push_back(model.GetImageId(src_image_name));
      }
    }

    if (problem.src_image_ids.empty()) {
      std::cout
          << StringPrintf(
                 "WARNING: Ignoring reference image %s, because it has no "
                 "source images.",
                 ref_image_name.c_str())
          << std::endl;
    } else {
      problems_.push_back(problem);
    }

    ref_image_name.clear();
  }
}

void PatchMatchController::ReadGpuIndices() {
  gpu_indices_ = CSVToVector<int>(options_.gpu_index);
  if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    CHECK_GT(num_cuda_devices, 0);
    gpu_indices_.resize(num_cuda_devices);
    std::iota(gpu_indices_.begin(), gpu_indices_.end(), 0);
  }
}

void PatchMatchController::ProcessProblem(const PatchMatch::Options& options,
                                          const size_t problem_idx) {
  if (IsStopped()) {
    return;
  }

  const auto model = workspace_->GetModel();

  auto& problem = problems_.at(problem_idx);
  const int gpu_index = gpu_indices_.at(thread_pool_->GetThreadIndex());
  CHECK_GE(gpu_index, -1);

  const std::string output_type =
      options.geom_consistency ? "geometric" : "photometric";
  const std::string image_name = model.GetImageName(problem.ref_image_id);
  const std::string file_name =
      StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
  const std::string depth_map_path =
      JoinPaths(workspace_path_, "stereo/depth_maps", file_name);
  const std::string normal_map_path =
      JoinPaths(workspace_path_, "stereo/normal_maps", file_name);
  const std::string consistency_graph_path =
      JoinPaths(workspace_path_, "stereo/consistency_graphs", file_name);

  if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
      (!options.write_consistency_graph ||
       ExistsFile(consistency_graph_path))) {
    return;
  }

  PrintHeading1(StringPrintf("Processing view %d / %d", problem_idx + 1,
                             problems_.size()));

  std::vector<Image> images = model.images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  if (options.geom_consistency) {
    depth_maps.resize(model.images.size());
    normal_maps.resize(model.images.size());
  }

  problem.images = &images;
  problem.depth_maps = &depth_maps;
  problem.normal_maps = &normal_maps;

  {
    // Collect all used images in current problem.
    std::unordered_set<int> used_image_ids(problem.src_image_ids.begin(),
                                           problem.src_image_ids.end());
    used_image_ids.insert(problem.ref_image_id);

    // Only access workspace from one thread at a time and only spawn resample
    // threads from one master thread at a time.
    std::unique_lock<std::mutex> lock(workspace_mutex_);

    std::cout << "Reading inputs..." << std::endl;
    for (const auto image_id : used_image_ids) {
      images.at(image_id).SetBitmap(workspace_->GetBitmap(image_id));
      if (options.geom_consistency) {
        depth_maps.at(image_id) = workspace_->GetDepthMap(image_id);
        normal_maps.at(image_id) = workspace_->GetNormalMap(image_id);
      }
    }
  }

  problem.Print();

  auto patch_match_options = options;
  patch_match_options.depth_min = depth_ranges_.at(problem.ref_image_id).first;
  patch_match_options.depth_max = depth_ranges_.at(problem.ref_image_id).second;
  patch_match_options.gpu_index = std::to_string(gpu_index);
  patch_match_options.Print();

  PatchMatch patch_match(patch_match_options, problem);
  patch_match.Run();

  std::cout << std::endl
            << StringPrintf("Writing %s output for %s", output_type.c_str(),
                            image_name.c_str())
            << std::endl;

  patch_match.GetDepthMap().Write(depth_map_path);
  patch_match.GetNormalMap().Write(normal_map_path);
  if (options.write_consistency_graph) {
    patch_match.GetConsistencyGraph().Write(consistency_graph_path);
  }
}

}  // namespace mvs
}  // namespace colmap
