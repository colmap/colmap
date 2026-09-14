// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/mvs_estimator_controller.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <set>
#include <utility>

namespace colmap {
namespace mvs {

PatchMatch::PatchMatch(const PatchMatchOptions& options, const Problem& problem)
    : options_(options), problem_(problem) {}

PatchMatch::~PatchMatch() = default;

void PatchMatch::Check() const {
  THROW_CHECK(options_.Check());
  THROW_CHECK(!options_.gpu_index.empty());
  const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  THROW_CHECK_EQ(gpu_indices.size(), 1);
  THROW_CHECK_GE(gpu_indices[0], -1);
  THROW_CHECK_NOTNULL(problem_.images);
  if (options_.geom_consistency) {
    THROW_CHECK_NOTNULL(problem_.depth_maps);
    THROW_CHECK_NOTNULL(problem_.normal_maps);
    THROW_CHECK_EQ(problem_.depth_maps->size(), problem_.images->size());
    THROW_CHECK_EQ(problem_.normal_maps->size(), problem_.images->size());
  }
  THROW_CHECK_GT(problem_.src_image_idxs.size(), 0);
  std::set<int> unique_image_idxs(problem_.src_image_idxs.begin(),
                                  problem_.src_image_idxs.end());
  unique_image_idxs.insert(problem_.ref_image_idx);
  THROW_CHECK_EQ(problem_.src_image_idxs.size() + 1, unique_image_idxs.size());
  for (const int image_idx : unique_image_idxs) {
    THROW_CHECK_GE(image_idx, 0) << image_idx;
    THROW_CHECK_LT(image_idx, problem_.images->size()) << image_idx;
    const Image& image = problem_.images->at(image_idx);
    THROW_CHECK_GT(image.GetBitmap().Width(), 0) << image_idx;
    THROW_CHECK_GT(image.GetBitmap().Height(), 0) << image_idx;
    THROW_CHECK(image.GetBitmap().IsGrey()) << image_idx;
    THROW_CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_idx;
    THROW_CHECK_EQ(image.GetHeight(), image.GetBitmap().Height()) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[1]), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[3]), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[6]), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[7]), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;
    if (options_.geom_consistency) {
      THROW_CHECK_LT(image_idx, problem_.depth_maps->size()) << image_idx;
      const DepthMap& depth_map = problem_.depth_maps->at(image_idx);
      THROW_CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
      THROW_CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
    }
  }
  if (options_.geom_consistency) {
    const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
    const NormalMap& ref_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
    THROW_CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
    THROW_CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
  }
}

void PatchMatch::Run() {
  LOG_HEADING2("PatchMatch::Run");
  Check();
  patch_match_cuda_ = std::make_unique<PatchMatchCuda>(options_, problem_);
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
  const auto& ref_image = problem_.images->at(problem_.ref_image_idx);
  return ConsistencyGraph(ref_image.GetWidth(),
                          ref_image.GetHeight(),
                          patch_match_cuda_->GetConsistentImageIdxs());
}

PatchMatchStereo::PatchMatchStereo(Options options)
    : options_(std::move(options)) {}

MVSEstimator::Capabilities PatchMatchStereo::GetCapabilities() const {
  Capabilities capabilities;
  capabilities.requires_rgb = false;
  capabilities.supports_geometric_pass = true;
  capabilities.min_num_source_images = 1;
  return capabilities;
}

MVSEstimator::Result PatchMatchStereo::Estimate(const Problem& problem,
                                                const Pass pass) {
  Options options = options_;
  options.depth_min = problem.depth_min;
  options.depth_max = problem.depth_max;
  options.geom_consistency = pass == Pass::GEOMETRIC;
  if (options_.geom_consistency && pass == Pass::PHOTOMETRIC) {
    options.filter = false;
  }
  options.filter_min_num_consistent =
      std::min(options.filter_min_num_consistent,
               static_cast<int>(problem.src_image_idxs.size()));
  PatchMatch patch_match(options, problem);
  patch_match.Run();
  Result result;
  result.depth_map = patch_match.GetDepthMap();
  result.normal_map = patch_match.GetNormalMap();
  if (options.write_consistency_graph) {
    result.consistency_graph = patch_match.GetConsistencyGraph();
  }
  return result;
}

PatchMatchController::PatchMatchController(
    const PatchMatchOptions& options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path) {
  MVSEstimator::Options estimator_options(MVSEstimator::Type::PATCH_MATCH);
  *estimator_options.patch_match = options;
  estimator_controller_ =
      std::make_unique<MVSEstimatorController>(estimator_options,
                                               workspace_path,
                                               workspace_format,
                                               pmvs_option_name,
                                               config_path);
}

PatchMatchController::~PatchMatchController() = default;

void PatchMatchController::Run() {
  estimator_controller_->SetCheckIfStoppedFunc(
      [this]() { return CheckIfStopped(); });
  estimator_controller_->Run();
}

}  // namespace mvs
}  // namespace colmap
