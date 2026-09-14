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
    THROW_CHECK_LT(std::abs(image.GetK()[1]), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[3]), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[6]), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[7]), 1e-6f) << image_idx;
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
