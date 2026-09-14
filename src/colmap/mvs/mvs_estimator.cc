// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mvs_estimator.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <sstream>
#include <stdexcept>

namespace colmap {
namespace mvs {

void MVSEstimator::Problem::Print() const {
  LOG_HEADING2("MVSEstimator::Problem");
  LOG(INFO) << "ref_image_idx: " << ref_image_idx;
  THROW_CHECK(!src_image_idxs.empty());
  std::ostringstream stream;
  for (size_t i = 0; i < src_image_idxs.size(); ++i) {
    if (i > 0) {
      stream << ' ';
    }
    stream << src_image_idxs[i];
  }
  LOG(INFO) << "src_image_idxs: " << stream.str();
}

MVSEstimator::Type MVSEstimator::Options::DefaultType() {
  return Type::PATCH_MATCH;
}

MVSEstimator::Options::Options(const Type type)
    : patch_match(std::make_shared<PatchMatchStereo::Options>()),
      mvsformer_pp(std::make_shared<MVSFormerPlusPlus::Options>()),
      type(type) {}

MVSEstimator::Options::Options(const Options& other) : type(other.type) {
  if (other.patch_match) {
    patch_match =
        std::make_shared<PatchMatchStereo::Options>(*other.patch_match);
  }
  if (other.mvsformer_pp) {
    mvsformer_pp =
        std::make_shared<MVSFormerPlusPlus::Options>(*other.mvsformer_pp);
  }
}

MVSEstimator::Options& MVSEstimator::Options::operator=(const Options& other) {
  if (this == &other) {
    return *this;
  }
  type = other.type;
  patch_match =
      other.patch_match
          ? std::make_shared<PatchMatchStereo::Options>(*other.patch_match)
          : nullptr;
  mvsformer_pp =
      other.mvsformer_pp
          ? std::make_shared<MVSFormerPlusPlus::Options>(*other.mvsformer_pp)
          : nullptr;
  return *this;
}

bool MVSEstimator::Options::Check() const {
  switch (type) {
    case Type::PATCH_MATCH:
      return THROW_CHECK_NOTNULL(patch_match)->Check();
    case Type::MVSFORMER_PP:
      return THROW_CHECK_NOTNULL(mvsformer_pp)->Check();
  }
  return false;
}

std::unique_ptr<MVSEstimator> MVSEstimator::Create(const Options& options,
                                                   const int device_index) {
  THROW_CHECK(options.Check());
  switch (options.type) {
    case Type::PATCH_MATCH:
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
    {
      auto patch_match_options = *options.patch_match;
      patch_match_options.gpu_index = std::to_string(device_index);
      return std::make_unique<PatchMatchStereo>(std::move(patch_match_options));
    }
#else
      throw std::runtime_error(
          "PatchMatch stereo requires a CUDA or HIP enabled build");
#endif
    case Type::MVSFORMER_PP:
#ifdef COLMAP_ONNX_ENABLED
      return std::make_unique<MVSFormerPlusPlus>(*options.mvsformer_pp,
                                                 device_index);
#else
      throw std::runtime_error(
          "MVSFormer++ requires an ONNX Runtime enabled build");
#endif
  }
  throw std::runtime_error("Unknown MVS estimator type");
}

void MVSFormerPlusPlus::Options::Print() const {
  LOG_HEADING2("MVSFormerPlusPlus::Options");
  LOG(INFO) << "model_path: " << model_path;
  LOG(INFO) << "num_views: " << num_views;
  LOG(INFO) << "max_image_size: " << max_image_size;
  LOG(INFO) << "depth_min: " << depth_min;
  LOG(INFO) << "depth_max: " << depth_max;
  LOG(INFO) << "min_confidence: " << min_confidence;
  LOG(INFO) << "geom_consistency: " << geom_consistency;
  LOG(INFO) << "use_gpu: " << use_gpu;
  LOG(INFO) << "gpu_index: " << gpu_index;
  LOG(INFO) << "num_threads: " << num_threads;
}

bool MVSFormerPlusPlus::Options::Check() const {
  CHECK_OPTION(num_views == 5 || num_views == 10);
  CHECK_OPTION_GT(max_image_size, 0);
  if (depth_min != -1.0 || depth_max != -1.0) {
    CHECK_OPTION_GE(depth_min, 0.0);
    CHECK_OPTION_LE(depth_min, depth_max);
  }
  CHECK_OPTION_GE(min_confidence, 0.0);
  CHECK_OPTION_LE(min_confidence, 1.0);
  CHECK_OPTION_GT(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GT(filter_max_depth_error, 0.0);
  CHECK_OPTION_GE(filter_max_normal_error, 0.0);
  CHECK_OPTION_LE(filter_max_normal_error, 180.0);
  CHECK_OPTION_GT(cache_size, 0.0);
  CHECK_OPTION_GT(filter_min_num_consistent, 0);
  CHECK_OPTION_GE(num_threads, -1);
  return true;
}

class MVSFormerPlusPlus::Impl {};

MVSFormerPlusPlus::MVSFormerPlusPlus(Options, int)
    : impl_(std::make_unique<Impl>()) {
  throw std::runtime_error("MVSFormer++ support is not available");
}

MVSFormerPlusPlus::~MVSFormerPlusPlus() = default;

MVSEstimator::Capabilities MVSFormerPlusPlus::GetCapabilities() const {
  Capabilities capabilities;
  capabilities.requires_rgb = true;
  capabilities.produces_confidence = true;
  capabilities.supports_geometric_pass = true;
  capabilities.min_num_source_images = 4;
  capabilities.max_num_source_images = 4;
  return capabilities;
}

MVSEstimator::Result MVSFormerPlusPlus::Estimate(const Problem&, Pass) {
  throw std::runtime_error("MVSFormer++ support is not available");
}

}  // namespace mvs
}  // namespace colmap
