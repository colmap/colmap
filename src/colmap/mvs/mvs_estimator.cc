// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mvs_estimator.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/patch_match.h"
#endif

#include <sstream>
#include <stdexcept>
#include <utility>

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
  LOG(INFO) << "depth_min: " << depth_min;
  LOG(INFO) << "depth_max: " << depth_max;
}

MVSEstimator::Type MVSEstimator::Options::DefaultType() {
  return Type::PATCH_MATCH;
}

MVSEstimator::Options::Options(const Type type)
    : patch_match(std::make_shared<PatchMatchOptions>()), type(type) {}

MVSEstimator::Options::Options(const Options& other)
    : patch_match(CloneSharedPtr(other.patch_match)), type(other.type) {}

MVSEstimator::Options& MVSEstimator::Options::operator=(const Options& other) {
  if (this == &other) {
    return *this;
  }
  patch_match = CloneSharedPtr(other.patch_match);
  type = other.type;
  return *this;
}

bool MVSEstimator::Options::Check() const {
  switch (type) {
    case Type::PATCH_MATCH:
      return THROW_CHECK_NOTNULL(patch_match)->Check();
  }
  return false;
}

std::unique_ptr<MVSEstimator> MVSEstimator::Create(const Options& options,
                                                   const int device_index) {
  THROW_CHECK(options.Check());
  switch (options.type) {
    case Type::PATCH_MATCH: {
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
      PatchMatchOptions patch_match_options = *options.patch_match;
      patch_match_options.gpu_index = std::to_string(device_index);
      return std::make_unique<PatchMatch>(std::move(patch_match_options));
#else
      (void)device_index;
      throw std::runtime_error(
          "PatchMatch stereo requires a CUDA or HIP enabled build");
#endif
    }
  }
  throw std::runtime_error("Unknown MVS estimator type");
}

}  // namespace mvs
}  // namespace colmap
