// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/matcher.h"

#include "colmap/feature/aliked.h"
#include "colmap/feature/loma.h"
#include "colmap/feature/onnx_matchers.h"
#include "colmap/feature/sift.h"
#include "colmap/util/misc.h"

namespace colmap {
namespace {

void ThrowUnknownFeatureMatcherType(FeatureMatcherType type) {
  std::ostringstream error;
  error << "Unknown feature matcher type: " << type;
  throw std::runtime_error(error.str());
}

}  // namespace

FeatureMatchingTypeOptions::FeatureMatchingTypeOptions()
    : sift(std::make_shared<SiftMatchingOptions>()),
      aliked(std::make_shared<AlikedMatchingOptions>()),
      loma(std::make_shared<LomaMatchingOptions>()) {}

FeatureMatchingTypeOptions::FeatureMatchingTypeOptions(
    const FeatureMatchingTypeOptions& other)
    : sift(CloneSharedPtr(other.sift)),
      aliked(CloneSharedPtr(other.aliked)),
      loma(CloneSharedPtr(other.loma)) {}

FeatureMatchingTypeOptions& FeatureMatchingTypeOptions::operator=(
    const FeatureMatchingTypeOptions& other) {
  if (this == &other) {
    return *this;
  }
  sift = CloneSharedPtr(other.sift);
  aliked = CloneSharedPtr(other.aliked);
  loma = CloneSharedPtr(other.loma);
  return *this;
}

FeatureMatchingOptions::FeatureMatchingOptions(FeatureMatcherType type)
    : FeatureMatchingTypeOptions(), type(type) {}

bool FeatureMatchingOptions::RequiresOpenGL() const {
  switch (type) {
    case FeatureMatcherType::SIFT_BRUTEFORCE: {
#ifdef COLMAP_CUDA_ENABLED
      return false;
#else
      return use_gpu;
#endif
    }
    case FeatureMatcherType::SIFT_LIGHTGLUE:
    case FeatureMatcherType::ALIKED_BRUTEFORCE:
    case FeatureMatcherType::ALIKED_LIGHTGLUE:
    case FeatureMatcherType::LOMA_BRUTEFORCE:
    case FeatureMatcherType::LOMA_B:
    case FeatureMatcherType::LOMA_B128:
    case FeatureMatcherType::LOMA_R:
    case FeatureMatcherType::LOMA_L:
    case FeatureMatcherType::LOMA_G:
      return false;
    default:
      ThrowUnknownFeatureMatcherType(type);
  }
  return false;
}

bool FeatureMatchingOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
#ifndef COLMAP_GPU_ENABLED
    LOG(ERROR) << "Cannot use GPU feature matching without CUDA or OpenGL "
                  "support. Set use_gpu or use_gpu to false.";
    return false;
#endif
  }
  CHECK_OPTION_GE(max_num_matches, 0);
  switch (type) {
    case FeatureMatcherType::SIFT_BRUTEFORCE:
    case FeatureMatcherType::SIFT_LIGHTGLUE:
      return THROW_CHECK_NOTNULL(sift)->Check();
    case FeatureMatcherType::ALIKED_BRUTEFORCE:
    case FeatureMatcherType::ALIKED_LIGHTGLUE:
      return THROW_CHECK_NOTNULL(aliked)->Check();
    case FeatureMatcherType::LOMA_BRUTEFORCE:
    case FeatureMatcherType::LOMA_B:
    case FeatureMatcherType::LOMA_B128:
    case FeatureMatcherType::LOMA_R:
    case FeatureMatcherType::LOMA_L:
    case FeatureMatcherType::LOMA_G:
      return THROW_CHECK_NOTNULL(loma)->Check();
    default:
      LOG(ERROR) << "Unknown feature matcher type: " << type;
      return false;
  }
  return true;
}

std::unique_ptr<FeatureMatcher> FeatureMatcher::Create(
    const FeatureMatchingOptions& options) {
  switch (options.type) {
    case FeatureMatcherType::SIFT_BRUTEFORCE:
    case FeatureMatcherType::SIFT_LIGHTGLUE:
      return CreateSiftFeatureMatcher(options);
    case FeatureMatcherType::ALIKED_BRUTEFORCE:
    case FeatureMatcherType::ALIKED_LIGHTGLUE:
      return CreateAlikedFeatureMatcher(options);
    case FeatureMatcherType::LOMA_BRUTEFORCE:
    case FeatureMatcherType::LOMA_B:
    case FeatureMatcherType::LOMA_B128:
    case FeatureMatcherType::LOMA_R:
    case FeatureMatcherType::LOMA_L:
    case FeatureMatcherType::LOMA_G:
      return CreateLomaFeatureMatcher(options);
    default:
      ThrowUnknownFeatureMatcherType(options.type);
  }
  return nullptr;
}

}  // namespace colmap
