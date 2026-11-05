// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/matcher.h"

#include "colmap/feature/aliked.h"
#include "colmap/feature/cusift_matcher.h"
#include "colmap/feature/flexglue.h"
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
      loma(std::make_shared<LomaMatchingOptions>()),
      flexglue(std::make_shared<FlexGlueMatchingOptions>()),
      cusift(std::make_shared<CusiftMatchingOptions>()) {}

FeatureMatchingTypeOptions::FeatureMatchingTypeOptions(
    const FeatureMatchingTypeOptions& other)
    : sift(CloneSharedPtr(other.sift)),
      aliked(CloneSharedPtr(other.aliked)),
      loma(CloneSharedPtr(other.loma)),
      flexglue(CloneSharedPtr(other.flexglue)),
      cusift(CloneSharedPtr(other.cusift)) {}

FeatureMatchingTypeOptions& FeatureMatchingTypeOptions::operator=(
    const FeatureMatchingTypeOptions& other) {
  if (this == &other) {
    return *this;
  }
  sift = CloneSharedPtr(other.sift);
  aliked = CloneSharedPtr(other.aliked);
  loma = CloneSharedPtr(other.loma);
  flexglue = CloneSharedPtr(other.flexglue);
  cusift = CloneSharedPtr(other.cusift);
  return *this;
}

FeatureMatchingOptions::FeatureMatchingOptions(FeatureMatcherType type)
    : FeatureMatchingTypeOptions(), type(type) {}

bool FeatureMatchingOptions::UsesDescriptors() const {
  return type != FeatureMatcherType::FLEXGLUE;
}

bool FeatureMatchingOptions::RequiresOpenGL() const {
  switch (type) {
    case FeatureMatcherType::SIFT_BRUTEFORCE: {
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
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
    case FeatureMatcherType::FLEXGLUE:
    case FeatureMatcherType::CUSIFT_GEMM:
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
    LOG(ERROR) << "Cannot use GPU feature matching without CUDA, HIP, or "
                  "OpenGL support. Consider setting use_gpu to false.";
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
    case FeatureMatcherType::FLEXGLUE:
      // Not supported by FlexGlue. Fail before loading the models.
      CHECK_OPTION(!guided_matching);
      return THROW_CHECK_NOTNULL(flexglue)->Check();
    case FeatureMatcherType::CUSIFT_GEMM:
#if !defined(COLMAP_CUSIFT_ENABLED)
      LOG(ERROR) << "COLMAP was built without cusift (CUSIFT_ENABLED, which "
                    "needs CUDA >= 11.2).";
      return false;
#endif
      // No CPU implementation to fall back to.
      if (!use_gpu) {
        LOG(ERROR) << "cusift feature matching requires GPU. Consider "
                      "setting the feature matcher type to SIFT_BRUTEFORCE.";
        return false;
      }
      // Not supported by CUSIFT_GEMM.
      CHECK_OPTION(!guided_matching);
      return THROW_CHECK_NOTNULL(sift)->Check() &&
             THROW_CHECK_NOTNULL(cusift)->Check();
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
    case FeatureMatcherType::FLEXGLUE:
      return CreateFlexGlueFeatureMatcher(options);
    case FeatureMatcherType::CUSIFT_GEMM:
      return CreateCusiftFeatureMatcher(options);
    default:
      ThrowUnknownFeatureMatcherType(options.type);
  }
  return nullptr;
}

}  // namespace colmap
