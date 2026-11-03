// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/matcher.h"

#include "colmap/feature/aliked.h"
#include "colmap/feature/flexglue.h"
#include "colmap/feature/cusift_matcher.h"
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

FeatureMatcherType FeatureMatchingOptions::ResolvedType() const {
  // An explicit request always wins, including an explicit SIFT_BRUTEFORCE.
  if (type != FeatureMatcherType::UNDEFINED) {
    return type;
  }
  // cusift is CUDA only and has no CPU implementation to fall back on.
  if (!use_gpu) {
    return FeatureMatcherType::SIFT_BRUTEFORCE;
  }
  // CusiftFeatureMatcher THROW_CHECKs on guided_matching and its MatchGuided()
  // is an unconditional throw (cusift_matcher.cc), so auto must not select it
  // for a guided run. An explicit CUSIFT_GEMM still throws there, as before.
  if (guided_matching) {
    return FeatureMatcherType::SIFT_BRUTEFORCE;
  }
  return FeatureMatcherType::CUSIFT_GEMM;
}

bool FeatureMatchingOptions::RequiresOpenGL() const {
  switch (ResolvedType()) {
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
  switch (ResolvedType()) {
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
      // cusift has no CPU implementation, so this is a hard requirement
      // rather than a fallback -- same rule FeatureExtractionOptions applies
      // to FeatureExtractorType::CUSIFT (extractor.cc).
      if (!use_gpu) {
        LOG(ERROR) << "cusift feature matching requires GPU. Consider "
                      "setting the feature matcher type to SIFT_BRUTEFORCE.";
        return false;
      }
      return THROW_CHECK_NOTNULL(cusift)->Check();
    default:
      LOG(ERROR) << "Unknown feature matcher type: " << type;
      return false;
  }
  return true;
}

std::unique_ptr<FeatureMatcher> FeatureMatcher::Create(
    const FeatureMatchingOptions& options) {
  // Resolve auto once, here -- see the matching comment in
  // FeatureExtractor::Create. CreateSiftFeatureMatcher in particular branches
  // on options.type again to split LIGHTGLUE from BRUTEFORCE and throws on
  // anything else, so it must not receive UNDEFINED.
  FeatureMatchingOptions resolved_options = options;
  resolved_options.type = options.ResolvedType();
  switch (resolved_options.type) {
    case FeatureMatcherType::SIFT_BRUTEFORCE:
    case FeatureMatcherType::SIFT_LIGHTGLUE:
      return CreateSiftFeatureMatcher(resolved_options);
    case FeatureMatcherType::ALIKED_BRUTEFORCE:
    case FeatureMatcherType::ALIKED_LIGHTGLUE:
      return CreateAlikedFeatureMatcher(resolved_options);
    case FeatureMatcherType::LOMA_BRUTEFORCE:
    case FeatureMatcherType::LOMA_B:
    case FeatureMatcherType::LOMA_B128:
    case FeatureMatcherType::LOMA_R:
    case FeatureMatcherType::LOMA_L:
    case FeatureMatcherType::LOMA_G:
      return CreateLomaFeatureMatcher(resolved_options);
    case FeatureMatcherType::FLEXGLUE:
      return CreateFlexGlueFeatureMatcher(resolved_options);
    case FeatureMatcherType::CUSIFT_GEMM:
      return CreateCusiftFeatureMatcher(resolved_options);
    default:
      ThrowUnknownFeatureMatcherType(resolved_options.type);
  }
  return nullptr;
}

}  // namespace colmap
