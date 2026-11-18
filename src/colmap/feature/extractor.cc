// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/extractor.h"

#include "colmap/feature/aliked.h"
#include "colmap/feature/cusift.h"
#include "colmap/feature/loma.h"
#include "colmap/feature/sift.h"
#include "colmap/util/misc.h"

namespace colmap {
namespace {

void ThrowUnknownFeatureExtractorType(FeatureExtractorType type) {
  std::ostringstream error;
  error << "Unknown feature extractor type: " << type;
  throw std::runtime_error(error.str());
}

}  // namespace

FeatureExtractionTypeOptions::FeatureExtractionTypeOptions()
    : sift(std::make_shared<SiftExtractionOptions>()),
      aliked(std::make_shared<AlikedExtractionOptions>()),
      loma(std::make_shared<LomaExtractionOptions>()),
      cusift(std::make_shared<CusiftExtractionOptions>()) {}

FeatureExtractionTypeOptions::FeatureExtractionTypeOptions(
    const FeatureExtractionTypeOptions& other)
    : sift(CloneSharedPtr(other.sift)),
      aliked(CloneSharedPtr(other.aliked)),
      loma(CloneSharedPtr(other.loma)),
      cusift(CloneSharedPtr(other.cusift)) {}

FeatureExtractionTypeOptions& FeatureExtractionTypeOptions::operator=(
    const FeatureExtractionTypeOptions& other) {
  if (this == &other) {
    return *this;
  }
  sift = CloneSharedPtr(other.sift);
  aliked = CloneSharedPtr(other.aliked);
  loma = CloneSharedPtr(other.loma);
  cusift = CloneSharedPtr(other.cusift);
  return *this;
}

FeatureExtractionOptions::FeatureExtractionOptions(FeatureExtractorType type)
    : FeatureExtractionTypeOptions(), type(type) {}

bool FeatureExtractionOptions::RequiresRGB() const {
  switch (type) {
    case FeatureExtractorType::SIFT:
    case FeatureExtractorType::CUSIFT:
      return false;
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32:
    case FeatureExtractorType::LOMA_B:
    case FeatureExtractorType::LOMA_B128:
      return true;
    default:
      ThrowUnknownFeatureExtractorType(type);
  }
  return false;
}

bool FeatureExtractionOptions::RequiresOpenGL() const {
  switch (type) {
    case FeatureExtractorType::SIFT: {
      // Sync with logic in CreateSiftFeatureExtractor().
      if (sift->estimate_affine_shape || sift->domain_size_pooling ||
          sift->force_covariant_extractor) {
        return false;
      }
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
      return false;
#else
      return use_gpu;
#endif
    }
    case FeatureExtractorType::CUSIFT:  // CUDA only, no OpenGL path.
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32:
    case FeatureExtractorType::LOMA_B:
    case FeatureExtractorType::LOMA_B128:
      return false;
    default:
      ThrowUnknownFeatureExtractorType(type);
  }
  return false;
}

int FeatureExtractionOptions::EffMaxImageSize() const {
  if (max_image_size > 0) {
    return max_image_size;
  } else {
    switch (type) {
      case FeatureExtractorType::SIFT:
      case FeatureExtractorType::CUSIFT:
        return 3200;
      case FeatureExtractorType::ALIKED_N16ROT:
      case FeatureExtractorType::ALIKED_N32:
      case FeatureExtractorType::LOMA_B:
      case FeatureExtractorType::LOMA_B128:
        return 1600;
      default:
        ThrowUnknownFeatureExtractorType(type);
    }
  }
  return 0;
}

bool FeatureExtractionOptions::Check() const {
  CHECK_OPTION_GT(EffMaxImageSize(), 0);
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
#if !defined(COLMAP_GPU_ENABLED)
    LOG(ERROR) << "Cannot use GPU feature extraction without CUDA, HIP or "
                  "OpenGL support. Consider setting use_gpu to false.";
    return false;
#endif
  }
  switch (type) {
    case FeatureExtractorType::SIFT:
      return THROW_CHECK_NOTNULL(sift)->Check();
    case FeatureExtractorType::CUSIFT:
#if !defined(COLMAP_CUSIFT_ENABLED)
      LOG(ERROR) << "COLMAP was built without cusift (CUSIFT_ENABLED, which "
                    "needs CUDA >= 11.2).";
      return false;
#else
      // No CPU implementation to fall back to.
      if (!use_gpu) {
        LOG(ERROR) << "cusift feature extraction requires GPU. Consider "
                      "setting the feature extractor type to SIFT.";
        return false;
      }
      return THROW_CHECK_NOTNULL(sift)->Check() &&
             THROW_CHECK_NOTNULL(cusift)->Check() &&
             CheckCusiftSiftOptions(*sift);
#endif
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32:
      return THROW_CHECK_NOTNULL(aliked)->Check();
    case FeatureExtractorType::LOMA_B:
    case FeatureExtractorType::LOMA_B128:
      return THROW_CHECK_NOTNULL(loma)->Check();
    default:
      LOG(ERROR) << "Unknown feature extractor type: " << type;
      return false;
  }
}

std::unique_ptr<FeatureExtractor> FeatureExtractor::Create(
    const FeatureExtractionOptions& options) {
  switch (options.type) {
    case FeatureExtractorType::SIFT:
      return CreateSiftFeatureExtractor(options);
    case FeatureExtractorType::CUSIFT:
      return CreateCusiftFeatureExtractor(options);
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32:
      return CreateAlikedFeatureExtractor(options);
    case FeatureExtractorType::LOMA_B:
    case FeatureExtractorType::LOMA_B128:
      return CreateLomaFeatureExtractor(options);
    default:
      ThrowUnknownFeatureExtractorType(options.type);
  }
  return nullptr;
}

}  // namespace colmap
