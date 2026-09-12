// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/extractor.h"

#include "colmap/feature/aliked.h"
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
      loma(std::make_shared<LomaExtractionOptions>()) {}

FeatureExtractionTypeOptions::FeatureExtractionTypeOptions(
    const FeatureExtractionTypeOptions& other) {
  if (other.sift) {
    sift = std::make_shared<SiftExtractionOptions>(*other.sift);
  }
  if (other.aliked) {
    aliked = std::make_shared<AlikedExtractionOptions>(*other.aliked);
  }
  if (other.loma) {
    loma = std::make_shared<LomaExtractionOptions>(*other.loma);
  }
}

FeatureExtractionTypeOptions& FeatureExtractionTypeOptions::operator=(
    const FeatureExtractionTypeOptions& other) {
  if (this == &other) {
    return *this;
  }
  if (other.sift) {
    sift = std::make_shared<SiftExtractionOptions>(*other.sift);
  } else {
    sift.reset();
  }
  if (other.aliked) {
    aliked = std::make_shared<AlikedExtractionOptions>(*other.aliked);
  } else {
    aliked.reset();
  }
  if (other.loma) {
    loma = std::make_shared<LomaExtractionOptions>(*other.loma);
  } else {
    loma.reset();
  }
  return *this;
}

FeatureExtractionOptions::FeatureExtractionOptions(FeatureExtractorType type)
    : FeatureExtractionTypeOptions(), type(type) {}

bool FeatureExtractionOptions::RequiresRGB() const {
  switch (type) {
    case FeatureExtractorType::SIFT:
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
#ifdef COLMAP_CUDA_ENABLED
      return false;
#else
      return use_gpu;
#endif
    }
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
#if !defined(COLMAP_GPU_ENABLED) && !defined(COLMAP_CUDA_ENABLED)
    LOG(ERROR) << "Cannot use GPU feature extraction without CUDA or OpenGL "
                  "support. Consider setting use_gpu to false.";
    return false;
#endif
  }
  switch (type) {
    case FeatureExtractorType::SIFT:
      return THROW_CHECK_NOTNULL(sift)->Check();
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
