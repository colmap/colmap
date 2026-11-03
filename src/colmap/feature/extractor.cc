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
      loma(std::make_shared<LomaExtractionOptions>()) {}

FeatureExtractionTypeOptions::FeatureExtractionTypeOptions(
    const FeatureExtractionTypeOptions& other)
    : sift(CloneSharedPtr(other.sift)),
      aliked(CloneSharedPtr(other.aliked)),
      loma(CloneSharedPtr(other.loma)) {}

FeatureExtractionTypeOptions& FeatureExtractionTypeOptions::operator=(
    const FeatureExtractionTypeOptions& other) {
  if (this == &other) {
    return *this;
  }
  sift = CloneSharedPtr(other.sift);
  aliked = CloneSharedPtr(other.aliked);
  loma = CloneSharedPtr(other.loma);
  return *this;
}

FeatureExtractionOptions::FeatureExtractionOptions(FeatureExtractorType type)
    : FeatureExtractionTypeOptions(), type(type) {}

FeatureExtractorType FeatureExtractionOptions::ResolvedType() const {
  // An explicit request always wins, including an explicit SIFT.
  if (type != FeatureExtractorType::UNDEFINED) {
    return type;
  }
  // cusift is CUDA only and has no CPU implementation to fall back on.
  if (!use_gpu) {
    return FeatureExtractorType::SIFT;
  }
  // CusiftFeatureExtractor THROW_CHECKs on all three of these (cusift.cc), so
  // auto-selecting it here would convert a valid covariant configuration into
  // a hard failure. Leave those on SIFT; an explicit CUSIFT still throws,
  // which is the correct answer to an impossible explicit request.
  if (sift && (sift->estimate_affine_shape || sift->domain_size_pooling ||
               sift->force_covariant_extractor)) {
    return FeatureExtractorType::SIFT;
  }
  return FeatureExtractorType::CUSIFT;
}

bool FeatureExtractionOptions::RequiresRGB() const {
  switch (ResolvedType()) {
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
  switch (ResolvedType()) {
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
    case FeatureExtractorType::CUSIFT:
      // CUDA only, no OpenGL path.
      return false;
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
    switch (ResolvedType()) {
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
  switch (ResolvedType()) {
    case FeatureExtractorType::SIFT:
      return THROW_CHECK_NOTNULL(sift)->Check();
    case FeatureExtractorType::CUSIFT:
      // cusift has no CPU implementation, so this is a hard requirement rather
      // than a fallback. Checked here so the failure is reported as a bad
      // option rather than surfacing later as an unknown-extractor error in
      // the CPU branch of the extraction controller.
      if (!use_gpu) {
        LOG(ERROR) << "cusift feature extraction requires GPU. Consider "
                      "setting the feature extractor type to SIFT.";
        return false;
      }
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
  // Resolve auto into a concrete type ONCE, here, and pass the resolved copy
  // downstream. The Create* implementations re-read options.type themselves
  // (e.g. aliked.cc dispatches N16ROT vs N32 on it), so handing them the
  // original UNDEFINED would surface as an unknown-type throw deep inside a
  // backend rather than as a decision made at the front door.
  FeatureExtractionOptions resolved_options = options;
  resolved_options.type = options.ResolvedType();
  switch (resolved_options.type) {
    case FeatureExtractorType::SIFT:
      return CreateSiftFeatureExtractor(resolved_options);
    case FeatureExtractorType::CUSIFT:
      return CreateCusiftFeatureExtractor(resolved_options);
    case FeatureExtractorType::ALIKED_N16ROT:
    case FeatureExtractorType::ALIKED_N32:
      return CreateAlikedFeatureExtractor(resolved_options);
    case FeatureExtractorType::LOMA_B:
    case FeatureExtractorType::LOMA_B128:
      return CreateLomaFeatureExtractor(resolved_options);
    default:
      ThrowUnknownFeatureExtractorType(resolved_options.type);
  }
  return nullptr;
}

}  // namespace colmap
