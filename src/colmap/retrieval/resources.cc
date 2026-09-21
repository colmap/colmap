// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/resources.h"

#include "colmap/util/logging.h"

namespace colmap {

const std::filesystem::path& GetVocabTreeUriForFeatureType(
    FeatureExtractorType feature_type) {
  switch (feature_type) {
    case FeatureExtractorType::SIFT:
      return kDefaultSiftVocabTreeUri;
    case FeatureExtractorType::ALIKED_N16ROT:
      return kDefaultAlikedN16RotVocabTreeUri;
    case FeatureExtractorType::ALIKED_N32:
      return kDefaultAlikedN32VocabTreeUri;
    case FeatureExtractorType::LOMA_B:
      return kDefaultLomaBVocabTreeUri;
    case FeatureExtractorType::LOMA_B128:
      return kDefaultLomaB128VocabTreeUri;
    default:
      LOG(FATAL_THROW)
          << "No default vocabulary tree available for feature type: "
          << FeatureExtractorTypeToString(feature_type);
  }
  const static std::filesystem::path kEmptyString;
  return kEmptyString;
}

}  // namespace colmap
