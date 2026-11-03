// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/feature/types.h"
#include "colmap/sensor/bitmap.h"

#include <memory>

namespace colmap {

struct SiftExtractionOptions;
struct AlikedExtractionOptions;
struct LomaExtractionOptions;

struct FeatureExtractionTypeOptions {
  explicit FeatureExtractionTypeOptions();

  std::shared_ptr<SiftExtractionOptions> sift;
  std::shared_ptr<AlikedExtractionOptions> aliked;
  std::shared_ptr<LomaExtractionOptions> loma;

  FeatureExtractionTypeOptions(const FeatureExtractionTypeOptions& other);
  FeatureExtractionTypeOptions& operator=(
      const FeatureExtractionTypeOptions& other);
  FeatureExtractionTypeOptions(FeatureExtractionTypeOptions&& other) = default;
  FeatureExtractionTypeOptions& operator=(
      FeatureExtractionTypeOptions&& other) = default;
};

struct FeatureExtractionOptions : public FeatureExtractionTypeOptions {
  explicit FeatureExtractionOptions(
      FeatureExtractorType type = FeatureExtractorType::UNDEFINED);

  // UNDEFINED means AUTO, not "unset": it defers the choice to ResolvedType(),
  // which prefers cusift on GPU. Any other value is an explicit request and is
  // always honored. The distinction is the whole point of defaulting to
  // UNDEFINED rather than to SIFT -- with SIFT as the default there is no way
  // to tell "the user asked for SIFT" from "the user said nothing", so an
  // explicit --FeatureExtraction.type SIFT could not be respected.
  FeatureExtractorType type = FeatureExtractorType::UNDEFINED;

  // Maximum image size, otherwise image will be down-scaled.
  // If max_image_size is non-positive, the appropriate size is selected
  // automatically based on the extractor type.
  int max_image_size = -1;

  // Number of threads for feature extraction.
  int num_threads = -1;

  // Whether to use the GPU for feature extraction.
#ifdef COLMAP_GPU_ENABLED
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif

  // Index of the GPU used for feature extraction. For multi-GPU extraction,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  // The extractor actually used, resolving UNDEFINED (=auto) to a concrete
  // type. Every decision inside COLMAP switches on this rather than on `type`,
  // so auto never leaks past this point.
  //
  // Auto resolves to CUSIFT when the GPU is in play, and to SIFT otherwise.
  // The fallbacks are not preferences but hard requirements: cusift is CUDA
  // only (no CPU path), and CusiftFeatureExtractor THROW_CHECKs on the three
  // covariant SIFT options below, which it does not implement. Silently
  // auto-selecting it there would turn a working CPU covariant configuration
  // into a crash, so auto declines it and leaves SIFT in place.
  FeatureExtractorType ResolvedType() const;

  // Whether the selected extractor requires RGB (or grayscale) images.
  bool RequiresRGB() const;

  // Whether the selected extractor requires OpenGL.
  bool RequiresOpenGL() const;

  // Returns the effective maximum image size. If max_image_size is set to -1,
  // the appropriate size is selected automatically based on the extractor type.
  int EffMaxImageSize() const;

  bool Check() const;
};

class FeatureExtractor {
 public:
  virtual ~FeatureExtractor() = default;

  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options);

  virtual bool Extract(const Bitmap& bitmap,
                       FeatureKeypoints* keypoints,
                       FeatureDescriptors* descriptors) = 0;
};

}  // namespace colmap
