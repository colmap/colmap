// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/feature/types.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/enum_utils.h"

#include <memory>

namespace colmap {

MAKE_ENUM_CLASS_OVERLOAD_STREAM(FeatureExtractorType, 0, SIFT, ALIKED);

struct SiftExtractionOptions;
struct ALIKEDExtractionOptions;

struct FeatureExtractionOptions {
  explicit FeatureExtractionOptions(
      FeatureExtractorType type = FeatureExtractorType::SIFT);

  FeatureExtractorType type = FeatureExtractorType::SIFT;

  // Number of threads for feature extraction.
  int num_threads = -1;

  // Whether to use the GPU for feature extraction.
  bool use_gpu = true;

  // Index of the GPU used for feature extraction. For multi-GPU extraction,
  // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
  std::string gpu_index = "-1";

  std::shared_ptr<SiftExtractionOptions> sift;
  std::shared_ptr<ALIKEDExtractionOptions> aliked;

  int MaxImageSize() const;

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
