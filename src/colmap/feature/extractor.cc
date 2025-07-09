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

#include "colmap/feature/extractor.h"

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

FeatureExtractionOptions::FeatureExtractionOptions(FeatureExtractorType type)
    : type(type), sift(std::make_shared<SiftExtractionOptions>()) {}

int FeatureExtractionOptions::MaxImageSize() const {
  switch (type) {
    case FeatureExtractorType::SIFT:
      return sift->max_image_size;
    default:
      ThrowUnknownFeatureExtractorType(type);
  }
  return -1;
}

bool FeatureExtractionOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
#ifndef COLMAP_GPU_ENABLED
    LOG(ERROR) << "Cannot use GPU feature Extraction without CUDA or OpenGL "
                  "support. Set use_gpu or use_gpu to false.";
    return false;
#endif
  }
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
#ifndef COLMAP_GPU_ENABLED
    LOG(ERROR) << "Cannot use GPU feature extraction without CUDA or OpenGL "
                  "support. Set use_gpu or use_gpu to false.";
    return false;
#endif
  }
  if (type == FeatureExtractorType::SIFT) {
    return THROW_CHECK_NOTNULL(sift)->Check();
  } else {
    LOG(ERROR) << "Unknown feature extractor type: " << type;
    return false;
  }
  return true;
}

std::unique_ptr<FeatureExtractor> FeatureExtractor::Create(
    const FeatureExtractionOptions& options) {
  switch (options.type) {
    case FeatureExtractorType::SIFT:
      return CreateSiftFeatureExtractor(options);
    default:
      ThrowUnknownFeatureExtractorType(options.type);
  }
  return nullptr;
}

}  // namespace colmap
