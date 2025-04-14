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

#include "colmap/feature/aliked.h"

#include "colmap/feature/lightglue.h"
#include "colmap/feature/torch_utils.h"
#include "colmap/util/file.h"

#include <memory>

#ifdef COLMAP_TORCH_ENABLED
#include "thirdparty/ALIKED/aliked.hpp"

#include <torch/torch.h>
#endif

namespace colmap {
namespace {

#ifdef COLMAP_TORCH_ENABLED

class ALIKEDFeatureExtractor : public FeatureExtractor {
 public:
  explicit ALIKEDFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options),
        aliked_(options.aliked->model_name,
                MaybeDownloadAndCacheFile(options.aliked->model_path),
                GetDeviceName(options.use_gpu, options.gpu_index),
                /*top_k=*/options.aliked->top_k,
                /*scores_th=*/options.aliked->score_threshold,
                /*n_limit=*/options.aliked->max_num_features) {}

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);

    const int orig_width = bitmap.Width();
    const int orig_height = bitmap.Height();
    const int image_size = std::max(orig_width, orig_height);
    const bool needs_rescale = image_size > options_.max_image_size;

    // Only copy bitmap if we need to rescale or convert to RGB.
    const Bitmap* bitmap_ptr = &bitmap;
    std::unique_ptr<Bitmap> maybe_bitmap_copy;
    if (!bitmap.IsRGB() || needs_rescale) {
      maybe_bitmap_copy = std::make_unique<Bitmap>(bitmap.CloneAsRGB());
      bitmap_ptr = maybe_bitmap_copy.get();
    }

    if (needs_rescale) {
      const double scale = static_cast<double>(options_.max_image_size) /
                           static_cast<double>(image_size);
      maybe_bitmap_copy->Rescale(scale * orig_width, scale * orig_height);
    }

    const int width = bitmap_ptr->Width();
    const int height = bitmap_ptr->Height();
    auto row_major_array = bitmap_ptr->ConvertToRowMajorArray();
    // Clone to ensure ownership
    auto torch_image =
        torch::from_blob(
            row_major_array.data(), {height, width, 3}, torch::kUInt8)
            .clone()
            .permute({2, 0, 1})   // [C, H, W]
            .to(torch::kFloat32)  // Convert to float
            .mul_(1.0f / 255.0f)  // Normalize in-place
            .unsqueeze(0);        // Add batch dimension [1, C, H, W]

    const torch::Dict<std::string, torch::Tensor> outputs =
        aliked_.forward(torch_image);

    const auto torch_keypoints = outputs.at("keypoints");      // shape [N, 2]
    const auto torch_descriptors = outputs.at("descriptors");  // shape [N, D]
    const int num_keypoints = torch_keypoints.size(0);
    const int descriptor_dim = torch_descriptors.size(1);
    THROW_CHECK_EQ(num_keypoints, torch_descriptors.size(0));

    keypoints->resize(num_keypoints);
    {
      // Move to CPU, ensure contiguous
      const auto torch_keypoints_cpu = torch_keypoints.contiguous().cpu();
      const float* torch_keypoints_data = torch_keypoints_cpu.data_ptr<float>();
      for (int i = 0; i < num_keypoints; ++i) {
        (*keypoints)[i].x =
            0.5f * orig_width * (torch_keypoints_data[2 * i] + 1.f);
        (*keypoints)[i].y =
            0.5f * orig_height * (torch_keypoints_data[2 * i + 1] + 1.f);
      }
    }

    // Bulk copy the descriptors
    descriptors->resize(num_keypoints, descriptor_dim * sizeof(float));
    {
      // Move to CPU, ensure contiguous
      const auto torch_descriptors_cpu = torch_descriptors.contiguous().cpu();
      const float* torch_descriptors_data =
          torch_descriptors_cpu.data_ptr<float>();
      std::memcpy(descriptors->data(),
                  torch_descriptors_data,
                  num_keypoints * descriptor_dim * sizeof(float));
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  ALIKED aliked_;
};

#endif

}  // namespace

bool ALIKEDExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(score_threshold, 0);
  CHECK_OPTION_GE(top_k, -1);
  return true;
}

std::unique_ptr<FeatureExtractor> CreateALIKEDFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_TORCH_ENABLED
  return std::make_unique<ALIKEDFeatureExtractor>(options);
#else
  throw std::runtime_error(
      "ALIKED feature extraction requires libtorch support.");
#endif
}

bool ALIKEDMatchingOptions::Check() const {
  if (lightglue) {
    CHECK_OPTION(!lightglue_model_path.empty());
  }
  return true;
}

std::unique_ptr<FeatureMatcher> CreateALIKEDFeatureMatcher(
    const FeatureMatchingOptions& options) {
  if (options.aliked->lightglue) {
#ifdef COLMAP_TORCH_ENABLED
    LightGlueMatchingOptions lightglue_options;
    lightglue_options.model_path = options.aliked->lightglue_model_path;
    return CreateLightGlueFeatureMatcher(options, lightglue_options);
#else
    throw std::runtime_error(
        "ALIKED feature matching requires libtorch support.");
#endif
  } else {
    throw std::runtime_error("Not implemented.");
  }
}

}  // namespace colmap
