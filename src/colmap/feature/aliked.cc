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
    const bool needs_rescale = image_size > options_.aliked->max_image_size;

    // Only copy bitmap if we need to rescale or convert to RGB.
    const Bitmap* bitmap_ptr = &bitmap;
    std::unique_ptr<Bitmap> maybe_bitmap_copy;
    if (!bitmap.IsRGB() || needs_rescale) {
      maybe_bitmap_copy = std::make_unique<Bitmap>(bitmap.CloneAsRGB());
      bitmap_ptr = maybe_bitmap_copy.get();
    }

    if (needs_rescale) {
      const double scale =
          static_cast<double>(options_.aliked->max_image_size) /
          static_cast<double>(image_size);
      maybe_bitmap_copy->Rescale(scale * orig_width, scale * orig_height);
    }

    const int width = bitmap_ptr->Width();
    const int height = bitmap_ptr->Height();
    auto row_major_array = bitmap_ptr->ConvertToRowMajorArray();
    // Clone to ensure ownership
    const torch::Tensor torch_image =
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

class ALIKEDDescriptorFeatureMatcher : public FeatureMatcher {
 public:
  explicit ALIKEDDescriptorFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options),
        device_(GetDeviceName(options.use_gpu, options.gpu_index)) {
    if (!options.Check()) {
      throw std::runtime_error("Invalid feature matching options.");
    }
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();

    const int num_keypoints1 = image1.descriptors->rows();
    const int num_keypoints2 = image2.descriptors->rows();
    if (num_keypoints1 == 0 || num_keypoints2 == 0) {
      return;
    }

    auto descriptors1 = FeaturesFromImage(image1);
    auto descriptors2 = FeaturesFromImage(image2);

    auto sim = torch::matmul(descriptors1, descriptors2.transpose(0, 1));
    sim = torch::where(sim < options_.aliked->min_similarity, 0, sim);
    const auto nn12 = torch::argmax(sim, /*axis=*/1).contiguous().cpu();
    const int64_t* nn12_data = nn12.data_ptr<int64_t>();
    const auto nn21 = torch::argmax(sim, /*axis=*/0).contiguous().cpu();
    const int64_t* nn21_data = nn21.data_ptr<int64_t>();

    matches->reserve(num_keypoints1);
    for (int i1 = 0; i1 < num_keypoints1; ++i1) {
      const int i2 = nn12_data[i1];
      if (i1 == nn21_data[i2]) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = i2;
        THROW_CHECK_LT(match.point2D_idx2, num_keypoints2);
        matches->push_back(match);
      }
    }
  }

  void MatchGuided(double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    THROW_CHECK_GE(max_error, 0);
    Match(image1, image2, &two_view_geometry->inlier_matches);
  }

 private:
  torch::Tensor FeaturesFromImage(const Image& image) {
    THROW_CHECK_NE(image.image_id, kInvalidImageId);
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK_NOTNULL(image.keypoints);
    THROW_CHECK_EQ(image.descriptors->rows(), image.keypoints->size());

    const int num_keypoints = image.keypoints->size();
    THROW_CHECK_EQ(image.descriptors->cols() % sizeof(float), 0);
    const int descriptor_dim = image.descriptors->cols() / sizeof(float);

    return torch::from_blob(const_cast<uint8_t*>(image.descriptors->data()),
                            {num_keypoints, descriptor_dim},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .to(device_);
  }

  const FeatureMatchingOptions options_;
  const torch::Device device_;
};

#endif

}  // namespace

bool ALIKEDExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_image_size, 0);
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
  throw std::runtime_error("ALIKED feature extraction requires torch support.");
#endif
}

bool ALIKEDMatchingOptions::Check() const {
  CHECK_OPTION_GE(min_similarity, -1);
  CHECK_OPTION_LE(min_similarity, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateALIKEDFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_TORCH_ENABLED
  if (options.type == FeatureMatcherType::LIGHTGLUE_ALIKED) {
    LightGlueMatchingOptions lightglue_options;
    lightglue_options.model_path = options.aliked->lightglue_model_path;
    lightglue_options.descriptor_data_type =
        LightGlueMatchingOptions::DescriptorDataType::FLOAT32;
    return CreateLightGlueFeatureMatcher(options, lightglue_options);
  } else {
    return std::make_unique<ALIKEDDescriptorFeatureMatcher>(options);
  }
#else
  throw std::runtime_error("ALIKED feature matching requires torch support.");
#endif
}

}  // namespace colmap
