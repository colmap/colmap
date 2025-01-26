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

#include "colmap/util/file.h"

#include <memory>

#ifdef COLMAP_TORCH_ENABLED
#include "thirdparty/ALIKED/aliked.hpp"
#include "thirdparty/LightGlue/matcher.hpp"

#include <torch/torch.h>
#endif

namespace colmap {
namespace {

#ifdef COLMAP_TORCH_ENABLED

std::string GetDeviceName() {
  if (torch::cuda::is_available()) {
    return "cuda";
  } else {
    return "cpu";
  }
}

class ALIKEDFeatureExtractor : public FeatureExtractor {
 public:
  explicit ALIKEDFeatureExtractor(const ALIKEDFeatureExtractionOptions& options)
      : options_(options),
        aliked_(options.model_name,
                MaybeDownloadAndCacheFile(options.model_path),
                GetDeviceName(),
                /*top_k=*/options.top_k,
                /*scores_th=*/options.score_threshold,
                /*n_limit=*/options.max_num_features) {}

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
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
    torch::Tensor torch_image = torch::empty({1, 3, height, width});
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        BitmapColor<uint8_t> color;
        CHECK(bitmap_ptr->GetPixel(x, y, &color));
        constexpr float kNorm = 1.f / 255.f;
        torch_image[0][0][y][x] = kNorm * color.r;
        torch_image[0][1][y][x] = kNorm * color.g;
        torch_image[0][2][y][x] = kNorm * color.b;
      }
    }

    const torch::Dict<std::string, torch::Tensor> outputs =
        aliked_.forward(torch_image);

    const auto& torch_keypoints = outputs.at("keypoints");
    const int num_keypoints = torch_keypoints.size(0);

    keypoints->resize(num_keypoints);
    for (int i = 0; i < num_keypoints; ++i) {
      (*keypoints)[i].x =
          0.5f * orig_width * (torch_keypoints[i][0].item<float>() + 1.f);
      (*keypoints)[i].y =
          0.5f * orig_height * (torch_keypoints[i][1].item<float>() + 1.f);
    }

    const auto& torch_descriptors = outputs.at("descriptors");
    const int num_descriptors = torch_descriptors.size(0);
    THROW_CHECK_EQ(num_descriptors, num_keypoints);
    const int descriptor_dim = torch_descriptors.size(1);
    descriptors->resize(num_descriptors, descriptor_dim * sizeof(float));
    torch::from_blob(reinterpret_cast<float*>(descriptors->data()),
                     {num_descriptors, descriptor_dim},
                     torch::TensorOptions().dtype(torch::kFloat32)) =
        outputs.at("descriptors");

    return true;
  }

 private:
  const ALIKEDFeatureExtractionOptions options_;
  ALIKED aliked_;
};

class ALIKEDLightGlueFeatureMatcher : public FeatureMatcher {
 public:
  explicit ALIKEDLightGlueFeatureMatcher(
      const ALIKEDFeatureMatchingOptions& options)
      : lightglue_("aliked", MaybeDownloadAndCacheFile(options.model_path)) {}

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);

    // TODO: Cache the torch tensors if the same image is passed.

    const torch::Dict<std::string, torch::Tensor> features1 =
        FeaturesFromImage(image1);
    const torch::Dict<std::string, torch::Tensor> features2 =
        FeaturesFromImage(image2);

    const torch::Dict<std::string, torch::Tensor> outputs =
        lightglue_.forward(features1, features2);

    const auto& torch_matches0 = outputs.at("matches0");
    THROW_CHECK_EQ(torch_matches0.size(0), 1);
    const int num_matches = torch_matches0.size(1);
    const int num_keypoints1 = image1.keypoints->size();
    const int num_keypoints2 = image2.keypoints->size();
    THROW_CHECK_EQ(num_matches, num_keypoints1);
    matches->reserve(num_matches);
    for (int i = 0; i < num_keypoints1; ++i) {
      const int64_t j = torch_matches0[0][i].item<int64_t>();
      if (j >= 0) {
        FeatureMatch match;
        match.point2D_idx1 = i;
        match.point2D_idx2 = j;
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
  torch::Dict<std::string, torch::Tensor> FeaturesFromImage(
      const Image& image) {
    THROW_CHECK_NE(image.image_id, kInvalidImageId);
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK_NOTNULL(image.keypoints);
    THROW_CHECK_EQ(image.descriptors->rows(), image.keypoints->size());

    const int num_keypoints = image.keypoints->size();
    THROW_CHECK_EQ(image.descriptors->cols() % sizeof(float), 0);
    const int descriptor_dim = image.descriptors->cols() / sizeof(float);

    torch::Dict<std::string, torch::Tensor> features;
    features.insert("image_size",
                    torch::tensor({static_cast<float>(image.width),
                                   static_cast<float>(image.height)},
                                  torch::kFloat32)
                        .unsqueeze(0));
    torch::Tensor torch_keypoints = torch::empty({num_keypoints, 2});
    for (int i = 0; i < num_keypoints; ++i) {
      const FeatureKeypoint& keypoint = (*image.keypoints)[i];
      torch_keypoints[i][0] = 2.f * keypoint.x / image.width - 1.f;
      torch_keypoints[i][1] = 2.f * keypoint.y / image.height - 1.f;
    }
    features.insert("keypoints", std::move(torch_keypoints));
    // TODO: The const_cast here is a little evil.
    features.insert(
        "descriptors",
        torch::from_blob(const_cast<uint8_t*>(image.descriptors->data()),
                         {num_keypoints, descriptor_dim},
                         torch::TensorOptions().dtype(torch::kFloat32)));

    return features;
  }

  matcher::LightGlue lightglue_;
};

#endif

}  // namespace

bool ALIKEDFeatureExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_image_size, 0);
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(score_threshold, 0);
  CHECK_OPTION_GE(top_k, -1);
  return true;
}

std::unique_ptr<FeatureExtractor> CreateALIKEDFeatureExtractor(
    const ALIKEDFeatureExtractionOptions& options) {
#ifdef COLMAP_TORCH_ENABLED
  return std::make_unique<ALIKEDFeatureExtractor>(options);
#else
  throw std::runtime_error(
      "ALIKED feature extraction requires libtorch support.");
#endif
}

std::unique_ptr<FeatureMatcher> CreateALIKEDLightGlueFeatureMatcher(
    const ALIKEDFeatureMatchingOptions& options) {
#ifdef COLMAP_TORCH_ENABLED
  return std::make_unique<ALIKEDLightGlueFeatureMatcher>(options);
#else
  throw std::runtime_error(
      "ALIKED feature matching requires libtorch support.");
#endif
}

}  // namespace colmap
