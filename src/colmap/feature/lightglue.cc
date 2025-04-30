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

#include "colmap/feature/lightglue.h"

#include "colmap/feature/torch_utils.h"
#include "colmap/util/file.h"

#include <memory>

#ifdef COLMAP_TORCH_ENABLED
#include "thirdparty/LightGlue/matcher.hpp"

#include <torch/torch.h>
#endif

namespace colmap {
namespace {

#ifdef COLMAP_TORCH_ENABLED

int GetInputDim(FeatureMatcherType type) {
  const static std::unordered_map<FeatureMatcherType, int> kTypeToDescDim{
      {FeatureMatcherType::LIGHTGLUE_SIFT, 128},
      {FeatureMatcherType::LIGHTGLUE_ALIKED, 128},
  };
  auto it = kTypeToDescDim.find(type);
  if (it == kTypeToDescDim.end()) {
    std::ostringstream error;
    error << "LightGlue input dimension not defined for feature matcher type: "
          << type;
    throw std::runtime_error(error.str());
  }
  return it->second;
}

class LightGlueFeatureMatcher : public FeatureMatcher {
 public:
  explicit LightGlueFeatureMatcher(
      const FeatureMatchingOptions& options,
      const LightGlueMatchingOptions& lightglue_options)
      : lightglue_(GetInputDim(options.type),
                   MaybeDownloadAndCacheFile(lightglue_options.model_path),
                   GetDeviceName(options.use_gpu, options.gpu_index)) {
    if (!options.Check()) {
      throw std::runtime_error("Invalid feature matching options.");
    }
    if (!lightglue_options.Check()) {
      throw std::runtime_error("Invalid LightGlue feature matching options.");
    }
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();

    if (image1.descriptors->rows() == 0 || image2.descriptors->rows() == 0) {
      return;
    }

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
    torch::Tensor matches_cpu = torch_matches0[0].cpu().contiguous();
    const int64_t* matches_ptr = matches_cpu.data_ptr<int64_t>();
    matches->reserve(num_keypoints1);
    for (int i = 0; i < num_keypoints1; ++i) {
      const int64_t j = matches_ptr[i];
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

    // Convert to contiguous array so we can use torch::from_blob below.
    std::vector<float> keypoints_array(num_keypoints * 2);
    for (int i = 0; i < num_keypoints; ++i) {
      const FeatureKeypoint& keypoint = (*image.keypoints)[i];
      keypoints_array[2 * i + 0] = keypoint.x;
      keypoints_array[2 * i + 1] = keypoint.y;
    }

    const auto kFloat32Options = torch::TensorOptions().dtype(torch::kFloat32);

    auto torch_keypoints =
        torch::from_blob(
            keypoints_array.data(), {num_keypoints, 2}, kFloat32Options)
            .clone();
    // Normalize to [-1, 1].
    torch_keypoints.index({torch::indexing::Ellipsis, 0}) =
        (2.0f * torch_keypoints.index({torch::indexing::Ellipsis, 0}) /
         image.width) -
        1.0f;
    torch_keypoints.index({torch::indexing::Ellipsis, 1}) =
        (2.0f * torch_keypoints.index({torch::indexing::Ellipsis, 1}) /
         image.height) -
        1.0f;

    torch::Dict<std::string, torch::Tensor> features;
    features.insert("keypoints", std::move(torch_keypoints));
    // TODO: The const_cast here is a little evil.
    features.insert(
        "descriptors",
        torch::from_blob(const_cast<uint8_t*>(image.descriptors->data()),
                         {num_keypoints, descriptor_dim},
                         kFloat32Options)
            .clone());

    return features;
  }

  matcher::LightGlue lightglue_;
};

#endif

}  // namespace

bool LightGlueMatchingOptions::Check() const {
  CHECK_OPTION(!model_path.empty());
  return true;
}

std::unique_ptr<FeatureMatcher> CreateLightGlueFeatureMatcher(
    const FeatureMatchingOptions& options,
    const LightGlueMatchingOptions& lightglue_options) {
#ifdef COLMAP_TORCH_ENABLED
  return std::make_unique<LightGlueFeatureMatcher>(options, lightglue_options);
#else
  throw std::runtime_error(
      "LightGlue feature matching requires torch support.");
#endif
}

}  // namespace colmap
