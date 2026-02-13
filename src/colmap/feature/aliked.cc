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

#include "colmap/feature/onnx_matchers.h"
#include "colmap/feature/onnx_utils.h"

#include <algorithm>
#include <memory>

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

const std::string& GetExtractorModelPath(
    const FeatureExtractionOptions& options) {
  switch (options.type) {
    case FeatureExtractorType::ALIKED_N16ROT:
      return options.aliked->n16rot_model_path;
    case FeatureExtractorType::ALIKED_N32:
      return options.aliked->n32_model_path;
    default:
      throw std::runtime_error("Unknown ALIKED feature extractor type.");
  }
}

// Convert bitmap to row-major [C, H, W] float tensor, normalized to [0, 1].
std::vector<float> BitmapToInputTensor(const Bitmap& bitmap) {
  THROW_CHECK(bitmap.IsRGB());

  const int width = bitmap.Width();
  const int height = bitmap.Height();
  const int pitch = bitmap.Pitch();
  const int num_pixels = width * height;

  std::vector<float> input(num_pixels * 3);
  const std::vector<uint8_t>& data = bitmap.RowMajorData();
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        constexpr float kImageNormalization = 1.0f / 255.0f;
        input[c * num_pixels + y * width + x] =
            kImageNormalization * data[y * pitch + 3 * x + c];
      }
    }
  }

  return input;
}

// Pads image dimensions to be divisible by a given factor.
struct InputPadder {
  InputPadder(int height, int width, int divisor = 32)
      : original_height(height),
        original_width(width),
        padded_height(((height + divisor - 1) / divisor) * divisor),
        padded_width(((width + divisor - 1) / divisor) * divisor) {}

  const int original_height;
  const int original_width;
  const int padded_height;
  const int padded_width;

  // Pad a [C, H, W] float array by replicating edge pixels on the right and
  // bottom of the multi-channel image. Either returns a reference to the
  // input, if no padding is necessary, or a reference to the padded data.
  std::vector<float>* MaybePad(std::vector<float>& input, int channels) {
    if (padded_height == original_height && padded_width == original_width) {
      return &input;
    }

    padded_.resize(channels * padded_height * padded_width, 0.0f);
    for (int c = 0; c < channels; ++c) {
      for (int y = 0; y < padded_height; ++y) {
        const int src_y = std::min(y, original_height - 1);
        for (int x = 0; x < padded_width; ++x) {
          const int src_x = std::min(x, original_width - 1);
          padded_[c * padded_height * padded_width + y * padded_width + x] =
              input[c * original_height * original_width +
                    src_y * original_width + src_x];
        }
      }
    }
    return &padded_;
  }

 private:
  std::vector<float> padded_;
};

class AlikedFeatureExtractor : public FeatureExtractor {
 public:
  explicit AlikedFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options),
        model_(GetExtractorModelPath(options),
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options.Check());

    // Validate sparse model inputs: image [1, 3, H, W], max_keypoints (scalar),
    // min_score (scalar).
    THROW_CHECK_EQ(model_.input_shapes().size(), 3);
    ThrowCheckONNXNode(model_.input_names()[0],
                       "image",
                       model_.input_shapes()[0],
                       {-1, 3, -1, -1});
    ThrowCheckONNXNode(
        model_.input_names()[1], "max_keypoints", model_.input_shapes()[1], {});
    ThrowCheckONNXNode(
        model_.input_names()[2], "min_score", model_.input_shapes()[2], {});

    // Validate sparse model outputs: keypoints [1, K, 2], descriptors [1, K,
    // D], scores [1, K]. Note: Some dimensions may be dynamic (-1) in ONNX.
    THROW_CHECK_EQ(model_.output_shapes().size(), 3);
    ThrowCheckONNXNode(model_.output_names()[0],
                       "keypoints",
                       model_.output_shapes()[0],
                       {-1, -1, -1});
    ThrowCheckONNXNode(model_.output_names()[1],
                       "descriptors",
                       model_.output_shapes()[1],
                       {-1, -1, -1});
    descriptor_dim_ = static_cast<int>(model_.output_shapes()[1][2]);
    THROW_CHECK_GT(descriptor_dim_, 0);
    VLOG(2) << "ALIKED descriptor dimension: " << descriptor_dim_;
    ThrowCheckONNXNode(model_.output_names()[2],
                       "scores",
                       model_.output_shapes()[2],
                       {-1, -1});
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);
    THROW_CHECK(bitmap.IsRGB());

    const int width = bitmap.Width();
    const int height = bitmap.Height();

    std::vector<float> input = BitmapToInputTensor(bitmap);

    // Pad image to dimensions divisible by 32.
    InputPadder padder(height, width, /*divisor=*/32);
    std::vector<float>* padded_input = padder.MaybePad(input, 3);

    // Prepare image input tensor.
    std::vector<int64_t> image_shape = model_.input_shapes()[0];
    image_shape[0] = 1;
    image_shape[1] = 3;
    image_shape[2] = padder.padded_height;
    image_shape[3] = padder.padded_width;

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        padded_input->data(),
        padded_input->size(),
        image_shape.data(),
        image_shape.size()));

    // Prepare max_keypoints input tensor (scalar).
    int64_t max_keypoints = options_.aliked->max_num_features;
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        &max_keypoints,
        1,
        model_.input_shapes()[1].data(),
        model_.input_shapes()[1].size()));

    // Prepare min_score input tensor (scalar).
    float min_score = static_cast<float>(options_.aliked->min_score);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        &min_score,
        1,
        model_.input_shapes()[2].data(),
        model_.input_shapes()[2].size()));

    // Run model inference.
    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);
    THROW_CHECK_EQ(output_tensors.size(), 3);

    // Parse keypoints shape: [1, K, 2].
    const std::vector<int64_t> keypoints_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(keypoints_shape.size(), 3);
    THROW_CHECK_EQ(keypoints_shape[0], 1);
    const int num_keypoints = static_cast<int>(keypoints_shape[1]);
    THROW_CHECK_EQ(keypoints_shape[2], 2);

    // Parse descriptors shape: [1, K, D].
    const std::vector<int64_t> descriptors_shape =
        output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(descriptors_shape.size(), 3);
    THROW_CHECK_EQ(descriptors_shape[0], 1);
    THROW_CHECK_EQ(descriptors_shape[1], num_keypoints);
    THROW_CHECK_EQ(descriptors_shape[2], descriptor_dim_);

    // Parse scores shape: [1, K].
    const std::vector<int64_t> scores_shape =
        output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(scores_shape.size(), 2);
    THROW_CHECK_EQ(scores_shape[0], 1);
    THROW_CHECK_EQ(scores_shape[1], num_keypoints);

    const float* keypoints_data = output_tensors[0].GetTensorData<float>();
    const float* descriptors_data = output_tensors[1].GetTensorData<float>();

    // Convert keypoints from normalized [-1, 1] to pixel coordinates,
    // where ALIKED uses the center of the top-left pixel as (0, 0),
    // while COLMAP uses the top-left pixel's corner as (0, 0).
    // Filter out keypoints in the padded region (outside original image
    // bounds).
    const float scale_x = 0.5f * static_cast<float>(padder.padded_width - 1);
    const float scale_y = 0.5f * static_cast<float>(padder.padded_height - 1);

    // Collect valid keypoints, their pixel coordinates, and descriptor indices.
    struct ValidKeypoint {
      float x, y;
      int index;
    };
    std::vector<ValidKeypoint> valid_keypoints;
    valid_keypoints.reserve(num_keypoints);
    for (int i = 0; i < num_keypoints; ++i) {
      const float norm_x = keypoints_data[2 * i + 0];
      const float norm_y = keypoints_data[2 * i + 1];
      const float px = (norm_x + 1.0f) * scale_x + 0.5f;
      const float py = (norm_y + 1.0f) * scale_y + 0.5f;
      if (px >= 0.0f && px < width && py >= 0.0f && py < height) {
        valid_keypoints.push_back({px, py, i});
      }
    }

    // Populate output with valid keypoints and descriptors.
    const int num_valid = static_cast<int>(valid_keypoints.size());
    keypoints->resize(num_valid);
    descriptors->type = options_.type;
    descriptors->data.resize(num_valid, descriptor_dim_ * sizeof(float));
    for (int j = 0; j < num_valid; ++j) {
      const auto& kp = valid_keypoints[j];
      (*keypoints)[j].x = kp.x;
      (*keypoints)[j].y = kp.y;
      std::memcpy(
          descriptors->data.data() + j * descriptor_dim_ * sizeof(float),
          descriptors_data + kp.index * descriptor_dim_,
          descriptor_dim_ * sizeof(float));
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  ONNXModel model_;
  int descriptor_dim_;
};

#endif

}  // namespace

bool AlikedExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_LE(min_score, 1);
  return true;
}

std::unique_ptr<FeatureExtractor> CreateAlikedFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<AlikedFeatureExtractor>(options);
#else
  throw std::runtime_error("ALIKED feature extraction requires ONNX support.");
#endif
}

bool AlikedMatchingOptions::Check() const {
  if (!brute_force.Check() || !lightglue.Check()) {
    return false;
  }
  return true;
}

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  switch (options.type) {
    case FeatureMatcherType::ALIKED_BRUTEFORCE:
      return CreateBruteForceONNXFeatureMatcher(options,
                                                options.aliked->brute_force);
    case FeatureMatcherType::ALIKED_LIGHTGLUE:
      return CreateLightGlueONNXFeatureMatcher(options,
                                               options.aliked->lightglue);
    default:
      throw std::runtime_error("Unknown ALIKED matcher type.");
  }
#else
  throw std::runtime_error("ALIKED feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
