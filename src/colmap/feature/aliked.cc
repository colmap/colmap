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

#include "colmap/feature/onnx_utils.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

// Pads image dimensions to be divisible by a given factor.
class InputPadder {
 public:
  InputPadder(int height, int width, int divisor = 32)
      : original_height_(height), original_width_(width) {
    padded_height_ = ((height + divisor - 1) / divisor) * divisor;
    padded_width_ = ((width + divisor - 1) / divisor) * divisor;
  }

  int OriginalHeight() const { return original_height_; }
  int OriginalWidth() const { return original_width_; }
  int PaddedHeight() const { return padded_height_; }
  int PaddedWidth() const { return padded_width_; }

  // Pad a CHW float array by replicating edge pixels.
  std::vector<float> Pad(const std::vector<float>& input, int channels) const {
    if (padded_height_ == original_height_ &&
        padded_width_ == original_width_) {
      return input;
    }

    std::vector<float> output(channels * padded_height_ * padded_width_, 0.0f);
    for (int c = 0; c < channels; ++c) {
      for (int y = 0; y < padded_height_; ++y) {
        const int src_y = std::min(y, original_height_ - 1);
        for (int x = 0; x < padded_width_; ++x) {
          const int src_x = std::min(x, original_width_ - 1);
          output[c * padded_height_ * padded_width_ + y * padded_width_ + x] =
              input[c * original_height_ * original_width_ +
                    src_y * original_width_ + src_x];
        }
      }
    }
    return output;
  }

  // Unpad a score map from [1, 1, padded_H, padded_W] to [original_H,
  // original_W].
  std::vector<float> UnpadScoreMap(const float* data) const {
    std::vector<float> output(original_height_ * original_width_);
    for (int y = 0; y < original_height_; ++y) {
      for (int x = 0; x < original_width_; ++x) {
        output[y * original_width_ + x] = data[y * padded_width_ + x];
      }
    }
    return output;
  }

 private:
  int original_height_;
  int original_width_;
  int padded_height_;
  int padded_width_;
};

// Simple NMS using max filter (morphological dilation).
void SimpleNms(std::vector<float>& scores, int height, int width, int radius) {
  std::vector<float> max_vals(height * width);

  // Apply max filter.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float max_val = scores[y * width + x];
      for (int ky = -radius; ky <= radius; ++ky) {
        const int ny = y + ky;
        if (ny < 0 || ny >= height) continue;
        for (int kx = -radius; kx <= radius; ++kx) {
          const int nx = x + kx;
          if (nx < 0 || nx >= width) continue;
          max_val = std::max(max_val, scores[ny * width + nx]);
        }
      }
      max_vals[y * width + x] = max_val;
    }
  }

  // Suppress non-maxima.
  for (int i = 0; i < height * width; ++i) {
    if (scores[i] != max_vals[i]) {
      scores[i] = 0.0f;
    }
  }
}

struct Keypoint {
  float x;
  float y;
  float score;
};

// Detect keypoints from score map.
std::vector<Keypoint> DetectKeypoints(const std::vector<float>& scores,
                                      int height,
                                      int width,
                                      double min_score) {
  std::vector<Keypoint> keypoints;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float score = scores[y * width + x];
      if (score > min_score) {
        keypoints.push_back({static_cast<float>(x) + 0.5f,
                             static_cast<float>(y) + 0.5f,
                             score});
      }
    }
  }
  return keypoints;
}

// Apply soft-argmax refinement to keypoint positions.
void RefineKeypoints(std::vector<Keypoint>& keypoints,
                     const std::vector<float>& scores,
                     int height,
                     int width,
                     int window_size = 5) {
  const int half_window = window_size / 2;

  for (auto& kp : keypoints) {
    const int cx = static_cast<int>(kp.x);
    const int cy = static_cast<int>(kp.y);

    float sum_weight = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (int dy = -half_window; dy <= half_window; ++dy) {
      const int ny = cy + dy;
      if (ny < 0 || ny >= height) continue;
      for (int dx = -half_window; dx <= half_window; ++dx) {
        const int nx = cx + dx;
        if (nx < 0 || nx >= width) continue;

        const float weight = std::exp(scores[ny * width + nx]);
        sum_weight += weight;
        sum_x += weight * (nx + 0.5f);
        sum_y += weight * (ny + 0.5f);
      }
    }

    if (sum_weight > 0.0f) {
      kp.x = sum_x / sum_weight;
      kp.y = sum_y / sum_weight;
    }
  }
}

// Sample descriptors using bilinear interpolation.
void SampleDescriptors(const float* descriptor_map,
                       int desc_dim,
                       int map_height,
                       int map_width,
                       const std::vector<Keypoint>& keypoints,
                       std::vector<float>& descriptors) {
  const int num_keypoints = keypoints.size();
  descriptors.resize(num_keypoints * desc_dim);

  for (int i = 0; i < num_keypoints; ++i) {
    // Convert keypoint coordinates to descriptor map coordinates.
    // The descriptor map has the same spatial dimensions as the score map
    // after padding.
    const float x = keypoints[i].x - 0.5f;
    const float y = keypoints[i].y - 0.5f;

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float wx1 = x - x0;
    const float wy1 = y - y0;
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;

    // Clamp coordinates.
    const int x0c = std::max(0, std::min(x0, map_width - 1));
    const int x1c = std::max(0, std::min(x1, map_width - 1));
    const int y0c = std::max(0, std::min(y0, map_height - 1));
    const int y1c = std::max(0, std::min(y1, map_height - 1));

    // Bilinear interpolation for each descriptor dimension.
    for (int d = 0; d < desc_dim; ++d) {
      const float v00 =
          descriptor_map[d * map_height * map_width + y0c * map_width + x0c];
      const float v01 =
          descriptor_map[d * map_height * map_width + y0c * map_width + x1c];
      const float v10 =
          descriptor_map[d * map_height * map_width + y1c * map_width + x0c];
      const float v11 =
          descriptor_map[d * map_height * map_width + y1c * map_width + x1c];

      descriptors[i * desc_dim + d] =
          wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
    }
  }
}

// L2 normalize descriptors.
void NormalizeDescriptors(std::vector<float>& descriptors, int desc_dim) {
  const int num_keypoints = descriptors.size() / desc_dim;
  for (int i = 0; i < num_keypoints; ++i) {
    float norm = 0.0f;
    for (int d = 0; d < desc_dim; ++d) {
      norm += descriptors[i * desc_dim + d] * descriptors[i * desc_dim + d];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
      for (int d = 0; d < desc_dim; ++d) {
        descriptors[i * desc_dim + d] /= norm;
      }
    }
  }
}

class AlikedFeatureExtractor : public FeatureExtractor {
 public:
  explicit AlikedFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options),
        model_(options.aliked->model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options.Check());

    // Verify model inputs: expects single image input.
    THROW_CHECK_EQ(model_.input_shapes.size(), 1);
    THROW_CHECK_EQ(model_.input_shapes[0].size(), 4);
    THROW_CHECK_EQ(model_.input_shapes[0][0], 1);   // batch
    THROW_CHECK_EQ(model_.input_shapes[0][1], 3);   // channels
    THROW_CHECK_EQ(model_.input_shapes[0][2], -1);  // height (dynamic)
    THROW_CHECK_EQ(model_.input_shapes[0][3], -1);  // width (dynamic)

    // Verify model outputs: descriptor_map and score_map.
    THROW_CHECK_EQ(model_.output_shapes.size(), 2);

    // descriptor_map: [1, D, H, W] where D is descriptor dimension.
    THROW_CHECK_EQ(model_.output_shapes[0].size(), 4);
    descriptor_dim_ = static_cast<int>(model_.output_shapes[0][1]);
    THROW_CHECK_GT(descriptor_dim_, 0);
    VLOG(2) << "ALIKED descriptor dimension: " << descriptor_dim_;

    // score_map: [1, 1, H, W].
    THROW_CHECK_EQ(model_.output_shapes[1].size(), 4);
    THROW_CHECK_EQ(model_.output_shapes[1][1], 1);
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);
    THROW_CHECK(bitmap.IsRGB());

    const int width = bitmap.Width();
    const int height = bitmap.Height();
    const int num_pixels = width * height;

    // Create padder to make dimensions divisible by 32.
    InputPadder padder(height, width, 32);

    // Convert bitmap to CHW float array, normalized to [0, 1].
    std::vector<float> row_major_array(num_pixels * 3);
    const std::vector<uint8_t>& data = bitmap.RowMajorData();
    const int pitch = bitmap.Pitch();
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < 3; ++c) {
          constexpr float kImageNormalization = 1.0f / 255.0f;
          row_major_array[c * num_pixels + y * width + x] =
              kImageNormalization * data[y * pitch + 3 * x + c];
        }
      }
    }

    // Pad the image if necessary.
    std::vector<float> padded_array = padder.Pad(row_major_array, 3);

    const int padded_height = padder.PaddedHeight();
    const int padded_width = padder.PaddedWidth();

    model_.input_shapes[0][0] = 1;
    model_.input_shapes[0][1] = 3;
    model_.input_shapes[0][2] = padded_height;
    model_.input_shapes[0][3] = padded_width;

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        padded_array.data(),
        padded_array.size(),
        model_.input_shapes[0].data(),
        model_.input_shapes[0].size()));

    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);
    THROW_CHECK_EQ(output_tensors.size(), 2);

    // Parse descriptor_map shape.
    const std::vector<int64_t> desc_map_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(desc_map_shape.size(), 4);
    THROW_CHECK_EQ(desc_map_shape[0], 1);
    const int desc_dim = static_cast<int>(desc_map_shape[1]);
    THROW_CHECK_EQ(desc_dim, descriptor_dim_);
    const int map_height = static_cast<int>(desc_map_shape[2]);
    const int map_width = static_cast<int>(desc_map_shape[3]);

    // Parse score_map shape.
    const std::vector<int64_t> score_map_shape =
        output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(score_map_shape.size(), 4);
    THROW_CHECK_EQ(score_map_shape[0], 1);
    THROW_CHECK_EQ(score_map_shape[1], 1);
    THROW_CHECK_EQ(score_map_shape[2], map_height);
    THROW_CHECK_EQ(score_map_shape[3], map_width);

    const float* descriptor_map_data = output_tensors[0].GetTensorData<float>();
    const float* score_map_data = output_tensors[1].GetTensorData<float>();

    // Unpad score map to original dimensions.
    // The score map from the model is at the padded resolution.
    std::vector<float> scores(height * width);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        scores[y * width + x] = score_map_data[y * map_width + x];
      }
    }

    // Apply NMS.
    SimpleNms(scores, height, width, options_.aliked->nms_radius);

    // Detect keypoints.
    std::vector<Keypoint> detected_keypoints =
        DetectKeypoints(scores, height, width, options_.aliked->min_score);

    // Apply soft-argmax refinement.
    RefineKeypoints(detected_keypoints, scores, height, width);

    // Sort by score and select top-k if specified.
    std::sort(
        detected_keypoints.begin(),
        detected_keypoints.end(),
        [](const Keypoint& a, const Keypoint& b) { return a.score > b.score; });

    int num_selected = detected_keypoints.size();
    if (options_.aliked->max_num_features > 0 &&
        num_selected > options_.aliked->max_num_features) {
      num_selected = options_.aliked->max_num_features;
    }
    detected_keypoints.resize(num_selected);

    // Sample descriptors at keypoint locations.
    // The descriptor map is at padded resolution, but keypoints are in original
    // coordinates. We need to sample from the padded descriptor map.
    std::vector<float> sampled_descriptors;
    SampleDescriptors(descriptor_map_data,
                      desc_dim,
                      map_height,
                      map_width,
                      detected_keypoints,
                      sampled_descriptors);

    // Normalize descriptors.
    NormalizeDescriptors(sampled_descriptors, desc_dim);

    // Copy to output.
    keypoints->resize(num_selected);
    descriptors->resize(num_selected, desc_dim * sizeof(float));
    for (int i = 0; i < num_selected; ++i) {
      (*keypoints)[i].x = detected_keypoints[i].x;
      (*keypoints)[i].y = detected_keypoints[i].y;
      std::memcpy(descriptors->data() + i * desc_dim * sizeof(float),
                  sampled_descriptors.data() + i * desc_dim,
                  desc_dim * sizeof(float));
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  ONNXModel model_;
  int descriptor_dim_;
};

class AlikedBruteForceFeatureMatcher : public FeatureMatcher {
 public:
  explicit AlikedBruteForceFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options),
        model_(options.aliked->bruteforce_model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options.Check());
    THROW_CHECK_EQ(model_.input_shapes.size(), 3);
    THROW_CHECK_EQ(model_.output_shapes.size(), 1);
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

    auto features1 = FeaturesFromImage(image1);
    auto features2 = FeaturesFromImage(image2);

    float min_cossim = static_cast<float>(options_.aliked->min_cossim);
    const std::vector<int64_t> min_cossim_shape = {1};
    auto min_cossim_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        &min_cossim,
        sizeof(float),
        min_cossim_shape.data(),
        min_cossim_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(features1.descriptors_tensor));
    input_tensors.emplace_back(std::move(features2.descriptors_tensor));
    input_tensors.emplace_back(std::move(min_cossim_tensor));

    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);
    THROW_CHECK_EQ(output_tensors.size(), 1);

    const std::vector<int64_t> matches_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(matches_shape.size(), 2);
    const int num_matches = matches_shape[0];
    THROW_CHECK_EQ(matches_shape[1], 2);

    if (num_matches == 0) {
      return;
    }

    const int64_t* matches_data = reinterpret_cast<const int64_t*>(
        output_tensors[0].GetTensorData<void>());
    matches->resize(num_matches);
    for (int i = 0; i < num_matches; ++i) {
      FeatureMatch& match = (*matches)[i];
      match.point2D_idx1 = matches_data[2 * i + 0];
      match.point2D_idx2 = matches_data[2 * i + 1];
      THROW_CHECK_GE(match.point2D_idx1, 0);
      THROW_CHECK_LT(match.point2D_idx1, num_keypoints1);
      THROW_CHECK_GE(match.point2D_idx2, 0);
      THROW_CHECK_LT(match.point2D_idx2, num_keypoints2);
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
  struct Features {
    std::vector<float> descriptors_data;
    std::vector<int64_t> descriptors_shape;
    Ort::Value descriptors_tensor;
  };

  Features FeaturesFromImage(const Image& image) {
    THROW_CHECK_NE(image.image_id, kInvalidImageId);
    THROW_CHECK_NOTNULL(image.descriptors);

    const int num_keypoints = image.descriptors->rows();
    THROW_CHECK_EQ(image.descriptors->cols() % sizeof(float), 0);
    const int descriptor_dim = image.descriptors->cols() / sizeof(float);

    Features features;
    features.descriptors_shape = {num_keypoints, descriptor_dim};
    features.descriptors_data.resize(num_keypoints * descriptor_dim);
    THROW_CHECK_EQ(image.descriptors->size(),
                   features.descriptors_data.size() * sizeof(float));
    std::memcpy(features.descriptors_data.data(),
                reinterpret_cast<const void*>(image.descriptors->data()),
                image.descriptors->size());

    features.descriptors_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        features.descriptors_data.data(),
        features.descriptors_data.size(),
        features.descriptors_shape.data(),
        features.descriptors_shape.size());

    return features;
  }

  const FeatureMatchingOptions options_;
  ONNXModel model_;
};

#endif

}  // namespace

bool AlikedExtractionOptions::Check() const {
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_GE(nms_radius, 0);
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
  CHECK_OPTION_GE(min_cossim, -1);
  CHECK_OPTION_LE(min_cossim, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateAlikedFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  switch (options.type) {
    case FeatureMatcherType::ALIKED_BRUTEFORCE:
      return std::make_unique<AlikedBruteForceFeatureMatcher>(options);
    default:
      throw std::runtime_error("Unknown ALIKED matcher type.");
  }
#else
  throw std::runtime_error("ALIKED feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
