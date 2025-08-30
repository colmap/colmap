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

#include "colmap/feature/xfeat.h"

#include "colmap/util/file.h"
#include "colmap/util/threading.h"

#include <memory>

#ifdef COLMAP_ONNX_ENABLED
#include <onnxruntime_cxx_api.h>
#endif

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

static constexpr int kDescriptorDim = 64;

std::string FormatShape(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i < shape.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

// TODO(jsch): Use std::span for shape when we move to C++20.
void ThrowCheckNode(const std::string_view name,
                    const std::string_view expected_name,
                    const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& expected_shape) {
  THROW_CHECK_EQ(name, expected_name);
  THROW_CHECK_EQ(shape.size(), expected_shape.size())
      << "Invalid shape for " << name << ": " << FormatShape(shape)
      << " != " << FormatShape(expected_shape);
  for (size_t i = 0; i < shape.size(); ++i) {
    THROW_CHECK_EQ(shape[i], expected_shape[i])
        << "Invalid shape for " << name << ": " << FormatShape(shape)
        << " != " << FormatShape(expected_shape);
  }
}

struct ONNXModel {
  ONNXModel(std::string model_path,
            int num_threads,
            const std::string& gpu_index) {
    model_path = MaybeDownloadAndCacheFile(model_path);

    const int num_eff_threads = GetEffectiveNumThreads(num_threads);
    session_options.SetInterOpNumThreads(num_eff_threads);
    session_options.SetIntraOpNumThreads(num_eff_threads);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef COLMAP_CUDA_ENABLED
    const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1)
        << "ONNX model can only run on one GPU";
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = gpu_indices[0];
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif

    VLOG(2) << "Loading ONNX model from " << model_path;
    session = std::make_unique<Ort::Session>(
        env, model_path.c_str(), session_options);

    VLOG(2) << "Parsing the inputs";
    const int num_inputs = session->GetInputCount();
    input_name_strs.reserve(num_inputs);
    input_names.reserve(num_inputs);
    input_shapes.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      input_name_strs.emplace_back(
          session->GetInputNameAllocated(i, allocator));
      input_names.emplace_back(input_name_strs[i].get());
      input_shapes.emplace_back(
          session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    VLOG(2) << "Parsing the outputs";
    const int num_outputs = session->GetOutputCount();
    output_name_strs.reserve(num_outputs);
    output_names.reserve(num_outputs);
    output_shapes.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      output_name_strs.emplace_back(
          session->GetOutputNameAllocated(i, allocator));
      output_names.emplace_back(output_name_strs[i].get());
      output_shapes.emplace_back(
          session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
  }

  std::vector<Ort::Value> Run(const std::vector<Ort::Value>& input_tensors) {
    return session->Run(Ort::RunOptions(),
                        input_names.data(),
                        input_tensors.data(),
                        input_tensors.size(),
                        output_names.data(),
                        output_names.size());
  }

  Ort::Env env;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<Ort::AllocatedStringPtr> input_name_strs;
  std::vector<char*> input_names;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<Ort::AllocatedStringPtr> output_name_strs;
  std::vector<char*> output_names;
};

class XFeatFeatureExtractor : public FeatureExtractor {
 public:
  explicit XFeatFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options),
        model_(
            options.xfeat->model_path, options.num_threads, options.gpu_index) {
    THROW_CHECK(options.Check());

    THROW_CHECK_EQ(model_.input_shapes.size(), 1);
    ThrowCheckNode(model_.input_names[0],
                   "images",
                   model_.input_shapes[0],
                   {1, 3, -1, -1});

    THROW_CHECK_EQ(model_.output_shapes.size(), 3);
    ThrowCheckNode(
        model_.output_names[0], "keypoints", model_.output_shapes[0], {-1, 2});
    ThrowCheckNode(model_.output_names[1],
                   "descriptors",
                   model_.output_shapes[1],
                   {-1, 64});
    ThrowCheckNode(
        model_.output_names[2], "scores", model_.output_shapes[2], {-1});
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

    std::vector<float> row_major_array(num_pixels * 3);
    for (int y = 0; y < height; ++y) {
      const uint8_t* line = bitmap.GetScanline(y);
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < 3; ++c) {
          constexpr float kImageNormalization = 1.0f / 255.0f;
          row_major_array[c * num_pixels + y * width + x] =
              kImageNormalization * line[3 * x + c];
        }
      }
    }

    model_.input_shapes[0][0] = 1;
    model_.input_shapes[0][1] = 3;
    model_.input_shapes[0][2] = bitmap.Height();
    model_.input_shapes[0][3] = bitmap.Width();

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        row_major_array.data(),
        row_major_array.size(),
        model_.input_shapes[0].data(),
        model_.input_shapes[0].size()));

    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);
    THROW_CHECK_EQ(output_tensors.size(), 3);

    const std::vector<int64_t> keypoints_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(keypoints_shape.size(), 2);
    const int num_keypoints = keypoints_shape[0];
    THROW_CHECK_EQ(keypoints_shape[1], 2);

    const std::vector<int64_t> descriptors_shape =
        output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(descriptors_shape.size(), 2);
    THROW_CHECK_EQ(descriptors_shape[0], num_keypoints);
    THROW_CHECK_EQ(descriptors_shape[1], kDescriptorDim);

    const std::vector<int64_t> scores_shape =
        output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(scores_shape.size(), 1);
    THROW_CHECK_EQ(scores_shape[0], num_keypoints);

    const float* keypoints_data =
        reinterpret_cast<const float*>(output_tensors[0].GetTensorData<void>());
    const float* descriptors_data =
        reinterpret_cast<const float*>(output_tensors[1].GetTensorData<void>());
    const float* scores_data =
        reinterpret_cast<const float*>(output_tensors[2].GetTensorData<void>());

    // Filter features by score.
    int num_filtered_keypoints = 0;
    for (int i = 0; i < num_keypoints; ++i) {
      if (scores_data[i] >= options_.xfeat->min_score) {
        ++num_filtered_keypoints;
      }
    }

    const int num_selected_keypoints =
        std::min(num_filtered_keypoints, options_.xfeat->max_num_features);

    // Rank features by score.
    std::vector<int> sorted_indices(num_keypoints);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::partial_sort(sorted_indices.begin(),
                      sorted_indices.begin() + num_selected_keypoints,
                      sorted_indices.end(),
                      [&scores_data](const int i, const int j) {
                        return scores_data[i] > scores_data[j];
                      });

    // Copy selected keypoints and descriptors.
    keypoints->resize(num_selected_keypoints);
    descriptors->resize(num_selected_keypoints, kDescriptorDim * sizeof(float));
    for (int i = 0; i < num_selected_keypoints; ++i) {
      const int index = sorted_indices[i];
      (*keypoints)[i].x = keypoints_data[2 * index + 0];
      (*keypoints)[i].y = keypoints_data[2 * index + 1];
      std::memcpy(descriptors->data() + i * kDescriptorDim * sizeof(float),
                  descriptors_data + index * kDescriptorDim,  // float pointer
                  kDescriptorDim * sizeof(float));
    }

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  ONNXModel model_;
};

class LightGlueFeatureMatcher : public FeatureMatcher {
 public:
  explicit LightGlueFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options),
        model_(
            options.xfeat->model_path, options.num_threads, options.gpu_index) {
    THROW_CHECK(options.Check());
    THROW_CHECK_EQ(model_.input_shapes.size(), 3);
    ThrowCheckNode(model_.input_names[0],
                   "feats0",
                   model_.input_shapes[0],
                   {-1, kDescriptorDim});
    ThrowCheckNode(model_.input_names[1],
                   "feats1",
                   model_.input_shapes[1],
                   {-1, kDescriptorDim});
    ThrowCheckNode(
        model_.input_names[2], "min_cossim", model_.input_shapes[2], {1});
    THROW_CHECK_EQ(model_.output_shapes.size(), 1);
    ThrowCheckNode(
        model_.output_names[0], "matches", model_.output_shapes[0], {-1, 2});
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

    auto features1 = FeaturesFromImage(image1, model_.input_shapes[0]);
    auto features2 = FeaturesFromImage(image2, model_.input_shapes[1]);

    const std::vector<int64_t> min_cossim_shape = {1};
    auto min_sim_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        &options_.xfeat->min_cossim,
        sizeof(float),
        min_cossim_shape.data(),
        min_cossim_shape.size());

    std::vector<Ort::Value> input_tensors;
    // input_tensors.emplace_back(std::move(features1.keypoints_tensor));
    input_tensors.emplace_back(std::move(features1.descriptors_tensor));
    // input_tensors.emplace_back(std::move(features2.keypoints_tensor));
    input_tensors.emplace_back(std::move(features2.descriptors_tensor));
    input_tensors.emplace_back(std::move(min_sim_tensor));

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
    Ort::Value descriptors_tensor;
  };

  Features FeaturesFromImage(const Image& image,
                             std::vector<int64_t>& descriptors_shape) {
    THROW_CHECK_NE(image.image_id, kInvalidImageId);
    THROW_CHECK_NOTNULL(image.descriptors);

    const int num_keypoints = image.descriptors->rows();
    THROW_CHECK_EQ(image.descriptors->cols() % sizeof(float), 0);
    const int descriptor_dim = image.descriptors->cols() / sizeof(float);

    Features features;

    THROW_CHECK_EQ(descriptors_shape.size(), 2);
    descriptors_shape[0] = num_keypoints;
    descriptors_shape[1] = descriptor_dim;
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
        descriptors_shape.data(),
        descriptors_shape.size());

    return features;
  }

  const FeatureMatchingOptions options_;
  ONNXModel model_;
};

#endif

}  // namespace

bool XFeatExtractionOptions::Check() const {
  CHECK_OPTION_GE(max_num_features, 0);
  CHECK_OPTION_GE(min_score, 0);
  return true;
}

std::unique_ptr<FeatureExtractor> CreateXFeatFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<XFeatFeatureExtractor>(options);
#else
  throw std::runtime_error("XFeat feature extraction requires torch support.");
#endif
}

bool XFeatMatchingOptions::Check() const {
  CHECK_OPTION_GE(min_cossim, -1);
  CHECK_OPTION_LE(min_cossim, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateXFeatFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<LightGlueFeatureMatcher>(options);
#else
  throw std::runtime_error("XFeat feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
