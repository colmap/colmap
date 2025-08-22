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
#include "colmap/util/threading.h"

#include <memory>

// TODO(jsch): Remove and configure in CMake.
#define COLMAP_ONNX_ENABLED

#ifdef COLMAP_ONNX_ENABLED
#include <onnxruntime_cxx_api.h>
#endif

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

struct ONNXModel {
  ONNXModel(const std::string& model_path, int num_threads) {
    session_options.SetInterOpNumThreads(GetEffectiveNumThreads(num_threads));
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(
        env, model_path.c_str(), session_options);

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

class ALIKEDFeatureExtractor : public FeatureExtractor {
 public:
  explicit ALIKEDFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options),
        model_(options.aliked->model_path, options.num_threads) {
    THROW_CHECK(options.Check());
    THROW_CHECK_EQ(model_.input_shapes.size(), 1);
    THROW_CHECK_EQ(std::string_view(model_.input_names[0]), "image");
    THROW_CHECK(model_.input_shapes[0] == std::vector<int64_t>({1, 1, -1, -1}));
    THROW_CHECK_EQ(model_.output_shapes.size(), 3);
    THROW_CHECK_EQ(std::string_view(model_.output_names[0]), "keypoints");
    THROW_CHECK(model_.output_shapes[0] == std::vector<int64_t>({1, -1, 2}));
    THROW_CHECK_EQ(std::string_view(model_.output_names[1]), "scores");
    THROW_CHECK(model_.output_shapes[1] == std::vector<int64_t>({1, -1}));
    THROW_CHECK_EQ(std::string_view(model_.output_names[2]), "descriptors");
    THROW_CHECK(model_.output_shapes[2] == std::vector<int64_t>({1, -1, 256}));
  }

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
    if (!bitmap.IsGrey() || needs_rescale) {
      maybe_bitmap_copy = std::make_unique<Bitmap>(bitmap.CloneAsGrey());
      bitmap_ptr = maybe_bitmap_copy.get();
    }

    if (needs_rescale) {
      const double scale =
          static_cast<double>(options_.aliked->max_image_size) /
          static_cast<double>(image_size);
      maybe_bitmap_copy->Rescale(scale * orig_width, scale * orig_height);
    }

    const std::vector<uint8_t> row_major_array =
        bitmap_ptr->ConvertToRowMajorArray();
    std::vector<float> input_data(row_major_array.size());
    for (size_t i = 0; i < row_major_array.size(); ++i) {
      input_data[i] = (1.0f / 255.0f) * static_cast<float>(row_major_array[i]);
    }

    model_.input_shapes[0][0] = 1;
    model_.input_shapes[0][1] = 1;
    model_.input_shapes[0][2] = bitmap_ptr->Height();
    model_.input_shapes[0][3] = bitmap_ptr->Width();

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        input_data.data(),
        input_data.size(),
        model_.input_shapes[0].data(),
        model_.input_shapes[0].size()));

    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);

    const std::vector<int64_t> keypoints_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(keypoints_shape.size(), 3);
    THROW_CHECK_EQ(keypoints_shape[0], 1);
    const int num_keypoints = keypoints_shape[1];
    THROW_CHECK_EQ(keypoints_shape[2], 2);
    // TODO(jsch): Change to float for ALIKED.
    const int64_t* keypoints_data = reinterpret_cast<const int64_t*>(
        output_tensors[0].GetTensorData<void>());
    keypoints->resize(num_keypoints);
    for (int i = 0; i < keypoints->size(); ++i) {
      (*keypoints)[i].x = keypoints_data[2 * i + 0];
      (*keypoints)[i].y = keypoints_data[2 * i + 1];
    }

    const std::vector<int64_t> descriptors_shape =
        output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(descriptors_shape.size(), 3);
    THROW_CHECK_EQ(descriptors_shape[0], 1);
    THROW_CHECK_EQ(descriptors_shape[1], num_keypoints);
    THROW_CHECK_EQ(descriptors_shape[2], 256);
    const int descriptors_dim = descriptors_shape[2];
    const float* descriptors_data =
        reinterpret_cast<const float*>(output_tensors[2].GetTensorData<void>());
    descriptors->resize(keypoints->size(), descriptors_dim * sizeof(float));
    std::memcpy(descriptors->data(),
                descriptors_data,
                num_keypoints * descriptors_dim * sizeof(float));

    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  ONNXModel model_;
};

class ALIKEDFeatureMatcher : public FeatureMatcher {
 public:
  explicit ALIKEDFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options),
        model_(options.aliked->model_path, options.num_threads) {
    THROW_CHECK(options.Check());
    THROW_CHECK_EQ(model_.input_shapes.size(), 4);
    THROW_CHECK_EQ(std::string_view(model_.input_names[0]), "kpts0");
    THROW_CHECK(model_.input_shapes[0] == std::vector<int64_t>({1, -1, 2}));
    THROW_CHECK_EQ(std::string_view(model_.input_names[1]), "kpts1");
    THROW_CHECK(model_.input_shapes[1] == std::vector<int64_t>({1, -1, 2}));
    THROW_CHECK_EQ(std::string_view(model_.input_names[2]), "desc0");
    THROW_CHECK(model_.input_shapes[2] == std::vector<int64_t>({1, -1, 256}));
    THROW_CHECK_EQ(std::string_view(model_.input_names[3]), "desc1");
    THROW_CHECK(model_.input_shapes[3] == std::vector<int64_t>({1, -1, 256}));
    THROW_CHECK_EQ(model_.output_shapes.size(), 4);
    THROW_CHECK_EQ(std::string_view(model_.output_names[0]), "matches0");
    THROW_CHECK(model_.output_shapes[0] == std::vector<int64_t>({-1, -1}));
    THROW_CHECK_EQ(std::string_view(model_.output_names[1]), "matches1");
    THROW_CHECK(model_.output_shapes[1] == std::vector<int64_t>({-1, -1}));
    THROW_CHECK_EQ(std::string_view(model_.output_names[2]), "mscores0");
    THROW_CHECK(model_.output_shapes[2] == std::vector<int64_t>({-1, -1}));
    THROW_CHECK_EQ(std::string_view(model_.output_names[3]), "mscores1");
    THROW_CHECK(model_.output_shapes[3] == std::vector<int64_t>({-1, -1}));
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

    auto features1 = FeaturesFromImage(
        image1, model_.input_shapes[0], model_.input_shapes[2]);
    auto features2 = FeaturesFromImage(
        image2, model_.input_shapes[1], model_.input_shapes[3]);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(features1.keypoints_tensor));
    input_tensors.emplace_back(std::move(features2.keypoints_tensor));
    input_tensors.emplace_back(std::move(features1.descriptors_tensor));
    input_tensors.emplace_back(std::move(features2.descriptors_tensor));

    LOG(INFO) << "Running model...";
    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);

    const Ort::Value& matches0 = output_tensors[0];

    const std::vector<int64_t> matches_shape =
        matches0.GetTensorTypeAndShapeInfo().GetShape();
    const int num_matches = matches_shape[1];
    THROW_CHECK_EQ(num_matches, num_keypoints1);
    const int64_t* matches0_data =
        reinterpret_cast<const int64_t*>(matches0.GetTensorData<void>());
    matches->reserve(num_keypoints1);
    for (int i = 0; i < num_keypoints1; ++i) {
      const int64_t j = matches0_data[i];
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
  struct Features {
    std::vector<float> keypoints_data;
    Ort::Value keypoints_tensor;
    std::vector<float> descriptors_data;
    Ort::Value descriptors_tensor;
  };

  Features FeaturesFromImage(const Image& image,
                             std::vector<int64_t>& keypoints_shape,
                             std::vector<int64_t>& descriptors_shape) {
    THROW_CHECK_NE(image.image_id, kInvalidImageId);
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK_NOTNULL(image.keypoints);
    THROW_CHECK_EQ(image.descriptors->rows(), image.keypoints->size());

    const int num_keypoints = image.keypoints->size();
    THROW_CHECK_EQ(image.descriptors->cols() % sizeof(float), 0);
    const int descriptor_dim = image.descriptors->cols() / sizeof(float);

    Features features;

    THROW_CHECK_EQ(keypoints_shape.size(), 3);
    keypoints_shape[0] = 1;
    keypoints_shape[1] = num_keypoints;
    keypoints_shape[2] = 2;
    features.keypoints_data.resize(num_keypoints * 2);
    for (int i = 0; i < num_keypoints; ++i) {
      features.keypoints_data[2 * i + 0] = (*image.keypoints)[i].x;
      features.keypoints_data[2 * i + 1] = (*image.keypoints)[i].y;
    }

    features.keypoints_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        features.keypoints_data.data(),
        features.keypoints_data.size(),
        keypoints_shape.data(),
        keypoints_shape.size());

    THROW_CHECK_EQ(descriptors_shape.size(), 3);
    descriptors_shape[0] = 1;
    descriptors_shape[1] = num_keypoints;
    descriptors_shape[2] = descriptor_dim;
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

bool ALIKEDExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_image_size, 0);
  // CHECK_OPTION_GT(max_num_features, 0);
  // CHECK_OPTION_GT(score_threshold, 0);
  // CHECK_OPTION_GE(top_k, -1);
  return true;
}

std::unique_ptr<FeatureExtractor> CreateALIKEDFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<ALIKEDFeatureExtractor>(options);
#else
  throw std::runtime_error("ALIKED feature extraction requires torch support.");
#endif
}

bool ALIKEDMatchingOptions::Check() const {
  // CHECK_OPTION_GE(min_similarity, -1);
  // CHECK_OPTION_LE(min_similarity, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateALIKEDFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<ALIKEDFeatureMatcher>(options);
#else
  throw std::runtime_error("ALIKED feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
