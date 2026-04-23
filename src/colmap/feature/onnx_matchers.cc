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

#include "colmap/feature/onnx_matchers.h"

#include "colmap/feature/onnx_utils.h"
#include "colmap/feature/utils.h"
#include "colmap/geometry/pose_prior.h"

#include <algorithm>
#include <memory>

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

class BruteForceONNXFeatureMatcher : public FeatureMatcher {
 public:
  explicit BruteForceONNXFeatureMatcher(
      const FeatureMatchingOptions& options,
      const BruteForceONNXMatchingOptions& brute_force_options)
      : options_(options),
        brute_force_options_(brute_force_options),
        model_(brute_force_options.model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options.Check());
    THROW_CHECK_EQ(model_.input_shapes().size(), 5);
    ThrowCheckONNXNode(
        model_.input_names()[0], "descs1", model_.input_shapes()[0], {-1, -1});
    ThrowCheckONNXNode(
        model_.input_names()[1], "descs2", model_.input_shapes()[1], {-1, -1});
    ThrowCheckONNXNode(
        model_.input_names()[2], "min_cossim", model_.input_shapes()[2], {});
    ThrowCheckONNXNode(
        model_.input_names()[3], "max_ratio", model_.input_shapes()[3], {});
    ThrowCheckONNXNode(
        model_.input_names()[4], "cross_check", model_.input_shapes()[4], {});
    THROW_CHECK_EQ(model_.output_shapes().size(), 3);
    ThrowCheckONNXNode(
        model_.output_names()[0], "idx0", model_.output_shapes()[0], {-1});
    ThrowCheckONNXNode(
        model_.output_names()[1], "idx1", model_.output_shapes()[1], {-1});
    ThrowCheckONNXNode(
        model_.output_names()[2], "scores", model_.output_shapes()[2], {-1});
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();

    const int num_keypoints1 = image1.descriptors->data.rows();
    const int num_keypoints2 = image2.descriptors->data.rows();
    // Model requires at least 2 descriptors in each set for ratio test.
    if (num_keypoints1 < 2 || num_keypoints2 < 2) {
      return;
    }

    // Cache features if image changed. Swap cached features when possible
    // to avoid redundant copies (e.g., matching (A, B) then (B, C)).
    if (prev_features1_.image_id == kInvalidImageId ||
        prev_features1_.image_id != image1.image_id) {
      if (image1.image_id != kInvalidImageId &&
          prev_features2_.image_id == image1.image_id) {
        std::swap(prev_features1_, prev_features2_);
      } else {
        prev_features1_ = FeaturesFromImage(image1);
      }
    }
    if (prev_features2_.image_id == kInvalidImageId ||
        prev_features2_.image_id != image2.image_id) {
      if (image2.image_id != kInvalidImageId &&
          prev_features1_.image_id == image2.image_id) {
        // This shouldn't happen, as it means we are self-matching an image.
        prev_features2_ = prev_features1_;
      } else {
        prev_features2_ = FeaturesFromImage(image2);
      }
    }

    // Create tensors from cached data (tensors must be recreated each call
    // since they reference the underlying data and get consumed by Run()).
    auto desc1_tensor = CreateDescriptorTensor(prev_features1_);
    auto desc2_tensor = CreateDescriptorTensor(prev_features2_);

    float min_cossim = static_cast<float>(brute_force_options_.min_cossim);
    const std::vector<int64_t> scalar_shape = {};
    auto min_cossim_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        &min_cossim,
        1,
        scalar_shape.data(),
        scalar_shape.size());

    float max_ratio = static_cast<float>(brute_force_options_.max_ratio);
    auto max_ratio_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        &max_ratio,
        1,
        scalar_shape.data(),
        scalar_shape.size());

    int64_t cross_check = brute_force_options_.cross_check ? 1 : 0;
    auto cross_check_tensor = Ort::Value::CreateTensor<int64_t>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        &cross_check,
        1,
        scalar_shape.data(),
        scalar_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(std::move(desc1_tensor));
    input_tensors.emplace_back(std::move(desc2_tensor));
    input_tensors.emplace_back(std::move(min_cossim_tensor));
    input_tensors.emplace_back(std::move(max_ratio_tensor));
    input_tensors.emplace_back(std::move(cross_check_tensor));

    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);
    THROW_CHECK_EQ(output_tensors.size(), 3);

    // Get num_matches from shape of idx0 output
    const std::vector<int64_t> idx0_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(idx0_shape.size(), 1);
    const int64_t num_matches = idx0_shape[0];

    // Ensure idx1 has the same 1D shape and length as idx0
    const std::vector<int64_t> idx1_shape =
        output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(idx1_shape.size(), 1);
    THROW_CHECK_EQ(idx1_shape[0], num_matches);

    if (num_matches == 0) {
      return;
    }

    const int64_t* idx0_data = output_tensors[0].GetTensorData<int64_t>();
    const int64_t* idx1_data = output_tensors[1].GetTensorData<int64_t>();

    matches->resize(num_matches);
    for (int64_t i = 0; i < num_matches; ++i) {
      FeatureMatch& match = (*matches)[i];
      match.point2D_idx1 = idx0_data[i];
      match.point2D_idx2 = idx1_data[i];
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
    LOG(FATAL_THROW) << "Guided matching not supported for ALIKED.";
  }

 private:
  struct Features {
    image_t image_id = kInvalidImageId;
    std::vector<float> descriptors_data;
    std::vector<int64_t> descriptors_shape;
  };

  Features FeaturesFromImage(const Image& image) {
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK(image.descriptors->type ==
                    FeatureExtractorType::ALIKED_N16ROT ||
                image.descriptors->type == FeatureExtractorType::ALIKED_N32)
        << "Unsupported feature type: "
        << FeatureExtractorTypeToString(image.descriptors->type);
    THROW_CHECK_EQ(image.descriptors->data.cols() % sizeof(float), 0);

    const int num_keypoints = image.descriptors->data.rows();
    const int descriptor_dim = image.descriptors->data.cols() / sizeof(float);
    THROW_CHECK_GT(descriptor_dim, 0);

    Features features;
    features.image_id = image.image_id;
    features.descriptors_shape = {num_keypoints, descriptor_dim};
    features.descriptors_data.resize(num_keypoints * descriptor_dim);
    THROW_CHECK_EQ(image.descriptors->data.size(),
                   features.descriptors_data.size() * sizeof(float));
    std::memcpy(features.descriptors_data.data(),
                reinterpret_cast<const void*>(image.descriptors->data.data()),
                image.descriptors->data.size());

    return features;
  }

  Ort::Value CreateDescriptorTensor(Features& features) {
    return Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        features.descriptors_data.data(),
        features.descriptors_data.size(),
        features.descriptors_shape.data(),
        features.descriptors_shape.size());
  }

  const FeatureMatchingOptions options_;
  const BruteForceONNXMatchingOptions brute_force_options_;
  ONNXModel model_;

  // Cached features for avoiding redundant data copies.
  Features prev_features1_;
  Features prev_features2_;
};

class LightGlueONNXFeatureMatcher : public FeatureMatcher {
 public:
  explicit LightGlueONNXFeatureMatcher(
      const FeatureMatchingOptions& options,
      const LightGlueONNXMatchingOptions& lightglue_options)
      : options_(options),
        lightglue_options_(lightglue_options),
        model_(lightglue_options.model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options.Check());

    const size_t num_inputs = model_.input_shapes().size();
    if (num_inputs == 6) {
      has_scale_ori_ = false;
    } else if (num_inputs == 10) {
      has_scale_ori_ = true;
    } else {
      LOG(FATAL_THROW) << "LightGlue ONNX model must have 6 or 10 inputs, "
                       << "got " << num_inputs;
    }

    ThrowCheckONNXNode(model_.input_names()[0],
                       "kpts0",
                       model_.input_shapes()[0],
                       {-1, -1, -1});
    ThrowCheckONNXNode(model_.input_names()[1],
                       "kpts1",
                       model_.input_shapes()[1],
                       {-1, -1, -1});
    ThrowCheckONNXNode(model_.input_names()[2],
                       "desc0",
                       model_.input_shapes()[2],
                       {-1, -1, -1});
    ThrowCheckONNXNode(model_.input_names()[3],
                       "desc1",
                       model_.input_shapes()[3],
                       {-1, -1, -1});
    ThrowCheckONNXNode(model_.input_names()[4],
                       "image_size0",
                       model_.input_shapes()[4],
                       {-1, -1});
    ThrowCheckONNXNode(model_.input_names()[5],
                       "image_size1",
                       model_.input_shapes()[5],
                       {-1, -1});

    if (has_scale_ori_) {
      ThrowCheckONNXNode(model_.input_names()[6],
                         "scales0",
                         model_.input_shapes()[6],
                         {-1, -1});
      ThrowCheckONNXNode(model_.input_names()[7],
                         "scales1",
                         model_.input_shapes()[7],
                         {-1, -1});
      ThrowCheckONNXNode(
          model_.input_names()[8], "oris0", model_.input_shapes()[8], {-1, -1});
      ThrowCheckONNXNode(
          model_.input_names()[9], "oris1", model_.input_shapes()[9], {-1, -1});
    }

    THROW_CHECK_EQ(model_.output_shapes().size(), 2);
    ThrowCheckONNXNode(
        model_.output_names()[0], "matches0", model_.output_shapes()[0], {-1});
    ThrowCheckONNXNode(
        model_.output_names()[1], "mscores0", model_.output_shapes()[1], {-1});
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();

    const int num_keypoints1 = image1.descriptors->data.rows();
    const int num_keypoints2 = image2.descriptors->data.rows();
    if (num_keypoints1 < 1 || num_keypoints2 < 1) {
      return;
    }

    // Cache features with swap optimization (identical to ALIKED pattern).
    if (prev_features1_.image_id == kInvalidImageId ||
        prev_features1_.image_id != image1.image_id) {
      if (image1.image_id != kInvalidImageId &&
          prev_features2_.image_id == image1.image_id) {
        std::swap(prev_features1_, prev_features2_);
      } else {
        prev_features1_ = FeaturesFromImage(image1);
      }
    }
    if (prev_features2_.image_id == kInvalidImageId ||
        prev_features2_.image_id != image2.image_id) {
      if (image2.image_id != kInvalidImageId &&
          prev_features1_.image_id == image2.image_id) {
        prev_features2_ = prev_features1_;
      } else {
        prev_features2_ = FeaturesFromImage(image2);
      }
    }

    // Create input tensors.
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(has_scale_ori_ ? 10 : 6);

    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        prev_features1_.keypoints_data.data(),
        prev_features1_.keypoints_data.size(),
        prev_features1_.keypoints_shape.data(),
        prev_features1_.keypoints_shape.size()));

    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        prev_features2_.keypoints_data.data(),
        prev_features2_.keypoints_data.size(),
        prev_features2_.keypoints_shape.data(),
        prev_features2_.keypoints_shape.size()));

    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        prev_features1_.descriptors_data.data(),
        prev_features1_.descriptors_data.size(),
        prev_features1_.descriptors_shape.data(),
        prev_features1_.descriptors_shape.size()));

    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        prev_features2_.descriptors_data.data(),
        prev_features2_.descriptors_data.size(),
        prev_features2_.descriptors_shape.data(),
        prev_features2_.descriptors_shape.size()));

    std::vector<int64_t> image_size_shape = {1, 2};
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        prev_features1_.image_size,
        2,
        image_size_shape.data(),
        image_size_shape.size()));

    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        prev_features2_.image_size,
        2,
        image_size_shape.data(),
        image_size_shape.size()));

    if (has_scale_ori_) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                     OrtMemType::OrtMemTypeCPU),
          prev_features1_.scales_data.data(),
          prev_features1_.scales_data.size(),
          prev_features1_.scales_shape.data(),
          prev_features1_.scales_shape.size()));

      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                     OrtMemType::OrtMemTypeCPU),
          prev_features2_.scales_data.data(),
          prev_features2_.scales_data.size(),
          prev_features2_.scales_shape.data(),
          prev_features2_.scales_shape.size()));

      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                     OrtMemType::OrtMemTypeCPU),
          prev_features1_.orientations_data.data(),
          prev_features1_.orientations_data.size(),
          prev_features1_.orientations_shape.data(),
          prev_features1_.orientations_shape.size()));

      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                     OrtMemType::OrtMemTypeCPU),
          prev_features2_.orientations_data.data(),
          prev_features2_.orientations_data.size(),
          prev_features2_.orientations_shape.data(),
          prev_features2_.orientations_shape.size()));
    }

    // Run model inference.
    const std::vector<Ort::Value> output_tensors = model_.Run(input_tensors);
    THROW_CHECK_EQ(output_tensors.size(), 2);

    // Parse matches0 shape: [M].
    const auto matches0_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> matches0_shape = matches0_info.GetShape();
    THROW_CHECK_EQ(matches0_shape.size(), 1);
    const int64_t num_kpts = matches0_shape[0];
    THROW_CHECK_EQ(num_kpts, num_keypoints1);

    // Parse mscores0 shape: [M].
    const auto mscores0_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> mscores0_shape = mscores0_info.GetShape();
    THROW_CHECK_EQ(mscores0_shape.size(), 1);
    THROW_CHECK_EQ(mscores0_shape[0], num_kpts);

    const float min_score = static_cast<float>(lightglue_options_.min_score);
    const float* mscores0_data = output_tensors[1].GetTensorData<float>();

    // Handle both int64 and float output types for matches0.
    const auto matches0_type = matches0_info.GetElementType();
    if (matches0_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* matches0_data = output_tensors[0].GetTensorData<int64_t>();
      for (int64_t i = 0; i < num_kpts; ++i) {
        if (matches0_data[i] >= 0 && mscores0_data[i] >= min_score) {
          THROW_CHECK_LT(matches0_data[i], num_keypoints2);
          matches->emplace_back(static_cast<point2D_t>(i),
                                static_cast<point2D_t>(matches0_data[i]));
        }
      }
    } else if (matches0_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const float* matches0_data = output_tensors[0].GetTensorData<float>();
      for (int64_t i = 0; i < num_kpts; ++i) {
        const int64_t match_idx = static_cast<int64_t>(matches0_data[i]);
        if (match_idx >= 0 && mscores0_data[i] >= min_score) {
          THROW_CHECK_LT(match_idx, num_keypoints2);
          matches->emplace_back(static_cast<point2D_t>(i),
                                static_cast<point2D_t>(match_idx));
        }
      }
    } else {
      LOG(FATAL_THROW) << "Unexpected matches0 output type: " << matches0_type;
    }
  }

  void MatchGuided(double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    LOG(FATAL_THROW) << "Guided matching not supported for LightGlue.";
  }

 private:
  struct CachedFeatures {
    image_t image_id = kInvalidImageId;
    std::vector<float> keypoints_data;
    std::vector<int64_t> keypoints_shape;
    std::vector<float> descriptors_data;
    std::vector<int64_t> descriptors_shape;
    float image_size[2];
    std::vector<float> scales_data;
    std::vector<int64_t> scales_shape;
    std::vector<float> orientations_data;
    std::vector<int64_t> orientations_shape;
  };

  CachedFeatures FeaturesFromImage(const Image& image) {
    THROW_CHECK_NOTNULL(image.keypoints);
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK_NOTNULL(image.camera);

    const bool is_sift = image.descriptors->type == FeatureExtractorType::SIFT;
    const bool is_aliked =
        image.descriptors->type == FeatureExtractorType::ALIKED_N16ROT ||
        image.descriptors->type == FeatureExtractorType::ALIKED_N32;
    THROW_CHECK(is_sift || is_aliked)
        << "Unsupported feature type: "
        << FeatureExtractorTypeToString(image.descriptors->type);

    if ((options_.type == FeatureMatcherType::SIFT_LIGHTGLUE && !is_sift) ||
        (options_.type == FeatureMatcherType::ALIKED_LIGHTGLUE && !is_aliked)) {
      LOG(FATAL_THROW) << FeatureMatcherTypeToString(options_.type)
                       << " feature matcher got unsupported feature type: "
                       << FeatureExtractorTypeToString(image.descriptors->type);
    }

    const int num_keypoints = image.descriptors->data.rows();
    THROW_CHECK_EQ(static_cast<int>(image.keypoints->size()), num_keypoints);

    CachedFeatures features;
    features.image_id = image.image_id;

    const int rot90 =
        (image.pose_prior != nullptr && image.pose_prior->HasGravity())
            ? ComputeRot90FromGravity(image.pose_prior->gravity)
            : 0;
    const int image_width = image.camera->width;
    const int image_height = image.camera->height;

    std::vector<FeatureKeypoint> rotated_keypoints;
    const FeatureKeypoints* keypoints_to_use = image.keypoints.get();
    if (rot90 != 0) {
      rotated_keypoints = *image.keypoints;
      for (auto& kp : rotated_keypoints) {
        kp.Rot90(rot90, image_width, image_height);
      }
      keypoints_to_use = &rotated_keypoints;
    }

    // Convert keypoints: COLMAP (origin at pixel corner, top-left center =
    // (0.5, 0.5)) to LightGlue (top-left center = (0, 0)).
    features.keypoints_shape = {1, num_keypoints, 2};
    features.keypoints_data.resize(num_keypoints * 2);
    for (int i = 0; i < num_keypoints; ++i) {
      const FeatureKeypoint& kp = (*keypoints_to_use)[i];
      features.keypoints_data[2 * i + 0] = kp.x - 0.5f;
      features.keypoints_data[2 * i + 1] = kp.y - 0.5f;
    }

    if (is_aliked) {
      // ALIKED descriptors: stored as float bytes, reinterpret directly.
      THROW_CHECK_EQ(image.descriptors->data.cols() % sizeof(float), 0);
      const int descriptor_dim = image.descriptors->data.cols() / sizeof(float);
      THROW_CHECK_GT(descriptor_dim, 0);

      features.descriptors_shape = {1, num_keypoints, descriptor_dim};
      features.descriptors_data.resize(num_keypoints * descriptor_dim);
      THROW_CHECK_EQ(image.descriptors->data.size(),
                     features.descriptors_data.size() * sizeof(float));
      std::memcpy(features.descriptors_data.data(),
                  reinterpret_cast<const void*>(image.descriptors->data.data()),
                  image.descriptors->data.size());
    } else {
      // SIFT descriptors: stored as uint8, cast to float32 and root-normalize.
      const int descriptor_dim = image.descriptors->data.cols();
      THROW_CHECK_GT(descriptor_dim, 0);

      // LightGlue was trained on root-normalized descriptors.
      FeatureDescriptorsFloat descriptors_float = image.descriptors->ToFloat();
      L1RootNormalizeFeatureDescriptors(&descriptors_float.data);

      features.descriptors_shape = {1, num_keypoints, descriptor_dim};
      features.descriptors_data.resize(num_keypoints * descriptor_dim);
      THROW_CHECK_EQ(descriptors_float.data.size(),
                     features.descriptors_data.size());
      std::memcpy(features.descriptors_data.data(),
                  reinterpret_cast<const void*>(descriptors_float.data.data()),
                  descriptors_float.data.size());

      // Extract scale and orientation from keypoints.
      features.scales_shape = {1, num_keypoints};
      features.scales_data.resize(num_keypoints);
      features.orientations_shape = {1, num_keypoints};
      features.orientations_data.resize(num_keypoints);
      for (int i = 0; i < num_keypoints; ++i) {
        const FeatureKeypoint& kp = (*keypoints_to_use)[i];
        features.scales_data[i] = kp.ComputeScale();
        // LightGlue was trained with radians.
        features.orientations_data[i] = kp.ComputeOrientation();
      }
    }

    // Image size as (width, height).
    const bool swap_dims = rot90 % 2;
    features.image_size[0] =
        static_cast<float>(swap_dims ? image_height : image_width);
    features.image_size[1] =
        static_cast<float>(swap_dims ? image_width : image_height);

    return features;
  }

  const FeatureMatchingOptions options_;
  const LightGlueONNXMatchingOptions lightglue_options_;
  ONNXModel model_;
  bool has_scale_ori_ = false;

  CachedFeatures prev_features1_;
  CachedFeatures prev_features2_;
};

#endif

}  // namespace

bool BruteForceONNXMatchingOptions::Check() const {
  CHECK_OPTION_GE(min_cossim, -1);
  CHECK_OPTION_LE(min_cossim, 1);
  CHECK_OPTION_GE(max_ratio, 0);
  CHECK_OPTION_LE(max_ratio, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateBruteForceONNXFeatureMatcher(
    const FeatureMatchingOptions& options,
    const BruteForceONNXMatchingOptions& brute_force_options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<BruteForceONNXFeatureMatcher>(options,
                                                        brute_force_options);
#else
  throw std::runtime_error("Brute-force ONNX matching requires ONNX support.");
#endif
}

bool LightGlueONNXMatchingOptions::Check() const {
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_LE(min_score, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateLightGlueONNXFeatureMatcher(
    const FeatureMatchingOptions& options,
    const LightGlueONNXMatchingOptions& lightglue_options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<LightGlueONNXFeatureMatcher>(options,
                                                       lightglue_options);
#else
  throw std::runtime_error("LightGlue feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
