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

#include "colmap/feature/loma.h"

#include "colmap/feature/onnx_utils.h"
#include "colmap/feature/utils.h"
#include "colmap/geometry/pose_prior.h"

#include <cstring>
#include <memory>

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

std::vector<float> BitmapToInputTensor(const Bitmap& bitmap) {
  THROW_CHECK(bitmap.IsRGB());
  const int width = bitmap.Width();
  const int height = bitmap.Height();
  const int pitch = bitmap.Pitch();
  const int num_pixels = width * height;

  std::vector<float> input(static_cast<size_t>(3) * num_pixels);
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

std::vector<float> ResizeToInputTensor(const Bitmap& bitmap,
                                       int target_width,
                                       int target_height) {
  THROW_CHECK(bitmap.IsRGB());
  Bitmap resized = bitmap.Clone();
  resized.Rescale(target_width, target_height);

  std::vector<float> input(static_cast<size_t>(3) * target_height * target_width);
  const int num_pixels = target_width * target_height;
  const std::vector<uint8_t>& data = resized.RowMajorData();
  const int pitch = resized.Pitch();
  for (int y = 0; y < target_height; ++y) {
    for (int x = 0; x < target_width; ++x) {
      for (int c = 0; c < 3; ++c) {
        constexpr float kImageNormalization = 1.0f / 255.0f;
        input[c * num_pixels + y * target_width + x] =
            kImageNormalization * data[y * pitch + 3 * x + c];
      }
    }
  }
  return input;
}

Ort::Value MakeTensor(std::vector<float>& data,
                      const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                 OrtMemType::OrtMemTypeCPU),
      data.data(),
      data.size(),
      shape.data(),
      shape.size());
}

Ort::Value MakeInt64Tensor(std::vector<int64_t>& data,
                           const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor<int64_t>(
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                 OrtMemType::OrtMemTypeCPU),
      data.data(),
      data.size(),
      shape.data(),
      shape.size());
}

class LomaFeatureExtractor : public FeatureExtractor {
 public:
  explicit LomaFeatureExtractor(const FeatureExtractionOptions& options)
      : options_(options),
        detector_(options.loma->detector_model_path,
                  options.num_threads,
                  options.use_gpu,
                  options.gpu_index),
        descriptor_(options.loma->descriptor_model_path,
                    options.num_threads,
                    options.use_gpu,
                    options.gpu_index) {
    THROW_CHECK(options.Check());

    // Detector: image [1, 3, H, W], num_keypoints [1] (int64, runtime input --
    // see deployment/export_onnx.py's make_detector_dynamic_k() in the LoMa
    // repo, matching ALIKED's max_keypoints convention) -> keypoints [1, N, 2]
    // (normalized [-1, 1]), keypoint_probs [1, N]. Shared across all variants.
    THROW_CHECK_EQ(detector_.input_shapes().size(), 2);
    ThrowCheckONNXNode(detector_.input_names()[0],
                       "image",
                       detector_.input_shapes()[0],
                       {1, 3, -1, -1});
    ThrowCheckONNXNode(detector_.input_names()[1],
                       "num_keypoints",
                       detector_.input_shapes()[1],
                       {1});
    THROW_CHECK_EQ(detector_.output_shapes().size(), 2);
    ThrowCheckONNXNode(detector_.output_names()[0],
                       "keypoints",
                       detector_.output_shapes()[0],
                       {1, -1, 2});
    ThrowCheckONNXNode(detector_.output_names()[1],
                       "keypoint_probs",
                       detector_.output_shapes()[1],
                       {1, -1});

    // Descriptor: image [1, 3, S, S], keypoints [1, N, 2] -> descriptions
    // [1, N, D]. Variant-specific (dim differs between DeDoDe-B / DeDoDe-G).
    THROW_CHECK_EQ(descriptor_.input_shapes().size(), 2);
    THROW_CHECK_EQ(descriptor_.output_shapes().size(), 1);
    const auto& desc_out_shape = descriptor_.output_shapes()[0];
    THROW_CHECK_EQ(desc_out_shape.size(), 3);
    descriptor_dim_ = static_cast<int>(desc_out_shape[2]);
    THROW_CHECK_GT(descriptor_dim_, 0);
    VLOG(2) << "LoMa descriptor dimension: " << descriptor_dim_;
  }

  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override {
    THROW_CHECK_NOTNULL(keypoints);
    THROW_CHECK_NOTNULL(descriptors);
    THROW_CHECK(bitmap.IsRGB());

    const int width = bitmap.Width();
    const int height = bitmap.Height();
    const int desc_size = options_.loma->descriptor_size;
    const int64_t num_keypoints_requested = options_.loma->max_num_features;

    std::vector<float> det_input = BitmapToInputTensor(bitmap);
    std::vector<int64_t> det_shape{1, 3, height, width};
    std::vector<int64_t> num_kpts_data{num_keypoints_requested};
    std::vector<int64_t> num_kpts_shape{1};

    std::vector<Ort::Value> det_inputs_unordered;
    det_inputs_unordered.push_back(MakeTensor(det_input, det_shape));
    det_inputs_unordered.push_back(MakeInt64Tensor(num_kpts_data, num_kpts_shape));
    std::vector<Ort::Value> det_inputs;
    for (const char* name : detector_.input_names()) {
      det_inputs.push_back(std::move(
          std::string(name) == "image" ? det_inputs_unordered[0]
                                       : det_inputs_unordered[1]));
    }
    const std::vector<Ort::Value> det_outputs = detector_.Run(det_inputs);
    THROW_CHECK_EQ(det_outputs.size(), 2);

    const int64_t num_kpts =
        det_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    THROW_CHECK_EQ(num_kpts, num_keypoints_requested);
    const float* kpts_norm = det_outputs[0].GetTensorData<float>();
    const float* scores = det_outputs[1].GetTensorData<float>();

    std::vector<float> kpts_norm_copy(kpts_norm, kpts_norm + num_kpts * 2);

    // --- descriptor: still fixed-shape, resized (not padded) to desc_size.
    std::vector<float> desc_input =
        ResizeToInputTensor(bitmap, desc_size, desc_size);
    std::vector<int64_t> desc_img_shape{1, 3, desc_size, desc_size};
    std::vector<int64_t> desc_kpt_shape{1, num_kpts, 2};

    std::vector<Ort::Value> desc_inputs_unordered;
    desc_inputs_unordered.push_back(MakeTensor(desc_input, desc_img_shape));
    desc_inputs_unordered.push_back(
        MakeTensor(kpts_norm_copy, desc_kpt_shape));
    std::vector<Ort::Value> desc_inputs;
    for (const char* name : descriptor_.input_names()) {
      desc_inputs.push_back(std::move(
          std::string(name) == "image" ? desc_inputs_unordered[0]
                                       : desc_inputs_unordered[1]));
    }
    const std::vector<Ort::Value> desc_outputs = descriptor_.Run(desc_inputs);
    THROW_CHECK_EQ(desc_outputs.size(), 1);
    const float* desc_data = desc_outputs[0].GetTensorData<float>();

    // Convert normalized [-1, 1] keypoints to pixel coordinates in the
    // original image, and filter by min_score, same convention as
    // aliked.cc (COLMAP: top-left pixel corner = (0, 0)).
    const float min_score = static_cast<float>(options_.loma->min_score);
    struct ValidKeypoint {
      float x, y;
      int64_t index;
    };
    std::vector<ValidKeypoint> valid;
    valid.reserve(num_kpts);
    for (int64_t i = 0; i < num_kpts; ++i) {
      if (scores[i] < min_score) continue;
      const float nx = kpts_norm[2 * i + 0];
      const float ny = kpts_norm[2 * i + 1];
      const float px = 0.5f * (nx + 1.0f) * width;
      const float py = 0.5f * (ny + 1.0f) * height;
      valid.push_back({px, py, i});
    }

    const int num_valid = static_cast<int>(valid.size());
    keypoints->resize(num_valid);
    descriptors->type = options_.type;
    descriptors->data.resize(num_valid, descriptor_dim_ * sizeof(float));
    for (int j = 0; j < num_valid; ++j) {
      (*keypoints)[j].x = valid[j].x;
      (*keypoints)[j].y = valid[j].y;
      std::memcpy(descriptors->data.data() + j * descriptor_dim_ * sizeof(float),
                  desc_data + valid[j].index * descriptor_dim_,
                  descriptor_dim_ * sizeof(float));
    }
    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  ONNXModel detector_;
  ONNXModel descriptor_;
  int descriptor_dim_ = 0;
};

class LomaFeatureMatcher : public FeatureMatcher {
 public:
  explicit LomaFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(options),
        loma_options_(*options.loma),
        model_(options.loma->model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options.Check());
    THROW_CHECK_EQ(model_.input_shapes().size(), 4);
    THROW_CHECK_GE(model_.output_shapes().size(), 2);
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();

    Features f1 = FeaturesFromImage(image1);
    Features f2 = FeaturesFromImage(image2);
    const int num_keypoints1 = static_cast<int>(f1.kpts.size() / 2);
    const int num_keypoints2 = static_cast<int>(f2.kpts.size() / 2);
    if (num_keypoints1 == 0 || num_keypoints2 == 0) return;

    std::vector<int64_t> k0s{1, num_keypoints1, 2};
    std::vector<int64_t> k1s{1, num_keypoints2, 2};
    std::vector<int64_t> d0s{1, num_keypoints1, f1.desc_dim};
    std::vector<int64_t> d1s{1, num_keypoints2, f2.desc_dim};

    std::vector<Ort::Value> inputs;
    for (const char* name_c : model_.input_names()) {
      const std::string name(name_c);
      if (name == "kpts0") inputs.push_back(MakeTensor(f1.kpts, k0s));
      else if (name == "kpts1") inputs.push_back(MakeTensor(f2.kpts, k1s));
      else if (name == "desc0") inputs.push_back(MakeTensor(f1.desc, d0s));
      else if (name == "desc1") inputs.push_back(MakeTensor(f2.desc, d1s));
      else LOG(FATAL_THROW) << "Unexpected LoMa matcher input: " << name;
    }

    const std::vector<Ort::Value> outputs = model_.Run(inputs);
    THROW_CHECK_GE(outputs.size(), 2);  // m0, m1, mscores0, mscores1 -- see ctor.

    int m0_idx = -1, mscores0_idx = -1;
    const auto& names = model_.output_names();
    for (size_t i = 0; i < names.size(); ++i) {
      if (std::string(names[i]) == "m0") m0_idx = static_cast<int>(i);
      if (std::string(names[i]) == "mscores0") mscores0_idx = static_cast<int>(i);
    }
    THROW_CHECK_GE(m0_idx, 0);
    THROW_CHECK_GE(mscores0_idx, 0);

    const auto m0_shape = outputs[m0_idx].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(m0_shape.size(), 2);
    THROW_CHECK_EQ(m0_shape[0], 1);
    THROW_CHECK_EQ(m0_shape[1], num_keypoints1);

    const int64_t* m0 = outputs[m0_idx].GetTensorData<int64_t>();
    const float* mscores0 = outputs[mscores0_idx].GetTensorData<float>();
    const float min_score = static_cast<float>(loma_options_.min_score);

    for (int i = 0; i < num_keypoints1; ++i) {
      const int64_t j = m0[i];
      if (j < 0) continue;  // -1 = filtered out by the matcher itself.
      if (mscores0[i] < min_score) continue;
      THROW_CHECK_LT(j, num_keypoints2);
      matches->emplace_back(static_cast<point2D_t>(i),
                            static_cast<point2D_t>(j));
    }
  }

  void MatchGuided(double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    LOG(FATAL_THROW) << "Guided matching not supported for LoMa.";
  }

 private:
  struct Features {
    std::vector<float> kpts;  // normalized [-1, 1], [1, N, 2] flattened
    std::vector<float> desc;  // [1, N, D] flattened
    int desc_dim = 0;
  };

  // Converts COLMAP pixel-space keypoints (top-left pixel corner = (0, 0)) back to LoMa's normalized [-1, 1] convention
  Features FeaturesFromImage(const Image& image) const {
    THROW_CHECK_NOTNULL(image.keypoints);
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK_NOTNULL(image.camera);
    THROW_CHECK(image.descriptors->type == FeatureExtractorType::LOMA_B)
        << "LoMa matcher got unsupported feature type: "
        << FeatureExtractorTypeToString(image.descriptors->type);
    THROW_CHECK_EQ(image.descriptors->data.cols() % sizeof(float), 0);

    const int num_keypoints = image.descriptors->data.rows();
    const int desc_dim = image.descriptors->data.cols() / sizeof(float);
    const float width = static_cast<float>(image.camera->width);
    const float height = static_cast<float>(image.camera->height);

    Features f;
    f.desc_dim = desc_dim;
    f.kpts.resize(num_keypoints * 2);
    for (int i = 0; i < num_keypoints; ++i) {
      const FeatureKeypoint& kp = (*image.keypoints)[i];
      f.kpts[2 * i + 0] = 2.0f * kp.x / width - 1.0f;
      f.kpts[2 * i + 1] = 2.0f * kp.y / height - 1.0f;
    }
    f.desc.resize(num_keypoints * desc_dim);
    std::memcpy(f.desc.data(),
               reinterpret_cast<const void*>(image.descriptors->data.data()),
               image.descriptors->data.size());
    return f;
  }

  const FeatureMatchingOptions options_;
  const LomaMatchingOptions loma_options_;
  ONNXModel model_;
};

#endif

}  // namespace

bool LomaExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_LE(min_score, 1);
  CHECK_OPTION_GT(descriptor_size, 0);
  return true;
}

std::unique_ptr<FeatureExtractor> CreateLomaFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<LomaFeatureExtractor>(options);
#else
  throw std::runtime_error("LoMa feature extraction requires ONNX support.");
#endif
}

bool LomaMatchingOptions::Check() const {
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_LE(min_score, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateLomaFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<LomaFeatureMatcher>(options);
#else
  throw std::runtime_error("LoMa feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
