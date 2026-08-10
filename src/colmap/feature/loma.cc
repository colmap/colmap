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

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

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

// Fast bilinear resample instead of Bitmap::Rescale()'s filtered resize --
// see LomaExtractionOptions::use_fast_resize for the speed/accuracy tradeoff.
// Reads directly from `bitmap` instead of cloning first.
std::vector<float> FastResizeToInputTensor(const Bitmap& bitmap,
                                           int target_width,
                                           int target_height) {
  const OIIO::ImageBuf src_buf(
      OIIO::ImageSpec(
          bitmap.Width(), bitmap.Height(), 3, OIIO::TypeDesc::UINT8),
      const_cast<uint8_t*>(bitmap.RowMajorData().data()));

  std::vector<uint8_t> resized_data(static_cast<size_t>(target_width) *
                                    target_height * 3);
  OIIO::ImageBuf dst_buf(
      OIIO::ImageSpec(target_width, target_height, 3, OIIO::TypeDesc::UINT8),
      resized_data.data());
  THROW_CHECK(
      OIIO::ImageBufAlgo::resample(dst_buf, src_buf, /*interpolate=*/true));

  std::vector<float> input(static_cast<size_t>(3) * target_height *
                           target_width);
  const int num_pixels = target_width * target_height;
  const int pitch = target_width * 3;
  for (int y = 0; y < target_height; ++y) {
    for (int x = 0; x < target_width; ++x) {
      for (int c = 0; c < 3; ++c) {
        constexpr float kImageNormalization = 1.0f / 255.0f;
        input[c * num_pixels + y * target_width + x] =
            kImageNormalization * resized_data[y * pitch + 3 * x + c];
      }
    }
  }
  return input;
}

// use_fast_resize=false fallback: Bitmap::Rescale()'s filtered resize.
std::vector<float> SlowResizeToInputTensor(const Bitmap& bitmap,
                                           int target_width,
                                           int target_height) {
  Bitmap resized = bitmap.Clone();
  resized.Rescale(target_width, target_height);

  std::vector<float> input(static_cast<size_t>(3) * target_height *
                           target_width);
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

std::vector<float> ResizeToInputTensor(const Bitmap& bitmap,
                                       int target_width,
                                       int target_height,
                                       bool use_fast_resize) {
  THROW_CHECK(bitmap.IsRGB());
  return use_fast_resize
             ? FastResizeToInputTensor(bitmap, target_width, target_height)
             : SlowResizeToInputTensor(bitmap, target_width, target_height);
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
  LomaFeatureExtractor(const FeatureExtractionOptions& options,
                       const std::string& descriptor_model_path,
                       bool probe_descriptor_support = false)
      : options_(options),
        detector_(options.loma->detector_model_path,
                  options.num_threads,
                  options.use_gpu,
                  options.gpu_index),
        descriptor_(descriptor_model_path,
                    options.num_threads,
                    options.use_gpu,
                    options.gpu_index,
                    probe_descriptor_support) {
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
    bool found_image_input = false;
    bool found_keypoints_input = false;
    for (size_t i = 0; i < descriptor_.input_names().size(); ++i) {
      const std::string_view name = descriptor_.input_names()[i];
      const auto& shape = descriptor_.input_shapes()[i];
      if (name == "image") {
        ThrowCheckONNXNode(name, "image", shape, {1, 3, -1, -1});
        THROW_CHECK_GT(shape[2], 0);
        THROW_CHECK_EQ(shape[2], shape[3]);
        descriptor_size_ = static_cast<int>(shape[2]);
        found_image_input = true;
      } else if (name == "keypoints") {
        ThrowCheckONNXNode(name, "keypoints", shape, {1, -1, 2});
        found_keypoints_input = true;
      } else {
        LOG(FATAL_THROW) << "Unexpected LoMa descriptor input: " << name;
      }
    }
    THROW_CHECK(found_image_input);
    THROW_CHECK(found_keypoints_input);
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
    const int64_t num_keypoints_requested = options_.loma->max_num_features;

    std::vector<float> det_input = BitmapToInputTensor(bitmap);
    std::vector<int64_t> det_shape{1, 3, height, width};
    std::vector<int64_t> num_kpts_data{num_keypoints_requested};
    std::vector<int64_t> num_kpts_shape{1};

    std::vector<Ort::Value> det_inputs_unordered;
    det_inputs_unordered.push_back(MakeTensor(det_input, det_shape));
    det_inputs_unordered.push_back(
        MakeInt64Tensor(num_kpts_data, num_kpts_shape));
    std::vector<Ort::Value> det_inputs;
    for (const char* name : detector_.input_names()) {
      det_inputs.push_back(std::move(std::string(name) == "image"
                                         ? det_inputs_unordered[0]
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

    // --- descriptor: still fixed-shape, resized (not padded) to the model's
    // input size.
    std::vector<float> desc_input =
        ResizeToInputTensor(bitmap,
                            descriptor_size_,
                            descriptor_size_,
                            options_.loma->use_fast_resize);
    std::vector<int64_t> desc_img_shape{
        1, 3, descriptor_size_, descriptor_size_};
    std::vector<int64_t> desc_kpt_shape{1, num_kpts, 2};

    std::vector<Ort::Value> desc_inputs_unordered;
    desc_inputs_unordered.push_back(MakeTensor(desc_input, desc_img_shape));
    desc_inputs_unordered.push_back(MakeTensor(kpts_norm_copy, desc_kpt_shape));
    std::vector<Ort::Value> desc_inputs;
    for (const char* name : descriptor_.input_names()) {
      desc_inputs.push_back(std::move(std::string(name) == "image"
                                          ? desc_inputs_unordered[0]
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
      std::memcpy(
          descriptors->data.data() + j * descriptor_dim_ * sizeof(float),
          desc_data + valid[j].index * descriptor_dim_,
          descriptor_dim_ * sizeof(float));
    }
    return true;
  }

 private:
  const FeatureExtractionOptions options_;
  ONNXModel detector_;
  ONNXModel descriptor_;
  int descriptor_size_ = 0;
  int descriptor_dim_ = 0;
};

class LomaFeatureMatcher : public FeatureMatcher {
 public:
  LomaFeatureMatcher(const FeatureMatchingOptions& options,
                     const std::string& model_path,
                     FeatureExtractorType expected_extractor_type,
                     bool probe_model_support = false)
      : options_(options),
        loma_options_(*options.loma),
        expected_extractor_type_(expected_extractor_type),
        model_(model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index,
               probe_model_support) {
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
      if (name == "kpts0")
        inputs.push_back(MakeTensor(f1.kpts, k0s));
      else if (name == "kpts1")
        inputs.push_back(MakeTensor(f2.kpts, k1s));
      else if (name == "desc0")
        inputs.push_back(MakeTensor(f1.desc, d0s));
      else if (name == "desc1")
        inputs.push_back(MakeTensor(f2.desc, d1s));
      else
        LOG(FATAL_THROW) << "Unexpected LoMa matcher input: " << name;
    }

    const std::vector<Ort::Value> outputs = model_.Run(inputs);
    THROW_CHECK_GE(outputs.size(),
                   2);  // m0, m1, mscores0, mscores1 -- see ctor.

    int m0_idx = -1, mscores0_idx = -1;
    const auto& names = model_.output_names();
    for (size_t i = 0; i < names.size(); ++i) {
      if (std::string(names[i]) == "m0") m0_idx = static_cast<int>(i);
      if (std::string(names[i]) == "mscores0")
        mscores0_idx = static_cast<int>(i);
    }
    THROW_CHECK_GE(m0_idx, 0);
    THROW_CHECK_GE(mscores0_idx, 0);

    const auto m0_shape =
        outputs[m0_idx].GetTensorTypeAndShapeInfo().GetShape();
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

  // Converts COLMAP pixel-space keypoints (top-left pixel corner = (0, 0)) back
  // to LoMa's normalized [-1, 1] convention
  Features FeaturesFromImage(const Image& image) const {
    THROW_CHECK_NOTNULL(image.keypoints);
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK_NOTNULL(image.camera);
    THROW_CHECK(image.descriptors->type == expected_extractor_type_)
        << "LoMa matcher expected features of type "
        << FeatureExtractorTypeToString(expected_extractor_type_) << ", got "
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
  const FeatureExtractorType expected_extractor_type_;
  ONNXModel model_;
};

#endif

}  // namespace

bool LomaExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_LE(min_score, 1);
  return true;
}

std::unique_ptr<FeatureExtractor> CreateLomaFeatureExtractor(
    const FeatureExtractionOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  switch (options.type) {
    case FeatureExtractorType::LOMA_B: {
      if (options.loma->use_bf16) {
        try {
          return std::make_unique<LomaFeatureExtractor>(
              options,
              options.loma->descriptor_model_path_bf16,
              /*probe_descriptor_support=*/true);
        } catch (const Ort::Exception& e) {
          LOG(WARNING) << "Failed to initialize the bf16 LoMa descriptor ("
                       << e.what() << "); falling back to fp32";
        }
      }
      return std::make_unique<LomaFeatureExtractor>(
          options, options.loma->descriptor_model_path);
    }
    case FeatureExtractorType::LOMA_B128:
      // No bf16 variant for dedode_b (VGG-only, no DINO) -- use_bf16 is a
      // no-op here.
      return std::make_unique<LomaFeatureExtractor>(
          options, options.loma->descriptor_b128_model_path);
    default:
      throw std::runtime_error("Unknown LoMa extractor type.");
  }
#else
  throw std::runtime_error("LoMa feature extraction requires ONNX support.");
#endif
}

bool LomaMatchingOptions::Check() const {
  CHECK_OPTION_GE(min_score, 0);
  CHECK_OPTION_LE(min_score, 1);
  return brute_force.Check();
}

#ifdef COLMAP_ONNX_ENABLED
namespace {
std::unique_ptr<FeatureMatcher> CreateLomaVariantMatcher(
    const FeatureMatchingOptions& options,
    const LomaVariantMatcherOptions& variant,
    FeatureExtractorType expected_extractor_type) {
  if (options.loma->use_bf16) {
    try {
      return std::make_unique<LomaFeatureMatcher>(options,
                                                  variant.model_path_bf16,
                                                  expected_extractor_type,
                                                  /*probe_model_support=*/true);
    } catch (const Ort::Exception& e) {
      LOG(WARNING) << "Failed to initialize the bf16 LoMa matcher (" << e.what()
                   << "); falling back to fp32";
    }
  }
  return std::make_unique<LomaFeatureMatcher>(
      options, variant.model_path, expected_extractor_type);
}
}  // namespace
#endif

std::unique_ptr<FeatureMatcher> CreateLomaFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  switch (options.type) {
    case FeatureMatcherType::LOMA_BRUTEFORCE:
      return CreateBruteForceONNXFeatureMatcher(options,
                                                options.loma->brute_force);
    case FeatureMatcherType::LOMA_B:
      return CreateLomaVariantMatcher(
          options, options.loma->b, FeatureExtractorType::LOMA_B);
    case FeatureMatcherType::LOMA_B128:
      return CreateLomaVariantMatcher(
          options, options.loma->b128, FeatureExtractorType::LOMA_B128);
    case FeatureMatcherType::LOMA_R:
      return CreateLomaVariantMatcher(
          options, options.loma->r, FeatureExtractorType::LOMA_B);
    case FeatureMatcherType::LOMA_L:
      return CreateLomaVariantMatcher(
          options, options.loma->l, FeatureExtractorType::LOMA_B);
    case FeatureMatcherType::LOMA_G:
      return CreateLomaVariantMatcher(
          options, options.loma->g, FeatureExtractorType::LOMA_B);
    default:
      throw std::runtime_error("Unknown LoMa matcher type.");
  }
#else
  throw std::runtime_error("LoMa feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
