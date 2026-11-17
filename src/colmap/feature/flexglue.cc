// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/flexglue.h"

#include "colmap/sensor/bitmap.h"
#include "colmap/util/cache.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/onnx.h"

#ifdef COLMAP_CUDA_ENABLED
#include "colmap/util/cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

// Bilinear resize of an RGB bitmap to a row-major [3, H, W] float tensor with
// values in [0, 1]. Matches torch.nn.functional.interpolate(mode="bilinear",
// align_corners=False, antialias=False), which the reference implementation
// uses to resize the images fed to the descriptor.
std::vector<float> ResizeBilinearToCHW(const Bitmap& bitmap,
                                       const int out_width,
                                       const int out_height) {
  THROW_CHECK(bitmap.IsRGB());
  const int in_width = bitmap.Width();
  const int in_height = bitmap.Height();
  THROW_CHECK_GT(in_width, 0);
  THROW_CHECK_GT(in_height, 0);
  const uint8_t* data = bitmap.RowMajorData().data();
  const int pitch = bitmap.Pitch();

  struct Tap {
    int index0;
    int index1;
    float weight1;
  };
  auto compute_taps = [](int in_size, int out_size) {
    std::vector<Tap> taps(out_size);
    const float scale = static_cast<float>(in_size) / out_size;
    for (int i = 0; i < out_size; ++i) {
      const float src = std::max(0.f, scale * (i + 0.5f) - 0.5f);
      const int index0 = std::min(static_cast<int>(src), in_size - 1);
      const int index1 = index0 < in_size - 1 ? index0 + 1 : index0;
      taps[i] = {index0, index1, src - index0};
    }
    return taps;
  };
  const std::vector<Tap> taps_x = compute_taps(in_width, out_width);
  const std::vector<Tap> taps_y = compute_taps(in_height, out_height);

  const size_t plane_size = static_cast<size_t>(out_width) * out_height;
  std::vector<float> chw(3 * plane_size);
  constexpr float kInv255 = 1.f / 255.f;
  for (int y = 0; y < out_height; ++y) {
    const Tap& ty = taps_y[y];
    const uint8_t* row0 = data + static_cast<size_t>(ty.index0) * pitch;
    const uint8_t* row1 = data + static_cast<size_t>(ty.index1) * pitch;
    for (int x = 0; x < out_width; ++x) {
      const Tap& tx = taps_x[x];
      const int c0 = 3 * tx.index0;
      const int c1 = 3 * tx.index1;
      for (int c = 0; c < 3; ++c) {
        const float top =
            (1.f - tx.weight1) * row0[c0 + c] + tx.weight1 * row0[c1 + c];
        const float bottom =
            (1.f - tx.weight1) * row1[c0 + c] + tx.weight1 * row1[c1 + c];
        chw[c * plane_size + static_cast<size_t>(y) * out_width + x] =
            ((1.f - ty.weight1) * top + ty.weight1 * bottom) * kInv255;
      }
    }
  }
  return chw;
}

// Dense features of an image, as output by the descriptor model.
struct DenseFeatures {
  // Owned by ONNX Runtime, in host or GPU memory.
  Ort::Value value{nullptr};
  // Size of the image file.
  int width = 0;
  int height = 0;
  // Whether the image size was compared with the camera size.
  bool checked_camera_size = false;
};

// Non-owning tensor referencing the memory of the given tensor.
Ort::Value TensorView(const Ort::Value& value) {
  const Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> shape = info.GetShape();
  return Ort::Value::CreateTensor(value.GetTensorMemoryInfo(),
                                  const_cast<void*>(value.GetTensorRawData()),
                                  info.GetElementCount() * sizeof(float),
                                  shape.data(),
                                  shape.size(),
                                  info.GetElementType());
}

// Whether to load the matcher with float16 transformer layers instead of the
// float32 matcher. This is decided up front, as for the bf16 LoMa models: the
// float16 matcher only runs with the CUDA execution provider, on GPUs with
// compute capability 7.0 or newer (Volta and later, which have fast float16
// arithmetic, with or without tensor cores). On older GPUs, float16
// arithmetic is slow or emulated (e.g., at 1/64 of the float32 rate on most
// Pascal GPUs), and a session that initializes may still fail at inference,
// which the fallback on initialization failures would not catch.
bool UseFlexGlueFp16(const FeatureMatchingOptions& options) {
  const FlexGlueMatchingOptions& flexglue_options = *options.flexglue;
  if (!flexglue_options.use_fp16 ||
      flexglue_options.matcher_model_path_fp16.empty() ||
      SelectONNXExecutionProvider(options.use_gpu) !=
          ONNXExecutionProvider::CUDA) {
    return false;
  }
#ifdef COLMAP_CUDA_ENABLED
  const int device = SelectONNXCudaDevice(options.gpu_index);
  if (CudaDeviceHasComputeCapability(device, /*major=*/7)) {
    return true;
  }
  LOG(WARNING) << "The fp16 FlexGlue matcher requires a CUDA GPU with compute "
                  "capability 7.0 or newer, which GPU "
               << device << " is not; falling back to fp32";
#endif
  return false;
}

// Matcher with float16 or float32 transformer layers (see UseFlexGlueFp16).
// The float16 matcher is loaded as a regular session, with a fallback to the
// float32 matcher if its initialization fails.
std::unique_ptr<ONNXModel> CreateMatcherModel(
    const FeatureMatchingOptions& options) {
  const FlexGlueMatchingOptions& flexglue_options = *options.flexglue;
  if (UseFlexGlueFp16(options)) {
    try {
      return std::make_unique<ONNXModel>(
          flexglue_options.matcher_model_path_fp16,
          options.num_threads,
          options.use_gpu,
          options.gpu_index);
    } catch (const Ort::Exception& e) {
      LOG(WARNING) << "Failed to initialize the fp16 FlexGlue matcher ("
                   << e.what() << "); falling back to fp32";
    }
  }
  return std::make_unique<ONNXModel>(flexglue_options.matcher_model_path,
                                     options.num_threads,
                                     options.use_gpu,
                                     options.gpu_index);
}

class FlexGlueFeatureMatcher : public FeatureMatcher {
 public:
  explicit FlexGlueFeatureMatcher(const FeatureMatchingOptions& options)
      : options_(*THROW_CHECK_NOTNULL(options.flexglue)),
        descriptor_(options_.descriptor_model_path,
                    options.num_threads,
                    options.use_gpu,
                    options.gpu_index),
        matcher_(CreateMatcherModel(options)),
        dense_cache_(options_.feature_cache_size,
                     [this](const std::string& name) {
                       return ComputeDenseFeatures(name);
                     }) {
    // Descriptor: image [1, 3, H, W] -> features [1, C, H/8, W/8].
    THROW_CHECK_EQ(descriptor_.input_shapes().size(), 1);
    ThrowCheckONNXNode(descriptor_.input_names()[0],
                       "image",
                       descriptor_.input_shapes()[0],
                       {1, 3, -1, -1});
    ThrowCheckONNXElementType(descriptor_.input_names()[0],
                              descriptor_.input_element_types()[0],
                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    descriptor_height_ = static_cast<int>(descriptor_.input_shapes()[0][2]);
    descriptor_width_ = static_cast<int>(descriptor_.input_shapes()[0][3]);
    THROW_CHECK_GT(descriptor_height_, 0)
        << "The FlexGlue descriptor must have a static input size.";
    THROW_CHECK_GT(descriptor_width_, 0)
        << "The FlexGlue descriptor must have a static input size.";
    THROW_CHECK_EQ(descriptor_.output_shapes().size(), 1);
    ThrowCheckONNXNode(descriptor_.output_names()[0],
                       "features",
                       descriptor_.output_shapes()[0],
                       {1, -1, -1, -1});
    ThrowCheckONNXElementType(descriptor_.output_names()[0],
                              descriptor_.output_element_types()[0],
                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // Keep the dense features on the GPU, where the matcher reads them, which
    // avoids copying both feature maps to the GPU for every pair. The GPU
    // memory is bounded by feature_cache_size.
    if (matcher_->execution_provider() == ONNXExecutionProvider::CUDA &&
        descriptor_.execution_provider() == ONNXExecutionProvider::CUDA) {
      feature_memory_info_.emplace("Cuda",
                                   OrtArenaAllocator,
                                   SelectONNXCudaDevice(options.gpu_index),
                                   OrtMemTypeDefault);
    }

    // Matcher: inputs are matched by name, see Match().
    const std::vector<std::string_view> expected_inputs = {"features0",
                                                           "features1",
                                                           "kpts0",
                                                           "kpts1",
                                                           "image_size0",
                                                           "image_size1",
                                                           "filter_threshold",
                                                           "max_distance_px",
                                                           "min_covisibility",
                                                           "early_stop_decay"};
    THROW_CHECK_EQ(matcher_->input_names().size(), expected_inputs.size());
    for (size_t i = 0; i < matcher_->input_names().size(); ++i) {
      const std::string_view name = matcher_->input_names()[i];
      THROW_CHECK(std::find(expected_inputs.begin(),
                            expected_inputs.end(),
                            name) != expected_inputs.end())
          << "Unexpected FlexGlue matcher input: " << name;
      ThrowCheckONNXElementType(name,
                                matcher_->input_element_types()[i],
                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

    for (size_t i = 0; i < matcher_->output_names().size(); ++i) {
      const std::string_view name = matcher_->output_names()[i];
      if (name == "matches0") {
        matches0_idx_ = static_cast<int>(i);
        ThrowCheckONNXNode(
            name, "matches0", matcher_->output_shapes()[i], {1, -1});
        ThrowCheckONNXElementType(name,
                                  matcher_->output_element_types()[i],
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      } else if (name == "covisibility_score") {
        covisibility_score_idx_ = static_cast<int>(i);
        ThrowCheckONNXNode(
            name, "covisibility_score", matcher_->output_shapes()[i], {1});
        ThrowCheckONNXElementType(name,
                                  matcher_->output_element_types()[i],
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      }
    }
    THROW_CHECK_GE(matches0_idx_, 0) << "Missing FlexGlue output: matches0";
    THROW_CHECK_GE(covisibility_score_idx_, 0)
        << "Missing FlexGlue output: covisibility_score";
  }

  void Match(const Image& image1,
             const Image& image2,
             FeatureMatches* matches) override {
    THROW_CHECK_NOTNULL(matches);
    matches->clear();
    THROW_CHECK_NOTNULL(image1.keypoints);
    THROW_CHECK_NOTNULL(image2.keypoints);

    const int64_t num_keypoints1 = image1.keypoints->size();
    const int64_t num_keypoints2 = image2.keypoints->size();
    if (num_keypoints1 == 0 || num_keypoints2 == 0) {
      return;
    }

    const std::shared_ptr<DenseFeatures> dense1 =
        dense_cache_.Get(std::string(image1.name));
    const std::shared_ptr<DenseFeatures> dense2 =
        dense_cache_.Get(std::string(image2.name));
    std::array<float, 2> image_size1 = ImageSize(image1, *dense1);
    std::array<float, 2> image_size2 = ImageSize(image2, *dense2);
    std::vector<float> kpts1 = KeypointsToTensorData(*image1.keypoints);
    std::vector<float> kpts2 = KeypointsToTensorData(*image2.keypoints);

    const std::vector<int64_t> kpts1_shape = {1, num_keypoints1, 2};
    const std::vector<int64_t> kpts2_shape = {1, num_keypoints2, 2};
    const std::vector<int64_t> image_size_shape = {1, 2};
    float filter_threshold = static_cast<float>(options_.filter_threshold);
    float max_distance_px = options_.max_distance_px > 0
                                ? static_cast<float>(options_.max_distance_px)
                                : std::numeric_limits<float>::infinity();
    float min_covisibility = static_cast<float>(options_.min_covisibility);
    float early_stop_decay = static_cast<float>(options_.early_stop_decay);

    std::vector<Ort::Value> inputs;
    inputs.reserve(matcher_->input_names().size());
    for (const char* name_cstr : matcher_->input_names()) {
      const std::string_view name(name_cstr);
      if (name == "features0") {
        inputs.push_back(TensorView(dense1->value));
      } else if (name == "features1") {
        inputs.push_back(TensorView(dense2->value));
      } else if (name == "kpts0") {
        inputs.push_back(CreateONNXTensor(kpts1, kpts1_shape));
      } else if (name == "kpts1") {
        inputs.push_back(CreateONNXTensor(kpts2, kpts2_shape));
      } else if (name == "image_size0") {
        inputs.push_back(
            CreateONNXTensor(image_size1.data(), 2, image_size_shape));
      } else if (name == "image_size1") {
        inputs.push_back(
            CreateONNXTensor(image_size2.data(), 2, image_size_shape));
      } else if (name == "filter_threshold") {
        inputs.push_back(CreateONNXScalarTensor(filter_threshold));
      } else if (name == "max_distance_px") {
        inputs.push_back(CreateONNXScalarTensor(max_distance_px));
      } else if (name == "min_covisibility") {
        inputs.push_back(CreateONNXScalarTensor(min_covisibility));
      } else if (name == "early_stop_decay") {
        inputs.push_back(CreateONNXScalarTensor(early_stop_decay));
      } else {
        LOG(FATAL_THROW) << "Unexpected FlexGlue matcher input: " << name;
      }
    }

    const std::vector<Ort::Value> outputs = matcher_->Run(inputs);
    THROW_CHECK_EQ(outputs.size(), matcher_->output_names().size());

    const std::vector<int64_t> matches0_shape =
        outputs[matches0_idx_].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(matches0_shape.size(), 2);
    THROW_CHECK_EQ(matches0_shape[0], 1);
    THROW_CHECK_EQ(matches0_shape[1], num_keypoints1);
    const int64_t* matches0 = outputs[matches0_idx_].GetTensorData<int64_t>();

    VLOG(2) << "FlexGlue covisibility score for images " << image1.name
            << " and " << image2.name << ": "
            << outputs[covisibility_score_idx_].GetTensorData<float>()[0];

    // The model already applies the confidence, distance, mutual nearest
    // neighbor and pair checks, so every non-negative entry is a valid match.
    for (int64_t i = 0; i < num_keypoints1; ++i) {
      const int64_t j = matches0[i];
      if (j < 0) {
        continue;
      }
      THROW_CHECK_LT(j, num_keypoints2);
      matches->emplace_back(static_cast<point2D_t>(i),
                            static_cast<point2D_t>(j));
    }
  }

  void MatchGuided(double max_error,
                   const Image& image1,
                   const Image& image2,
                   TwoViewGeometry* two_view_geometry) override {
    LOG(FATAL_THROW) << "Guided matching not supported for FlexGlue.";
  }

 private:
  // COLMAP and FlexGlue both place the origin at the upper left image corner,
  // i.e., the center of the upper left pixel is at (0.5, 0.5).
  static std::vector<float> KeypointsToTensorData(
      const FeatureKeypoints& keypoints) {
    std::vector<float> data(2 * keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
      data[2 * i + 0] = keypoints[i].x;
      data[2 * i + 1] = keypoints[i].y;
    }
    return data;
  }

  // Size (width, height) of the image in the frame of its keypoints: that of
  // the camera, which normally coincides with the image file, else that of
  // the image file. The image is squashed to the descriptor input size anyway,
  // so only warn (once per cached image) if the aspect ratios disagree.
  static std::array<float, 2> ImageSize(const Image& image,
                                        DenseFeatures& dense) {
    if (image.camera == nullptr) {
      return {static_cast<float>(dense.width),
              static_cast<float>(dense.height)};
    }
    if (!dense.checked_camera_size) {
      dense.checked_camera_size = true;
      const double camera_aspect =
          static_cast<double>(image.camera->width) / image.camera->height;
      const double image_aspect =
          static_cast<double>(dense.width) / dense.height;
      if (std::abs(camera_aspect - image_aspect) > 1e-2 * camera_aspect) {
        LOG(WARNING) << "Image " << image.name << " has size " << dense.width
                     << "x" << dense.height
                     << ", which does not match the camera size "
                     << image.camera->width << "x" << image.camera->height;
      }
    }
    return {static_cast<float>(image.camera->width),
            static_cast<float>(image.camera->height)};
  }

  std::shared_ptr<DenseFeatures> ComputeDenseFeatures(const std::string& name) {
    THROW_CHECK(!name.empty())
        << "FlexGlue requires the image name to read the image pixels.";
    const std::filesystem::path path =
        std::filesystem::path(options_.image_path) / name;
    Bitmap bitmap;
    THROW_CHECK(bitmap.Read(path, /*as_rgb=*/true))
        << "Failed to read image: " << path;

    auto dense = std::make_shared<DenseFeatures>();
    dense->width = bitmap.Width();
    dense->height = bitmap.Height();

    std::vector<float> input =
        ResizeBilinearToCHW(bitmap, descriptor_width_, descriptor_height_);
    const std::vector<int64_t> input_shape = {
        1, 3, descriptor_height_, descriptor_width_};
    std::vector<Ort::Value> inputs;
    inputs.push_back(CreateONNXTensor(input, input_shape));
    std::vector<Ort::Value> outputs =
        feature_memory_info_ ? descriptor_.Run(inputs, *feature_memory_info_)
                             : descriptor_.Run(inputs);
    THROW_CHECK_EQ(outputs.size(), 1);
    THROW_CHECK_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetShape().size(), 4);
    dense->value = std::move(outputs[0]);
    return dense;
  }

  const FlexGlueMatchingOptions options_;
  ONNXModel descriptor_;
  std::unique_ptr<ONNXModel> matcher_;
  int descriptor_height_ = 0;
  int descriptor_width_ = 0;
  int matches0_idx_ = -1;
  int covisibility_score_idx_ = -1;

  // Memory of the cached dense features, if not host memory.
  std::optional<Ort::MemoryInfo> feature_memory_info_;
  // Dense features keyed by image name. Unlike ImageFeatureCache, it retains
  // many images, since pairs of exhaustive or retrieval-based matching revisit
  // the same images far apart in time. Neither the cache nor its entries
  // (see ImageSize) are thread-safe, so a matcher must not run concurrent
  // Match() calls (see CreateFlexGlueFeatureMatcher).
  LRUCache<std::string, DenseFeatures> dense_cache_;
};

#endif

}  // namespace

bool FlexGlueMatchingOptions::Check() const {
  CHECK_OPTION(!image_path.empty());
  CHECK_OPTION(ExistsDir(image_path));
  CHECK_OPTION_IN(filter_threshold, 0, 1);
  CHECK_OPTION(!std::isnan(max_distance_px));
  CHECK_OPTION_IN(min_covisibility, 0, 1);
  CHECK_OPTION_IN(early_stop_decay, 0, 1);
  CHECK_OPTION_GT(feature_cache_size, 0);
  CHECK_OPTION(!descriptor_model_path.empty());
  CHECK_OPTION(!matcher_model_path.empty());
  return true;
}

std::unique_ptr<FeatureMatcher> CreateFlexGlueFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  THROW_CHECK(options.Check());
  return std::make_unique<FlexGlueFeatureMatcher>(options);
#else
  throw std::runtime_error("FlexGlue feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
