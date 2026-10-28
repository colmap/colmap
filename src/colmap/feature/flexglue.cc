// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/flexglue.h"

#include "colmap/feature/utils.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/onnx.h"
#include "colmap/util/string.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <list>
#include <memory>
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
  std::vector<float> data;
  std::vector<int64_t> shape;
  // Size (width, height) of the image in the keypoint coordinate frame.
  float image_size[2] = {0.f, 0.f};
};

// Least-recently-used cache of dense features, keyed by image id. Unlike
// ImageFeatureCache, it retains many images, since pairs of exhaustive or
// retrieval-based matching revisit the same images far apart in time.
class DenseFeatureCache {
 public:
  explicit DenseFeatureCache(size_t capacity) : capacity_(capacity) {
    THROW_CHECK_GT(capacity_, 0);
  }

  template <typename CreateFn>
  std::shared_ptr<DenseFeatures> GetOrCreate(image_t image_id,
                                             CreateFn&& create) {
    if (image_id == kInvalidImageId) {
      return create();
    }
    const auto it = map_.find(image_id);
    if (it != map_.end()) {
      elems_.splice(elems_.begin(), elems_, it->second);
      return it->second->second;
    }
    std::shared_ptr<DenseFeatures> value = create();
    elems_.emplace_front(image_id, value);
    map_.emplace(image_id, elems_.begin());
    while (elems_.size() > capacity_) {
      map_.erase(elems_.back().first);
      elems_.pop_back();
    }
    return value;
  }

 private:
  using Elem = std::pair<image_t, std::shared_ptr<DenseFeatures>>;
  const size_t capacity_;
  std::list<Elem> elems_;
  FlatHashMap<image_t, std::list<Elem>::iterator> map_;
};

// ONNX Runtime's MatMulAddFusion turns each Linear layer (MatMul + bias Add)
// into a Gemm. Its CUDA kernel broadcasts the bias into the output with an
// extra rank-1 GEMM that runs on a slow non-tensor-core cuBLAS kernel (162
// times per pair). A separate, fused broadcast Add is faster: about 1.3 ms of
// GPU time per pair with the float16 matcher.
std::vector<std::string> MatcherDisabledOptimizers(bool use_gpu) {
  if (SelectONNXExecutionProvider(use_gpu) == ONNXExecutionProvider::CUDA) {
    return {"MatMulAddFusion"};
  }
  return {};
}

class FlexGlueFeatureMatcher : public FeatureMatcher {
 public:
  FlexGlueFeatureMatcher(const FeatureMatchingOptions& options,
                         const std::string& matcher_model_path)
      : options_(*THROW_CHECK_NOTNULL(options.flexglue)),
        descriptor_(options_.descriptor_model_path,
                    options.num_threads,
                    options.use_gpu,
                    options.gpu_index),
        matcher_(matcher_model_path,
                 options.num_threads,
                 options.use_gpu,
                 options.gpu_index,
                 /*is_capability_probe=*/false,
                 MatcherDisabledOptimizers(options.use_gpu)),
        dense_cache_(options_.feature_cache_size) {
    THROW_CHECK(options.Check());

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

    // Matcher: inputs are matched by name, see Match().
    const std::vector<std::string_view> expected_inputs = {"features0",
                                                           "features1",
                                                           "kpts0",
                                                           "kpts1",
                                                           "image_size0",
                                                           "image_size1",
                                                           "filter_threshold",
                                                           "max_distance_px",
                                                           "min_covisibility"};
    THROW_CHECK_EQ(matcher_.input_names().size(), expected_inputs.size());
    for (size_t i = 0; i < matcher_.input_names().size(); ++i) {
      const std::string_view name = matcher_.input_names()[i];
      THROW_CHECK(std::find(expected_inputs.begin(),
                            expected_inputs.end(),
                            name) != expected_inputs.end())
          << "Unexpected FlexGlue matcher input: " << name;
      ThrowCheckONNXElementType(name,
                                matcher_.input_element_types()[i],
                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

    for (size_t i = 0; i < matcher_.output_names().size(); ++i) {
      const std::string_view name = matcher_.output_names()[i];
      if (name == "matches0") {
        matches0_idx_ = static_cast<int>(i);
        ThrowCheckONNXNode(
            name, "matches0", matcher_.output_shapes()[i], {1, -1});
        ThrowCheckONNXElementType(name,
                                  matcher_.output_element_types()[i],
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      } else if (name == "mscores0") {
        mscores0_idx_ = static_cast<int>(i);
      } else if (name == "covisibility_score") {
        covisibility_score_idx_ = static_cast<int>(i);
      }
    }
    THROW_CHECK_GE(matches0_idx_, 0) << "Missing FlexGlue output: matches0";
    THROW_CHECK_GE(mscores0_idx_, 0) << "Missing FlexGlue output: mscores0";
  }

  ~FlexGlueFeatureMatcher() override {
    if (stats_.num_pairs == 0) {
      return;
    }
    LOG(INFO) << StringPrintf(
        "FlexGlue: matched %d pairs (matcher: %.1fms/pair), computed dense "
        "features for %d images (read: %.1fms/image, resize: %.1fms/image, "
        "descriptor: %.1fms/image)",
        stats_.num_pairs,
        1e3 * stats_.matcher_seconds / stats_.num_pairs,
        stats_.num_images,
        1e3 * stats_.read_seconds / std::max(1, stats_.num_images),
        1e3 * stats_.resize_seconds / std::max(1, stats_.num_images),
        1e3 * stats_.descriptor_seconds / std::max(1, stats_.num_images));
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

    const std::shared_ptr<DenseFeatures> dense1 = GetDenseFeatures(image1);
    const std::shared_ptr<DenseFeatures> dense2 = GetDenseFeatures(image2);

    auto create_keypoints = [](const Image& image) {
      return KeypointsToTensorData(*image.keypoints);
    };
    // Copy, since the second lookup may evict the first entry.
    std::vector<float> kpts1 =
        kpts_cache_.GetOrCreate(image1, create_keypoints);
    std::vector<float>& kpts2 =
        kpts_cache_.GetOrCreate(image2, create_keypoints);

    const std::vector<int64_t> kpts1_shape = {1, num_keypoints1, 2};
    const std::vector<int64_t> kpts2_shape = {1, num_keypoints2, 2};
    const std::vector<int64_t> image_size_shape = {1, 2};
    float filter_threshold = static_cast<float>(options_.filter_threshold);
    float max_distance_px = options_.max_distance_px > 0
                                ? static_cast<float>(options_.max_distance_px)
                                : std::numeric_limits<float>::infinity();
    float min_covisibility = static_cast<float>(options_.min_covisibility);

    std::vector<Ort::Value> inputs;
    inputs.reserve(matcher_.input_names().size());
    for (const char* name_cstr : matcher_.input_names()) {
      const std::string_view name(name_cstr);
      if (name == "features0") {
        inputs.push_back(CreateONNXTensor(dense1->data, dense1->shape));
      } else if (name == "features1") {
        inputs.push_back(CreateONNXTensor(dense2->data, dense2->shape));
      } else if (name == "kpts0") {
        inputs.push_back(CreateONNXTensor(kpts1, kpts1_shape));
      } else if (name == "kpts1") {
        inputs.push_back(CreateONNXTensor(kpts2, kpts2_shape));
      } else if (name == "image_size0") {
        inputs.push_back(
            CreateONNXTensor(dense1->image_size, 2, image_size_shape));
      } else if (name == "image_size1") {
        inputs.push_back(
            CreateONNXTensor(dense2->image_size, 2, image_size_shape));
      } else if (name == "filter_threshold") {
        inputs.push_back(CreateONNXScalarTensor(filter_threshold));
      } else if (name == "max_distance_px") {
        inputs.push_back(CreateONNXScalarTensor(max_distance_px));
      } else if (name == "min_covisibility") {
        inputs.push_back(CreateONNXScalarTensor(min_covisibility));
      } else {
        LOG(FATAL_THROW) << "Unexpected FlexGlue matcher input: " << name;
      }
    }

    Timer timer;
    timer.Start();
    const std::vector<Ort::Value> outputs = matcher_.Run(inputs);
    stats_.matcher_seconds += timer.ElapsedSeconds();
    stats_.num_pairs += 1;
    THROW_CHECK_EQ(outputs.size(), matcher_.output_names().size());

    const std::vector<int64_t> matches0_shape =
        outputs[matches0_idx_].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK_EQ(matches0_shape.size(), 2);
    THROW_CHECK_EQ(matches0_shape[0], 1);
    THROW_CHECK_EQ(matches0_shape[1], num_keypoints1);
    const int64_t* matches0 = outputs[matches0_idx_].GetTensorData<int64_t>();

    if (covisibility_score_idx_ >= 0 && VLOG_IS_ON(2)) {
      VLOG(2) << "FlexGlue covisibility score for images " << image1.image_id
              << " and " << image2.image_id << ": "
              << outputs[covisibility_score_idx_].GetTensorData<float>()[0];
    }

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

  std::shared_ptr<DenseFeatures> GetDenseFeatures(const Image& image) {
    return dense_cache_.GetOrCreate(image.image_id, [this, &image]() {
      return ComputeDenseFeatures(image);
    });
  }

  std::shared_ptr<DenseFeatures> ComputeDenseFeatures(const Image& image) {
    THROW_CHECK(!image.name.empty())
        << "FlexGlue requires the image name to read the image pixels.";
    THROW_CHECK(!options_.image_path.empty())
        << "FlexGlue requires the image path to read the image pixels; "
           "set FlexGlueMatchingOptions::image_path.";
    const std::filesystem::path path =
        std::filesystem::path(options_.image_path) / std::string(image.name);

    Timer timer;
    timer.Start();
    Bitmap bitmap;
    THROW_CHECK(bitmap.Read(path, /*as_rgb=*/true))
        << "Failed to read image: " << path;
    stats_.read_seconds += timer.ElapsedSeconds();

    auto dense = std::make_shared<DenseFeatures>();
    if (image.camera != nullptr) {
      // Keypoints are defined in the frame of the camera, which normally
      // coincides with the image. The image is squashed to the descriptor
      // input size anyway, so only warn if the aspect ratios disagree.
      dense->image_size[0] = static_cast<float>(image.camera->width);
      dense->image_size[1] = static_cast<float>(image.camera->height);
      const double camera_aspect =
          static_cast<double>(image.camera->width) / image.camera->height;
      const double image_aspect =
          static_cast<double>(bitmap.Width()) / bitmap.Height();
      if (std::abs(camera_aspect - image_aspect) > 1e-2 * camera_aspect) {
        LOG(WARNING) << "Image " << path << " has size " << bitmap.Width()
                     << "x" << bitmap.Height()
                     << ", which does not match the camera size "
                     << image.camera->width << "x" << image.camera->height;
      }
    } else {
      dense->image_size[0] = static_cast<float>(bitmap.Width());
      dense->image_size[1] = static_cast<float>(bitmap.Height());
    }

    timer.Restart();
    std::vector<float> input =
        ResizeBilinearToCHW(bitmap, descriptor_width_, descriptor_height_);
    stats_.resize_seconds += timer.ElapsedSeconds();
    const std::vector<int64_t> input_shape = {
        1, 3, descriptor_height_, descriptor_width_};
    std::vector<Ort::Value> inputs;
    inputs.push_back(CreateONNXTensor(input, input_shape));
    timer.Restart();
    const std::vector<Ort::Value> outputs = descriptor_.Run(inputs);
    stats_.descriptor_seconds += timer.ElapsedSeconds();
    stats_.num_images += 1;
    THROW_CHECK_EQ(outputs.size(), 1);

    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    dense->shape = info.GetShape();
    THROW_CHECK_EQ(dense->shape.size(), 4);
    const float* features = outputs[0].GetTensorData<float>();
    dense->data.assign(features, features + info.GetElementCount());
    return dense;
  }

  const FlexGlueMatchingOptions options_;
  ONNXModel descriptor_;
  ONNXModel matcher_;
  int descriptor_height_ = 0;
  int descriptor_width_ = 0;
  int matches0_idx_ = -1;
  int mscores0_idx_ = -1;
  int covisibility_score_idx_ = -1;

  DenseFeatureCache dense_cache_;
  ImageFeatureCache<std::vector<float>> kpts_cache_;

  // Runtime statistics, logged on destruction.
  struct Stats {
    int num_pairs = 0;
    int num_images = 0;
    double matcher_seconds = 0;
    double read_seconds = 0;
    double resize_seconds = 0;
    double descriptor_seconds = 0;
  };
  Stats stats_;
};

#endif

}  // namespace

bool FlexGlueMatchingOptions::Check() const {
  CHECK_OPTION_IN(filter_threshold, 0, 1);
  CHECK_OPTION_IN(min_covisibility, 0, 1);
  CHECK_OPTION_GT(feature_cache_size, 0);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateFlexGlueFeatureMatcher(
    const FeatureMatchingOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  const FlexGlueMatchingOptions& flexglue_options =
      *THROW_CHECK_NOTNULL(options.flexglue);
  // The float16 matcher is only beneficial (and supported) on CUDA GPUs.
  if (flexglue_options.use_fp16 && options.use_gpu &&
      !flexglue_options.matcher_model_path_fp16.empty() &&
      SelectONNXExecutionProvider(options.use_gpu) ==
          ONNXExecutionProvider::CUDA) {
    try {
      return std::make_unique<FlexGlueFeatureMatcher>(
          options, flexglue_options.matcher_model_path_fp16);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to initialize the fp16 FlexGlue matcher ("
                   << e.what() << "); falling back to fp32";
    }
  }
  return std::make_unique<FlexGlueFeatureMatcher>(
      options, flexglue_options.matcher_model_path);
#else
  throw std::runtime_error("FlexGlue feature matching requires ONNX support.");
#endif
}

}  // namespace colmap
