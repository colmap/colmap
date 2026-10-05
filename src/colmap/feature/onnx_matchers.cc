// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/feature/onnx_matchers.h"

#include "colmap/feature/utils.h"
#include "colmap/util/onnx.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace colmap {
namespace {

#ifdef COLMAP_ONNX_ENABLED

class BruteForceONNXFeatureMatcher : public FeatureMatcher {
 public:
  explicit BruteForceONNXFeatureMatcher(
      const FeatureMatchingOptions& options,
      const BruteForceONNXMatchingOptions& brute_force_options,
      std::vector<FeatureExtractorType> supported_feature_types,
      bool normalize_descriptors)
      : brute_force_options_(brute_force_options),
        supported_feature_types_(std::move(supported_feature_types)),
        normalize_descriptors_(normalize_descriptors),
        model_(brute_force_options.model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options.Check());
    THROW_CHECK(!supported_feature_types_.empty());
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

    auto create = [this](const Image& image) {
      return FeaturesFromImage(image);
    };
    Features& cached1 = cache_.GetOrCreate(image1, create);
    Features& cached2 = cache_.GetOrCreate(image2, create);

    // Create tensors from cached data (tensors must be recreated each call
    // since they reference the underlying data and get consumed by Run()).
    float min_cossim = static_cast<float>(brute_force_options_.min_cossim);
    float max_ratio = static_cast<float>(brute_force_options_.max_ratio);
    int64_t cross_check = brute_force_options_.cross_check ? 1 : 0;

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(
        CreateONNXTensor(cached1.descriptors_data, cached1.descriptors_shape));
    input_tensors.emplace_back(
        CreateONNXTensor(cached2.descriptors_data, cached2.descriptors_shape));
    input_tensors.emplace_back(CreateONNXScalarTensor(min_cossim));
    input_tensors.emplace_back(CreateONNXScalarTensor(max_ratio));
    input_tensors.emplace_back(CreateONNXScalarTensor(cross_check));

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
    LOG(FATAL_THROW) << "Guided matching not supported for ONNX brute-force "
                        "matching.";
  }

 private:
  struct Features {
    std::vector<float> descriptors_data;
    std::vector<int64_t> descriptors_shape;
  };

  Features FeaturesFromImage(const Image& image) {
    THROW_CHECK_NOTNULL(image.descriptors);
    THROW_CHECK(std::find(supported_feature_types_.begin(),
                          supported_feature_types_.end(),
                          image.descriptors->type) !=
                supported_feature_types_.end())
        << "Unsupported feature type: "
        << FeatureExtractorTypeToString(image.descriptors->type);
    FeatureDescriptorsFloat descriptors = image.descriptors->ToFloat();
    if (normalize_descriptors_) {
      L2NormalizeFeatureDescriptors(&descriptors.data);
    }

    const int num_keypoints = descriptors.data.rows();
    const int descriptor_dim = descriptors.data.cols();
    THROW_CHECK_GT(descriptor_dim, 0);

    Features features;
    features.descriptors_shape = {num_keypoints, descriptor_dim};
    features.descriptors_data.assign(
        descriptors.data.data(),
        descriptors.data.data() + descriptors.data.size());

    return features;
  }

  const BruteForceONNXMatchingOptions brute_force_options_;
  const std::vector<FeatureExtractorType> supported_feature_types_;
  const bool normalize_descriptors_;
  ONNXModel model_;

  // Cached features for avoiding redundant data copies.
  ImageFeatureCache<Features> cache_;
};

#endif

}  // namespace

bool BruteForceONNXMatchingOptions::Check() const {
  CHECK_OPTION_IN(min_cossim, -1, 1);
  CHECK_OPTION_IN(max_ratio, 0, 1);
  return true;
}

std::unique_ptr<FeatureMatcher> CreateBruteForceONNXFeatureMatcher(
    const FeatureMatchingOptions& options,
    const BruteForceONNXMatchingOptions& brute_force_options,
    std::vector<FeatureExtractorType> supported_feature_types,
    bool normalize_descriptors) {
#ifdef COLMAP_ONNX_ENABLED
  FeatureMatchingOptions effective_options = options;
  if (SelectONNXExecutionProvider(effective_options.use_gpu) ==
      ONNXExecutionProvider::COREML) {
    // CoreML cannot handle the zero-length dynamic output from NonZero.
    LOG_FIRST_N(WARNING, 1)
        << "The ONNX brute-force matcher is not supported by CoreML; using "
           "the CPU execution provider instead";
    effective_options.use_gpu = false;
  }
  return std::make_unique<BruteForceONNXFeatureMatcher>(
      effective_options,
      brute_force_options,
      std::move(supported_feature_types),
      normalize_descriptors);
#else
  throw std::runtime_error("Brute-force ONNX matching requires ONNX support.");
#endif
}

}  // namespace colmap
