// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mvs_estimator.h"
#include "colmap/mvs/resources.h"
#include "colmap/util/logging.h"
#include "colmap/util/onnx_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace colmap {
namespace mvs {

#ifdef COLMAP_ONNX_ENABLED

namespace {

constexpr int kSpatialMultiple = 64;
constexpr int kNumDepthValues = 192;
constexpr double kPi = 3.14159265358979323846;
constexpr std::array<float, 3> kImageMean = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kImageStd = {0.229f, 0.224f, 0.225f};
constexpr std::array<float, 4> kStageScales = {0.125f, 0.25f, 0.5f, 1.0f};

size_t RoundUp(const size_t value, const size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

Eigen::Matrix4f ProjectionMatrix(const Image& image, const float scale) {
  Eigen::Matrix3f K =
      Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
          image.GetK());
  K.row(0) *= scale;
  K.row(1) *= scale;
  const Eigen::Matrix3f R =
      Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
          image.GetR());
  const Eigen::Vector3f T = Eigen::Map<const Eigen::Vector3f>(image.GetT());
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  projection.topLeftCorner<3, 3>() = K * R;
  projection.topRightCorner<3, 1>() = K * T;
  return projection;
}

Eigen::Vector3f CameraPoint(const Image& image,
                            const float x,
                            const float y,
                            const float depth) {
  const float* K = image.GetK();
  return Eigen::Vector3f(
      (x - K[2]) / K[0] * depth, (y - K[5]) / K[4] * depth, depth);
}

NormalMap ComputeNormalMap(const Image& image, const DepthMap& depth_map) {
  NormalMap normal_map(depth_map.GetWidth(), depth_map.GetHeight());
  if (depth_map.GetWidth() < 3 || depth_map.GetHeight() < 3) {
    return normal_map;
  }
  for (size_t row = 1; row + 1 < depth_map.GetHeight(); ++row) {
    for (size_t col = 1; col + 1 < depth_map.GetWidth(); ++col) {
      const float center_depth = depth_map.Get(row, col);
      const float left_depth = depth_map.Get(row, col - 1);
      const float right_depth = depth_map.Get(row, col + 1);
      const float up_depth = depth_map.Get(row - 1, col);
      const float down_depth = depth_map.Get(row + 1, col);
      if (center_depth <= 0 || left_depth <= 0 || right_depth <= 0 ||
          up_depth <= 0 || down_depth <= 0) {
        continue;
      }
      const Eigen::Vector3f center = CameraPoint(image, col, row, center_depth);
      const Eigen::Vector3f horizontal =
          CameraPoint(image, col + 1, row, right_depth) -
          CameraPoint(image, col - 1, row, left_depth);
      const Eigen::Vector3f vertical =
          CameraPoint(image, col, row + 1, down_depth) -
          CameraPoint(image, col, row - 1, up_depth);
      Eigen::Vector3f normal = horizontal.cross(vertical);
      const float norm = normal.norm();
      if (norm <= std::numeric_limits<float>::epsilon()) {
        continue;
      }
      normal /= norm;
      if (normal.dot(center) > 0) {
        normal = -normal;
      }
      normal_map.Set(row, col, 0, normal.x());
      normal_map.Set(row, col, 1, normal.y());
      normal_map.Set(row, col, 2, normal.z());
    }
  }
  return normal_map;
}

}  // namespace

class MVSFormerPlusPlus::Impl {
 public:
  Impl(Options options, const int device_index) : options_(std::move(options)) {
    if (options_.model_path.empty()) {
      options_.model_path = options_.num_views == 5
                                ? kDefaultMVSFormerPlusPlus5ViewUri
                                : kDefaultMVSFormerPlusPlus10ViewUri;
    }
    THROW_CHECK(!options_.model_path.empty())
        << "MVSFormer++ model_path is required until release artifacts are "
           "available";
    const std::string gpu_index =
        device_index >= 0 ? std::to_string(device_index) : options_.gpu_index;
    if (SelectONNXExecutionProvider(options_.use_gpu) ==
        ONNXExecutionProvider::CUDA) {
      ONNXModel probe(options_.model_path,
                      options_.num_threads,
                      options_.use_gpu,
                      gpu_index,
                      /*is_capability_probe=*/true);
    }
    model_ = std::make_unique<ONNXModel>(
        options_.model_path, options_.num_threads, options_.use_gpu, gpu_index);
    CheckModelContract();
  }

  Capabilities GetCapabilities() const {
    Capabilities capabilities;
    capabilities.requires_rgb = true;
    capabilities.produces_confidence = true;
    capabilities.supports_geometric_pass = true;
    capabilities.supports_repeated_source_images = true;
    capabilities.min_num_source_images = options_.num_views - 1;
    capabilities.max_num_source_images = options_.num_views - 1;
    return capabilities;
  }

  Result Estimate(const Problem& problem, const Pass pass) const {
    if (pass == Pass::GEOMETRIC) {
      return FilterGeometrically(problem);
    }
    return Infer(problem);
  }

 private:
  template <typename T>
  static Ort::Value MakeTensor(std::vector<T>* data,
                               const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<T>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        data->data(),
        data->size(),
        shape.data(),
        shape.size());
  }

  void CheckModelContract() const {
    const std::vector<std::string> input_names = {
        "images",
        "source_from_reference_stage1",
        "source_from_reference_stage2",
        "source_from_reference_stage3",
        "source_from_reference_stage4",
        "reference_inverse_intrinsics_stage1",
        "depth_values"};
    const int64_t num_sources = options_.num_views - 1;
    const std::vector<std::vector<int64_t>> input_shapes = {
        {1, options_.num_views, 3, -1, -1},
        {1, num_sources, 4, 4},
        {1, num_sources, 4, 4},
        {1, num_sources, 4, 4},
        {1, num_sources, 4, 4},
        {1, 3, 3},
        {1, kNumDepthValues}};
    THROW_CHECK_EQ(model_->input_names().size(), input_names.size());
    for (size_t i = 0; i < input_names.size(); ++i) {
      ThrowCheckONNXNode(model_->input_names()[i],
                         input_names[i],
                         model_->input_shapes()[i],
                         input_shapes[i]);
    }
    THROW_CHECK_EQ(model_->output_names().size(), 2);
    ThrowCheckONNXNode(model_->output_names()[0],
                       "depth",
                       model_->output_shapes()[0],
                       {-1, -1, -1});
    ThrowCheckONNXNode(model_->output_names()[1],
                       "confidence",
                       model_->output_shapes()[1],
                       {-1, -1, -1});
  }

  Result Infer(const Problem& problem) const {
    THROW_CHECK_NOTNULL(problem.images);
    THROW_CHECK_EQ(problem.src_image_idxs.size(), options_.num_views - 1);
    const Image& ref_image = problem.images->at(problem.ref_image_idx);
    size_t padded_width = ref_image.GetWidth();
    size_t padded_height = ref_image.GetHeight();
    std::vector<int> image_idxs{problem.ref_image_idx};
    image_idxs.insert(image_idxs.end(),
                      problem.src_image_idxs.begin(),
                      problem.src_image_idxs.end());
    for (const int image_idx : image_idxs) {
      const Image& image = problem.images->at(image_idx);
      THROW_CHECK(image.GetBitmap().IsRGB());
      padded_width = std::max(padded_width, image.GetWidth());
      padded_height = std::max(padded_height, image.GetHeight());
    }
    padded_width = RoundUp(padded_width, kSpatialMultiple);
    padded_height = RoundUp(padded_height, kSpatialMultiple);

    const size_t image_area = padded_width * padded_height;
    std::vector<float> image_data(options_.num_views * 3 * image_area, 0.0f);
    for (size_t view_idx = 0; view_idx < image_idxs.size(); ++view_idx) {
      const Bitmap& bitmap =
          problem.images->at(image_idxs[view_idx]).GetBitmap();
      for (int row = 0; row < bitmap.Height(); ++row) {
        for (int col = 0; col < bitmap.Width(); ++col) {
          const BitmapColor<uint8_t> color = *bitmap.GetPixel(col, row);
          const std::array<float, 3> rgb = {
              static_cast<float>(color.r) / 255.0f,
              static_cast<float>(color.g) / 255.0f,
              static_cast<float>(color.b) / 255.0f};
          for (size_t channel = 0; channel < 3; ++channel) {
            const size_t offset = (view_idx * 3 + channel) * image_area +
                                  row * padded_width + col;
            image_data[offset] =
                (rgb[channel] - kImageMean[channel]) / kImageStd[channel];
          }
        }
      }
    }

    std::array<std::vector<float>, 4> projection_data;
    for (size_t stage = 0; stage < projection_data.size(); ++stage) {
      projection_data[stage].resize(problem.src_image_idxs.size() * 16);
      const Eigen::Matrix4f ref_projection =
          ProjectionMatrix(ref_image, kStageScales[stage]);
      for (size_t src_idx = 0; src_idx < problem.src_image_idxs.size();
           ++src_idx) {
        const Eigen::Matrix4f relative =
            ProjectionMatrix(
                problem.images->at(problem.src_image_idxs[src_idx]),
                kStageScales[stage]) *
            ref_projection.inverse();
        Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> relative_map(
            projection_data[stage].data() + src_idx * 16);
        relative_map = relative;
      }
    }

    Eigen::Matrix3f ref_K =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            ref_image.GetK());
    ref_K.row(0) *= kStageScales[0];
    ref_K.row(1) *= kStageScales[0];
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> ref_inv_K =
        ref_K.inverse();
    std::vector<float> inverse_intrinsics(ref_inv_K.data(),
                                          ref_inv_K.data() + 9);

    std::vector<float> depth_values(kNumDepthValues);
    for (int i = 0; i < kNumDepthValues; ++i) {
      depth_values[i] =
          problem.depth_min +
          (problem.depth_max - problem.depth_min) * i / (kNumDepthValues - 1);
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(7);
    inputs.push_back(MakeTensor(&image_data,
                                {1,
                                 options_.num_views,
                                 3,
                                 static_cast<int64_t>(padded_height),
                                 static_cast<int64_t>(padded_width)}));
    for (auto& stage_data : projection_data) {
      inputs.push_back(
          MakeTensor(&stage_data, {1, options_.num_views - 1, 4, 4}));
    }
    inputs.push_back(MakeTensor(&inverse_intrinsics, {1, 3, 3}));
    inputs.push_back(MakeTensor(&depth_values, {1, kNumDepthValues}));

    std::vector<Ort::Value> outputs = model_->Run(inputs);
    THROW_CHECK_EQ(outputs.size(), 2);
    const std::vector<int64_t> depth_shape =
        outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> confidence_shape =
        outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    THROW_CHECK(depth_shape ==
                (std::vector<int64_t>{1,
                                      static_cast<int64_t>(padded_height),
                                      static_cast<int64_t>(padded_width)}));
    THROW_CHECK(confidence_shape == depth_shape);
    const float* output_depth = outputs[0].GetTensorData<float>();
    const float* output_confidence = outputs[1].GetTensorData<float>();
    DepthMap depth_map(ref_image.GetWidth(),
                       ref_image.GetHeight(),
                       problem.depth_min,
                       problem.depth_max);
    Mat<float> confidence_map(ref_image.GetWidth(), ref_image.GetHeight(), 1);
    for (size_t row = 0; row < ref_image.GetHeight(); ++row) {
      for (size_t col = 0; col < ref_image.GetWidth(); ++col) {
        const size_t offset = row * padded_width + col;
        const float confidence = output_confidence[offset];
        const float depth = output_depth[offset];
        confidence_map.Set(row, col, confidence);
        if (confidence >= options_.min_confidence &&
            depth >= problem.depth_min && depth <= problem.depth_max) {
          depth_map.Set(row, col, depth);
        }
      }
    }
    Result result;
    result.depth_map = std::move(depth_map);
    result.normal_map = ComputeNormalMap(ref_image, result.depth_map);
    result.confidence_map = std::move(confidence_map);
    return result;
  }

  Result FilterGeometrically(const Problem& problem) const {
    THROW_CHECK_NOTNULL(problem.images);
    THROW_CHECK_NOTNULL(problem.depth_maps);
    THROW_CHECK_NOTNULL(problem.normal_maps);
    const Image& ref_image = problem.images->at(problem.ref_image_idx);
    const DepthMap& ref_depth = problem.depth_maps->at(problem.ref_image_idx);
    const NormalMap& ref_normal =
        problem.normal_maps->at(problem.ref_image_idx);
    DepthMap filtered(ref_depth.GetWidth(),
                      ref_depth.GetHeight(),
                      ref_depth.GetDepthMin(),
                      ref_depth.GetDepthMax());
    NormalMap filtered_normals(ref_depth.GetWidth(), ref_depth.GetHeight());
    std::vector<int> consistency_data;
    const float max_reproj_error_sq =
        options_.filter_max_reproj_error * options_.filter_max_reproj_error;
    const float min_normal_cos =
        std::cos(options_.filter_max_normal_error * kPi / 180.0);
    const Eigen::Matrix3f ref_K =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            ref_image.GetK());
    struct SourceGeometry {
      int image_idx;
      Eigen::Matrix3f source_from_reference_R;
      Eigen::Vector3f source_from_reference_T;
      Eigen::Matrix3f K;
    };
    std::vector<SourceGeometry> source_geometries;
    source_geometries.reserve(problem.src_image_idxs.size());
    for (const int src_image_idx : problem.src_image_idxs) {
      const Image& src_image = problem.images->at(src_image_idx);
      float R_data[9];
      float T_data[3];
      ComputeRelativePose(ref_image.GetR(),
                          ref_image.GetT(),
                          src_image.GetR(),
                          src_image.GetT(),
                          R_data,
                          T_data);
      SourceGeometry geometry;
      geometry.image_idx = src_image_idx;
      geometry.source_from_reference_R =
          Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R_data);
      geometry.source_from_reference_T =
          Eigen::Map<const Eigen::Vector3f>(T_data);
      geometry.K =
          Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
              src_image.GetK());
      source_geometries.push_back(std::move(geometry));
    }

    for (size_t row = 0; row < ref_depth.GetHeight(); ++row) {
      for (size_t col = 0; col < ref_depth.GetWidth(); ++col) {
        const float depth = ref_depth.Get(row, col);
        if (depth <= 0) {
          continue;
        }
        const Eigen::Vector3f point_ref =
            CameraPoint(ref_image, col, row, depth);
        const Eigen::Vector3f normal_ref(ref_normal.Get(row, col, 0),
                                         ref_normal.Get(row, col, 1),
                                         ref_normal.Get(row, col, 2));
        std::vector<int> consistent_images;
        for (const SourceGeometry& geometry : source_geometries) {
          const int src_image_idx = geometry.image_idx;
          const Image& src_image = problem.images->at(src_image_idx);
          const Eigen::Vector3f point_src =
              geometry.source_from_reference_R * point_ref +
              geometry.source_from_reference_T;
          if (point_src.z() <= 0) {
            continue;
          }
          const Eigen::Vector3f projected_src = geometry.K * point_src;
          const int src_col =
              std::lround(projected_src.x() / projected_src.z());
          const int src_row =
              std::lround(projected_src.y() / projected_src.z());
          const DepthMap& src_depth = problem.depth_maps->at(src_image_idx);
          if (src_col < 0 || src_row < 0 ||
              src_col >= static_cast<int>(src_depth.GetWidth()) ||
              src_row >= static_cast<int>(src_depth.GetHeight())) {
            continue;
          }
          const float measured_src_depth = src_depth.Get(src_row, src_col);
          if (measured_src_depth <= 0) {
            continue;
          }
          const Eigen::Vector3f measured_point_src =
              CameraPoint(src_image, src_col, src_row, measured_src_depth);
          const Eigen::Vector3f measured_point_ref =
              geometry.source_from_reference_R.transpose() *
              (measured_point_src - geometry.source_from_reference_T);
          if (measured_point_ref.z() <= 0) {
            continue;
          }
          const Eigen::Vector3f projected_ref = ref_K * measured_point_ref;
          const float reproj_col = projected_ref.x() / projected_ref.z();
          const float reproj_row = projected_ref.y() / projected_ref.z();
          const float reproj_error_sq =
              (reproj_col - col) * (reproj_col - col) +
              (reproj_row - row) * (reproj_row - row);
          const float relative_depth_error =
              std::abs(measured_point_ref.z() - depth) / depth;
          const NormalMap& src_normal = problem.normal_maps->at(src_image_idx);
          const Eigen::Vector3f normal_src(src_normal.Get(src_row, src_col, 0),
                                           src_normal.Get(src_row, src_col, 1),
                                           src_normal.Get(src_row, src_col, 2));
          const float normal_cos = normal_ref.dot(
              geometry.source_from_reference_R.transpose() * normal_src);
          if (reproj_error_sq <= max_reproj_error_sq &&
              relative_depth_error <= options_.filter_max_depth_error &&
              normal_cos >= min_normal_cos) {
            consistent_images.push_back(src_image_idx);
          }
        }
        if (consistent_images.size() <
            static_cast<size_t>(options_.filter_min_num_consistent)) {
          continue;
        }
        filtered.Set(row, col, depth);
        for (size_t channel = 0; channel < 3; ++channel) {
          filtered_normals.Set(
              row, col, channel, ref_normal.Get(row, col, channel));
        }
        consistency_data.push_back(static_cast<int>(row));
        consistency_data.push_back(static_cast<int>(col));
        consistency_data.push_back(static_cast<int>(consistent_images.size()));
        consistency_data.insert(consistency_data.end(),
                                consistent_images.begin(),
                                consistent_images.end());
      }
    }
    Result result;
    result.depth_map = std::move(filtered);
    result.normal_map = std::move(filtered_normals);
    if (options_.write_consistency_graph) {
      result.consistency_graph = ConsistencyGraph(
          ref_depth.GetWidth(), ref_depth.GetHeight(), consistency_data);
    }
    return result;
  }

  Options options_;
  std::unique_ptr<ONNXModel> model_;
};

MVSFormerPlusPlus::MVSFormerPlusPlus(Options options, const int device_index)
    : impl_(std::make_unique<Impl>(std::move(options), device_index)) {}

MVSFormerPlusPlus::~MVSFormerPlusPlus() = default;

MVSEstimator::Capabilities MVSFormerPlusPlus::GetCapabilities() const {
  return impl_->GetCapabilities();
}

MVSEstimator::Result MVSFormerPlusPlus::Estimate(const Problem& problem,
                                                 const Pass pass) {
  return impl_->Estimate(problem, pass);
}

#else

class MVSFormerPlusPlus::Impl {};

MVSFormerPlusPlus::MVSFormerPlusPlus(Options, int)
    : impl_(std::make_unique<Impl>()) {
  throw std::runtime_error(
      "MVSFormer++ requires an ONNX Runtime enabled build");
}

MVSFormerPlusPlus::~MVSFormerPlusPlus() = default;

MVSEstimator::Capabilities MVSFormerPlusPlus::GetCapabilities() const {
  return {};
}

MVSEstimator::Result MVSFormerPlusPlus::Estimate(const Problem&, Pass) {
  throw std::runtime_error(
      "MVSFormer++ requires an ONNX Runtime enabled build");
}

#endif

}  // namespace mvs
}  // namespace colmap
