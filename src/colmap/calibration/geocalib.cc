// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/geocalib.h"

#include "colmap/util/logging.h"
#include "colmap/util/onnx.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace colmap {
namespace {

constexpr int kEdgeDivisibleBy = 32;

#ifdef COLMAP_ONNX_ENABLED

struct GeoCalibOutputIndices {
  size_t up_field_idx = 0;
  size_t up_confidence_idx = 0;
  size_t latitude_field_idx = 0;
  size_t latitude_confidence_idx = 0;
};

GeoCalibOutputIndices CheckONNXSignatureAndGetOutputIndices(
    const ONNXModel& model) {
  THROW_CHECK_EQ(model.input_shapes().size(), 1);
  THROW_CHECK_EQ(model.input_element_types().size(), 1);
  ThrowCheckONNXNode(
      model.input_names()[0], "image", model.input_shapes()[0], {1, 3, -1, -1});
  ThrowCheckONNXElementType(model.input_names()[0],
                            model.input_element_types()[0],
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  THROW_CHECK_EQ(model.output_shapes().size(), 4);
  THROW_CHECK_EQ(model.output_element_types().size(), 4);

  std::optional<size_t> up_field_idx;
  std::optional<size_t> up_confidence_idx;
  std::optional<size_t> latitude_field_idx;
  std::optional<size_t> latitude_confidence_idx;

  for (size_t i = 0; i < model.output_names().size(); ++i) {
    const std::string_view name = model.output_names()[i];
    const auto& shape = model.output_shapes()[i];
    ThrowCheckONNXElementType(name,
                              model.output_element_types()[i],
                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (name == "up_field") {
      ThrowCheckONNXNode(name, "up_field", shape, {-1, 2, -1, -1});
      up_field_idx = i;
    } else if (name == "up_confidence") {
      ThrowCheckONNXNode(name, "up_confidence", shape, {-1, 1, -1, -1});
      up_confidence_idx = i;
    } else if (name == "latitude_field") {
      ThrowCheckONNXNode(name, "latitude_field", shape, {-1, 1, -1, -1});
      latitude_field_idx = i;
    } else if (name == "latitude_confidence") {
      ThrowCheckONNXNode(name, "latitude_confidence", shape, {-1, 1, -1, -1});
      latitude_confidence_idx = i;
    } else {
      LOG(FATAL_THROW) << "Unexpected GeoCalib output: " << name;
    }
  }

  THROW_CHECK(up_field_idx.has_value());
  THROW_CHECK(up_confidence_idx.has_value());
  THROW_CHECK(latitude_field_idx.has_value());
  THROW_CHECK(latitude_confidence_idx.has_value());

  return {*up_field_idx,
          *up_confidence_idx,
          *latitude_field_idx,
          *latitude_confidence_idx};
}

class GeoCalibImpl : public GeoCalib {
 public:
  explicit GeoCalibImpl(const GeoCalibOptions& options)
      : options_(options),
        model_(options.model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options_.Check());
    output_indices_ = CheckONNXSignatureAndGetOutputIndices(model_);
  }

  PerspectiveField PredictPerspectiveField(
      const Bitmap& bitmap) const override {
    Bitmap rgb_bitmap;
    const Bitmap* input_bitmap = &bitmap;
    if (!bitmap.IsRGB()) {
      rgb_bitmap = bitmap.CloneAsRGB();
      input_bitmap = &rgb_bitmap;
    }

    GeoCalibInput input = PrepareGeoCalibInput(
        *input_bitmap, options_.image_size, options_.force_square);

    const std::vector<int64_t> input_shape = {1, 3, input.height, input.width};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(CreateONNXTensor(input.data, input_shape));

    const std::vector<Ort::Value> outputs = model_.Run(input_tensors);

    const float* up_data =
        outputs[output_indices_.up_field_idx].GetTensorData<float>();
    const float* up_conf_data =
        outputs[output_indices_.up_confidence_idx].GetTensorData<float>();
    const float* lat_data =
        outputs[output_indices_.latitude_field_idx].GetTensorData<float>();
    const float* lat_conf_data =
        outputs[output_indices_.latitude_confidence_idx].GetTensorData<float>();

    const int width = input.width;
    const int height = input.height;
    const Eigen::Index num_pixels = static_cast<Eigen::Index>(width) * height;

    PerspectiveField field;
    field.width = width;
    field.height = height;
    field.points2D_in_img.resize(num_pixels, 2);
    field.up_in_img.resize(num_pixels, 2);
    field.latitude.resize(num_pixels);
    field.up_confidence.resize(num_pixels);
    field.latitude_confidence.resize(num_pixels);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const Eigen::Index idx = static_cast<Eigen::Index>(y) * width + x;
        field.points2D_in_img.row(idx) =
            input.ImgToOrig(Eigen::Vector2d(x + 0.5, y + 0.5));
        const Eigen::Vector2d up_net(up_data[idx], up_data[num_pixels + idx]);
        field.up_in_img.row(idx) = input.UpToOrig(up_net);
        field.latitude(idx) = static_cast<double>(lat_data[idx]);
        field.up_confidence(idx) = static_cast<double>(up_conf_data[idx]);
        field.latitude_confidence(idx) =
            static_cast<double>(lat_conf_data[idx]);
      }
    }

    return field;
  }

  FittedPerspectiveFields Calibrate(const Bitmap& bitmap,
                                    Camera* camera,
                                    const bool refine_camera) const override {
    THROW_CHECK_NOTNULL(camera);
    const PerspectiveField field = PredictPerspectiveField(bitmap);
    return FitPerspectiveField(options_.fitting, field, camera, refine_camera);
  }

 private:
  GeoCalibOptions options_;
  ONNXModel model_;
  GeoCalibOutputIndices output_indices_;
};

#endif  // COLMAP_ONNX_ENABLED

}  // namespace

bool GeoCalibOptions::Check() const {
  CHECK_OPTION_GE(image_size, kEdgeDivisibleBy);
  CHECK_OPTION(fitting.Check());
  return true;
}

Eigen::Vector2d GeoCalibInput::ImgToOrig(const Eigen::Vector2d& point) const {
  return (point - shift_xy).cwiseQuotient(scale_xy);
}

Eigen::Vector2d GeoCalibInput::UpToOrig(const Eigen::Vector2d& up) const {
  const Eigen::Vector2d scaled = up.cwiseQuotient(scale_xy);
  const double norm = scaled.norm();
  if (norm < 1e-12) {
    return up;
  }
  return scaled / norm;
}

GeoCalibInput PrepareGeoCalibInput(const Bitmap& bitmap,
                                   const int image_size,
                                   const bool force_square) {
  THROW_CHECK(bitmap.IsRGB());
  THROW_CHECK_GT(bitmap.Width(), 0);
  THROW_CHECK_GT(bitmap.Height(), 0);
  THROW_CHECK_GE(image_size, kEdgeDivisibleBy);

  const int orig_width = bitmap.Width();
  const int orig_height = bitmap.Height();

  // 1. Resize the shorter edge to `image_size`, preserving aspect ratio.
  const double scale =
      static_cast<double>(image_size) / std::min(orig_width, orig_height);
  const int resized_width = std::max(
      kEdgeDivisibleBy, static_cast<int>(std::lround(orig_width * scale)));
  const int resized_height = std::max(
      kEdgeDivisibleBy, static_cast<int>(std::lround(orig_height * scale)));

  Bitmap image = bitmap.Clone();
  image.Rescale(resized_width, resized_height);

  GeoCalibInput input;
  input.scale_xy =
      Eigen::Vector2d(static_cast<double>(resized_width) / orig_width,
                      static_cast<double>(resized_height) / orig_height);

  // 2. Center-crop to dimensions divisible by 32 (or square if force_square).
  int crop_width = (resized_width / kEdgeDivisibleBy) * kEdgeDivisibleBy;
  int crop_height = (resized_height / kEdgeDivisibleBy) * kEdgeDivisibleBy;
  if (force_square) {
    const int side = std::min(crop_width, crop_height);
    crop_width = side;
    crop_height = side;
  }
  const int crop_left = (resized_width - crop_width) / 2;
  const int crop_top = (resized_height - crop_height) / 2;
  input.shift_xy = Eigen::Vector2d(-crop_left, -crop_top);

  if (crop_left != 0 || crop_top != 0 || crop_width != resized_width ||
      crop_height != resized_height) {
    image.Crop(crop_left, crop_top, crop_width, crop_height);
  }

  input.width = crop_width;
  input.height = crop_height;

  // 3. Convert to row-major [C, H, W] float tensor in [0, 1].
  const int num_pixels = crop_width * crop_height;
  input.data.resize(3 * num_pixels);
  const std::vector<uint8_t>& raw_data = image.RowMajorData();
  const int pitch = image.Pitch();
  constexpr float kNorm = 1.0f / 255.0f;
  for (int y = 0; y < crop_height; ++y) {
    for (int x = 0; x < crop_width; ++x) {
      for (int c = 0; c < 3; ++c) {
        input.data[c * num_pixels + y * crop_width + x] =
            kNorm * raw_data[y * pitch + 3 * x + c];
      }
    }
  }

  return input;
}

std::unique_ptr<GeoCalib> GeoCalib::Create(const GeoCalibOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  THROW_CHECK(options.Check());
  return std::make_unique<GeoCalibImpl>(options);
#else
  throw std::runtime_error("GeoCalib requires ONNX support.");
#endif
}

}  // namespace colmap
