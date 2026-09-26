// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/anycalib.h"

#include "colmap/calibration/calibrator.h"
#include "colmap/util/logging.h"
#include "colmap/util/onnx.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace colmap {
namespace {

// Fixed square network input size: round(sqrt(102400) / 14) * 14, the square
// training resolution of AnyCalib. See `scripts/anycalib/export_onnx.py` for
// why dynamic input sizes are not supported.
constexpr int kAnyCalibInputSize = 322;

// Upsample small images in place, preserving the aspect ratio, and update
// `scale_xy` with the applied per-axis rescaling.
void UpsampleToMinSize(Bitmap& image, int min_size, Eigen::Vector2d& scale_xy) {
  const int width = image.Width();
  const int height = image.Height();
  if (width >= min_size && height >= min_size) {
    return;
  }
  const double s = std::max(static_cast<double>(min_size) / height,
                            static_cast<double>(min_size) / width);
  const int new_width = std::lround(width * s);
  const int new_height = std::lround(height * s);
  image.Rescale(new_width, new_height);
  scale_xy.x() = static_cast<double>(image.Width()) / width;
  scale_xy.y() = static_cast<double>(image.Height()) / height;
}

// Origin (left, top) of the centered square crop of a `width` x `height`
// image. The integer divisions center the crop up to one pixel.
Eigen::Vector2i CenterCropOrigin(int width, int height) {
  const int side = std::min(width, height);
  return Eigen::Vector2i((width - side) / 2, (height - side) / 2);
}

#ifdef COLMAP_ONNX_ENABLED

// Validate the AnyCalib input/output signature and return the index of the
// `rays` output. Throws on a signature mismatch or if `rays` is missing.
size_t CheckONNXSignatureAndGetRayIndex(const ONNXModel& model) {
  THROW_CHECK_EQ(model.input_shapes().size(), 1);
  THROW_CHECK_EQ(model.input_element_types().size(), 1);
  ThrowCheckONNXNode(model.input_names()[0],
                     "image",
                     model.input_shapes()[0],
                     {1, 3, kAnyCalibInputSize, kAnyCalibInputSize});
  ThrowCheckONNXElementType(model.input_names()[0],
                            model.input_element_types()[0],
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  THROW_CHECK_EQ(model.output_shapes().size(), 2);
  THROW_CHECK_EQ(model.output_element_types().size(), 2);
  std::optional<size_t> rays_idx;
  for (size_t i = 0; i < model.output_names().size(); ++i) {
    const std::string_view name = model.output_names()[i];
    const auto& shape = model.output_shapes()[i];
    // The outputs are read as float below; validate the element type with
    // the same rigor as the shapes rather than relying on the ORT version's
    // `GetTensorData` checking.
    ThrowCheckONNXElementType(name,
                              model.output_element_types()[i],
                              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (name == "rays") {
      ThrowCheckONNXNode(
          name, "rays", shape, {1, kAnyCalibInputSize * kAnyCalibInputSize, 3});
      rays_idx = i;
    } else if (name == "tangent_coords") {
      ThrowCheckONNXNode(name,
                         "tangent_coords",
                         shape,
                         {1, kAnyCalibInputSize * kAnyCalibInputSize, 2});
    } else {
      LOG(FATAL_THROW) << "Unexpected AnyCalib output: " << name;
    }
  }
  THROW_CHECK(rays_idx.has_value()) << "AnyCalib outputs must contain \"rays\"";
  return *rays_idx;
}

class AnyCalibCalibrator : public MonocularCalibrator {
 public:
  explicit AnyCalibCalibrator(const MonocularCalibrationOptions& options)
      : options_(options),
        model_(options.anycalib->model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options_.Check());
    rays_idx_ = CheckONNXSignatureAndGetRayIndex(model_);
  }

  bool Calibrate(const Bitmap& bitmap,
                 Camera* camera,
                 PosePrior* pose_prior) const override {
    THROW_CHECK_NOTNULL(camera);
    THROW_CHECK_NOTNULL(pose_prior);
    // The pose prior is populated independently of whether intrinsics
    // calibration succeeds below. EXIF gravity filled here also drives the
    // upright rotation of the network input.
    SetPosePriorFromExif(bitmap, pose_prior);
    // Feature extraction may read grayscale images; the network needs RGB.
    Bitmap rgb_bitmap;
    const Bitmap* input_bitmap = &bitmap;
    if (!bitmap.IsRGB()) {
      rgb_bitmap = bitmap.CloneAsRGB();
      input_bitmap = &rgb_bitmap;
    }

    // An empty target model preserves the camera's existing model. Fitting a
    // non-perspective model fails gracefully below (`FitCameraFromRays`
    // rejects it), leaving the camera unmodified.
    const CameraModelId model_id =
        options_.camera_model.empty()
            ? camera->model_id
            : CameraModelNameToId(options_.camera_model);
    // The camera-model switches throw on invalid models instead of returning
    // false, so reject them explicitly before running inference.
    if (model_id == CameraModelId::kInvalid) {
      LOG(WARNING) << "Cannot calibrate camera with invalid model";
      return false;
    }

    AnyCalibInput input = PrepareAnyCalibInput(*input_bitmap, *pose_prior);
    const std::vector<int64_t> input_shape(
        {1, 3, kAnyCalibInputSize, kAnyCalibInputSize});
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        input.data.data(),
        input.data.size(),
        input_shape.data(),
        input_shape.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    const std::vector<Ort::Value> outputs = model_.Run(input_tensors);

    const float* rays_data = outputs[rays_idx_].GetTensorData<float>();
    const int size = kAnyCalibInputSize;
    std::vector<Eigen::Vector2d> img_points;
    std::vector<Eigen::Vector3d> cam_rays;
    img_points.reserve(size * size);
    cam_rays.reserve(size * size);
    for (int y = 0; y < size; ++y) {
      for (int x = 0; x < size; ++x) {
        const size_t i = static_cast<size_t>(y) * size + x;
        // The network predicts one ray per pixel; reference pixel centers.
        img_points.push_back(
            input.ImgToOrig(Eigen::Vector2d(x + 0.5, y + 0.5)));
        cam_rays.push_back(input.CamToOrig(Eigen::Vector3d(
            rays_data[3 * i], rays_data[3 * i + 1], rays_data[3 * i + 2])));
      }
    }

    std::vector<double> prior_focal_lengths;
    if (options_.anycalib->fitting.prior_focal_length_weight > 0.0 &&
        camera->IsPerspective()) {
      if (camera->has_prior_focal_length) {
        if (CameraModelFocalLengthIdxs(model_id).size() == 1) {
          prior_focal_lengths.push_back(camera->MeanFocalLength());
        } else {
          prior_focal_lengths = {camera->FocalLengthX(),
                                 camera->FocalLengthY()};
        }
      } else {
        // Newly created feature-extraction cameras intentionally contain the
        // default focal length until this calibration stage. Recover the EXIF
        // prior directly from the bitmap so that prior_focal_length_weight is
        // effective for the integrated AnyCalib path too.
        if (const std::optional<double> focal_length = bitmap.ExifFocalLength();
            focal_length.has_value()) {
          prior_focal_lengths.assign(
              CameraModelFocalLengthIdxs(model_id).size(),
              focal_length.value());
        }
      }
    }

    const FittedCamera fitted = FitCameraFromRays(model_id,
                                                  img_points,
                                                  cam_rays,
                                                  options_.anycalib->fitting,
                                                  prior_focal_lengths);
    if (!fitted.success) {
      return false;
    }
    Camera calibrated;
    calibrated.camera_id = camera->camera_id;
    calibrated.model_id = model_id;
    calibrated.width = bitmap.Width();
    calibrated.height = bitmap.Height();
    calibrated.params = fitted.params;
    calibrated.has_prior_focal_length = true;
    if (!IsValidCalibration(calibrated)) {
      return false;
    }
    // Reject implausible predictions, which are numerically well behaved but
    // far from the true intrinsics. NOTE: this is only checked per image,
    // because the coefficient-wise median of calibrations that all satisfy
    // these per-coefficient bounds satisfies them as well.
    if (calibrated.HasBogusParams(options_.min_focal_length_ratio,
                                  options_.max_focal_length_ratio,
                                  options_.max_extra_param)) {
      // Log the rejected parameters: a systematic rejection here (e.g. by
      // `max_extra_param` for high-order models) would otherwise silently fall
      // back to the existing intrinsics for every image.
      LOG(WARNING) << "Rejecting implausible learned calibration for "
                   << calibrated.ModelName()
                   << " camera: " << calibrated.ParamsToString();
      return false;
    }
    *camera = calibrated;
    return true;
  }

 private:
  MonocularCalibrationOptions options_;
  ONNXModel model_;
  size_t rays_idx_ = 0;
};

#endif

}  // namespace

bool AnyCalibOptions::Check() const {
  // NOTE: `model_path` is intentionally not validated here: like the
  // feature extractor model paths, it may be empty until set, and a missing
  // file surfaces as an exception when the calibrator is created.
  CHECK_OPTION(fitting.Check());
  return true;
}

Eigen::Vector2d AnyCalibInput::ImgToOrig(const Eigen::Vector2d& point) const {
  Eigen::Vector2d upright = (point - shift_xy).cwiseQuotient(scale_xy);
  switch (image_rot90) {
    case 0:
      return upright;
    case 1:
      return Eigen::Vector2d(upright_height - upright.y(), upright.x());
    case 2:
      return Eigen::Vector2d(upright_width - upright.x(),
                             upright_height - upright.y());
    case 3:
      return Eigen::Vector2d(upright.y(), upright_width - upright.x());
    default:
      break;
  }
  LOG(FATAL_THROW) << "Invalid image rotation: " << image_rot90;
  return Eigen::Vector2d::Zero();
}

Eigen::Vector3d AnyCalibInput::CamToOrig(const Eigen::Vector3d& ray) const {
  switch (image_rot90) {
    case 0:
      return ray;
    case 1:
      return Eigen::Vector3d(-ray.y(), ray.x(), ray.z());
    case 2:
      return Eigen::Vector3d(-ray.x(), -ray.y(), ray.z());
    case 3:
      return Eigen::Vector3d(ray.y(), -ray.x(), ray.z());
    default:
      break;
  }
  LOG(FATAL_THROW) << "Invalid image rotation: " << image_rot90;
  return Eigen::Vector3d::Zero();
}

AnyCalibInput PrepareAnyCalibInput(const Bitmap& bitmap,
                                   const PosePrior& pose_prior) {
  THROW_CHECK(bitmap.IsRGB());
  THROW_CHECK_GT(bitmap.Width(), 0);
  THROW_CHECK_GT(bitmap.Height(), 0);
  constexpr int kSize = kAnyCalibInputSize;
  AnyCalibInput input;
  Eigen::Vector2d& scale_xy = input.scale_xy;
  Eigen::Vector2d& shift_xy = input.shift_xy;

  Bitmap image = bitmap.Clone();
  input.image_rot90 =
      pose_prior.HasGravity() ? ComputeRot90FromGravity(pose_prior.gravity) : 0;
  if (input.image_rot90 != 0) {
    image.Rot90(input.image_rot90);
  }
  input.upright_width = image.Width();
  input.upright_height = image.Height();

  // Upsample small images, preserving the aspect ratio (set_im_size step 1).
  UpsampleToMinSize(image, kSize, scale_xy);

  // Center-crop to square (set_im_size step 2 with target aspect ratio 1).
  // Separate fixed landscape and portrait exports preserve more image
  // content, but the landscape export consistently reduced reconstruction AUC
  // relative to this square model on ETH3D DSLR. That dataset did not
  // exercise the portrait export, so keep the empirically stronger single
  // square model.
  // NOTE: upstream bicubic resampling is replaced with COLMAP's bilinear
  // `Bitmap::Rescale`; the network is robust to the resampling kernel.
  const Eigen::Vector2i crop_origin =
      CenterCropOrigin(image.Width(), image.Height());
  const int side = std::min(image.Width(), image.Height());
  // Cropping translates image coordinates by the negative crop origin.
  shift_xy = Eigen::Vector2d(-crop_origin.x(), -crop_origin.y());
  image.Crop(crop_origin.x(), crop_origin.y(), side, side);

  // Downsample to the network resolution (set_im_size step 3).
  const Eigen::Vector2d scale_2(static_cast<double>(kSize) / image.Width(),
                                static_cast<double>(kSize) / image.Height());
  image.Rescale(kSize, kSize);
  scale_xy = scale_xy.cwiseProduct(scale_2);
  shift_xy = shift_xy.cwiseProduct(scale_2);

  // Convert to row-major [C, H, W] float tensor, normalized to [0, 1].
  const int num_pixels = kSize * kSize;
  input.data.resize(3 * num_pixels);
  const std::vector<uint8_t>& data = image.RowMajorData();
  const int pitch = image.Pitch();
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      for (int c = 0; c < 3; ++c) {
        constexpr float kImageNormalization = 1.0f / 255.0f;
        input.data[c * num_pixels + y * kSize + x] =
            kImageNormalization * data[y * pitch + 3 * x + c];
      }
    }
  }
  return input;
}

std::unique_ptr<MonocularCalibrator> CreateAnyCalibCalibrator(
    const MonocularCalibrationOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  // Validate before constructing: the constructor dereferences
  // `options.anycalib` in its initializer list and loads the network model, so
  // invalid options must be rejected before any of that happens. The
  // backend-specific settings are validated explicitly, as `Check()` only
  // validates them when this backend is the selected `type`.
  THROW_CHECK(options.Check());
  THROW_CHECK(options.anycalib != nullptr);
  THROW_CHECK(options.anycalib->Check());
  return std::make_unique<AnyCalibCalibrator>(options);
#else
  throw std::runtime_error("AnyCalib calibration requires ONNX support.");
#endif
}

}  // namespace colmap
