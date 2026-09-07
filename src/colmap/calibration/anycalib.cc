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

#include "colmap/calibration/anycalib.h"

#include "colmap/calibration/calibrator.h"
#include "colmap/util/logging.h"
#include "colmap/util/onnx.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace colmap {
namespace {

Bitmap CenterCropToAspectRatio(const Bitmap& bitmap,
                               int target_width,
                               int target_height,
                               Eigen::Vector2d* shift_xy) {
  THROW_CHECK(bitmap.IsRGB());
  THROW_CHECK_NOTNULL(shift_xy);
  const int width = bitmap.Width();
  const int height = bitmap.Height();
  const double target_aspect =
      static_cast<double>(target_width) / target_height;
  int crop_width = width;
  int crop_height = height;
  int left = 0;
  int top = 0;
  if (static_cast<double>(width) / height > target_aspect) {
    crop_width = static_cast<int>(std::round(height * target_aspect));
    left = (width - crop_width) / 2;
    shift_xy->x() = -left;
  } else {
    crop_height = static_cast<int>(std::round(width / target_aspect));
    top = (height - crop_height) / 2;
    shift_xy->y() = -top;
  }
  if (crop_width == width && crop_height == height) {
    return bitmap.Clone();
  }
  Bitmap cropped(crop_width, crop_height, /*as_rgb=*/true);
  const std::vector<uint8_t>& src = bitmap.RowMajorData();
  std::vector<uint8_t>& dst = cropped.RowMajorData();
  const int src_pitch = bitmap.Pitch();
  const int dst_pitch = cropped.Pitch();
  for (int y = 0; y < crop_height; ++y) {
    std::memcpy(dst.data() + y * dst_pitch,
                src.data() + (top + y) * src_pitch + left * 3,
                dst_pitch);
  }
  return cropped;
}

#ifdef COLMAP_ONNX_ENABLED

bool UseLandscapeModel(int width, int height) {
  const double aspect = static_cast<double>(width) / height;
  const double landscape_aspect =
      static_cast<double>(kAnyCalibLandscapeWidth) / kAnyCalibLandscapeHeight;
  const double portrait_aspect =
      static_cast<double>(kAnyCalibPortraitWidth) / kAnyCalibPortraitHeight;
  return std::abs(std::log(aspect / landscape_aspect)) <=
         std::abs(std::log(aspect / portrait_aspect));
}

struct AnyCalibModel {
  AnyCalibModel(const std::string& path,
                int width,
                int height,
                const CameraCalibrationOptions& options)
      : model(path, options.num_threads, options.use_gpu, options.gpu_index),
        width(width),
        height(height) {
    THROW_CHECK_EQ(model.input_shapes().size(), 1);
    ThrowCheckONNXNode(model.input_names()[0],
                       "image",
                       model.input_shapes()[0],
                       {1, 3, height, width});
    THROW_CHECK_EQ(model.output_shapes().size(), 2);
    for (size_t i = 0; i < model.output_names().size(); ++i) {
      const std::string_view name = model.output_names()[i];
      const auto& shape = model.output_shapes()[i];
      if (name == "rays") {
        ThrowCheckONNXNode(name, "rays", shape, {1, width * height, 3});
        rays_idx = i;
      } else if (name == "tangent_coords") {
        ThrowCheckONNXNode(
            name, "tangent_coords", shape, {1, width * height, 2});
      } else {
        LOG(FATAL_THROW) << "Unexpected AnyCalib output: " << name;
      }
    }
  }

  ONNXModel model;
  int width;
  int height;
  size_t rays_idx = 0;
};

class AnyCalibCalibrator : public CameraCalibrator {
 public:
  explicit AnyCalibCalibrator(const CameraCalibrationOptions& options)
      : options_(options),
        landscape_model_(options.anycalib.landscape_model_path,
                         kAnyCalibLandscapeWidth,
                         kAnyCalibLandscapeHeight,
                         options),
        portrait_model_(options.anycalib.portrait_model_path,
                        kAnyCalibPortraitWidth,
                        kAnyCalibPortraitHeight,
                        options) {
    THROW_CHECK(options_.Check());
  }

  bool Calibrate(const Bitmap& bitmap,
                 Camera* camera,
                 const PosePrior& pose_prior) const override {
    THROW_CHECK_NOTNULL(camera);
    THROW_CHECK(bitmap.IsRGB());
    const int image_rot90 = pose_prior.HasGravity()
                                ? ComputeRot90FromGravity(pose_prior.gravity)
                                : 0;
    const int upright_width =
        image_rot90 % 2 == 0 ? bitmap.Width() : bitmap.Height();
    const int upright_height =
        image_rot90 % 2 == 0 ? bitmap.Height() : bitmap.Width();
    const AnyCalibModel& selected_model =
        UseLandscapeModel(upright_width, upright_height) ? landscape_model_
                                                         : portrait_model_;
    AnyCalibInput input = PrepareAnyCalibInput(
        bitmap, selected_model.width, selected_model.height, pose_prior);
    const std::vector<int64_t> input_shape(
        {1, 3, selected_model.height, selected_model.width});
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                   OrtMemType::OrtMemTypeCPU),
        input.data.data(),
        input.data.size(),
        input_shape.data(),
        input_shape.size());
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    const std::vector<Ort::Value> outputs =
        selected_model.model.Run(input_tensors);

    const float* rays_data =
        outputs[selected_model.rays_idx].GetTensorData<float>();
    const int num_pixels = selected_model.width * selected_model.height;
    std::vector<Eigen::Vector2d> img_points;
    std::vector<Eigen::Vector3d> cam_rays;
    img_points.reserve(num_pixels);
    cam_rays.reserve(num_pixels);
    for (int y = 0; y < selected_model.height; ++y) {
      for (int x = 0; x < selected_model.width; ++x) {
        const size_t i = static_cast<size_t>(y * selected_model.width + x);
        img_points.push_back(
            input.ImagePointToOriginal(Eigen::Vector2d(x + 0.5, y + 0.5)));
        cam_rays.push_back(input.CameraRayToOriginal(Eigen::Vector3d(
            rays_data[3 * i], rays_data[3 * i + 1], rays_data[3 * i + 2])));
      }
    }

    const CameraModelId model_id = CameraModelNameToId(options_.camera_model);
    std::vector<double> prior_focal_lengths;
    if (options_.anycalib.fitting.prior_focal_length_weight > 0.0 &&
        camera->has_prior_focal_length && camera->IsPerspective()) {
      if (CameraModelFocalLengthIdxs(model_id).size() == 1) {
        prior_focal_lengths.push_back(camera->MeanFocalLength());
      } else {
        prior_focal_lengths = {camera->FocalLengthX(), camera->FocalLengthY()};
      }
    }

    const FittedCamera fitted = FitCameraFromRays(model_id,
                                                  img_points,
                                                  cam_rays,
                                                  options_.anycalib.fitting,
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
    if (!calibrated.VerifyParams() ||
        !std::all_of(calibrated.params.begin(),
                     calibrated.params.end(),
                     [](const double param) { return std::isfinite(param); })) {
      return false;
    }
    for (const size_t idx : calibrated.FocalLengthIdxs()) {
      if (!std::isfinite(calibrated.params[idx]) ||
          calibrated.params[idx] <= 0) {
        return false;
      }
    }
    *camera = calibrated;
    return true;
  }

 private:
  CameraCalibrationOptions options_;
  AnyCalibModel landscape_model_;
  AnyCalibModel portrait_model_;
};

#endif

}  // namespace

bool AnyCalibCalibrationOptions::Check() const {
  // NOTE: Model paths are intentionally not validated here: like feature
  // extractor model paths, they may be empty until set. Missing files surface
  // as exceptions when the calibrator is created.
  CHECK_OPTION(fitting.Check());
  return true;
}

Eigen::Vector2d AnyCalibInput::ImagePointToOriginal(
    const Eigen::Vector2d& point) const {
  const Eigen::Vector2d upright = (point - shift_xy).cwiseQuotient(scale_xy);
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
  }
  LOG(FATAL_THROW) << "Invalid image rotation: " << image_rot90;
  return Eigen::Vector2d::Zero();
}

Eigen::Vector3d AnyCalibInput::CameraRayToOriginal(
    const Eigen::Vector3d& ray) const {
  switch (image_rot90) {
    case 0:
      return ray;
    case 1:
      return Eigen::Vector3d(-ray.y(), ray.x(), ray.z());
    case 2:
      return Eigen::Vector3d(-ray.x(), -ray.y(), ray.z());
    case 3:
      return Eigen::Vector3d(ray.y(), -ray.x(), ray.z());
  }
  LOG(FATAL_THROW) << "Invalid image rotation: " << image_rot90;
  return Eigen::Vector3d::Zero();
}

AnyCalibInput PrepareAnyCalibInput(const Bitmap& bitmap,
                                   int target_width,
                                   int target_height,
                                   const PosePrior& pose_prior) {
  THROW_CHECK(bitmap.IsRGB());
  THROW_CHECK_GT(target_width, 0);
  THROW_CHECK_GT(target_height, 0);
  AnyCalibInput input;
  input.width = target_width;
  input.height = target_height;
  input.image_rot90 =
      pose_prior.HasGravity() ? ComputeRot90FromGravity(pose_prior.gravity) : 0;

  Bitmap image = bitmap.Clone();
  if (input.image_rot90 != 0) {
    image.Rot90(input.image_rot90);
  }
  input.upright_width = image.Width();
  input.upright_height = image.Height();

  const int width = image.Width();
  const int height = image.Height();
  if (height < target_height || width < target_width) {
    const double scale = std::max(static_cast<double>(target_height) / height,
                                  static_cast<double>(target_width) / width);
    image.Rescale(static_cast<int>(width * scale),
                  static_cast<int>(height * scale));
    input.scale_xy.x() = static_cast<double>(image.Width()) / width;
    input.scale_xy.y() = static_cast<double>(image.Height()) / height;
  }

  Eigen::Vector2d crop_shift = Eigen::Vector2d::Zero();
  image =
      CenterCropToAspectRatio(image, target_width, target_height, &crop_shift);
  input.shift_xy += crop_shift;

  const Eigen::Vector2d resize_scale(
      static_cast<double>(target_width) / image.Width(),
      static_cast<double>(target_height) / image.Height());
  image.Rescale(target_width, target_height);
  input.scale_xy = input.scale_xy.cwiseProduct(resize_scale);
  input.shift_xy = input.shift_xy.cwiseProduct(resize_scale);

  const int num_pixels = target_width * target_height;
  input.data.resize(3 * num_pixels);
  const std::vector<uint8_t>& data = image.RowMajorData();
  const int pitch = image.Pitch();
  for (int y = 0; y < target_height; ++y) {
    for (int x = 0; x < target_width; ++x) {
      for (int c = 0; c < 3; ++c) {
        constexpr float kImageNormalization = 1.0f / 255.0f;
        input.data[c * num_pixels + y * target_width + x] =
            kImageNormalization * data[y * pitch + 3 * x + c];
      }
    }
  }
  return input;
}

std::unique_ptr<CameraCalibrator> CreateAnyCalibCalibrator(
    const CameraCalibrationOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<AnyCalibCalibrator>(options);
#else
  throw std::runtime_error("AnyCalib calibration requires ONNX support.");
#endif
}

}  // namespace colmap
