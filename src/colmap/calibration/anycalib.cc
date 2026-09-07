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

// Separate fixed landscape and portrait exports preserve more image content,
// but the landscape export consistently reduced reconstruction AUC relative
// to this square model on ETH3D DSLR. That dataset did not exercise the
// portrait export, so keep the empirically stronger single square model.
Bitmap CenterCropToSquare(const Bitmap& bitmap) {
  THROW_CHECK(bitmap.IsRGB());
  const int width = bitmap.Width();
  const int height = bitmap.Height();
  const int side = std::min(width, height);
  if (width == height) {
    return bitmap.Clone();
  }
  const int left = (width - side) / 2;
  const int top = (height - side) / 2;
  Bitmap cropped(side, side, /*as_rgb=*/true);
  const std::vector<uint8_t>& src = bitmap.RowMajorData();
  std::vector<uint8_t>& dst = cropped.RowMajorData();
  const int src_pitch = bitmap.Pitch();
  const int dst_pitch = cropped.Pitch();
  for (int y = 0; y < side; ++y) {
    std::memcpy(dst.data() + y * dst_pitch,
                src.data() + (top + y) * src_pitch + left * 3,
                dst_pitch);
  }
  return cropped;
}

#ifdef COLMAP_ONNX_ENABLED

class AnyCalibCalibrator : public CameraCalibrator {
 public:
  explicit AnyCalibCalibrator(const CameraCalibrationOptions& options)
      : options_(options),
        model_(options.anycalib.model_path,
               options.num_threads,
               options.use_gpu,
               options.gpu_index) {
    THROW_CHECK(options_.Check());
    THROW_CHECK_EQ(model_.input_shapes().size(), 1);
    ThrowCheckONNXNode(model_.input_names()[0],
                       "image",
                       model_.input_shapes()[0],
                       {1, 3, kAnyCalibInputSize, kAnyCalibInputSize});
    THROW_CHECK_EQ(model_.output_shapes().size(), 2);
    for (size_t i = 0; i < model_.output_names().size(); ++i) {
      const std::string_view name = model_.output_names()[i];
      const auto& shape = model_.output_shapes()[i];
      if (name == "rays") {
        ThrowCheckONNXNode(name,
                           "rays",
                           shape,
                           {1, kAnyCalibInputSize * kAnyCalibInputSize, 3});
        rays_idx_ = i;
      } else if (name == "tangent_coords") {
        ThrowCheckONNXNode(name,
                           "tangent_coords",
                           shape,
                           {1, kAnyCalibInputSize * kAnyCalibInputSize, 2});
      } else {
        LOG(FATAL_THROW) << "Unexpected AnyCalib output: " << name;
      }
    }
  }

  bool Calibrate(const Bitmap& bitmap,
                 Camera* camera,
                 const PosePrior& pose_prior) const override {
    THROW_CHECK_NOTNULL(camera);
    THROW_CHECK(bitmap.IsRGB());

    const CameraModelId model_id = CameraModelNameToId(options_.camera_model);

    AnyCalibInput input = PrepareAnyCalibInput(bitmap, pose_prior);
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
        const size_t i = static_cast<size_t>(y * size + x);
        img_points.push_back(
            input.ImagePointToOriginal(Eigen::Vector2d(x + 0.5, y + 0.5)));
        cam_rays.push_back(input.CameraRayToOriginal(Eigen::Vector3d(
            rays_data[3 * i], rays_data[3 * i + 1], rays_data[3 * i + 2])));
      }
    }

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
  ONNXModel model_;
  size_t rays_idx_ = 0;
};

#endif

}  // namespace

bool AnyCalibCalibrationOptions::Check() const {
  // NOTE: `model_path` is intentionally not validated here: like the
  // feature extractor model paths, it may be empty until set, and a missing
  // file surfaces as an exception when the calibrator is created.
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
                                   const PosePrior& pose_prior) {
  THROW_CHECK(bitmap.IsRGB());
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
  const int width = image.Width();
  const int height = image.Height();
  if (height < kSize || width < kSize) {
    const double s = std::max(static_cast<double>(kSize) / height,
                              static_cast<double>(kSize) / width);
    const int new_width = static_cast<int>(width * s);
    const int new_height = static_cast<int>(height * s);
    image.Rescale(new_width, new_height);
    scale_xy.x() = static_cast<double>(image.Width()) / width;
    scale_xy.y() = static_cast<double>(image.Height()) / height;
  }

  // Center-crop to square (set_im_size step 2 with target aspect ratio 1).
  // NOTE: upstream bicubic resampling is replaced with COLMAP's bilinear
  // `Bitmap::Rescale`; the network is robust to the resampling kernel.
  const int w = image.Width();
  const int h = image.Height();
  if (w > h) {
    const int crop_w = w - h;
    shift_xy.x() = -(crop_w / 2);
  } else {
    const int crop_h = h - w;
    shift_xy.y() = -(crop_h / 2);
  }
  image = CenterCropToSquare(image);

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

std::unique_ptr<CameraCalibrator> CreateAnyCalibCalibrator(
    const CameraCalibrationOptions& options) {
#ifdef COLMAP_ONNX_ENABLED
  return std::make_unique<AnyCalibCalibrator>(options);
#else
  throw std::runtime_error("AnyCalib calibration requires ONNX support.");
#endif
}

}  // namespace colmap
