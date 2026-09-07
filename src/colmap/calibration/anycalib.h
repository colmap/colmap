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

#pragma once

#include "colmap/calibration/ray_fitting.h"
#include "colmap/calibration/resources.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/sensor/bitmap.h"

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

class CameraCalibrator;
struct CameraCalibrationOptions;

// Fixed 3:2 network input sizes near AnyCalib's 102400-pixel training
// resolution, with dimensions divisible by the DINOv2 patch size (14).
constexpr int kAnyCalibLandscapeWidth = 392;
constexpr int kAnyCalibLandscapeHeight = 266;
constexpr int kAnyCalibPortraitWidth = kAnyCalibLandscapeHeight;
constexpr int kAnyCalibPortraitHeight = kAnyCalibLandscapeWidth;

struct AnyCalibCalibrationOptions {
  // Paths or download URIs of the exported AnyCalib ONNX models.
  std::string landscape_model_path = kDefaultAnyCalibGenLandscapeUri;
  std::string portrait_model_path = kDefaultAnyCalibGenPortraitUri;

  // Fitting options shared with future ray-based backends.
  RayFittingOptions fitting;

  bool Check() const;
};

// Network input plus the transform from the original image to the upright,
// cropped, and resized network image, mirroring `AnyCalib.set_im_size`.
struct AnyCalibInput {
  // RGB in [0, 1], row-major [C, H, W].
  std::vector<float> data;
  // Per-axis scale and shift of the intrinsics digitizing transform.
  Eigen::Vector2d scale_xy = Eigen::Vector2d::Ones();
  Eigen::Vector2d shift_xy = Eigen::Vector2d::Zero();
  int width = 0;
  int height = 0;
  int upright_width = 0;
  int upright_height = 0;
  int image_rot90 = 0;

  // Map network-image coordinates and camera rays back to the original image
  // orientation and resolution.
  Eigen::Vector2d ImagePointToOriginal(const Eigen::Vector2d& point) const;
  Eigen::Vector3d CameraRayToOriginal(const Eigen::Vector3d& ray) const;
};

AnyCalibInput PrepareAnyCalibInput(const Bitmap& bitmap,
                                   int target_width,
                                   int target_height,
                                   const PosePrior& pose_prior = PosePrior());

std::unique_ptr<CameraCalibrator> CreateAnyCalibCalibrator(
    const CameraCalibrationOptions& options);

}  // namespace colmap
