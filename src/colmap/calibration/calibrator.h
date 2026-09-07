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

#include "colmap/calibration/anycalib.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/enum_utils.h"

#include <memory>
#include <string>
#include <vector>

namespace colmap {

// Backend for learned single-image camera calibration. Additional backends
// (e.g. GeoCalib) plug in here without changing the pipeline or CLI.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(CameraCalibratorType, 0, ANYCALIB);

// Aggregate per-image fitted parameters into a single camera by taking the
// coefficient-wise median. Sets `has_prior_focal_length` on success. Returns
// false if `params_list` is empty, leaving `camera` unmodified.
bool AggregateCameraCalibrations(
    const std::vector<std::vector<double>>& params_list, Camera* camera);

struct CameraCalibrationOptions {
  CameraCalibratorType type = CameraCalibratorType::ANYCALIB;

  // Target COLMAP camera model for the fitted intrinsics. Any perspective
  // model is supported; the fitting refines that model's parameters directly.
  std::string camera_model = "SIMPLE_RADIAL";

  // Number of threads for calibration.
  int num_threads = -1;

  // Whether to use the GPU for neural-network inference.
#ifdef COLMAP_GPU_ENABLED
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif

  // Index of the GPU used for inference. Only a single GPU is supported.
  std::string gpu_index = "-1";

  AnyCalibCalibrationOptions anycalib;

  bool Check() const;
};

// Abstract single-image camera calibrator: estimates intrinsics for one image,
// replacing e.g. EXIF-based initialization with a learned prediction.
class CameraCalibrator {
 public:
  virtual ~CameraCalibrator() = default;

  static std::unique_ptr<CameraCalibrator> Create(
      const CameraCalibrationOptions& options);

  // Calibrate the camera for `bitmap`, optionally using an existing focal
  // length prior in `camera`, and write the target-model intrinsics at full
  // image resolution into `camera` (model, dimensions, params, and focal
  // length prior flag). Returns false if calibration fails, in which case
  // `camera` is left unmodified.
  virtual bool Calibrate(const Bitmap& bitmap, Camera* camera) const = 0;
};

}  // namespace colmap
