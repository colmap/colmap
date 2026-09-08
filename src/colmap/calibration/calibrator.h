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

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/enum_utils.h"

#include <memory>
#include <string>
#include <vector>

namespace colmap {

struct AnyCalibCalibrationOptions;

// Backend for learned single-image camera calibration. Additional backends
// (e.g. GeoCalib) plug in here without changing the pipeline or CLI.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(CameraCalibratorType, 0, ANYCALIB);

// Whether the intrinsics are numerically well behaved: finite parameters,
// positive focal lengths, and a projection that round-trips over the image.
// Note that this says nothing about whether the intrinsics are plausible; see
// `Camera::HasBogusParams` for that.
bool IsValidCalibration(const Camera& camera);

// Aggregate per-image fitted parameters for the given target model into a
// single camera by taking the coefficient-wise median. Because the median of
// individually valid calibrations is not itself guaranteed to be valid, the
// aggregate is re-validated and, if it fails, replaced by the single-image
// calibration closest to it. Sets the model, parameters, and
// `has_prior_focal_length` on success. Returns false if `params_list` is empty
// or no valid calibration is found, leaving `camera` unmodified. The
// dimensions of `camera` must be those of the images it was calibrated from.
bool AggregateCameraCalibrations(
    CameraModelId model_id,
    const std::vector<std::vector<double>>& params_list,
    Camera* camera);

struct CameraCalibrationTypeOptions {
  CameraCalibrationTypeOptions();

  std::shared_ptr<AnyCalibCalibrationOptions> anycalib;

  CameraCalibrationTypeOptions(const CameraCalibrationTypeOptions& other);
  CameraCalibrationTypeOptions& operator=(
      const CameraCalibrationTypeOptions& other);
  CameraCalibrationTypeOptions(CameraCalibrationTypeOptions&& other) = default;
  CameraCalibrationTypeOptions& operator=(
      CameraCalibrationTypeOptions&& other) = default;
};

struct CameraCalibrationOptions : public CameraCalibrationTypeOptions {
  explicit CameraCalibrationOptions(
      CameraCalibratorType type = CameraCalibratorType::ANYCALIB);

  CameraCalibratorType type = CameraCalibratorType::ANYCALIB;

  // Target COLMAP camera model for the fitted intrinsics. Any perspective
  // model is supported; the fitting refines that model's parameters directly.
  std::string camera_model = "SIMPLE_RADIAL";

  // Number of threads for calibration.
  int num_threads = -1;

  // Whether to use the GPU for neural-network inference.
#if defined(COLMAP_GPU_ENABLED) || defined(COLMAP_COREML_ENABLED)
  bool use_gpu = true;
#else
  bool use_gpu = false;
#endif

  // Index of the GPU used for inference. Only a single GPU is supported.
  std::string gpu_index = "-1";

  // Plausibility bounds on the calibrated intrinsics, matching the defaults of
  // the incremental mapper. A learned prediction can be perfectly
  // self-consistent and still be far off, in which case the existing (e.g.
  // EXIF) intrinsics are kept instead. Focal length ratios are relative to the
  // maximum image dimension and correspond to opening angles of ~130 and ~5
  // degrees.
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  // Maximum absolute value of any distortion parameter. NOTE: this is a single
  // bound for all coefficients of the target model, so high-order models with
  // legitimately large coefficients (e.g. the rational denominator terms k4,
  // k5, k6 of FULL_OPENCV) may require a larger value.
  double max_extra_param = 1.0;

  bool Check() const;
};

// Abstract single-image camera calibrator: estimates intrinsics for one image,
// replacing e.g. EXIF-based initialization with a learned prediction.
class CameraCalibrator {
 public:
  virtual ~CameraCalibrator() = default;

  // Create the calibrator backend selected by `options.type`. Throws if the
  // options are invalid, the type is unknown, or the backend cannot be created
  // (e.g. the network model cannot be loaded); never returns nullptr.
  static std::unique_ptr<CameraCalibrator> Create(
      const CameraCalibrationOptions& options);

  // Calibrate the camera for `bitmap`, optionally using an existing focal
  // length prior in `camera` and gravity in `pose_prior`, and write the
  // target-model intrinsics at full image resolution into `camera` (model,
  // dimensions, params, and focal length prior flag). Returns false if
  // calibration fails, in which case `camera` is left unmodified.
  virtual bool Calibrate(const Bitmap& bitmap,
                         Camera* camera,
                         const PosePrior& pose_prior = PosePrior()) const = 0;
};

}  // namespace colmap
