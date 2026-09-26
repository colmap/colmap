// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/camera.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/enum_utils.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace colmap {

struct AnyCalibOptions;

// Backend for monocular single-image camera calibration. Additional backends
// (e.g. GeoCalib) plug in here without changing the pipeline or CLI.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(MonocularCalibratorType, 0, EXIF, ANYCALIB);

// Whether the intrinsics are numerically well behaved: finite parameters,
// positive focal lengths, and a projection that round-trips over the image.
// Note that this says nothing about whether the intrinsics are plausible; see
// `Camera::HasBogusParams` for that.
bool IsValidCalibration(const Camera& camera);

// Aggregate per-image fitted parameters for the given target model into a
// single camera by taking the coefficient-wise median. Fitting once to the
// accumulated rays of all images would repeat the dense refinement over every
// image's rays and let a single bad image corrupt the shared estimate, while
// the median stays robust to individual failures. The refinement estimates no
// covariances, so uncertainty weighting is not available either. Because the
// median of individually valid calibrations is not itself guaranteed to be
// valid, the aggregate is re-validated and, if it fails, replaced by the
// single-image calibration closest to it. Joint fitting over the accumulated
// rays was evaluated as an alternative: it matches the median only with a
// tuned robust loss and fails on a single bad image without one. Sets the
// model, parameters, and `has_prior_focal_length` on success. Returns false
// if `params_list` is empty or no valid calibration is found, leaving
// `camera` unmodified. The dimensions of `camera` must be those of the images
// it was calibrated from.
bool AggregateMonocularCalibrations(
    CameraModelId model_id,
    const std::vector<std::vector<double>>& params_list,
    Camera* camera);

struct MonocularCalibrationTypeOptions {
  MonocularCalibrationTypeOptions();

  std::shared_ptr<AnyCalibOptions> anycalib;

  MonocularCalibrationTypeOptions(const MonocularCalibrationTypeOptions& other);
  MonocularCalibrationTypeOptions& operator=(
      const MonocularCalibrationTypeOptions& other);
  MonocularCalibrationTypeOptions(MonocularCalibrationTypeOptions&& other) =
      default;
  MonocularCalibrationTypeOptions& operator=(
      MonocularCalibrationTypeOptions&& other) = default;
};

struct MonocularCalibrationOptions : public MonocularCalibrationTypeOptions {
  explicit MonocularCalibrationOptions(
      MonocularCalibratorType type = MonocularCalibratorType::EXIF);

  // EXIF is the default, so that feature extraction reproduces the
  // long-standing EXIF/default focal length initialization unless a learned
  // backend is explicitly selected.
  MonocularCalibratorType type = MonocularCalibratorType::EXIF;

  // Target COLMAP camera model for the fitted intrinsics. Any perspective
  // model is supported; the fitting refines that model's parameters directly.
  // Empty preserves each camera's existing model instead of converting it.
  std::string camera_model;

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
  // Maximum absolute value of any distortion parameter. Matches the
  // incremental mapper default (`IncrementalMapper::Options::max_extra_param`),
  // so accepted calibrations also pass downstream checks. NOTE: this is a
  // single bound for all coefficients of the target model, so high-order
  // models with legitimately large coefficients (e.g. the rational
  // denominator terms k4, k5, k6 of FULL_OPENCV) may require a larger value.
  double max_extra_param = 1.0;

  bool Check() const;
};

// Populate the pose prior from the image EXIF tags: GPS position (WGS84) and
// gravity from the orientation tag. Only fields that are missing in
// `pose_prior` are set; already present values (e.g. from an IMU) are never
// overwritten.
void SetPosePriorFromExif(const Bitmap& bitmap, PosePrior* pose_prior);

// Abstract single-image camera calibrator: estimates intrinsics for one image,
// replacing e.g. EXIF-based initialization with a learned prediction.
class MonocularCalibrator {
 public:
  virtual ~MonocularCalibrator() = default;

  // Create the calibrator backend selected by `options.type`. Throws if the
  // options are invalid, the type is unknown, or the backend cannot be created
  // (e.g. the network model cannot be loaded); never returns nullptr.
  static std::unique_ptr<MonocularCalibrator> Create(
      const MonocularCalibrationOptions& options);

  // Calibrate the camera for `bitmap`, optionally using an existing focal
  // length prior in `camera` and gravity in `pose_prior`, and write the
  // target-model intrinsics at full image resolution into `camera` (model,
  // dimensions, params, and focal length prior flag). The pose prior is an
  // additional output: backends populate it from the image (see
  // `SetPosePriorFromExif`), independently of whether intrinsics calibration
  // succeeds. Returns false if calibration fails, in which case `camera` is
  // left unmodified.
  virtual bool Calibrate(const Bitmap& bitmap,
                         Camera* camera,
                         PosePrior* pose_prior) const = 0;
};

// Factory for the calibrator backend, injectable for testing. Defaults to
// `MonocularCalibrator::Create` when empty.
using MonocularCalibratorFactory =
    std::function<std::unique_ptr<MonocularCalibrator>(
        const MonocularCalibrationOptions& options)>;

}  // namespace colmap
