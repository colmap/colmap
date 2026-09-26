// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/exif.h"

#include "colmap/util/logging.h"

#include <optional>

namespace colmap {
namespace {

class ExifCalibrator : public MonocularCalibrator {
 public:
  explicit ExifCalibrator(const MonocularCalibrationOptions& options)
      : options_(options) {
    THROW_CHECK(options_.Check());
  }

  bool Calibrate(const Bitmap& bitmap,
                 Camera* camera,
                 PosePrior* pose_prior) const override {
    THROW_CHECK_NOTNULL(camera);
    THROW_CHECK_NOTNULL(pose_prior);
    // The pose prior is populated independently of whether intrinsics
    // calibration succeeds below.
    SetPosePriorFromExif(bitmap, pose_prior);
    // EXIF parameters cannot be converted across models, so only the
    // existing model is supported.
    const CameraModelId model_id =
        options_.camera_model.empty()
            ? camera->model_id
            : CameraModelNameToId(options_.camera_model);
    if (model_id == CameraModelId::kInvalid || model_id != camera->model_id) {
      return false;
    }
    if (bitmap.Width() != static_cast<int>(camera->width) ||
        bitmap.Height() != static_cast<int>(camera->height)) {
      return false;
    }
    // Images without an EXIF focal length fail calibration, keeping the
    // existing (default) intrinsics.
    const std::optional<double> focal_length = bitmap.ExifFocalLength();
    if (!focal_length.has_value()) {
      return false;
    }
    // Validate on a candidate, so that `camera` is only modified on success.
    // A corrupt or implausible EXIF tag then falls back to the existing
    // intrinsics instead of poisoning the camera.
    Camera calibrated = *camera;
    calibrated.SetFocalLength(focal_length.value());
    calibrated.has_prior_focal_length = true;
    if (!IsValidCalibration(calibrated) ||
        calibrated.HasBogusParams(options_.min_focal_length_ratio,
                                  options_.max_focal_length_ratio,
                                  options_.max_extra_param)) {
      return false;
    }
    *camera = calibrated;
    return true;
  }

 private:
  MonocularCalibrationOptions options_;
};

}  // namespace

std::unique_ptr<MonocularCalibrator> CreateExifCalibrator(
    const MonocularCalibrationOptions& options) {
  return std::make_unique<ExifCalibrator>(options);
}

}  // namespace colmap
