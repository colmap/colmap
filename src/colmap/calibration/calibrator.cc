// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/calibration/calibrator.h"

#include "colmap/calibration/anycalib.h"
#include "colmap/calibration/exif.h"
#include "colmap/math/math.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace colmap {
namespace {

// Index of the calibration closest to `reference` under a spread-normalized
// L1 distance, so that parameters on different scales (e.g. focal lengths and
// distortion coefficients) contribute comparably.
size_t FindClosestCalibrationIdx(
    const std::vector<std::vector<double>>& params_list,
    const std::vector<double>& reference) {
  const size_t num_calibrations = params_list.size();
  std::vector<double> distances(num_calibrations, 0.0);
  std::vector<double> deviations(num_calibrations);
  std::vector<double> sorted_deviations(num_calibrations);
  for (size_t d = 0; d < reference.size(); ++d) {
    for (size_t i = 0; i < num_calibrations; ++i) {
      deviations[i] = std::abs(params_list[i][d] - reference[d]);
    }
    // Normalize by the median absolute deviation. Dimensions without spread
    // then contribute nothing to the distance.
    sorted_deviations = deviations;
    const double mad = Median(sorted_deviations);
    if (mad > 0) {
      for (size_t i = 0; i < num_calibrations; ++i) {
        distances[i] += deviations[i] / mad;
      }
    }
  }
  return std::min_element(distances.begin(), distances.end()) -
         distances.begin();
}

}  // namespace

bool IsValidCalibration(const Camera& camera) {
  if (camera.width == 0 || camera.height == 0) {
    return false;
  }
  if (!std::all_of(camera.params.begin(),
                   camera.params.end(),
                   [](const double param) { return std::isfinite(param); }) ||
      !camera.VerifyParams()) {
    return false;
  }
  for (const size_t idx : camera.FocalLengthIdxs()) {
    if (camera.params[idx] <= 0) {
      return false;
    }
  }
  // Unproject a coarse grid of pixels to rays and project them back. Diverging
  // distortion shows up as either a failed or an inaccurate round-trip. The
  // grid covers bin centers rather than the exact image edges, and the
  // tolerance admits small numerical errors while rejecting divergence.
  constexpr int kNumGridSteps = 8;
  constexpr double kMaxRelativeRoundTripError = 0.001;
  const double max_error =
      kMaxRelativeRoundTripError * std::max(camera.width, camera.height);
  for (int y = 0; y <= kNumGridSteps; ++y) {
    for (int x = 0; x <= kNumGridSteps; ++x) {
      const Eigen::Vector2d image_point(
          camera.width * (static_cast<double>(x) + 0.5) / (kNumGridSteps + 1),
          camera.height * (static_cast<double>(y) + 0.5) / (kNumGridSteps + 1));
      const std::optional<Eigen::Vector3d> cam_ray =
          camera.CamRayFromImg(image_point);
      if (!cam_ray.has_value() || !cam_ray->allFinite()) {
        return false;
      }
      const std::optional<Eigen::Vector2d> projected =
          camera.ImgFromCam(*cam_ray);
      if (!projected.has_value() || !projected->allFinite() ||
          (*projected - image_point).norm() > max_error) {
        return false;
      }
    }
  }
  return true;
}

bool AggregateMonocularCalibrations(
    const CameraModelId model_id,
    const std::vector<std::vector<double>>& params_list,
    Camera* camera) {
  THROW_CHECK_NOTNULL(camera);
  if (params_list.empty()) {
    return false;
  }
  const size_t dim = params_list[0].size();
  for (const auto& params : params_list) {
    THROW_CHECK_EQ(params.size(), dim);
  }

  std::vector<double> median(dim);
  std::vector<double> values(params_list.size());
  for (size_t d = 0; d < dim; ++d) {
    for (size_t i = 0; i < params_list.size(); ++i) {
      values[i] = params_list[i][d];
    }
    median[d] = Median(values);
  }

  // Validate on a candidate, so that `camera` is only modified on success.
  Camera candidate = *camera;
  candidate.model_id = model_id;
  candidate.params = std::move(median);
  if (!IsValidCalibration(candidate)) {
    // Fall back to the single-image calibration closest to the median, which
    // trades the noise averaging of the median for a parameter vector that is
    // guaranteed to have been observed (and is re-validated below).
    LOG(WARNING) << "Aggregated calibration for camera " << camera->camera_id
                 << " is invalid, falling back to the closest single-image "
                    "calibration";
    candidate.params =
        params_list[FindClosestCalibrationIdx(params_list, candidate.params)];
    if (!IsValidCalibration(candidate)) {
      LOG(WARNING) << "Fallback calibration for camera " << camera->camera_id
                   << " is invalid too, keeping existing intrinsics";
      return false;
    }
  }

  camera->model_id = candidate.model_id;
  camera->params = std::move(candidate.params);
  camera->has_prior_focal_length = true;
  return true;
}

MonocularCalibrationTypeOptions::MonocularCalibrationTypeOptions()
    : anycalib(std::make_shared<AnyCalibOptions>()) {}

MonocularCalibrationTypeOptions::MonocularCalibrationTypeOptions(
    const MonocularCalibrationTypeOptions& other) {
  if (other.anycalib) {
    anycalib = std::make_shared<AnyCalibOptions>(*other.anycalib);
  }
}

MonocularCalibrationTypeOptions& MonocularCalibrationTypeOptions::operator=(
    const MonocularCalibrationTypeOptions& other) {
  if (this == &other) {
    return *this;
  }
  if (other.anycalib) {
    anycalib = std::make_shared<AnyCalibOptions>(*other.anycalib);
  } else {
    anycalib.reset();
  }
  return *this;
}

MonocularCalibrationOptions::MonocularCalibrationOptions(
    MonocularCalibratorType type)
    : MonocularCalibrationTypeOptions(), type(type) {}

bool MonocularCalibrationOptions::Check() const {
  if (!camera_model.empty()) {
    CHECK_OPTION(ExistsCameraModelWithName(camera_model));
    const CameraModelId model_id = CameraModelNameToId(camera_model);
    CHECK_OPTION(CameraModelIsPerspective(model_id));
  }
  CHECK_OPTION_GT(num_threads, -2);
  CHECK_OPTION_GT(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GT(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GT(max_extra_param, 0.0);
  switch (type) {
    case MonocularCalibratorType::ANYCALIB:
      CHECK_OPTION(anycalib != nullptr);
      CHECK_OPTION(anycalib->Check());
      break;
    case MonocularCalibratorType::EXIF:
      // No backend-specific settings to validate.
      break;
    default:
      LOG(ERROR) << "Unknown monocular calibrator type";
      return false;
  }
  return true;
}

std::unique_ptr<MonocularCalibrator> MonocularCalibrator::Create(
    const MonocularCalibrationOptions& options) {
  THROW_CHECK(options.Check());
  switch (options.type) {
    case MonocularCalibratorType::ANYCALIB:
      return CreateAnyCalibCalibrator(options);
    case MonocularCalibratorType::EXIF:
      return CreateExifCalibrator(options);
    default:
      LOG(FATAL_THROW) << "Unknown monocular calibrator type";
  }
  return nullptr;
}

void SetPosePriorFromExif(const Bitmap& bitmap, PosePrior* pose_prior) {
  THROW_CHECK_NOTNULL(pose_prior);
  if (!pose_prior->HasPosition()) {
    const std::optional<double> latitude = bitmap.ExifLatitude();
    const std::optional<double> longitude = bitmap.ExifLongitude();
    const std::optional<double> altitude = bitmap.ExifAltitude();
    if (latitude.has_value() && longitude.has_value() && altitude.has_value()) {
      pose_prior->position = Eigen::Vector3d(*latitude, *longitude, *altitude);
      pose_prior->coordinate_system = PosePrior::CoordinateSystem::WGS84;
    }
  }
  if (!pose_prior->HasGravity()) {
    const std::optional<int> orientation = bitmap.ExifOrientation();
    if (orientation.has_value()) {
      if (const auto gravity = GravityFromExifOrientation(orientation.value());
          gravity.has_value()) {
        pose_prior->gravity = gravity.value();
      }
    }
  }
}

}  // namespace colmap
