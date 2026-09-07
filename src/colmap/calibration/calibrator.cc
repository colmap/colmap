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

#include "colmap/calibration/calibrator.h"

#include "colmap/calibration/anycalib.h"
#include "colmap/math/math.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <cmath>

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
  THROW_CHECK_GT(camera.width, 0);
  THROW_CHECK_GT(camera.height, 0);
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
  // distortion shows up as either a failed or an inaccurate round-trip.
  constexpr int kNumGridSteps = 8;
  constexpr double kMaxRelativeRoundTripError = 0.01;
  const double max_error =
      kMaxRelativeRoundTripError * std::max(camera.width, camera.height);
  for (int y = 0; y <= kNumGridSteps; ++y) {
    for (int x = 0; x <= kNumGridSteps; ++x) {
      const Eigen::Vector2d image_point(
          camera.width * static_cast<double>(x) / kNumGridSteps,
          camera.height * static_cast<double>(y) / kNumGridSteps);
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

bool AggregateCameraCalibrations(
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
    // Fall back to the individually validated calibration closest to the
    // median, which trades the noise averaging of the median for a parameter
    // vector that is guaranteed to have been observed.
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

CameraCalibrationTypeOptions::CameraCalibrationTypeOptions()
    : anycalib(std::make_shared<AnyCalibCalibrationOptions>()) {}

CameraCalibrationTypeOptions::CameraCalibrationTypeOptions(
    const CameraCalibrationTypeOptions& other) {
  if (other.anycalib) {
    anycalib = std::make_shared<AnyCalibCalibrationOptions>(*other.anycalib);
  }
}

CameraCalibrationTypeOptions& CameraCalibrationTypeOptions::operator=(
    const CameraCalibrationTypeOptions& other) {
  if (this == &other) {
    return *this;
  }
  if (other.anycalib) {
    anycalib = std::make_shared<AnyCalibCalibrationOptions>(*other.anycalib);
  } else {
    anycalib.reset();
  }
  return *this;
}

CameraCalibrationOptions::CameraCalibrationOptions(CameraCalibratorType type)
    : CameraCalibrationTypeOptions(), type(type) {}

bool CameraCalibrationOptions::Check() const {
  CHECK_OPTION(ExistsCameraModelWithName(camera_model));
  const CameraModelId model_id = CameraModelNameToId(camera_model);
  CHECK_OPTION(CameraModelIsPerspective(model_id));
  CHECK_OPTION_GT(num_threads, -2);
  CHECK_OPTION_GT(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GT(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GT(max_extra_param, 0.0);
  switch (type) {
    case CameraCalibratorType::ANYCALIB:
      CHECK_OPTION(THROW_CHECK_NOTNULL(anycalib)->Check());
      break;
    default:
      LOG(FATAL_THROW) << "Unknown camera calibrator type";
  }
  return true;
}

std::unique_ptr<CameraCalibrator> CameraCalibrator::Create(
    const CameraCalibrationOptions& options) {
  THROW_CHECK(options.Check());
  switch (options.type) {
    case CameraCalibratorType::ANYCALIB:
      return CreateAnyCalibCalibrator(options);
    default:
      LOG(FATAL_THROW) << "Unknown camera calibrator type";
  }
}

}  // namespace colmap
