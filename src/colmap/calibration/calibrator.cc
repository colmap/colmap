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

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>

namespace colmap {

bool AggregateCameraCalibrations(
    const std::vector<std::vector<double>>& params_list, Camera* camera) {
  THROW_CHECK_NOTNULL(camera);
  if (params_list.empty()) {
    return false;
  }
  const size_t dim = params_list[0].size();
  std::vector<double> median(dim);
  std::vector<double> values;
  values.reserve(params_list.size());
  for (size_t d = 0; d < dim; ++d) {
    values.clear();
    for (const auto& params : params_list) {
      THROW_CHECK_EQ(params.size(), dim);
      values.push_back(params[d]);
    }
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    median[d] = (n % 2 == 1) ? values[n / 2]
                             : 0.5 * (values[n / 2 - 1] + values[n / 2]);
  }
  camera->params = median;
  camera->has_prior_focal_length = true;
  return true;
}

bool CameraCalibrationOptions::Check() const {
  CHECK_OPTION(ExistsCameraModelWithName(camera_model));
  const CameraModelId model_id = CameraModelNameToId(camera_model);
  CHECK_OPTION(CameraModelIsPerspective(model_id));
  CHECK_OPTION_GT(num_threads, -2);
  switch (type) {
    case CameraCalibratorType::ANYCALIB:
      CHECK_OPTION(anycalib.Check());
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
