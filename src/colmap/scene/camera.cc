// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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

#include "colmap/scene/camera.h"

#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <iomanip>

namespace colmap {

Camera Camera::CreateFromModelId(camera_t camera_id,
                                 const CameraModelId model_id,
                                 const double focal_length,
                                 const size_t width,
                                 const size_t height) {
  THROW_CHECK(ExistsCameraModelWithId(model_id));
  Camera camera;
  camera.camera_id = camera_id;
  camera.model_id = model_id;
  camera.width = width;
  camera.height = height;
  camera.params =
      CameraModelInitializeParams(model_id, focal_length, width, height);
  return camera;
}

Camera Camera::CreateFromModelName(camera_t camera_id,
                                   const std::string& model_name,
                                   const double focal_length,
                                   const size_t width,
                                   const size_t height) {
  return CreateFromModelId(
      camera_id, CameraModelNameToId(model_name), focal_length, width, height);
}

Eigen::Matrix3d Camera::CalibrationMatrix() const {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

  const span<const size_t> idxs = FocalLengthIdxs();
  if (idxs.size() == 1) {
    K(0, 0) = params[idxs[0]];
    K(1, 1) = params[idxs[0]];
  } else if (idxs.size() == 2) {
    K(0, 0) = params[idxs[0]];
    K(1, 1) = params[idxs[1]];
  } else {
    LOG(FATAL_THROW)
        << "Camera model must either have 1 or 2 focal length parameters.";
  }

  K(0, 2) = PrincipalPointX();
  K(1, 2) = PrincipalPointY();

  return K;
}

double Camera::MeanFocalLength() const {
  const span<const size_t> focal_length_idxs = FocalLengthIdxs();
  double focal_length = 0;
  for (const auto idx : focal_length_idxs) {
    focal_length += params[idx];
  }
  return focal_length / focal_length_idxs.size();
}

std::string Camera::ParamsToString() const { return VectorToCSV(params); }

bool Camera::SetParamsFromString(const std::string& string) {
  std::vector<double> new_camera_params = CSVToVector<double>(string);
  if (!CameraModelVerifyParams(model_id, new_camera_params)) {
    return false;
  }
  params = std::move(new_camera_params);
  return true;
}

bool Camera::IsUndistorted() const {
  for (const size_t idx : ExtraParamsIdxs()) {
    if (std::abs(params[idx]) > 1e-8) {
      return false;
    }
  }
  return true;
}

void Camera::Rescale(const double scale) {
  THROW_CHECK_GT(scale, 0.0);
  const double scale_x = std::round(scale * width) / static_cast<double>(width);
  const double scale_y =
      std::round(scale * height) / static_cast<double>(height);
  width = static_cast<size_t>(std::round(scale * width));
  height = static_cast<size_t>(std::round(scale * height));
  SetPrincipalPointX(scale_x * PrincipalPointX());
  SetPrincipalPointY(scale_y * PrincipalPointY());
  if (FocalLengthIdxs().size() == 1) {
    SetFocalLength((scale_x + scale_y) / 2.0 * FocalLength());
  } else if (FocalLengthIdxs().size() == 2) {
    SetFocalLengthX(scale_x * FocalLengthX());
    SetFocalLengthY(scale_y * FocalLengthY());
  } else {
    LOG(FATAL_THROW)
        << "Camera model must either have 1 or 2 focal length parameters.";
  }
}

void Camera::Rescale(const size_t new_width, const size_t new_height) {
  const double scale_x =
      static_cast<double>(new_width) / static_cast<double>(width);
  const double scale_y =
      static_cast<double>(new_height) / static_cast<double>(height);
  width = new_width;
  height = new_height;
  SetPrincipalPointX(scale_x * PrincipalPointX());
  SetPrincipalPointY(scale_y * PrincipalPointY());
  if (FocalLengthIdxs().size() == 1) {
    SetFocalLength((scale_x + scale_y) / 2.0 * FocalLength());
  } else if (FocalLengthIdxs().size() == 2) {
    SetFocalLengthX(scale_x * FocalLengthX());
    SetFocalLengthY(scale_y * FocalLengthY());
  } else {
    LOG(FATAL_THROW)
        << "Camera model must either have 1 or 2 focal length parameters.";
  }
}

}  // namespace colmap
