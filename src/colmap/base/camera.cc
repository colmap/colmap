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
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "colmap/base/camera.h"

#include "colmap/camera/models.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <iomanip>

namespace colmap {

Camera::Camera()
    : camera_id_(kInvalidCameraId),
      model_id_(kInvalidCameraModelId),
      width_(0),
      height_(0),
      prior_focal_length_(false) {}

std::string Camera::ModelName() const { return CameraModelIdToName(model_id_); }

void Camera::SetModelId(const int model_id) {
  CHECK(ExistsCameraModelWithId(model_id));
  model_id_ = model_id;
  params_.resize(CameraModelNumParams(model_id_), 0);
}

void Camera::SetModelIdFromName(const std::string& model_name) {
  CHECK(ExistsCameraModelWithName(model_name));
  model_id_ = CameraModelNameToId(model_name);
  params_.resize(CameraModelNumParams(model_id_), 0);
}

const std::vector<size_t>& Camera::FocalLengthIdxs() const {
  return CameraModelFocalLengthIdxs(model_id_);
}

const std::vector<size_t>& Camera::PrincipalPointIdxs() const {
  return CameraModelPrincipalPointIdxs(model_id_);
}

const std::vector<size_t>& Camera::ExtraParamsIdxs() const {
  return CameraModelExtraParamsIdxs(model_id_);
}

Eigen::Matrix3d Camera::CalibrationMatrix() const {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

  const std::vector<size_t>& idxs = FocalLengthIdxs();
  if (idxs.size() == 1) {
    K(0, 0) = params_[idxs[0]];
    K(1, 1) = params_[idxs[0]];
  } else if (idxs.size() == 2) {
    K(0, 0) = params_[idxs[0]];
    K(1, 1) = params_[idxs[1]];
  } else {
    LOG(FATAL)
        << "Camera model must either have 1 or 2 focal length parameters.";
  }

  K(0, 2) = PrincipalPointX();
  K(1, 2) = PrincipalPointY();

  return K;
}

std::string Camera::ParamsInfo() const {
  return CameraModelParamsInfo(model_id_);
}

double Camera::MeanFocalLength() const {
  const auto& focal_length_idxs = FocalLengthIdxs();
  double focal_length = 0;
  for (const auto idx : focal_length_idxs) {
    focal_length += params_[idx];
  }
  return focal_length / focal_length_idxs.size();
}

double Camera::FocalLength() const {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 1);
  return params_[idxs[0]];
}

double Camera::FocalLengthX() const {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 2);
  return params_[idxs[0]];
}

double Camera::FocalLengthY() const {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 2);
  return params_[idxs[1]];
}

void Camera::SetFocalLength(const double focal_length) {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  for (const auto idx : idxs) {
    params_[idx] = focal_length;
  }
}

void Camera::SetFocalLengthX(const double focal_length_x) {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[0]] = focal_length_x;
}

void Camera::SetFocalLengthY(const double focal_length_y) {
  const std::vector<size_t>& idxs = FocalLengthIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[1]] = focal_length_y;
}

double Camera::PrincipalPointX() const {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  return params_[idxs[0]];
}

double Camera::PrincipalPointY() const {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  return params_[idxs[1]];
}

void Camera::SetPrincipalPointX(const double ppx) {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[0]] = ppx;
}

void Camera::SetPrincipalPointY(const double ppy) {
  const std::vector<size_t>& idxs = PrincipalPointIdxs();
  CHECK_EQ(idxs.size(), 2);
  params_[idxs[1]] = ppy;
}

std::string Camera::ParamsToString() const { return VectorToCSV(params_); }

bool Camera::SetParamsFromString(const std::string& string) {
  const std::vector<double> new_camera_params = CSVToVector<double>(string);
  if (!CameraModelVerifyParams(model_id_, new_camera_params)) {
    return false;
  }

  params_ = new_camera_params;
  return true;
}

bool Camera::VerifyParams() const {
  return CameraModelVerifyParams(model_id_, params_);
}

bool Camera::HasBogusParams(const double min_focal_length_ratio,
                            const double max_focal_length_ratio,
                            const double max_extra_param) const {
  return CameraModelHasBogusParams(model_id_,
                                   params_,
                                   width_,
                                   height_,
                                   min_focal_length_ratio,
                                   max_focal_length_ratio,
                                   max_extra_param);
}

bool Camera::IsUndistorted() const {
  for (const size_t idx : ExtraParamsIdxs()) {
    if (std::abs(params_[idx]) > 1e-8) {
      return false;
    }
  }
  return true;
}

void Camera::InitializeWithId(const int model_id,
                              const double focal_length,
                              const size_t width,
                              const size_t height) {
  CHECK(ExistsCameraModelWithId(model_id));
  model_id_ = model_id;
  width_ = width;
  height_ = height;
  params_ = CameraModelInitializeParams(model_id, focal_length, width, height);
}

void Camera::InitializeWithName(const std::string& model_name,
                                const double focal_length,
                                const size_t width,
                                const size_t height) {
  InitializeWithId(
      CameraModelNameToId(model_name), focal_length, width, height);
}

Eigen::Vector2d Camera::ImageToWorld(const Eigen::Vector2d& image_point) const {
  Eigen::Vector2d world_point;
  CameraModelImageToWorld(model_id_,
                          params_,
                          image_point(0),
                          image_point(1),
                          &world_point(0),
                          &world_point(1));
  return world_point;
}

double Camera::ImageToWorldThreshold(const double threshold) const {
  return CameraModelImageToWorldThreshold(model_id_, params_, threshold);
}

Eigen::Vector2d Camera::WorldToImage(const Eigen::Vector2d& world_point) const {
  Eigen::Vector2d image_point;
  CameraModelWorldToImage(model_id_,
                          params_,
                          world_point(0),
                          world_point(1),
                          &image_point(0),
                          &image_point(1));
  return image_point;
}

void Camera::Rescale(const double scale) {
  CHECK_GT(scale, 0.0);
  const double scale_x =
      std::round(scale * width_) / static_cast<double>(width_);
  const double scale_y =
      std::round(scale * height_) / static_cast<double>(height_);
  width_ = static_cast<size_t>(std::round(scale * width_));
  height_ = static_cast<size_t>(std::round(scale * height_));
  SetPrincipalPointX(scale_x * PrincipalPointX());
  SetPrincipalPointY(scale_y * PrincipalPointY());
  if (FocalLengthIdxs().size() == 1) {
    SetFocalLength((scale_x + scale_y) / 2.0 * FocalLength());
  } else if (FocalLengthIdxs().size() == 2) {
    SetFocalLengthX(scale_x * FocalLengthX());
    SetFocalLengthY(scale_y * FocalLengthY());
  } else {
    LOG(FATAL)
        << "Camera model must either have 1 or 2 focal length parameters.";
  }
}

void Camera::Rescale(const size_t width, const size_t height) {
  const double scale_x =
      static_cast<double>(width) / static_cast<double>(width_);
  const double scale_y =
      static_cast<double>(height) / static_cast<double>(height_);
  width_ = width;
  height_ = height;
  SetPrincipalPointX(scale_x * PrincipalPointX());
  SetPrincipalPointY(scale_y * PrincipalPointY());
  if (FocalLengthIdxs().size() == 1) {
    SetFocalLength((scale_x + scale_y) / 2.0 * FocalLength());
  } else if (FocalLengthIdxs().size() == 2) {
    SetFocalLengthX(scale_x * FocalLengthX());
    SetFocalLengthY(scale_y * FocalLengthY());
  } else {
    LOG(FATAL)
        << "Camera model must either have 1 or 2 focal length parameters.";
  }
}

}  // namespace colmap
