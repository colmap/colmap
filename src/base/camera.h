// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_BASE_CAMERA_H_
#define COLMAP_SRC_BASE_CAMERA_H_

#include <vector>

#include "base/camera_models.h"
#include "util/types.h"

namespace colmap {

// Camera class that holds the intrinsic parameters. Cameras may be shared
// between multiple images, e.g., if the same "physical" camera took multiple
// pictures with the exact same lens and intrinsics (focal length, etc.).
// This class has a specific distortion model defined by a camera model class.
class Camera {
 public:
  Camera();

  // Access the unique identifier of the camera.
  inline camera_t CameraId() const;
  inline void SetCameraId(const camera_t camera_id);

  // Access the camera model.
  inline int ModelId() const;
  std::string ModelName() const;
  void SetModelId(const int model_id);
  void SetModelIdFromName(const std::string& name);

  // Access dimensions of the camera sensor.
  inline size_t Width() const;
  inline size_t Height() const;
  inline void SetWidth(const size_t width);
  inline void SetHeight(const size_t height);

  // Access focal length parameters.
  double MeanFocalLength() const;
  double FocalLength() const;
  double FocalLengthX() const;
  double FocalLengthY() const;
  void SetFocalLength(const double focal_length);
  void SetFocalLengthX(const double focal_length_x);
  void SetFocalLengthY(const double focal_length_y);

  // Check if camera has prior focal length.
  inline bool HasPriorFocalLength() const;
  inline void SetPriorFocalLength(const bool prior);

  // Access principal point parameters. Only works if there are two
  // principal point parameters.
  double PrincipalPointX() const;
  double PrincipalPointY() const;
  void SetPrincipalPointX(const double ppx);
  void SetPrincipalPointY(const double ppy);

  // Get the indices of the parameter groups in the parameter vector.
  const std::vector<size_t>& FocalLengthIdxs() const;
  const std::vector<size_t>& PrincipalPointIdxs() const;
  const std::vector<size_t>& ExtraParamsIdxs() const;

  // Get intrinsic calibration matrix composed from focal length and principal
  // point parameters, excluding distortion parameters.
  Eigen::Matrix3d CalibrationMatrix() const;

  // Get human-readable information about the parameter vector ordering.
  std::string ParamsInfo() const;

  // Access the raw parameter vector.
  inline size_t NumParams() const;
  inline const std::vector<double>& Params() const;
  inline std::vector<double>& Params();
  inline double Params(const size_t idx) const;
  inline double& Params(const size_t idx);
  inline const double* ParamsData() const;
  inline double* ParamsData();
  inline void SetParams(const std::vector<double>& params);

  // Concatenate parameters as comma-separated list.
  std::string ParamsToString() const;

  // Set camera parameters from comma-separated list.
  bool SetParamsFromString(const std::string& string);

  // Check whether parameters are valid, i.e. the parameter vector has
  // the correct dimensions that match the specified camera model.
  bool VerifyParams() const;

  // Check whether camera has bogus parameters.
  bool HasBogusParams(const double min_focal_length_ratio,
                      const double max_focal_length_ratio,
                      const double max_extra_param) const;

  // Initialize parameters for given camera model and focal length, and set
  // the principal point to be the image center.
  void InitializeWithId(const int model_id, const double focal_length,
                        const size_t width, const size_t height);
  void InitializeWithName(const std::string& model_name,
                          const double focal_length, const size_t width,
                          const size_t height);

  // Project point in image plane to world / infinity.
  Eigen::Vector2d ImageToWorld(const Eigen::Vector2d& image_point) const;

  // Convert pixel threshold in image plane to world space.
  double ImageToWorldThreshold(const double threshold) const;

  // Project point from world / infinity to image plane.
  Eigen::Vector2d WorldToImage(const Eigen::Vector2d& world_point) const;

  // Rescale camera dimensions and accordingly the focal length and
  // and the principal point.
  void Rescale(const double scale);

 private:
  // The unique identifier of the camera. If the identifier is not specified
  // it is set to `kInvalidCameraId`.
  camera_t camera_id_;

  // The identifier of the camera model. If the camera model is not specified
  // the identifier is `kInvalidCameraModelId`.
  int model_id_;

  // The dimensions of the image, 0 if not initialized.
  size_t width_;
  size_t height_;

  // The focal length, principal point, and extra parameters. If the camera
  // model is not specified, this vector is empty.
  std::vector<double> params_;

  // Whether there is a safe prior for the focal length,
  // e.g. manually provided or extracted from EXIF
  bool prior_focal_length_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

camera_t Camera::CameraId() const { return camera_id_; }

void Camera::SetCameraId(const camera_t camera_id) { camera_id_ = camera_id; }

int Camera::ModelId() const { return model_id_; }

size_t Camera::Width() const { return width_; }

size_t Camera::Height() const { return height_; }

void Camera::SetWidth(const size_t width) { width_ = width; }

void Camera::SetHeight(const size_t height) { height_ = height; }

bool Camera::HasPriorFocalLength() const { return prior_focal_length_; }

void Camera::SetPriorFocalLength(const bool prior) {
  prior_focal_length_ = prior;
}

size_t Camera::NumParams() const { return params_.size(); }

const std::vector<double>& Camera::Params() const { return params_; }

std::vector<double>& Camera::Params() { return params_; }

double Camera::Params(const size_t idx) const { return params_[idx]; }

double& Camera::Params(const size_t idx) { return params_[idx]; }

const double* Camera::ParamsData() const { return params_.data(); }

double* Camera::ParamsData() { return params_.data(); }

void Camera::SetParams(const std::vector<double>& params) { params_ = params; }

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_CAMERA_H_
