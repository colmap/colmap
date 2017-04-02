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

#include "base/camera_models.h"

#include <boost/algorithm/string.hpp>

namespace colmap {

// Initialize params_info, focal_length_idxs, principal_point_idxs,
// extra_params_idxs
#define CAMERA_MODEL_CASE(CameraModel)                          \
  const std::string CameraModel::params_info =                  \
      CameraModel::InitializeParamsInfo();                      \
  const std::vector<size_t> CameraModel::focal_length_idxs =    \
      CameraModel::InitializeFocalLengthIdxs();                 \
  const std::vector<size_t> CameraModel::principal_point_idxs = \
      CameraModel::InitializePrincipalPointIdxs();              \
  const std::vector<size_t> CameraModel::extra_params_idxs =    \
      CameraModel::InitializeExtraParamsIdxs();

CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE

int CameraModelNameToId(const std::string& name) {
  std::string uppercast_name = name;
  boost::to_upper(uppercast_name);
  if (uppercast_name == "SIMPLE_PINHOLE") {
    return SimplePinholeCameraModel::model_id;
  } else if (uppercast_name == "PINHOLE") {
    return PinholeCameraModel::model_id;
  } else if (uppercast_name == "SIMPLE_RADIAL") {
    return SimpleRadialCameraModel::model_id;
  } else if (uppercast_name == "SIMPLE_RADIAL_FISHEYE") {
    return SimpleRadialFisheyeCameraModel::model_id;
  } else if (uppercast_name == "RADIAL") {
    return RadialCameraModel::model_id;
  } else if (uppercast_name == "RADIAL_FISHEYE") {
    return RadialFisheyeCameraModel::model_id;
  } else if (uppercast_name == "OPENCV") {
    return OpenCVCameraModel::model_id;
  } else if (uppercast_name == "OPENCV_FISHEYE") {
    return OpenCVFisheyeCameraModel::model_id;
  } else if (uppercast_name == "FULL_OPENCV") {
    return FullOpenCVCameraModel::model_id;
  } else if (uppercast_name == "FOV") {
    return FOVCameraModel::model_id;
  } else if (uppercast_name == "THIN_PRISM_FISHEYE") {
    return ThinPrismFisheyeCameraModel::model_id;
  }
  return kInvalidCameraModelId;
}

std::string CameraModelIdToName(const int model_id) {
  if (model_id == SimplePinholeCameraModel::model_id) {
    return "SIMPLE_PINHOLE";
  } else if (model_id == PinholeCameraModel::model_id) {
    return "PINHOLE";
  } else if (model_id == SimpleRadialCameraModel::model_id) {
    return "SIMPLE_RADIAL";
  } else if (model_id == SimpleRadialFisheyeCameraModel::model_id) {
    return "SIMPLE_RADIAL_FISHEYE";
  } else if (model_id == RadialCameraModel::model_id) {
    return "RADIAL";
  } else if (model_id == RadialFisheyeCameraModel::model_id) {
    return "RADIAL_FISHEYE";
  } else if (model_id == OpenCVCameraModel::model_id) {
    return "OPENCV";
  } else if (model_id == OpenCVFisheyeCameraModel::model_id) {
    return "OPENCV_FISHEYE";
  } else if (model_id == FullOpenCVCameraModel::model_id) {
    return "FULL_OPENCV";
  } else if (model_id == FOVCameraModel::model_id) {
    return "FOV";
  } else if (model_id == ThinPrismFisheyeCameraModel::model_id) {
    return "THIN_PRISM_FISHEYE";
  }
  return "INVALID_CAMERA_MODEL";
}

void CameraModelInitializeParams(const int model_id, const double focal_length,
                                 const size_t width, const size_t height,
                                 std::vector<double>* params) {
  // Assuming that image measurements are within [0, dim], i.e. that the
  // upper left corner is the (0, 0) coordinate (rather than the center of
  // the upper left pixel). This complies with the default SiftGPU convention.
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                              \
  case CameraModel::model_id:                                       \
    params->resize(CameraModel::num_params);                        \
    for (const int idx : CameraModel::focal_length_idxs) {          \
      (*params)[idx] = focal_length;                                \
    }                                                               \
    (*params)[CameraModel::principal_point_idxs[0]] = width / 2.0;  \
    (*params)[CameraModel::principal_point_idxs[1]] = height / 2.0; \
    for (const int idx : CameraModel::extra_params_idxs) {          \
      (*params)[idx] = 0;                                           \
    }                                                               \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

std::string CameraModelParamsInfo(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::params_info;   \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return "Camera model does not exist";
}

std::vector<size_t> CameraModelFocalLengthIdxs(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)     \
  case CameraModel::model_id:              \
    return CameraModel::focal_length_idxs; \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return std::vector<size_t>{};
}

std::vector<size_t> CameraModelPrincipalPointIdxs(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)        \
  case CameraModel::model_id:                 \
    return CameraModel::principal_point_idxs; \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return std::vector<size_t>{};
}

std::vector<size_t> CameraModelExtraParamsIdxs(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)     \
  case CameraModel::model_id:              \
    return CameraModel::extra_params_idxs; \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return std::vector<size_t>{};
}

bool CameraModelVerifyParams(const int model_id,
                             const std::vector<double>& params) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    if (params.size() == CameraModel::num_params) { \
      return true;                                  \
    }                                               \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

bool CameraModelHasBogusParams(const int model_id,
                               const std::vector<double>& params,
                               const size_t width, const size_t height,
                               const double min_focal_length_ratio,
                               const double max_focal_length_ratio,
                               const double max_extra_param) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return CameraModel::HasBogusParams(                                        \
        params, width, height, min_focal_length_ratio, max_focal_length_ratio, \
        max_extra_param);                                                      \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

}  // namespace colmap
