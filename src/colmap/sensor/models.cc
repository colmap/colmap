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

#include "colmap/sensor/models.h"

#include <unordered_map>

#include <boost/algorithm/string.hpp>

namespace colmap {

// Initialize params_info, focal_length_idxs, principal_point_idxs,
// extra_params_idxs
#define CAMERA_MODEL_CASE(CameraModel)                    \
  constexpr int CameraModel::model_id;                    \
  const std::string CameraModel::model_name =             \
      CameraModel::InitializeModelName();                 \
  constexpr size_t CameraModel::num_params;               \
  const std::string CameraModel::params_info =            \
      CameraModel::InitializeParamsInfo();                \
  const std::array<size_t, CameraModel::num_focal_params> \
      CameraModel::focal_length_idxs =                    \
          CameraModel::InitializeFocalLengthIdxs();       \
  const std::array<size_t, CameraModel::num_pp_params>    \
      CameraModel::principal_point_idxs =                 \
          CameraModel::InitializePrincipalPointIdxs();    \
  const std::array<size_t, CameraModel::num_extra_params> \
      CameraModel::extra_params_idxs =                    \
          CameraModel::InitializeExtraParamsIdxs();

CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE

std::unordered_map<std::string, int> InitialzeCameraModelNameToId() {
  std::unordered_map<std::string, int> camera_model_name_to_id;

#define CAMERA_MODEL_CASE(CameraModel)                     \
  camera_model_name_to_id.emplace(CameraModel::model_name, \
                                  CameraModel::model_id);

  CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE

  return camera_model_name_to_id;
}

std::unordered_map<int, std::string> InitialzeCameraModelIdToName() {
  std::unordered_map<int, std::string> camera_model_id_to_name;

#define CAMERA_MODEL_CASE(CameraModel)                   \
  camera_model_id_to_name.emplace(CameraModel::model_id, \
                                  CameraModel::model_name);

  CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE

  return camera_model_id_to_name;
}

static const std::unordered_map<std::string, int> CAMERA_MODEL_NAME_TO_ID =
    InitialzeCameraModelNameToId();

static const std::unordered_map<int, std::string> CAMERA_MODEL_ID_TO_NAME =
    InitialzeCameraModelIdToName();

bool ExistsCameraModelWithName(const std::string& model_name) {
  return CAMERA_MODEL_NAME_TO_ID.count(model_name) > 0;
}

bool ExistsCameraModelWithId(const int model_id) {
  return CAMERA_MODEL_ID_TO_NAME.count(model_id) > 0;
}

int CameraModelNameToId(const std::string& model_name) {
  const auto it = CAMERA_MODEL_NAME_TO_ID.find(model_name);
  if (it == CAMERA_MODEL_NAME_TO_ID.end()) {
    return kInvalidCameraModelId;
  } else {
    return it->second;
  }
}

std::string CameraModelIdToName(const int model_id) {
  const auto it = CAMERA_MODEL_ID_TO_NAME.find(model_id);
  if (it == CAMERA_MODEL_ID_TO_NAME.end()) {
    return "";
  } else {
    return it->second;
  }
}

std::vector<double> CameraModelInitializeParams(const int model_id,
                                                const double focal_length,
                                                const size_t width,
                                                const size_t height) {
  // Assuming that image measurements are within [0, dim], i.e. that the
  // upper left corner is the (0, 0) coordinate (rather than the center of
  // the upper left pixel). This complies with the default SiftGPU convention.
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                 \
  case CameraModel::model_id:                                          \
    return CameraModel::InitializeParams(focal_length, width, height); \
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

span<const size_t> CameraModelFocalLengthIdxs(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    return {CameraModel::focal_length_idxs.data(),  \
            CameraModel::focal_length_idxs.size()}; \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return {nullptr, 0};
}

span<const size_t> CameraModelPrincipalPointIdxs(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                 \
  case CameraModel::model_id:                          \
    return {CameraModel::principal_point_idxs.data(),  \
            CameraModel::principal_point_idxs.size()}; \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return {nullptr, 0};
}

span<const size_t> CameraModelExtraParamsIdxs(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    return {CameraModel::extra_params_idxs.data(),  \
            CameraModel::extra_params_idxs.size()}; \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return {nullptr, 0};
}

size_t CameraModelNumParams(const int model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::num_params;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return 0;
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
                               const size_t width,
                               const size_t height,
                               const double min_focal_length_ratio,
                               const double max_focal_length_ratio,
                               const double max_extra_param) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                         \
  case CameraModel::model_id:                                  \
    return CameraModel::HasBogusParams(params,                 \
                                       width,                  \
                                       height,                 \
                                       min_focal_length_ratio, \
                                       max_focal_length_ratio, \
                                       max_extra_param);       \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return false;
}

}  // namespace colmap
