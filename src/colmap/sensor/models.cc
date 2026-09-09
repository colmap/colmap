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

#include "colmap/sensor/models.h"

#include "colmap/util/hash_containers.h"

namespace colmap {

bool ExistsCameraModelWithName(const std::string& model_name) {
  return CameraModelNameToId(model_name) != CameraModelId::kInvalid;
}

bool ExistsCameraModelWithId(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return true;

    CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
    default:
      return false;
  }
}

CameraModelId CameraModelNameToId(const std::string& model_name) {
  // Function-local static: built once on first use (thread-safe), keeping
  // the name lookup O(1) without paying for eager static initialization.
  static const NodeHashMap<std::string, CameraModelId> kNameToId = [] {
    NodeHashMap<std::string, CameraModelId> name_to_id;
#define CAMERA_MODEL_CASE(CameraModel) \
  name_to_id.emplace(CameraModel::model_name, CameraModel::model_id);

    CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE

    return name_to_id;
  }();

  const auto it = kNameToId.find(model_name);
  if (it == kNameToId.end()) {
    return CameraModelId::kInvalid;
  } else {
    return it->second;
  }
}

const std::string& CameraModelIdToName(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::model_name;

    CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
    default:
      break;
  }

  const static std::string kEmptyModelName = "";
  return kEmptyModelName;
}

std::vector<double> CameraModelInitializeParams(const CameraModelId model_id,
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

const std::string& CameraModelParamsInfo(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::params_info;   \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  const static std::string kEmptyParamsInfo = "";
  return kEmptyParamsInfo;
}

span<const size_t> CameraModelFocalLengthIdxs(const CameraModelId model_id) {
  switch (model_id) {
    // Only perspective models have focal-length parameters; spherical models
    // have none. Unknown models throw via the default case.
#define CAMERA_MODEL_CASE(CameraModel)             \
  case CameraModel::model_id:                      \
    return {CameraModel::focal_length_idxs.data(), \
            CameraModel::focal_length_idxs.size()};
    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return {nullptr, 0};
    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }
  return {nullptr, 0};
}

span<const size_t> CameraModelPrincipalPointIdxs(const CameraModelId model_id) {
  switch (model_id) {
    // Only perspective models have principal-point parameters; spherical models
    // have none. Unknown models throw via the default case.
#define CAMERA_MODEL_CASE(CameraModel)                \
  case CameraModel::model_id:                         \
    return {CameraModel::principal_point_idxs.data(), \
            CameraModel::principal_point_idxs.size()};
    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return {nullptr, 0};
    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }
  return {nullptr, 0};
}

span<const size_t> CameraModelExtraParamsIdxs(const CameraModelId model_id) {
  switch (model_id) {
    // Only perspective models have extra parameters; spherical models have
    // none. Unknown models throw via the default case.
#define CAMERA_MODEL_CASE(CameraModel)             \
  case CameraModel::model_id:                      \
    return {CameraModel::extra_params_idxs.data(), \
            CameraModel::extra_params_idxs.size()};
    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return {nullptr, 0};
    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }
  return {nullptr, 0};
}

span<const size_t> CameraModelMetaDataParamsIdxs(const CameraModelId model_id) {
  switch (model_id) {
    // Only spherical models have metadata parameters; perspective models have
    // none. Unknown models throw via the default case.
#define CAMERA_MODEL_CASE(CameraModel)         \
  case CameraModel::model_id:                  \
    return {CameraModel::metadata_idxs.data(), \
            CameraModel::metadata_idxs.size()};
    SPHERICAL_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
#define CAMERA_MODEL_CASE(CameraModel) case CameraModel::model_id:
    PERSPECTIVE_CAMERA_MODEL_CASES
#undef CAMERA_MODEL_CASE
    return {nullptr, 0};
    default:
      CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION
  }
  return {nullptr, 0};
}

size_t CameraModelNumParams(const CameraModelId model_id) {
  switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::num_params;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }

  return 0;
}

bool CameraModelVerifyParams(const CameraModelId model_id,
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

bool CameraModelHasBogusParams(const CameraModelId model_id,
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
