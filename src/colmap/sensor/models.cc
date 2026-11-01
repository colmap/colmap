// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/models.h"

#include "colmap/util/hash_containers.h"

#include <cmath>

namespace colmap {

namespace {

bool CameraModelParamsAreFinite(const std::vector<double>& params) {
  for (const double param : params) {
    if (!std::isfinite(param)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool ExistsCameraModelWithName(const std::string& model_name) {
  return CameraModelNameToId(model_name) != CameraModelId::kInvalid;
}

bool ExistsCameraModelWithId(const CameraModelId model_id) {
  return VisitCameraModelOrValue(
      model_id, [](auto /*tag*/) { return true; }, false);
}

CameraModelId CameraModelNameToId(const std::string& model_name) {
  // Function-local static: built once on first use (thread-safe), keeping
  // the name lookup O(1) without paying for eager static initialization.
  static const NodeHashMap<std::string, CameraModelId> kNameToId = [] {
    NodeHashMap<std::string, CameraModelId> name_to_id;
    ForEachCameraModel([&](auto tag) {
      using CameraModel = typename decltype(tag)::type;
      name_to_id.emplace(CameraModel::model_name, CameraModel::model_id);
    });
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
  const static std::string kEmptyModelName = "";
  return VisitCameraModelOrValue(
      model_id,
      [](auto tag) -> const std::string& {
        using CameraModel = typename decltype(tag)::type;
        return CameraModel::model_name;
      },
      kEmptyModelName);
}

std::vector<double> CameraModelInitializeParams(const CameraModelId model_id,
                                                const double focal_length,
                                                const size_t width,
                                                const size_t height) {
  // Assuming that image measurements are within [0, dim], i.e. that the
  // upper left corner is the (0, 0) coordinate (rather than the center of
  // the upper left pixel). This complies with the default SiftGPU convention.
  return VisitCameraModelOrThrow(model_id, [&](auto tag) {
    using CameraModel = typename decltype(tag)::type;
    return CameraModel::InitializeParams(focal_length, width, height);
  });
}

const std::string& CameraModelParamsInfo(const CameraModelId model_id) {
  return VisitCameraModelOrThrow(model_id, [](auto tag) -> const std::string& {
    using CameraModel = typename decltype(tag)::type;
    return CameraModel::params_info;
  });
}

span<const size_t> CameraModelFocalLengthIdxs(const CameraModelId model_id) {
  // Only perspective models have focal-length parameters; spherical models
  // have none. Unknown models throw.
  return VisitCameraModelOrThrow(model_id, [](auto tag) -> span<const size_t> {
    using CameraModel = typename decltype(tag)::type;
    if constexpr (std::is_base_of_v<BasePerspectiveCameraModel<CameraModel>,
                                    CameraModel>) {
      return {CameraModel::focal_length_idxs.data(),
              CameraModel::focal_length_idxs.size()};
    } else {
      return {nullptr, 0};
    }
  });
}

span<const size_t> CameraModelPrincipalPointIdxs(const CameraModelId model_id) {
  // Only perspective models have principal-point parameters; spherical models
  // have none. Unknown models throw.
  return VisitCameraModelOrThrow(model_id, [](auto tag) -> span<const size_t> {
    using CameraModel = typename decltype(tag)::type;
    if constexpr (std::is_base_of_v<BasePerspectiveCameraModel<CameraModel>,
                                    CameraModel>) {
      return {CameraModel::principal_point_idxs.data(),
              CameraModel::principal_point_idxs.size()};
    } else {
      return {nullptr, 0};
    }
  });
}

span<const size_t> CameraModelExtraParamsIdxs(const CameraModelId model_id) {
  // Only perspective models have extra parameters; spherical models have
  // none. Unknown models throw.
  return VisitCameraModelOrThrow(model_id, [](auto tag) -> span<const size_t> {
    using CameraModel = typename decltype(tag)::type;
    if constexpr (std::is_base_of_v<BasePerspectiveCameraModel<CameraModel>,
                                    CameraModel>) {
      return {CameraModel::extra_params_idxs.data(),
              CameraModel::extra_params_idxs.size()};
    } else {
      return {nullptr, 0};
    }
  });
}

span<const size_t> CameraModelMetaDataParamsIdxs(const CameraModelId model_id) {
  // Only spherical models have metadata parameters; perspective models have
  // none. Unknown models throw.
  return VisitCameraModelOrThrow(model_id, [](auto tag) -> span<const size_t> {
    using CameraModel = typename decltype(tag)::type;
    if constexpr (std::is_base_of_v<BaseSphericalCameraModel<CameraModel>,
                                    CameraModel>) {
      return {CameraModel::metadata_idxs.data(),
              CameraModel::metadata_idxs.size()};
    } else {
      return {nullptr, 0};
    }
  });
}

size_t CameraModelNumParams(const CameraModelId model_id) {
  return VisitCameraModelOrThrow(model_id, [](auto tag) {
    using CameraModel = typename decltype(tag)::type;
    return CameraModel::num_params;
  });
}

bool CameraModelVerifyParams(const CameraModelId model_id,
                             const std::vector<double>& params) {
  return VisitCameraModelOrThrow(model_id, [&](auto tag) {
    using CameraModel = typename decltype(tag)::type;
    return params.size() == CameraModel::num_params &&
           CameraModelParamsAreFinite(params);
  });
}

bool CameraModelHasBogusParams(const CameraModelId model_id,
                               const std::vector<double>& params,
                               const size_t width,
                               const size_t height,
                               const double min_focal_length_ratio,
                               const double max_focal_length_ratio,
                               const double max_extra_param) {
  return VisitCameraModelOrThrow(model_id, [&](auto tag) {
    using CameraModel = typename decltype(tag)::type;
    return !CameraModelParamsAreFinite(params) ||
           CameraModel::HasBogusParams(params,
                                       width,
                                       height,
                                       min_focal_length_ratio,
                                       max_focal_length_ratio,
                                       max_extra_param);
  });
}

}  // namespace colmap
