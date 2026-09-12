// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"

#include <functional>
#include <vector>

namespace colmap {

// Helper method to extract sorted camera, image, point3D identifiers.
// We sort the identifiers before writing to the stream, such that we produce
// deterministic output independent of standard library dependent ordering of
// the unordered map container.
template <typename ID_TYPE, typename DATA_TYPE>
std::vector<ID_TYPE> ExtractSortedIds(
    const NodeHashMap<ID_TYPE, DATA_TYPE>& data,
    const std::function<bool(const DATA_TYPE&)>& filter = nullptr) {
  std::vector<ID_TYPE> ids;
  ids.reserve(data.size());
  for (const auto& [id, d] : data) {
    if (filter == nullptr || filter(d)) {
      ids.push_back(id);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

void CreateOneRigPerCamera(Reconstruction& reconstruction);

void CreateFrameForImage(const Image& image,
                         const Rigid3d& cam_from_world,
                         Reconstruction& reconstruction);

NodeHashMap<image_t, Frame*> ExtractImageToFramePtr(
    Reconstruction& reconstruction);

}  // namespace colmap
