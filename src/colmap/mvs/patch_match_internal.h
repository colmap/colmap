// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/hash_containers.h"

#include <vector>

namespace colmap {
namespace mvs {
namespace internal {

// Preserve the first occurrence of each available non-reference source.
// Availability is already resolved by the controller's input policy.
inline std::vector<int> SelectPatchMatchSources(
    const int ref_image_idx,
    const std::vector<int>& configured_image_idxs,
    FlatHashSet<int> available_image_idxs) {
  std::vector<int> src_image_idxs;
  for (const int image_idx : configured_image_idxs) {
    if (image_idx != ref_image_idx &&
        available_image_idxs.erase(image_idx) > 0) {
      src_image_idxs.push_back(image_idx);
    }
  }
  return src_image_idxs;
}

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
