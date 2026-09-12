// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/util/hash_containers.h"

#include <string>
#include <vector>

namespace colmap {

// { make1 : ({ model1 : sensor-width in mm }, ...), ... }
using camera_make_specs_t = std::vector<std::pair<std::string, float>>;
using camera_specs_t = NodeHashMap<std::string, camera_make_specs_t>;

camera_specs_t InitializeCameraSpecs();

}  // namespace colmap
