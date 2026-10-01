// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/scene/reconstruction.h"

namespace colmap {

std::vector<point3D_t> FindRedundantPoints3D(
    double min_coverage_gain, const Reconstruction& reconstruction);

}  // namespace colmap
