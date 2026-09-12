// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/scene/two_view_geometry.h"

namespace colmap {

// Out-of-line on purpose; see the header.
TwoViewGeometry::TwoViewGeometry(const TwoViewGeometry&) = default;
TwoViewGeometry::TwoViewGeometry(TwoViewGeometry&&) noexcept = default;
TwoViewGeometry& TwoViewGeometry::operator=(const TwoViewGeometry&) = default;
TwoViewGeometry& TwoViewGeometry::operator=(TwoViewGeometry&&) noexcept =
    default;
TwoViewGeometry::~TwoViewGeometry() = default;

void TwoViewGeometry::Invert() {
  if (F) {
    F->transposeInPlace();
  }
  if (E) {
    E->transposeInPlace();
  }
  if (H) {
    *H = H->inverse().eval();
  }
  if (cam2_from_cam1) {
    cam2_from_cam1 = Inverse(*cam2_from_cam1);
  }
  camera1.swap(camera2);
  for (auto& match : inlier_matches) {
    std::swap(match.point2D_idx1, match.point2D_idx2);
  }
}

}  // namespace colmap
