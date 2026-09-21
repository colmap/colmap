// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/geometry/bbox.h"

#include "colmap/util/logging.h"

namespace colmap {

std::vector<Eigen::AlignedBox3d> ComputeEqualPartsBboxes(
    const Eigen::AlignedBox3d& bbox, const Eigen::Vector3i& split) {
  THROW_CHECK_GT(split(0), 0);
  THROW_CHECK_GT(split(1), 0);
  THROW_CHECK_GT(split(2), 0);

  const Eigen::Vector3d extent = bbox.diagonal();
  const Eigen::Vector3d size(
      extent(0) / split(0), extent(1) / split(1), extent(2) / split(2));

  std::vector<Eigen::AlignedBox3d> bboxes;
  bboxes.reserve(split(0) * split(1) * split(2));
  for (int k = 0; k < split(2); ++k) {
    for (int j = 0; j < split(1); ++j) {
      for (int i = 0; i < split(0); ++i) {
        Eigen::Vector3d min(bbox.min().x() + i * size(0),
                            bbox.min().y() + j * size(1),
                            bbox.min().z() + k * size(2));
        bboxes.emplace_back(min, min + size);
      }
    }
  }

  return bboxes;
}

}  // namespace colmap
