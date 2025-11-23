#pragma once

#include "glomap/scene/types.h"

#include <vector>

#include <Eigen/Core>

namespace glomap {
typedef std::pair<image_t, feature_t> Observation;

struct Track {
  // The id of the track
  track_t track_id;

  // The 3D point of the track
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();

  // The color of the track (now not used)
  Eigen::Vector3ub color = Eigen::Vector3ub::Zero();

  // Whether the track has been estimated
  bool is_initialized = false;

  // The list where the track is observed (image_id, feature_id)
  std::vector<Observation> observations;
};

}  // namespace glomap
