#pragma once

#include "glomap/scene/pose_graph.h"

#include <limits>
#include <vector>

#include <Eigen/Core>

namespace glomap {

struct TrackEstablishmentOptions {
  // Max pixel distance between observations of the same track within one image.
  double intra_image_consistency_threshold = 10.;

  // Required number of tracks per view before early stopping.
  int required_tracks_per_view = std::numeric_limits<int>::max();

  // Minimum number of views per track.
  int min_num_views_per_track = 3;
};

// Establish tracks from feature matches in the pose graph.
// Creates tracks using union-find, validates consistency, and filters
// to select tracks suitable for the optimization problem.
// image_id_to_keypoints: map from image_id to vector of 2D points (indexed by
// point2D_idx).
size_t EstablishTracks(
    const PoseGraph& pose_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>&
        image_id_to_keypoints,
    const TrackEstablishmentOptions& options,
    std::unordered_map<point3D_t, Point3D>& points3D);

}  // namespace glomap
