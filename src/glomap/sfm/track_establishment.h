#pragma once

#include "colmap/scene/image.h"

#include "glomap/scene/pose_graph.h"

#include <limits>

namespace glomap {

struct TrackEstablishmentOptions {
  // Max pixel distance between observations of the same track within one image.
  double intra_image_consistency_threshold = 10.;

  // Required number of tracks per view before early stopping.
  size_t required_tracks_per_view = std::numeric_limits<size_t>::max();

  // Minimal number of views per track.
  size_t min_num_views_per_track = 3;

  // Maximal number of views per track.
  size_t max_num_views_per_track = 100;

  // Maximal number of tracks.
  size_t max_num_tracks = std::numeric_limits<size_t>::max();
};

// Establish tracks from feature matches in the pose graph.
// Creates tracks using union-find, validates consistency, and filters
// to select tracks suitable for the optimization problem.
size_t EstablishTracks(const PoseGraph& pose_graph,
                       const std::unordered_map<image_t, colmap::Image>& images,
                       const TrackEstablishmentOptions& options,
                       std::unordered_map<point3D_t, Point3D>& points3D);

}  // namespace glomap
