#pragma once

#include "colmap/math/union_find.h"

#include "glomap/scene/view_graph.h"

namespace glomap {

struct TrackEstablishmentOptions {
  // the max allowed distance for features in the same track in the same image
  double thres_inconsistency = 10.;

  // The minimal number of tracks for each view,
  int min_num_tracks_per_view = -1;

  // The minimal number of tracks for each view pair
  int min_num_view_per_track = 3;

  // The maximal number of tracks for each view pair
  int max_num_view_per_track = 100;

  // The maximal number of tracks
  int max_num_tracks = 10000000;
};

class TrackEngine {
 public:
  TrackEngine(const ViewGraph& view_graph,
              const std::unordered_map<image_t, Image>& images,
              const TrackEstablishmentOptions& options)
      : options_(options), view_graph_(view_graph), images_(images) {}

  // Establish tracks from the view graph. Exclude the tracks that are not
  // consistent Return the number of tracks
  size_t EstablishFullTracks(std::unordered_map<point3D_t, Point3D>& tracks);

  // Subsample the tracks, and exclude too short / long tracks
  // Return the number of tracks
  size_t FindTracksForProblem(
      const std::unordered_map<point3D_t, Point3D>& tracks_full,
      std::unordered_map<point3D_t, Point3D>& tracks_selected);

 private:
  // Blindly concatenate tracks if any matches occur
  void BlindConcatenation();

  // Iterate through the collected tracks and record the items for each track
  void TrackCollection(std::unordered_map<point3D_t, Point3D>& tracks);

  const TrackEstablishmentOptions& options_;

  const ViewGraph& view_graph_;
  const std::unordered_map<image_t, Image>& images_;

  // Internal structure used for concatenating tracks
  colmap::UnionFind<image_pair_t> uf_;
};

}  // namespace glomap
