#include "glomap/sfm/track_establishment.h"

#include "colmap/util/logging.h"

namespace glomap {

// Type alias for a 2D observation: (image_id, point2D_idx).
using Observation = std::pair<image_t, colmap::point2D_t>;

size_t TrackEngine::EstablishFullTracks(
    std::unordered_map<point3D_t, Point3D>& points3D) {
  points3D.clear();
  uf_ = {};

  // Blindly concatenate tracks if any matches occur
  BlindConcatenation();

  // Iterate through the collected tracks and record the items for each track
  TrackCollection(points3D);

  return points3D.size();
}

void TrackEngine::BlindConcatenation() {
  // Initialize the union find data structure by connecting all the
  // correspondences
  size_t counter = 0;
  for (const auto& [pair_id, image_pair] : view_graph_.ImagePairs()) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.NumImagePairs() - 1) {
      VLOG(1) << "Initializing pairs " << counter + 1 << " / "
              << view_graph_.NumImagePairs();
    }
    counter++;

    if (!view_graph_.IsValid(pair_id)) continue;

    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    for (const auto& match : image_pair.inlier_matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);

      // Link the two observations. Use consistent ordering for root selection.
      if (obs2 < obs1) {
        uf_.Union(obs1, obs2);
      } else {
        uf_.Union(obs2, obs1);
      }
    }
  }
  LOG(INFO) << "Initialized " << view_graph_.NumImagePairs() << " pairs";
}

void TrackEngine::TrackCollection(
    std::unordered_map<point3D_t, Point3D>& points3D) {
  const auto& images = images_;
  std::unordered_map<Observation, std::unordered_set<Observation>> track_map;

  // Create tracks from the connected components of the point correspondences
  size_t counter = 0;
  for (const auto& [pair_id, image_pair] : view_graph_.ImagePairs()) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.NumImagePairs() - 1) {
      VLOG(1) << "Establishing pairs " << counter + 1 << " / "
              << view_graph_.NumImagePairs();
    }
    counter++;

    if (!view_graph_.IsValid(pair_id)) continue;

    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    for (const auto& match : image_pair.inlier_matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);

      const Observation track_id = uf_.Find(obs1);

      track_map[track_id].insert(obs1);
      track_map[track_id].insert(obs2);
    }
  }
  LOG(INFO) << "Established " << view_graph_.NumImagePairs() << " pairs";

  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;
  for (const auto& [track_id, correspondence_set] : track_map) {
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : correspondence_set) {
      const Eigen::Vector2d& xy = images.at(image_id).Point2D(feature_id).xy;

      // Check consistency: if this image already has observations,
      // verify the new one is close enough to existing ones.
      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          if ((existing_xy - xy).norm() > options_.thres_inconsistency) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          discarded_counter++;
          break;
        }
        it->second.push_back(xy);
      } else {
        image_id_set[image_id].push_back(xy);
      }

      point3D.track.AddElement(image_id, feature_id);
    }

    if (is_consistent) {
      points3D.emplace(next_point3D_id++, std::move(point3D));
    }
  }

  LOG(INFO) << "Established " << track_map.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";
}

size_t TrackEngine::FindTracksForProblem(
    const std::unordered_map<point3D_t, Point3D>& points3D_full,
    std::unordered_map<point3D_t, Point3D>& points3D_selected) {
  const auto& images = images_;

  // Sort the tracks by length
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  for (const auto& [point3D_id, point3D] : points3D_full) {
    if (point3D.track.Length() < options_.min_num_view_per_track) continue;
    // FUTURE: have a more elegant way of filtering tracks
    if (point3D.track.Length() > options_.max_num_view_per_track) continue;
    track_lengths.emplace_back(point3D.track.Length(), point3D_id);
  }
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  // Initialize the point3D per camera number to zero
  std::unordered_map<image_t, point3D_t> points3D_per_camera;

  // If we only want to select a subset of images, then only add the points3D
  // corresponding to those images
  std::unordered_map<point3D_t, Point3D> points3D;
  for (const auto& [image_id, image] : images) {
    if (!image.HasPose()) continue;
    points3D_per_camera[image_id] = 0;
  }

  int cameras_left = points3D_per_camera.size();
  for (const auto& [track_length, point3D_id] : track_lengths) {
    const auto& point3D = points3D_full.at(point3D_id);

    // Collect the image ids. For each image, only increment the counter by 1
    std::unordered_set<image_t> image_ids;
    Point3D point3D_temp;
    for (const auto& observation : point3D.track.Elements()) {
      if (points3D_per_camera.count(observation.image_id) == 0) continue;

      point3D_temp.track.AddElement(observation.image_id,
                                    observation.point2D_idx);
      image_ids.insert(observation.image_id);
    }

    if (image_ids.size() < options_.min_num_view_per_track) continue;

    // A flag to see if the point3D has already been added or not to avoid
    // multiple insertion into the set to be efficient
    bool added = false;
    // for (auto &image_id : image_ids) {
    for (const auto& observation : point3D_temp.track.Elements()) {
      // Getting the current number of points3D
      auto& num_points3D = points3D_per_camera[observation.image_id];
      if (num_points3D > options_.min_num_tracks_per_view) continue;

      // Otherwise, increase the point3D number per camera
      ++num_points3D;
      if (num_points3D > options_.min_num_tracks_per_view) --cameras_left;

      if (!added) {
        points3D.insert(std::make_pair(point3D_id, point3D_temp));
        added = true;
      }
    }
    // Stop iterating if all cameras have enough points3D assigned
    if (cameras_left == 0) break;
    if (points3D.size() > options_.max_num_tracks) break;
  }

  // Move the selected points3D to the output
  size_t num_points3D = points3D.size();
  points3D_selected = std::move(points3D);

  return num_points3D;
}

}  // namespace glomap
