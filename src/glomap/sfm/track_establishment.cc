#include "glomap/sfm/track_establishment.h"

#include "colmap/util/logging.h"

namespace glomap {

size_t TrackEngine::EstablishFullTracks(
    std::unordered_map<point3D_t, Point3D>& tracks) {
  tracks.clear();
  uf_ = {};

  // Blindly concatenate tracks if any matches occur
  BlindConcatenation();

  // Iterate through the collected tracks and record the items for each track
  TrackCollection(tracks);

  return tracks.size();
}

void TrackEngine::BlindConcatenation() {
  // Initialize the union find data structure by connecting all the
  // correspondences
  image_pair_t counter = 0;
  for (const auto& [pair_id, image_pair] : view_graph_.ImagePairs()) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.NumImagePairs() - 1) {
      std::cout << "\r Initializing pairs " << counter + 1 << " / "
                << view_graph_.NumImagePairs() << std::flush;
    }
    counter++;

    if (!view_graph_.IsValid(pair_id)) continue;

    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    // Get the matches
    const Eigen::MatrixXi& matches = image_pair.matches;

    // Get the inlier mask
    const std::vector<int>& inliers = image_pair.inliers;

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // Get point indices
      const uint32_t& point1_idx = matches(idx, 0);
      const uint32_t& point2_idx = matches(idx, 1);

      image_pair_t point_global_id1 = static_cast<image_pair_t>(image_id1)
                                          << 32 |
                                      static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 = static_cast<image_pair_t>(image_id2)
                                          << 32 |
                                      static_cast<image_pair_t>(point2_idx);

      // Link the first point to the second point. Take the smallest one as the
      // root
      if (point_global_id2 < point_global_id1) {
        uf_.Union(point_global_id1, point_global_id2);
      } else
        uf_.Union(point_global_id2, point_global_id1);
    }
  }
  LOG(INFO) << "Initialized " << view_graph_.NumImagePairs() << " pairs";
}

void TrackEngine::TrackCollection(
    std::unordered_map<point3D_t, Point3D>& tracks) {
  const auto& images = images_;
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> track_map;
  std::unordered_map<uint64_t, int> track_counter;

  // Create tracks from the connected components of the point correspondences
  size_t counter = 0;
  for (const auto& [pair_id, image_pair] : view_graph_.ImagePairs()) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.NumImagePairs() - 1) {
      std::cout << "\r Establishing pairs " << counter + 1 << " / "
                << view_graph_.NumImagePairs() << std::flush;
    }
    counter++;

    if (!view_graph_.IsValid(pair_id)) continue;

    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    // Get the matches
    const Eigen::MatrixXi& matches = image_pair.matches;

    // Get the inlier mask
    const std::vector<int>& inliers = image_pair.inliers;

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // Get point indices
      const uint32_t& point1_idx = matches(idx, 0);
      const uint32_t& point2_idx = matches(idx, 1);

      image_pair_t point_global_id1 = static_cast<image_pair_t>(image_id1)
                                          << 32 |
                                      static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 = static_cast<image_pair_t>(image_id2)
                                          << 32 |
                                      static_cast<image_pair_t>(point2_idx);

      image_pair_t track_id = uf_.Find(point_global_id1);

      track_map[track_id].insert(point_global_id1);
      track_map[track_id].insert(point_global_id2);
      track_counter[track_id]++;
    }
  }
  LOG(INFO) << "Established " << view_graph_.NumImagePairs() << " pairs";

  size_t discarded_counter = 0;
  for (const auto& [track_id, correspondence_set] : track_map) {
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    for (const uint64_t point_global_id : correspondence_set) {
      // image_id is the higher 32 bits and feature_id is the lower 32 bits
      const image_t image_id = point_global_id >> 32;
      const uint64_t feature_id = point_global_id & 0xFFFFFFFF;
      if (image_id_set.find(image_id) != image_id_set.end()) {
        for (const auto& feature : image_id_set.at(image_id)) {
          if ((feature - images.at(image_id).Point2D(feature_id).xy).norm() >
              options_.thres_inconsistency) {
            tracks[track_id].track.SetElements({});
            break;
          }
        }
        if (tracks[track_id].track.Length() == 0) {
          discarded_counter++;
          break;
        }
      } else
        image_id_set.insert(
            std::make_pair(image_id, std::vector<Eigen::Vector2d>()));

      image_id_set[image_id].push_back(
          images.at(image_id).Point2D(feature_id).xy);

      tracks[track_id].track.AddElement(image_id, feature_id);
    }
  }

  LOG(INFO) << "Established " << track_map.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";
}

size_t TrackEngine::FindTracksForProblem(
    const std::unordered_map<point3D_t, Point3D>& tracks_full,
    std::unordered_map<point3D_t, Point3D>& tracks_selected) {
  const auto& images = images_;

  // Sort the tracks by length
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  for (const auto& [track_id, track] : tracks_full) {
    if (track.track.Length() < options_.min_num_view_per_track) continue;
    // FUTURE: have a more elegant way of filtering tracks
    if (track.track.Length() > options_.max_num_view_per_track) continue;
    track_lengths.emplace_back(track.track.Length(), track_id);
  }
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  // Initialize the track per camera number to zero
  std::unordered_map<image_t, point3D_t> tracks_per_camera;

  // If we only want to select a subset of images, then only add the tracks
  // corresponding to those images
  std::unordered_map<point3D_t, Point3D> tracks;
  for (const auto& [image_id, image] : images) {
    if (!image.HasPose()) continue;
    tracks_per_camera[image_id] = 0;
  }

  int cameras_left = tracks_per_camera.size();
  for (const auto& [track_length, track_id] : track_lengths) {
    const auto& track = tracks_full.at(track_id);

    // Collect the image ids. For each image, only increment the counter by 1
    std::unordered_set<image_t> image_ids;
    Point3D track_temp;
    for (const auto& observation : track.track.Elements()) {
      if (tracks_per_camera.count(observation.image_id) == 0) continue;

      track_temp.track.AddElement(observation.image_id,
                                  observation.point2D_idx);
      image_ids.insert(observation.image_id);
    }

    if (image_ids.size() < options_.min_num_view_per_track) continue;

    // A flag to see if the track has already been added or not to avoid
    // multiple insertion into the set to be efficient
    bool added = false;
    // for (auto &image_id : image_ids) {
    for (const auto& observation : track_temp.track.Elements()) {
      // Getting the current number of tracks
      auto& track_per_camera = tracks_per_camera[observation.image_id];
      if (track_per_camera > options_.min_num_tracks_per_view) continue;

      // Otherwise, increase the track number per camera
      ++track_per_camera;
      if (track_per_camera > options_.min_num_tracks_per_view) --cameras_left;

      if (!added) {
        tracks.insert(std::make_pair(track_id, track_temp));
        added = true;
      }
    }
    // Stop iterating if all cameras have enough tracks assigned
    if (cameras_left == 0) break;
    if (tracks.size() > options_.max_num_tracks) break;
  }

  // Move the selected tracks to the output
  size_t num_tracks = tracks.size();
  tracks_selected = std::move(tracks);

  return num_tracks;
}

}  // namespace glomap
