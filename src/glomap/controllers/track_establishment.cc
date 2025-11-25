#include "track_establishment.h"

namespace glomap {

size_t TrackEngine::EstablishFullTracks(
    std::unordered_map<track_t, Track>& tracks) {
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
  for (const auto& pair : view_graph_.image_pairs) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.image_pairs.size() - 1) {
      std::cout << "\r Initializing pairs " << counter + 1 << " / "
                << view_graph_.image_pairs.size() << std::flush;
    }
    counter++;

    const ImagePair& image_pair = pair.second;
    if (!image_pair.is_valid) continue;

    // Get the matches
    const Eigen::MatrixXi& matches = image_pair.matches;

    // Get the inlier mask
    const std::vector<int>& inliers = image_pair.inliers;

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // Get point indices
      const uint32_t& point1_idx = matches(idx, 0);
      const uint32_t& point2_idx = matches(idx, 1);

      image_pair_t point_global_id1 =
          static_cast<image_pair_t>(image_pair.image_id1) << 32 |
          static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 =
          static_cast<image_pair_t>(image_pair.image_id2) << 32 |
          static_cast<image_pair_t>(point2_idx);

      // Link the first point to the second point. Take the smallest one as the
      // root
      if (point_global_id2 < point_global_id1) {
        uf_.Union(point_global_id1, point_global_id2);
      } else
        uf_.Union(point_global_id2, point_global_id1);
    }
  }
  std::cout << '\n';
}

void TrackEngine::TrackCollection(std::unordered_map<track_t, Track>& tracks) {
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> track_map;
  std::unordered_map<uint64_t, int> track_counter;

  // Create tracks from the connected components of the point correspondences
  size_t counter = 0;
  for (const auto& pair : view_graph_.image_pairs) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.image_pairs.size() - 1) {
      std::cout << "\r Establishing pairs " << counter + 1 << " / "
                << view_graph_.image_pairs.size() << std::flush;
    }
    counter++;

    const ImagePair& image_pair = pair.second;
    if (!image_pair.is_valid) continue;

    // Get the matches
    const Eigen::MatrixXi& matches = image_pair.matches;

    // Get the inlier mask
    const std::vector<int>& inliers = image_pair.inliers;

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // Get point indices
      const uint32_t& point1_idx = matches(idx, 0);
      const uint32_t& point2_idx = matches(idx, 1);

      image_pair_t point_global_id1 =
          static_cast<image_pair_t>(image_pair.image_id1) << 32 |
          static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 =
          static_cast<image_pair_t>(image_pair.image_id2) << 32 |
          static_cast<image_pair_t>(point2_idx);

      image_pair_t track_id = uf_.Find(point_global_id1);

      track_map[track_id].insert(point_global_id1);
      track_map[track_id].insert(point_global_id2);
      track_counter[track_id]++;
    }
  }
  std::cout << '\n';

  counter = 0;
  size_t discarded_counter = 0;
  for (auto& [track_id, correspondence_set] : track_map) {
    if ((counter + 1) % 1000 == 0 || counter == track_map.size() - 1) {
      std::cout << "\r Establishing tracks " << counter + 1 << " / "
                << track_map.size() << std::flush;
    }
    counter++;

    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    for (auto point_global_id : correspondence_set) {
      // image_id is the higher 32 bits and feature_id is the lower 32 bits
      image_t image_id = point_global_id >> 32;
      feature_t feature_id = point_global_id & 0xFFFFFFFF;
      if (image_id_set.find(image_id) != image_id_set.end()) {
        for (const auto& feature : image_id_set.at(image_id)) {
          if ((feature - images_.at(image_id).features[feature_id]).norm() >
              options_.thres_inconsistency) {
            tracks[track_id].observations.clear();
            break;
          }
        }
        if (tracks[track_id].observations.size() == 0) {
          discarded_counter++;
          break;
        }
      } else
        image_id_set.insert(
            std::make_pair(image_id, std::vector<Eigen::Vector2d>()));

      image_id_set[image_id].push_back(
          images_.at(image_id).features[feature_id]);

      tracks[track_id].observations.emplace_back(image_id, feature_id);
    }
  }

  std::cout << '\n';
  LOG(INFO) << "Discarded " << discarded_counter
            << " tracks due to inconsistency";
}

size_t TrackEngine::FindTracksForProblem(
    const std::unordered_map<track_t, Track>& tracks_full,
    std::unordered_map<track_t, Track>& tracks_selected) {
  // Sort the tracks by length
  std::vector<std::pair<size_t, track_t>> track_lengths;

  // std::unordered_map<ViewId, std::vector<TrackId>> map_track;
  for (const auto& [track_id, track] : tracks_full) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;
    // FUTURE: have a more elegant way of filtering tracks
    if (track.observations.size() > options_.max_num_view_per_track) continue;
    track_lengths.emplace_back(
        std::make_pair(track.observations.size(), track_id));
  }
  std::sort(std::rbegin(track_lengths), std::rend(track_lengths));

  // Initialize the track per camera number to zero
  std::unordered_map<image_t, track_t> tracks_per_camera;

  // If we only want to select a subset of images, then only add the tracks
  // corresponding to those images
  std::unordered_map<track_t, Track> tracks;
  for (const auto& [image_id, image] : images_) {
    if (!image.IsRegistered()) continue;

    tracks_per_camera[image_id] = 0;
  }

  int cameras_left = tracks_per_camera.size();
  for (const auto& [track_length, track_id] : track_lengths) {
    const auto& track = tracks_full.at(track_id);

    // Collect the image ids. For each image, only increment the counter by 1
    std::unordered_set<image_t> image_ids;
    Track track_temp;
    for (const auto& [image_id, feature_id] : track.observations) {
      if (tracks_per_camera.count(image_id) == 0) continue;

      track_temp.track_id = track_id;
      track_temp.observations.emplace_back(
          std::make_pair(image_id, feature_id));
      image_ids.insert(image_id);
    }

    if (image_ids.size() < options_.min_num_view_per_track) continue;

    // A flag to see if the track has already been added or not to avoid
    // multiple insertion into the set to be efficient
    bool added = false;
    // for (auto &image_id : image_ids) {
    for (const auto& [image_id, feature_id] : track_temp.observations) {
      // Getting the current number of tracks
      auto& track_per_camera = tracks_per_camera[image_id];
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

  // To avoid flushing the track_full, we copy the selected tracks to the
  // selected tracks
  tracks_selected = tracks;

  return tracks.size();
}

}  // namespace glomap
