#include "glomap/sfm/track_establishment.h"

#include "colmap/math/union_find.h"
#include "colmap/util/logging.h"

#include <algorithm>

namespace glomap {
namespace {

using Observation = std::pair<image_t, colmap::point2D_t>;

size_t EstablishFullTracks(
    const PoseGraph& pose_graph,
    const std::unordered_map<image_t, colmap::Image>& images,
    const TrackEstablishmentOptions& options,
    std::unordered_map<point3D_t, Point3D>& points3D) {
  colmap::UnionFind<Observation> uf;

  // Single pass: union all matching observations
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

    for (const auto& match : edge.inlier_matches) {
      const Observation obs1(image_id1, match.point2D_idx1);
      const Observation obs2(image_id2, match.point2D_idx2);

      // Link the two observations. Use consistent ordering for root selection.
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    }
  }

  // Group observations by their root
  std::unordered_map<Observation, std::vector<Observation>> track_map;
  for (const auto& [obs, parent] : uf.Elements()) {
    track_map[uf.Find(obs)].push_back(obs);
  }
  LOG(INFO) << "Established " << track_map.size() << " tracks from "
            << uf.Elements().size() << " observations";

  // Validate tracks and create Point3D objects
  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;
  for (const auto& [track_id, observations] : track_map) {
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : observations) {
      const Eigen::Vector2d& xy = images.at(image_id).Point2D(feature_id).xy;

      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          if ((existing_xy - xy).norm() >
              options.intra_image_consistency_threshold) {
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

  LOG(INFO) << "Kept " << points3D.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";
  return points3D.size();
}

size_t SelectTracksForProblem(
    const std::unordered_map<image_t, colmap::Image>& images,
    const TrackEstablishmentOptions& options,
    const std::unordered_map<point3D_t, Point3D>& points3D_full,
    std::unordered_map<point3D_t, Point3D>& points3D_selected) {
  // Sort tracks by length (descending)
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  for (const auto& [point3D_id, point3D] : points3D_full) {
    const size_t length = point3D.track.Length();
    if (length < options.min_num_views_per_track) continue;
    if (length > options.max_num_views_per_track) continue;
    track_lengths.emplace_back(length, point3D_id);
  }
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  // Track how many points each camera has
  std::unordered_map<image_t, point3D_t> points3D_per_camera;
  for (const auto& [image_id, image] : images) {
    if (!image.HasPose()) continue;
    points3D_per_camera[image_id] = 0;
  }

  size_t cameras_left = points3D_per_camera.size();
  for (const auto& [track_length, point3D_id] : track_lengths) {
    const auto& point3D = points3D_full.at(point3D_id);

    // Filter observations to only include images with poses
    std::unordered_set<image_t> image_ids;
    Point3D point3D_filtered;
    for (const auto& obs : point3D.track.Elements()) {
      if (points3D_per_camera.count(obs.image_id) == 0) continue;
      point3D_filtered.track.AddElement(obs.image_id, obs.point2D_idx);
      image_ids.insert(obs.image_id);
    }

    if (image_ids.size() < options.min_num_views_per_track) continue;

    // Check if any camera in this track still needs more observations
    const bool should_add =
        std::any_of(point3D_filtered.track.Elements().begin(),
                    point3D_filtered.track.Elements().end(),
                    [&](const auto& obs) {
                      return points3D_per_camera[obs.image_id] <=
                             options.required_tracks_per_view;
                    });
    if (!should_add) continue;

    // Add track and update camera counts
    points3D_selected.emplace(point3D_id, std::move(point3D_filtered));
    for (const auto& obs : points3D_selected.at(point3D_id).track.Elements()) {
      auto& count = points3D_per_camera[obs.image_id];
      if (count == options.required_tracks_per_view) --cameras_left;
      ++count;
    }

    if (cameras_left == 0) break;
    if (points3D_selected.size() > options.max_num_tracks) break;
  }

  return points3D_selected.size();
}

}  // namespace

size_t EstablishTracks(const PoseGraph& pose_graph,
                       const std::unordered_map<image_t, colmap::Image>& images,
                       const TrackEstablishmentOptions& options,
                       std::unordered_map<point3D_t, Point3D>& points3D) {
  std::unordered_map<point3D_t, Point3D> unfiltered_points3D;
  EstablishFullTracks(pose_graph, images, options, unfiltered_points3D);

  points3D.clear();
  size_t num_selected =
      SelectTracksForProblem(images, options, unfiltered_points3D, points3D);

  LOG(INFO) << "Before filtering: " << unfiltered_points3D.size()
            << ", after filtering: " << num_selected;
  return num_selected;
}

}  // namespace glomap
