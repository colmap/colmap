#include "glomap/processors/reconstruction_pruning.h"

#include "glomap/processors/view_graph_manipulation.h"

namespace glomap {
void PruneWeaklyConnectedImages(colmap::Reconstruction& reconstruction,
                                std::unordered_map<frame_t, int>& cluster_ids,
                                int min_num_images,
                                int min_num_observations) {
  // Prepare the 2d-3d correspondences
  std::unordered_map<image_pair_t, int> pair_covisibility_count;
  std::unordered_map<frame_t, int> frame_observation_count;
  for (const auto& [track_id, track] : reconstruction.Points3D()) {
    if (track.track.Length() <= 2) continue;

    for (size_t i = 0; i < track.track.Length(); i++) {
      const image_t image_id1 = track.track.Element(i).image_id;
      const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();

      frame_observation_count[frame_id1]++;
      for (size_t j = i + 1; j < track.track.Length(); j++) {
        const image_t image_id2 = track.track.Element(j).image_id;
        const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
        if (frame_id1 == frame_id2) continue;
        const image_pair_t pair_id =
            colmap::ImagePairToPairId(frame_id1, frame_id2);
        if (pair_covisibility_count.find(pair_id) ==
            pair_covisibility_count.end()) {
          pair_covisibility_count[pair_id] = 1;
        } else {
          pair_covisibility_count[pair_id]++;
        }
      }
    }
  }

  // Establish the visibility graph
  size_t counter = 0;
  ViewGraph visibility_graph_frame;
  std::vector<int> pair_count;
  for (auto& [pair_id, count] : pair_covisibility_count) {
    // since the relative pose is only fixed if there are more than 5 points,
    // then require each pair to have at least 5 points
    if (count >= 5) {
      counter++;
      const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);

      if (frame_observation_count[image_id1] < min_num_observations ||
          frame_observation_count[image_id2] < min_num_observations)
        continue;

      ImagePair image_pair;
      image_pair.is_valid = true;
      image_pair.weight = count;
      visibility_graph_frame.AddImagePair(image_id1, image_id2, std::move(image_pair));

      pair_count.push_back(count);
    }
  }
  LOG(INFO) << "Established visibility graph with " << counter << " pairs";

  // Create the visibility graph
  // Connect the reference image of each frame with other reference image
  std::unordered_map<frame_t, image_t> frame_id_to_begin_img;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    for (const auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (!reconstruction.ExistsImage(image_id)) continue;
      frame_id_to_begin_img[frame_id] = image_id;
      break;
    }
  }

  ViewGraph visibility_graph;
  for (auto& [pair_id, frame_image_pair] : visibility_graph_frame.image_pairs) {
    const auto [frame_id1, frame_id2] = colmap::PairIdToImagePair(pair_id);
    const image_t image_id1 = frame_id_to_begin_img[frame_id1];
    const image_t image_id2 = frame_id_to_begin_img[frame_id2];
    ImagePair image_pair;
    image_pair.weight = frame_image_pair.weight;
    visibility_graph.AddImagePair(image_id1, image_id2, std::move(image_pair));
  }

  int max_weight = std::max_element(pair_count.begin(), pair_count.end()) -
                   pair_count.begin();

  // within each frame, connect the reference image with all other images
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    image_t begin_image_id = frame_id_to_begin_img[frame_id];
    for (const auto& data_id : frame.ImageIds()) {
      image_t image_id = data_id.id;
      if (image_id == begin_image_id || !reconstruction.ExistsImage(image_id))
        continue;
      // Never break the inner edge
      ImagePair image_pair;
      image_pair.weight = max_weight;
      visibility_graph.AddImagePair(begin_image_id, image_id, std::move(image_pair));
    }
  }

  // sort the pair count
  std::sort(pair_count.begin(), pair_count.end());
  double median_count = pair_count[pair_count.size() / 2];

  // calculate the MAD (median absolute deviation)
  std::vector<int> pair_count_diff(pair_count.size());
  for (size_t i = 0; i < pair_count.size(); i++) {
    pair_count_diff[i] = std::abs(pair_count[i] - median_count);
  }
  std::sort(pair_count_diff.begin(), pair_count_diff.end());
  double median_count_diff = pair_count_diff[pair_count_diff.size() / 2];

  // The median
  LOG(INFO) << "Threshold for Strong Clustering: "
            << median_count - median_count_diff;

  ViewGraphManipulater::EstablishStrongClusters(
      visibility_graph,
      reconstruction,
      cluster_ids,
      ViewGraphManipulater::WEIGHT,
      std::max(median_count - median_count_diff, 20.),
      min_num_images);
}

}  // namespace glomap
