#include "view_graph_manipulation.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/util/threading.h"

#include <mutex>

namespace glomap {

// Decompose relative poses from the two-view geometry matrices.
void ViewGraphManipulator::DecomposeRelativePoses(
    ViewGraph& view_graph,
    colmap::Reconstruction& reconstruction,
    int num_threads) {
  std::vector<image_pair_t> image_pair_ids;
  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    image_pair_ids.push_back(pair_id);
  }

  const int64_t num_image_pairs = image_pair_ids.size();
  LOG(INFO) << "Decompose relative pose for " << num_image_pairs << " pairs";

  std::mutex invalid_pairs_mutex;
  std::vector<image_pair_t> invalid_pair_ids;

  colmap::ThreadPool thread_pool(num_threads);
  for (int64_t idx = 0; idx < num_image_pairs; idx++) {
    thread_pool.AddTask([&, idx]() {
      const image_pair_t pair_id = image_pair_ids[idx];
      const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
      ImagePair& image_pair = view_graph.ImagePair(image_id1, image_id2).first;
      const Image& image1 = reconstruction.Image(image_id1);
      const Image& image2 = reconstruction.Image(image_id2);

      std::vector<Eigen::Vector2d> points1(image1.NumPoints2D());
      for (colmap::point2D_t point2D_idx = 0;
           point2D_idx < image1.NumPoints2D();
           point2D_idx++) {
        points1[point2D_idx] = image1.Point2D(point2D_idx).xy;
      }
      std::vector<Eigen::Vector2d> points2(image2.NumPoints2D());
      for (colmap::point2D_t point2D_idx = 0;
           point2D_idx < image2.NumPoints2D();
           point2D_idx++) {
        points2[point2D_idx] = image2.Point2D(point2D_idx).xy;
      }

      // ImagePair inherits from TwoViewGeometry, so pass it directly.
      const bool success =
          colmap::EstimateTwoViewGeometryPose(*image1.CameraPtr(),
                                              points1,
                                              *image2.CameraPtr(),
                                              points2,
                                              &image_pair);

      if (!success || !image_pair.cam2_from_cam1.has_value()) {
        std::lock_guard<std::mutex> lock(invalid_pairs_mutex);
        invalid_pair_ids.push_back(pair_id);
        return;
      }

      const double norm = image_pair.cam2_from_cam1->translation.norm();
      if (norm > 1e-12) {
        image_pair.cam2_from_cam1->translation /= norm;
      }
    });
  }
  thread_pool.Wait();

  for (const image_pair_t pair_id : invalid_pair_ids) {
    view_graph.SetInvalidImagePair(pair_id);
  }

  LOG(INFO) << "Decompose relative pose done. " << invalid_pair_ids.size()
            << " pairs failed.";
}

}  // namespace glomap
