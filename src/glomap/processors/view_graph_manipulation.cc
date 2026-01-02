#include "view_graph_manipulation.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/util/threading.h"

namespace glomap {

// Decompose relative poses from the two-view geometry matrices.
void ViewGraphManipulator::DecomposeRelativePoses(
    ViewGraph& view_graph,
    colmap::Reconstruction& reconstruction,
    int num_threads) {
  std::vector<image_pair_t> image_pair_ids;
  for (const auto& [pair_id, image_pair] : view_graph.ValidImagePairs()) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    if (!reconstruction.Image(image_id1).CameraPtr()->has_prior_focal_length ||
        !reconstruction.Image(image_id2).CameraPtr()->has_prior_focal_length)
      continue;
    image_pair_ids.push_back(pair_id);
  }

  const int64_t num_image_pairs = image_pair_ids.size();
  LOG(INFO) << "Decompose relative pose for " << num_image_pairs << " pairs";

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
      colmap::EstimateTwoViewGeometryPose(*image1.CameraPtr(),
                                          points1,
                                          *image2.CameraPtr(),
                                          points2,
                                          &image_pair);
      THROW_CHECK(image_pair.cam2_from_cam1.has_value());

      if (image_pair.cam2_from_cam1->translation.norm() > 1e-12) {
        image_pair.cam2_from_cam1->translation =
            image_pair.cam2_from_cam1->translation.normalized();
      }
    });
  }
  thread_pool.Wait();

  LOG(INFO) << "Decompose relative pose done. ";
}

}  // namespace glomap
