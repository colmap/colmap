#include "glomap/processors/image_undistorter.h"

#include "colmap/util/threading.h"

namespace glomap {

void UndistortImages(std::unordered_map<camera_t, Camera>& cameras,
                     std::unordered_map<image_t, Image>& images,
                     bool clean_points) {
  std::vector<image_t> image_ids;
  for (auto& [image_id, image] : images) {
    const int num_points = image.features.size();
    if (image.features_undist.size() == num_points && !clean_points)
      continue;  // already undistorted
    image_ids.push_back(image_id);
  }

  colmap::ThreadPool thread_pool(colmap::ThreadPool::kMaxNumThreads);

  LOG(INFO) << "Undistorting images..";
  const int num_images = image_ids.size();
  for (int image_idx = 0; image_idx < num_images; image_idx++) {
    Image& image = images[image_ids[image_idx]];
    const int num_points = image.features.size();
    if (image.features_undist.size() == num_points && !clean_points)
      continue;  // already undistorted

    const Camera& camera = cameras[image.camera_id];

    thread_pool.AddTask([&image, &camera, num_points]() {
      image.features_undist.clear();
      image.features_undist.reserve(num_points);
      for (int i = 0; i < num_points; i++) {
        image.features_undist.emplace_back(
            camera.CamFromImg(image.features[i])
                .value_or(Eigen::Vector2d::Zero())
                .homogeneous()
                .normalized());
      }
    });
  }

  thread_pool.Wait();
  LOG(INFO) << "Image undistortion done";
}

}  // namespace glomap
