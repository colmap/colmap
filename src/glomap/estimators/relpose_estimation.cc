#include "glomap/estimators/relpose_estimation.h"

#include "colmap/util/threading.h"

#include <PoseLib/robust.h>

namespace glomap {
namespace {

inline poselib::Camera ColmapCameraToPoseLibCamera(
    const colmap::Camera& camera) {
  poselib::Camera pose_lib_camera(
      camera.ModelName(), camera.params, camera.width, camera.height);
  return pose_lib_camera;
}

}  // namespace

void EstimateRelativePoses(ViewGraph& view_graph,
                           colmap::Reconstruction& reconstruction,
                           const RelativePoseEstimationOptions& options) {
  std::vector<image_pair_t> valid_pair_ids;
  for (auto& [image_pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    valid_pair_ids.push_back(image_pair_id);
  }

  const int64_t num_image_pairs = valid_pair_ids.size();
  const int64_t kNumChunks = 10;
  const int64_t interval =
      std::ceil(static_cast<double>(num_image_pairs) / kNumChunks);

  colmap::ThreadPool thread_pool(colmap::ThreadPool::kMaxNumThreads);

  LOG(INFO) << "Estimating relative pose for " << num_image_pairs << " pairs";
  for (int64_t chunk_id = 0; chunk_id < kNumChunks; chunk_id++) {
    std::cout << "\r Estimating relative pose: " << chunk_id * kNumChunks << "%"
              << std::flush;
    const int64_t start = chunk_id * interval;
    const int64_t end =
        std::min<int64_t>((chunk_id + 1) * interval, num_image_pairs);

    for (int64_t pair_idx = start; pair_idx < end; pair_idx++) {
      thread_pool.AddTask([&, pair_idx]() {
        // Define as thread-local to reuse memory allocation in different tasks.
        thread_local std::vector<Eigen::Vector2d> points2D_1;
        thread_local std::vector<Eigen::Vector2d> points2D_2;
        thread_local std::vector<char> inliers;

        ImagePair& image_pair =
            view_graph.image_pairs[valid_pair_ids[pair_idx]];
        const Image& image1 = reconstruction.Image(image_pair.image_id1);
        const Image& image2 = reconstruction.Image(image_pair.image_id2);
        const Eigen::MatrixXi& matches = image_pair.matches;

        const colmap::Camera& camera1 =
            reconstruction.Camera(image1.CameraId());
        const colmap::Camera& camera2 =
            reconstruction.Camera(image2.CameraId());
        poselib::Camera camera_poselib1 = ColmapCameraToPoseLibCamera(camera1);
        poselib::Camera camera_poselib2 = ColmapCameraToPoseLibCamera(camera2);
        bool valid_camera_model =
            (camera_poselib1.model_id >= 0 && camera_poselib2.model_id >= 0);

        // Collect the original 2D points
        points2D_1.clear();
        points2D_2.clear();
        for (size_t idx = 0; idx < matches.rows(); idx++) {
          points2D_1.push_back(image1.Point2D(matches(idx, 0)).xy);
          points2D_2.push_back(image2.Point2D(matches(idx, 1)).xy);
        }
        // If the camera model is not supported by poselib
        if (!valid_camera_model) {
          // Undistort points
          // Note that here, we still rescale by the focal length (to avoid
          // change the RANSAC threshold)
          Eigen::Matrix2d K1_new = Eigen::Matrix2d::Zero();
          Eigen::Matrix2d K2_new = Eigen::Matrix2d::Zero();
          K1_new(0, 0) = camera1.FocalLengthX();
          K1_new(1, 1) = camera1.FocalLengthY();
          K2_new(0, 0) = camera2.FocalLengthX();
          K2_new(1, 1) = camera2.FocalLengthY();
          for (size_t idx = 0; idx < matches.rows(); idx++) {
            points2D_1[idx] = K1_new * camera1.CamFromImg(points2D_1[idx])
                                           .value_or(Eigen::Vector2d::Zero());
            points2D_2[idx] = K2_new * camera2.CamFromImg(points2D_2[idx])
                                           .value_or(Eigen::Vector2d::Zero());
          }

          // Reset the camera to be the pinhole camera with original focal
          // length and zero principal point
          camera_poselib1 = poselib::Camera(
              "PINHOLE",
              {camera1.FocalLengthX(), camera1.FocalLengthY(), 0., 0.},
              camera1.width,
              camera1.height);
          camera_poselib2 = poselib::Camera(
              "PINHOLE",
              {camera2.FocalLengthX(), camera2.FocalLengthY(), 0., 0.},
              camera2.width,
              camera2.height);
        }
        inliers.clear();
        poselib::CameraPose pose_rel_calc;
        try {
          poselib::estimate_relative_pose(points2D_1,
                                          points2D_2,
                                          camera_poselib1,
                                          camera_poselib2,
                                          options.ransac_options,
                                          options.bundle_options,
                                          &pose_rel_calc,
                                          &inliers);
        } catch (const std::exception& e) {
          LOG(ERROR) << "Error in relative pose estimation: " << e.what();
          image_pair.is_valid = false;
          return;
        }

        // Convert the relative pose to the glomap format
        for (int i = 0; i < 4; i++) {
          image_pair.cam2_from_cam1.rotation.coeffs()[i] =
              pose_rel_calc.q[(i + 1) % 4];
        }
        image_pair.cam2_from_cam1.translation = pose_rel_calc.t;
      });
    }

    thread_pool.Wait();
  }

  std::cout << "\r Estimating relative pose: 100%" << '\n';
  LOG(INFO) << "Estimating relative pose done";
}

}  // namespace glomap
