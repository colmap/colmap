// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/apriltag.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/rigid3_matchers.h"
#include "colmap/geometry/sim3.h"
#include "colmap/image/apriltag.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/sensor/models.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <gtest/gtest.h>

namespace colmap {
namespace {

int InvokeAprilTagAligner(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  return RunAprilTagAligner(static_cast<int>(argv.size()), argv.data());
}

#ifdef COLMAP_APRILTAG_ENABLED

struct SyntheticTagBoard {
  int tag_id = -1;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d u_axis = Eigen::Vector3d::UnitX();
  Eigen::Vector3d v_axis = Eigen::Vector3d::UnitY();
  double full_size = 0.5;
  Bitmap texture;
};

void RenderTagBoardsIntoBitmap(const Camera& camera,
                               const Rigid3d& cam_from_world,
                               const std::vector<SyntheticTagBoard>& boards,
                               Bitmap* out_bitmap) {
  out_bitmap->Fill(BitmapColor<uint8_t>(210));
  const Rigid3d world_from_cam = Inverse(cam_from_world);

  for (const SyntheticTagBoard& board : boards) {
    const double half = 0.5 * board.full_size;
    const Eigen::Vector3d normal =
        board.u_axis.cross(board.v_axis).normalized();
    const std::array<Eigen::Vector3d, 4> corners_world = {
        board.center - half * board.u_axis - half * board.v_axis,
        board.center + half * board.u_axis - half * board.v_axis,
        board.center + half * board.u_axis + half * board.v_axis,
        board.center - half * board.u_axis + half * board.v_axis,
    };

    double min_x = static_cast<double>(out_bitmap->Width());
    double max_x = -1.0;
    double min_y = static_cast<double>(out_bitmap->Height());
    double max_y = -1.0;
    for (const Eigen::Vector3d& corner : corners_world) {
      const auto px = camera.ImgFromCam(cam_from_world * corner);
      if (!px.has_value()) {
        continue;
      }
      min_x = std::min(min_x, px->x());
      max_x = std::max(max_x, px->x());
      min_y = std::min(min_y, px->y());
      max_y = std::max(max_y, px->y());
    }

    const int x0 = std::max(0, static_cast<int>(std::floor(min_x)) - 2);
    const int x1 = std::min(out_bitmap->Width() - 1,
                            static_cast<int>(std::ceil(max_x)) + 2);
    const int y0 = std::max(0, static_cast<int>(std::floor(min_y)) - 2);
    const int y1 = std::min(out_bitmap->Height() - 1,
                            static_cast<int>(std::ceil(max_y)) + 2);

    const Eigen::Vector3d ray_origin = world_from_cam.translation();
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const auto cam_ray =
            camera.CamRayFromImg(Eigen::Vector2d(x + 0.5, y + 0.5));
        if (!cam_ray.has_value()) {
          continue;
        }
        const Eigen::Vector3d world_ray =
            world_from_cam.rotation() * (*cam_ray);
        const double denom = world_ray.dot(normal);
        if (std::abs(denom) < 1e-9) {
          continue;
        }
        const double t = (board.center - ray_origin).dot(normal) / denom;
        if (t <= 0.0) {
          continue;
        }
        const Eigen::Vector3d hit = ray_origin + t * world_ray;
        const Eigen::Vector3d rel = hit - board.center;
        const double u = rel.dot(board.u_axis) / board.full_size + 0.5;
        const double v = rel.dot(board.v_axis) / board.full_size + 0.5;
        if (u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0) {
          continue;
        }
        const int tx = std::clamp(
            static_cast<int>(std::floor(u * board.texture.Width())),
            0,
            board.texture.Width() - 1);
        const int ty = std::clamp(
            static_cast<int>(std::floor(v * board.texture.Height())),
            0,
            board.texture.Height() - 1);
        const auto pixel = board.texture.GetPixel(tx, ty);
        if (pixel.has_value()) {
          out_bitmap->SetPixel(x, y, *pixel);
        }
      }
    }
  }
}

struct SyntheticAprilTagScene {
  Reconstruction gt_reconstruction;
  std::vector<SyntheticTagBoard> boards;
};

SyntheticAprilTagScene BuildSyntheticScene(
    const std::filesystem::path& img_dir) {
  CreateDirIfNotExists(img_dir);
  SyntheticAprilTagScene scene;

  Camera camera;
  camera.camera_id = 1;
  camera.model_id = OpenCVCameraModel::model_id;
  camera.width = 640;
  camera.height = 480;
  camera.params = {550.0, 550.0, 320.0, 240.0, -0.08, 0.02, 0.0, 0.0};
  THROW_CHECK(camera.VerifyParams());
  scene.gt_reconstruction.AddCameraWithTrivialRig(camera);

  const std::vector<std::pair<int, Eigen::Vector3d>> tag_specs = {
      {0, {-0.45, -0.35, 2.5}},
      {1, {0.45, -0.35, 2.5}},
      {2, {-0.45, 0.35, 2.8}},
      {3, {0.45, 0.35, 2.2}},
  };
  for (const auto& [tag_id, center] : tag_specs) {
    SyntheticTagBoard board;
    board.tag_id = tag_id;
    board.center = center;
    board.u_axis = Eigen::Vector3d::UnitX();
    board.v_axis = Eigen::Vector3d::UnitY();
    board.full_size = 0.55;
    board.texture = SynthesizeAprilTagBitmap(AprilTagFamily::TAG_36H11,
                                             tag_id,
                                             /*quiet_zone_bits=*/2,
                                             /*pixel_scale=*/4);
    scene.boards.push_back(std::move(board));
  }

  const std::vector<Eigen::Vector3d> cam_positions = {
      {-0.9, -0.3, -0.1},
      {-0.4, 0.4, 0.0},
      {0.0, -0.3, 0.1},
      {0.5, 0.35, -0.05},
      {0.9, -0.25, 0.05},
  };
  const Eigen::Vector3d look_at(0.0, 0.0, 2.5);
  for (size_t i = 0; i < cam_positions.size(); ++i) {
    const image_t image_id = static_cast<image_t>(i + 1);
    const Eigen::Vector3d z_axis = (look_at - cam_positions[i]).normalized();
    const Eigen::Vector3d x_axis =
        Eigen::Vector3d::UnitY().cross(z_axis).normalized();
    const Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    Eigen::Matrix3d world_from_cam_rot;
    world_from_cam_rot << x_axis, y_axis, z_axis;
    const Rigid3d cam_from_world =
        Inverse(Rigid3d(Eigen::Quaterniond(world_from_cam_rot),
                        cam_positions[i]));

    Image image;
    image.SetImageId(image_id);
    image.SetCameraId(1);
    image.SetName("img_" + std::to_string(image_id) + ".png");
    scene.gt_reconstruction.AddImageWithTrivialFrame(image, cam_from_world);

    Bitmap bitmap(static_cast<int>(camera.width),
                  static_cast<int>(camera.height),
                  /*as_rgb=*/false);
    RenderTagBoardsIntoBitmap(camera, cam_from_world, scene.boards, &bitmap);
    THROW_CHECK(bitmap.Write(img_dir / image.Name()));
  }

  return scene;
}

TEST(AprilTagAligner, AlignWithKnownDistances) {
  const auto test_dir = CreateTestDir();
  const auto img_dir = test_dir / "images";
  const auto input_dir = test_dir / "input";
  const auto output_dir = test_dir / "output";
  const auto distances_path = test_dir / "distances.txt";
  const auto transform_path = test_dir / "transform.txt";
  const auto report_path = test_dir / "report.json";
  CreateDirIfNotExists(input_dir);

  const SyntheticAprilTagScene scene = BuildSyntheticScene(img_dir);

  // Scale the ground-truth model by 1 / 2.25 to simulate unscaled SfM output.
  constexpr double kTrueScale = 2.25;
  Reconstruction unscaled = scene.gt_reconstruction;
  unscaled.Transform(Sim3d(1.0 / kTrueScale,
                           Eigen::Quaterniond::Identity(),
                           Eigen::Vector3d::Zero()));
  unscaled.Write(input_dir);

  {
    std::ofstream file(distances_path);
    for (size_t i = 0; i < scene.boards.size(); ++i) {
      for (size_t j = i + 1; j < scene.boards.size(); ++j) {
        const double dist =
            (scene.boards[j].center - scene.boards[i].center).norm();
        file << scene.boards[i].tag_id << ' ' << scene.boards[j].tag_id << ' '
             << dist << '\n';
      }
    }
  }

  const int exit_code = InvokeAprilTagAligner({
      "apriltag_aligner",
      "--input_path",
      input_dir.string(),
      "--output_path",
      output_dir.string(),
      "--image_path",
      img_dir.string(),
      "--distances_path",
      distances_path.string(),
      "--transform_path",
      transform_path.string(),
      "--report_path",
      report_path.string(),
      "--AprilTag.quad_decimate",
      "1.0",
      "--AprilTag.num_threads",
      "2",
  });
  ASSERT_EQ(exit_code, EXIT_SUCCESS);

  Reconstruction aligned;
  aligned.Read(output_dir);
  for (const image_t image_id : scene.gt_reconstruction.RegImageIds()) {
    EXPECT_THAT(
        aligned.Image(image_id).CamFromWorld(),
        Rigid3dNear(scene.gt_reconstruction.Image(image_id).CamFromWorld(),
                    /*rtol=*/1e-3,
                    /*ttol=*/5e-3));
  }

  const Sim3d saved_tform = Sim3d::FromFile(transform_path);
  EXPECT_NEAR(saved_tform.scale(), kTrueScale, 5e-3);

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(report_path.string(), pt);
  EXPECT_TRUE(pt.get<bool>("success"));
  EXPECT_EQ(pt.get<std::string>("mode"), "distances");
  EXPECT_NEAR(pt.get<double>("transform.scale"), kTrueScale, 5e-3);
  EXPECT_EQ(pt.get<size_t>("metrics.num_inliers"), 6);
  // Verify resolution mismatch (640x480 image on disk vs 1280x960 camera).
  const auto scaled_input_dir = test_dir / "scaled_input";
  const auto scaled_output_dir = test_dir / "scaled_output";
  CreateDirIfNotExists(scaled_input_dir);
  Reconstruction rescaled_cam_recon = unscaled;
  rescaled_cam_recon.Camera(1).Rescale(1280, 960);
  rescaled_cam_recon.Write(scaled_input_dir);

  const int scaled_exit_code = InvokeAprilTagAligner({
      "apriltag_aligner",
      "--input_path",
      scaled_input_dir.string(),
      "--output_path",
      scaled_output_dir.string(),
      "--image_path",
      img_dir.string(),
      "--distances_path",
      distances_path.string(),
      "--AprilTag.quad_decimate",
      "1.0",
  });
  ASSERT_EQ(scaled_exit_code, EXIT_SUCCESS);

  Reconstruction scaled_aligned;
  scaled_aligned.Read(scaled_output_dir);
  for (const image_t image_id : scene.gt_reconstruction.RegImageIds()) {
    EXPECT_THAT(
        scaled_aligned.Image(image_id).CamFromWorld(),
        Rigid3dNear(scene.gt_reconstruction.Image(image_id).CamFromWorld(),
                    /*rtol=*/1e-3,
                    /*ttol=*/5e-3));
  }
}

TEST(AprilTagAligner, AlignWithKnownPositions) {
  const auto test_dir = CreateTestDir();
  const auto img_dir = test_dir / "images";
  const auto input_dir = test_dir / "input";
  const auto output_dir = test_dir / "output";
  const auto positions_path = test_dir / "positions.txt";
  const auto transform_path = test_dir / "transform.txt";
  const auto report_path = test_dir / "report.json";
  CreateDirIfNotExists(input_dir);

  const SyntheticAprilTagScene scene = BuildSyntheticScene(img_dir);

  // Move the reconstruction into an arbitrary unaligned frame.
  const Sim3d gt_world_from_local(
      1.6,
      Eigen::Quaterniond(Eigen::AngleAxisd(
          0.35, Eigen::Vector3d(0.2, 1.0, -0.3).normalized())),
      Eigen::Vector3d(1.5, -0.8, 0.7));
  Reconstruction local_recon = scene.gt_reconstruction;
  local_recon.Transform(Inverse(gt_world_from_local));
  local_recon.Write(input_dir);

  {
    std::ofstream file(positions_path);
    for (const SyntheticTagBoard& board : scene.boards) {
      file << board.tag_id << ' ' << board.center.x() << ' ' << board.center.y()
           << ' ' << board.center.z() << '\n';
    }
  }

  const int exit_code = InvokeAprilTagAligner({
      "apriltag_aligner",
      "--input_path",
      input_dir.string(),
      "--output_path",
      output_dir.string(),
      "--image_path",
      img_dir.string(),
      "--positions_path",
      positions_path.string(),
      "--transform_path",
      transform_path.string(),
      "--report_path",
      report_path.string(),
      "--AprilTag.family",
      "tag36h11",
      "--AprilTag.quad_decimate",
      "1.0",
      "--Alignment.max_position_error",
      "0.02",
  });
  ASSERT_EQ(exit_code, EXIT_SUCCESS);

  Reconstruction aligned;
  aligned.Read(output_dir);
  for (const image_t image_id : scene.gt_reconstruction.RegImageIds()) {
    EXPECT_THAT(
        aligned.Image(image_id).CamFromWorld(),
        Rigid3dNear(scene.gt_reconstruction.Image(image_id).CamFromWorld(),
                    /*rtol=*/5e-3,
                    /*ttol=*/2e-2));
  }

  const Sim3d saved_tform = Sim3d::FromFile(transform_path);
  EXPECT_NEAR(saved_tform.scale(), gt_world_from_local.scale(), 1e-2);
  EXPECT_LT(
      saved_tform.rotation().angularDistance(gt_world_from_local.rotation()),
      5e-3);
  EXPECT_LT(
      (saved_tform.translation() - gt_world_from_local.translation()).norm(),
      1e-2);

  boost::property_tree::ptree pt;
  boost::property_tree::read_json(report_path.string(), pt);
  EXPECT_TRUE(pt.get<bool>("success"));
  EXPECT_EQ(pt.get<std::string>("mode"), "positions");
  EXPECT_EQ(pt.get<size_t>("metrics.num_inliers"), 4);
}

TEST(AprilTagAligner, RejectsInvalidArguments) {
  const auto test_dir = CreateTestDir();
  const auto input_dir = test_dir / "input";
  const auto output_dir = test_dir / "output";
  const auto img_dir = test_dir / "images";
  const auto distances_path = test_dir / "distances.txt";
  const auto positions_path = test_dir / "positions.txt";
  CreateDirIfNotExists(input_dir);
  CreateDirIfNotExists(img_dir);
  {
    std::ofstream f1(distances_path);
    f1 << "0 1 1.0\n";
    std::ofstream f2(positions_path);
    f2 << "0 0 0 0\n";
  }

  // Neither distances nor positions provided.
  EXPECT_EQ(InvokeAprilTagAligner({
                "apriltag_aligner",
                "--input_path",
                input_dir.string(),
                "--output_path",
                output_dir.string(),
                "--image_path",
                img_dir.string(),
            }),
            EXIT_FAILURE);

  // Both distances and positions provided.
  EXPECT_EQ(InvokeAprilTagAligner({
                "apriltag_aligner",
                "--input_path",
                input_dir.string(),
                "--output_path",
                output_dir.string(),
                "--image_path",
                img_dir.string(),
                "--distances_path",
                distances_path.string(),
                "--positions_path",
                positions_path.string(),
            }),
            EXIT_FAILURE);
}

#else

TEST(AprilTagAligner, ReturnsFailureWhenDisabled) {
  const auto test_dir = CreateTestDir();
  const auto input_dir = test_dir / "input";
  const auto output_dir = test_dir / "output";
  const auto img_dir = test_dir / "images";
  const auto distances_path = test_dir / "distances.txt";
  CreateDirIfNotExists(input_dir);
  CreateDirIfNotExists(img_dir);
  {
    std::ofstream f(distances_path);
    f << "0 1 1.0\n";
  }
  EXPECT_EQ(InvokeAprilTagAligner({
                "apriltag_aligner",
                "--input_path",
                input_dir.string(),
                "--output_path",
                output_dir.string(),
                "--image_path",
                img_dir.string(),
                "--distances_path",
                distances_path.string(),
            }),
            EXIT_FAILURE);
}

#endif

}  // namespace
}  // namespace colmap
