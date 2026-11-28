// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_metal.h"

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

constexpr int kWidth = 200;
constexpr int kHeight = 150;
constexpr float kFocalLength = 400.0f;

// A textured plane that is seen by a reference and two source cameras. All
// quantities are in the world frame, the reference camera is at the origin.
class SyntheticPlane {
 public:
  SyntheticPlane() {
    // Plane through (0, 0, 5), tilted by 20 degrees around the x-axis, with
    // the normal pointing towards the cameras.
    const float tilt = 20.0f * M_PI / 180.0f;
    normal_ = Eigen::Vector3d(0, std::sin(tilt), -std::cos(tilt));
    dist_ = normal_.dot(Eigen::Vector3d(0, 0, 5));
    axis_u_ = Eigen::Vector3d(1, 0, 0);
    axis_v_ = normal_.cross(axis_u_);

    // A smooth texture without repetitions: random plane waves with
    // wavelengths of 4 to 10 pixels at the depth of the plane.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int i = 0; i < 12; ++i) {
      const double wavelength = 0.05 + 0.08 * uniform(rng);
      const double angle = 2.0 * M_PI * uniform(rng);
      waves_.push_back({2.0 * M_PI / wavelength * std::cos(angle),
                        2.0 * M_PI / wavelength * std::sin(angle),
                        2.0 * M_PI * uniform(rng)});
    }

    // Reference camera and two source cameras with different baselines,
    // both looking at the plane.
    cams_.push_back(MakeCamera(Eigen::Vector3d(0, 0, 0), 0.0));
    cams_.push_back(MakeCamera(Eigen::Vector3d(0.4, 0, 0), -4.0));
    cams_.push_back(MakeCamera(Eigen::Vector3d(-0.3, 0.2, 0), 3.0));
  }

  // Ground truth depth of the pixel in the given camera.
  float Depth(const int cam_idx, const int col, const int row) const {
    const Eigen::Vector3d ray_in_cam((col + 0.5 - kWidth / 2.0) / kFocalLength,
                                     (row + 0.5 - kHeight / 2.0) / kFocalLength,
                                     1.0);
    const Eigen::Matrix3d& R = cams_[cam_idx].R;
    const Eigen::Vector3d center = cams_[cam_idx].center;
    const Eigen::Vector3d ray = R.transpose() * ray_in_cam;
    const double t = (dist_ - normal_.dot(center)) / normal_.dot(ray);
    if (!(t > 0)) {
      return 0.0f;
    }
    return static_cast<float>(t);  // The depth is t since ray_in_cam.z = 1.
  }

  // Normal of the plane in the frame of the given camera.
  Eigen::Vector3d Normal(const int cam_idx) const {
    return cams_[cam_idx].R * normal_;
  }

  // Returns the projection of the 3D point of the reference pixel into the
  // given camera.
  Eigen::Vector2d Project(const int cam_idx,
                          const int col,
                          const int row) const {
    const double depth = Depth(0, col, row);
    const Eigen::Vector3d point(
        depth * (col + 0.5 - kWidth / 2.0) / kFocalLength,
        depth * (row + 0.5 - kHeight / 2.0) / kFocalLength,
        depth);
    const Eigen::Vector3d point_in_cam =
        cams_[cam_idx].R * point + cams_[cam_idx].T;
    return Eigen::Vector2d(
        kFocalLength * point_in_cam.x() / point_in_cam.z() + kWidth / 2.0 - 0.5,
        kFocalLength * point_in_cam.y() / point_in_cam.z() + kHeight / 2.0 -
            0.5);
  }

  Image CreateImage(const int cam_idx) const {
    const float K[9] = {kFocalLength,
                        0,
                        kWidth / 2.0f,
                        0,
                        kFocalLength,
                        kHeight / 2.0f,
                        0,
                        0,
                        1};
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R =
        cams_[cam_idx].R.cast<float>();
    const Eigen::Vector3f T = cams_[cam_idx].T.cast<float>();
    Image image("", kWidth, kHeight, K, R.data(), T.data());

    Bitmap bitmap(kWidth, kHeight, /*as_rgb=*/false);
    for (int row = 0; row < kHeight; ++row) {
      for (int col = 0; col < kWidth; ++col) {
        const float depth = Depth(cam_idx, col, row);
        const Eigen::Vector3d ray_in_cam(
            (col + 0.5 - kWidth / 2.0) / kFocalLength,
            (row + 0.5 - kHeight / 2.0) / kFocalLength,
            1.0);
        const Eigen::Vector3d point =
            cams_[cam_idx].center +
            cams_[cam_idx].R.transpose() * ray_in_cam * depth;
        bitmap.SetPixel(
            col,
            row,
            BitmapColor<uint8_t>(Texture(point.x(), point.dot(axis_v_))));
      }
    }
    image.SetBitmap(std::move(bitmap));
    return image;
  }

  DepthMap CreateDepthMap(const int cam_idx) const {
    DepthMap depth_map(kWidth, kHeight, 3.0f, 8.0f);
    for (int row = 0; row < kHeight; ++row) {
      for (int col = 0; col < kWidth; ++col) {
        depth_map.Set(row, col, Depth(cam_idx, col, row));
      }
    }
    return depth_map;
  }

  int NumCameras() const { return cams_.size(); }

 private:
  struct Camera {
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    Eigen::Vector3d center;
  };

  // Camera at the given center, rotated around the y-axis (in degrees).
  static Camera MakeCamera(const Eigen::Vector3d& center,
                           const double yaw_deg) {
    Camera cam;
    cam.R = Eigen::AngleAxisd(yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitY())
                .toRotationMatrix();
    cam.T = -cam.R * center;
    cam.center = center;
    return cam;
  }

  uint8_t Texture(const double u, const double v) const {
    double value = 0;
    for (const auto& wave : waves_) {
      value += std::sin(wave[0] * u + wave[1] * v + wave[2]);
    }
    value = 0.5 + 0.12 * value;
    return static_cast<uint8_t>(255.0 * std::min(1.0, std::max(0.0, value)));
  }

  Eigen::Vector3d normal_;
  Eigen::Vector3d axis_u_;
  Eigen::Vector3d axis_v_;
  double dist_ = 0;
  std::vector<std::array<double, 3>> waves_;
  std::vector<Camera> cams_;
};

struct TestProblem {
  explicit TestProblem(const SyntheticPlane& plane) {
    for (int i = 0; i < plane.NumCameras(); ++i) {
      images.push_back(plane.CreateImage(i));
    }
    problem.ref_image_idx = 0;
    problem.src_image_idxs = {1, 2};
    problem.images = &images;
  }

  std::vector<Image> images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  PatchMatch::Problem problem;
};

PatchMatchOptions CreateOptions() {
  PatchMatchOptions options;
  options.gpu_index = "0";
  options.depth_min = 3.0f;
  options.depth_max = 8.0f;
  options.sigma_spatial = options.window_radius;
  options.geom_consistency = false;
  options.filter = false;
  return options;
}

// Whether the pixel of the reference image is visible in all source images.
bool IsVisible(const SyntheticPlane& plane, const int col, const int row) {
  const int kBorder = 12;
  for (int cam_idx = 1; cam_idx < plane.NumCameras(); ++cam_idx) {
    const Eigen::Vector2d xy = plane.Project(cam_idx, col, row);
    if (xy.x() < kBorder || xy.y() < kBorder || xy.x() >= kWidth - kBorder ||
        xy.y() >= kHeight - kBorder) {
      return false;
    }
  }
  return col >= kBorder && row >= kBorder && col < kWidth - kBorder &&
         row < kHeight - kBorder;
}

// Fraction of the visible pixels with a depth within the relative tolerance of
// the ground truth.
double ComputeDepthInlierRatio(const SyntheticPlane& plane,
                               const DepthMap& depth_map,
                               const double rel_tolerance,
                               size_t* num_visible = nullptr) {
  size_t num_inliers = 0;
  size_t num_total = 0;
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      if (!IsVisible(plane, col, row)) {
        continue;
      }
      ++num_total;
      const float gt_depth = plane.Depth(0, col, row);
      if (std::abs(depth_map.Get(row, col) - gt_depth) <
          rel_tolerance * gt_depth) {
        ++num_inliers;
      }
    }
  }
  if (num_visible != nullptr) {
    *num_visible = num_total;
  }
  return num_total == 0 ? 0.0 : static_cast<double>(num_inliers) / num_total;
}

// Median angle between the estimated and the ground truth normal.
double ComputeMedianNormalError(const SyntheticPlane& plane,
                                const NormalMap& normal_map) {
  const Eigen::Vector3d gt_normal = plane.Normal(0);
  std::vector<double> errors;
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      if (!IsVisible(plane, col, row)) {
        continue;
      }
      const Eigen::Vector3d normal(normal_map.Get(row, col, 0),
                                   normal_map.Get(row, col, 1),
                                   normal_map.Get(row, col, 2));
      errors.push_back(
          std::acos(std::min(1.0, std::max(-1.0, normal.dot(gt_normal)))) *
          180.0 / M_PI);
    }
  }
  std::sort(errors.begin(), errors.end());
  return errors[errors.size() / 2];
}

bool HasMetalDevice() { return GetNumMetalDevices() > 0; }

TEST(PatchMatchMetal, PhotometricPlane) {
  if (!HasMetalDevice()) {
    GTEST_SKIP() << "No Metal device available";
  }

  SyntheticPlane plane;
  TestProblem test_problem(plane);

  PatchMatch patch_match(CreateOptions(), test_problem.problem);
  patch_match.Run();

  const DepthMap depth_map = patch_match.GetDepthMap();
  ASSERT_EQ(depth_map.GetWidth(), kWidth);
  ASSERT_EQ(depth_map.GetHeight(), kHeight);
  size_t num_visible = 0;
  EXPECT_GT(ComputeDepthInlierRatio(plane, depth_map, 0.02, &num_visible),
            0.95);
  EXPECT_GT(num_visible, 5000);

  const NormalMap normal_map = patch_match.GetNormalMap();
  EXPECT_LT(ComputeMedianNormalError(plane, normal_map), 5.0);

  const Mat<float> sel_prob_map = patch_match.GetSelProbMap();
  EXPECT_EQ(sel_prob_map.GetWidth(), kWidth);
  EXPECT_EQ(sel_prob_map.GetHeight(), kHeight);
  EXPECT_EQ(sel_prob_map.GetDepth(), 2);

  // Without filtering, no consistency graph is generated.
  int num_images = 0;
  const int* image_idxs = nullptr;
  patch_match.GetConsistencyGraph().GetImageIdxs(
      kHeight / 2, kWidth / 2, &num_images, &image_idxs);
  EXPECT_EQ(num_images, 0);
}

TEST(PatchMatchMetal, WindowStep) {
  if (!HasMetalDevice()) {
    GTEST_SKIP() << "No Metal device available";
  }

  SyntheticPlane plane;
  TestProblem test_problem(plane);

  PatchMatchOptions options = CreateOptions();
  options.window_radius = 7;
  options.window_step = 2;
  options.sigma_spatial = options.window_radius;
  PatchMatch patch_match(options, test_problem.problem);
  patch_match.Run();

  EXPECT_GT(ComputeDepthInlierRatio(plane, patch_match.GetDepthMap(), 0.02),
            0.95);
}

TEST(PatchMatchMetal, GeometricConsistencyAndFilter) {
  if (!HasMetalDevice()) {
    GTEST_SKIP() << "No Metal device available";
  }

  SyntheticPlane plane;
  TestProblem test_problem(plane);

  // Photometric pass without filtering.
  PatchMatch photometric(CreateOptions(), test_problem.problem);
  photometric.Run();

  // Geometric pass with filtering, given the depth maps of the source images
  // and the result of the photometric pass for the reference image.
  test_problem.depth_maps.resize(plane.NumCameras());
  test_problem.normal_maps.resize(plane.NumCameras());
  for (int i = 1; i < plane.NumCameras(); ++i) {
    test_problem.depth_maps[i] = plane.CreateDepthMap(i);
  }
  test_problem.depth_maps[0] = photometric.GetDepthMap();
  test_problem.normal_maps[0] = photometric.GetNormalMap();
  test_problem.problem.depth_maps = &test_problem.depth_maps;
  test_problem.problem.normal_maps = &test_problem.normal_maps;

  PatchMatchOptions options = CreateOptions();
  options.geom_consistency = true;
  options.filter = true;
  PatchMatch geometric(options, test_problem.problem);
  geometric.Run();

  // Filtered pixels have a zero depth, all others must be correct.
  const DepthMap depth_map = geometric.GetDepthMap();
  size_t num_valid = 0;
  size_t num_correct = 0;
  size_t num_visible = 0;
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      if (!IsVisible(plane, col, row)) {
        continue;
      }
      ++num_visible;
      if (depth_map.Get(row, col) > 0) {
        ++num_valid;
        const float gt_depth = plane.Depth(0, col, row);
        num_correct +=
            std::abs(depth_map.Get(row, col) - gt_depth) < 0.02 * gt_depth;
      }
    }
  }
  EXPECT_GT(num_valid, 0.9 * num_visible);
  EXPECT_GT(num_correct, 0.98 * num_valid);

  // Pixels of the consistency graph are valid pixels and list both sources.
  const ConsistencyGraph graph = geometric.GetConsistencyGraph();
  size_t num_in_graph = 0;
  size_t num_valid_total = 0;
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      int num_images = 0;
      const int* image_idxs = nullptr;
      graph.GetImageIdxs(row, col, &num_images, &image_idxs);
      num_valid_total += depth_map.Get(row, col) > 0;
      if (num_images > 0) {
        ++num_in_graph;
        EXPECT_GT(depth_map.Get(row, col), 0);
        for (int i = 0; i < num_images; ++i) {
          EXPECT_TRUE(image_idxs[i] == 1 || image_idxs[i] == 2);
        }
      } else {
        EXPECT_EQ(depth_map.Get(row, col), 0);
      }
    }
  }
  EXPECT_EQ(num_in_graph, num_valid_total);
}

TEST(PatchMatchMetal, UnsupportedWindowRadius) {
  if (!HasMetalDevice()) {
    GTEST_SKIP() << "No Metal device available";
  }

  SyntheticPlane plane;
  TestProblem test_problem(plane);

  PatchMatchOptions options = CreateOptions();
  options.window_radius = 21;
  options.sigma_spatial = options.window_radius;
  PatchMatch patch_match(options, test_problem.problem);
  EXPECT_ANY_THROW(patch_match.Run());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
