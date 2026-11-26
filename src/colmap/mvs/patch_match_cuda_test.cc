// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/patch_match_cuda.h"

#include "colmap/mvs/patch_match_cuda_test.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace internal {
float ProbePropagatedDepth(const std::array<float, 4>& K,
                           float depth,
                           const std::array<float, 3>& normal,
                           float row1,
                           float row2,
                           float col);
}  // namespace internal
namespace {
constexpr float kDepthMin = 1.0f;
constexpr float kDepthMax = 8.0f;
constexpr float kPlaneZ = 4.0f;
struct Inputs {
  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
};

// Two independent reference problems, each with differently calibrated source
// views. All images are rendered from the same textured world plane. Sizes are
// deliberately non-square, odd, and not CUDA block multiples.
Inputs MakeInputs(const int variant,
                  const int num_source_images = 2,
                  const int ref_height = 0,
                  const int ref_width = 0) {
  Inputs inputs;
  for (int i = 0; i <= num_source_images; ++i) {
    const int width =
        (ref_width > 0 ? ref_width : (variant == 0 ? 47 : 39)) + 2 * i;
    const int height =
        (ref_height > 0 ? ref_height : (variant == 0 ? 35 : 53)) + 2 * i;
    const float fx = (variant == 0 ? 48.0f : 69.0f) + 3.0f * i;
    const float fy = (variant == 0 ? 57.0f : 44.0f) + 2.0f * i;
    const float cx = width * 0.5f + 0.7f + i * 0.2f;
    const float cy = height * 0.5f - 1.1f - i * 0.15f;
    const float yaw = (i - 1) * 0.035f;
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    const float camera_x = (i - 1) * 0.25f;
    const float camera_y = (i - 1) * 0.04f;
    const float K[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    const float R[9] = {c, 0, s, 0, 1, 0, -s, 0, c};
    const float T[3] = {-c * camera_x, -camera_y, s * camera_x};

    Image image(
        "synthetic-" + std::to_string(variant) + "-" + std::to_string(i),
        width,
        height,
        K,
        R,
        T);
    Bitmap bitmap(width, height, false);
    DepthMap depth(width, height, kDepthMin, kDepthMax);
    NormalMap normal(width, height);
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        // COLMAP's undistorted calibration uses pixel centers at (col+.5,
        // row+.5). Invert R to intersect the camera ray with world Z=4.
        const float ray_x = (col + 0.5f - cx) / fx;
        const float ray_y = (row + 0.5f - cy) / fy;
        const float world_ray_x = c * ray_x - s;
        const float world_ray_z = s * ray_x + c;
        const float z = kPlaneZ / world_ray_z;
        const float x = camera_x + z * world_ray_x;
        const float y = camera_y + z * ray_y;
        const float value = 128.0f + 42.0f * std::sin(8.0f * x + 3.0f * y) +
                            35.0f * std::cos(5.0f * x - 11.0f * y) +
                            24.0f * std::sin(17.0f * x + 13.0f * y);
        bitmap.SetPixel(col,
                        row,
                        BitmapColor<uint8_t>(static_cast<uint8_t>(
                            std::clamp(std::round(value), 0.0f, 255.0f))));
        depth.Set(row, col, z);
        normal.Set(row, col, 0, -s);
        normal.Set(row, col, 1, 0.0f);
        normal.Set(row, col, 2, -c);
      }
    }
    image.SetBitmap(std::move(bitmap));
    inputs.images.push_back(std::move(image));
    inputs.depths.push_back(std::move(depth));
    inputs.normals.push_back(std::move(normal));
  }
  return inputs;
}

PatchMatchOptions MakeOptions(const bool geometry,
                              const bool filter,
                              const bool /*unused*/) {
  PatchMatchOptions options;
  options.gpu_index = "0";
  options.depth_min = kDepthMin;
  options.depth_max = kDepthMax;
  options.window_radius = 2;
  options.window_step = 1;
  options.num_samples = 3;
  // Two iterations cross both alternating orientations repeatedly.
  options.num_iterations = 2;
  options.geom_consistency = geometry;
  options.filter = filter;
  // Keep full-storage controls explicit when the public CUDA default changes.
  options.filter_min_ncc = -1.0;
  options.filter_min_triangulation_angle = 0.0;
  options.filter_min_num_consistent = 1;
  options.filter_geom_consistency_max_cost = 10.0;
  return options;
}

class CudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int num_devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&num_devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && num_devices == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  }
};

TEST_F(CudaTest, PropagationPreservesCompletePlaneAcrossSweepRotations) {
  std::array<float, 4> K{800, 640, 750, 480};
  std::array<float, 3> normal{0.6f, 0.4f, -std::sqrt(0.48f)};
  int width = 1280;
  int height = 960;
  for (int rotation = 0; rotation < 4; ++rotation) {
    SCOPED_TRACE(rotation);
    const Eigen::Vector3d n(normal[0], normal[1], normal[2]);
    for (const float col : {0.1f * width, K[1], 0.86f * width}) {
      const float row1 = 0.68f * height;
      const float row2 = row1 + 1;
      const Eigen::Vector3d ray1(
          (double(col) - K[1]) / K[0], (double(row1) - K[3]) / K[2], 1);
      const Eigen::Vector3d ray2(
          (double(col) - K[1]) / K[0], (double(row2) - K[3]) / K[2], 1);
      ASSERT_LT(n.dot(ray1), 0);
      ASSERT_LT(n.dot(ray2), 0);
      for (const float depth : {0.0001f, 10.0f, 1000000.0f}) {
        SCOPED_TRACE(depth);
        const Eigen::Vector3d point1 = double(depth) * ray1;
        const double expected = n.dot(point1) / n.dot(ray2);
        const float actual =
            internal::ProbePropagatedDepth(K, depth, normal, row1, row2, col);
        ASSERT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, expected, 2e-6 * std::abs(expected));
        // A propagated candidate must still lie on the original 3D plane.
        EXPECT_NEAR(n.dot(double(actual) * ray2),
                    n.dot(point1),
                    2e-6 * std::abs(n.dot(point1)));
      }
    }
    const float previous_cx = K[1];
    K = {K[2], K[3], K[0], width - 1.0f - previous_cx};
    normal = {normal[1], -normal[0], normal[2]};
    std::swap(width, height);
  }
}

TEST_F(CudaTest, PropagationRetainsDepthForNearParallelRay) {
  const std::array<float, 4> K{1, 0, 1, 0};
  const std::array<float, 3> normal{0, -1, 0};
  for (const float row2 : {0.0f, 0.000005f}) {
    EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, 3, normal, 1, row2, 0),
                    3);
  }
  EXPECT_NEAR(internal::ProbePropagatedDepth(K, 3, normal, 1, 0.00002f, 0),
              150000.0f,
              1.0f);
}

TEST_F(CudaTest, PropagationRetainsDepthAcrossPlaneHorizon) {
  const std::array<float, 4> K{1000, 0, 1000, 0};
  Eigen::Vector3f n(0.7f, 0.7f, -0.70035f);
  n.normalize();
  const std::array<float, 3> normal{n.x(), n.y(), n.z()};
  ASSERT_LT(n.cast<double>().dot(Eigen::Vector3d(1, 0, 1)), 0);
  ASSERT_GT(n.cast<double>().dot(Eigen::Vector3d(1, 0.001, 1)), 0);
  EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, 10, normal, 0, 1, 1000),
                  10);
}

TEST_F(CudaTest, PropagationAvoidsUnrepresentableDepth) {
  const std::array<float, 4> K{1, 0, 1, 0};
  const float component = std::sqrt(0.5f);
  const std::array<float, 3> normal{0, -component, -component};
  const float depth = std::numeric_limits<float>::max();
  EXPECT_FLOAT_EQ(internal::ProbePropagatedDepth(K, depth, normal, 1, 0, 0),
                  depth);
  // Dividing the plane terms before multiplying avoids an intermediate
  // overflow when the resulting depth is still representable.
  const double expected = double(depth) * 2 / 3;
  EXPECT_NEAR(internal::ProbePropagatedDepth(K, depth, normal, 1, 2, 0),
              expected,
              2e-6 * expected);
}

// Use original pixel-center calibration, independently of the device helper.
std::array<double, 3> ReferenceRay(const Image& image,
                                   const int row,
                                   const int col) {
  const float* K = image.GetK();
  return {(col + 0.5 - K[2]) / K[0], (row + 0.5 - K[5]) / K[4], 1};
}

void ExpectPropagationAnchors(const internal::SweepTraceResult& trace,
                              const Image& image) {
  const int width = static_cast<int>(trace.hypotheses.GetWidth());
  const int height = static_cast<int>(trace.hypotheses.GetHeight());
  ASSERT_GT(width, 0);
  ASSERT_GT(height, 1);
  for (int row = 0; row < height; ++row) {
    int changed_depths = 0;
    for (int col = 0; col < width; ++col) {
      SCOPED_TRACE(::testing::Message() << "row=" << row << " col=" << col);
      // Zero NCC emissions keep the deliberately oblique incumbent selected.
      ASSERT_EQ(trace.support.Get(row, col), 0);
      ASSERT_EQ(trace.winners.Get(row, col), 0);
      const int source_row = row == 0 ? 0 : row - 1;
      const float source_depth = trace.hypotheses.Get(source_row, col, 0);
      const float propagated_depth = trace.hypotheses.Get(row, col, 4);
      for (int axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(trace.hypotheses.Get(row, col, 5 + axis),
                  trace.hypotheses.Get(source_row, col, 1 + axis));
      }
      if (row == 0) {
        EXPECT_EQ(propagated_depth, source_depth);
        continue;
      }
      const auto source_ray = ReferenceRay(image, row - 1, col);
      const auto target_ray = ReferenceRay(image, row, col);
      double source_dot = 0;
      double target_dot = 0;
      for (int axis = 0; axis < 3; ++axis) {
        const double normal = trace.hypotheses.Get(source_row, col, 1 + axis);
        source_dot += normal * source_ray[axis];
        target_dot += normal * target_ray[axis];
      }
      // This fixture stays far from the production fallback and plane horizon.
      ASSERT_GT(std::abs(target_dot), 0.1);
      const double expected = source_depth * source_dot / target_dot;
      ASSERT_GT(expected, 0);
      const double tolerance = 2e-5 * expected;
      EXPECT_NEAR(propagated_depth, expected, tolerance);
      changed_depths += std::abs(expected - source_depth) > 10 * tolerance;
    }
    // Every nonzero row, including each band's first row, needs a real
    // transfer.
    if (row > 0) {
      EXPECT_GT(changed_depths, 0) << "row=" << row;
    }
  }
}

TEST_F(CudaTest, PropagationUsesTheActualSourcePixel) {
  constexpr int kWidth = 35;
  constexpr int kHeight = 259;
  auto inputs = MakeInputs(0, 1, kHeight, kWidth);
  for (Image& image : inputs.images) {
    Bitmap bitmap(image.GetWidth(), image.GetHeight(), false);
    bitmap.Fill(BitmapColor<uint8_t>(0));
    image.SetBitmap(std::move(bitmap));
  }
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      inputs.depths[0].Set(row, col, 3.2f + 0.001f * row);
      inputs.normals[0].Set(row, col, 0, 0.2f);
      inputs.normals[0].Set(row, col, 1, 0.3f);
      inputs.normals[0].Set(row, col, 2, -std::sqrt(0.87f));
    }
  }
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1};
  problem.images = &inputs.images;
  problem.depth_maps = &inputs.depths;
  problem.normal_maps = &inputs.normals;
  auto options = MakeOptions(true, false, false);
  options.window_radius = 5;
  options.sigma_spatial = 5.0;
  options.ncc_sigma = 0.01;
  options.num_samples = 1;
  options.num_iterations = 1;
  ASSERT_TRUE(options.Check());

  // Constant images give cost=2. At this sigma the emission is exactly zero,
  // so the real sweep retains the seed while still tracing every hypothesis.
  // This makes every propagation boundary informative without relying on RNG
  // winners or claiming that the incorrect candidate must win its comparison.
  PatchMatchCuda patch_match(options, problem);
  const auto trace = internal::ProbeSweep(patch_match, 1873);
  ExpectPropagationAnchors(trace, inputs.images[0]);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
