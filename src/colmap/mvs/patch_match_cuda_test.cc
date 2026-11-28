// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/patch_match_cuda.h"

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_cuda_test.h"
#include "colmap/util/cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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
std::vector<uint32_t> ProbeSelectionProbability(const std::vector<uint32_t>&);
std::pair<std::vector<int>, std::vector<float>> ProbeSampledSources(
    const std::vector<int>&, const std::vector<float>&, int, bool);
int ProbeSamplingSupport();
Mat<float> ProbeOwnershipInitialCost(PatchMatchCuda&, unsigned long long);
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

void MakeSourceTexturesDistinct(Inputs* inputs) {
  // Spatial errors distinguish sources; an affine intensity change would not
  // distinguish their NCC values reliably.
  for (int source = 1; source < static_cast<int>(inputs->images.size());
       ++source) {
    Bitmap bitmap = inputs->images[source].GetBitmap().Clone();
    for (int row = 0; row < bitmap.Height(); ++row) {
      for (int col = 0; col < bitmap.Width(); ++col) {
        const int perturbation =
            ((row * (source + 1) + col * (2 * source + 1)) % 17) - 8;
        const int value = bitmap.GetPixel(col, row)->r + perturbation;
        bitmap.SetPixel(col,
                        row,
                        BitmapColor<uint8_t>(
                            static_cast<uint8_t>(std::clamp(value, 0, 255))));
      }
    }
    inputs->images[source].SetBitmap(std::move(bitmap));
  }
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

struct Result {
  DepthMap depths;
  NormalMap normals;
  Mat<float> selection;
  // Query graph through its public API, preserving each pixel and ordered IDs.
  std::vector<std::vector<int>> consistent_images;
};

Result RunPatchMatch(Inputs* inputs,
                     const PatchMatchOptions& options,
                     const std::vector<int>& ordered_sources = {}) {
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = ordered_sources;
  if (problem.src_image_idxs.empty()) {
    for (size_t i = 1; i < inputs->images.size(); ++i) {
      problem.src_image_idxs.push_back(static_cast<int>(i));
    }
  }
  problem.images = &inputs->images;
  problem.depth_maps = &inputs->depths;
  problem.normal_maps = &inputs->normals;
  PatchMatch patch_match(options, problem);
  patch_match.Run();
  Result result{patch_match.GetDepthMap(),
                patch_match.GetNormalMap(),
                patch_match.GetSelProbMap(),
                {}};
  const ConsistencyGraph graph = patch_match.GetConsistencyGraph();
  for (size_t row = 0; row < result.depths.GetHeight(); ++row) {
    for (size_t col = 0; col < result.depths.GetWidth(); ++col) {
      int num_images = 0;
      const int* image_idxs = nullptr;
      graph.GetImageIdxs(row, col, &num_images, &image_idxs);
      result.consistent_images.emplace_back();
      if (num_images > 0) {
        result.consistent_images.back().assign(image_idxs,
                                               image_idxs + num_images);
      }
    }
  }
  return result;
}

void ExpectExactlyEqual(const Mat<float>& expected, const Mat<float>& actual) {
  ASSERT_EQ(expected.GetWidth(), actual.GetWidth());
  ASSERT_EQ(expected.GetHeight(), actual.GetHeight());
  ASSERT_EQ(expected.GetDepth(), actual.GetDepth());
  ASSERT_EQ(expected.GetData().size(), actual.GetData().size());
  for (size_t i = 0; i < expected.GetData().size(); ++i) {
    const float a = expected.GetData()[i];
    const float b = actual.GetData()[i];
    ASSERT_TRUE(std::isfinite(a)) << i;
    ASSERT_TRUE(std::isfinite(b)) << i;
    // These delivery/allocation/scheduling paths preserve the arithmetic.
    // Require byte equality within the same executable and CUDA device only.
    ASSERT_EQ(std::memcmp(&a, &b, sizeof(float)), 0)
        << "element " << i << ": " << a << " vs " << b;
  }
}

void ExpectEquivalent(const Result& expected, const Result& actual) {
  ExpectExactlyEqual(expected.depths, actual.depths);
  ExpectExactlyEqual(expected.normals, actual.normals);
  ExpectExactlyEqual(expected.selection, actual.selection);
  // Discrete source memberships must agree exactly, including empty pixels.
  ASSERT_EQ(expected.consistent_images.size(), actual.consistent_images.size());
  for (size_t i = 0; i < expected.consistent_images.size(); ++i) {
    ASSERT_EQ(expected.consistent_images[i], actual.consistent_images[i]) << i;
  }
}

void ExpectNonempty(const Result& result,
                    const Inputs& inputs,
                    const bool filter) {
  EXPECT_EQ(result.depths.GetWidth(), inputs.images[0].GetWidth());
  EXPECT_EQ(result.depths.GetHeight(), inputs.images[0].GetHeight());
  EXPECT_EQ(result.normals.GetDepth(), 3);
  EXPECT_EQ(result.selection.GetDepth(), inputs.images.size() - 1);
  EXPECT_EQ(result.depths.GetData().size(),
            result.depths.GetWidth() * result.depths.GetHeight());
  size_t valid_depths = 0;
  for (const float depth : result.depths.GetData()) {
    ASSERT_TRUE(std::isfinite(depth));
    if (depth > 0.0f) {
      ++valid_depths;
    }
  }
  // Propagation and perturbation can leave the initial sampling range.
  EXPECT_GT(valid_depths, 0);
  size_t consistent_pixels = 0;
  for (const auto& image_idxs : result.consistent_images) {
    consistent_pixels += !image_idxs.empty();
    for (const int image_idx : image_idxs) {
      EXPECT_GT(image_idx, 0);
      EXPECT_LT(image_idx, inputs.images.size());
    }
  }
  if (filter) {
    EXPECT_GT(consistent_pixels, 0);
  } else {
    EXPECT_EQ(consistent_pixels, 0);
  }
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
  for (const bool cooperative : {false, true}) {
    SCOPED_TRACE(cooperative);
    PatchMatchCuda patch_match(options, problem);
    const auto trace = internal::ProbeSweep(patch_match, cooperative, 1873);
    ExpectPropagationAnchors(trace, inputs.images[0]);
  }
}

TEST_F(CudaTest, SelectionProbabilityCapturedTupleAndDomain) {
  // Captured alpha, beta, previous and weight, retained as exact binary32 bits.
  const std::vector<uint32_t> input = {
      0x3f7ff9b0, 0x3f7ff844, 0x3f000000, 0x00000000, 0x00000000, 0x00000000,
      0x3f000000, 0x00000000, 0x3f800000, 0x3f800000, 0x3f000000, 0x00000000,
      0x3f000000, 0x3f000000, 0x3f000000, 0x3e800000, 0x3f800000, 0x3f800000,
      0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x3f000000, 0x00000000,
      0x7fc00000, 0x3f000000, 0x3f000000, 0x00000000, 0xbe800000, 0x3f000000,
      0x3f000000, 0x00000000};
  const auto out = internal::ProbeSelectionProbability(input);
  ASSERT_EQ(out.size(), 8);
  EXPECT_EQ(out[0], 0x3f800000u);  // Captured valid tuple: strict <= 1.
  EXPECT_EQ(out[1], 0x00000000u);
  EXPECT_EQ(out[2], 0x3f800000u);
  EXPECT_EQ(out[3], 0x3f000000u);
  EXPECT_EQ(out[4], 0x00000000u);
  EXPECT_GT(out[5] & 0x7fffffffu, 0x7f800000u);  // Singular 0/0 remains NaN.
  EXPECT_GT(out[6] & 0x7fffffffu, 0x7f800000u);  // NaN input remains NaN.
  EXPECT_EQ(out[7], 0xbe800000u);  // Invalid input is not silently clamped.
  std::vector<uint32_t> many;
  for (int i = 0; i < 40; ++i) {
    many.insert(many.end(), input.begin(), input.begin() + 4);
  }
  const auto many_out = internal::ProbeSelectionProbability(many);
  ASSERT_EQ(many_out.size(), 40);
  for (const auto value : many_out) EXPECT_EQ(value, 0x3f800000u);
}

TEST_F(CudaTest, DirectInstancesOwnReferenceCalibration) {
  auto a = MakeInputs(0);
  auto b = MakeInputs(1);
  ASSERT_NE(a.images[0].GetK()[0], b.images[0].GetK()[0]);
  ASSERT_NE(a.images[0].GetK()[4], b.images[0].GetK()[4]);
  auto options = MakeOptions(true, false, false);
  options.window_radius = 5;
  options.window_step = 1;
  options.sigma_spatial = 5.0;
  options.num_samples = 17;
  auto make_problem = [](Inputs& inputs) {
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    problem.src_image_idxs = {1, 2};
    problem.images = &inputs.images;
    problem.depth_maps = &inputs.depths;
    problem.normal_maps = &inputs.normals;
    return problem;
  };
  const auto problem_a = make_problem(a);
  const auto problem_b = make_problem(b);
  constexpr unsigned long long kSeed = 12345;

  PatchMatchCuda isolated(options, problem_a);
  const Mat<float> isolated_cost =
      internal::ProbeOwnershipInitialCost(isolated, kSeed);
  isolated.Run();
  const auto isolated_depth = isolated.GetDepthMap();
  const auto isolated_normal = isolated.GetNormalMap();
  const auto isolated_selection = isolated.GetSelProbMap();

  PatchMatchCuda interleaved(options, problem_a);
  PatchMatchCuda other(options, problem_b);
  const Mat<float> interleaved_cost =
      internal::ProbeOwnershipInitialCost(interleaved, kSeed);
  interleaved.Run();
  ASSERT_GT(isolated_cost.GetData().size(), 0);
  EXPECT_TRUE(std::any_of(
      isolated_cost.GetData().begin(),
      isolated_cost.GetData().end(),
      [](float cost) { return std::isfinite(cost) && cost < 1.9f; }));
  ExpectExactlyEqual(isolated_cost, interleaved_cost);
  ExpectExactlyEqual(isolated_depth, interleaved.GetDepthMap());
  ExpectExactlyEqual(isolated_normal, interleaved.GetNormalMap());
  ExpectExactlyEqual(isolated_selection, interleaved.GetSelProbMap());
  EXPECT_TRUE(std::any_of(
      isolated_depth.GetData().begin(),
      isolated_depth.GetData().end(),
      [](float depth) { return std::isfinite(depth) && depth > 0; }));
}

TEST_F(CudaTest, SameWindowScalarAndCooperativeSweep) {
  constexpr int kWidth = 35;
  constexpr int kHeight = 65;
  constexpr unsigned long long kSeed = 9182;
  static_assert(kWidth % 8 != 0 && kWidth % 32 != 0);

  struct Case {
    int sources, samples;
  };
  const Case cases[] = {{32, 15}, {32, 16}, {32, 17}, {33, 17}, {33, 33}};
  for (const auto& c : cases) {
    const int sources = c.sources;
    const int kSamples = c.samples;
    SCOPED_TRACE(::testing::Message()
                 << "sources=" << sources << " samples=" << kSamples);
    auto inputs = MakeInputs(0, 2, kHeight, kWidth);
    // Repeating a real overlapping source makes the high IDs eligible for
    // sampling rather than placing later cameras outside the reference view.
    const Image source_image = inputs.images[2];
    const DepthMap source_depth = inputs.depths[2];
    const NormalMap source_normal = inputs.normals[2];
    inputs.images[1] = source_image;
    inputs.depths[1] = source_depth;
    inputs.normals[1] = source_normal;
    while (inputs.images.size() < static_cast<size_t>(sources) + 1) {
      inputs.images.push_back(source_image);
      inputs.depths.push_back(source_depth);
      inputs.normals.push_back(source_normal);
    }
    MakeSourceTexturesDistinct(&inputs);
    // Alternating non-ground-truth depths provide a useful propagation and
    // perturbation challenge; the source geometry remains valid.
    for (int row = 0; row < kHeight; ++row) {
      for (int col = 0; col < kWidth; ++col) {
        inputs.depths[0].Set(row, col, row % 2 == 0 ? 3.2f : 5.2f);
      }
    }
    PatchMatch::Problem problem;
    problem.ref_image_idx = 0;
    for (int i = 1; i <= sources; ++i) problem.src_image_idxs.push_back(i);
    problem.images = &inputs.images;
    problem.depth_maps = &inputs.depths;
    problem.normal_maps = &inputs.normals;
    auto options = MakeOptions(true, false, false);
    options.window_radius = 5;
    options.window_step = 1;
    options.sigma_spatial = 5.0;
    options.num_samples = kSamples;

    PatchMatchCuda scalar(options, problem);
    PatchMatchCuda cooperative(options, problem);
    const auto a = internal::ProbeSweep(scalar, false, kSeed);
    const auto b = internal::ProbeSweep(cooperative, true, kSeed);
    ASSERT_EQ(a.draws.GetWidth(), kWidth);
    ASSERT_EQ(a.draws.GetHeight(), kHeight);
    ASSERT_EQ(a.draws.GetDepth(), kSamples);
    ASSERT_EQ(a.draws.GetData(), b.draws.GetData());
    ASSERT_EQ(a.winners.GetData(), b.winners.GetData());
    ASSERT_EQ(a.support.GetData(), b.support.GetData());
    ExpectExactlyEqual(a.initial_cost, b.initial_cost);
    ExpectExactlyEqual(a.hypotheses, b.hypotheses);
    ExpectExactlyEqual(a.hypothesis_costs, b.hypothesis_costs);
    ExpectExactlyEqual(a.forward, b.forward);
    ExpectExactlyEqual(a.cost, b.cost);
    ExpectExactlyEqual(a.depth, b.depth);
    ExpectExactlyEqual(a.normal, b.normal);
    ExpectExactlyEqual(a.selection, b.selection);
    for (const auto* map : {&a.selection, &b.selection}) {
      for (float value : map->GetData()) {
        ASSERT_TRUE(std::isfinite(value));
        ASSERT_GE(value, 0.0f);
        ASSERT_LE(value, 1.0f);
      }
    }
    ASSERT_EQ(a.continuation.GetWidth(), kWidth);
    ASSERT_EQ(a.continuation.GetDepth(), 3);
    ASSERT_EQ(a.continuation.GetData(), b.continuation.GetData());
    EXPECT_TRUE(std::any_of(a.continuation.GetData().begin(),
                            a.continuation.GetData().end(),
                            [](uint32_t draw) { return draw != 0; }));

    int supported_pixels = 0;
    int supported_rows = 0;
    int duplicate_pixels = 0;
    int high31 = 0;
    int high32 = 0;
    int strict_new_winners = 0;
    int distinct_sampled_costs = 0;
    for (int row = 0; row < kHeight; ++row) {
      bool row_supported = false;
      for (int col = 0; col < kWidth; ++col) {
        const int supported = a.support.Get(row, col);
        ASSERT_TRUE(supported == 0 || supported == 1);
        supported_pixels += supported;
        row_supported = row_supported || supported;
        std::vector<bool> seen(sources, false);
        bool duplicate = false;
        int first_sampled_source = -1;
        bool distinct_sampled_cost = false;
        for (int sample = 0; sample < kSamples; ++sample) {
          const int source = a.draws.Get(row, col, sample);
          ASSERT_GE(source, -1);
          ASSERT_LT(source, sources);
          if (source >= 0) {
            ASSERT_EQ(supported, 1);
            if (sample < 16) {
              duplicate = duplicate || seen[source];
              seen[source] = true;
            }
            if (first_sampled_source < 0) {
              first_sampled_source = source;
            } else {
              distinct_sampled_cost =
                  distinct_sampled_cost ||
                  a.initial_cost.Get(row, col, source) !=
                      a.initial_cost.Get(row, col, first_sampled_source);
            }
            high31 += source == 31;
            high32 += source == 32;
          }
        }
        duplicate_pixels += duplicate;
        distinct_sampled_costs += distinct_sampled_cost;
        const int winner = a.winners.Get(row, col);
        ASSERT_GE(winner, 0);
        ASSERT_LT(winner, 5);
        bool different = false;
        for (int component = 0; component < 4; ++component) {
          const float chosen =
              a.hypotheses.Get(row, col, winner * 4 + component);
          const float incumbent = a.hypotheses.Get(row, col, component);
          different = different || chosen != incumbent;
          const float published = component == 0
                                      ? a.depth.Get(row, col)
                                      : a.normal.Get(row, col, component - 1);
          // Preserve the selected tuple's exact bits, including signed zero.
          ASSERT_EQ(std::memcmp(&chosen, &published, sizeof(float)), 0);
        }
        if (supported && winner > 0 && different &&
            a.hypothesis_costs.Get(row, col, winner) <
                a.hypothesis_costs.Get(row, col, 0)) {
          ++strict_new_winners;
        }
      }
      supported_rows += row_supported;
    }
    EXPECT_GT(supported_pixels, 0);
    EXPECT_GT(supported_rows, 1);
    EXPECT_GT(duplicate_pixels, 0);
    EXPECT_GT(distinct_sampled_costs, 0);
    EXPECT_GT(high31, 0);
    if (sources == 33) {
      EXPECT_GT(high32, 0);
    }
    EXPECT_GT(strict_new_winners, 0);
    const std::string property_prefix = "sources_" + std::to_string(sources) +
                                        "_samples_" + std::to_string(kSamples);
    RecordProperty(property_prefix + "_supported_pixels", supported_pixels);
    RecordProperty(property_prefix + "_duplicate_pixels", duplicate_pixels);
    RecordProperty(property_prefix + "_distinct_sampled_costs",
                   distinct_sampled_costs);
    RecordProperty(property_prefix + "_draws_31", high31);
    RecordProperty(property_prefix + "_draws_32", high32);
    RecordProperty(property_prefix + "_strict_winners", strict_new_winners);
  }
}

TEST_F(CudaTest, SamplingSupportHelpersPreserveNativeDrawCount) {
  EXPECT_EQ(internal::ProbeSamplingSupport(), 0);
}

TEST_F(CudaTest, EmptySamplingSupportRetainsScalarAndCooperativeIncumbents) {
  int configurations = 0;
  for (const int radius : {2, 5}) {
    for (const int sources : {2, 33}) {
      for (const int samples : {1, 17}) {
        SCOPED_TRACE(::testing::Message() << "radius=" << radius << " sources="
                                          << sources << " samples=" << samples);
        auto inputs = MakeInputs(0, sources, 19, 23);
        for (auto& image : inputs.images) {
          Bitmap black(image.GetWidth(), image.GetHeight(), false);
          black.Fill(BitmapColor<uint8_t>(0));
          image.SetBitmap(std::move(black));
        }
        auto options = MakeOptions(true, false, false);
        options.window_radius = radius;
        options.num_samples = samples;
        // Constant patches have cost2; its FP32 likelihood exp(-512)
        // underflows to zero, giving finite empty sampling support.
        options.ncc_sigma = 1.0f / 16.0f;
        const auto result = RunPatchMatch(&inputs, options);
        ++configurations;
        ASSERT_EQ(result.depths.GetWidth(), 23);
        ASSERT_EQ(result.depths.GetHeight(), 19);
        ASSERT_EQ(result.depths.GetDepth(), 1);
        ASSERT_EQ(result.depths.GetData().size(), 23 * 19);
        ASSERT_EQ(result.normals.GetWidth(), 23);
        ASSERT_EQ(result.normals.GetHeight(), 19);
        ASSERT_EQ(result.normals.GetDepth(), 3);
        ASSERT_EQ(result.normals.GetData().size(), 23 * 19 * 3);
        ASSERT_EQ(result.selection.GetWidth(), 23);
        ASSERT_EQ(result.selection.GetHeight(), 19);
        ASSERT_EQ(result.selection.GetDepth(), sources);
        ASSERT_EQ(result.selection.GetData().size(), 23 * 19 * sources);
        ExpectExactlyEqual(inputs.depths[0], result.depths);
        ExpectExactlyEqual(inputs.normals[0], result.normals);
        EXPECT_TRUE(
            std::all_of(result.selection.GetData().begin(),
                        result.selection.GetData().end(),
                        [](float probability) { return probability == 0.0f; }));
      }
    }
  }
  EXPECT_EQ(configurations, 8);
  RecordProperty("empty_support_configurations", configurations);
}

TEST_F(CudaTest, SourcePackingPreservesDrawIdentityAndOrderedFold) {
  constexpr int kLanes = 8;
  for (const int sources : {32, 33}) {
    std::vector<float> values(sources);
    for (int source = 0; source < sources; ++source) {
      values[source] = (source % 7 + 1) * 0.125f;
    }
    values[0] = 1.0f;
    values[31] = std::ldexp(1.0f, -24);
    values[17] = 0.125f;
    values[1] = 0.75f;
    const auto check = [&](const std::vector<int>& pattern) {
      const size_t count = pattern.size();
      std::vector<int> draws(count * kLanes);
      std::vector<float> expected(kLanes, 0.0f);
      for (int lane = 0; lane < kLanes; ++lane) {
        for (size_t sample = 0; sample < count; ++sample) {
          const int original = pattern[sample];
          const int source = original < 0 ? -1 : (original + lane) % sources;
          draws[lane * count + sample] = source;
          // Independent scalar oracle: no packing, deduplication, ranking,
          // masks or alias mapping, and no production fold helper.
          if (source >= 0) expected[lane] += values[source];
        }
      }
      for (const auto& config :
           {std::pair{false, 1}, std::pair{true, 1}, std::pair{true, 16}}) {
        SCOPED_TRACE(::testing::Message()
                     << "sources=" << sources << " samples=" << count
                     << " cooperative=" << config.first
                     << " window=" << config.second);
        const auto actual = internal::ProbeSampledSources(
            draws, values, config.second, config.first);
        ASSERT_EQ(actual.first, draws);
        ASSERT_EQ(actual.second.size(), expected.size());
        for (int lane = 0; lane < kLanes; ++lane) {
          EXPECT_EQ(actual.second[lane], expected[lane]) << lane;
        }
      }
    };
    const std::array<int, 6> seed = {31, 0, 31, 17, -1, 1};
    for (const size_t samples : {15, 16, 17}) {
      std::vector<int> pattern(samples);
      for (size_t i = 0; i < samples; ++i) pattern[i] = seed[i % seed.size()];
      if (sources == 33) pattern[7] = 32;
      pattern.back() = -1;
      check(pattern);
    }
    check(std::vector<int>(17, -1));
    // Confirm that this fixture detects addition reordering in the valid,
    // finite nonnegative cost range instead of using only exact sums.
    float ordered = 0;
    for (const int source : {0, 31, 31}) ordered += values[source];
    float reordered = 0;
    for (const int source : {31, 31, 0}) reordered += values[source];
    ASSERT_NE(ordered, reordered);
    check({0, 31, 31});
  }
}

TEST_F(CudaTest, NativeDenseSingleReferenceVariedShapesAndModes) {
  struct Case {
    int sources, samples, width, height, radius, step;
    bool geo, filter;
  };
  const Case cases[] = {
      {1, 15, 47, 35, 5, 1, false, false},
      {17, 16, 39, 67, 5, 1, false, true},
      {32, 17, 35, 65, 5, 1, true, true},
      {33, 15, 17, 129, 5, 1, true, false},
      {2, 3, 39, 53, 2, 1, true, true},
      {2, 4, 35, 67, 5, 2, false, true},
      {1, 17, 17, 65, 17, 2, false, false},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message()
                 << c.sources << "/" << c.samples << " " << c.width << "x"
                 << c.height << " radius=" << c.radius << " step=" << c.step
                 << " geo=" << c.geo << " filter=" << c.filter);
    Inputs inputs = MakeInputs(0, c.sources, c.height, c.width);
    auto options = MakeOptions(c.geo, c.filter, false);
    options.window_radius = c.radius;
    options.window_step = c.step;
    options.num_samples = c.samples;
    const Result first = RunPatchMatch(&inputs, options);
    const Result repeated = RunPatchMatch(&inputs, options);
    ExpectNonempty(first, inputs, c.filter);
    ExpectNonempty(repeated, inputs, c.filter);
    ExpectEquivalent(first, repeated);
  }
}

TEST_F(CudaTest, InvalidSourceDepthIsExcludedWithoutDiscardingUsefulViews) {
  auto inputs = MakeInputs(0, 6);
  auto options = MakeOptions(true, true, false);
  const PatchMatchOptions defaults;
  options.window_radius = 5;
  options.sigma_spatial = defaults.sigma_spatial;
  options.geom_consistency_max_cost = defaults.geom_consistency_max_cost;
  options.filter_geom_consistency_max_cost =
      defaults.filter_geom_consistency_max_cost;
  const std::vector<int> order{3, 6, 1, 5, 2, 4};
  const Result valid = RunPatchMatch(&inputs, options, order);
  ExpectNonempty(valid, inputs, true);
  size_t source_six_memberships = 0;
  for (const auto& members : valid.consistent_images)
    source_six_memberships += std::count(members.begin(), members.end(), 6);
  ASSERT_GT(source_six_memberships, 0)
      << "The all-valid control must observe source 6";
  for (size_t row = 0; row < inputs.depths[6].GetHeight(); ++row) {
    for (size_t col = 0; col < inputs.depths[6].GetWidth(); ++col)
      inputs.depths[6].Set(row, col, 0.0f);
  }
  const Result result = RunPatchMatch(&inputs, options, order);
  ExpectNonempty(result, inputs, true);
  size_t useful_memberships = 0;
  for (const auto& members : result.consistent_images) {
    for (const int source : members) {
      EXPECT_NE(source, 6);
      ++useful_memberships;
    }
  }
  EXPECT_GT(useful_memberships, 0);
}

TEST_F(CudaTest, AllInvalidSourceDepthsPublishSizedRejectedOutputs) {
  auto inputs = MakeInputs(0);
  for (size_t source = 1; source < inputs.depths.size(); ++source) {
    for (size_t row = 0; row < inputs.depths[source].GetHeight(); ++row) {
      for (size_t col = 0; col < inputs.depths[source].GetWidth(); ++col)
        inputs.depths[source].Set(row, col, 0.0f);
    }
  }
  auto options = MakeOptions(true, true, false);
  const PatchMatchOptions defaults;
  options.window_radius = 5;
  options.sigma_spatial = defaults.sigma_spatial;
  options.geom_consistency_max_cost = defaults.geom_consistency_max_cost;
  options.filter_geom_consistency_max_cost =
      defaults.filter_geom_consistency_max_cost;
  const Result result = RunPatchMatch(&inputs, options);
  ASSERT_EQ(result.depths.GetWidth(), inputs.images[0].GetWidth());
  ASSERT_EQ(result.depths.GetHeight(), inputs.images[0].GetHeight());
  ASSERT_GT(result.depths.GetData().size(), 0);
  ASSERT_EQ(result.normals.GetWidth(), result.depths.GetWidth());
  ASSERT_EQ(result.normals.GetHeight(), result.depths.GetHeight());
  ASSERT_EQ(result.normals.GetDepth(), 3);
  ASSERT_EQ(result.selection.GetWidth(), result.depths.GetWidth());
  ASSERT_EQ(result.selection.GetHeight(), result.depths.GetHeight());
  ASSERT_EQ(result.selection.GetDepth(), inputs.images.size() - 1);
  ASSERT_EQ(result.normals.GetData().size(),
            result.depths.GetData().size() * 3);
  ASSERT_EQ(result.selection.GetData().size(),
            result.depths.GetData().size() * (inputs.images.size() - 1));
  ASSERT_EQ(result.consistent_images.size(), result.depths.GetData().size());
  for (const float depth : result.depths.GetData()) EXPECT_EQ(depth, 0.0f);
  for (const float normal : result.normals.GetData()) EXPECT_EQ(normal, 0.0f);
  for (const float probability : result.selection.GetData()) {
    EXPECT_TRUE(std::isfinite(probability));
    EXPECT_GE(probability, 0.0f);
    EXPECT_LE(probability, 1.0f);
  }
  for (const auto& members : result.consistent_images)
    EXPECT_TRUE(members.empty());
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
