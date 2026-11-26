// SPDX-License-Identifier: BSD-3-Clause

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BOOST_PP_VARIADICS 0
#endif

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/packed_consistency_mask.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// Inspect the representation through a reference, independently of
// floating-point classification assumptions under fast-math.
bool HasFiniteFloatBits(const float& value) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  static_assert(std::numeric_limits<float>::is_iec559);
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000U) != 0x7f800000U;
}

class PackedConsistencyMaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};

// Exactly one thread owns each pixel, as required by the production view.
__global__ void SetAll(PackedConsistencyMaskView mask, size_t sources) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= mask.bytes.width || row >= mask.bytes.height) return;
  for (size_t source = 0; source < sources; ++source) {
    mask.Set(row, col, source, 1);
  }
}

__global__ void ClearOne(PackedConsistencyMaskView mask, size_t sources) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= mask.bytes.width || row >= mask.bytes.height) return;
  mask.Set(row, col, (row * 7 + col * 11) % sources, 0);
}

// Reference is a dense boolean matrix per pixel, not a byte-packed formula.
std::vector<uint8_t> RotateReference(const std::vector<uint8_t>& dense,
                                     size_t width,
                                     size_t height,
                                     size_t sources) {
  std::vector<uint8_t> out(dense.size());
  for (size_t r = 0; r < width; ++r) {
    for (size_t c = 0; c < height; ++c) {
      for (size_t s = 0; s < sources; ++s) {
        out[(r * height + c) * sources + s] =
            dense[(c * width + (width - 1 - r)) * sources + s];
      }
    }
  }
  return out;
}

void CheckDecoded(const GpuMat<uint8_t>& mask,
                  const std::vector<uint8_t>& dense,
                  size_t sources) {
  const size_t width = mask.GetWidth(), height = mask.GetHeight();
  const size_t planes = PackedConsistencyMaskDepth(sources);
  ASSERT_EQ(mask.GetDepth(), planes);
  ASSERT_GE(mask.GetPitch(), width * sizeof(uint8_t));
  std::vector<uint8_t> bytes(width * height * planes);
  mask.CopyToHost(bytes.data(), width * sizeof(uint8_t));
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      std::vector<int> actual, expected;
      for (size_t source = 0; source < sources; ++source) {
        // Deliberately descending IDs verify source-order decode, not sorting.
        const int id = 1000 + static_cast<int>((sources - source) * 17);
        if (bytes[(source / 8) * width * height + row * width + col] &
            (1u << (source % 8)))
          actual.push_back(id);
        if (dense[(row * width + col) * sources + source])
          expected.push_back(id);
      }
      EXPECT_EQ(actual, expected) << "pixel " << row << "," << col;
      if (sources % 8 != 0) {
        const auto tail =
            bytes[(planes - 1) * width * height + row * width + col];
        EXPECT_EQ(tail >> (sources % 8), 0)
            << "tail bits at " << row << "," << col;
      }
    }
  }
}

TEST_F(PackedConsistencyMaskTest, DeviceWriteClearDecodeAndFourRotations) {
  constexpr std::pair<uint32_t, bool> finite_cases[] = {
      {0x00000000U, true},
      {0x80000000U, true},  // Signed zero.
      {0x00000001U, true},
      {0x80000001U, true},  // Signed subnormals.
      {0x007fffffU, true},
      {0x00800000U, true},  // Normal/subnormal boundary.
      {0x3f800000U, true},
      {0xbf800000U, true},  // Signed one.
      {0x7f7fffffU, true},
      {0xff7fffffU, true},  // Largest finite values.
      {0x7f800000U, false},
      {0xff800000U, false},  // Signed infinity.
      {0x7fc00000U, false},
      {0xffc12345U, false},  // Quiet NaNs.
      {0x7f800001U, false},
      {0xff800001U, false}  // Signaling NaNs.
  };
  for (const auto& [bits, expected] : finite_cases) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    EXPECT_EQ(HasFiniteFloatBits(value), expected) << "float bits: " << bits;
  }

  constexpr size_t width = 35, height = 19;
  for (const size_t sources : {size_t{1},
                               size_t{7},
                               size_t{8},
                               size_t{9},
                               size_t{17},
                               size_t{31},
                               size_t{32},
                               size_t{33}}) {
    SCOPED_TRACE(::testing::Message() << "sources=" << sources);
    const size_t planes = PackedConsistencyMaskDepth(sources);
    ASSERT_EQ(planes, (sources + 7) / 8);
    GpuMat<uint8_t> a(width, height, planes), b(height, width, planes);
    a.FillWithScalar(0);
    const dim3 block(16, 8);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    SetAll<<<grid, block>>>(PackedConsistencyMaskView{a.View()}, sources);
    CUDA_SYNC_AND_CHECK();
    std::vector<uint8_t> dense(width * height * sources, 1);
    CheckDecoded(a, dense, sources);
    ClearOne<<<grid, block>>>(PackedConsistencyMaskView{a.View()}, sources);
    CUDA_SYNC_AND_CHECK();
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        dense[(row * width + col) * sources + (row * 7 + col * 11) % sources] =
            0;
      }
    }
    // The complete decoded vectors prove clearing did not alter neighbors.
    CheckDecoded(a, dense, sources);
    auto* current = &a;
    auto* next = &b;
    size_t current_width = width, current_height = height;
    for (int rotation = 0; rotation < 4; ++rotation) {
      current->Rotate(next);
      dense = RotateReference(dense, current_width, current_height, sources);
      std::swap(current_width, current_height);
      std::swap(current, next);
      CheckDecoded(*current, dense, sources);
    }
  }
}

constexpr float kDepthMin = 1.0f;
constexpr float kDepthMax = 8.0f;
constexpr float kPlaneZ = 4.0f;
struct Inputs {
  std::vector<Image> images;
  std::vector<DepthMap> depths;
  std::vector<NormalMap> normals;
};

// Render the same textured world plane through differently calibrated views.
Inputs MakeInputs(const int num_source_images,
                  const int width,
                  const int height) {
  Inputs inputs;
  for (int i = 0; i <= num_source_images; ++i) {
    const int image_width = width + 2 * i;
    const int image_height = height + 2 * i;
    const float fx = 48.0f + 3.0f * i;
    const float fy = 57.0f + 2.0f * i;
    const float cx = image_width * 0.5f + 0.7f + i * 0.2f;
    const float cy = image_height * 0.5f - 1.1f - i * 0.15f;
    const float yaw = (i - 1) * 0.035f;
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    const float camera_x = (i - 1) * 0.25f;
    const float camera_y = (i - 1) * 0.04f;
    const float K[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
    const float R[9] = {c, 0, s, 0, 1, 0, -s, 0, c};
    const float T[3] = {-c * camera_x, -camera_y, s * camera_x};

    Image image(
        "synthetic-" + std::to_string(i), image_width, image_height, K, R, T);
    Bitmap bitmap(image_width, image_height, false);
    DepthMap depth(image_width, image_height, kDepthMin, kDepthMax);
    NormalMap normal(image_width, image_height);
    for (int row = 0; row < image_height; ++row) {
      for (int col = 0; col < image_width; ++col) {
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
                              const int samples) {
  PatchMatchOptions options;
  options.gpu_index = "0";
  options.depth_min = kDepthMin;
  options.depth_max = kDepthMax;
  options.window_radius = 5;
  options.window_step = 1;
  options.num_samples = samples;
  options.num_iterations = 2;
  options.geom_consistency = geometry;
  options.filter = filter;
  options.filter_min_ncc = -1.0;
  options.filter_min_triangulation_angle = 0.0;
  options.filter_min_num_consistent = 1;
  options.filter_geom_consistency_max_cost = 10.0;
  return options;
}

PatchMatch::Problem MakeProblem(Inputs* inputs) {
  PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  for (int source = static_cast<int>(inputs->images.size()) - 1; source > 0;
       --source) {
    problem.src_image_idxs.push_back(source);
  }
  problem.images = &inputs->images;
  problem.depth_maps = &inputs->depths;
  problem.normal_maps = &inputs->normals;
  return problem;
}

void CheckProductionResult(const PatchMatch& matcher,
                           const Inputs& inputs,
                           const bool expect_graph,
                           const bool reject_all) {
  const auto depth = matcher.GetDepthMap();
  const auto normal = matcher.GetNormalMap();
  const auto selection = matcher.GetSelProbMap();
  const auto graph = matcher.GetConsistencyGraph();
  const size_t width = inputs.images[0].GetWidth();
  const size_t height = inputs.images[0].GetHeight();
  const int sources = static_cast<int>(inputs.images.size()) - 1;
  ASSERT_EQ(depth.GetDepth(), 1);
  ASSERT_EQ(normal.GetDepth(), 3);
  ASSERT_EQ(selection.GetDepth(), sources);
  for (const auto* map : {static_cast<const Mat<float>*>(&depth),
                          static_cast<const Mat<float>*>(&normal),
                          &selection}) {
    ASSERT_EQ(map->GetWidth(), width);
    ASSERT_EQ(map->GetHeight(), height);
    ASSERT_EQ(map->GetData().size(), width * height * map->GetDepth());
    ASSERT_TRUE(std::all_of(
        map->GetData().begin(), map->GetData().end(), HasFiniteFloatBits));
  }
  if (reject_all) {
    for (const auto* map : {static_cast<const Mat<float>*>(&depth),
                            static_cast<const Mat<float>*>(&normal)}) {
      EXPECT_TRUE(std::all_of(map->GetData().begin(),
                              map->GetData().end(),
                              [](float value) { return value == 0.0f; }));
    }
  } else {
    EXPECT_TRUE(std::any_of(depth.GetData().begin(),
                            depth.GetData().end(),
                            [](float value) { return value > 0.0f; }));
  }

  int max_membership = 0;
  bool saw_tail_source = false;
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      int count = 0;
      const int* ids = nullptr;
      graph.GetImageIdxs(row, col, &count, &ids);
      ASSERT_GE(count, 0);
      ASSERT_LE(count, sources);
      int previous = sources + 1;
      if (count > 0) {
        ASSERT_NE(ids, nullptr);
        for (int i = 0; i < count; ++i) {
          ASSERT_GT(ids[i], 0);
          ASSERT_LT(ids[i], previous) << row << "," << col;
          previous = ids[i];
          // Descending source order puts image 1 in the final logical slot.
          saw_tail_source = saw_tail_source || ids[i] == 1;
        }
      } else {
        EXPECT_EQ(ids, nullptr);
      }
      max_membership = std::max(max_membership, count);
    }
  }
  if (expect_graph) {
    EXPECT_GT(max_membership, 8);
    EXPECT_TRUE(saw_tail_source);
  } else {
    EXPECT_EQ(max_membership, 0);
  }
}

void RunProductionProblem(const int sources,
                          const int width,
                          const int height,
                          const int samples,
                          const bool geometry,
                          const bool filter,
                          const bool reject_all = false,
                          const int window_radius = 5) {
  Inputs inputs = MakeInputs(sources, width, height);
  auto options = MakeOptions(geometry, filter, samples);
  options.window_radius = window_radius;
  if (reject_all) {
    options.filter_min_num_consistent = sources + 1;
  }
  const auto problem = MakeProblem(&inputs);
  PatchMatch matcher(options, problem);
  matcher.Run();
  CheckProductionResult(matcher, inputs, filter && !reject_all, reject_all);
}

TEST_F(PackedConsistencyMaskTest, InitializedOwnerHasNoConsistencyBeforeRun) {
  Inputs inputs = MakeInputs(2, 39, 67);
  const auto options = MakeOptions(false, true, 3);
  const auto problem = MakeProblem(&inputs);
  PatchMatchCuda matcher(options, problem);
  EXPECT_TRUE(matcher.GetConsistentImageIdxs().empty());
}

TEST_F(PackedConsistencyMaskTest, PhotoGraphDecodesOrderedSourcesAndTail) {
  RunProductionProblem(17, 39, 67, 16, false, true);
}

TEST_F(PackedConsistencyMaskTest,
       ScalarFallbackGraphDecodesOrderedSourcesAndTail) {
  RunProductionProblem(17, 39, 67, 17, false, true, false, 4);
}

TEST_F(PackedConsistencyMaskTest, GeometricGraphDecodesOrderedSourcesAndTail) {
  RunProductionProblem(33, 35, 67, 17, true, true);
}

TEST_F(PackedConsistencyMaskTest, DisabledFilterHasNoConsistencyAfterRun) {
  RunProductionProblem(33, 17, 129, 15, true, false);
}

TEST_F(PackedConsistencyMaskTest, RejectedPixelsClearAllSourceConsistency) {
  RunProductionProblem(17, 39, 67, 16, false, true, true);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
