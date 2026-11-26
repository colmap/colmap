// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/gpu_mat_prng.h"
#include "colmap/mvs/patch_match_rng_perimeter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
namespace colmap {
namespace mvs {
namespace {
class PerimeterDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count == 0)) {
      GTEST_SKIP() << cudaGetErrorString(status);
    }
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};
// Independent oracle: original 137f full-map initializer, unchanged seed math.
__global__ void LegacySeed(GpuMatView<curandState> output) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t uniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
  const size_t id = uniqueBlockIndex * blockDim.y * blockDim.x +
                    threadIdx.y * blockDim.x + threadIdx.x;
  if (col < output.GetWidth() && row < output.GetHeight()) {
    curand_init(id, 0, 0, &output.GetRef(row, col));
  }
}
// Original 137f fill kernel; do not call the candidate interval helper here.
template <typename T>
__global__ void LegacyFill(GpuMatView<T> output,
                           GpuMatView<curandState> states,
                           T min_value,
                           T max_value) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    curandState state = states.Get(row, col);
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      const T value =
          curand_uniform(&state) * (max_value - min_value) + min_value;
      output.Set(row, col, slice, value);
    }
    states.Set(row, col, state);
  }
}
// GenerateRandomNormal's original draw/rejection sequence. Its remaining vector
// arithmetic and facing correction consume no random numbers.
__device__ int NormalDraws(curandState* state) {
  float s = 2.0f;
  int pairs = 0;
  while (s >= 1.0f) {
    const float v1 = 2.0f * curand_uniform(state) - 1.0f;
    const float v2 = 2.0f * curand_uniform(state) - 1.0f;
    s = v1 * v1 + v2 * v2;
    ++pairs;
  }
  return pairs;
}
__global__ void AdvanceFullNormals(GpuMatView<curandState> states,
                                   GpuMatView<int> retries) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= states.height || col >= states.width) return;
  curandState state = states.Get(row, col);
  retries.Set(row, col, NormalDraws(&state));
  states.Set(row, col, state);
}
__global__ void LocalInitialize(GpuMatView<curandState> perimeter,
                                GpuMatView<float> depths,
                                PatchMatchRngPerimeter layout,
                                bool photo) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= layout.height || col >= layout.width) return;
  curandState state;
  curand_init(internal::GpuMatPRNGSeed(row, col, layout.width), 0, 0, &state);
  if (photo) {
    depths.Set(row, col, internal::DrawUniformInterval(3.0f, 5.0f, &state));
    NormalDraws(&state);
  }
  if (row == 0 || col == 0 || row + 1 == layout.height ||
      col + 1 == layout.width)
    perimeter.Set(0, layout.Index(row, col), state);
}
__global__ void DrawPerimeter(GpuMatView<curandState> perimeter,
                              GpuMatView<float> output,
                              PatchMatchRngPerimeter layout,
                              unsigned rotation) {
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= output.width) return;
  const size_t slot = layout.SweepIndex(rotation, col);
  curandState state = perimeter.Get(0, slot);
  for (size_t slice = 0; slice < output.depth; ++slice)
    output.Set(
        0, col, slice, internal::DrawUniformInterval(-3.0f, 7.0f, &state));
  perimeter.Set(0, slot, state);
}
template <typename T>
std::vector<T> Read(const GpuMat<T>& map) {
  std::vector<T> out(map.GetWidth() * map.GetHeight() * map.GetDepth());
  map.CopyToHost(out.data(), map.GetWidth() * sizeof(T));
  return out;
}
void CheckEvolution(bool photo) {
  bool saw_retry = false;
  for (const auto& shape :
       std::array<std::pair<size_t, size_t>, 9>{{{1, 1},
                                                 {1, 9},
                                                 {9, 1},
                                                 {2, 7},
                                                 {7, 2},
                                                 {31, 15},
                                                 {32, 16},
                                                 {33, 17},
                                                 {47, 35}}}) {
    const auto [width, height] = shape;
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    const PatchMatchRngPerimeter layout{width, height};
    auto full = std::make_unique<GpuMatPRNG>(width, height);
    const dim3 legacy_block(32, 16),
        grid((width + 31) / 32, (height + 15) / 16);
    LegacySeed<<<grid, legacy_block>>>(full->View());
    GpuMat<curandState> perimeter(layout.Count(), 1);
    GpuMat<float> expected_depth(width, height), actual_depth(width, height);
    if (photo) {
      LegacyFill<<<grid, legacy_block>>>(
          expected_depth.View(), full->View(), 3.0f, 5.0f);
      GpuMat<int> retries(width, height);
      AdvanceFullNormals<<<grid, legacy_block>>>(full->View(), retries.View());
      if (width * height > 1) {
        const auto counts = Read(retries);
        saw_retry = saw_retry || std::any_of(counts.begin(),
                                             counts.end(),
                                             [](int n) { return n > 1; });
      }
    }
    // Deliberately different launch shape: seeds depend on original image
    // tiles.
    const dim3 local_block(8, 8), local_grid((width + 7) / 8, (height + 7) / 8);
    LocalInitialize<<<local_grid, local_block>>>(
        perimeter.View(), actual_depth.View(), layout, photo);
    CUDA_SYNC_AND_CHECK();
    if (photo) EXPECT_EQ(Read(expected_depth), Read(actual_depth));
    for (unsigned sweep = 0; sweep < 8; ++sweep) {
      const unsigned rotation = sweep % 4;
      GpuMat<float> expected(full->GetWidth(), 1, 65),
          actual(full->GetWidth(), 1, 65);
      const dim3 row_grid((full->GetWidth() + 31) / 32, 1);
      LegacyFill<<<row_grid, legacy_block>>>(
          expected.View(), full->View(), -3.0f, 7.0f);
      DrawPerimeter<<<(full->GetWidth() + 127) / 128, 128>>>(
          perimeter.View(), actual.View(), layout, rotation);
      CUDA_SYNC_AND_CHECK();
      EXPECT_EQ(Read(expected), Read(actual)) << "sweep " << sweep;
      auto rotated =
          std::make_unique<GpuMatPRNG>(full->GetHeight(), full->GetWidth());
      full->Rotate(rotated.get());
      full.swap(rotated);
    }
  }
  if (photo) EXPECT_TRUE(saw_retry);
}
TEST_F(PerimeterDeviceTest, PhotoStateEvolutionMatchesFullMap) {
  CheckEvolution(true);
}
TEST_F(PerimeterDeviceTest, GeometricStateEvolutionMatchesFullMap) {
  CheckEvolution(false);
}
template <typename T>
void CheckScalarDraws() {
  GpuMatPRNG old_states(33, 17), new_states(33, 17);
  const dim3 block(32, 16), grid(2, 2);
  LegacySeed<<<grid, block>>>(old_states.View());
  LegacySeed<<<grid, block>>>(new_states.View());
  GpuMat<T> expected(33, 17, 3), actual(33, 17, 3);
  LegacyFill<<<grid, block>>>(expected.View(), old_states.View(), T(2), T(9));
  actual.FillWithRandomNumbers(T(2), T(9), new_states);
  CUDA_SYNC_AND_CHECK();
  EXPECT_EQ(Read(expected), Read(actual));
}
TEST_F(PerimeterDeviceTest, SharedDrawPreservesScalarConversions) {
  CheckScalarDraws<int8_t>();
  CheckScalarDraws<int16_t>();
  CheckScalarDraws<int32_t>();
  CheckScalarDraws<int64_t>();
  CheckScalarDraws<float>();
  CheckScalarDraws<double>();
}
TEST(PatchMatchRngPerimeter, RngPerimeterLayoutPreservesRotatedRowZero) {
  // Host-only arithmetic oracle: the original CUDA operands are uint32_t.
  // Large coordinates exercise wrapping without allocating an image or state.
  const auto legacy_seed = [](uint32_t block_x,
                              uint32_t block_y,
                              uint32_t grid_x,
                              uint32_t thread_x,
                              uint32_t thread_y) {
    const size_t unique_block_index = block_y * grid_x + block_x;
    return unique_block_index * uint32_t{16} * uint32_t{32} +
           thread_y * uint32_t{32} + thread_x;
  };
  const std::array<std::array<size_t, 4>, 10> seed_cases = {
      {{1, 1, 0, 0},
       {1, 9, 8, 0},
       {9, 1, 0, 8},
       {31, 15, 14, 30},
       {32, 16, 15, 31},
       {33, 17, 16, 32},
       {47, 35, 34, 46},
       {4194303, 524289, 524287, 4194302},
       {4194303, 524289, 524288, 0},
       {4194303, 524289, 524288, 32}}};
  for (const auto& [width, height, row, col] : seed_cases) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height) + " at " +
                 std::to_string(row) + "," + std::to_string(col));
    ASSERT_LT(row, height);
    ASSERT_LT(col, width);
    const size_t expected =
        legacy_seed(static_cast<uint32_t>(col / 32),
                    static_cast<uint32_t>(row / 16),
                    static_cast<uint32_t>((width - 1) / 32 + 1),
                    static_cast<uint32_t>(col % 32),
                    static_cast<uint32_t>(row % 16));
    EXPECT_EQ(internal::GpuMatPRNGSeed(row, col, width), expected);
  }

  const std::array<std::pair<std::size_t, std::size_t>, 7> shapes = {
      {{1, 1}, {1, 9}, {9, 1}, {2, 7}, {7, 2}, {33, 17}, {47, 35}}};
  for (const auto& [width, height] : shapes) {
    SCOPED_TRACE(std::to_string(width) + "x" + std::to_string(height));
    const PatchMatchRngPerimeter layout{width, height};
    const std::size_t expected_count =
        width == 1 ? height
                   : (height == 1 ? width : 2 * width + 2 * height - 4);
    ASSERT_EQ(layout.Count(), expected_count);
    std::vector<bool> seen(expected_count, false);
    using Row = std::vector<std::size_t>;
    std::vector<Row> rotated(height, Row(width));
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        rotated[row][col] = row * width + col;
        if (row != 0 && col != 0 && row + 1 != height && col + 1 != width) {
          continue;
        }
        const std::size_t index = layout.Index(row, col);
        ASSERT_LT(index, expected_count);
        EXPECT_FALSE(seen[index]);
        seen[index] = true;
        std::size_t recovered_row = 0, recovered_col = 0;
        layout.Coordinates(index, &recovered_row, &recovered_col);
        EXPECT_EQ(recovered_row, row);
        EXPECT_EQ(recovered_col, col);
      }
    }
    for (const bool visited : seen) EXPECT_TRUE(visited);

    for (unsigned rotation = 0; rotation < 4; ++rotation) {
      const std::size_t current_height = rotated.size();
      const std::size_t current_width = rotated.front().size();
      std::vector<bool> active(expected_count, false);
      for (std::size_t col = 0; col < current_width; ++col) {
        const std::size_t index = layout.SweepIndex(rotation, col);
        ASSERT_LT(index, expected_count);
        EXPECT_FALSE(active[index]);
        active[index] = true;
        std::size_t original_row = 0, original_col = 0;
        layout.Coordinates(index, &original_row, &original_col);
        EXPECT_EQ(original_row * width + original_col, rotated[0][col]);
      }
      std::vector<Row> next(current_width, Row(current_height));
      for (std::size_t row = 0; row < current_height; ++row) {
        for (std::size_t col = 0; col < current_width; ++col) {
          next[current_width - 1 - col][row] = rotated[row][col];
        }
      }
      rotated.swap(next);
    }
    EXPECT_EQ(layout.SweepIndex(0, width - 1), layout.SweepIndex(1, 0));
    EXPECT_EQ(layout.SweepIndex(1, height - 1), layout.SweepIndex(2, 0));
    EXPECT_EQ(layout.SweepIndex(2, width - 1), layout.SweepIndex(3, 0));
    EXPECT_EQ(layout.SweepIndex(3, height - 1), layout.SweepIndex(0, 0));
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
