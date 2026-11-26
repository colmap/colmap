// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/hash_containers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#if defined(COLMAP_MVS_CUDA_ALLOCATION_WRAP)
namespace {
// Test-local, single-threaded interception.
struct AllocationState {
  colmap::FlatHashSet<void*> live_pitched;
  size_t successful_allocations = 0;
  size_t tracked_frees = 0;
  size_t scalar_width = 0;
  size_t scalar_height = 0;
  size_t refuse_ordinal = 0;
  size_t matching_calls = 0;
  size_t refusals = 0;
  bool tracking = false;
  bool duplicate_owner = false;
};
AllocationState allocations;

class AllocationScope {
 public:
  AllocationScope() {
    EXPECT_TRUE(allocations.live_pitched.empty());
    allocations = AllocationState{};
    allocations.tracking = true;
  }
  ~AllocationScope() {
    // The owner is declared after this guard, so even a fatal assertion or an
    // exception destroys it before these checks and resets the one-shot arm.
    allocations.refuse_ordinal = 0;
    EXPECT_FALSE(allocations.duplicate_owner);
    EXPECT_TRUE(allocations.live_pitched.empty());
    EXPECT_EQ(allocations.tracked_frees, allocations.successful_allocations);
    allocations = AllocationState{};
  }
  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;
};
}  // namespace

extern "C" cudaError_t __real_cudaMallocPitch(void**, size_t*, size_t, size_t);
extern "C" cudaError_t __real_cudaFree(void*);
extern "C" cudaError_t __wrap_cudaMallocPitch(void** pointer,
                                              size_t* pitch,
                                              size_t width,
                                              size_t height) {
  if (allocations.tracking && allocations.refuse_ordinal != 0 &&
      width == allocations.scalar_width &&
      height == allocations.scalar_height &&
      ++allocations.matching_calls == allocations.refuse_ordinal) {
    allocations.refuse_ordinal = 0;
    ++allocations.refusals;
    *pointer = nullptr;
    *pitch = 0;
    return cudaErrorMemoryAllocation;
  }
  const auto status = __real_cudaMallocPitch(pointer, pitch, width, height);
  if (allocations.tracking && status == cudaSuccess && *pointer != nullptr) {
    ++allocations.successful_allocations;
    if (!allocations.live_pitched.insert(*pointer).second) {
      allocations.duplicate_owner = true;
    }
  }
  return status;
}
extern "C" cudaError_t __wrap_cudaFree(void* pointer) {
  // Unknown pointers (including runtime-owned allocations) are still forwarded.
  const auto status = __real_cudaFree(pointer);
  if (allocations.tracking && status == cudaSuccess) {
    allocations.tracked_frees += allocations.live_pitched.erase(pointer);
  }
  return status;
}
#endif  // COLMAP_MVS_CUDA_ALLOCATION_WRAP

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

// Local fixture bounds; the supplied physical plane is world Z=4.
constexpr float kDepthMin = 1.0f;
constexpr float kDepthMax = 8.0f;
constexpr float kPlaneZ = 4.0f;

class PatchMatchCudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
      GTEST_SKIP() << GetCudaErrorMessage(status);
    }
    ASSERT_EQ(status, cudaSuccess) << GetCudaErrorMessage(status);
    if (count == 0) {
      GTEST_SKIP() << "No CUDA/HIP devices available";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};

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
                          const bool reject_all = false) {
  Inputs inputs = MakeInputs(sources, width, height);
  auto options = MakeOptions(geometry, filter, samples);
  if (reject_all) {
    options.filter_min_num_consistent = sources + 1;
  }
  const auto problem = MakeProblem(&inputs);
  PatchMatch matcher(options, problem);
  matcher.Run();
  CheckProductionResult(matcher, inputs, filter && !reject_all, reject_all);
}

TEST_F(PatchMatchCudaTest, ProductionGraphPreservesSourceOrder) {
  RunProductionProblem(17, 67, 39, 15, true, true);
}

TEST_F(PatchMatchCudaTest, FilterOffHasEmptyGraph) {
  RunProductionProblem(17, 67, 39, 15, true, false);
}

TEST_F(PatchMatchCudaTest, FinalFilterRejectsAllPixels) {
  RunProductionProblem(17, 67, 39, 15, true, true, true);
}

#if defined(COLMAP_MVS_CUDA_ALLOCATION_WRAP)
template <typename Call>
void ExpectInvalidArgument(Call&& call, const std::string& diagnostic) {
  try {
    call();
    ADD_FAILURE() << "Expected std::invalid_argument containing: "
                  << diagnostic;
  } catch (const std::invalid_argument& error) {
    // Source locations may vary; the full owner-specific diagnostic must match.
    EXPECT_NE(std::string(error.what()).find(diagnostic), std::string::npos)
        << error.what();
  } catch (const std::exception& error) {
    ADD_FAILURE() << "Expected std::invalid_argument, got: " << error.what();
  } catch (...) {
    ADD_FAILURE() << "Expected std::invalid_argument";
  }
}

void CheckRefusal(const size_t ordinal, const bool after_success) {
  ASSERT_TRUE(ordinal == 1 || ordinal == 2);
  auto inputs = MakeInputs(17, 67, 39);
  auto options = MakeOptions(true, false, 15);
  options.num_iterations = 1;
  const auto problem = MakeProblem(&inputs);
  AllocationScope scope;
  PatchMatch matcher(options, problem);
  if (after_success) {
    ASSERT_NO_THROW(matcher.Run());
    CheckProductionResult(matcher, inputs, false, false);
  }
  const size_t allocations_before = allocations.successful_allocations;

  // The constructor allocates unrotated source maps. Matches 1 and 2 at
  // the first rotated size are the previous map and the cost destination.
  allocations.scalar_width = 39 * sizeof(float);
  allocations.scalar_height = 67 * 17;
  allocations.refuse_ordinal = ordinal;
  ExpectInvalidArgument([&] { matcher.Run(); },
                        GetCudaErrorMessage(cudaErrorMemoryAllocation));
  allocations.refuse_ordinal = 0;
  ASSERT_EQ(allocations.refusals, 1u);
  ASSERT_EQ(allocations.matching_calls, ordinal);
  ASSERT_GT(allocations.successful_allocations, allocations_before);
  EXPECT_TRUE(allocations.live_pitched.empty());

  EXPECT_THROW(matcher.GetDepthMap(), std::invalid_argument);
  EXPECT_THROW(matcher.GetNormalMap(), std::invalid_argument);
  EXPECT_THROW(matcher.GetSelProbMap(), std::invalid_argument);
  EXPECT_THROW(matcher.GetConsistencyGraph(), std::invalid_argument);

  ASSERT_NO_THROW(matcher.Run());
  CheckProductionResult(matcher, inputs, false, false);
}
#endif  // COLMAP_MVS_CUDA_ALLOCATION_WRAP

TEST_F(PatchMatchCudaTest, PreviousReplacementAllocationRefusal) {
#if defined(COLMAP_MVS_CUDA_ALLOCATION_WRAP)
  CheckRefusal(1, /*after_success=*/false);
#else
  GTEST_SKIP()
      << "Allocation refusal requires CUDA, a static production owner, "
         "and a supported non-Windows/non-Apple wrap linker; "
         "HIP allocation refusal is not implemented";
#endif
}

TEST_F(PatchMatchCudaTest, CostDestinationAllocationRefusal) {
#if defined(COLMAP_MVS_CUDA_ALLOCATION_WRAP)
  CheckRefusal(2, /*after_success=*/true);
#else
  GTEST_SKIP()
      << "Allocation refusal requires CUDA, a static production owner, "
         "and a supported non-Windows/non-Apple wrap linker; "
         "HIP allocation refusal is not implemented";
#endif
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
