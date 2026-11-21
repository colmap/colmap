// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/gpu_mat.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"

#include <cstring>
#include <exception>
#include <string>

#include <gtest/gtest.h>

namespace {

struct TextureCalls {
  cudaError_t alloc_error = cudaSuccess;
  cudaError_t create_error = cudaSuccess;
  cudaError_t destroy_error = cudaSuccess;
  cudaError_t free_error = cudaSuccess;
  int free_count = 0;
  bool freed_expected_array = false;
  std::string order;
};

TextureCalls calls;
int array_token;

cudaArray_t TestArray() { return reinterpret_cast<cudaArray_t>(&array_token); }

cudaError_t TestMalloc3DArray(cudaArray_t* array,
                              const cudaChannelFormatDesc*,
                              cudaExtent,
                              unsigned int) {
  calls.order += 'A';
  if (calls.alloc_error == cudaSuccess) {
    *array = TestArray();
  }
  return calls.alloc_error;
}

cudaError_t TestCreateTextureObject(cudaTextureObject_t* texture,
                                    const cudaResourceDesc*,
                                    const cudaTextureDesc*,
                                    const void*) {
  calls.order += 'C';
  if (calls.create_error == cudaSuccess) {
    *texture = static_cast<cudaTextureObject_t>(1);
  }
  return calls.create_error;
}

cudaError_t TestDestroyTextureObject(cudaTextureObject_t) {
  calls.order += 'D';
  return calls.destroy_error;
}

cudaError_t TestFreeArray(cudaArray_t array) {
  calls.order += 'F';
  ++calls.free_count;
  calls.freed_expected_array = (array == TestArray());
  return calls.free_error;
}

}  // namespace

// Isolate the template and route only its four ownership calls. In HIP builds,
// save and restore the CUDA-to-HIP aliases defined by cuda_to_hip.h.
#pragma push_macro("CudaArrayLayeredTexture")
#pragma push_macro("cudaMalloc3DArray")
#pragma push_macro("cudaCreateTextureObject")
#pragma push_macro("cudaDestroyTextureObject")
#pragma push_macro("cudaFreeArray")
#undef CudaArrayLayeredTexture
#undef cudaMalloc3DArray
#undef cudaCreateTextureObject
#undef cudaDestroyTextureObject
#undef cudaFreeArray
#define CudaArrayLayeredTexture CudaArrayLayeredTextureTestOnly
#define cudaMalloc3DArray TestMalloc3DArray
#define cudaCreateTextureObject TestCreateTextureObject
#define cudaDestroyTextureObject TestDestroyTextureObject
#define cudaFreeArray TestFreeArray
#include "colmap/mvs/cuda_texture.h"
#pragma pop_macro("cudaFreeArray")
#pragma pop_macro("cudaDestroyTextureObject")
#pragma pop_macro("cudaCreateTextureObject")
#pragma pop_macro("cudaMalloc3DArray")
#pragma pop_macro("CudaArrayLayeredTexture")

namespace colmap {
namespace mvs {
namespace {

using Texture = CudaArrayLayeredTextureTestOnly<float>;

class CudaTextureConstructionTest : public testing::Test {
 protected:
  void SetUp() override { calls = TextureCalls{}; }

  void ExpectCreateError() {
    try {
      Texture texture(cudaTextureDesc{}, 1, 1, 1);
      FAIL() << "Expected texture creation to throw";
    } catch (const std::exception& error) {
      const std::string code =
          "(error " + std::to_string(static_cast<int>(cudaErrorNoDevice)) + ")";
      EXPECT_NE(std::string(error.what()).find(code), std::string::npos)
          << error.what();
    }
  }
};

TEST_F(CudaTextureConstructionTest, CreateFailureFreesArray) {
  calls.create_error = cudaErrorNoDevice;
  ExpectCreateError();
  EXPECT_EQ(calls.free_count, 1);
  EXPECT_TRUE(calls.freed_expected_array);
  EXPECT_EQ(calls.order, "ACF");
}

TEST_F(CudaTextureConstructionTest, FreeFailurePreservesCreateError) {
  calls.create_error = cudaErrorNoDevice;
  calls.free_error = cudaErrorInsufficientDriver;
  ExpectCreateError();
  EXPECT_EQ(calls.free_count, 1);
  EXPECT_TRUE(calls.freed_expected_array);
  EXPECT_EQ(calls.order, "ACF");
}

TEST_F(CudaTextureConstructionTest, AllocationFailureDoesNotFree) {
  calls.alloc_error = cudaErrorNoDevice;
  EXPECT_THROW((Texture(cudaTextureDesc{}, 1, 1, 1)), std::exception);
  EXPECT_EQ(calls.free_count, 0);
  EXPECT_EQ(calls.order, "A");
}

TEST_F(CudaTextureConstructionTest, SuccessDestroysTextureBeforeFreeingArray) {
  {
    Texture texture(cudaTextureDesc{}, 1, 1, 1);
    EXPECT_EQ(calls.free_count, 0);
    EXPECT_EQ(calls.order, "AC");
  }
  EXPECT_EQ(calls.free_count, 1);
  EXPECT_TRUE(calls.freed_expected_array);
  EXPECT_EQ(calls.order, "ACDF");
}

TEST_F(CudaTextureConstructionTest, CleanupFailuresStillAttemptBothReleases) {
  for (const cudaError_t destroy_error : {cudaSuccess, cudaErrorNoDevice}) {
    for (const cudaError_t free_error :
         {cudaSuccess, cudaErrorInsufficientDriver}) {
      SCOPED_TRACE(testing::Message()
                   << "destroy=" << destroy_error << ", free=" << free_error);
      calls = TextureCalls{};
      calls.destroy_error = destroy_error;
      calls.free_error = free_error;
      {
        Texture texture(cudaTextureDesc{}, 1, 1, 1);
      }
      EXPECT_EQ(calls.free_count, 1);
      EXPECT_TRUE(calls.freed_expected_array);
      EXPECT_EQ(calls.order, "ACDF");
    }
  }
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
