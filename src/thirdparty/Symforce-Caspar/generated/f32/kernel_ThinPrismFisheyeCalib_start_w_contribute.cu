#include "kernel_ThinPrismFisheyeCalib_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibStartWContributeKernel(
        float* ThinPrismFisheyeCalib_precond_diag,
        unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
        const float* const diag,
        float* ThinPrismFisheyeCalib_p,
        unsigned int ThinPrismFisheyeCalib_p_num_alloc,
        float* out_ThinPrismFisheyeCalib_w,
        unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_precond_diag,
        0 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r4;
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_p,
                                         0 * ThinPrismFisheyeCalib_p_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7,
                                         r8);
    r0 = r0 * r5;
    r1 = r1 * r4;
    r1 = r1 * r6;
    r2 = r2 * r4;
    r2 = r2 * r7;
    r3 = r3 * r4;
    r3 = r3 * r8;
    AddIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_w,
        0 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_precond_diag,
        4 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r3,
        r2,
        r1,
        r0);
    r3 = r3 * r4;
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_p,
                                         4 * ThinPrismFisheyeCalib_p_num_alloc,
                                         global_thread_idx,
                                         r8,
                                         r7,
                                         r6,
                                         r5);
    r3 = r3 * r8;
    r2 = r2 * r4;
    r2 = r2 * r7;
    r1 = r1 * r4;
    r1 = r1 * r6;
    r0 = r0 * r4;
    r0 = r0 * r5;
    AddIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_w,
        4 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r3,
        r2,
        r1,
        r0);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_precond_diag,
        8 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    r0 = r0 * r4;
    ReadIdx4<1024, float, float, float4>(ThinPrismFisheyeCalib_p,
                                         8 * ThinPrismFisheyeCalib_p_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7,
                                         r8);
    r0 = r0 * r5;
    r1 = r1 * r4;
    r1 = r1 * r6;
    r2 = r2 * r4;
    r2 = r2 * r7;
    r4 = r3 * r4;
    r4 = r4 * r8;
    AddIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_w,
        8 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r4);
  };
}

void ThinPrismFisheyeCalibStartWContribute(
    float* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyeCalib_p,
    unsigned int ThinPrismFisheyeCalib_p_num_alloc,
    float* out_ThinPrismFisheyeCalib_w,
    unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibStartWContributeKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_precond_diag,
      ThinPrismFisheyeCalib_precond_diag_num_alloc,
      diag,
      ThinPrismFisheyeCalib_p,
      ThinPrismFisheyeCalib_p_num_alloc,
      out_ThinPrismFisheyeCalib_w,
      out_ThinPrismFisheyeCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar