#include "kernel_ThinPrismFisheyeCalib_update_step.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibUpdateStepKernel(
        float* ThinPrismFisheyeCalib_step_k,
        unsigned int ThinPrismFisheyeCalib_step_k_num_alloc,
        float* ThinPrismFisheyeCalib_p_kp1,
        unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
        const float* const alpha,
        float* out_ThinPrismFisheyeCalib_step_kp1,
        unsigned int out_ThinPrismFisheyeCalib_step_kp1_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_step_k,
        0 * ThinPrismFisheyeCalib_step_k_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        0 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
  };
  LoadUnique<1, float, float>(alpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r4, r8, r0);
    r5 = fmaf(r5, r8, r1);
    r6 = fmaf(r6, r8, r2);
    r7 = fmaf(r7, r8, r3);
    WriteIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_step_kp1,
        0 * out_ThinPrismFisheyeCalib_step_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_step_k,
        4 * ThinPrismFisheyeCalib_step_k_num_alloc,
        global_thread_idx,
        r7,
        r6,
        r5,
        r4);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        4 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r3,
        r2,
        r1,
        r0);
    r3 = fmaf(r3, r8, r7);
    r2 = fmaf(r2, r8, r6);
    r1 = fmaf(r1, r8, r5);
    r0 = fmaf(r0, r8, r4);
    WriteIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_step_kp1,
        4 * out_ThinPrismFisheyeCalib_step_kp1_num_alloc,
        global_thread_idx,
        r3,
        r2,
        r1,
        r0);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_step_k,
        8 * ThinPrismFisheyeCalib_step_k_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_p_kp1,
        8 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    r4 = fmaf(r4, r8, r0);
    r5 = fmaf(r5, r8, r1);
    r6 = fmaf(r6, r8, r2);
    r8 = fmaf(r7, r8, r3);
    WriteIdx4<1024, float, float, float4>(
        out_ThinPrismFisheyeCalib_step_kp1,
        8 * out_ThinPrismFisheyeCalib_step_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r8);
  };
}

void ThinPrismFisheyeCalibUpdateStep(
    float* ThinPrismFisheyeCalib_step_k,
    unsigned int ThinPrismFisheyeCalib_step_k_num_alloc,
    float* ThinPrismFisheyeCalib_p_kp1,
    unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
    const float* const alpha,
    float* out_ThinPrismFisheyeCalib_step_kp1,
    unsigned int out_ThinPrismFisheyeCalib_step_kp1_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibUpdateStepKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_step_k,
      ThinPrismFisheyeCalib_step_k_num_alloc,
      ThinPrismFisheyeCalib_p_kp1,
      ThinPrismFisheyeCalib_p_kp1_num_alloc,
      alpha,
      out_ThinPrismFisheyeCalib_step_kp1,
      out_ThinPrismFisheyeCalib_step_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar