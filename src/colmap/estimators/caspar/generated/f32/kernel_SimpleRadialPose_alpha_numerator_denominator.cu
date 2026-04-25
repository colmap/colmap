#include "kernel_SimpleRadialPose_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialPose_alpha_numerator_denominator_kernel(
        float* SimpleRadialPose_p_kp1,
        unsigned int SimpleRadialPose_p_kp1_num_alloc,
        float* SimpleRadialPose_r_k,
        unsigned int SimpleRadialPose_r_k_num_alloc,
        float* SimpleRadialPose_w,
        unsigned int SimpleRadialPose_w_num_alloc,
        float* const SimpleRadialPose_total_ag,
        float* const SimpleRadialPose_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float SimpleRadialPose_total_ag_local[1];

  __shared__ float SimpleRadialPose_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(SimpleRadialPose_p_kp1,
                                           0 * SimpleRadialPose_p_kp1_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2,
                                           r3);
    read_idx_4<1024, float, float, float4>(SimpleRadialPose_r_k,
                                           0 * SimpleRadialPose_r_k_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
    r6 = fmaf(r2, r6, r3 * r7);
    read_idx_2<1024, float, float, float2>(SimpleRadialPose_p_kp1,
                                           4 * SimpleRadialPose_p_kp1_num_alloc,
                                           global_thread_idx,
                                           r7,
                                           r8);
    read_idx_2<1024, float, float, float2>(SimpleRadialPose_r_k,
                                           4 * SimpleRadialPose_r_k_num_alloc,
                                           global_thread_idx,
                                           r9,
                                           r10);
    r6 = fmaf(r0, r4, r6);
    r6 = fmaf(r7, r9, r6);
    r6 = fmaf(r1, r5, r6);
    r6 = fmaf(r8, r10, r6);
  };
  sum_store<float>(SimpleRadialPose_total_ag_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r6);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(SimpleRadialPose_w,
                                           4 * SimpleRadialPose_w_num_alloc,
                                           global_thread_idx,
                                           r6,
                                           r10);
    read_idx_4<1024, float, float, float4>(SimpleRadialPose_w,
                                           0 * SimpleRadialPose_w_num_alloc,
                                           global_thread_idx,
                                           r5,
                                           r9,
                                           r4,
                                           r11);
    r5 = fmaf(r0, r5, r8 * r10);
    r5 = fmaf(r1, r9, r5);
    r5 = fmaf(r3, r11, r5);
    r5 = fmaf(r7, r6, r5);
    r5 = fmaf(r2, r4, r5);
  };
  sum_store<float>(SimpleRadialPose_total_ac_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  sum_flush_final<float>(
      SimpleRadialPose_total_ag_local, SimpleRadialPose_total_ag, 1);
  sum_flush_final<float>(
      SimpleRadialPose_total_ac_local, SimpleRadialPose_total_ac, 1);
}

void SimpleRadialPose_alpha_numerator_denominator(
    float* SimpleRadialPose_p_kp1,
    unsigned int SimpleRadialPose_p_kp1_num_alloc,
    float* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    float* SimpleRadialPose_w,
    unsigned int SimpleRadialPose_w_num_alloc,
    float* const SimpleRadialPose_total_ag,
    float* const SimpleRadialPose_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPose_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      SimpleRadialPose_p_kp1,
      SimpleRadialPose_p_kp1_num_alloc,
      SimpleRadialPose_r_k,
      SimpleRadialPose_r_k_num_alloc,
      SimpleRadialPose_w,
      SimpleRadialPose_w_num_alloc,
      SimpleRadialPose_total_ag,
      SimpleRadialPose_total_ac,
      problem_size);
}

}  // namespace caspar