#include "kernel_SimpleRadialCalib_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialCalib_alpha_numerator_denominator_kernel(
        float* SimpleRadialCalib_p_kp1,
        unsigned int SimpleRadialCalib_p_kp1_num_alloc,
        float* SimpleRadialCalib_r_k,
        unsigned int SimpleRadialCalib_r_k_num_alloc,
        float* SimpleRadialCalib_w,
        unsigned int SimpleRadialCalib_w_num_alloc,
        float* const SimpleRadialCalib_total_ag,
        float* const SimpleRadialCalib_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float SimpleRadialCalib_total_ag_local[1];

  __shared__ float SimpleRadialCalib_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        SimpleRadialCalib_p_kp1,
        0 * SimpleRadialCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    read_idx_4<1024, float, float, float4>(SimpleRadialCalib_r_k,
                                           0 * SimpleRadialCalib_r_k_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
    r5 = fmaf(r1, r5, r2 * r6);
    r5 = fmaf(r3, r7, r5);
    r5 = fmaf(r0, r4, r5);
  };
  sum_store<float>(SimpleRadialCalib_total_ag_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(SimpleRadialCalib_w,
                                           0 * SimpleRadialCalib_w_num_alloc,
                                           global_thread_idx,
                                           r5,
                                           r4,
                                           r7,
                                           r6);
    r7 = fmaf(r2, r7, r1 * r4);
    r7 = fmaf(r0, r5, r7);
    r7 = fmaf(r3, r6, r7);
  };
  sum_store<float>(SimpleRadialCalib_total_ac_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r7);
  sum_flush_final<float>(
      SimpleRadialCalib_total_ag_local, SimpleRadialCalib_total_ag, 1);
  sum_flush_final<float>(
      SimpleRadialCalib_total_ac_local, SimpleRadialCalib_total_ac, 1);
}

void SimpleRadialCalib_alpha_numerator_denominator(
    float* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    float* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    float* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    float* const SimpleRadialCalib_total_ag,
    float* const SimpleRadialCalib_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialCalib_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      SimpleRadialCalib_p_kp1,
      SimpleRadialCalib_p_kp1_num_alloc,
      SimpleRadialCalib_r_k,
      SimpleRadialCalib_r_k_num_alloc,
      SimpleRadialCalib_w,
      SimpleRadialCalib_w_num_alloc,
      SimpleRadialCalib_total_ag,
      SimpleRadialCalib_total_ac,
      problem_size);
}

}  // namespace caspar