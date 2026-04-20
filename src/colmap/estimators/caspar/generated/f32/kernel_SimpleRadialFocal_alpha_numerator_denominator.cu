#include "kernel_SimpleRadialFocal_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocal_alpha_numerator_denominator_kernel(
        float* SimpleRadialFocal_p_kp1,
        unsigned int SimpleRadialFocal_p_kp1_num_alloc,
        float* SimpleRadialFocal_r_k,
        unsigned int SimpleRadialFocal_r_k_num_alloc,
        float* SimpleRadialFocal_w,
        unsigned int SimpleRadialFocal_w_num_alloc,
        float* const SimpleRadialFocal_total_ag,
        float* const SimpleRadialFocal_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float SimpleRadialFocal_total_ag_local[1];

  __shared__ float SimpleRadialFocal_total_ac_local[1];

  float r0, r1;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, float, float, float>(SimpleRadialFocal_p_kp1,
                                          0 * SimpleRadialFocal_p_kp1_num_alloc,
                                          global_thread_idx,
                                          r0);
    read_idx_1<1024, float, float, float>(SimpleRadialFocal_r_k,
                                          0 * SimpleRadialFocal_r_k_num_alloc,
                                          global_thread_idx,
                                          r1);
    r1 = r0 * r1;
  };
  sum_store<float>(SimpleRadialFocal_total_ag_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  if (global_thread_idx < problem_size) {
    read_idx_1<1024, float, float, float>(SimpleRadialFocal_w,
                                          0 * SimpleRadialFocal_w_num_alloc,
                                          global_thread_idx,
                                          r1);
    r1 = r0 * r1;
  };
  sum_store<float>(SimpleRadialFocal_total_ac_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  sum_flush_final<float>(
      SimpleRadialFocal_total_ag_local, SimpleRadialFocal_total_ag, 1);
  sum_flush_final<float>(
      SimpleRadialFocal_total_ac_local, SimpleRadialFocal_total_ac, 1);
}

void SimpleRadialFocal_alpha_numerator_denominator(
    float* SimpleRadialFocal_p_kp1,
    unsigned int SimpleRadialFocal_p_kp1_num_alloc,
    float* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    float* SimpleRadialFocal_w,
    unsigned int SimpleRadialFocal_w_num_alloc,
    float* const SimpleRadialFocal_total_ag,
    float* const SimpleRadialFocal_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocal_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      SimpleRadialFocal_p_kp1,
      SimpleRadialFocal_p_kp1_num_alloc,
      SimpleRadialFocal_r_k,
      SimpleRadialFocal_r_k_num_alloc,
      SimpleRadialFocal_w,
      SimpleRadialFocal_w_num_alloc,
      SimpleRadialFocal_total_ag,
      SimpleRadialFocal_total_ac,
      problem_size);
}

}  // namespace caspar