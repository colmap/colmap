#include "kernel_PinholeFocal_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFocal_alpha_numerator_denominator_kernel(
        float* PinholeFocal_p_kp1,
        unsigned int PinholeFocal_p_kp1_num_alloc,
        float* PinholeFocal_r_k,
        unsigned int PinholeFocal_r_k_num_alloc,
        float* PinholeFocal_w,
        unsigned int PinholeFocal_w_num_alloc,
        float* const PinholeFocal_total_ag,
        float* const PinholeFocal_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float PinholeFocal_total_ag_local[1];

  __shared__ float PinholeFocal_total_ac_local[1];

  float r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(PinholeFocal_p_kp1,
                                           0 * PinholeFocal_p_kp1_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1);
    read_idx_2<1024, float, float, float2>(PinholeFocal_r_k,
                                           0 * PinholeFocal_r_k_num_alloc,
                                           global_thread_idx,
                                           r2,
                                           r3);
    r2 = fmaf(r0, r2, r1 * r3);
  };
  sum_store<float>(PinholeFocal_total_ag_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r2);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(PinholeFocal_w,
                                           0 * PinholeFocal_w_num_alloc,
                                           global_thread_idx,
                                           r2,
                                           r3);
    r3 = fmaf(r1, r3, r0 * r2);
  };
  sum_store<float>(PinholeFocal_total_ac_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  sum_flush_final<float>(PinholeFocal_total_ag_local, PinholeFocal_total_ag, 1);
  sum_flush_final<float>(PinholeFocal_total_ac_local, PinholeFocal_total_ac, 1);
}

void PinholeFocal_alpha_numerator_denominator(
    float* PinholeFocal_p_kp1,
    unsigned int PinholeFocal_p_kp1_num_alloc,
    float* PinholeFocal_r_k,
    unsigned int PinholeFocal_r_k_num_alloc,
    float* PinholeFocal_w,
    unsigned int PinholeFocal_w_num_alloc,
    float* const PinholeFocal_total_ag,
    float* const PinholeFocal_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFocal_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      PinholeFocal_p_kp1,
      PinholeFocal_p_kp1_num_alloc,
      PinholeFocal_r_k,
      PinholeFocal_r_k_num_alloc,
      PinholeFocal_w,
      PinholeFocal_w_num_alloc,
      PinholeFocal_total_ag,
      PinholeFocal_total_ac,
      problem_size);
}

}  // namespace caspar