#include "kernel_PinholePose_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_alpha_numerator_denominator_kernel(
        float* PinholePose_p_kp1,
        unsigned int PinholePose_p_kp1_num_alloc,
        float* PinholePose_r_k,
        unsigned int PinholePose_r_k_num_alloc,
        float* PinholePose_w,
        unsigned int PinholePose_w_num_alloc,
        float* const PinholePose_total_ag,
        float* const PinholePose_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float PinholePose_total_ag_local[1];

  __shared__ float PinholePose_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(PinholePose_p_kp1,
                                           4 * PinholePose_p_kp1_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1);
    read_idx_2<1024, float, float, float2>(PinholePose_r_k,
                                           4 * PinholePose_r_k_num_alloc,
                                           global_thread_idx,
                                           r2,
                                           r3);
    read_idx_4<1024, float, float, float4>(PinholePose_p_kp1,
                                           0 * PinholePose_p_kp1_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
    read_idx_4<1024, float, float, float4>(PinholePose_r_k,
                                           0 * PinholePose_r_k_num_alloc,
                                           global_thread_idx,
                                           r8,
                                           r9,
                                           r10,
                                           r11);
    r9 = fmaf(r5, r9, r1 * r3);
    r9 = fmaf(r4, r8, r9);
    r9 = fmaf(r7, r11, r9);
    r9 = fmaf(r6, r10, r9);
    r9 = fmaf(r0, r2, r9);
  };
  sum_store<float>(PinholePose_total_ag_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r9);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(PinholePose_w,
                                           0 * PinholePose_w_num_alloc,
                                           global_thread_idx,
                                           r9,
                                           r2,
                                           r10,
                                           r11);
    read_idx_2<1024, float, float, float2>(
        PinholePose_w, 4 * PinholePose_w_num_alloc, global_thread_idx, r8, r3);
    r8 = fmaf(r0, r8, r7 * r11);
    r8 = fmaf(r6, r10, r8);
    r8 = fmaf(r1, r3, r8);
    r8 = fmaf(r4, r9, r8);
    r8 = fmaf(r5, r2, r8);
  };
  sum_store<float>(PinholePose_total_ac_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r8);
  sum_flush_final<float>(PinholePose_total_ag_local, PinholePose_total_ag, 1);
  sum_flush_final<float>(PinholePose_total_ac_local, PinholePose_total_ac, 1);
}

void PinholePose_alpha_numerator_denominator(
    float* PinholePose_p_kp1,
    unsigned int PinholePose_p_kp1_num_alloc,
    float* PinholePose_r_k,
    unsigned int PinholePose_r_k_num_alloc,
    float* PinholePose_w,
    unsigned int PinholePose_w_num_alloc,
    float* const PinholePose_total_ag,
    float* const PinholePose_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      PinholePose_p_kp1,
      PinholePose_p_kp1_num_alloc,
      PinholePose_r_k,
      PinholePose_r_k_num_alloc,
      PinholePose_w,
      PinholePose_w_num_alloc,
      PinholePose_total_ag,
      PinholePose_total_ac,
      problem_size);
}

}  // namespace caspar