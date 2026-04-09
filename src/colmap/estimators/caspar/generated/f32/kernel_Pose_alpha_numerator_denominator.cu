#include "kernel_Pose_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Pose_alpha_numerator_denominator_kernel(float* Pose_p_kp1,
                                            unsigned int Pose_p_kp1_num_alloc,
                                            float* Pose_r_k,
                                            unsigned int Pose_r_k_num_alloc,
                                            float* Pose_w,
                                            unsigned int Pose_w_num_alloc,
                                            float* const Pose_total_ag,
                                            float* const Pose_total_ac,
                                            size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float Pose_total_ag_local[1];

  __shared__ float Pose_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        Pose_p_kp1, 4 * Pose_p_kp1_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, float, float, float2>(
        Pose_r_k, 4 * Pose_r_k_num_alloc, global_thread_idx, r2, r3);
    read_idx_4<1024, float, float, float4>(Pose_p_kp1,
                                           0 * Pose_p_kp1_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
    read_idx_4<1024, float, float, float4>(
        Pose_r_k, 0 * Pose_r_k_num_alloc, global_thread_idx, r8, r9, r10, r11);
    r10 = fmaf(r6, r10, r1 * r3);
    r10 = fmaf(r0, r2, r10);
    r10 = fmaf(r7, r11, r10);
    r10 = fmaf(r5, r9, r10);
    r10 = fmaf(r4, r8, r10);
  };
  sum_store<float>(Pose_total_ag_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r10);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        Pose_w, 0 * Pose_w_num_alloc, global_thread_idx, r10, r8, r9, r11);
    r9 = fmaf(r6, r9, r4 * r10);
    read_idx_2<1024, float, float, float2>(
        Pose_w, 4 * Pose_w_num_alloc, global_thread_idx, r6, r10);
    r9 = fmaf(r5, r8, r9);
    r9 = fmaf(r1, r10, r9);
    r9 = fmaf(r0, r6, r9);
    r9 = fmaf(r7, r11, r9);
  };
  sum_store<float>(Pose_total_ac_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r9);
  sum_flush_final<float>(Pose_total_ag_local, Pose_total_ag, 1);
  sum_flush_final<float>(Pose_total_ac_local, Pose_total_ac, 1);
}

void Pose_alpha_numerator_denominator(float* Pose_p_kp1,
                                      unsigned int Pose_p_kp1_num_alloc,
                                      float* Pose_r_k,
                                      unsigned int Pose_r_k_num_alloc,
                                      float* Pose_w,
                                      unsigned int Pose_w_num_alloc,
                                      float* const Pose_total_ag,
                                      float* const Pose_total_ac,
                                      size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Pose_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      Pose_p_kp1,
      Pose_p_kp1_num_alloc,
      Pose_r_k,
      Pose_r_k_num_alloc,
      Pose_w,
      Pose_w_num_alloc,
      Pose_total_ag,
      Pose_total_ac,
      problem_size);
}

}  // namespace caspar