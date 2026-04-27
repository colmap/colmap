#include "kernel_Point_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Point_alpha_numerator_denominator_kernel(float* Point_p_kp1,
                                             unsigned int Point_p_kp1_num_alloc,
                                             float* Point_r_k,
                                             unsigned int Point_r_k_num_alloc,
                                             float* Point_w,
                                             unsigned int Point_w_num_alloc,
                                             float* const Point_total_ag,
                                             float* const Point_total_ac,
                                             size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float Point_total_ag_local[1];

  __shared__ float Point_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        Point_p_kp1, 0 * Point_p_kp1_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_3<1024, float, float, float4>(
        Point_r_k, 0 * Point_r_k_num_alloc, global_thread_idx, r3, r4, r5);
    r4 = fmaf(r1, r4, r2 * r5);
    r4 = fmaf(r0, r3, r4);
  };
  sum_store<float>(Point_total_ag_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        Point_w, 0 * Point_w_num_alloc, global_thread_idx, r4, r3, r5);
    r4 = fmaf(r0, r4, r1 * r3);
    r4 = fmaf(r2, r5, r4);
  };
  sum_store<float>(Point_total_ac_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  sum_flush_final<float>(Point_total_ag_local, Point_total_ag, 1);
  sum_flush_final<float>(Point_total_ac_local, Point_total_ac, 1);
}

void Point_alpha_numerator_denominator(float* Point_p_kp1,
                                       unsigned int Point_p_kp1_num_alloc,
                                       float* Point_r_k,
                                       unsigned int Point_r_k_num_alloc,
                                       float* Point_w,
                                       unsigned int Point_w_num_alloc,
                                       float* const Point_total_ag,
                                       float* const Point_total_ac,
                                       size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      Point_p_kp1,
      Point_p_kp1_num_alloc,
      Point_r_k,
      Point_r_k_num_alloc,
      Point_w,
      Point_w_num_alloc,
      Point_total_ag,
      Point_total_ac,
      problem_size);
}

}  // namespace caspar