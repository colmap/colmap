#include "kernel_Point_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Point_alpha_denumerator_or_beta_nummerator_kernel(
        float* Point_p_kp1,
        unsigned int Point_p_kp1_num_alloc,
        float* Point_w,
        unsigned int Point_w_num_alloc,
        float* const Point_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float Point_out_local[1];

  float r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        Point_p_kp1, 0 * Point_p_kp1_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_3<1024, float, float, float4>(
        Point_w, 0 * Point_w_num_alloc, global_thread_idx, r3, r4, r5);
    r3 = fmaf(r0, r3, r1 * r4);
    r3 = fmaf(r2, r5, r3);
  };
  sum_store<float>(Point_out_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  sum_flush_final<float>(Point_out_local, Point_out, 1);
}

void Point_alpha_denumerator_or_beta_nummerator(
    float* Point_p_kp1,
    unsigned int Point_p_kp1_num_alloc,
    float* Point_w,
    unsigned int Point_w_num_alloc,
    float* const Point_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_alpha_denumerator_or_beta_nummerator_kernel<<<n_blocks, 1024>>>(
      Point_p_kp1,
      Point_p_kp1_num_alloc,
      Point_w,
      Point_w_num_alloc,
      Point_out,
      problem_size);
}

}  // namespace caspar