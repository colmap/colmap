#include "kernel_Point_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Point_start_w_contribute_kernel(float* Point_precond_diag,
                                    unsigned int Point_precond_diag_num_alloc,
                                    const float* const diag,
                                    float* Point_p,
                                    unsigned int Point_p_num_alloc,
                                    float* out_Point_w,
                                    unsigned int out_Point_w_num_alloc,
                                    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(Point_precond_diag,
                                           0 * Point_precond_diag_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2);
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r3;
    read_idx_3<1024, float, float, float4>(
        Point_p, 0 * Point_p_num_alloc, global_thread_idx, r4, r5, r6);
    r0 = r0 * r4;
    r1 = r1 * r3;
    r1 = r1 * r5;
    r3 = r2 * r3;
    r3 = r3 * r6;
    add_idx_3<1024, float, float, float4>(
        out_Point_w, 0 * out_Point_w_num_alloc, global_thread_idx, r0, r1, r3);
  };
}

void Point_start_w_contribute(float* Point_precond_diag,
                              unsigned int Point_precond_diag_num_alloc,
                              const float* const diag,
                              float* Point_p,
                              unsigned int Point_p_num_alloc,
                              float* out_Point_w,
                              unsigned int out_Point_w_num_alloc,
                              size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_start_w_contribute_kernel<<<n_blocks, 1024>>>(
      Point_precond_diag,
      Point_precond_diag_num_alloc,
      diag,
      Point_p,
      Point_p_num_alloc,
      out_Point_w,
      out_Point_w_num_alloc,
      problem_size);
}

}  // namespace caspar