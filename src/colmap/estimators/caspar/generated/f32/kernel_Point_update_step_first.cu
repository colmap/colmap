#include "kernel_Point_update_step_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Point_update_step_first_kernel(float* Point_p_kp1,
                                   unsigned int Point_p_kp1_num_alloc,
                                   const float* const alpha,
                                   float* out_Point_step_kp1,
                                   unsigned int out_Point_step_kp1_num_alloc,
                                   size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        Point_p_kp1, 0 * Point_p_kp1_num_alloc, global_thread_idx, r0, r1, r2);
  };
  load_unique<1, float, float>(alpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r3;
    r1 = r1 * r3;
    r3 = r2 * r3;
    write_idx_3<1024, float, float, float4>(out_Point_step_kp1,
                                            0 * out_Point_step_kp1_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1,
                                            r3);
  };
}

void Point_update_step_first(float* Point_p_kp1,
                             unsigned int Point_p_kp1_num_alloc,
                             const float* const alpha,
                             float* out_Point_step_kp1,
                             unsigned int out_Point_step_kp1_num_alloc,
                             size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_update_step_first_kernel<<<n_blocks, 1024>>>(
      Point_p_kp1,
      Point_p_kp1_num_alloc,
      alpha,
      out_Point_step_kp1,
      out_Point_step_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar