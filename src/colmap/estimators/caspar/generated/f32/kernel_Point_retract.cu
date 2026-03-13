#include "kernel_Point_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Point_retract_kernel(float* Point,
                         unsigned int Point_num_alloc,
                         float* delta,
                         unsigned int delta_num_alloc,
                         float* out_Point_retracted,
                         unsigned int out_Point_retracted_num_alloc,
                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        Point, 0 * Point_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_3<1024, float, float, float4>(
        delta, 0 * delta_num_alloc, global_thread_idx, r3, r4, r5);
    r3 = r0 + r3;
    r4 = r1 + r4;
    r5 = r2 + r5;
    write_idx_3<1024, float, float, float4>(out_Point_retracted,
                                            0 * out_Point_retracted_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r4,
                                            r5);
  };
}

void Point_retract(float* Point,
                   unsigned int Point_num_alloc,
                   float* delta,
                   unsigned int delta_num_alloc,
                   float* out_Point_retracted,
                   unsigned int out_Point_retracted_num_alloc,
                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_retract_kernel<<<n_blocks, 1024>>>(Point,
                                           Point_num_alloc,
                                           delta,
                                           delta_num_alloc,
                                           out_Point_retracted,
                                           out_Point_retracted_num_alloc,
                                           problem_size);
}

}  // namespace caspar