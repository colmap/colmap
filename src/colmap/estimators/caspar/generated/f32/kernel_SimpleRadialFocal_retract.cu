#include "kernel_SimpleRadialFocal_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialFocal_retract_kernel(
    float* SimpleRadialFocal,
    unsigned int SimpleRadialFocal_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SimpleRadialFocal_retracted,
    unsigned int out_SimpleRadialFocal_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, float, float, float>(SimpleRadialFocal,
                                          0 * SimpleRadialFocal_num_alloc,
                                          global_thread_idx,
                                          r0);
    read_idx_1<1024, float, float, float>(
        delta, 0 * delta_num_alloc, global_thread_idx, r1);
    r1 = r0 + r1;
    write_idx_1<1024, float, float, float>(
        out_SimpleRadialFocal_retracted,
        0 * out_SimpleRadialFocal_retracted_num_alloc,
        global_thread_idx,
        r1);
  };
}

void SimpleRadialFocal_retract(
    float* SimpleRadialFocal,
    unsigned int SimpleRadialFocal_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SimpleRadialFocal_retracted,
    unsigned int out_SimpleRadialFocal_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocal_retract_kernel<<<n_blocks, 1024>>>(
      SimpleRadialFocal,
      SimpleRadialFocal_num_alloc,
      delta,
      delta_num_alloc,
      out_SimpleRadialFocal_retracted,
      out_SimpleRadialFocal_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar