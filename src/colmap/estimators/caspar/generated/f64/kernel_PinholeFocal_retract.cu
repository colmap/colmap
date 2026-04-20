#include "kernel_PinholeFocal_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeFocal_retract_kernel(
    double* PinholeFocal,
    unsigned int PinholeFocal_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_PinholeFocal_retracted,
    unsigned int out_PinholeFocal_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholeFocal, 0 * PinholeFocal_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, double, double, double2>(
        delta, 0 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    write_idx_2<1024, double, double, double2>(
        out_PinholeFocal_retracted,
        0 * out_PinholeFocal_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
}

void PinholeFocal_retract(double* PinholeFocal,
                          unsigned int PinholeFocal_num_alloc,
                          double* delta,
                          unsigned int delta_num_alloc,
                          double* out_PinholeFocal_retracted,
                          unsigned int out_PinholeFocal_retracted_num_alloc,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFocal_retract_kernel<<<n_blocks, 1024>>>(
      PinholeFocal,
      PinholeFocal_num_alloc,
      delta,
      delta_num_alloc,
      out_PinholeFocal_retracted,
      out_PinholeFocal_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar