#include "kernel_PinholeFocal_start_w.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeFocal_start_w_kernel(
    float* PinholeFocal_precond_diag,
    unsigned int PinholeFocal_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeFocal_p,
    unsigned int PinholeFocal_p_num_alloc,
    float* out_PinholeFocal_w,
    unsigned int out_PinholeFocal_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        PinholeFocal_precond_diag,
        0 * PinholeFocal_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1);
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r2;
    read_idx_2<1024, float, float, float2>(PinholeFocal_p,
                                           0 * PinholeFocal_p_num_alloc,
                                           global_thread_idx,
                                           r3,
                                           r4);
    r0 = r0 * r3;
    r2 = r1 * r2;
    r2 = r2 * r4;
    write_idx_2<1024, float, float, float2>(out_PinholeFocal_w,
                                            0 * out_PinholeFocal_w_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r2);
  };
}

void PinholeFocal_start_w(float* PinholeFocal_precond_diag,
                          unsigned int PinholeFocal_precond_diag_num_alloc,
                          const float* const diag,
                          float* PinholeFocal_p,
                          unsigned int PinholeFocal_p_num_alloc,
                          float* out_PinholeFocal_w,
                          unsigned int out_PinholeFocal_w_num_alloc,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFocal_start_w_kernel<<<n_blocks, 1024>>>(
      PinholeFocal_precond_diag,
      PinholeFocal_precond_diag_num_alloc,
      diag,
      PinholeFocal_p,
      PinholeFocal_p_num_alloc,
      out_PinholeFocal_w,
      out_PinholeFocal_w_num_alloc,
      problem_size);
}

}  // namespace caspar