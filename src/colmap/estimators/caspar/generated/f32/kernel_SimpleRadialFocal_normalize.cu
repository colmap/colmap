#include "kernel_SimpleRadialFocal_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocal_normalize_kernel(float* precond_diag,
                                       unsigned int precond_diag_num_alloc,
                                       float* precond_tril,
                                       unsigned int precond_tril_num_alloc,
                                       float* njtr,
                                       unsigned int njtr_num_alloc,
                                       const float* const diag,
                                       float* out_normalized,
                                       unsigned int out_normalized_num_alloc,
                                       size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, float, float, float>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r0);
    read_idx_1<1024, float, float, float>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r1);
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = 1.00000000000000000e+00;
    r3 = r2 + r3;
    r4 = 9.99999999999999955e-07;
    r4 = fmaf(r2, r4, r1 * r3);
    r4 = 1.0 / r4;
    r4 = r0 * r4;
    write_idx_1<1024, float, float, float>(
        out_normalized, 0 * out_normalized_num_alloc, global_thread_idx, r4);
  };
}

void SimpleRadialFocal_normalize(float* precond_diag,
                                 unsigned int precond_diag_num_alloc,
                                 float* precond_tril,
                                 unsigned int precond_tril_num_alloc,
                                 float* njtr,
                                 unsigned int njtr_num_alloc,
                                 const float* const diag,
                                 float* out_normalized,
                                 unsigned int out_normalized_num_alloc,
                                 size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocal_normalize_kernel<<<n_blocks, 1024>>>(
      precond_diag,
      precond_diag_num_alloc,
      precond_tril,
      precond_tril_num_alloc,
      njtr,
      njtr_num_alloc,
      diag,
      out_normalized,
      out_normalized_num_alloc,
      problem_size);
}

}  // namespace caspar