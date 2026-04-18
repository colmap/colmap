#include "kernel_SimpleRadialExtraCalib_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_normalize_kernel(
        float* precond_diag,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r0, r1, r2);
    r3 = -1.00000000000000000e+00;
    read_idx_3<1024, float, float, float4>(precond_tril,
                                           0 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6);
    read_idx_3<1024, float, float, float4>(precond_diag,
                                           0 * precond_diag_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r7,
                                           r8);
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 1.00000000000000000e+00;
    r10 = r9 + r10;
    r11 = 9.99999999999999955e-07;
    r11 = r9 * r11;
    r4 = fmaf(r4, r10, r11);
    r4 = 1.0 / r4;
    r9 = r5 * r4;
    r12 = r3 * r9;
    r2 = fmaf(r0, r12, r2);
    r13 = r1 * r3;
    r7 = fmaf(r7, r10, r11);
    r7 = 1.0 / r7;
    r14 = r6 * r7;
    r2 = fmaf(r14, r13, r2);
    r9 = fmaf(r5, r9, r6 * r14);
    r9 = fmaf(r3, r9, r11);
    r9 = fmaf(r8, r10, r9);
    r9 = 1.0 / r9;
    r9 = r2 * r9;
    r12 = fmaf(r9, r12, r0 * r4);
    r4 = r3 * r14;
    r4 = fmaf(r9, r4, r1 * r7);
    write_idx_3<1024, float, float, float4>(out_normalized,
                                            0 * out_normalized_num_alloc,
                                            global_thread_idx,
                                            r12,
                                            r4,
                                            r9);
  };
}

void SimpleRadialExtraCalib_normalize(float* precond_diag,
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
  SimpleRadialExtraCalib_normalize_kernel<<<n_blocks, 1024>>>(
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