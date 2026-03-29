#include "kernel_PinholeCalib_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeCalib_normalize_kernel(float* precond_diag,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = 9.99999999999999955e-07;
    r1 = r0 * r1;
    read_idx_4<1024, float, float, float4>(precond_diag,
                                           0 * precond_diag_num_alloc,
                                           global_thread_idx,
                                           r2,
                                           r3,
                                           r4,
                                           r5);
    r6 = 1.00000000000000000e+00;
    r6 = r0 + r6;
    r4 = fmaf(r4, r6, r1);
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           0 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r7,
                                           r8,
                                           r9);
    r9 = -1.00000000000000000e+00;
    r8 = r7 * r9;
    r2 = fmaf(r2, r6, r1);
    r2 = 1.0 / r2;
    r8 = r8 * r2;
    r4 = fmaf(r7, r8, r4);
    r4 = 1.0 / r4;
    read_idx_4<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r7, r0, r10, r11);
    r10 = fmaf(r7, r8, r10);
    r10 = r4 * r10;
    r5 = fmaf(r5, r6, r1);
    read_idx_2<1024, float, float, float2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r4, r12);
    r9 = r4 * r9;
    r6 = fmaf(r3, r6, r1);
    r6 = 1.0 / r6;
    r9 = r9 * r6;
    r5 = fmaf(r4, r9, r5);
    r5 = 1.0 / r5;
    r11 = fmaf(r0, r9, r11);
    r11 = r5 * r11;
    r8 = fmaf(r8, r10, r7 * r2);
    r9 = fmaf(r9, r11, r0 * r6);
    write_idx_4<1024, float, float, float4>(out_normalized,
                                            0 * out_normalized_num_alloc,
                                            global_thread_idx,
                                            r8,
                                            r9,
                                            r10,
                                            r11);
  };
}

void PinholeCalib_normalize(float* precond_diag,
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
  PinholeCalib_normalize_kernel<<<n_blocks, 1024>>>(precond_diag,
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