#include "kernel_SimpleRadialCalib_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialCalib_normalize_kernel(float* precond_diag,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r0, r1, r2, r3);
    r4 = -1.00000000000000000e+00;
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           0 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r5,
                                           r6,
                                           r7,
                                           r8);
    read_idx_4<1024, float, float, float4>(precond_diag,
                                           0 * precond_diag_num_alloc,
                                           global_thread_idx,
                                           r8,
                                           r9,
                                           r10,
                                           r11);
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r13 = 1.00000000000000000e+00;
    r13 = r12 + r13;
    r14 = 9.99999999999999955e-07;
    r14 = r12 * r14;
    r8 = fmaf(r8, r13, r14);
    r12 = 1.0 / r8;
    r15 = r4 * r12;
    r16 = r5 * r15;
    r1 = fmaf(r0, r16, r1);
    r17 = r4 * r1;
    read_idx_2<1024, float, float, float2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r18, r19);
    r18 = fmaf(r7, r16, r18);
    r9 = fmaf(r9, r13, r14);
    r9 = fmaf(r5, r16, r9);
    r9 = 1.0 / r9;
    r17 = r17 * r18;
    r17 = fmaf(r9, r17, r3);
    r3 = r7 * r0;
    r17 = fmaf(r15, r3, r17);
    r20 = r6 * r0;
    r20 = fmaf(r15, r20, r2);
    r2 = r6 * r9;
    r21 = r5 * r2;
    r22 = r12 * r21;
    r20 = fmaf(r1, r22, r20);
    r23 = r4 * r20;
    r24 = r18 * r16;
    r25 = r6 * r7;
    r25 = fmaf(r12, r25, r2 * r24);
    r25 = fmaf(r4, r25, r19);
    r10 = fmaf(r10, r13, r14);
    r19 = r5 * r6;
    r8 = r8 * r8;
    r8 = 1.0 / r8;
    r19 = r19 * r8;
    r8 = r6 * r6;
    r8 = fmaf(r12, r8, r21 * r19);
    r10 = fmaf(r4, r8, r10);
    r10 = 1.0 / r10;
    r8 = r25 * r10;
    r17 = fmaf(r8, r23, r17);
    r13 = fmaf(r11, r13, r14);
    r11 = r18 * r18;
    r11 = fmaf(r9, r11, r25 * r8);
    r25 = r7 * r7;
    r11 = fmaf(r12, r25, r11);
    r13 = fmaf(r4, r11, r13);
    r13 = 1.0 / r13;
    r13 = r17 * r13;
    r17 = r4 * r13;
    r10 = fmaf(r20, r10, r8 * r17);
    r22 = fmaf(r10, r22, r1 * r9);
    r8 = r18 * r9;
    r22 = fmaf(r17, r8, r22);
    r12 = fmaf(r22, r16, r0 * r12);
    r8 = r6 * r10;
    r12 = fmaf(r15, r8, r12);
    r17 = r7 * r15;
    r12 = fmaf(r13, r17, r12);
    write_idx_4<1024, float, float, float4>(out_normalized,
                                            0 * out_normalized_num_alloc,
                                            global_thread_idx,
                                            r12,
                                            r22,
                                            r10,
                                            r13);
  };
}

void SimpleRadialCalib_normalize(float* precond_diag,
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
  SimpleRadialCalib_normalize_kernel<<<n_blocks, 1024>>>(
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