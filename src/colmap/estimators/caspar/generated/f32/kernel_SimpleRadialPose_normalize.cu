#include "kernel_SimpleRadialPose_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialPose_normalize_kernel(float* precond_diag,
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = 9.99999999999999955e-07;
    r2 = r1 * r2;
    read_idx_2<1024, float, float, float2>(
        precond_diag, 4 * precond_diag_num_alloc, global_thread_idx, r3, r4);
    r5 = 1.00000000000000000e+00;
    r5 = r1 + r5;
    r3 = fmaf(r3, r5, r2);
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           4 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r1,
                                           r6,
                                           r7,
                                           r8);
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           0 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r9,
                                           r10,
                                           r11,
                                           r12);
    read_idx_4<1024, float, float, float4>(precond_diag,
                                           0 * precond_diag_num_alloc,
                                           global_thread_idx,
                                           r13,
                                           r14,
                                           r15,
                                           r16);
    r13 = fmaf(r13, r5, r2);
    r13 = 1.0 / r13;
    r17 = r0 * r13;
    r18 = r9 * r17;
    r8 = fmaf(r12, r18, r8);
    r14 = fmaf(r14, r5, r2);
    r14 = fmaf(r9, r18, r14);
    r14 = 1.0 / r14;
    r9 = r8 * r14;
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           8 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r19,
                                           r20,
                                           r21,
                                           r22);
    r6 = fmaf(r10, r18, r6);
    r23 = r10 * r12;
    r23 = fmaf(r13, r23, r6 * r9);
    r23 = fmaf(r0, r23, r21);
    r15 = fmaf(r15, r5, r2);
    r21 = r6 * r6;
    r24 = r10 * r10;
    r24 = fmaf(r13, r24, r14 * r21);
    r15 = fmaf(r0, r24, r15);
    r15 = 1.0 / r15;
    r24 = r23 * r15;
    r23 = fmaf(r23, r24, r8 * r9);
    read_idx_3<1024, float, float, float4>(precond_tril,
                                           12 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r8,
                                           r21,
                                           r25);
    r26 = r10 * r11;
    r7 = fmaf(r11, r18, r7);
    r27 = r6 * r7;
    r27 = fmaf(r14, r27, r13 * r26);
    r27 = fmaf(r0, r27, r20);
    r20 = r11 * r12;
    r20 = fmaf(r13, r20, r27 * r24);
    r20 = fmaf(r7, r9, r20);
    r20 = fmaf(r0, r20, r8);
    r16 = fmaf(r16, r5, r2);
    r8 = r27 * r27;
    r26 = r7 * r7;
    r26 = fmaf(r14, r26, r15 * r8);
    r8 = r11 * r11;
    r26 = fmaf(r13, r8, r26);
    r16 = fmaf(r0, r26, r16);
    r16 = 1.0 / r16;
    r26 = r20 * r16;
    r8 = r12 * r12;
    r23 = fmaf(r20, r26, r23);
    r23 = fmaf(r13, r8, r23);
    r3 = fmaf(r0, r23, r3);
    r3 = 1.0 / r3;
    read_idx_2<1024, float, float, float2>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r23, r8);
    read_idx_4<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r20, r28, r29, r30);
    r31 = r0 * r6;
    r28 = fmaf(r20, r18, r28);
    r31 = r31 * r28;
    r31 = fmaf(r14, r31, r29);
    r29 = r10 * r20;
    r31 = fmaf(r17, r29, r31);
    r29 = r0 * r31;
    r29 = fmaf(r24, r29, r23);
    r23 = r0 * r7;
    r23 = r23 * r28;
    r23 = fmaf(r14, r23, r30);
    r30 = r11 * r20;
    r23 = fmaf(r17, r30, r23);
    r32 = r0 * r27;
    r32 = r32 * r31;
    r23 = fmaf(r15, r32, r23);
    r32 = r0 * r23;
    r29 = fmaf(r26, r32, r29);
    r30 = r12 * r20;
    r29 = fmaf(r17, r30, r29);
    r33 = r0 * r28;
    r29 = fmaf(r9, r33, r29);
    r19 = fmaf(r1, r18, r19);
    r33 = r6 * r19;
    r30 = r10 * r1;
    r30 = fmaf(r13, r30, r14 * r33);
    r30 = fmaf(r0, r30, r22);
    r22 = fmaf(r19, r9, r30 * r24);
    r33 = r12 * r1;
    r22 = fmaf(r13, r33, r22);
    r32 = r27 * r30;
    r34 = r11 * r1;
    r34 = fmaf(r13, r34, r15 * r32);
    r32 = r7 * r19;
    r34 = fmaf(r14, r32, r34);
    r34 = fmaf(r0, r34, r21);
    r22 = fmaf(r34, r26, r22);
    r22 = fmaf(r0, r22, r25);
    r25 = r22 * r3;
    r5 = fmaf(r4, r5, r2);
    r4 = r34 * r34;
    r22 = fmaf(r22, r25, r16 * r4);
    r4 = r30 * r30;
    r2 = r19 * r19;
    r33 = r1 * r1;
    r22 = fmaf(r15, r4, r22);
    r22 = fmaf(r14, r2, r22);
    r22 = fmaf(r13, r33, r22);
    r5 = fmaf(r0, r22, r5);
    r5 = 1.0 / r5;
    r22 = r0 * r29;
    r22 = fmaf(r25, r22, r8);
    r8 = r0 * r31;
    r8 = r8 * r30;
    r22 = fmaf(r15, r8, r22);
    r33 = r0 * r23;
    r33 = r33 * r34;
    r22 = fmaf(r16, r33, r22);
    r2 = r0 * r28;
    r2 = r2 * r19;
    r22 = fmaf(r14, r2, r22);
    r4 = r1 * r20;
    r22 = fmaf(r17, r4, r22);
    r22 = r5 * r22;
    r5 = r0 * r22;
    r25 = fmaf(r5, r25, r29 * r3);
    r3 = r0 * r25;
    r3 = fmaf(r23, r16, r26 * r3);
    r26 = r34 * r16;
    r3 = fmaf(r5, r26, r3);
    r26 = r0 * r27;
    r26 = r26 * r3;
    r26 = fmaf(r31, r15, r15 * r26);
    r4 = r0 * r25;
    r26 = fmaf(r24, r4, r26);
    r24 = r30 * r15;
    r26 = fmaf(r5, r24, r26);
    r24 = r0 * r25;
    r24 = fmaf(r28, r14, r9 * r24);
    r9 = r0 * r6;
    r9 = r9 * r26;
    r24 = fmaf(r14, r9, r24);
    r4 = r19 * r14;
    r24 = fmaf(r5, r4, r24);
    r5 = r0 * r7;
    r5 = r5 * r3;
    r24 = fmaf(r14, r5, r24);
    r18 = fmaf(r24, r18, r20 * r13);
    r13 = r11 * r3;
    r18 = fmaf(r17, r13, r18);
    r5 = r10 * r26;
    r18 = fmaf(r17, r5, r18);
    r4 = r12 * r25;
    r18 = fmaf(r17, r4, r18);
    r9 = r1 * r17;
    r18 = fmaf(r22, r9, r18);
    write_idx_4<1024, float, float, float4>(out_normalized,
                                            0 * out_normalized_num_alloc,
                                            global_thread_idx,
                                            r18,
                                            r24,
                                            r26,
                                            r3);
    write_idx_2<1024, float, float, float2>(out_normalized,
                                            4 * out_normalized_num_alloc,
                                            global_thread_idx,
                                            r25,
                                            r22);
  };
}

void SimpleRadialPose_normalize(float* precond_diag,
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
  SimpleRadialPose_normalize_kernel<<<n_blocks, 1024>>>(
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