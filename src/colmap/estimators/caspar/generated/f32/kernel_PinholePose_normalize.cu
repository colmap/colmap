#include "kernel_PinholePose_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_normalize_kernel(float* precond_diag,
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
      r31, r32, r33;
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
    r5 = fmaf(r5, r6, r1);
    r0 = -1.00000000000000000e+00;
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           8 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r7,
                                           r8,
                                           r9,
                                           r10);
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           0 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r11,
                                           r12,
                                           r13,
                                           r14);
    r15 = r12 * r13;
    r2 = fmaf(r2, r6, r1);
    r2 = 1.0 / r2;
    read_idx_4<1024, float, float, float4>(precond_tril,
                                           4 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r16,
                                           r17,
                                           r18,
                                           r19);
    r20 = r0 * r2;
    r21 = r11 * r20;
    r18 = fmaf(r13, r21, r18);
    r3 = fmaf(r3, r6, r1);
    r3 = fmaf(r11, r21, r3);
    r3 = 1.0 / r3;
    r11 = r18 * r3;
    r17 = fmaf(r12, r21, r17);
    r15 = fmaf(r17, r11, r2 * r15);
    r15 = fmaf(r0, r15, r8);
    r4 = fmaf(r4, r6, r1);
    r8 = r17 * r17;
    r22 = r12 * r12;
    r22 = fmaf(r2, r22, r3 * r8);
    r4 = fmaf(r0, r22, r4);
    r4 = 1.0 / r4;
    r22 = r15 * r4;
    r18 = fmaf(r18, r11, r15 * r22);
    r15 = r13 * r13;
    r18 = fmaf(r2, r15, r18);
    r5 = fmaf(r0, r18, r5);
    r5 = 1.0 / r5;
    read_idx_4<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r18, r15, r8, r23);
    r15 = fmaf(r18, r21, r15);
    r24 = r0 * r15;
    r24 = fmaf(r11, r24, r23);
    r23 = r13 * r18;
    r24 = fmaf(r20, r23, r24);
    r25 = r0 * r17;
    r25 = r25 * r15;
    r25 = fmaf(r3, r25, r8);
    r8 = r12 * r18;
    r25 = fmaf(r20, r8, r25);
    r8 = r0 * r25;
    r24 = fmaf(r22, r8, r24);
    r19 = fmaf(r14, r21, r19);
    r8 = r17 * r19;
    r23 = r12 * r14;
    r23 = fmaf(r2, r23, r3 * r8);
    r23 = fmaf(r0, r23, r9);
    r9 = r13 * r14;
    r9 = fmaf(r2, r9, r23 * r22);
    r9 = fmaf(r19, r11, r9);
    read_idx_3<1024, float, float, float4>(precond_tril,
                                           12 * precond_tril_num_alloc,
                                           global_thread_idx,
                                           r8,
                                           r26,
                                           r27);
    r7 = fmaf(r16, r21, r7);
    r8 = r17 * r7;
    r28 = r12 * r16;
    r28 = fmaf(r2, r28, r3 * r8);
    r28 = fmaf(r0, r28, r10);
    r10 = r28 * r23;
    r8 = r0 * r9;
    r29 = r13 * r16;
    r29 = fmaf(r2, r29, r28 * r22);
    r29 = fmaf(r7, r11, r29);
    r29 = fmaf(r0, r29, r26);
    r26 = r29 * r5;
    r8 = fmaf(r26, r8, r4 * r10);
    r10 = r7 * r19;
    r8 = fmaf(r3, r10, r8);
    r30 = r14 * r16;
    r8 = fmaf(r2, r30, r8);
    r8 = fmaf(r0, r8, r27);
    read_idx_2<1024, float, float, float2>(
        precond_diag, 4 * precond_diag_num_alloc, global_thread_idx, r27, r30);
    r27 = fmaf(r27, r6, r1);
    r10 = r9 * r9;
    r31 = r19 * r19;
    r31 = fmaf(r3, r31, r5 * r10);
    r10 = r23 * r23;
    r32 = r14 * r14;
    r31 = fmaf(r4, r10, r31);
    r31 = fmaf(r2, r32, r31);
    r27 = fmaf(r0, r31, r27);
    r27 = 1.0 / r27;
    r31 = r8 * r27;
    r6 = fmaf(r30, r6, r1);
    r8 = fmaf(r8, r31, r29 * r26);
    r29 = r28 * r28;
    r30 = r7 * r7;
    r1 = r16 * r16;
    r8 = fmaf(r4, r29, r8);
    r8 = fmaf(r3, r30, r8);
    r8 = fmaf(r2, r1, r8);
    r6 = fmaf(r0, r8, r6);
    r6 = 1.0 / r6;
    read_idx_2<1024, float, float, float2>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r8, r1);
    r30 = r0 * r28;
    r30 = r30 * r25;
    r30 = fmaf(r4, r30, r1);
    r1 = r0 * r24;
    r30 = fmaf(r26, r1, r30);
    r29 = r0 * r7;
    r29 = r29 * r15;
    r30 = fmaf(r3, r29, r30);
    r32 = r0 * r23;
    r32 = r32 * r25;
    r32 = fmaf(r4, r32, r8);
    r8 = r9 * r24;
    r32 = fmaf(r5, r8, r32);
    r10 = r14 * r18;
    r32 = fmaf(r20, r10, r32);
    r33 = r0 * r19;
    r33 = r33 * r15;
    r32 = fmaf(r3, r33, r32);
    r33 = r0 * r32;
    r30 = fmaf(r31, r33, r30);
    r10 = r16 * r18;
    r30 = fmaf(r20, r10, r30);
    r30 = r6 * r30;
    r6 = r0 * r30;
    r27 = fmaf(r32, r27, r6 * r31);
    r31 = r9 * r27;
    r31 = fmaf(r5, r31, r24 * r5);
    r31 = fmaf(r26, r6, r31);
    r26 = r0 * r31;
    r5 = r0 * r23;
    r5 = r5 * r27;
    r5 = fmaf(r4, r5, r22 * r26);
    r26 = r28 * r4;
    r5 = fmaf(r6, r26, r5);
    r5 = fmaf(r25, r4, r5);
    r26 = r0 * r19;
    r26 = r26 * r27;
    r22 = r7 * r3;
    r22 = fmaf(r6, r22, r3 * r26);
    r26 = r0 * r17;
    r26 = r26 * r5;
    r22 = fmaf(r3, r26, r22);
    r6 = r0 * r31;
    r22 = fmaf(r11, r6, r22);
    r22 = fmaf(r15, r3, r22);
    r6 = r14 * r27;
    r6 = fmaf(r20, r6, r18 * r2);
    r2 = r12 * r5;
    r6 = fmaf(r20, r2, r6);
    r26 = r16 * r20;
    r6 = fmaf(r30, r26, r6);
    r11 = r13 * r31;
    r6 = fmaf(r20, r11, r6);
    r6 = fmaf(r22, r21, r6);
    write_idx_4<1024, float, float, float4>(out_normalized,
                                            0 * out_normalized_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r22,
                                            r5,
                                            r31);
    write_idx_2<1024, float, float, float2>(out_normalized,
                                            4 * out_normalized_num_alloc,
                                            global_thread_idx,
                                            r27,
                                            r30);
  };
}

void PinholePose_normalize(float* precond_diag,
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
  PinholePose_normalize_kernel<<<n_blocks, 1024>>>(precond_diag,
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