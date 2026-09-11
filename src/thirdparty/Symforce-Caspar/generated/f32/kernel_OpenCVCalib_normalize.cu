#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVCalib_normalize.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpenCVCalibNormalizeKernel(
    float *precond_diag, unsigned int precond_diag_num_alloc,
    float *precond_tril, unsigned int precond_tril_num_alloc, float *njtr,
    unsigned int njtr_num_alloc, const float *const diag, float *out_normalized,
    unsigned int out_normalized_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         24 * precond_tril_num_alloc,
                                         global_thread_idx, r1, r2, r3, r4);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         12 * precond_tril_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         8 * precond_tril_num_alloc,
                                         global_thread_idx, r8, r9, r10, r11);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         4 * precond_tril_num_alloc,
                                         global_thread_idx, r11, r12, r13, r14);
  };
  LoadUnique<1, float, float>(diag, 0, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float *)inout_shared, 0, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = 9.99999999999999955e-07;
    r15 = r13 * r15;
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         0 * precond_diag_num_alloc,
                                         global_thread_idx, r16, r17, r18, r19);
    r20 = 1.00000000000000000e+00;
    r20 = r13 + r20;
    r17 = fmaf(r17, r20, r15);
    r17 = 1.0 / r17;
    r13 = r14 * r17;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         0 * precond_tril_num_alloc,
                                         global_thread_idx, r21, r22, r23, r24);
    r21 = r22 * r11;
    r16 = fmaf(r16, r20, r15);
    r16 = 1.0 / r16;
    r21 = fmaf(r16, r21, r10 * r13);
    r21 = fmaf(r0, r21, r7);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         16 * precond_tril_num_alloc,
                                         global_thread_idx, r7, r25, r26, r27);
    r28 = r4 * r0;
    r28 = fmaf(r13, r28, r25);
    r25 = r21 * r28;
    r29 = r22 * r22;
    r29 = fmaf(r16, r29, r14 * r13);
    r29 = fmaf(r0, r29, r15);
    r29 = fmaf(r18, r20, r29);
    r29 = 1.0 / r29;
    r18 = r8 * r10;
    r14 = r23 * r11;
    r14 = fmaf(r16, r14, r17 * r18);
    r18 = r22 * r23;
    r18 = fmaf(r8, r13, r16 * r18);
    r18 = fmaf(r0, r18, r5);
    r5 = r18 * r21;
    r14 = fmaf(r29, r5, r14);
    r14 = fmaf(r0, r14, r27);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         20 * precond_tril_num_alloc,
                                         global_thread_idx, r27, r5, r30, r31);
    r32 = r18 * r28;
    r33 = r8 * r4;
    r33 = fmaf(r17, r33, r29 * r32);
    r33 = fmaf(r0, r33, r5);
    r5 = r14 * r33;
    r19 = fmaf(r19, r20, r15);
    r32 = r8 * r8;
    r34 = r23 * r23;
    r34 = fmaf(r16, r34, r17 * r32);
    r32 = r18 * r18;
    r34 = fmaf(r29, r32, r34);
    r19 = fmaf(r0, r34, r19);
    r19 = 1.0 / r19;
    r5 = fmaf(r19, r5, r29 * r25);
    r25 = r22 * r24;
    r25 = fmaf(r9, r13, r16 * r25);
    r25 = fmaf(r0, r25, r6);
    r6 = r25 * r18;
    r34 = r23 * r24;
    r34 = fmaf(r16, r34, r29 * r6);
    r6 = r8 * r9;
    r34 = fmaf(r17, r6, r34);
    r34 = fmaf(r0, r34, r26);
    r26 = r34 * r14;
    r6 = r25 * r21;
    r6 = fmaf(r29, r6, r19 * r26);
    r26 = r9 * r10;
    r6 = fmaf(r17, r26, r6);
    r32 = r24 * r11;
    r6 = fmaf(r16, r32, r6);
    r6 = fmaf(r0, r6, r30);
    r30 = r25 * r28;
    r32 = r34 * r33;
    r32 = fmaf(r19, r32, r29 * r30);
    r30 = r9 * r4;
    r32 = fmaf(r17, r30, r32);
    r32 = fmaf(r0, r32, r1);
    r1 = r6 * r32;
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         4 * precond_diag_num_alloc,
                                         global_thread_idx, r30, r26, r35, r36);
    r30 = fmaf(r30, r20, r15);
    r37 = r25 * r25;
    r38 = r34 * r34;
    r38 = fmaf(r19, r38, r29 * r37);
    r37 = r9 * r9;
    r39 = r24 * r24;
    r38 = fmaf(r17, r37, r38);
    r38 = fmaf(r16, r39, r38);
    r30 = fmaf(r0, r38, r30);
    r30 = 1.0 / r30;
    r5 = fmaf(r30, r1, r5);
    r38 = r10 * r4;
    r5 = fmaf(r17, r38, r5);
    r5 = fmaf(r0, r5, r3);
    r3 = r22 * r12;
    r38 = r0 * r16;
    r3 = fmaf(r38, r3, r7);
    r7 = r3 * r29;
    r1 = r23 * r12;
    r1 = fmaf(r16, r1, r18 * r7);
    r1 = fmaf(r0, r1, r27);
    r27 = r1 * r19;
    r39 = r24 * r12;
    r39 = fmaf(r16, r39, r34 * r27);
    r39 = fmaf(r25, r7, r39);
    r39 = fmaf(r0, r39, r31);
    r31 = r39 * r30;
    r37 = fmaf(r6, r31, r14 * r27);
    r40 = r11 * r12;
    r37 = fmaf(r16, r40, r37);
    r37 = fmaf(r21, r7, r37);
    r37 = fmaf(r0, r37, r2);
    r26 = fmaf(r26, r20, r15);
    r2 = r21 * r21;
    r40 = r14 * r14;
    r40 = fmaf(r19, r40, r29 * r2);
    r2 = r6 * r6;
    r41 = r11 * r11;
    r42 = r10 * r10;
    r40 = fmaf(r30, r2, r40);
    r40 = fmaf(r16, r41, r40);
    r40 = fmaf(r17, r42, r40);
    r26 = fmaf(r0, r40, r26);
    r26 = 1.0 / r26;
    r40 = r37 * r26;
    r42 = fmaf(r28, r7, r5 * r40);
    r42 = fmaf(r33, r27, r42);
    r42 = fmaf(r32, r31, r42);
    r35 = fmaf(r35, r20, r15);
    r39 = fmaf(r39, r31, r3 * r7);
    r3 = r12 * r12;
    r39 = fmaf(r1, r27, r39);
    r39 = fmaf(r16, r3, r39);
    r39 = fmaf(r37, r40, r39);
    r35 = fmaf(r0, r39, r35);
    r35 = 1.0 / r35;
    r39 = r42 * r35;
    ReadIdx4<1024, float, float, float4>(njtr, 4 * njtr_num_alloc,
                                         global_thread_idx, r37, r3, r1, r41);
    r2 = r0 * r32;
    r43 = r0 * r34;
    ReadIdx4<1024, float, float, float4>(njtr, 0 * njtr_num_alloc,
                                         global_thread_idx, r44, r45, r46, r47);
    r48 = r44 * r38;
    r47 = fmaf(r23, r48, r47);
    r49 = r8 * r45;
    r49 = r49 * r0;
    r47 = fmaf(r17, r49, r47);
    r50 = r0 * r18;
    r46 = fmaf(r22, r48, r46);
    r51 = r45 * r0;
    r46 = fmaf(r13, r51, r46);
    r50 = r50 * r46;
    r47 = fmaf(r29, r50, r47);
    r43 = r43 * r47;
    r43 = fmaf(r19, r43, r37);
    r37 = r9 * r45;
    r37 = r37 * r0;
    r43 = fmaf(r17, r37, r43);
    r50 = r0 * r25;
    r50 = r50 * r46;
    r43 = fmaf(r29, r50, r43);
    r43 = fmaf(r24, r48, r43);
    r2 = r2 * r43;
    r2 = fmaf(r30, r2, r41);
    r41 = r4 * r45;
    r41 = r41 * r0;
    r2 = fmaf(r17, r41, r2);
    r50 = r0 * r33;
    r50 = r50 * r47;
    r2 = fmaf(r19, r50, r2);
    r37 = r0 * r43;
    r37 = fmaf(r31, r37, r1);
    r40 = r0 * r40;
    r1 = r0 * r6;
    r1 = r1 * r43;
    r1 = fmaf(r30, r1, r3);
    r3 = r10 * r45;
    r3 = r3 * r0;
    r1 = fmaf(r17, r3, r1);
    r49 = r0 * r14;
    r49 = r49 * r47;
    r1 = fmaf(r19, r49, r1);
    r51 = r0 * r21;
    r51 = r51 * r46;
    r1 = fmaf(r29, r51, r1);
    r1 = fmaf(r11, r48, r1);
    r51 = r0 * r46;
    r37 = fmaf(r7, r51, r37);
    r49 = r0 * r47;
    r37 = fmaf(r27, r49, r37);
    r37 = fmaf(r1, r40, r37);
    r37 = fmaf(r12, r48, r37);
    r49 = r0 * r28;
    r49 = r49 * r46;
    r2 = fmaf(r29, r49, r2);
    r48 = r0 * r1;
    r48 = r48 * r5;
    r2 = fmaf(r26, r48, r2);
    r2 = fmaf(r37, r39, r2);
    r48 = r5 * r5;
    r49 = r33 * r33;
    r49 = fmaf(r19, r49, r26 * r48);
    r48 = r28 * r28;
    r50 = r4 * r4;
    r41 = r32 * r32;
    r49 = fmaf(r29, r48, r49);
    r49 = fmaf(r42, r39, r49);
    r49 = fmaf(r17, r50, r49);
    r49 = fmaf(r30, r41, r49);
    r49 = fmaf(r0, r49, r15);
    r49 = fmaf(r36, r20, r49);
    r49 = 1.0 / r49;
    r49 = r2 * r49;
    r35 = fmaf(r37, r35, r49 * r39);
    r37 = r0 * r35;
    r37 = fmaf(r47, r19, r27 * r37);
    r27 = r33 * r19;
    r39 = r0 * r49;
    r37 = fmaf(r39, r27, r37);
    r2 = r0 * r34;
    r20 = r32 * r30;
    r20 = fmaf(r43, r30, r39 * r20);
    r36 = r0 * r6;
    r15 = r5 * r26;
    r15 = fmaf(r39, r15, r35 * r40);
    r15 = fmaf(r1, r26, r15);
    r36 = r36 * r15;
    r20 = fmaf(r30, r36, r20);
    r40 = r0 * r35;
    r20 = fmaf(r31, r40, r20);
    r2 = r2 * r20;
    r37 = fmaf(r19, r2, r37);
    r40 = r0 * r14;
    r40 = r40 * r15;
    r37 = fmaf(r19, r40, r37);
    r40 = r0 * r18;
    r40 = r40 * r37;
    r2 = r0 * r35;
    r2 = fmaf(r7, r2, r29 * r40);
    r40 = r0 * r21;
    r40 = r40 * r15;
    r2 = fmaf(r29, r40, r2);
    r7 = r28 * r29;
    r2 = fmaf(r39, r7, r2);
    r27 = r0 * r25;
    r27 = r27 * r20;
    r2 = fmaf(r29, r27, r2);
    r2 = fmaf(r46, r29, r2);
    r27 = r12 * r35;
    r27 = fmaf(r38, r27, r44 * r16);
    r16 = r23 * r37;
    r27 = fmaf(r38, r16, r27);
    r44 = r24 * r20;
    r27 = fmaf(r38, r44, r27);
    r7 = r11 * r15;
    r27 = fmaf(r38, r7, r27);
    r40 = r22 * r2;
    r27 = fmaf(r38, r40, r27);
    r40 = r10 * r0;
    r40 = r40 * r15;
    r40 = fmaf(r17, r40, r45 * r17);
    r7 = r8 * r0;
    r7 = r7 * r37;
    r40 = fmaf(r17, r7, r40);
    r44 = r9 * r0;
    r44 = r44 * r20;
    r40 = fmaf(r17, r44, r40);
    r16 = r4 * r17;
    r40 = fmaf(r39, r16, r40);
    r39 = r0 * r2;
    r40 = fmaf(r13, r39, r40);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          0 * out_normalized_num_alloc,
                                          global_thread_idx, r27, r40, r2, r37);
    WriteIdx4<1024, float, float, float4>(
        out_normalized, 4 * out_normalized_num_alloc, global_thread_idx, r20,
        r15, r35, r49);
  };
}

void OpenCVCalibNormalize(
    float *precond_diag, unsigned int precond_diag_num_alloc,
    float *precond_tril, unsigned int precond_tril_num_alloc, float *njtr,
    unsigned int njtr_num_alloc, const float *const diag, float *out_normalized,
    unsigned int out_normalized_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVCalibNormalizeKernel<<<n_blocks, 1024>>>(
      precond_diag, precond_diag_num_alloc, precond_tril,
      precond_tril_num_alloc, njtr, njtr_num_alloc, diag, out_normalized,
      out_normalized_num_alloc, problem_size);
}

} // namespace caspar