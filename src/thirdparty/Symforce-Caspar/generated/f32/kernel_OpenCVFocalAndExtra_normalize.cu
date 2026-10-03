#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVFocalAndExtra_normalize.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpenCVFocalAndExtraNormalizeKernel(
    float *precond_diag, unsigned int precond_diag_num_alloc,
    float *precond_tril, unsigned int precond_tril_num_alloc, float *njtr,
    unsigned int njtr_num_alloc, const float *const diag, float *out_normalized,
    unsigned int out_normalized_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32;
  LoadUnique<1, float, float>(diag, 0, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float *)inout_shared, 0, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = 9.99999999999999955e-07;
    r1 = r0 * r1;
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         0 * precond_diag_num_alloc,
                                         global_thread_idx, r2, r3, r4, r5);
    r6 = 1.00000000000000000e+00;
    r6 = r0 + r6;
    r5 = fmaf(r5, r6, r1);
    r0 = -1.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         8 * precond_tril_num_alloc,
                                         global_thread_idx, r7, r8, r9, r10);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         0 * precond_tril_num_alloc,
                                         global_thread_idx, r11, r12, r13, r14);
    r11 = r12 * r13;
    r2 = fmaf(r2, r6, r1);
    r2 = 1.0 / r2;
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         4 * precond_tril_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    r3 = fmaf(r3, r6, r1);
    r3 = 1.0 / r3;
    r19 = r17 * r3;
    r11 = fmaf(r16, r19, r2 * r11);
    r11 = fmaf(r0, r11, r8);
    r8 = r16 * r16;
    r20 = r12 * r12;
    r20 = fmaf(r2, r20, r3 * r8);
    r20 = fmaf(r0, r20, r1);
    r20 = fmaf(r4, r6, r20);
    r20 = 1.0 / r20;
    r4 = r11 * r20;
    r17 = fmaf(r17, r19, r11 * r4);
    r11 = r13 * r13;
    r17 = fmaf(r2, r11, r17);
    r5 = fmaf(r0, r17, r5);
    r5 = 1.0 / r5;
    ReadIdx4<1024, float, float, float4>(njtr, 0 * njtr_num_alloc,
                                         global_thread_idx, r17, r11, r8, r21);
    r22 = r0 * r2;
    r23 = r17 * r22;
    r21 = fmaf(r13, r23, r21);
    r24 = r11 * r0;
    r21 = fmaf(r19, r24, r21);
    r8 = fmaf(r12, r23, r8);
    r25 = r16 * r11;
    r25 = r25 * r0;
    r8 = fmaf(r3, r25, r8);
    r25 = r0 * r8;
    r21 = fmaf(r4, r25, r21);
    ReadIdx3<1024, float, float, float4>(precond_tril,
                                         12 * precond_tril_num_alloc,
                                         global_thread_idx, r25, r24, r26);
    r27 = r12 * r15;
    r28 = r16 * r7;
    r28 = fmaf(r3, r28, r2 * r27);
    r28 = fmaf(r0, r28, r10);
    r10 = r13 * r15;
    r10 = fmaf(r2, r10, r28 * r4);
    r10 = fmaf(r7, r19, r10);
    r10 = fmaf(r0, r10, r24);
    r24 = r10 * r5;
    ReadIdx2<1024, float, float, float2>(
        precond_diag, 4 * precond_diag_num_alloc, global_thread_idx, r27, r29);
    r29 = fmaf(r29, r6, r1);
    r30 = r28 * r28;
    r10 = fmaf(r10, r24, r20 * r30);
    r30 = r14 * r15;
    r31 = r18 * r7;
    r31 = fmaf(r3, r31, r2 * r30);
    r30 = r12 * r14;
    r32 = r16 * r18;
    r32 = fmaf(r3, r32, r2 * r30);
    r32 = fmaf(r0, r32, r9);
    r9 = r13 * r14;
    r9 = fmaf(r2, r9, r32 * r4);
    r9 = fmaf(r18, r19, r9);
    r9 = fmaf(r0, r9, r25);
    r25 = r28 * r32;
    r31 = fmaf(r20, r25, r31);
    r31 = fmaf(r9, r24, r31);
    r31 = fmaf(r0, r31, r26);
    r6 = fmaf(r27, r6, r1);
    r27 = r32 * r32;
    r1 = r18 * r18;
    r1 = fmaf(r3, r1, r20 * r27);
    r27 = r14 * r14;
    r26 = r9 * r9;
    r1 = fmaf(r2, r27, r1);
    r1 = fmaf(r5, r26, r1);
    r6 = fmaf(r0, r1, r6);
    r6 = 1.0 / r6;
    r1 = r31 * r6;
    r26 = r7 * r7;
    r27 = r15 * r15;
    r10 = fmaf(r31, r1, r10);
    r10 = fmaf(r3, r26, r10);
    r10 = fmaf(r2, r27, r10);
    r29 = fmaf(r0, r10, r29);
    r29 = 1.0 / r29;
    ReadIdx2<1024, float, float, float2>(njtr, 4 * njtr_num_alloc,
                                         global_thread_idx, r10, r27);
    r26 = r0 * r8;
    r26 = r26 * r32;
    r26 = fmaf(r20, r26, r10);
    r10 = r18 * r11;
    r10 = r10 * r0;
    r26 = fmaf(r3, r10, r26);
    r31 = r0 * r21;
    r31 = r31 * r9;
    r26 = fmaf(r5, r31, r26);
    r26 = fmaf(r14, r23, r26);
    r31 = r0 * r26;
    r31 = fmaf(r1, r31, r27);
    r27 = r0 * r8;
    r27 = r27 * r28;
    r31 = fmaf(r20, r27, r31);
    r10 = r7 * r11;
    r10 = r10 * r0;
    r31 = fmaf(r3, r10, r31);
    r25 = r0 * r21;
    r31 = fmaf(r24, r25, r31);
    r31 = fmaf(r15, r23, r31);
    r31 = r29 * r31;
    r29 = r0 * r31;
    r24 = fmaf(r24, r29, r21 * r5);
    r25 = r0 * r9;
    r1 = fmaf(r29, r1, r26 * r6);
    r25 = r25 * r1;
    r24 = fmaf(r5, r25, r24);
    r25 = r0 * r24;
    r5 = r0 * r32;
    r5 = r5 * r1;
    r5 = fmaf(r20, r5, r4 * r25);
    r25 = r28 * r20;
    r5 = fmaf(r29, r25, r5);
    r5 = fmaf(r8, r20, r5);
    r25 = r12 * r5;
    r25 = fmaf(r22, r25, r17 * r2);
    r2 = r15 * r22;
    r25 = fmaf(r31, r2, r25);
    r17 = r13 * r24;
    r25 = fmaf(r22, r17, r25);
    r4 = r14 * r1;
    r25 = fmaf(r22, r4, r25);
    r4 = r16 * r0;
    r4 = r4 * r5;
    r4 = fmaf(r3, r4, r11 * r3);
    r17 = r0 * r24;
    r4 = fmaf(r19, r17, r4);
    r19 = r7 * r3;
    r4 = fmaf(r29, r19, r4);
    r29 = r18 * r0;
    r29 = r29 * r1;
    r4 = fmaf(r3, r29, r4);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          0 * out_normalized_num_alloc,
                                          global_thread_idx, r25, r4, r5, r24);
    WriteIdx2<1024, float, float, float2>(out_normalized,
                                          4 * out_normalized_num_alloc,
                                          global_thread_idx, r1, r31);
  };
}

void OpenCVFocalAndExtraNormalize(
    float *precond_diag, unsigned int precond_diag_num_alloc,
    float *precond_tril, unsigned int precond_tril_num_alloc, float *njtr,
    unsigned int njtr_num_alloc, const float *const diag, float *out_normalized,
    unsigned int out_normalized_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVFocalAndExtraNormalizeKernel<<<n_blocks, 1024>>>(
      precond_diag, precond_diag_num_alloc, precond_tril,
      precond_tril_num_alloc, njtr, njtr_num_alloc, diag, out_normalized,
      out_normalized_num_alloc, problem_size);
}

} // namespace caspar