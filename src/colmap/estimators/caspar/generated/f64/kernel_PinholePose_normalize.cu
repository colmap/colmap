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
    PinholePose_normalize_kernel(double* precond_diag,
                                 unsigned int precond_diag_num_alloc,
                                 double* precond_tril,
                                 unsigned int precond_tril_num_alloc,
                                 double* njtr,
                                 unsigned int njtr_num_alloc,
                                 const double* const diag,
                                 double* out_normalized,
                                 unsigned int out_normalized_num_alloc,
                                 size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
    read_idx_2<1024, double, double, double2>(
        precond_tril, 6 * precond_tril_num_alloc, global_thread_idx, r1, r2);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 2 * precond_tril_num_alloc, global_thread_idx, r3, r4);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 0 * precond_tril_num_alloc, global_thread_idx, r5, r6);
    read_idx_2<1024, double, double, double2>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r7, r8);
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = 1.00000000000000000e+00;
    r10 = r9 + r10;
    r11 = 1.00000000000000008e-15;
    r11 = r9 * r11;
    r7 = fma(r7, r10, r11);
    r7 = 1.0 / r7;
    r9 = r0 * r7;
    r12 = r5 * r9;
    r1 = fma(r3, r12, r1);
    r13 = r0 * r1;
    r14 = r3 * r4;
    r2 = fma(r4, r12, r2);
    r8 = fma(r8, r10, r11);
    r8 = fma(r5, r12, r8);
    r8 = 1.0 / r8;
    r5 = r2 * r8;
    r14 = fma(r1, r5, r7 * r14);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 8 * precond_tril_num_alloc, global_thread_idx, r15, r16);
    r17 = r6 * r3;
    read_idx_2<1024, double, double, double2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r18, r19);
    r19 = fma(r6, r12, r19);
    r20 = r19 * r1;
    r20 = fma(r8, r20, r7 * r17);
    r20 = fma(r0, r20, r16);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 10 * precond_tril_num_alloc, global_thread_idx, r16, r17);
    r21 = r6 * r4;
    r21 = fma(r19, r5, r7 * r21);
    r21 = fma(r0, r21, r16);
    r16 = r6 * r6;
    r22 = r19 * r19;
    r22 = fma(r8, r22, r7 * r16);
    r22 = fma(r0, r22, r11);
    read_idx_2<1024, double, double, double2>(
        precond_diag, 2 * precond_diag_num_alloc, global_thread_idx, r16, r23);
    r22 = fma(r16, r10, r22);
    r22 = 1.0 / r22;
    r16 = r21 * r22;
    r14 = fma(r20, r16, r14);
    r23 = fma(r23, r10, r11);
    r24 = r3 * r3;
    r25 = r20 * r20;
    r25 = fma(r22, r25, r7 * r24);
    r24 = r1 * r1;
    r25 = fma(r8, r24, r25);
    r23 = fma(r0, r25, r23);
    r23 = 1.0 / r23;
    r25 = r14 * r23;
    read_idx_2<1024, double, double, double2>(
        precond_diag, 4 * precond_diag_num_alloc, global_thread_idx, r24, r26);
    r24 = fma(r24, r10, r11);
    r27 = r4 * r4;
    r21 = fma(r21, r16, r7 * r27);
    r21 = fma(r2, r5, r21);
    r21 = fma(r14, r25, r21);
    r24 = fma(r0, r21, r24);
    r24 = 1.0 / r24;
    read_idx_2<1024, double, double, double2>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r21, r14);
    read_idx_2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r2, r27);
    r27 = fma(r2, r12, r27);
    r28 = r0 * r27;
    r28 = fma(r5, r28, r21);
    read_idx_2<1024, double, double, double2>(
        njtr, 2 * njtr_num_alloc, global_thread_idx, r21, r29);
    r30 = r0 * r1;
    r30 = r30 * r27;
    r30 = fma(r8, r30, r29);
    r29 = r0 * r20;
    r31 = r0 * r19;
    r31 = r31 * r27;
    r31 = fma(r8, r31, r21);
    r21 = r6 * r2;
    r31 = fma(r9, r21, r31);
    r29 = r29 * r31;
    r30 = fma(r22, r29, r30);
    r21 = r3 * r2;
    r30 = fma(r9, r21, r30);
    r21 = r4 * r2;
    r28 = fma(r9, r21, r28);
    r29 = r0 * r31;
    r28 = fma(r16, r29, r28);
    r28 = fma(r30, r25, r28);
    read_idx_1<1024, double, double, double>(
        precond_tril, 14 * precond_tril_num_alloc, global_thread_idx, r29);
    r21 = r4 * r18;
    read_idx_2<1024, double, double, double2>(
        precond_tril, 12 * precond_tril_num_alloc, global_thread_idx, r32, r33);
    r32 = r6 * r18;
    r15 = fma(r18, r12, r15);
    r34 = r19 * r15;
    r34 = fma(r8, r34, r7 * r32);
    r34 = fma(r0, r34, r17);
    r17 = r20 * r34;
    r32 = r3 * r18;
    r32 = fma(r7, r32, r22 * r17);
    r17 = r1 * r15;
    r32 = fma(r8, r17, r32);
    r32 = fma(r0, r32, r33);
    r33 = r0 * r32;
    r33 = fma(r25, r33, r7 * r21);
    r33 = fma(r15, r5, r33);
    r33 = fma(r34, r16, r33);
    r33 = fma(r0, r33, r29);
    r29 = r33 * r24;
    r10 = fma(r26, r10, r11);
    r26 = r18 * r18;
    r11 = r15 * r15;
    r11 = fma(r8, r11, r7 * r26);
    r26 = r32 * r32;
    r21 = r34 * r34;
    r11 = fma(r33, r29, r11);
    r11 = fma(r23, r26, r11);
    r11 = fma(r22, r21, r11);
    r10 = fma(r0, r11, r10);
    r10 = 1.0 / r10;
    r11 = r0 * r28;
    r11 = fma(r29, r11, r14);
    r14 = r0 * r27;
    r14 = r14 * r15;
    r11 = fma(r8, r14, r11);
    r21 = r0 * r30;
    r21 = r21 * r32;
    r11 = fma(r23, r21, r11);
    r26 = r18 * r2;
    r11 = fma(r9, r26, r11);
    r33 = r0 * r31;
    r33 = r33 * r34;
    r11 = fma(r22, r33, r11);
    r11 = r10 * r11;
    r10 = r0 * r11;
    r29 = fma(r10, r29, r28 * r24);
    r24 = r32 * r23;
    r24 = fma(r10, r24, r29 * r25);
    r24 = fma(r30, r23, r24);
    r13 = r13 * r24;
    r13 = fma(r27, r8, r8 * r13);
    r25 = r0 * r29;
    r13 = fma(r5, r25, r13);
    r5 = r0 * r19;
    r33 = r0 * r29;
    r33 = fma(r31, r22, r16 * r33);
    r16 = r0 * r20;
    r16 = r16 * r24;
    r33 = fma(r22, r16, r33);
    r26 = r34 * r22;
    r33 = fma(r10, r26, r33);
    r5 = r5 * r33;
    r13 = fma(r8, r5, r13);
    r26 = r15 * r8;
    r13 = fma(r10, r26, r13);
    r26 = r4 * r29;
    r12 = fma(r13, r12, r9 * r26);
    r26 = r6 * r33;
    r12 = fma(r9, r26, r12);
    r5 = r3 * r24;
    r12 = fma(r9, r5, r12);
    r25 = r18 * r9;
    r12 = fma(r11, r25, r12);
    r12 = fma(r2, r7, r12);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               0 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r12,
                                               r13);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               2 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r33,
                                               r24);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               4 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r29,
                                               r11);
  };
}

void PinholePose_normalize(double* precond_diag,
                           unsigned int precond_diag_num_alloc,
                           double* precond_tril,
                           unsigned int precond_tril_num_alloc,
                           double* njtr,
                           unsigned int njtr_num_alloc,
                           const double* const diag,
                           double* out_normalized,
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