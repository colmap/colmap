#include "kernel_Pose_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Pose_normalize_kernel(double* precond_diag,
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
      r31, r32, r33, r34, r35, r36;

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
    r2 = fma(r4, r12, r2);
    r13 = r0 * r2;
    read_idx_1<1024, double, double, double>(
        precond_tril, 14 * precond_tril_num_alloc, global_thread_idx, r14);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r15, r16);
    r17 = r4 * r15;
    read_idx_2<1024, double, double, double2>(
        precond_tril, 12 * precond_tril_num_alloc, global_thread_idx, r18, r19);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 10 * precond_tril_num_alloc, global_thread_idx, r20, r21);
    r22 = r6 * r15;
    r16 = fma(r6, r12, r16);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 8 * precond_tril_num_alloc, global_thread_idx, r23, r24);
    r23 = fma(r15, r12, r23);
    r25 = r16 * r23;
    r8 = fma(r8, r10, r11);
    r8 = fma(r5, r12, r8);
    r8 = 1.0 / r8;
    r25 = fma(r8, r25, r7 * r22);
    r25 = fma(r0, r25, r21);
    r21 = r6 * r3;
    r1 = fma(r3, r12, r1);
    r22 = r1 * r8;
    r21 = fma(r16, r22, r7 * r21);
    r21 = fma(r0, r21, r24);
    r24 = r6 * r6;
    r5 = r16 * r16;
    r5 = fma(r8, r5, r7 * r24);
    r5 = fma(r0, r5, r11);
    read_idx_2<1024, double, double, double2>(
        precond_diag, 2 * precond_diag_num_alloc, global_thread_idx, r24, r26);
    r5 = fma(r24, r10, r5);
    r5 = 1.0 / r5;
    r24 = r21 * r5;
    r27 = r3 * r15;
    r27 = fma(r7, r27, r25 * r24);
    r27 = fma(r23, r22, r27);
    r27 = fma(r0, r27, r19);
    r19 = r3 * r4;
    r19 = fma(r2, r22, r7 * r19);
    r28 = r6 * r4;
    r29 = r16 * r2;
    r29 = fma(r8, r29, r7 * r28);
    r29 = fma(r0, r29, r20);
    r19 = fma(r29, r24, r19);
    r19 = fma(r0, r19, r18);
    r26 = fma(r26, r10, r11);
    r18 = r3 * r3;
    r21 = fma(r21, r24, r7 * r18);
    r21 = fma(r1, r22, r21);
    r26 = fma(r0, r21, r26);
    r26 = 1.0 / r26;
    r21 = r19 * r26;
    r17 = fma(r27, r21, r7 * r17);
    r1 = r2 * r23;
    r17 = fma(r8, r1, r17);
    r18 = r29 * r25;
    r17 = fma(r5, r18, r17);
    r17 = fma(r0, r17, r14);
    read_idx_2<1024, double, double, double2>(
        precond_diag, 4 * precond_diag_num_alloc, global_thread_idx, r14, r18);
    r14 = fma(r14, r10, r11);
    r1 = r4 * r4;
    r1 = fma(r7, r1, r19 * r21);
    r19 = r29 * r29;
    r20 = r2 * r2;
    r1 = fma(r5, r19, r1);
    r1 = fma(r8, r20, r1);
    r14 = fma(r0, r1, r14);
    r14 = 1.0 / r14;
    r1 = r17 * r14;
    read_idx_2<1024, double, double, double2>(
        njtr, 4 * njtr_num_alloc, global_thread_idx, r20, r19);
    r28 = r0 * r2;
    read_idx_2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r30, r31);
    r31 = fma(r30, r12, r31);
    r28 = r28 * r31;
    r28 = fma(r8, r28, r20);
    read_idx_2<1024, double, double, double2>(
        njtr, 2 * njtr_num_alloc, global_thread_idx, r20, r32);
    r33 = r0 * r31;
    r33 = fma(r22, r33, r32);
    r32 = r0 * r16;
    r32 = r32 * r31;
    r32 = fma(r8, r32, r20);
    r20 = r6 * r30;
    r32 = fma(r9, r20, r32);
    r20 = r0 * r32;
    r33 = fma(r24, r20, r33);
    r34 = r3 * r30;
    r33 = fma(r9, r34, r33);
    r34 = r0 * r33;
    r28 = fma(r21, r34, r28);
    r20 = r4 * r30;
    r28 = fma(r9, r20, r28);
    r35 = r0 * r29;
    r35 = r35 * r32;
    r28 = fma(r5, r35, r28);
    r35 = r0 * r28;
    r35 = fma(r1, r35, r19);
    r19 = r0 * r23;
    r19 = r19 * r31;
    r35 = fma(r8, r19, r35);
    r20 = r0 * r27;
    r20 = r20 * r33;
    r35 = fma(r26, r20, r35);
    r34 = r15 * r30;
    r35 = fma(r9, r34, r35);
    r36 = r0 * r25;
    r36 = r36 * r32;
    r35 = fma(r5, r36, r35);
    r36 = r15 * r15;
    r36 = fma(r7, r36, r17 * r1);
    r17 = r23 * r23;
    r34 = r27 * r27;
    r20 = r25 * r25;
    r36 = fma(r8, r17, r36);
    r36 = fma(r26, r34, r36);
    r36 = fma(r5, r20, r36);
    r36 = fma(r0, r36, r11);
    r36 = fma(r18, r10, r36);
    r36 = 1.0 / r36;
    r36 = r35 * r36;
    r35 = r0 * r36;
    r14 = fma(r28, r14, r35 * r1);
    r13 = r13 * r14;
    r1 = r27 * r26;
    r10 = r0 * r14;
    r10 = fma(r21, r10, r35 * r1);
    r10 = fma(r33, r26, r10);
    r1 = r0 * r10;
    r1 = fma(r22, r1, r8 * r13);
    r13 = r0 * r16;
    r22 = r25 * r5;
    r22 = fma(r35, r22, r32 * r5);
    r21 = r0 * r10;
    r22 = fma(r24, r21, r22);
    r24 = r0 * r29;
    r24 = r24 * r14;
    r22 = fma(r5, r24, r22);
    r13 = r13 * r22;
    r1 = fma(r8, r13, r1);
    r24 = r23 * r8;
    r1 = fma(r35, r24, r1);
    r1 = fma(r31, r8, r1);
    r24 = r15 * r9;
    r12 = fma(r1, r12, r36 * r24);
    r24 = r6 * r22;
    r12 = fma(r9, r24, r12);
    r13 = r4 * r14;
    r12 = fma(r9, r13, r12);
    r35 = r3 * r10;
    r12 = fma(r9, r35, r12);
    r12 = fma(r30, r7, r12);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               0 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r12,
                                               r1);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               2 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r22,
                                               r10);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               4 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r14,
                                               r36);
  };
}

void Pose_normalize(double* precond_diag,
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
  Pose_normalize_kernel<<<n_blocks, 1024>>>(precond_diag,
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