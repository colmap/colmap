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
    SimpleRadialCalib_normalize_kernel(double* precond_diag,
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
      r16, r17, r18, r19, r20, r21, r22, r23, r24;
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = 1.00000000000000008e-15;
    r1 = r0 * r1;
    read_idx_2<1024, double, double, double2>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r2, r3);
    r4 = 1.00000000000000000e+00;
    r4 = r0 + r4;
    r3 = fma(r3, r4, r1);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 0 * precond_tril_num_alloc, global_thread_idx, r0, r5);
    r6 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r1);
    r7 = 1.0 / r2;
    r8 = r6 * r7;
    r9 = r0 * r8;
    r3 = fma(r0, r9, r3);
    r3 = 1.0 / r3;
    read_idx_2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r10, r11);
    r11 = fma(r10, r9, r11);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r12, r13);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 2 * precond_tril_num_alloc, global_thread_idx, r14, r15);
    r12 = fma(r14, r9, r12);
    r15 = r12 * r3;
    read_idx_2<1024, double, double, double2>(
        njtr, 2 * njtr_num_alloc, global_thread_idx, r16, r17);
    r18 = r5 * r10;
    r18 = fma(r8, r18, r16);
    r16 = r5 * r3;
    r19 = r0 * r16;
    r20 = r7 * r19;
    r18 = fma(r11, r20, r18);
    r21 = r6 * r18;
    r22 = r5 * r14;
    r23 = r12 * r9;
    r23 = fma(r16, r23, r7 * r22);
    r23 = fma(r6, r23, r13);
    read_idx_2<1024, double, double, double2>(
        precond_diag, 2 * precond_diag_num_alloc, global_thread_idx, r13, r22);
    r13 = fma(r13, r4, r1);
    r16 = r5 * r5;
    r24 = r0 * r5;
    r2 = r2 * r2;
    r2 = 1.0 / r2;
    r24 = r24 * r2;
    r24 = fma(r19, r24, r7 * r16);
    r13 = fma(r6, r24, r13);
    r13 = 1.0 / r13;
    r24 = r23 * r13;
    r21 = fma(r24, r21, r17);
    r17 = r6 * r12;
    r17 = r17 * r11;
    r21 = fma(r3, r17, r21);
    r16 = r14 * r10;
    r21 = fma(r8, r16, r21);
    r16 = r14 * r14;
    r23 = fma(r23, r24, r7 * r16);
    r16 = r12 * r12;
    r23 = fma(r3, r16, r23);
    r23 = fma(r6, r23, r1);
    r23 = fma(r22, r4, r23);
    r23 = 1.0 / r23;
    r23 = r21 * r23;
    r21 = r6 * r23;
    r15 = fma(r21, r15, r11 * r3);
    r13 = fma(r18, r13, r24 * r21);
    r15 = fma(r13, r20, r15);
    r20 = r14 * r8;
    r20 = fma(r15, r9, r23 * r20);
    r21 = r5 * r13;
    r20 = fma(r8, r21, r20);
    r20 = fma(r10, r7, r20);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               0 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r20,
                                               r15);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               2 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r13,
                                               r23);
  };
}

void SimpleRadialCalib_normalize(double* precond_diag,
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