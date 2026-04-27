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
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    r0 = -1.00000000000000000e+00;
    read_idx_2<1024, double, double, double2>(
        precond_tril, 2 * precond_tril_num_alloc, global_thread_idx, r1, r2);
    read_idx_2<1024, double, double, double2>(
        precond_tril, 0 * precond_tril_num_alloc, global_thread_idx, r3, r4);
    read_idx_2<1024, double, double, double2>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r5, r6);
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = 1.00000000000000000e+00;
    r8 = r7 + r8;
    r9 = 1.00000000000000008e-15;
    r9 = r7 * r9;
    r5 = fma(r5, r8, r9);
    r5 = 1.0 / r5;
    r7 = r0 * r5;
    r10 = r3 * r7;
    r2 = fma(r4, r10, r2);
    r6 = fma(r6, r8, r9);
    r6 = fma(r3, r10, r6);
    r6 = 1.0 / r6;
    r3 = r2 * r6;
    r11 = r0 * r3;
    r12 = r4 * r4;
    r2 = fma(r2, r3, r5 * r12);
    r2 = fma(r0, r2, r9);
    read_idx_2<1024, double, double, double2>(
        precond_diag, 2 * precond_diag_num_alloc, global_thread_idx, r12, r13);
    r2 = fma(r12, r8, r2);
    r2 = 1.0 / r2;
    read_idx_2<1024, double, double, double2>(
        njtr, 2 * njtr_num_alloc, global_thread_idx, r12, r14);
    read_idx_2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r15, r16);
    r16 = fma(r15, r10, r16);
    r12 = fma(r16, r11, r12);
    r17 = r4 * r15;
    r12 = fma(r7, r17, r12);
    r17 = r4 * r1;
    read_idx_2<1024, double, double, double2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r18, r19);
    r18 = fma(r1, r10, r18);
    r3 = fma(r18, r3, r5 * r17);
    r17 = r3 * r2;
    r14 = fma(r12, r17, r14);
    r19 = r0 * r16;
    r19 = r19 * r18;
    r14 = fma(r6, r19, r14);
    r20 = r1 * r15;
    r14 = fma(r7, r20, r14);
    r8 = fma(r13, r8, r9);
    r13 = r1 * r1;
    r3 = fma(r3, r17, r5 * r13);
    r13 = r18 * r18;
    r3 = fma(r6, r13, r3);
    r8 = fma(r0, r3, r8);
    r8 = 1.0 / r8;
    r8 = r14 * r8;
    r17 = fma(r8, r17, r12 * r2);
    r11 = fma(r16, r6, r17 * r11);
    r2 = r0 * r18;
    r2 = r2 * r6;
    r11 = fma(r8, r2, r11);
    r2 = r1 * r7;
    r2 = fma(r8, r2, r11 * r10);
    r10 = r4 * r17;
    r2 = fma(r7, r10, r2);
    r2 = fma(r15, r5, r2);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               0 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r2,
                                               r11);
    write_idx_2<1024, double, double, double2>(out_normalized,
                                               2 * out_normalized_num_alloc,
                                               global_thread_idx,
                                               r17,
                                               r8);
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