#include "kernel_PinholePose_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_pred_decrease_times_two_kernel(
        double* PinholePose_step,
        unsigned int PinholePose_step_num_alloc,
        double* PinholePose_precond_diag,
        unsigned int PinholePose_precond_diag_num_alloc,
        const double* const diag,
        double* PinholePose_njtr,
        unsigned int PinholePose_njtr_num_alloc,
        double* const out_PinholePose_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_PinholePose_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholePose_step,
                                              2 * PinholePose_step_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(PinholePose_njtr,
                                              2 * PinholePose_njtr_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    read_idx_2<1024, double, double, double2>(
        PinholePose_precond_diag,
        2 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r4,
        r5);
    r6 = r0 * r4;
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r2);
    read_idx_2<1024, double, double, double2>(PinholePose_step,
                                              0 * PinholePose_step_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r8);
    read_idx_2<1024, double, double, double2>(PinholePose_njtr,
                                              0 * PinholePose_njtr_num_alloc,
                                              global_thread_idx,
                                              r9,
                                              r10);
    read_idx_2<1024, double, double, double2>(
        PinholePose_precond_diag,
        0 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r13 = r2 * r11;
    r13 = fma(r7, r13, r9);
    r13 = fma(r2, r13, r0 * r6);
    r6 = r8 * r12;
    r6 = fma(r7, r6, r10);
    r10 = r1 * r5;
    r10 = fma(r7, r10, r3);
    read_idx_2<1024, double, double, double2>(PinholePose_step,
                                              4 * PinholePose_step_num_alloc,
                                              global_thread_idx,
                                              r3,
                                              r9);
    read_idx_2<1024, double, double, double2>(PinholePose_njtr,
                                              4 * PinholePose_njtr_num_alloc,
                                              global_thread_idx,
                                              r14,
                                              r15);
    read_idx_2<1024, double, double, double2>(
        PinholePose_precond_diag,
        4 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r16,
        r17);
    r18 = r3 * r16;
    r18 = fma(r7, r18, r14);
    r14 = r9 * r17;
    r14 = fma(r7, r14, r15);
    r13 = fma(r8, r6, r13);
    r13 = fma(r1, r10, r13);
    r13 = fma(r3, r18, r13);
    r13 = fma(r9, r14, r13);
  };
  sum_store<double>(out_PinholePose_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r13);
  sum_flush_final<double>(
      out_PinholePose_pred_dec_local, out_PinholePose_pred_dec, 1);
}

void PinholePose_pred_decrease_times_two(
    double* PinholePose_step,
    unsigned int PinholePose_step_num_alloc,
    double* PinholePose_precond_diag,
    unsigned int PinholePose_precond_diag_num_alloc,
    const double* const diag,
    double* PinholePose_njtr,
    unsigned int PinholePose_njtr_num_alloc,
    double* const out_PinholePose_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      PinholePose_step,
      PinholePose_step_num_alloc,
      PinholePose_precond_diag,
      PinholePose_precond_diag_num_alloc,
      diag,
      PinholePose_njtr,
      PinholePose_njtr_num_alloc,
      out_PinholePose_pred_dec,
      problem_size);
}

}  // namespace caspar