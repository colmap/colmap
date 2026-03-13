#include "kernel_PinholeCalib_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeCalib_pred_decrease_times_two_kernel(
        double* PinholeCalib_step,
        unsigned int PinholeCalib_step_num_alloc,
        double* PinholeCalib_precond_diag,
        unsigned int PinholeCalib_precond_diag_num_alloc,
        const double* const diag,
        double* PinholeCalib_njtr,
        unsigned int PinholeCalib_njtr_num_alloc,
        double* const out_PinholeCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_PinholeCalib_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholeCalib_step,
                                              2 * PinholeCalib_step_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(PinholeCalib_njtr,
                                              2 * PinholeCalib_njtr_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    read_idx_2<1024, double, double, double2>(
        PinholeCalib_precond_diag,
        2 * PinholeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r4,
        r5);
    r6 = r1 * r5;
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r3);
    r3 = r0 * r4;
    r3 = fma(r7, r3, r2);
    r3 = fma(r0, r3, r1 * r6);
    read_idx_2<1024, double, double, double2>(PinholeCalib_step,
                                              0 * PinholeCalib_step_num_alloc,
                                              global_thread_idx,
                                              r6,
                                              r2);
    read_idx_2<1024, double, double, double2>(PinholeCalib_njtr,
                                              0 * PinholeCalib_njtr_num_alloc,
                                              global_thread_idx,
                                              r8,
                                              r9);
    read_idx_2<1024, double, double, double2>(
        PinholeCalib_precond_diag,
        0 * PinholeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r10,
        r11);
    r12 = r6 * r10;
    r12 = fma(r7, r12, r8);
    r8 = r2 * r11;
    r8 = fma(r7, r8, r9);
    r3 = fma(r6, r12, r3);
    r3 = fma(r2, r8, r3);
  };
  sum_store<double>(out_PinholeCalib_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(
      out_PinholeCalib_pred_dec_local, out_PinholeCalib_pred_dec, 1);
}

void PinholeCalib_pred_decrease_times_two(
    double* PinholeCalib_step,
    unsigned int PinholeCalib_step_num_alloc,
    double* PinholeCalib_precond_diag,
    unsigned int PinholeCalib_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeCalib_njtr,
    unsigned int PinholeCalib_njtr_num_alloc,
    double* const out_PinholeCalib_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeCalib_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      PinholeCalib_step,
      PinholeCalib_step_num_alloc,
      PinholeCalib_precond_diag,
      PinholeCalib_precond_diag_num_alloc,
      diag,
      PinholeCalib_njtr,
      PinholeCalib_njtr_num_alloc,
      out_PinholeCalib_pred_dec,
      problem_size);
}

}  // namespace caspar