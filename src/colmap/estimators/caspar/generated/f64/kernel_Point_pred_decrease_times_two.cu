#include "kernel_Point_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) Point_pred_decrease_times_two_kernel(
    double* Point_step,
    unsigned int Point_step_num_alloc,
    double* Point_precond_diag,
    unsigned int Point_precond_diag_num_alloc,
    const double* const diag,
    double* Point_njtr,
    unsigned int Point_njtr_num_alloc,
    double* const out_Point_pred_dec,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_Point_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, double, double, double>(
        Point_step, 2 * Point_step_num_alloc, global_thread_idx, r0);
    read_idx_1<1024, double, double, double>(
        Point_njtr, 2 * Point_njtr_num_alloc, global_thread_idx, r1);
    read_idx_1<1024, double, double, double>(Point_precond_diag,
                                             2 * Point_precond_diag_num_alloc,
                                             global_thread_idx,
                                             r2);
    r3 = r0 * r2;
  };
  load_unique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fma(r4, r3, r1);
    read_idx_2<1024, double, double, double2>(
        Point_step, 0 * Point_step_num_alloc, global_thread_idx, r1, r5);
    read_idx_2<1024, double, double, double2>(
        Point_njtr, 0 * Point_njtr_num_alloc, global_thread_idx, r6, r7);
    read_idx_2<1024, double, double, double2>(Point_precond_diag,
                                              0 * Point_precond_diag_num_alloc,
                                              global_thread_idx,
                                              r8,
                                              r9);
    r10 = r1 * r8;
    r10 = fma(r4, r10, r6);
    r10 = fma(r1, r10, r0 * r3);
    r3 = r5 * r9;
    r3 = fma(r4, r3, r7);
    r10 = fma(r5, r3, r10);
  };
  sum_store<double>(out_Point_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r10);
  sum_flush_final<double>(out_Point_pred_dec_local, out_Point_pred_dec, 1);
}

void Point_pred_decrease_times_two(double* Point_step,
                                   unsigned int Point_step_num_alloc,
                                   double* Point_precond_diag,
                                   unsigned int Point_precond_diag_num_alloc,
                                   const double* const diag,
                                   double* Point_njtr,
                                   unsigned int Point_njtr_num_alloc,
                                   double* const out_Point_pred_dec,
                                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      Point_step,
      Point_step_num_alloc,
      Point_precond_diag,
      Point_precond_diag_num_alloc,
      diag,
      Point_njtr,
      Point_njtr_num_alloc,
      out_Point_pred_dec,
      problem_size);
}

}  // namespace caspar