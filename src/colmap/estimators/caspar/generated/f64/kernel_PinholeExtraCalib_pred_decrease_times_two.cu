#include "kernel_PinholeExtraCalib_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeExtraCalib_pred_decrease_times_two_kernel(
        double* PinholeExtraCalib_step,
        unsigned int PinholeExtraCalib_step_num_alloc,
        double* PinholeExtraCalib_precond_diag,
        unsigned int PinholeExtraCalib_precond_diag_num_alloc,
        const double* const diag,
        double* PinholeExtraCalib_njtr,
        unsigned int PinholeExtraCalib_njtr_num_alloc,
        double* const out_PinholeExtraCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_PinholeExtraCalib_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholeExtraCalib_step,
        0 * PinholeExtraCalib_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        PinholeExtraCalib_njtr,
        0 * PinholeExtraCalib_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(
        PinholeExtraCalib_precond_diag,
        0 * PinholeExtraCalib_precond_diag_num_alloc,
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
    r2 = r1 * r5;
    r2 = fma(r7, r2, r3);
    r2 = fma(r1, r2, r0 * r6);
  };
  sum_store<double>(out_PinholeExtraCalib_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  sum_flush_final<double>(
      out_PinholeExtraCalib_pred_dec_local, out_PinholeExtraCalib_pred_dec, 1);
}

void PinholeExtraCalib_pred_decrease_times_two(
    double* PinholeExtraCalib_step,
    unsigned int PinholeExtraCalib_step_num_alloc,
    double* PinholeExtraCalib_precond_diag,
    unsigned int PinholeExtraCalib_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeExtraCalib_njtr,
    unsigned int PinholeExtraCalib_njtr_num_alloc,
    double* const out_PinholeExtraCalib_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeExtraCalib_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      PinholeExtraCalib_step,
      PinholeExtraCalib_step_num_alloc,
      PinholeExtraCalib_precond_diag,
      PinholeExtraCalib_precond_diag_num_alloc,
      diag,
      PinholeExtraCalib_njtr,
      PinholeExtraCalib_njtr_num_alloc,
      out_PinholeExtraCalib_pred_dec,
      problem_size);
}

}  // namespace caspar