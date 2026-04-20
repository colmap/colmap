#include "kernel_SimpleRadialExtraCalib_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_pred_decrease_times_two_kernel(
        double* SimpleRadialExtraCalib_step,
        unsigned int SimpleRadialExtraCalib_step_num_alloc,
        double* SimpleRadialExtraCalib_precond_diag,
        unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
        const double* const diag,
        double* SimpleRadialExtraCalib_njtr,
        unsigned int SimpleRadialExtraCalib_njtr_num_alloc,
        double* const out_SimpleRadialExtraCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SimpleRadialExtraCalib_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_step,
        0 * SimpleRadialExtraCalib_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_njtr,
        0 * SimpleRadialExtraCalib_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_precond_diag,
        0 * SimpleRadialExtraCalib_precond_diag_num_alloc,
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
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_step,
        2 * SimpleRadialExtraCalib_step_num_alloc,
        global_thread_idx,
        r3);
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_njtr,
        2 * SimpleRadialExtraCalib_njtr_num_alloc,
        global_thread_idx,
        r8);
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_precond_diag,
        2 * SimpleRadialExtraCalib_precond_diag_num_alloc,
        global_thread_idx,
        r9);
    r10 = r3 * r9;
    r10 = fma(r7, r10, r8);
    r10 = fma(r3, r10, r1 * r6);
    r6 = r0 * r4;
    r6 = fma(r7, r6, r2);
    r10 = fma(r0, r6, r10);
  };
  sum_store<double>(out_SimpleRadialExtraCalib_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r10);
  sum_flush_final<double>(out_SimpleRadialExtraCalib_pred_dec_local,
                          out_SimpleRadialExtraCalib_pred_dec,
                          1);
}

void SimpleRadialExtraCalib_pred_decrease_times_two(
    double* SimpleRadialExtraCalib_step,
    unsigned int SimpleRadialExtraCalib_step_num_alloc,
    double* SimpleRadialExtraCalib_precond_diag,
    unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialExtraCalib_njtr,
    unsigned int SimpleRadialExtraCalib_njtr_num_alloc,
    double* const out_SimpleRadialExtraCalib_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialExtraCalib_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      SimpleRadialExtraCalib_step,
      SimpleRadialExtraCalib_step_num_alloc,
      SimpleRadialExtraCalib_precond_diag,
      SimpleRadialExtraCalib_precond_diag_num_alloc,
      diag,
      SimpleRadialExtraCalib_njtr,
      SimpleRadialExtraCalib_njtr_num_alloc,
      out_SimpleRadialExtraCalib_pred_dec,
      problem_size);
}

}  // namespace caspar