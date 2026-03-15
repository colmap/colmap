#include "kernel_SimpleRadialCalib_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialCalib_pred_decrease_times_two_kernel(
        double* SimpleRadialCalib_step,
        unsigned int SimpleRadialCalib_step_num_alloc,
        double* SimpleRadialCalib_precond_diag,
        unsigned int SimpleRadialCalib_precond_diag_num_alloc,
        const double* const diag,
        double* SimpleRadialCalib_njtr,
        unsigned int SimpleRadialCalib_njtr_num_alloc,
        double* const out_SimpleRadialCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SimpleRadialCalib_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_step,
        0 * SimpleRadialCalib_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_njtr,
        0 * SimpleRadialCalib_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_precond_diag,
        0 * SimpleRadialCalib_precond_diag_num_alloc,
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
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_step,
        2 * SimpleRadialCalib_step_num_alloc,
        global_thread_idx,
        r6,
        r2);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_njtr,
        2 * SimpleRadialCalib_njtr_num_alloc,
        global_thread_idx,
        r8,
        r9);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_precond_diag,
        2 * SimpleRadialCalib_precond_diag_num_alloc,
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
  sum_store<double>(out_SimpleRadialCalib_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(
      out_SimpleRadialCalib_pred_dec_local, out_SimpleRadialCalib_pred_dec, 1);
}

void SimpleRadialCalib_pred_decrease_times_two(
    double* SimpleRadialCalib_step,
    unsigned int SimpleRadialCalib_step_num_alloc,
    double* SimpleRadialCalib_precond_diag,
    unsigned int SimpleRadialCalib_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialCalib_njtr,
    unsigned int SimpleRadialCalib_njtr_num_alloc,
    double* const out_SimpleRadialCalib_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialCalib_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      SimpleRadialCalib_step,
      SimpleRadialCalib_step_num_alloc,
      SimpleRadialCalib_precond_diag,
      SimpleRadialCalib_precond_diag_num_alloc,
      diag,
      SimpleRadialCalib_njtr,
      SimpleRadialCalib_njtr_num_alloc,
      out_SimpleRadialCalib_pred_dec,
      problem_size);
}

}  // namespace caspar