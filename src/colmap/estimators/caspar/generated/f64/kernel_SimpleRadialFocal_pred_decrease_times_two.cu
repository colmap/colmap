#include "kernel_SimpleRadialFocal_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocal_pred_decrease_times_two_kernel(
        double* SimpleRadialFocal_step,
        unsigned int SimpleRadialFocal_step_num_alloc,
        double* SimpleRadialFocal_precond_diag,
        unsigned int SimpleRadialFocal_precond_diag_num_alloc,
        const double* const diag,
        double* SimpleRadialFocal_njtr,
        unsigned int SimpleRadialFocal_njtr_num_alloc,
        double* const out_SimpleRadialFocal_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SimpleRadialFocal_pred_dec_local[1];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, double, double, double>(
        SimpleRadialFocal_step,
        0 * SimpleRadialFocal_step_num_alloc,
        global_thread_idx,
        r0);
    read_idx_1<1024, double, double, double>(
        SimpleRadialFocal_njtr,
        0 * SimpleRadialFocal_njtr_num_alloc,
        global_thread_idx,
        r1);
    read_idx_1<1024, double, double, double>(
        SimpleRadialFocal_precond_diag,
        0 * SimpleRadialFocal_precond_diag_num_alloc,
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
    r3 = r0 * r3;
  };
  sum_store<double>(out_SimpleRadialFocal_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(
      out_SimpleRadialFocal_pred_dec_local, out_SimpleRadialFocal_pred_dec, 1);
}

void SimpleRadialFocal_pred_decrease_times_two(
    double* SimpleRadialFocal_step,
    unsigned int SimpleRadialFocal_step_num_alloc,
    double* SimpleRadialFocal_precond_diag,
    unsigned int SimpleRadialFocal_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialFocal_njtr,
    unsigned int SimpleRadialFocal_njtr_num_alloc,
    double* const out_SimpleRadialFocal_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocal_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      SimpleRadialFocal_step,
      SimpleRadialFocal_step_num_alloc,
      SimpleRadialFocal_precond_diag,
      SimpleRadialFocal_precond_diag_num_alloc,
      diag,
      SimpleRadialFocal_njtr,
      SimpleRadialFocal_njtr_num_alloc,
      out_SimpleRadialFocal_pred_dec,
      problem_size);
}

}  // namespace caspar