#include "kernel_SimpleRadialPose_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialPose_pred_decrease_times_two_kernel(
        double* SimpleRadialPose_step,
        unsigned int SimpleRadialPose_step_num_alloc,
        double* SimpleRadialPose_precond_diag,
        unsigned int SimpleRadialPose_precond_diag_num_alloc,
        const double* const diag,
        double* SimpleRadialPose_njtr,
        unsigned int SimpleRadialPose_njtr_num_alloc,
        double* const out_SimpleRadialPose_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_SimpleRadialPose_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_step,
        2 * SimpleRadialPose_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_njtr,
        2 * SimpleRadialPose_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_precond_diag,
        2 * SimpleRadialPose_precond_diag_num_alloc,
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
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_step,
        4 * SimpleRadialPose_step_num_alloc,
        global_thread_idx,
        r6,
        r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_njtr,
        4 * SimpleRadialPose_njtr_num_alloc,
        global_thread_idx,
        r8,
        r9);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_precond_diag,
        4 * SimpleRadialPose_precond_diag_num_alloc,
        global_thread_idx,
        r10,
        r11);
    r12 = r6 * r10;
    r12 = fma(r7, r12, r8);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_step,
        0 * SimpleRadialPose_step_num_alloc,
        global_thread_idx,
        r8,
        r13);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_njtr,
        0 * SimpleRadialPose_njtr_num_alloc,
        global_thread_idx,
        r14,
        r15);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialPose_precond_diag,
        0 * SimpleRadialPose_precond_diag_num_alloc,
        global_thread_idx,
        r16,
        r17);
    r18 = r8 * r16;
    r18 = fma(r7, r18, r14);
    r14 = r13 * r17;
    r14 = fma(r7, r14, r15);
    r15 = r3 * r11;
    r15 = fma(r7, r15, r9);
    r2 = fma(r6, r12, r2);
    r2 = fma(r8, r18, r2);
    r2 = fma(r13, r14, r2);
    r2 = fma(r3, r15, r2);
  };
  sum_store<double>(out_SimpleRadialPose_pred_dec_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  sum_flush_final<double>(
      out_SimpleRadialPose_pred_dec_local, out_SimpleRadialPose_pred_dec, 1);
}

void SimpleRadialPose_pred_decrease_times_two(
    double* SimpleRadialPose_step,
    unsigned int SimpleRadialPose_step_num_alloc,
    double* SimpleRadialPose_precond_diag,
    unsigned int SimpleRadialPose_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialPose_njtr,
    unsigned int SimpleRadialPose_njtr_num_alloc,
    double* const out_SimpleRadialPose_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPose_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      SimpleRadialPose_step,
      SimpleRadialPose_step_num_alloc,
      SimpleRadialPose_precond_diag,
      SimpleRadialPose_precond_diag_num_alloc,
      diag,
      SimpleRadialPose_njtr,
      SimpleRadialPose_njtr_num_alloc,
      out_SimpleRadialPose_pred_dec,
      problem_size);
}

}  // namespace caspar