#include "kernel_ThinPrismFisheyePose_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyePosePredDecreaseTimesTwoKernel(
        double* ThinPrismFisheyePose_step,
        unsigned int ThinPrismFisheyePose_step_num_alloc,
        double* ThinPrismFisheyePose_precond_diag,
        unsigned int ThinPrismFisheyePose_precond_diag_num_alloc,
        const double* const diag,
        double* ThinPrismFisheyePose_njtr,
        unsigned int ThinPrismFisheyePose_njtr_num_alloc,
        double* const out_ThinPrismFisheyePose_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_ThinPrismFisheyePose_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_step,
        0 * ThinPrismFisheyePose_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_njtr,
        0 * ThinPrismFisheyePose_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_precond_diag,
        0 * ThinPrismFisheyePose_precond_diag_num_alloc,
        global_thread_idx,
        r4,
        r5);
    r6 = r1 * r5;
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_step,
        4 * ThinPrismFisheyePose_step_num_alloc,
        global_thread_idx,
        r3,
        r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_njtr,
        4 * ThinPrismFisheyePose_njtr_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_precond_diag,
        4 * ThinPrismFisheyePose_precond_diag_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r13 = r8 * r12;
    r13 = fma(r7, r13, r10);
    r13 = fma(r8, r13, r1 * r6);
    r6 = r3 * r11;
    r6 = fma(r7, r6, r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_step,
        2 * ThinPrismFisheyePose_step_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_njtr,
        2 * ThinPrismFisheyePose_njtr_num_alloc,
        global_thread_idx,
        r14,
        r15);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_precond_diag,
        2 * ThinPrismFisheyePose_precond_diag_num_alloc,
        global_thread_idx,
        r16,
        r17);
    r18 = r9 * r16;
    r18 = fma(r7, r18, r14);
    r14 = r10 * r17;
    r14 = fma(r7, r14, r15);
    r15 = r0 * r4;
    r15 = fma(r7, r15, r2);
    r13 = fma(r3, r6, r13);
    r13 = fma(r9, r18, r13);
    r13 = fma(r10, r14, r13);
    r13 = fma(r0, r15, r13);
  };
  SumStore<double>(out_ThinPrismFisheyePose_pred_dec_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r13);
  SumFlushFinal<double>(out_ThinPrismFisheyePose_pred_dec_local,
                        out_ThinPrismFisheyePose_pred_dec,
                        1);
}

void ThinPrismFisheyePosePredDecreaseTimesTwo(
    double* ThinPrismFisheyePose_step,
    unsigned int ThinPrismFisheyePose_step_num_alloc,
    double* ThinPrismFisheyePose_precond_diag,
    unsigned int ThinPrismFisheyePose_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyePose_njtr,
    unsigned int ThinPrismFisheyePose_njtr_num_alloc,
    double* const out_ThinPrismFisheyePose_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePosePredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyePose_step,
      ThinPrismFisheyePose_step_num_alloc,
      ThinPrismFisheyePose_precond_diag,
      ThinPrismFisheyePose_precond_diag_num_alloc,
      diag,
      ThinPrismFisheyePose_njtr,
      ThinPrismFisheyePose_njtr_num_alloc,
      out_ThinPrismFisheyePose_pred_dec,
      problem_size);
}

}  // namespace caspar