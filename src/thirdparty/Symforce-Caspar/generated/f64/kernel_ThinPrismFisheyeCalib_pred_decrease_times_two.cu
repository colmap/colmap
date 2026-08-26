#include "kernel_ThinPrismFisheyeCalib_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibPredDecreaseTimesTwoKernel(
        double* ThinPrismFisheyeCalib_step,
        unsigned int ThinPrismFisheyeCalib_step_num_alloc,
        double* ThinPrismFisheyeCalib_precond_diag,
        unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
        const double* const diag,
        double* ThinPrismFisheyeCalib_njtr,
        unsigned int ThinPrismFisheyeCalib_njtr_num_alloc,
        double* const out_ThinPrismFisheyeCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_ThinPrismFisheyeCalib_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_step,
        10 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_njtr,
        10 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        10 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
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
    r3 = r0 * r4;
    r3 = fma(r7, r3, r2);
    r3 = fma(r0, r3, r1 * r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_step,
        6 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r6,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_njtr,
        6 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r8,
        r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        6 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r10,
        r11);
    r12 = r2 * r11;
    r12 = fma(r7, r12, r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_step,
        8 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r9,
        r13);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_njtr,
        8 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r14,
        r15);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        8 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r16,
        r17);
    r18 = r9 * r16;
    r18 = fma(r7, r18, r14);
    r14 = r13 * r17;
    r14 = fma(r7, r14, r15);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_step,
        2 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r15,
        r19);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_njtr,
        2 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r20,
        r21);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        2 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r22,
        r23);
    r24 = r19 * r23;
    r24 = fma(r7, r24, r21);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_step,
        4 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r21,
        r25);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_njtr,
        4 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r26,
        r27);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        4 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r28,
        r29);
    r30 = r21 * r28;
    r30 = fma(r7, r30, r26);
    r26 = r25 * r29;
    r26 = fma(r7, r26, r27);
    r27 = r15 * r22;
    r27 = fma(r7, r27, r20);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_step,
        0 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r20,
        r31);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_njtr,
        0 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r32,
        r33);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        0 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r34,
        r35);
    r36 = r31 * r35;
    r36 = fma(r7, r36, r33);
    r33 = r6 * r10;
    r33 = fma(r7, r33, r8);
    r8 = r20 * r34;
    r8 = fma(r7, r8, r32);
    r3 = fma(r2, r12, r3);
    r3 = fma(r9, r18, r3);
    r3 = fma(r13, r14, r3);
    r3 = fma(r19, r24, r3);
    r3 = fma(r21, r30, r3);
    r3 = fma(r25, r26, r3);
    r3 = fma(r15, r27, r3);
    r3 = fma(r31, r36, r3);
    r3 = fma(r6, r33, r3);
    r3 = fma(r20, r8, r3);
  };
  SumStore<double>(out_ThinPrismFisheyeCalib_pred_dec_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  SumFlushFinal<double>(out_ThinPrismFisheyeCalib_pred_dec_local,
                        out_ThinPrismFisheyeCalib_pred_dec,
                        1);
}

void ThinPrismFisheyeCalibPredDecreaseTimesTwo(
    double* ThinPrismFisheyeCalib_step,
    unsigned int ThinPrismFisheyeCalib_step_num_alloc,
    double* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyeCalib_njtr,
    unsigned int ThinPrismFisheyeCalib_njtr_num_alloc,
    double* const out_ThinPrismFisheyeCalib_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibPredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_step,
      ThinPrismFisheyeCalib_step_num_alloc,
      ThinPrismFisheyeCalib_precond_diag,
      ThinPrismFisheyeCalib_precond_diag_num_alloc,
      diag,
      ThinPrismFisheyeCalib_njtr,
      ThinPrismFisheyeCalib_njtr_num_alloc,
      out_ThinPrismFisheyeCalib_pred_dec,
      problem_size);
}

}  // namespace caspar