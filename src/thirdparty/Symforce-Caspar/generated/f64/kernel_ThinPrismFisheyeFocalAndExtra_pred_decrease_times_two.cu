#include "kernel_ThinPrismFisheyeFocalAndExtra_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraPredDecreaseTimesTwoKernel(
        double* ThinPrismFisheyeFocalAndExtra_step,
        unsigned int ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        double* ThinPrismFisheyeFocalAndExtra_precond_diag,
        unsigned int ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        const double* const diag,
        double* ThinPrismFisheyeFocalAndExtra_njtr,
        unsigned int ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        double* const out_ThinPrismFisheyeFocalAndExtra_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_ThinPrismFisheyeFocalAndExtra_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step,
        8 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        8 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        8 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
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
        ThinPrismFisheyeFocalAndExtra_step,
        2 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r3,
        r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        2 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        2 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r13 = r3 * r11;
    r13 = fma(r7, r13, r9);
    r13 = fma(r3, r13, r1 * r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step,
        6 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r6,
        r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        6 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r14,
        r15);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        6 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        global_thread_idx,
        r16,
        r17);
    r18 = r9 * r17;
    r18 = fma(r7, r18, r15);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step,
        0 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r15,
        r19);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        0 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r20,
        r21);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        0 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        global_thread_idx,
        r22,
        r23);
    r24 = r15 * r22;
    r24 = fma(r7, r24, r20);
    r20 = r0 * r4;
    r20 = fma(r7, r20, r2);
    r2 = r6 * r16;
    r2 = fma(r7, r2, r14);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step,
        4 * ThinPrismFisheyeFocalAndExtra_step_num_alloc,
        global_thread_idx,
        r14,
        r25);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_njtr,
        4 * ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
        global_thread_idx,
        r26,
        r27);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_precond_diag,
        4 * ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
        global_thread_idx,
        r28,
        r29);
    r30 = r14 * r28;
    r30 = fma(r7, r30, r26);
    r26 = r25 * r29;
    r26 = fma(r7, r26, r27);
    r27 = r8 * r12;
    r27 = fma(r7, r27, r10);
    r10 = r19 * r23;
    r10 = fma(r7, r10, r21);
    r13 = fma(r9, r18, r13);
    r13 = fma(r15, r24, r13);
    r13 = fma(r0, r20, r13);
    r13 = fma(r6, r2, r13);
    r13 = fma(r14, r30, r13);
    r13 = fma(r25, r26, r13);
    r13 = fma(r8, r27, r13);
    r13 = fma(r19, r10, r13);
  };
  SumStore<double>(out_ThinPrismFisheyeFocalAndExtra_pred_dec_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r13);
  SumFlushFinal<double>(out_ThinPrismFisheyeFocalAndExtra_pred_dec_local,
                        out_ThinPrismFisheyeFocalAndExtra_pred_dec,
                        1);
}

void ThinPrismFisheyeFocalAndExtraPredDecreaseTimesTwo(
    double* ThinPrismFisheyeFocalAndExtra_step,
    unsigned int ThinPrismFisheyeFocalAndExtra_step_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_precond_diag,
    unsigned int ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyeFocalAndExtra_njtr,
    unsigned int ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
    double* const out_ThinPrismFisheyeFocalAndExtra_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraPredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeFocalAndExtra_step,
      ThinPrismFisheyeFocalAndExtra_step_num_alloc,
      ThinPrismFisheyeFocalAndExtra_precond_diag,
      ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
      diag,
      ThinPrismFisheyeFocalAndExtra_njtr,
      ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
      out_ThinPrismFisheyeFocalAndExtra_pred_dec,
      problem_size);
}

}  // namespace caspar