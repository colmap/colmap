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
        float* ThinPrismFisheyeCalib_step,
        unsigned int ThinPrismFisheyeCalib_step_num_alloc,
        float* ThinPrismFisheyeCalib_precond_diag,
        unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
        const float* const diag,
        float* ThinPrismFisheyeCalib_njtr,
        unsigned int ThinPrismFisheyeCalib_njtr_num_alloc,
        float* const out_ThinPrismFisheyeCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_ThinPrismFisheyeCalib_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_step,
        8 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_njtr,
        8 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_precond_diag,
        8 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r8,
        r9,
        r10,
        r11);
    r12 = r3 * r11;
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r13, r12, r7);
    r7 = r2 * r10;
    r7 = fmaf(r13, r7, r6);
    r7 = fmaf(r2, r7, r3 * r12);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_step,
        4 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r12,
        r6,
        r14,
        r15);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_njtr,
        4 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r16,
        r17,
        r18,
        r19);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_precond_diag,
        4 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r20,
        r21,
        r22,
        r23);
    r24 = r15 * r23;
    r24 = fmaf(r13, r24, r19);
    r19 = r0 * r8;
    r19 = fmaf(r13, r19, r4);
    r4 = r1 * r9;
    r4 = fmaf(r13, r4, r5);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_step,
        0 * ThinPrismFisheyeCalib_step_num_alloc,
        global_thread_idx,
        r5,
        r25,
        r26,
        r27);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_njtr,
        0 * ThinPrismFisheyeCalib_njtr_num_alloc,
        global_thread_idx,
        r28,
        r29,
        r30,
        r31);
    ReadIdx4<1024, float, float, float4>(
        ThinPrismFisheyeCalib_precond_diag,
        0 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r32,
        r33,
        r34,
        r35);
    r36 = r27 * r35;
    r36 = fmaf(r13, r36, r31);
    r31 = r12 * r20;
    r31 = fmaf(r13, r31, r16);
    r16 = r6 * r21;
    r16 = fmaf(r13, r16, r17);
    r17 = r26 * r34;
    r17 = fmaf(r13, r17, r30);
    r30 = r25 * r33;
    r30 = fmaf(r13, r30, r29);
    r29 = r14 * r22;
    r29 = fmaf(r13, r29, r18);
    r18 = r5 * r32;
    r18 = fmaf(r13, r18, r28);
    r7 = fmaf(r15, r24, r7);
    r7 = fmaf(r0, r19, r7);
    r7 = fmaf(r1, r4, r7);
    r7 = fmaf(r27, r36, r7);
    r7 = fmaf(r12, r31, r7);
    r7 = fmaf(r6, r16, r7);
    r7 = fmaf(r26, r17, r7);
    r7 = fmaf(r25, r30, r7);
    r7 = fmaf(r14, r29, r7);
    r7 = fmaf(r5, r18, r7);
  };
  SumStore<float>(out_ThinPrismFisheyeCalib_pred_dec_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r7);
  SumFlushFinal<float>(out_ThinPrismFisheyeCalib_pred_dec_local,
                       out_ThinPrismFisheyeCalib_pred_dec,
                       1);
}

void ThinPrismFisheyeCalibPredDecreaseTimesTwo(
    float* ThinPrismFisheyeCalib_step,
    unsigned int ThinPrismFisheyeCalib_step_num_alloc,
    float* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyeCalib_njtr,
    unsigned int ThinPrismFisheyeCalib_njtr_num_alloc,
    float* const out_ThinPrismFisheyeCalib_pred_dec,
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