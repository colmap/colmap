#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVCalib_pred_decrease_times_two.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpenCVCalibPredDecreaseTimesTwoKernel(
        float *OpenCVCalib_step, unsigned int OpenCVCalib_step_num_alloc,
        float *OpenCVCalib_precond_diag,
        unsigned int OpenCVCalib_precond_diag_num_alloc,
        const float *const diag, float *OpenCVCalib_njtr,
        unsigned int OpenCVCalib_njtr_num_alloc,
        float *const out_OpenCVCalib_pred_dec, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_OpenCVCalib_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_step,
                                         0 * OpenCVCalib_step_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_njtr,
                                         0 * OpenCVCalib_njtr_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_precond_diag,
                                         0 * OpenCVCalib_precond_diag_num_alloc,
                                         global_thread_idx, r8, r9, r10, r11);
    r12 = r0 * r8;
  };
  LoadUnique<1, float, float>(diag, 0, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float *)inout_shared, 0, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r13, r12, r4);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_step,
                                         4 * OpenCVCalib_step_num_alloc,
                                         global_thread_idx, r4, r14, r15, r16);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_njtr,
                                         4 * OpenCVCalib_njtr_num_alloc,
                                         global_thread_idx, r17, r18, r19, r20);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_precond_diag,
                                         4 * OpenCVCalib_precond_diag_num_alloc,
                                         global_thread_idx, r21, r22, r23, r24);
    r25 = r14 * r22;
    r25 = fmaf(r13, r25, r18);
    r25 = fmaf(r14, r25, r0 * r12);
    r12 = r3 * r11;
    r12 = fmaf(r13, r12, r7);
    r7 = r4 * r21;
    r7 = fmaf(r13, r7, r17);
    r17 = r2 * r10;
    r17 = fmaf(r13, r17, r6);
    r6 = r15 * r23;
    r6 = fmaf(r13, r6, r19);
    r19 = r16 * r24;
    r19 = fmaf(r13, r19, r20);
    r20 = r1 * r9;
    r20 = fmaf(r13, r20, r5);
    r25 = fmaf(r3, r12, r25);
    r25 = fmaf(r4, r7, r25);
    r25 = fmaf(r2, r17, r25);
    r25 = fmaf(r15, r6, r25);
    r25 = fmaf(r16, r19, r25);
    r25 = fmaf(r1, r20, r25);
  };
  SumStore<float>(out_OpenCVCalib_pred_dec_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r25);
  SumFlushFinal<float>(out_OpenCVCalib_pred_dec_local, out_OpenCVCalib_pred_dec,
                       1);
}

void OpenCVCalibPredDecreaseTimesTwo(
    float *OpenCVCalib_step, unsigned int OpenCVCalib_step_num_alloc,
    float *OpenCVCalib_precond_diag,
    unsigned int OpenCVCalib_precond_diag_num_alloc, const float *const diag,
    float *OpenCVCalib_njtr, unsigned int OpenCVCalib_njtr_num_alloc,
    float *const out_OpenCVCalib_pred_dec, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVCalibPredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      OpenCVCalib_step, OpenCVCalib_step_num_alloc, OpenCVCalib_precond_diag,
      OpenCVCalib_precond_diag_num_alloc, diag, OpenCVCalib_njtr,
      OpenCVCalib_njtr_num_alloc, out_OpenCVCalib_pred_dec, problem_size);
}

} // namespace caspar