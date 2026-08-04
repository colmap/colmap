#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVPose_pred_decrease_times_two.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) OpenCVPosePredDecreaseTimesTwoKernel(
    float *OpenCVPose_step, unsigned int OpenCVPose_step_num_alloc,
    float *OpenCVPose_precond_diag,
    unsigned int OpenCVPose_precond_diag_num_alloc, const float *const diag,
    float *OpenCVPose_njtr, unsigned int OpenCVPose_njtr_num_alloc,
    float *const out_OpenCVPose_pred_dec, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_OpenCVPose_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVPose_step,
                                         0 * OpenCVPose_step_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx4<1024, float, float, float4>(OpenCVPose_njtr,
                                         0 * OpenCVPose_njtr_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
    ReadIdx4<1024, float, float, float4>(OpenCVPose_precond_diag,
                                         0 * OpenCVPose_precond_diag_num_alloc,
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
    r4 = r3 * r11;
    r4 = fmaf(r13, r4, r7);
    r4 = fmaf(r3, r4, r0 * r12);
    r12 = r2 * r10;
    r12 = fmaf(r13, r12, r6);
    ReadIdx2<1024, float, float, float2>(OpenCVPose_step,
                                         4 * OpenCVPose_step_num_alloc,
                                         global_thread_idx, r6, r7);
    ReadIdx2<1024, float, float, float2>(OpenCVPose_njtr,
                                         4 * OpenCVPose_njtr_num_alloc,
                                         global_thread_idx, r14, r15);
    ReadIdx2<1024, float, float, float2>(OpenCVPose_precond_diag,
                                         4 * OpenCVPose_precond_diag_num_alloc,
                                         global_thread_idx, r16, r17);
    r18 = r7 * r17;
    r18 = fmaf(r13, r18, r15);
    r15 = r6 * r16;
    r15 = fmaf(r13, r15, r14);
    r14 = r1 * r9;
    r14 = fmaf(r13, r14, r5);
    r4 = fmaf(r2, r12, r4);
    r4 = fmaf(r7, r18, r4);
    r4 = fmaf(r6, r15, r4);
    r4 = fmaf(r1, r14, r4);
  };
  SumStore<float>(out_OpenCVPose_pred_dec_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r4);
  SumFlushFinal<float>(out_OpenCVPose_pred_dec_local, out_OpenCVPose_pred_dec,
                       1);
}

void OpenCVPosePredDecreaseTimesTwo(
    float *OpenCVPose_step, unsigned int OpenCVPose_step_num_alloc,
    float *OpenCVPose_precond_diag,
    unsigned int OpenCVPose_precond_diag_num_alloc, const float *const diag,
    float *OpenCVPose_njtr, unsigned int OpenCVPose_njtr_num_alloc,
    float *const out_OpenCVPose_pred_dec, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVPosePredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      OpenCVPose_step, OpenCVPose_step_num_alloc, OpenCVPose_precond_diag,
      OpenCVPose_precond_diag_num_alloc, diag, OpenCVPose_njtr,
      OpenCVPose_njtr_num_alloc, out_OpenCVPose_pred_dec, problem_size);
}

} // namespace caspar