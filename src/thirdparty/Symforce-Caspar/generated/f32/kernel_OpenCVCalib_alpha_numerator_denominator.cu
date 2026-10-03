#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVCalib_alpha_numerator_denominator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpenCVCalibAlphaNumeratorDenominatorKernel(
        float *OpenCVCalib_p_kp1, unsigned int OpenCVCalib_p_kp1_num_alloc,
        float *OpenCVCalib_r_k, unsigned int OpenCVCalib_r_k_num_alloc,
        float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
        float *const OpenCVCalib_total_ag, float *const OpenCVCalib_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float OpenCVCalib_total_ag_local[1];

  __shared__ float OpenCVCalib_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_p_kp1,
                                         0 * OpenCVCalib_p_kp1_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_r_k,
                                         0 * OpenCVCalib_r_k_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_p_kp1,
                                         4 * OpenCVCalib_p_kp1_num_alloc,
                                         global_thread_idx, r8, r9, r10, r11);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_r_k,
                                         4 * OpenCVCalib_r_k_num_alloc,
                                         global_thread_idx, r12, r13, r14, r15);
    r12 = fmaf(r8, r12, r0 * r4);
    r12 = fmaf(r1, r5, r12);
    r12 = fmaf(r3, r7, r12);
    r12 = fmaf(r2, r6, r12);
    r12 = fmaf(r9, r13, r12);
    r12 = fmaf(r11, r15, r12);
    r12 = fmaf(r10, r14, r12);
  };
  SumStore<float>(OpenCVCalib_total_ag_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r12);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_w,
                                         4 * OpenCVCalib_w_num_alloc,
                                         global_thread_idx, r12, r14, r15, r13);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_w,
                                         0 * OpenCVCalib_w_num_alloc,
                                         global_thread_idx, r6, r7, r5, r4);
    r6 = fmaf(r0, r6, r9 * r14);
    r6 = fmaf(r11, r13, r6);
    r6 = fmaf(r3, r4, r6);
    r6 = fmaf(r10, r15, r6);
    r6 = fmaf(r2, r5, r6);
    r6 = fmaf(r8, r12, r6);
    r6 = fmaf(r1, r7, r6);
  };
  SumStore<float>(OpenCVCalib_total_ac_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r6);
  SumFlushFinal<float>(OpenCVCalib_total_ag_local, OpenCVCalib_total_ag, 1);
  SumFlushFinal<float>(OpenCVCalib_total_ac_local, OpenCVCalib_total_ac, 1);
}

void OpenCVCalibAlphaNumeratorDenominator(
    float *OpenCVCalib_p_kp1, unsigned int OpenCVCalib_p_kp1_num_alloc,
    float *OpenCVCalib_r_k, unsigned int OpenCVCalib_r_k_num_alloc,
    float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
    float *const OpenCVCalib_total_ag, float *const OpenCVCalib_total_ac,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVCalibAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      OpenCVCalib_p_kp1, OpenCVCalib_p_kp1_num_alloc, OpenCVCalib_r_k,
      OpenCVCalib_r_k_num_alloc, OpenCVCalib_w, OpenCVCalib_w_num_alloc,
      OpenCVCalib_total_ag, OpenCVCalib_total_ac, problem_size);
}

} // namespace caspar