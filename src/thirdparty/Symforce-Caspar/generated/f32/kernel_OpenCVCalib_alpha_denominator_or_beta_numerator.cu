#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVCalib_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpenCVCalibAlphaDenominatorOrBetaNumeratorKernel(
        float *OpenCVCalib_p_kp1, unsigned int OpenCVCalib_p_kp1_num_alloc,
        float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
        float *const OpenCVCalib_out, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float OpenCVCalib_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_p_kp1,
                                         4 * OpenCVCalib_p_kp1_num_alloc,
                                         global_thread_idx, r0, r1, r2, r3);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_w,
                                         4 * OpenCVCalib_w_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_p_kp1,
                                         0 * OpenCVCalib_p_kp1_num_alloc,
                                         global_thread_idx, r8, r9, r10, r11);
    ReadIdx4<1024, float, float, float4>(OpenCVCalib_w,
                                         0 * OpenCVCalib_w_num_alloc,
                                         global_thread_idx, r12, r13, r14, r15);
    r12 = fmaf(r8, r12, r1 * r5);
    r12 = fmaf(r3, r7, r12);
    r12 = fmaf(r11, r15, r12);
    r12 = fmaf(r2, r6, r12);
    r12 = fmaf(r10, r14, r12);
    r12 = fmaf(r0, r4, r12);
    r12 = fmaf(r9, r13, r12);
  };
  SumStore<float>(OpenCVCalib_out_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r12);
  SumFlushFinal<float>(OpenCVCalib_out_local, OpenCVCalib_out, 1);
}

void OpenCVCalibAlphaDenominatorOrBetaNumerator(
    float *OpenCVCalib_p_kp1, unsigned int OpenCVCalib_p_kp1_num_alloc,
    float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
    float *const OpenCVCalib_out, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVCalibAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      OpenCVCalib_p_kp1, OpenCVCalib_p_kp1_num_alloc, OpenCVCalib_w,
      OpenCVCalib_w_num_alloc, OpenCVCalib_out, problem_size);
}

} // namespace caspar