#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVPrincipalPoint_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpenCVPrincipalPointAlphaDenominatorOrBetaNumeratorKernel(
        float *OpenCVPrincipalPoint_p_kp1,
        unsigned int OpenCVPrincipalPoint_p_kp1_num_alloc,
        float *OpenCVPrincipalPoint_w,
        unsigned int OpenCVPrincipalPoint_w_num_alloc,
        float *const OpenCVPrincipalPoint_out, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float OpenCVPrincipalPoint_out_local[1];

  float r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        OpenCVPrincipalPoint_p_kp1, 0 * OpenCVPrincipalPoint_p_kp1_num_alloc,
        global_thread_idx, r0, r1);
    ReadIdx2<1024, float, float, float2>(OpenCVPrincipalPoint_w,
                                         0 * OpenCVPrincipalPoint_w_num_alloc,
                                         global_thread_idx, r2, r3);
    r2 = fmaf(r0, r2, r1 * r3);
  };
  SumStore<float>(OpenCVPrincipalPoint_out_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r2);
  SumFlushFinal<float>(OpenCVPrincipalPoint_out_local, OpenCVPrincipalPoint_out,
                       1);
}

void OpenCVPrincipalPointAlphaDenominatorOrBetaNumerator(
    float *OpenCVPrincipalPoint_p_kp1,
    unsigned int OpenCVPrincipalPoint_p_kp1_num_alloc,
    float *OpenCVPrincipalPoint_w,
    unsigned int OpenCVPrincipalPoint_w_num_alloc,
    float *const OpenCVPrincipalPoint_out, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVPrincipalPointAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      OpenCVPrincipalPoint_p_kp1, OpenCVPrincipalPoint_p_kp1_num_alloc,
      OpenCVPrincipalPoint_w, OpenCVPrincipalPoint_w_num_alloc,
      OpenCVPrincipalPoint_out, problem_size);
}

} // namespace caspar