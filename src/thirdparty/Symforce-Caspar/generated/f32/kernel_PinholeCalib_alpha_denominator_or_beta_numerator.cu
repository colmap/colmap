#include "kernel_PinholeCalib_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeCalibAlphaDenominatorOrBetaNumeratorKernel(
        float* PinholeCalib_p_kp1,
        unsigned int PinholeCalib_p_kp1_num_alloc,
        float* PinholeCalib_w,
        unsigned int PinholeCalib_w_num_alloc,
        float* const PinholeCalib_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float PinholeCalib_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(PinholeCalib_p_kp1,
                                         0 * PinholeCalib_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
    ReadIdx4<1024, float, float, float4>(PinholeCalib_w,
                                         0 * PinholeCalib_w_num_alloc,
                                         global_thread_idx,
                                         r4,
                                         r5,
                                         r6,
                                         r7);
    r5 = fmaf(r1, r5, r2 * r6);
    r5 = fmaf(r0, r4, r5);
    r5 = fmaf(r3, r7, r5);
  };
  SumStore<float>(PinholeCalib_out_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r5);
  SumFlushFinal<float>(PinholeCalib_out_local, PinholeCalib_out, 1);
}

void PinholeCalibAlphaDenominatorOrBetaNumerator(
    float* PinholeCalib_p_kp1,
    unsigned int PinholeCalib_p_kp1_num_alloc,
    float* PinholeCalib_w,
    unsigned int PinholeCalib_w_num_alloc,
    float* const PinholeCalib_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeCalibAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      PinholeCalib_p_kp1,
      PinholeCalib_p_kp1_num_alloc,
      PinholeCalib_w,
      PinholeCalib_w_num_alloc,
      PinholeCalib_out,
      problem_size);
}

}  // namespace caspar