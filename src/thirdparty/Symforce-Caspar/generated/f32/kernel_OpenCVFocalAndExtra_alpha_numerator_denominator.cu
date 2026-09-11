#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVFocalAndExtra_alpha_numerator_denominator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpenCVFocalAndExtraAlphaNumeratorDenominatorKernel(
        float *OpenCVFocalAndExtra_p_kp1,
        unsigned int OpenCVFocalAndExtra_p_kp1_num_alloc,
        float *OpenCVFocalAndExtra_r_k,
        unsigned int OpenCVFocalAndExtra_r_k_num_alloc,
        float *OpenCVFocalAndExtra_w,
        unsigned int OpenCVFocalAndExtra_w_num_alloc,
        float *const OpenCVFocalAndExtra_total_ag,
        float *const OpenCVFocalAndExtra_total_ac, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float OpenCVFocalAndExtra_total_ag_local[1];

  __shared__ float OpenCVFocalAndExtra_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        OpenCVFocalAndExtra_p_kp1, 4 * OpenCVFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx, r0, r1);
    ReadIdx2<1024, float, float, float2>(OpenCVFocalAndExtra_r_k,
                                         4 * OpenCVFocalAndExtra_r_k_num_alloc,
                                         global_thread_idx, r2, r3);
    r2 = fmaf(r0, r2, r1 * r3);
    ReadIdx4<1024, float, float, float4>(
        OpenCVFocalAndExtra_p_kp1, 0 * OpenCVFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx, r3, r4, r5, r6);
    ReadIdx4<1024, float, float, float4>(OpenCVFocalAndExtra_r_k,
                                         0 * OpenCVFocalAndExtra_r_k_num_alloc,
                                         global_thread_idx, r7, r8, r9, r10);
    r2 = fmaf(r6, r10, r2);
    r2 = fmaf(r5, r9, r2);
    r2 = fmaf(r3, r7, r2);
    r2 = fmaf(r4, r8, r2);
  };
  SumStore<float>(OpenCVFocalAndExtra_total_ag_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r2);
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(OpenCVFocalAndExtra_w,
                                         0 * OpenCVFocalAndExtra_w_num_alloc,
                                         global_thread_idx, r2, r8, r7, r9);
    r7 = fmaf(r5, r7, r3 * r2);
    ReadIdx2<1024, float, float, float2>(OpenCVFocalAndExtra_w,
                                         4 * OpenCVFocalAndExtra_w_num_alloc,
                                         global_thread_idx, r5, r2);
    r7 = fmaf(r1, r2, r7);
    r7 = fmaf(r0, r5, r7);
    r7 = fmaf(r4, r8, r7);
    r7 = fmaf(r6, r9, r7);
  };
  SumStore<float>(OpenCVFocalAndExtra_total_ac_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r7);
  SumFlushFinal<float>(OpenCVFocalAndExtra_total_ag_local,
                       OpenCVFocalAndExtra_total_ag, 1);
  SumFlushFinal<float>(OpenCVFocalAndExtra_total_ac_local,
                       OpenCVFocalAndExtra_total_ac, 1);
}

void OpenCVFocalAndExtraAlphaNumeratorDenominator(
    float *OpenCVFocalAndExtra_p_kp1,
    unsigned int OpenCVFocalAndExtra_p_kp1_num_alloc,
    float *OpenCVFocalAndExtra_r_k,
    unsigned int OpenCVFocalAndExtra_r_k_num_alloc,
    float *OpenCVFocalAndExtra_w, unsigned int OpenCVFocalAndExtra_w_num_alloc,
    float *const OpenCVFocalAndExtra_total_ag,
    float *const OpenCVFocalAndExtra_total_ac, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVFocalAndExtraAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      OpenCVFocalAndExtra_p_kp1, OpenCVFocalAndExtra_p_kp1_num_alloc,
      OpenCVFocalAndExtra_r_k, OpenCVFocalAndExtra_r_k_num_alloc,
      OpenCVFocalAndExtra_w, OpenCVFocalAndExtra_w_num_alloc,
      OpenCVFocalAndExtra_total_ag, OpenCVFocalAndExtra_total_ac, problem_size);
}

} // namespace caspar