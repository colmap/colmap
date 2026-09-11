#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_OpenCVFocalAndExtra_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpenCVFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel(
        float *OpenCVFocalAndExtra_p_kp1,
        unsigned int OpenCVFocalAndExtra_p_kp1_num_alloc,
        float *OpenCVFocalAndExtra_w,
        unsigned int OpenCVFocalAndExtra_w_num_alloc,
        float *const OpenCVFocalAndExtra_out, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float OpenCVFocalAndExtra_out_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        OpenCVFocalAndExtra_p_kp1, 0 * OpenCVFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx, r0, r1, r2, r3);
    ReadIdx4<1024, float, float, float4>(OpenCVFocalAndExtra_w,
                                         0 * OpenCVFocalAndExtra_w_num_alloc,
                                         global_thread_idx, r4, r5, r6, r7);
    r6 = fmaf(r2, r6, r0 * r4);
    ReadIdx2<1024, float, float, float2>(
        OpenCVFocalAndExtra_p_kp1, 4 * OpenCVFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx, r2, r4);
    ReadIdx2<1024, float, float, float2>(OpenCVFocalAndExtra_w,
                                         4 * OpenCVFocalAndExtra_w_num_alloc,
                                         global_thread_idx, r0, r8);
    r6 = fmaf(r4, r8, r6);
    r6 = fmaf(r2, r0, r6);
    r6 = fmaf(r1, r5, r6);
    r6 = fmaf(r3, r7, r6);
  };
  SumStore<float>(OpenCVFocalAndExtra_out_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r6);
  SumFlushFinal<float>(OpenCVFocalAndExtra_out_local, OpenCVFocalAndExtra_out,
                       1);
}

void OpenCVFocalAndExtraAlphaDenominatorOrBetaNumerator(
    float *OpenCVFocalAndExtra_p_kp1,
    unsigned int OpenCVFocalAndExtra_p_kp1_num_alloc,
    float *OpenCVFocalAndExtra_w, unsigned int OpenCVFocalAndExtra_w_num_alloc,
    float *const OpenCVFocalAndExtra_out, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpenCVFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      OpenCVFocalAndExtra_p_kp1, OpenCVFocalAndExtra_p_kp1_num_alloc,
      OpenCVFocalAndExtra_w, OpenCVFocalAndExtra_w_num_alloc,
      OpenCVFocalAndExtra_out, problem_size);
}

} // namespace caspar