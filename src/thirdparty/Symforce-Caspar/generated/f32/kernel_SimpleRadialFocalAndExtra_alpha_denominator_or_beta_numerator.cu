#include "kernel_SimpleRadialFocalAndExtra_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel(
        float* SimpleRadialFocalAndExtra_p_kp1,
        unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        float* SimpleRadialFocalAndExtra_w,
        unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
        float* const SimpleRadialFocalAndExtra_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float SimpleRadialFocalAndExtra_out_local[1];

  float r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        SimpleRadialFocalAndExtra_p_kp1,
        0 * SimpleRadialFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, float, float, float2>(
        SimpleRadialFocalAndExtra_w,
        0 * SimpleRadialFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r3 = fmaf(r1, r3, r0 * r2);
  };
  SumStore<float>(SimpleRadialFocalAndExtra_out_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  SumFlushFinal<float>(
      SimpleRadialFocalAndExtra_out_local, SimpleRadialFocalAndExtra_out, 1);
}

void SimpleRadialFocalAndExtraAlphaDenominatorOrBetaNumerator(
    float* SimpleRadialFocalAndExtra_p_kp1,
    unsigned int SimpleRadialFocalAndExtra_p_kp1_num_alloc,
    float* SimpleRadialFocalAndExtra_w,
    unsigned int SimpleRadialFocalAndExtra_w_num_alloc,
    float* const SimpleRadialFocalAndExtra_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks,
                                                                   1024>>>(
      SimpleRadialFocalAndExtra_p_kp1,
      SimpleRadialFocalAndExtra_p_kp1_num_alloc,
      SimpleRadialFocalAndExtra_w,
      SimpleRadialFocalAndExtra_w_num_alloc,
      SimpleRadialFocalAndExtra_out,
      problem_size);
}

}  // namespace caspar