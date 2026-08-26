#include "kernel_ThinPrismFisheyeFocalAndExtra_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel(
        double* ThinPrismFisheyeFocalAndExtra_p_kp1,
        unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        double* ThinPrismFisheyeFocalAndExtra_w,
        unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        double* const ThinPrismFisheyeFocalAndExtra_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double ThinPrismFisheyeFocalAndExtra_out_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        0 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        0 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        6 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        6 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r6,
        r7);
    r7 = fma(r5, r7, r0 * r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        4 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r5,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        4 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r0,
        r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        2 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        2 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r11,
        r12);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        8 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r13,
        r14);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        8 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r15,
        r16);
    r7 = fma(r5, r0, r7);
    r7 = fma(r9, r11, r7);
    r7 = fma(r10, r12, r7);
    r7 = fma(r13, r15, r7);
    r7 = fma(r14, r16, r7);
    r7 = fma(r2, r8, r7);
    r7 = fma(r1, r3, r7);
    r7 = fma(r4, r6, r7);
  };
  SumStore<double>(ThinPrismFisheyeFocalAndExtra_out_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r7);
  SumFlushFinal<double>(ThinPrismFisheyeFocalAndExtra_out_local,
                        ThinPrismFisheyeFocalAndExtra_out,
                        1);
}

void ThinPrismFisheyeFocalAndExtraAlphaDenominatorOrBetaNumerator(
    double* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    double* const ThinPrismFisheyeFocalAndExtra_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks,
                                                                       1024>>>(
      ThinPrismFisheyeFocalAndExtra_p_kp1,
      ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
      ThinPrismFisheyeFocalAndExtra_w,
      ThinPrismFisheyeFocalAndExtra_w_num_alloc,
      ThinPrismFisheyeFocalAndExtra_out,
      problem_size);
}

}  // namespace caspar