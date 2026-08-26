#include "kernel_ThinPrismFisheyeFocalAndExtra_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraAlphaNumeratorDenominatorKernel(
        double* ThinPrismFisheyeFocalAndExtra_p_kp1,
        unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        double* ThinPrismFisheyeFocalAndExtra_r_k,
        unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        double* ThinPrismFisheyeFocalAndExtra_w,
        unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        double* const ThinPrismFisheyeFocalAndExtra_total_ag,
        double* const ThinPrismFisheyeFocalAndExtra_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double ThinPrismFisheyeFocalAndExtra_total_ag_local[1];

  __shared__ double ThinPrismFisheyeFocalAndExtra_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        8 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        8 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        4 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        4 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r6,
        r7);
    r6 = fma(r4, r6, r1 * r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        0 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r3,
        r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        0 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        2 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r11,
        r12);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        2 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r13,
        r14);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        6 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r15,
        r16);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        6 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r17,
        r18);
    r6 = fma(r8, r10, r6);
    r6 = fma(r11, r13, r6);
    r6 = fma(r16, r18, r6);
    r6 = fma(r5, r7, r6);
    r6 = fma(r0, r2, r6);
    r6 = fma(r15, r17, r6);
    r6 = fma(r12, r14, r6);
    r6 = fma(r3, r9, r6);
  };
  SumStore<double>(ThinPrismFisheyeFocalAndExtra_total_ag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r6);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        0 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r6,
        r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        6 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r14,
        r17);
    r17 = fma(r16, r17, r3 * r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        4 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r16,
        r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        2 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        8 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r7,
        r18);
    r17 = fma(r4, r16, r17);
    r17 = fma(r11, r3, r17);
    r17 = fma(r12, r2, r17);
    r17 = fma(r0, r7, r17);
    r17 = fma(r1, r18, r17);
    r17 = fma(r5, r6, r17);
    r17 = fma(r8, r9, r17);
    r17 = fma(r15, r14, r17);
  };
  SumStore<double>(ThinPrismFisheyeFocalAndExtra_total_ac_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r17);
  SumFlushFinal<double>(ThinPrismFisheyeFocalAndExtra_total_ag_local,
                        ThinPrismFisheyeFocalAndExtra_total_ag,
                        1);
  SumFlushFinal<double>(ThinPrismFisheyeFocalAndExtra_total_ac_local,
                        ThinPrismFisheyeFocalAndExtra_total_ac,
                        1);
}

void ThinPrismFisheyeFocalAndExtraAlphaNumeratorDenominator(
    double* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    double* const ThinPrismFisheyeFocalAndExtra_total_ag,
    double* const ThinPrismFisheyeFocalAndExtra_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraAlphaNumeratorDenominatorKernel<<<n_blocks,
                                                                 1024>>>(
      ThinPrismFisheyeFocalAndExtra_p_kp1,
      ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
      ThinPrismFisheyeFocalAndExtra_r_k,
      ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
      ThinPrismFisheyeFocalAndExtra_w,
      ThinPrismFisheyeFocalAndExtra_w_num_alloc,
      ThinPrismFisheyeFocalAndExtra_total_ag,
      ThinPrismFisheyeFocalAndExtra_total_ac,
      problem_size);
}

}  // namespace caspar