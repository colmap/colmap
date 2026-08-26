#include "kernel_ThinPrismFisheyeCalib_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibAlphaDenominatorOrBetaNumeratorKernel(
        double* ThinPrismFisheyeCalib_p_kp1,
        unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
        double* ThinPrismFisheyeCalib_w,
        unsigned int ThinPrismFisheyeCalib_w_num_alloc,
        double* const ThinPrismFisheyeCalib_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double ThinPrismFisheyeCalib_out_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        4 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        4 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r2 = fma(r0, r2, r1 * r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        2 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        2 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r1,
        r4);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        0 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r5,
        r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        0 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r7,
        r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        6 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        6 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r11,
        r12);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        8 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r13,
        r14);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        8 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r15,
        r16);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        10 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r17,
        r18);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        10 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r19,
        r20);
    r2 = fma(r3, r4, r2);
    r2 = fma(r6, r8, r2);
    r2 = fma(r9, r11, r2);
    r2 = fma(r13, r15, r2);
    r2 = fma(r0, r1, r2);
    r2 = fma(r10, r12, r2);
    r2 = fma(r5, r7, r2);
    r2 = fma(r14, r16, r2);
    r2 = fma(r18, r20, r2);
    r2 = fma(r17, r19, r2);
  };
  SumStore<double>(ThinPrismFisheyeCalib_out_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r2);
  SumFlushFinal<double>(
      ThinPrismFisheyeCalib_out_local, ThinPrismFisheyeCalib_out, 1);
}

void ThinPrismFisheyeCalibAlphaDenominatorOrBetaNumerator(
    double* ThinPrismFisheyeCalib_p_kp1,
    unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
    double* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    double* const ThinPrismFisheyeCalib_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks,
                                                               1024>>>(
      ThinPrismFisheyeCalib_p_kp1,
      ThinPrismFisheyeCalib_p_kp1_num_alloc,
      ThinPrismFisheyeCalib_w,
      ThinPrismFisheyeCalib_w_num_alloc,
      ThinPrismFisheyeCalib_out,
      problem_size);
}

}  // namespace caspar