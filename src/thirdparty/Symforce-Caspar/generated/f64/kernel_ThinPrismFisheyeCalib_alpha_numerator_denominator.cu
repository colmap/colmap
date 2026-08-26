#include "kernel_ThinPrismFisheyeCalib_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeCalibAlphaNumeratorDenominatorKernel(
        double* ThinPrismFisheyeCalib_p_kp1,
        unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
        double* ThinPrismFisheyeCalib_r_k,
        unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
        double* ThinPrismFisheyeCalib_w,
        unsigned int ThinPrismFisheyeCalib_w_num_alloc,
        double* const ThinPrismFisheyeCalib_total_ag,
        double* const ThinPrismFisheyeCalib_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double ThinPrismFisheyeCalib_total_ag_local[1];

  __shared__ double ThinPrismFisheyeCalib_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        10 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        10 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r3 = fma(r1, r3, r0 * r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        2 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r2,
        r4);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        2 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r5,
        r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        6 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r7,
        r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        6 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        4 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r11,
        r12);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        4 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r13,
        r14);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        0 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r15,
        r16);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        0 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r17,
        r18);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p_kp1,
        8 * ThinPrismFisheyeCalib_p_kp1_num_alloc,
        global_thread_idx,
        r19,
        r20);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        8 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r21,
        r22);
    r3 = fma(r2, r5, r3);
    r3 = fma(r8, r10, r3);
    r3 = fma(r12, r14, r3);
    r3 = fma(r4, r6, r3);
    r3 = fma(r16, r18, r3);
    r3 = fma(r7, r9, r3);
    r3 = fma(r19, r21, r3);
    r3 = fma(r11, r13, r3);
    r3 = fma(r15, r17, r3);
    r3 = fma(r20, r22, r3);
  };
  SumStore<double>(ThinPrismFisheyeCalib_total_ag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        4 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r3,
        r22);
    r3 = fma(r11, r3, r12 * r22);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        2 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r11,
        r22);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        0 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r12,
        r17);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        6 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r13,
        r21);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        8 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r9,
        r18);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_w,
        10 * ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r6,
        r14);
    r3 = fma(r4, r22, r3);
    r3 = fma(r16, r17, r3);
    r3 = fma(r7, r13, r3);
    r3 = fma(r19, r9, r3);
    r3 = fma(r2, r11, r3);
    r3 = fma(r8, r21, r3);
    r3 = fma(r15, r12, r3);
    r3 = fma(r20, r18, r3);
    r3 = fma(r1, r14, r3);
    r3 = fma(r0, r6, r3);
  };
  SumStore<double>(ThinPrismFisheyeCalib_total_ac_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  SumFlushFinal<double>(
      ThinPrismFisheyeCalib_total_ag_local, ThinPrismFisheyeCalib_total_ag, 1);
  SumFlushFinal<double>(
      ThinPrismFisheyeCalib_total_ac_local, ThinPrismFisheyeCalib_total_ac, 1);
}

void ThinPrismFisheyeCalibAlphaNumeratorDenominator(
    double* ThinPrismFisheyeCalib_p_kp1,
    unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
    double* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    double* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    double* const ThinPrismFisheyeCalib_total_ag,
    double* const ThinPrismFisheyeCalib_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_p_kp1,
      ThinPrismFisheyeCalib_p_kp1_num_alloc,
      ThinPrismFisheyeCalib_r_k,
      ThinPrismFisheyeCalib_r_k_num_alloc,
      ThinPrismFisheyeCalib_w,
      ThinPrismFisheyeCalib_w_num_alloc,
      ThinPrismFisheyeCalib_total_ag,
      ThinPrismFisheyeCalib_total_ac,
      problem_size);
}

}  // namespace caspar