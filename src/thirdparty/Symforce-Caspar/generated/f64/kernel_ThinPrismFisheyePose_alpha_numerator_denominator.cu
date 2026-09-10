#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ThinPrismFisheyePose_alpha_numerator_denominator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyePoseAlphaNumeratorDenominatorKernel(
        double *ThinPrismFisheyePose_p_kp1,
        unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
        double *ThinPrismFisheyePose_r_k,
        unsigned int ThinPrismFisheyePose_r_k_num_alloc,
        double *ThinPrismFisheyePose_w,
        unsigned int ThinPrismFisheyePose_w_num_alloc,
        double *const ThinPrismFisheyePose_total_ag,
        double *const ThinPrismFisheyePose_total_ac, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double ThinPrismFisheyePose_total_ag_local[1];

  __shared__ double ThinPrismFisheyePose_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_p_kp1, 4 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_r_k, 4 * ThinPrismFisheyePose_r_k_num_alloc,
        global_thread_idx, r2, r3);
    r2 = fma(r0, r2, r1 * r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_p_kp1, 2 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx, r3, r4);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_r_k, 2 * ThinPrismFisheyePose_r_k_num_alloc,
        global_thread_idx, r5, r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_p_kp1, 0 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx, r7, r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_r_k, 0 * ThinPrismFisheyePose_r_k_num_alloc,
        global_thread_idx, r9, r10);
    r2 = fma(r4, r6, r2);
    r2 = fma(r8, r10, r2);
    r2 = fma(r3, r5, r2);
    r2 = fma(r7, r9, r2);
  };
  SumStore<double>(ThinPrismFisheyePose_total_ag_local, (double *)inout_shared,
                   0, global_thread_idx < problem_size, r2);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_w, 2 * ThinPrismFisheyePose_w_num_alloc,
        global_thread_idx, r2, r9);
    r9 = fma(r4, r9, r3 * r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_w, 0 * ThinPrismFisheyePose_w_num_alloc,
        global_thread_idx, r4, r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_w, 4 * ThinPrismFisheyePose_w_num_alloc,
        global_thread_idx, r3, r5);
    r9 = fma(r8, r2, r9);
    r9 = fma(r7, r4, r9);
    r9 = fma(r1, r5, r9);
    r9 = fma(r0, r3, r9);
  };
  SumStore<double>(ThinPrismFisheyePose_total_ac_local, (double *)inout_shared,
                   0, global_thread_idx < problem_size, r9);
  SumFlushFinal<double>(ThinPrismFisheyePose_total_ag_local,
                        ThinPrismFisheyePose_total_ag, 1);
  SumFlushFinal<double>(ThinPrismFisheyePose_total_ac_local,
                        ThinPrismFisheyePose_total_ac, 1);
}

void ThinPrismFisheyePoseAlphaNumeratorDenominator(
    double *ThinPrismFisheyePose_p_kp1,
    unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
    double *ThinPrismFisheyePose_r_k,
    unsigned int ThinPrismFisheyePose_r_k_num_alloc,
    double *ThinPrismFisheyePose_w,
    unsigned int ThinPrismFisheyePose_w_num_alloc,
    double *const ThinPrismFisheyePose_total_ag,
    double *const ThinPrismFisheyePose_total_ac, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePoseAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyePose_p_kp1, ThinPrismFisheyePose_p_kp1_num_alloc,
      ThinPrismFisheyePose_r_k, ThinPrismFisheyePose_r_k_num_alloc,
      ThinPrismFisheyePose_w, ThinPrismFisheyePose_w_num_alloc,
      ThinPrismFisheyePose_total_ag, ThinPrismFisheyePose_total_ac,
      problem_size);
}

} // namespace caspar