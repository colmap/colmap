#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ThinPrismFisheyePose_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyePoseAlphaDenominatorOrBetaNumeratorKernel(
        double *ThinPrismFisheyePose_p_kp1,
        unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
        double *ThinPrismFisheyePose_w,
        unsigned int ThinPrismFisheyePose_w_num_alloc,
        double *const ThinPrismFisheyePose_out, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double ThinPrismFisheyePose_out_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_p_kp1, 2 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_w, 2 * ThinPrismFisheyePose_w_num_alloc,
        global_thread_idx, r2, r3);
    r3 = fma(r1, r3, r0 * r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_p_kp1, 0 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx, r1, r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_w, 0 * ThinPrismFisheyePose_w_num_alloc,
        global_thread_idx, r0, r4);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_p_kp1, 4 * ThinPrismFisheyePose_p_kp1_num_alloc,
        global_thread_idx, r5, r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyePose_w, 4 * ThinPrismFisheyePose_w_num_alloc,
        global_thread_idx, r7, r8);
    r3 = fma(r2, r4, r3);
    r3 = fma(r1, r0, r3);
    r3 = fma(r6, r8, r3);
    r3 = fma(r5, r7, r3);
  };
  SumStore<double>(ThinPrismFisheyePose_out_local, (double *)inout_shared, 0,
                   global_thread_idx < problem_size, r3);
  SumFlushFinal<double>(ThinPrismFisheyePose_out_local,
                        ThinPrismFisheyePose_out, 1);
}

void ThinPrismFisheyePoseAlphaDenominatorOrBetaNumerator(
    double *ThinPrismFisheyePose_p_kp1,
    unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
    double *ThinPrismFisheyePose_w,
    unsigned int ThinPrismFisheyePose_w_num_alloc,
    double *const ThinPrismFisheyePose_out, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyePoseAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyePose_p_kp1, ThinPrismFisheyePose_p_kp1_num_alloc,
      ThinPrismFisheyePose_w, ThinPrismFisheyePose_w_num_alloc,
      ThinPrismFisheyePose_out, problem_size);
}

} // namespace caspar