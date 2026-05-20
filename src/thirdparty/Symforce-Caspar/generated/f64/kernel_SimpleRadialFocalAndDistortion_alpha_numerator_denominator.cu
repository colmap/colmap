#include "kernel_SimpleRadialFocalAndDistortion_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialFocalAndDistortionAlphaNumeratorDenominatorKernel(
        double* SimpleRadialFocalAndDistortion_p_kp1,
        unsigned int SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
        double* SimpleRadialFocalAndDistortion_r_k,
        unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
        double* SimpleRadialFocalAndDistortion_w,
        unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
        double* const SimpleRadialFocalAndDistortion_total_ag,
        double* const SimpleRadialFocalAndDistortion_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SimpleRadialFocalAndDistortion_total_ag_local[1];

  __shared__ double SimpleRadialFocalAndDistortion_total_ac_local[1];

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        SimpleRadialFocalAndDistortion_p_kp1,
        0 * SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        SimpleRadialFocalAndDistortion_r_k,
        0 * SimpleRadialFocalAndDistortion_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r3 = fma(r1, r3, r0 * r2);
  };
  SumStore<double>(SimpleRadialFocalAndDistortion_total_ag_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        SimpleRadialFocalAndDistortion_w,
        0 * SimpleRadialFocalAndDistortion_w_num_alloc,
        global_thread_idx,
        r3,
        r2);
    r2 = fma(r1, r2, r0 * r3);
  };
  SumStore<double>(SimpleRadialFocalAndDistortion_total_ac_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r2);
  SumFlushFinal<double>(SimpleRadialFocalAndDistortion_total_ag_local,
                        SimpleRadialFocalAndDistortion_total_ag,
                        1);
  SumFlushFinal<double>(SimpleRadialFocalAndDistortion_total_ac_local,
                        SimpleRadialFocalAndDistortion_total_ac,
                        1);
}

void SimpleRadialFocalAndDistortionAlphaNumeratorDenominator(
    double* SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    double* SimpleRadialFocalAndDistortion_r_k,
    unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
    double* SimpleRadialFocalAndDistortion_w,
    unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
    double* const SimpleRadialFocalAndDistortion_total_ag,
    double* const SimpleRadialFocalAndDistortion_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocalAndDistortionAlphaNumeratorDenominatorKernel<<<n_blocks,
                                                                  1024>>>(
      SimpleRadialFocalAndDistortion_p_kp1,
      SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
      SimpleRadialFocalAndDistortion_r_k,
      SimpleRadialFocalAndDistortion_r_k_num_alloc,
      SimpleRadialFocalAndDistortion_w,
      SimpleRadialFocalAndDistortion_w_num_alloc,
      SimpleRadialFocalAndDistortion_total_ag,
      SimpleRadialFocalAndDistortion_total_ac,
      problem_size);
}

}  // namespace caspar