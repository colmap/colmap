#include "kernel_SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator_kernel(
        double* SimpleRadialExtraCalib_p_kp1,
        unsigned int SimpleRadialExtraCalib_p_kp1_num_alloc,
        double* SimpleRadialExtraCalib_w,
        unsigned int SimpleRadialExtraCalib_w_num_alloc,
        double* const SimpleRadialExtraCalib_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SimpleRadialExtraCalib_out_local[1];

  double r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_p_kp1,
        2 * SimpleRadialExtraCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0);
    read_idx_1<1024, double, double, double>(
        SimpleRadialExtraCalib_w,
        2 * SimpleRadialExtraCalib_w_num_alloc,
        global_thread_idx,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_p_kp1,
        0 * SimpleRadialExtraCalib_p_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialExtraCalib_w,
        0 * SimpleRadialExtraCalib_w_num_alloc,
        global_thread_idx,
        r4,
        r5);
    r4 = fma(r2, r4, r0 * r1);
    r4 = fma(r3, r5, r4);
  };
  sum_store<double>(SimpleRadialExtraCalib_out_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r4);
  sum_flush_final<double>(
      SimpleRadialExtraCalib_out_local, SimpleRadialExtraCalib_out, 1);
}

void SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator(
    double* SimpleRadialExtraCalib_p_kp1,
    unsigned int SimpleRadialExtraCalib_p_kp1_num_alloc,
    double* SimpleRadialExtraCalib_w,
    unsigned int SimpleRadialExtraCalib_w_num_alloc,
    double* const SimpleRadialExtraCalib_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator_kernel<<<n_blocks,
                                                                       1024>>>(
      SimpleRadialExtraCalib_p_kp1,
      SimpleRadialExtraCalib_p_kp1_num_alloc,
      SimpleRadialExtraCalib_w,
      SimpleRadialExtraCalib_w_num_alloc,
      SimpleRadialExtraCalib_out,
      problem_size);
}

}  // namespace caspar