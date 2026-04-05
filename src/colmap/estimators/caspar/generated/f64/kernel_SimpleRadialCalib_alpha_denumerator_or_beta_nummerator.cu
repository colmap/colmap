#include "kernel_SimpleRadialCalib_alpha_denumerator_or_beta_nummerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialCalib_alpha_denumerator_or_beta_nummerator_kernel(
        double* SimpleRadialCalib_p_kp1,
        unsigned int SimpleRadialCalib_p_kp1_num_alloc,
        double* SimpleRadialCalib_w,
        unsigned int SimpleRadialCalib_w_num_alloc,
        double* const SimpleRadialCalib_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SimpleRadialCalib_out_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_p_kp1,
        0 * SimpleRadialCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(SimpleRadialCalib_w,
                                              0 * SimpleRadialCalib_w_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_p_kp1,
        2 * SimpleRadialCalib_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5);
    read_idx_2<1024, double, double, double2>(SimpleRadialCalib_w,
                                              2 * SimpleRadialCalib_w_num_alloc,
                                              global_thread_idx,
                                              r6,
                                              r7);
    r6 = fma(r4, r6, r1 * r3);
    r6 = fma(r0, r2, r6);
    r6 = fma(r5, r7, r6);
  };
  sum_store<double>(SimpleRadialCalib_out_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r6);
  sum_flush_final<double>(
      SimpleRadialCalib_out_local, SimpleRadialCalib_out, 1);
}

void SimpleRadialCalib_alpha_denumerator_or_beta_nummerator(
    double* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    double* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    double* const SimpleRadialCalib_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialCalib_alpha_denumerator_or_beta_nummerator_kernel<<<n_blocks,
                                                                  1024>>>(
      SimpleRadialCalib_p_kp1,
      SimpleRadialCalib_p_kp1_num_alloc,
      SimpleRadialCalib_w,
      SimpleRadialCalib_w_num_alloc,
      SimpleRadialCalib_out,
      problem_size);
}

}  // namespace caspar