#include "kernel_SimpleRadialCalib_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialCalib_alpha_numerator_denominator_kernel(
        double* SimpleRadialCalib_p_kp1,
        unsigned int SimpleRadialCalib_p_kp1_num_alloc,
        double* SimpleRadialCalib_r_k,
        unsigned int SimpleRadialCalib_r_k_num_alloc,
        double* SimpleRadialCalib_w,
        unsigned int SimpleRadialCalib_w_num_alloc,
        double* const SimpleRadialCalib_total_ag,
        double* const SimpleRadialCalib_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double SimpleRadialCalib_total_ag_local[1];

  __shared__ double SimpleRadialCalib_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_p_kp1,
        2 * SimpleRadialCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_r_k,
        2 * SimpleRadialCalib_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_p_kp1,
        0 * SimpleRadialCalib_p_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5);
    read_idx_2<1024, double, double, double2>(
        SimpleRadialCalib_r_k,
        0 * SimpleRadialCalib_r_k_num_alloc,
        global_thread_idx,
        r6,
        r7);
    r7 = fma(r5, r7, r0 * r2);
    r7 = fma(r1, r3, r7);
    r7 = fma(r4, r6, r7);
  };
  sum_store<double>(SimpleRadialCalib_total_ag_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r7);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(SimpleRadialCalib_w,
                                              0 * SimpleRadialCalib_w_num_alloc,
                                              global_thread_idx,
                                              r7,
                                              r6);
    read_idx_2<1024, double, double, double2>(SimpleRadialCalib_w,
                                              2 * SimpleRadialCalib_w_num_alloc,
                                              global_thread_idx,
                                              r3,
                                              r2);
    r3 = fma(r0, r3, r5 * r6);
    r3 = fma(r4, r7, r3);
    r3 = fma(r1, r2, r3);
  };
  sum_store<double>(SimpleRadialCalib_total_ac_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(
      SimpleRadialCalib_total_ag_local, SimpleRadialCalib_total_ag, 1);
  sum_flush_final<double>(
      SimpleRadialCalib_total_ac_local, SimpleRadialCalib_total_ac, 1);
}

void SimpleRadialCalib_alpha_numerator_denominator(
    double* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    double* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    double* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    double* const SimpleRadialCalib_total_ag,
    double* const SimpleRadialCalib_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialCalib_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      SimpleRadialCalib_p_kp1,
      SimpleRadialCalib_p_kp1_num_alloc,
      SimpleRadialCalib_r_k,
      SimpleRadialCalib_r_k_num_alloc,
      SimpleRadialCalib_w,
      SimpleRadialCalib_w_num_alloc,
      SimpleRadialCalib_total_ag,
      SimpleRadialCalib_total_ac,
      problem_size);
}

}  // namespace caspar