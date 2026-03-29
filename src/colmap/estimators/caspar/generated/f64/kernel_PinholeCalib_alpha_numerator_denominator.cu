#include "kernel_PinholeCalib_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeCalib_alpha_numerator_denominator_kernel(
        double* PinholeCalib_p_kp1,
        unsigned int PinholeCalib_p_kp1_num_alloc,
        double* PinholeCalib_r_k,
        unsigned int PinholeCalib_r_k_num_alloc,
        double* PinholeCalib_w,
        unsigned int PinholeCalib_w_num_alloc,
        double* const PinholeCalib_total_ag,
        double* const PinholeCalib_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double PinholeCalib_total_ag_local[1];

  __shared__ double PinholeCalib_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholeCalib_p_kp1,
                                              2 * PinholeCalib_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(PinholeCalib_r_k,
                                              2 * PinholeCalib_r_k_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    read_idx_2<1024, double, double, double2>(PinholeCalib_p_kp1,
                                              0 * PinholeCalib_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r4,
                                              r5);
    read_idx_2<1024, double, double, double2>(PinholeCalib_r_k,
                                              0 * PinholeCalib_r_k_num_alloc,
                                              global_thread_idx,
                                              r6,
                                              r7);
    r7 = fma(r5, r7, r1 * r3);
    r7 = fma(r0, r2, r7);
    r7 = fma(r4, r6, r7);
  };
  sum_store<double>(PinholeCalib_total_ag_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r7);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholeCalib_w,
                                              2 * PinholeCalib_w_num_alloc,
                                              global_thread_idx,
                                              r7,
                                              r6);
    read_idx_2<1024, double, double, double2>(PinholeCalib_w,
                                              0 * PinholeCalib_w_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    r3 = fma(r5, r3, r0 * r7);
    r3 = fma(r4, r2, r3);
    r3 = fma(r1, r6, r3);
  };
  sum_store<double>(PinholeCalib_total_ac_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r3);
  sum_flush_final<double>(
      PinholeCalib_total_ag_local, PinholeCalib_total_ag, 1);
  sum_flush_final<double>(
      PinholeCalib_total_ac_local, PinholeCalib_total_ac, 1);
}

void PinholeCalib_alpha_numerator_denominator(
    double* PinholeCalib_p_kp1,
    unsigned int PinholeCalib_p_kp1_num_alloc,
    double* PinholeCalib_r_k,
    unsigned int PinholeCalib_r_k_num_alloc,
    double* PinholeCalib_w,
    unsigned int PinholeCalib_w_num_alloc,
    double* const PinholeCalib_total_ag,
    double* const PinholeCalib_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeCalib_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      PinholeCalib_p_kp1,
      PinholeCalib_p_kp1_num_alloc,
      PinholeCalib_r_k,
      PinholeCalib_r_k_num_alloc,
      PinholeCalib_w,
      PinholeCalib_w_num_alloc,
      PinholeCalib_total_ag,
      PinholeCalib_total_ac,
      problem_size);
}

}  // namespace caspar