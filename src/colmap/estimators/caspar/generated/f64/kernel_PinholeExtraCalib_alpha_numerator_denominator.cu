#include "kernel_PinholeExtraCalib_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeExtraCalib_alpha_numerator_denominator_kernel(
        double* PinholeExtraCalib_p_kp1,
        unsigned int PinholeExtraCalib_p_kp1_num_alloc,
        double* PinholeExtraCalib_r_k,
        unsigned int PinholeExtraCalib_r_k_num_alloc,
        double* PinholeExtraCalib_w,
        unsigned int PinholeExtraCalib_w_num_alloc,
        double* const PinholeExtraCalib_total_ag,
        double* const PinholeExtraCalib_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double PinholeExtraCalib_total_ag_local[1];

  __shared__ double PinholeExtraCalib_total_ac_local[1];

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholeExtraCalib_p_kp1,
        0 * PinholeExtraCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(
        PinholeExtraCalib_r_k,
        0 * PinholeExtraCalib_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r2 = fma(r0, r2, r1 * r3);
  };
  sum_store<double>(PinholeExtraCalib_total_ag_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholeExtraCalib_w,
                                              0 * PinholeExtraCalib_w_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    r2 = fma(r0, r2, r1 * r3);
  };
  sum_store<double>(PinholeExtraCalib_total_ac_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  sum_flush_final<double>(
      PinholeExtraCalib_total_ag_local, PinholeExtraCalib_total_ag, 1);
  sum_flush_final<double>(
      PinholeExtraCalib_total_ac_local, PinholeExtraCalib_total_ac, 1);
}

void PinholeExtraCalib_alpha_numerator_denominator(
    double* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    double* PinholeExtraCalib_r_k,
    unsigned int PinholeExtraCalib_r_k_num_alloc,
    double* PinholeExtraCalib_w,
    unsigned int PinholeExtraCalib_w_num_alloc,
    double* const PinholeExtraCalib_total_ag,
    double* const PinholeExtraCalib_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeExtraCalib_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      PinholeExtraCalib_p_kp1,
      PinholeExtraCalib_p_kp1_num_alloc,
      PinholeExtraCalib_r_k,
      PinholeExtraCalib_r_k_num_alloc,
      PinholeExtraCalib_w,
      PinholeExtraCalib_w_num_alloc,
      PinholeExtraCalib_total_ag,
      PinholeExtraCalib_total_ac,
      problem_size);
}

}  // namespace caspar