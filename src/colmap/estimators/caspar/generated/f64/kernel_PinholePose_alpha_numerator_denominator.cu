#include "kernel_PinholePose_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_alpha_numerator_denominator_kernel(
        double* PinholePose_p_kp1,
        unsigned int PinholePose_p_kp1_num_alloc,
        double* PinholePose_r_k,
        unsigned int PinholePose_r_k_num_alloc,
        double* PinholePose_w,
        unsigned int PinholePose_w_num_alloc,
        double* const PinholePose_total_ag,
        double* const PinholePose_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double PinholePose_total_ag_local[1];

  __shared__ double PinholePose_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholePose_p_kp1,
                                              4 * PinholePose_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(PinholePose_r_k,
                                              4 * PinholePose_r_k_num_alloc,
                                              global_thread_idx,
                                              r2,
                                              r3);
    read_idx_2<1024, double, double, double2>(PinholePose_p_kp1,
                                              0 * PinholePose_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r4,
                                              r5);
    read_idx_2<1024, double, double, double2>(PinholePose_r_k,
                                              0 * PinholePose_r_k_num_alloc,
                                              global_thread_idx,
                                              r6,
                                              r7);
    r7 = fma(r5, r7, r1 * r3);
    read_idx_2<1024, double, double, double2>(PinholePose_p_kp1,
                                              2 * PinholePose_p_kp1_num_alloc,
                                              global_thread_idx,
                                              r3,
                                              r8);
    read_idx_2<1024, double, double, double2>(PinholePose_r_k,
                                              2 * PinholePose_r_k_num_alloc,
                                              global_thread_idx,
                                              r9,
                                              r10);
    r7 = fma(r4, r6, r7);
    r7 = fma(r8, r10, r7);
    r7 = fma(r3, r9, r7);
    r7 = fma(r0, r2, r7);
  };
  sum_store<double>(PinholePose_total_ag_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r7);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 2 * PinholePose_w_num_alloc, global_thread_idx, r7, r2);
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 4 * PinholePose_w_num_alloc, global_thread_idx, r9, r10);
    r9 = fma(r0, r9, r8 * r2);
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 0 * PinholePose_w_num_alloc, global_thread_idx, r0, r2);
    r9 = fma(r3, r7, r9);
    r9 = fma(r1, r10, r9);
    r9 = fma(r4, r0, r9);
    r9 = fma(r5, r2, r9);
  };
  sum_store<double>(PinholePose_total_ac_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r9);
  sum_flush_final<double>(PinholePose_total_ag_local, PinholePose_total_ag, 1);
  sum_flush_final<double>(PinholePose_total_ac_local, PinholePose_total_ac, 1);
}

void PinholePose_alpha_numerator_denominator(
    double* PinholePose_p_kp1,
    unsigned int PinholePose_p_kp1_num_alloc,
    double* PinholePose_r_k,
    unsigned int PinholePose_r_k_num_alloc,
    double* PinholePose_w,
    unsigned int PinholePose_w_num_alloc,
    double* const PinholePose_total_ag,
    double* const PinholePose_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      PinholePose_p_kp1,
      PinholePose_p_kp1_num_alloc,
      PinholePose_r_k,
      PinholePose_r_k_num_alloc,
      PinholePose_w,
      PinholePose_w_num_alloc,
      PinholePose_total_ag,
      PinholePose_total_ac,
      problem_size);
}

}  // namespace caspar