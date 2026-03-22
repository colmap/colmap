#include "kernel_Pose_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Pose_alpha_numerator_denominator_kernel(double* Pose_p_kp1,
                                            unsigned int Pose_p_kp1_num_alloc,
                                            double* Pose_r_k,
                                            unsigned int Pose_r_k_num_alloc,
                                            double* Pose_w,
                                            unsigned int Pose_w_num_alloc,
                                            double* const Pose_total_ag,
                                            double* const Pose_total_ac,
                                            size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double Pose_total_ag_local[1];

  __shared__ double Pose_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        Pose_p_kp1, 4 * Pose_p_kp1_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, double, double, double2>(
        Pose_r_k, 4 * Pose_r_k_num_alloc, global_thread_idx, r2, r3);
    read_idx_2<1024, double, double, double2>(
        Pose_p_kp1, 2 * Pose_p_kp1_num_alloc, global_thread_idx, r4, r5);
    read_idx_2<1024, double, double, double2>(
        Pose_r_k, 2 * Pose_r_k_num_alloc, global_thread_idx, r6, r7);
    r6 = fma(r4, r6, r1 * r3);
    read_idx_2<1024, double, double, double2>(
        Pose_p_kp1, 0 * Pose_p_kp1_num_alloc, global_thread_idx, r3, r8);
    read_idx_2<1024, double, double, double2>(
        Pose_r_k, 0 * Pose_r_k_num_alloc, global_thread_idx, r9, r10);
    r6 = fma(r0, r2, r6);
    r6 = fma(r5, r7, r6);
    r6 = fma(r8, r10, r6);
    r6 = fma(r3, r9, r6);
  };
  sum_store<double>(Pose_total_ag_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r6);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        Pose_w, 0 * Pose_w_num_alloc, global_thread_idx, r6, r9);
    read_idx_2<1024, double, double, double2>(
        Pose_w, 2 * Pose_w_num_alloc, global_thread_idx, r10, r7);
    r10 = fma(r4, r10, r3 * r6);
    read_idx_2<1024, double, double, double2>(
        Pose_w, 4 * Pose_w_num_alloc, global_thread_idx, r4, r6);
    r10 = fma(r8, r9, r10);
    r10 = fma(r1, r6, r10);
    r10 = fma(r0, r4, r10);
    r10 = fma(r5, r7, r10);
  };
  sum_store<double>(Pose_total_ac_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r10);
  sum_flush_final<double>(Pose_total_ag_local, Pose_total_ag, 1);
  sum_flush_final<double>(Pose_total_ac_local, Pose_total_ac, 1);
}

void Pose_alpha_numerator_denominator(double* Pose_p_kp1,
                                      unsigned int Pose_p_kp1_num_alloc,
                                      double* Pose_r_k,
                                      unsigned int Pose_r_k_num_alloc,
                                      double* Pose_w,
                                      unsigned int Pose_w_num_alloc,
                                      double* const Pose_total_ag,
                                      double* const Pose_total_ac,
                                      size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Pose_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      Pose_p_kp1,
      Pose_p_kp1_num_alloc,
      Pose_r_k,
      Pose_r_k_num_alloc,
      Pose_w,
      Pose_w_num_alloc,
      Pose_total_ag,
      Pose_total_ac,
      problem_size);
}

}  // namespace caspar