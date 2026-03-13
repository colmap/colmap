#include "kernel_Point_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Point_alpha_numerator_denominator_kernel(double* Point_p_kp1,
                                             unsigned int Point_p_kp1_num_alloc,
                                             double* Point_r_k,
                                             unsigned int Point_r_k_num_alloc,
                                             double* Point_w,
                                             unsigned int Point_w_num_alloc,
                                             double* const Point_total_ag,
                                             double* const Point_total_ac,
                                             size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double Point_total_ag_local[1];

  __shared__ double Point_total_ac_local[1];

  double r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, double, double, double>(
        Point_p_kp1, 2 * Point_p_kp1_num_alloc, global_thread_idx, r0);
    read_idx_1<1024, double, double, double>(
        Point_r_k, 2 * Point_r_k_num_alloc, global_thread_idx, r1);
    read_idx_2<1024, double, double, double2>(
        Point_p_kp1, 0 * Point_p_kp1_num_alloc, global_thread_idx, r2, r3);
    read_idx_2<1024, double, double, double2>(
        Point_r_k, 0 * Point_r_k_num_alloc, global_thread_idx, r4, r5);
    r5 = fma(r3, r5, r0 * r1);
    r5 = fma(r2, r4, r5);
  };
  sum_store<double>(Point_total_ag_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r5);
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        Point_w, 0 * Point_w_num_alloc, global_thread_idx, r5, r4);
    r5 = fma(r2, r5, r3 * r4);
    read_idx_1<1024, double, double, double>(
        Point_w, 2 * Point_w_num_alloc, global_thread_idx, r2);
    r5 = fma(r0, r2, r5);
  };
  sum_store<double>(Point_total_ac_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r5);
  sum_flush_final<double>(Point_total_ag_local, Point_total_ag, 1);
  sum_flush_final<double>(Point_total_ac_local, Point_total_ac, 1);
}

void Point_alpha_numerator_denominator(double* Point_p_kp1,
                                       unsigned int Point_p_kp1_num_alloc,
                                       double* Point_r_k,
                                       unsigned int Point_r_k_num_alloc,
                                       double* Point_w,
                                       unsigned int Point_w_num_alloc,
                                       double* const Point_total_ag,
                                       double* const Point_total_ac,
                                       size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_alpha_numerator_denominator_kernel<<<n_blocks, 1024>>>(
      Point_p_kp1,
      Point_p_kp1_num_alloc,
      Point_r_k,
      Point_r_k_num_alloc,
      Point_w,
      Point_w_num_alloc,
      Point_total_ag,
      Point_total_ac,
      problem_size);
}

}  // namespace caspar