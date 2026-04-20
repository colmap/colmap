#include "kernel_Point_update_r.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Point_update_r_kernel(double* Point_r_k,
                          unsigned int Point_r_k_num_alloc,
                          double* Point_w,
                          unsigned int Point_w_num_alloc,
                          const double* const negalpha,
                          double* out_Point_r_kp1,
                          unsigned int out_Point_r_kp1_num_alloc,
                          double* const out_Point_r_kp1_norm2_tot,
                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_Point_r_kp1_norm2_tot_local[1];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(
        Point_r_k, 0 * Point_r_k_num_alloc, global_thread_idx, r0, r1);
    read_idx_2<1024, double, double, double2>(
        Point_w, 0 * Point_w_num_alloc, global_thread_idx, r2, r3);
  };
  load_unique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    write_idx_2<1024, double, double, double2>(out_Point_r_kp1,
                                               0 * out_Point_r_kp1_num_alloc,
                                               global_thread_idx,
                                               r2,
                                               r3);
    read_idx_1<1024, double, double, double>(
        Point_r_k, 2 * Point_r_k_num_alloc, global_thread_idx, r1);
    read_idx_1<1024, double, double, double>(
        Point_w, 2 * Point_w_num_alloc, global_thread_idx, r0);
    r4 = fma(r0, r4, r1);
    write_idx_1<1024, double, double, double>(
        out_Point_r_kp1, 2 * out_Point_r_kp1_num_alloc, global_thread_idx, r4);
    r2 = fma(r2, r2, r4 * r4);
    r2 = fma(r3, r3, r2);
  };
  sum_store<double>(out_Point_r_kp1_norm2_tot_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r2);
  sum_flush_final<double>(
      out_Point_r_kp1_norm2_tot_local, out_Point_r_kp1_norm2_tot, 1);
}

void Point_update_r(double* Point_r_k,
                    unsigned int Point_r_k_num_alloc,
                    double* Point_w,
                    unsigned int Point_w_num_alloc,
                    const double* const negalpha,
                    double* out_Point_r_kp1,
                    unsigned int out_Point_r_kp1_num_alloc,
                    double* const out_Point_r_kp1_norm2_tot,
                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Point_update_r_kernel<<<n_blocks, 1024>>>(Point_r_k,
                                            Point_r_k_num_alloc,
                                            Point_w,
                                            Point_w_num_alloc,
                                            negalpha,
                                            out_Point_r_kp1,
                                            out_Point_r_kp1_num_alloc,
                                            out_Point_r_kp1_norm2_tot,
                                            problem_size);
}

}  // namespace caspar