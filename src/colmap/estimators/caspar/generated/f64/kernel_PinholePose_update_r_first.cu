#include "kernel_PinholePose_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholePose_update_r_first_kernel(
    double* PinholePose_r_k,
    unsigned int PinholePose_r_k_num_alloc,
    double* PinholePose_w,
    unsigned int PinholePose_w_num_alloc,
    const double* const negalpha,
    double* out_PinholePose_r_kp1,
    unsigned int out_PinholePose_r_kp1_num_alloc,
    double* const out_PinholePose_r_0_norm2_tot,
    double* const out_PinholePose_r_kp1_norm2_tot,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_PinholePose_r_0_norm2_tot_local[1];

  __shared__ double out_PinholePose_r_kp1_norm2_tot_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholePose_r_k,
                                              0 * PinholePose_r_k_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 0 * PinholePose_w_num_alloc, global_thread_idx, r2, r3);
  };
  load_unique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    write_idx_2<1024, double, double, double2>(
        out_PinholePose_r_kp1,
        0 * out_PinholePose_r_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    read_idx_2<1024, double, double, double2>(PinholePose_r_k,
                                              2 * PinholePose_r_k_num_alloc,
                                              global_thread_idx,
                                              r5,
                                              r6);
    read_idx_2<1024, double, double, double2>(
        PinholePose_w, 2 * PinholePose_w_num_alloc, global_thread_idx, r7, r8);
    r7 = fma(r7, r4, r5);
    r8 = fma(r8, r4, r6);
    write_idx_2<1024, double, double, double2>(
        out_PinholePose_r_kp1,
        2 * out_PinholePose_r_kp1_num_alloc,
        global_thread_idx,
        r7,
        r8);
    read_idx_2<1024, double, double, double2>(PinholePose_r_k,
                                              4 * PinholePose_r_k_num_alloc,
                                              global_thread_idx,
                                              r9,
                                              r10);
    read_idx_2<1024, double, double, double2>(PinholePose_w,
                                              4 * PinholePose_w_num_alloc,
                                              global_thread_idx,
                                              r11,
                                              r12);
    r11 = fma(r11, r4, r9);
    r4 = fma(r12, r4, r10);
    write_idx_2<1024, double, double, double2>(
        out_PinholePose_r_kp1,
        4 * out_PinholePose_r_kp1_num_alloc,
        global_thread_idx,
        r11,
        r4);
    r0 = fma(r0, r0, r1 * r1);
    r0 = fma(r6, r6, r0);
    r0 = fma(r5, r5, r0);
    r0 = fma(r10, r10, r0);
    r0 = fma(r9, r9, r0);
  };
  sum_store<double>(out_PinholePose_r_0_norm2_tot_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r0);
  if (global_thread_idx < problem_size) {
    r8 = fma(r8, r8, r11 * r11);
    r8 = fma(r2, r2, r8);
    r8 = fma(r3, r3, r8);
    r8 = fma(r7, r7, r8);
    r8 = fma(r4, r4, r8);
  };
  sum_store<double>(out_PinholePose_r_kp1_norm2_tot_local,
                    (double*)inout_shared,
                    0,
                    global_thread_idx < problem_size,
                    r8);
  sum_flush_final<double>(
      out_PinholePose_r_0_norm2_tot_local, out_PinholePose_r_0_norm2_tot, 1);
  sum_flush_final<double>(out_PinholePose_r_kp1_norm2_tot_local,
                          out_PinholePose_r_kp1_norm2_tot,
                          1);
}

void PinholePose_update_r_first(double* PinholePose_r_k,
                                unsigned int PinholePose_r_k_num_alloc,
                                double* PinholePose_w,
                                unsigned int PinholePose_w_num_alloc,
                                const double* const negalpha,
                                double* out_PinholePose_r_kp1,
                                unsigned int out_PinholePose_r_kp1_num_alloc,
                                double* const out_PinholePose_r_0_norm2_tot,
                                double* const out_PinholePose_r_kp1_norm2_tot,
                                size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_update_r_first_kernel<<<n_blocks, 1024>>>(
      PinholePose_r_k,
      PinholePose_r_k_num_alloc,
      PinholePose_w,
      PinholePose_w_num_alloc,
      negalpha,
      out_PinholePose_r_kp1,
      out_PinholePose_r_kp1_num_alloc,
      out_PinholePose_r_0_norm2_tot,
      out_PinholePose_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar