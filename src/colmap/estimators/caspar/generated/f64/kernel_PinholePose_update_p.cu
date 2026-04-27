#include "kernel_PinholePose_update_p.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_update_p_kernel(double* PinholePose_z,
                                unsigned int PinholePose_z_num_alloc,
                                double* PinholePose_p_k,
                                unsigned int PinholePose_p_k_num_alloc,
                                const double* const beta,
                                double* out_PinholePose_p_kp1,
                                unsigned int out_PinholePose_p_kp1_num_alloc,
                                size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, double, double, double2>(PinholePose_p_k,
                                              0 * PinholePose_p_k_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(
        PinholePose_z, 0 * PinholePose_z_num_alloc, global_thread_idx, r2, r3);
  };
  load_unique<1, double, double>(beta, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fma(r0, r4, r2);
    r1 = fma(r1, r4, r3);
    write_idx_2<1024, double, double, double2>(
        out_PinholePose_p_kp1,
        0 * out_PinholePose_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, double, double, double2>(PinholePose_p_k,
                                              2 * PinholePose_p_k_num_alloc,
                                              global_thread_idx,
                                              r1,
                                              r0);
    read_idx_2<1024, double, double, double2>(
        PinholePose_z, 2 * PinholePose_z_num_alloc, global_thread_idx, r3, r2);
    r1 = fma(r1, r4, r3);
    r0 = fma(r0, r4, r2);
    write_idx_2<1024, double, double, double2>(
        out_PinholePose_p_kp1,
        2 * out_PinholePose_p_kp1_num_alloc,
        global_thread_idx,
        r1,
        r0);
    read_idx_2<1024, double, double, double2>(PinholePose_p_k,
                                              4 * PinholePose_p_k_num_alloc,
                                              global_thread_idx,
                                              r0,
                                              r1);
    read_idx_2<1024, double, double, double2>(
        PinholePose_z, 4 * PinholePose_z_num_alloc, global_thread_idx, r2, r3);
    r0 = fma(r0, r4, r2);
    r4 = fma(r1, r4, r3);
    write_idx_2<1024, double, double, double2>(
        out_PinholePose_p_kp1,
        4 * out_PinholePose_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r4);
  };
}

void PinholePose_update_p(double* PinholePose_z,
                          unsigned int PinholePose_z_num_alloc,
                          double* PinholePose_p_k,
                          unsigned int PinholePose_p_k_num_alloc,
                          const double* const beta,
                          double* out_PinholePose_p_kp1,
                          unsigned int out_PinholePose_p_kp1_num_alloc,
                          size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_update_p_kernel<<<n_blocks, 1024>>>(
      PinholePose_z,
      PinholePose_z_num_alloc,
      PinholePose_p_k,
      PinholePose_p_k_num_alloc,
      beta,
      out_PinholePose_p_kp1,
      out_PinholePose_p_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar