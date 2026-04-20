#include "kernel_PinholePrincipalPoint_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePrincipalPoint_update_r_first_kernel(
        float* PinholePrincipalPoint_r_k,
        unsigned int PinholePrincipalPoint_r_k_num_alloc,
        float* PinholePrincipalPoint_w,
        unsigned int PinholePrincipalPoint_w_num_alloc,
        const float* const negalpha,
        float* out_PinholePrincipalPoint_r_kp1,
        unsigned int out_PinholePrincipalPoint_r_kp1_num_alloc,
        float* const out_PinholePrincipalPoint_r_0_norm2_tot,
        float* const out_PinholePrincipalPoint_r_kp1_norm2_tot,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_PinholePrincipalPoint_r_0_norm2_tot_local[1];

  __shared__ float out_PinholePrincipalPoint_r_kp1_norm2_tot_local[1];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        PinholePrincipalPoint_r_k,
        0 * PinholePrincipalPoint_r_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    read_idx_2<1024, float, float, float2>(
        PinholePrincipalPoint_w,
        0 * PinholePrincipalPoint_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
  load_unique<1, float, float>(negalpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r2, r4, r0);
    r4 = fmaf(r3, r4, r1);
    write_idx_2<1024, float, float, float2>(
        out_PinholePrincipalPoint_r_kp1,
        0 * out_PinholePrincipalPoint_r_kp1_num_alloc,
        global_thread_idx,
        r2,
        r4);
    r0 = fmaf(r0, r0, r1 * r1);
  };
  sum_store<float>(out_PinholePrincipalPoint_r_0_norm2_tot_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r0);
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r2, r2, r4 * r4);
  };
  sum_store<float>(out_PinholePrincipalPoint_r_kp1_norm2_tot_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r2);
  sum_flush_final<float>(out_PinholePrincipalPoint_r_0_norm2_tot_local,
                         out_PinholePrincipalPoint_r_0_norm2_tot,
                         1);
  sum_flush_final<float>(out_PinholePrincipalPoint_r_kp1_norm2_tot_local,
                         out_PinholePrincipalPoint_r_kp1_norm2_tot,
                         1);
}

void PinholePrincipalPoint_update_r_first(
    float* PinholePrincipalPoint_r_k,
    unsigned int PinholePrincipalPoint_r_k_num_alloc,
    float* PinholePrincipalPoint_w,
    unsigned int PinholePrincipalPoint_w_num_alloc,
    const float* const negalpha,
    float* out_PinholePrincipalPoint_r_kp1,
    unsigned int out_PinholePrincipalPoint_r_kp1_num_alloc,
    float* const out_PinholePrincipalPoint_r_0_norm2_tot,
    float* const out_PinholePrincipalPoint_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePrincipalPoint_update_r_first_kernel<<<n_blocks, 1024>>>(
      PinholePrincipalPoint_r_k,
      PinholePrincipalPoint_r_k_num_alloc,
      PinholePrincipalPoint_w,
      PinholePrincipalPoint_w_num_alloc,
      negalpha,
      out_PinholePrincipalPoint_r_kp1,
      out_PinholePrincipalPoint_r_kp1_num_alloc,
      out_PinholePrincipalPoint_r_0_norm2_tot,
      out_PinholePrincipalPoint_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar