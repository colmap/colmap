#include "kernel_PinholeFocal_update_r.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFocal_update_r_kernel(float* PinholeFocal_r_k,
                                 unsigned int PinholeFocal_r_k_num_alloc,
                                 float* PinholeFocal_w,
                                 unsigned int PinholeFocal_w_num_alloc,
                                 const float* const negalpha,
                                 float* out_PinholeFocal_r_kp1,
                                 unsigned int out_PinholeFocal_r_kp1_num_alloc,
                                 float* const out_PinholeFocal_r_kp1_norm2_tot,
                                 size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_PinholeFocal_r_kp1_norm2_tot_local[1];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(PinholeFocal_r_k,
                                           0 * PinholeFocal_r_k_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1);
    read_idx_2<1024, float, float, float2>(PinholeFocal_w,
                                           0 * PinholeFocal_w_num_alloc,
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
        out_PinholeFocal_r_kp1,
        0 * out_PinholeFocal_r_kp1_num_alloc,
        global_thread_idx,
        r2,
        r4);
    r2 = fmaf(r2, r2, r4 * r4);
  };
  sum_store<float>(out_PinholeFocal_r_kp1_norm2_tot_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r2);
  sum_flush_final<float>(out_PinholeFocal_r_kp1_norm2_tot_local,
                         out_PinholeFocal_r_kp1_norm2_tot,
                         1);
}

void PinholeFocal_update_r(float* PinholeFocal_r_k,
                           unsigned int PinholeFocal_r_k_num_alloc,
                           float* PinholeFocal_w,
                           unsigned int PinholeFocal_w_num_alloc,
                           const float* const negalpha,
                           float* out_PinholeFocal_r_kp1,
                           unsigned int out_PinholeFocal_r_kp1_num_alloc,
                           float* const out_PinholeFocal_r_kp1_norm2_tot,
                           size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFocal_update_r_kernel<<<n_blocks, 1024>>>(
      PinholeFocal_r_k,
      PinholeFocal_r_k_num_alloc,
      PinholeFocal_w,
      PinholeFocal_w_num_alloc,
      negalpha,
      out_PinholeFocal_r_kp1,
      out_PinholeFocal_r_kp1_num_alloc,
      out_PinholeFocal_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar