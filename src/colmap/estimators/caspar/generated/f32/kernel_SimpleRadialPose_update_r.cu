#include "kernel_SimpleRadialPose_update_r.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialPose_update_r_kernel(
    float* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    float* SimpleRadialPose_w,
    unsigned int SimpleRadialPose_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialPose_r_kp1,
    unsigned int out_SimpleRadialPose_r_kp1_num_alloc,
    float* const out_SimpleRadialPose_r_kp1_norm2_tot,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_SimpleRadialPose_r_kp1_norm2_tot_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(SimpleRadialPose_r_k,
                                           0 * SimpleRadialPose_r_k_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2,
                                           r3);
    read_idx_4<1024, float, float, float4>(SimpleRadialPose_w,
                                           0 * SimpleRadialPose_w_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
  };
  load_unique<1, float, float>(negalpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r4, r8, r0);
    r5 = fmaf(r5, r8, r1);
    r6 = fmaf(r6, r8, r2);
    r7 = fmaf(r7, r8, r3);
    write_idx_4<1024, float, float, float4>(
        out_SimpleRadialPose_r_kp1,
        0 * out_SimpleRadialPose_r_kp1_num_alloc,
        global_thread_idx,
        r4,
        r5,
        r6,
        r7);
    read_idx_2<1024, float, float, float2>(SimpleRadialPose_r_k,
                                           4 * SimpleRadialPose_r_k_num_alloc,
                                           global_thread_idx,
                                           r3,
                                           r2);
    read_idx_2<1024, float, float, float2>(SimpleRadialPose_w,
                                           4 * SimpleRadialPose_w_num_alloc,
                                           global_thread_idx,
                                           r1,
                                           r0);
    r1 = fmaf(r1, r8, r3);
    r8 = fmaf(r0, r8, r2);
    write_idx_2<1024, float, float, float2>(
        out_SimpleRadialPose_r_kp1,
        4 * out_SimpleRadialPose_r_kp1_num_alloc,
        global_thread_idx,
        r1,
        r8);
    r1 = fmaf(r1, r1, r5 * r5);
    r1 = fmaf(r8, r8, r1);
    r1 = fmaf(r4, r4, r1);
    r1 = fmaf(r7, r7, r1);
    r1 = fmaf(r6, r6, r1);
  };
  sum_store<float>(out_SimpleRadialPose_r_kp1_norm2_tot_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r1);
  sum_flush_final<float>(out_SimpleRadialPose_r_kp1_norm2_tot_local,
                         out_SimpleRadialPose_r_kp1_norm2_tot,
                         1);
}

void SimpleRadialPose_update_r(
    float* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    float* SimpleRadialPose_w,
    unsigned int SimpleRadialPose_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialPose_r_kp1,
    unsigned int out_SimpleRadialPose_r_kp1_num_alloc,
    float* const out_SimpleRadialPose_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPose_update_r_kernel<<<n_blocks, 1024>>>(
      SimpleRadialPose_r_k,
      SimpleRadialPose_r_k_num_alloc,
      SimpleRadialPose_w,
      SimpleRadialPose_w_num_alloc,
      negalpha,
      out_SimpleRadialPose_r_kp1,
      out_SimpleRadialPose_r_kp1_num_alloc,
      out_SimpleRadialPose_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar