#include "kernel_SimpleRadialPose_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialPose_update_Mp_kernel(
    float* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    float* SimpleRadialPose_Mp,
    unsigned int SimpleRadialPose_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialPose_Mp_kp1,
    unsigned int out_SimpleRadialPose_Mp_kp1_num_alloc,
    float* out_SimpleRadialPose_w,
    unsigned int out_SimpleRadialPose_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(SimpleRadialPose_Mp,
                                           0 * SimpleRadialPose_Mp_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2,
                                           r3);
    read_idx_4<1024, float, float, float4>(SimpleRadialPose_r_k,
                                           0 * SimpleRadialPose_r_k_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
  };
  load_unique<1, float, float>(beta, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r0, r8, r4);
    r1 = fmaf(r1, r8, r5);
    r2 = fmaf(r2, r8, r6);
    r3 = fmaf(r3, r8, r7);
    write_idx_4<1024, float, float, float4>(
        out_SimpleRadialPose_Mp_kp1,
        0 * out_SimpleRadialPose_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    read_idx_2<1024, float, float, float2>(SimpleRadialPose_Mp,
                                           4 * SimpleRadialPose_Mp_num_alloc,
                                           global_thread_idx,
                                           r7,
                                           r6);
    read_idx_2<1024, float, float, float2>(SimpleRadialPose_r_k,
                                           4 * SimpleRadialPose_r_k_num_alloc,
                                           global_thread_idx,
                                           r5,
                                           r4);
    r7 = fmaf(r7, r8, r5);
    r8 = fmaf(r6, r8, r4);
    write_idx_2<1024, float, float, float2>(
        out_SimpleRadialPose_Mp_kp1,
        4 * out_SimpleRadialPose_Mp_kp1_num_alloc,
        global_thread_idx,
        r7,
        r8);
    write_idx_4<1024, float, float, float4>(
        out_SimpleRadialPose_w,
        0 * out_SimpleRadialPose_w_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
    write_idx_2<1024, float, float, float2>(
        out_SimpleRadialPose_w,
        4 * out_SimpleRadialPose_w_num_alloc,
        global_thread_idx,
        r7,
        r8);
  };
}

void SimpleRadialPose_update_Mp(
    float* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    float* SimpleRadialPose_Mp,
    unsigned int SimpleRadialPose_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialPose_Mp_kp1,
    unsigned int out_SimpleRadialPose_Mp_kp1_num_alloc,
    float* out_SimpleRadialPose_w,
    unsigned int out_SimpleRadialPose_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialPose_update_Mp_kernel<<<n_blocks, 1024>>>(
      SimpleRadialPose_r_k,
      SimpleRadialPose_r_k_num_alloc,
      SimpleRadialPose_Mp,
      SimpleRadialPose_Mp_num_alloc,
      beta,
      out_SimpleRadialPose_Mp_kp1,
      out_SimpleRadialPose_Mp_kp1_num_alloc,
      out_SimpleRadialPose_w,
      out_SimpleRadialPose_w_num_alloc,
      problem_size);
}

}  // namespace caspar