#include "kernel_SimpleRadialCalib_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialCalib_update_Mp_kernel(
    float* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    float* SimpleRadialCalib_Mp,
    unsigned int SimpleRadialCalib_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialCalib_Mp_kp1,
    unsigned int out_SimpleRadialCalib_Mp_kp1_num_alloc,
    float* out_SimpleRadialCalib_w,
    unsigned int out_SimpleRadialCalib_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(SimpleRadialCalib_Mp,
                                           0 * SimpleRadialCalib_Mp_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2,
                                           r3);
    read_idx_4<1024, float, float, float4>(SimpleRadialCalib_r_k,
                                           0 * SimpleRadialCalib_r_k_num_alloc,
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
    r8 = fmaf(r3, r8, r7);
    write_idx_4<1024, float, float, float4>(
        out_SimpleRadialCalib_Mp_kp1,
        0 * out_SimpleRadialCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r8);
    write_idx_4<1024, float, float, float4>(
        out_SimpleRadialCalib_w,
        0 * out_SimpleRadialCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r8);
  };
}

void SimpleRadialCalib_update_Mp(
    float* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    float* SimpleRadialCalib_Mp,
    unsigned int SimpleRadialCalib_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialCalib_Mp_kp1,
    unsigned int out_SimpleRadialCalib_Mp_kp1_num_alloc,
    float* out_SimpleRadialCalib_w,
    unsigned int out_SimpleRadialCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialCalib_update_Mp_kernel<<<n_blocks, 1024>>>(
      SimpleRadialCalib_r_k,
      SimpleRadialCalib_r_k_num_alloc,
      SimpleRadialCalib_Mp,
      SimpleRadialCalib_Mp_num_alloc,
      beta,
      out_SimpleRadialCalib_Mp_kp1,
      out_SimpleRadialCalib_Mp_kp1_num_alloc,
      out_SimpleRadialCalib_w,
      out_SimpleRadialCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar