#include "kernel_SimpleRadialExtraCalib_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_update_Mp_kernel(
        float* SimpleRadialExtraCalib_r_k,
        unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
        float* SimpleRadialExtraCalib_Mp,
        unsigned int SimpleRadialExtraCalib_Mp_num_alloc,
        const float* const beta,
        float* out_SimpleRadialExtraCalib_Mp_kp1,
        unsigned int out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
        float* out_SimpleRadialExtraCalib_w,
        unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        SimpleRadialExtraCalib_Mp,
        0 * SimpleRadialExtraCalib_Mp_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2);
    read_idx_3<1024, float, float, float4>(
        SimpleRadialExtraCalib_r_k,
        0 * SimpleRadialExtraCalib_r_k_num_alloc,
        global_thread_idx,
        r3,
        r4,
        r5);
  };
  load_unique<1, float, float>(beta, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r0, r6, r3);
    r1 = fmaf(r1, r6, r4);
    r6 = fmaf(r2, r6, r5);
    write_idx_3<1024, float, float, float4>(
        out_SimpleRadialExtraCalib_Mp_kp1,
        0 * out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r6);
    write_idx_3<1024, float, float, float4>(
        out_SimpleRadialExtraCalib_w,
        0 * out_SimpleRadialExtraCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r6);
  };
}

void SimpleRadialExtraCalib_update_Mp(
    float* SimpleRadialExtraCalib_r_k,
    unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
    float* SimpleRadialExtraCalib_Mp,
    unsigned int SimpleRadialExtraCalib_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialExtraCalib_Mp_kp1,
    unsigned int out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
    float* out_SimpleRadialExtraCalib_w,
    unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialExtraCalib_update_Mp_kernel<<<n_blocks, 1024>>>(
      SimpleRadialExtraCalib_r_k,
      SimpleRadialExtraCalib_r_k_num_alloc,
      SimpleRadialExtraCalib_Mp,
      SimpleRadialExtraCalib_Mp_num_alloc,
      beta,
      out_SimpleRadialExtraCalib_Mp_kp1,
      out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
      out_SimpleRadialExtraCalib_w,
      out_SimpleRadialExtraCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar