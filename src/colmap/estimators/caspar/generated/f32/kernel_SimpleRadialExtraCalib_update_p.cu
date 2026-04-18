#include "kernel_SimpleRadialExtraCalib_update_p.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_update_p_kernel(
        float* SimpleRadialExtraCalib_z,
        unsigned int SimpleRadialExtraCalib_z_num_alloc,
        float* SimpleRadialExtraCalib_p_k,
        unsigned int SimpleRadialExtraCalib_p_k_num_alloc,
        const float* const beta,
        float* out_SimpleRadialExtraCalib_p_kp1,
        unsigned int out_SimpleRadialExtraCalib_p_kp1_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        SimpleRadialExtraCalib_p_k,
        0 * SimpleRadialExtraCalib_p_k_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2);
    read_idx_3<1024, float, float, float4>(
        SimpleRadialExtraCalib_z,
        0 * SimpleRadialExtraCalib_z_num_alloc,
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
        out_SimpleRadialExtraCalib_p_kp1,
        0 * out_SimpleRadialExtraCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r6);
  };
}

void SimpleRadialExtraCalib_update_p(
    float* SimpleRadialExtraCalib_z,
    unsigned int SimpleRadialExtraCalib_z_num_alloc,
    float* SimpleRadialExtraCalib_p_k,
    unsigned int SimpleRadialExtraCalib_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialExtraCalib_p_kp1,
    unsigned int out_SimpleRadialExtraCalib_p_kp1_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialExtraCalib_update_p_kernel<<<n_blocks, 1024>>>(
      SimpleRadialExtraCalib_z,
      SimpleRadialExtraCalib_z_num_alloc,
      SimpleRadialExtraCalib_p_k,
      SimpleRadialExtraCalib_p_k_num_alloc,
      beta,
      out_SimpleRadialExtraCalib_p_kp1,
      out_SimpleRadialExtraCalib_p_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar