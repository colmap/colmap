#include "kernel_SimpleRadialFocal_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialFocal_update_Mp_kernel(
    float* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    float* SimpleRadialFocal_Mp,
    unsigned int SimpleRadialFocal_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocal_Mp_kp1,
    unsigned int out_SimpleRadialFocal_Mp_kp1_num_alloc,
    float* out_SimpleRadialFocal_w,
    unsigned int out_SimpleRadialFocal_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2;

  if (global_thread_idx < problem_size) {
    read_idx_1<1024, float, float, float>(SimpleRadialFocal_Mp,
                                          0 * SimpleRadialFocal_Mp_num_alloc,
                                          global_thread_idx,
                                          r0);
    read_idx_1<1024, float, float, float>(SimpleRadialFocal_r_k,
                                          0 * SimpleRadialFocal_r_k_num_alloc,
                                          global_thread_idx,
                                          r1);
  };
  load_unique<1, float, float>(beta, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r0, r2, r1);
    write_idx_1<1024, float, float, float>(
        out_SimpleRadialFocal_Mp_kp1,
        0 * out_SimpleRadialFocal_Mp_kp1_num_alloc,
        global_thread_idx,
        r2);
    write_idx_1<1024, float, float, float>(
        out_SimpleRadialFocal_w,
        0 * out_SimpleRadialFocal_w_num_alloc,
        global_thread_idx,
        r2);
  };
}

void SimpleRadialFocal_update_Mp(
    float* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    float* SimpleRadialFocal_Mp,
    unsigned int SimpleRadialFocal_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocal_Mp_kp1,
    unsigned int out_SimpleRadialFocal_Mp_kp1_num_alloc,
    float* out_SimpleRadialFocal_w,
    unsigned int out_SimpleRadialFocal_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFocal_update_Mp_kernel<<<n_blocks, 1024>>>(
      SimpleRadialFocal_r_k,
      SimpleRadialFocal_r_k_num_alloc,
      SimpleRadialFocal_Mp,
      SimpleRadialFocal_Mp_num_alloc,
      beta,
      out_SimpleRadialFocal_Mp_kp1,
      out_SimpleRadialFocal_Mp_kp1_num_alloc,
      out_SimpleRadialFocal_w,
      out_SimpleRadialFocal_w_num_alloc,
      problem_size);
}

}  // namespace caspar