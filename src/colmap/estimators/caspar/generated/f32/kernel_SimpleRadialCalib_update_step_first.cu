#include "kernel_SimpleRadialCalib_update_step_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialCalib_update_step_first_kernel(
        float* SimpleRadialCalib_p_kp1,
        unsigned int SimpleRadialCalib_p_kp1_num_alloc,
        const float* const alpha,
        float* out_SimpleRadialCalib_step_kp1,
        unsigned int out_SimpleRadialCalib_step_kp1_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        SimpleRadialCalib_p_kp1,
        0 * SimpleRadialCalib_p_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
  };
  load_unique<1, float, float>(alpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r4;
    r1 = r1 * r4;
    r2 = r2 * r4;
    r4 = r3 * r4;
    write_idx_4<1024, float, float, float4>(
        out_SimpleRadialCalib_step_kp1,
        0 * out_SimpleRadialCalib_step_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r4);
  };
}

void SimpleRadialCalib_update_step_first(
    float* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    const float* const alpha,
    float* out_SimpleRadialCalib_step_kp1,
    unsigned int out_SimpleRadialCalib_step_kp1_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialCalib_update_step_first_kernel<<<n_blocks, 1024>>>(
      SimpleRadialCalib_p_kp1,
      SimpleRadialCalib_p_kp1_num_alloc,
      alpha,
      out_SimpleRadialCalib_step_kp1,
      out_SimpleRadialCalib_step_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar