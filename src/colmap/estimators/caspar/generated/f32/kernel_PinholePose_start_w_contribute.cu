#include "kernel_PinholePose_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_start_w_contribute_kernel(
        float* PinholePose_precond_diag,
        unsigned int PinholePose_precond_diag_num_alloc,
        const float* const diag,
        float* PinholePose_p,
        unsigned int PinholePose_p_num_alloc,
        float* out_PinholePose_w,
        unsigned int out_PinholePose_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        PinholePose_precond_diag,
        0 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2,
        r3);
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r4;
    read_idx_4<1024, float, float, float4>(PinholePose_p,
                                           0 * PinholePose_p_num_alloc,
                                           global_thread_idx,
                                           r5,
                                           r6,
                                           r7,
                                           r8);
    r0 = r0 * r5;
    r1 = r1 * r4;
    r1 = r1 * r6;
    r2 = r2 * r4;
    r2 = r2 * r7;
    r3 = r3 * r4;
    r3 = r3 * r8;
    add_idx_4<1024, float, float, float4>(out_PinholePose_w,
                                          0 * out_PinholePose_w_num_alloc,
                                          global_thread_idx,
                                          r0,
                                          r1,
                                          r2,
                                          r3);
    read_idx_2<1024, float, float, float2>(
        PinholePose_precond_diag,
        4 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r3,
        r2);
    r3 = r3 * r4;
    read_idx_2<1024, float, float, float2>(
        PinholePose_p, 4 * PinholePose_p_num_alloc, global_thread_idx, r1, r0);
    r3 = r3 * r1;
    r4 = r2 * r4;
    r4 = r4 * r0;
    add_idx_2<1024, float, float, float2>(out_PinholePose_w,
                                          4 * out_PinholePose_w_num_alloc,
                                          global_thread_idx,
                                          r3,
                                          r4);
  };
}

void PinholePose_start_w_contribute(
    float* PinholePose_precond_diag,
    unsigned int PinholePose_precond_diag_num_alloc,
    const float* const diag,
    float* PinholePose_p,
    unsigned int PinholePose_p_num_alloc,
    float* out_PinholePose_w,
    unsigned int out_PinholePose_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_start_w_contribute_kernel<<<n_blocks, 1024>>>(
      PinholePose_precond_diag,
      PinholePose_precond_diag_num_alloc,
      diag,
      PinholePose_p,
      PinholePose_p_num_alloc,
      out_PinholePose_w,
      out_PinholePose_w_num_alloc,
      problem_size);
}

}  // namespace caspar