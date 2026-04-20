#include "kernel_SimpleRadialExtraCalib_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialExtraCalib_pred_decrease_times_two_kernel(
        float* SimpleRadialExtraCalib_step,
        unsigned int SimpleRadialExtraCalib_step_num_alloc,
        float* SimpleRadialExtraCalib_precond_diag,
        unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
        const float* const diag,
        float* SimpleRadialExtraCalib_njtr,
        unsigned int SimpleRadialExtraCalib_njtr_num_alloc,
        float* const out_SimpleRadialExtraCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_SimpleRadialExtraCalib_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        SimpleRadialExtraCalib_step,
        0 * SimpleRadialExtraCalib_step_num_alloc,
        global_thread_idx,
        r0,
        r1,
        r2);
    read_idx_3<1024, float, float, float4>(
        SimpleRadialExtraCalib_njtr,
        0 * SimpleRadialExtraCalib_njtr_num_alloc,
        global_thread_idx,
        r3,
        r4,
        r5);
    read_idx_3<1024, float, float, float4>(
        SimpleRadialExtraCalib_precond_diag,
        0 * SimpleRadialExtraCalib_precond_diag_num_alloc,
        global_thread_idx,
        r6,
        r7,
        r8);
    r9 = r1 * r7;
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r10, r9, r4);
    r4 = r2 * r8;
    r4 = fmaf(r10, r4, r5);
    r4 = fmaf(r2, r4, r1 * r9);
    r9 = r0 * r6;
    r9 = fmaf(r10, r9, r3);
    r4 = fmaf(r0, r9, r4);
  };
  sum_store<float>(out_SimpleRadialExtraCalib_pred_dec_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r4);
  sum_flush_final<float>(out_SimpleRadialExtraCalib_pred_dec_local,
                         out_SimpleRadialExtraCalib_pred_dec,
                         1);
}

void SimpleRadialExtraCalib_pred_decrease_times_two(
    float* SimpleRadialExtraCalib_step,
    unsigned int SimpleRadialExtraCalib_step_num_alloc,
    float* SimpleRadialExtraCalib_precond_diag,
    unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialExtraCalib_njtr,
    unsigned int SimpleRadialExtraCalib_njtr_num_alloc,
    float* const out_SimpleRadialExtraCalib_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialExtraCalib_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      SimpleRadialExtraCalib_step,
      SimpleRadialExtraCalib_step_num_alloc,
      SimpleRadialExtraCalib_precond_diag,
      SimpleRadialExtraCalib_precond_diag_num_alloc,
      diag,
      SimpleRadialExtraCalib_njtr,
      SimpleRadialExtraCalib_njtr_num_alloc,
      out_SimpleRadialExtraCalib_pred_dec,
      problem_size);
}

}  // namespace caspar