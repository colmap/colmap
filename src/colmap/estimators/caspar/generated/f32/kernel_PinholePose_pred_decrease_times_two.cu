#include "kernel_PinholePose_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholePose_pred_decrease_times_two_kernel(
        float* PinholePose_step,
        unsigned int PinholePose_step_num_alloc,
        float* PinholePose_precond_diag,
        unsigned int PinholePose_precond_diag_num_alloc,
        const float* const diag,
        float* PinholePose_njtr,
        unsigned int PinholePose_njtr_num_alloc,
        float* const out_PinholePose_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_PinholePose_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(PinholePose_step,
                                           0 * PinholePose_step_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2,
                                           r3);
    read_idx_4<1024, float, float, float4>(PinholePose_njtr,
                                           0 * PinholePose_njtr_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
    read_idx_4<1024, float, float, float4>(
        PinholePose_precond_diag,
        0 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r8,
        r9,
        r10,
        r11);
    r12 = r2 * r10;
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r13, r12, r6);
    r6 = r0 * r8;
    r6 = fmaf(r13, r6, r4);
    r6 = fmaf(r0, r6, r2 * r12);
    r12 = r1 * r9;
    r12 = fmaf(r13, r12, r5);
    r5 = r3 * r11;
    r5 = fmaf(r13, r5, r7);
    read_idx_2<1024, float, float, float2>(PinholePose_step,
                                           4 * PinholePose_step_num_alloc,
                                           global_thread_idx,
                                           r7,
                                           r4);
    read_idx_2<1024, float, float, float2>(PinholePose_njtr,
                                           4 * PinholePose_njtr_num_alloc,
                                           global_thread_idx,
                                           r14,
                                           r15);
    read_idx_2<1024, float, float, float2>(
        PinholePose_precond_diag,
        4 * PinholePose_precond_diag_num_alloc,
        global_thread_idx,
        r16,
        r17);
    r18 = r7 * r16;
    r18 = fmaf(r13, r18, r14);
    r14 = r4 * r17;
    r14 = fmaf(r13, r14, r15);
    r6 = fmaf(r1, r12, r6);
    r6 = fmaf(r3, r5, r6);
    r6 = fmaf(r7, r18, r6);
    r6 = fmaf(r4, r14, r6);
  };
  sum_store<float>(out_PinholePose_pred_dec_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r6);
  sum_flush_final<float>(
      out_PinholePose_pred_dec_local, out_PinholePose_pred_dec, 1);
}

void PinholePose_pred_decrease_times_two(
    float* PinholePose_step,
    unsigned int PinholePose_step_num_alloc,
    float* PinholePose_precond_diag,
    unsigned int PinholePose_precond_diag_num_alloc,
    const float* const diag,
    float* PinholePose_njtr,
    unsigned int PinholePose_njtr_num_alloc,
    float* const out_PinholePose_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholePose_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      PinholePose_step,
      PinholePose_step_num_alloc,
      PinholePose_precond_diag,
      PinholePose_precond_diag_num_alloc,
      diag,
      PinholePose_njtr,
      PinholePose_njtr_num_alloc,
      out_PinholePose_pred_dec,
      problem_size);
}

}  // namespace caspar