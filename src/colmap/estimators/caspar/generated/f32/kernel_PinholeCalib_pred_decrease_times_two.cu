#include "kernel_PinholeCalib_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeCalib_pred_decrease_times_two_kernel(
        float* PinholeCalib_step,
        unsigned int PinholeCalib_step_num_alloc,
        float* PinholeCalib_precond_diag,
        unsigned int PinholeCalib_precond_diag_num_alloc,
        const float* const diag,
        float* PinholeCalib_njtr,
        unsigned int PinholeCalib_njtr_num_alloc,
        float* const out_PinholeCalib_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_PinholeCalib_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(PinholeCalib_step,
                                           0 * PinholeCalib_step_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1,
                                           r2,
                                           r3);
    read_idx_4<1024, float, float, float4>(PinholeCalib_njtr,
                                           0 * PinholeCalib_njtr_num_alloc,
                                           global_thread_idx,
                                           r4,
                                           r5,
                                           r6,
                                           r7);
    read_idx_4<1024, float, float, float4>(
        PinholeCalib_precond_diag,
        0 * PinholeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r8,
        r9,
        r10,
        r11);
    r12 = r3 * r11;
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r13, r12, r7);
    r7 = r2 * r10;
    r7 = fmaf(r13, r7, r6);
    r7 = fmaf(r2, r7, r3 * r12);
    r12 = r0 * r8;
    r12 = fmaf(r13, r12, r4);
    r4 = r1 * r9;
    r4 = fmaf(r13, r4, r5);
    r7 = fmaf(r0, r12, r7);
    r7 = fmaf(r1, r4, r7);
  };
  sum_store<float>(out_PinholeCalib_pred_dec_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r7);
  sum_flush_final<float>(
      out_PinholeCalib_pred_dec_local, out_PinholeCalib_pred_dec, 1);
}

void PinholeCalib_pred_decrease_times_two(
    float* PinholeCalib_step,
    unsigned int PinholeCalib_step_num_alloc,
    float* PinholeCalib_precond_diag,
    unsigned int PinholeCalib_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeCalib_njtr,
    unsigned int PinholeCalib_njtr_num_alloc,
    float* const out_PinholeCalib_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeCalib_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      PinholeCalib_step,
      PinholeCalib_step_num_alloc,
      PinholeCalib_precond_diag,
      PinholeCalib_precond_diag_num_alloc,
      diag,
      PinholeCalib_njtr,
      PinholeCalib_njtr_num_alloc,
      out_PinholeCalib_pred_dec,
      problem_size);
}

}  // namespace caspar