#include "kernel_Pose_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) Pose_pred_decrease_times_two_kernel(
    float* Pose_step,
    unsigned int Pose_step_num_alloc,
    float* Pose_precond_diag,
    unsigned int Pose_precond_diag_num_alloc,
    const float* const diag,
    float* Pose_njtr,
    unsigned int Pose_njtr_num_alloc,
    float* const out_Pose_pred_dec,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_Pose_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        Pose_step, 0 * Pose_step_num_alloc, global_thread_idx, r0, r1, r2, r3);
    read_idx_4<1024, float, float, float4>(
        Pose_njtr, 0 * Pose_njtr_num_alloc, global_thread_idx, r4, r5, r6, r7);
    read_idx_4<1024, float, float, float4>(Pose_precond_diag,
                                           0 * Pose_precond_diag_num_alloc,
                                           global_thread_idx,
                                           r8,
                                           r9,
                                           r10,
                                           r11);
    r12 = r1 * r9;
  };
  load_unique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_1<float>((float*)inout_shared, 0, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r13, r12, r5);
    read_idx_2<1024, float, float, float2>(
        Pose_step, 4 * Pose_step_num_alloc, global_thread_idx, r5, r14);
    read_idx_2<1024, float, float, float2>(
        Pose_njtr, 4 * Pose_njtr_num_alloc, global_thread_idx, r15, r16);
    read_idx_2<1024, float, float, float2>(Pose_precond_diag,
                                           4 * Pose_precond_diag_num_alloc,
                                           global_thread_idx,
                                           r17,
                                           r18);
    r19 = r5 * r17;
    r19 = fmaf(r13, r19, r15);
    r19 = fmaf(r5, r19, r1 * r12);
    r12 = r3 * r11;
    r12 = fmaf(r13, r12, r7);
    r7 = r0 * r8;
    r7 = fmaf(r13, r7, r4);
    r4 = r14 * r18;
    r4 = fmaf(r13, r4, r16);
    r16 = r2 * r10;
    r16 = fmaf(r13, r16, r6);
    r19 = fmaf(r3, r12, r19);
    r19 = fmaf(r0, r7, r19);
    r19 = fmaf(r14, r4, r19);
    r19 = fmaf(r2, r16, r19);
  };
  sum_store<float>(out_Pose_pred_dec_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r19);
  sum_flush_final<float>(out_Pose_pred_dec_local, out_Pose_pred_dec, 1);
}

void Pose_pred_decrease_times_two(float* Pose_step,
                                  unsigned int Pose_step_num_alloc,
                                  float* Pose_precond_diag,
                                  unsigned int Pose_precond_diag_num_alloc,
                                  const float* const diag,
                                  float* Pose_njtr,
                                  unsigned int Pose_njtr_num_alloc,
                                  float* const out_Pose_pred_dec,
                                  size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Pose_pred_decrease_times_two_kernel<<<n_blocks, 1024>>>(
      Pose_step,
      Pose_step_num_alloc,
      Pose_precond_diag,
      Pose_precond_diag_num_alloc,
      diag,
      Pose_njtr,
      Pose_njtr_num_alloc,
      out_Pose_pred_dec,
      problem_size);
}

}  // namespace caspar