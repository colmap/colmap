#include "kernel_pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct_kernel(
        float* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        float* pose_jac,
        unsigned int pose_jac_num_alloc,
        float* principal_point_njtr,
        unsigned int principal_point_njtr_num_alloc,
        SharedIndex* principal_point_njtr_indices,
        float* principal_point_jac,
        unsigned int principal_point_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex principal_point_njtr_indices_loc[1024];
  principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1, r2, r3);
  };
  load_shared<2, float, float>(principal_point_njtr,
                               0 * principal_point_njtr_num_alloc,
                               principal_point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         principal_point_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r1, r5, r0 * r4);
    r7 = fmaf(r3, r5, r2 * r4);
    read_idx_4<1024, float, float, float4>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r8, r9, r10, r11);
    r12 = fmaf(r9, r5, r8 * r4);
    r13 = r10 * r4;
    write_sum_4<float, float>((float*)inout_shared, r6, r7, r12, r13);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r11 * r5;
    read_idx_2<1024, float, float, float2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r12, r7);
    r5 = fmaf(r7, r5, r12 * r4);
    write_sum_2<float, float>((float*)inout_shared, r13, r5);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  load_shared<2, float, float>(pose_njtr,
                               4 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_2<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r5,
                         r13);
  };
  __syncthreads();
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r4,
                         r6,
                         r14,
                         r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r14, r8, r13 * r12);
    r8 = fmaf(r15, r10, r8);
    r8 = fmaf(r4, r0, r8);
    r8 = fmaf(r6, r2, r8);
    r9 = fmaf(r14, r9, r13 * r7);
    r9 = fmaf(r4, r1, r9);
    r9 = fmaf(r6, r3, r9);
    r9 = fmaf(r5, r11, r9);
    write_sum_2<float, float>((float*)inout_shared, r8, r9);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_njtr_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    float* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct_kernel<<<n_blocks,
                                                                    1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      principal_point_njtr,
      principal_point_njtr_num_alloc,
      principal_point_njtr_indices,
      principal_point_jac,
      principal_point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar