#include "kernel_simple_radial_fixed_focal_and_extra_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_and_extra_jtjnjtr_direct_kernel(
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
        float* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        float* point_jac,
        unsigned int point_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26;

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
  load_shared<3, float, float>(point_njtr,
                               0 * point_njtr_num_alloc,
                               point_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         point_njtr_indices_loc[threadIdx.x].target,
                         r6,
                         r7,
                         r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r9, r10);
    read_idx_4<1024, float, float, float4>(point_jac,
                                           0 * point_jac_num_alloc,
                                           global_thread_idx,
                                           r11,
                                           r12,
                                           r13,
                                           r14);
    r15 = fmaf(r7, r13, r8 * r9);
    r15 = fmaf(r6, r11, r15);
    r16 = r4 + r15;
    r7 = fmaf(r7, r14, r8 * r10);
    r7 = fmaf(r6, r12, r7);
    r6 = r5 + r7;
    r8 = fmaf(r1, r6, r0 * r16);
    r17 = fmaf(r3, r6, r2 * r16);
    read_idx_4<1024, float, float, float4>(pose_jac,
                                           4 * pose_jac_num_alloc,
                                           global_thread_idx,
                                           r18,
                                           r19,
                                           r20,
                                           r21);
    r22 = fmaf(r19, r6, r18 * r16);
    r23 = fmaf(r21, r6, r20 * r16);
    write_sum_4<float, float>((float*)inout_shared, r8, r17, r22, r23);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_idx_4<1024, float, float, float4>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r23, r22, r17, r8);
    r24 = fmaf(r22, r6, r23 * r16);
    r6 = fmaf(r8, r6, r17 * r16);
    write_sum_2<float, float>((float*)inout_shared, r24, r6);
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
                         r6,
                         r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r6, r23, r24 * r17);
  };
  load_shared<4, float, float>(pose_njtr,
                               0 * pose_njtr_num_alloc,
                               pose_njtr_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_njtr_indices_loc[threadIdx.x].target,
                         r17,
                         r16,
                         r25,
                         r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r25, r18, r23);
    r23 = fmaf(r26, r20, r23);
    r23 = fmaf(r17, r0, r23);
    r23 = fmaf(r16, r2, r23);
    r15 = r23 + r15;
    r21 = fmaf(r26, r21, r24 * r8);
    r21 = fmaf(r25, r19, r21);
    r21 = fmaf(r17, r1, r21);
    r21 = fmaf(r16, r3, r21);
    r21 = fmaf(r6, r22, r21);
    r7 = r21 + r7;
    write_sum_2<float, float>((float*)inout_shared, r15, r7);
  };
  flush_sum_shared<2, float>(out_principal_point_njtr,
                             0 * out_principal_point_njtr_num_alloc,
                             principal_point_njtr_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r4 + r23;
    r21 = r5 + r21;
    r12 = fmaf(r12, r21, r11 * r23);
    r14 = fmaf(r14, r21, r13 * r23);
    r21 = fmaf(r10, r21, r9 * r23);
    write_sum_3<float, float>((float*)inout_shared, r12, r14, r21);
  };
  flush_sum_shared<3, float>(out_point_njtr,
                             0 * out_point_njtr_num_alloc,
                             point_njtr_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_focal_and_extra_jtjnjtr_direct(
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
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_and_extra_jtjnjtr_direct_kernel<<<n_blocks, 1024>>>(
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
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar