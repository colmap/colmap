#include "kernel_pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(principal_point,
                                           0 * principal_point_num_alloc,
                                           global_thread_idx,
                                           r0,
                                           r1);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = 9.99999999999999955e-07;
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r5, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
  };
  load_shared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_indices_loc[threadIdx.x].target,
                         r11,
                         r12,
                         r13,
                         r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r15 = r12 * r13;
    r16 = 2.00000000000000000e+00;
    r15 = r15 * r16;
    r17 = r11 * r16;
    r18 = r14 * r17;
    r19 = r15 + r18;
    r7 = fmaf(r9, r19, r7);
    r20 = r12 * r14;
    r21 = -2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = r13 * r17;
    r23 = r20 + r22;
    r24 = r11 * r11;
    r24 = r24 * r21;
    r25 = 1.00000000000000000e+00;
    r26 = r12 * r12;
    r27 = fmaf(r21, r26, r25);
    r28 = r24 + r27;
    r7 = fmaf(r8, r23, r7);
    r7 = fmaf(r10, r28, r7);
    r28 = copysign(1.0, r7);
    r28 = fmaf(r0, r28, r7);
    r0 = 1.0 / r28;
    read_idx_2<1024, float, float, float2>(focal_and_extra,
                                           0 * focal_and_extra_num_alloc,
                                           global_thread_idx,
                                           r7,
                                           r29);
    r30 = r13 * r21;
    r31 = r14 * r30;
    r17 = r12 * r17;
    r32 = r31 + r17;
    r5 = fmaf(r9, r32, r5);
    r33 = r12 * r14;
    r33 = r33 * r16;
    r22 = r33 + r22;
    r34 = r13 * r30;
    r27 = r34 + r27;
    r5 = fmaf(r10, r22, r5);
    r5 = fmaf(r8, r27, r5);
    r27 = r7 * r5;
    r2 = fmaf(r0, r27, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r13 * r14;
    r1 = r1 * r16;
    r17 = r1 + r17;
    r6 = fmaf(r8, r17, r6);
    r16 = r11 * r14;
    r16 = r16 * r21;
    r15 = r15 + r16;
    r34 = r25 + r34;
    r34 = r34 + r24;
    r6 = fmaf(r10, r15, r6);
    r6 = fmaf(r9, r34, r6);
    r34 = r29 * r6;
    r3 = fmaf(r0, r34, r3);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r24 = r4 * r3;
    r25 = r13 * r13;
    r14 = r14 * r14;
    r35 = r4 * r14;
    r36 = r25 + r35;
    r37 = r11 * r11;
    r38 = r4 * r26;
    r39 = r37 + r38;
    r40 = r36 + r39;
    r40 = fmaf(r10, r40, r9 * r15);
    r15 = r29 * r40;
    r41 = r12 * r30;
    r16 = r16 + r41;
    r38 = r25 + r38;
    r25 = r11 * r11;
    r25 = r25 * r4;
    r42 = r14 + r25;
    r38 = r38 + r42;
    r38 = fmaf(r9, r38, r10 * r16);
    r16 = r28 * r28;
    r43 = 1.0 / r16;
    r44 = r4 * r43;
    r45 = r38 * r44;
    r45 = fmaf(r34, r45, r0 * r15);
    r15 = r4 * r2;
    r46 = r44 * r27;
    r12 = r11 * r12;
    r12 = r12 * r21;
    r1 = r1 + r12;
    r22 = fmaf(r9, r22, r10 * r1);
    r1 = r7 * r22;
    r1 = fmaf(r0, r1, r38 * r46);
    r15 = fmaf(r1, r15, r45 * r24);
    r24 = r4 * r2;
    r35 = r37 + r35;
    r13 = r13 * r13;
    r13 = r13 * r4;
    r37 = r26 + r13;
    r35 = r35 + r37;
    r35 = fmaf(r8, r35, r10 * r23);
    r30 = r11 * r30;
    r20 = r20 + r30;
    r13 = r14 + r13;
    r13 = r13 + r39;
    r13 = fmaf(r10, r13, r8 * r20);
    r20 = r7 * r13;
    r20 = fmaf(r0, r20, r35 * r46);
    r39 = r4 * r3;
    r14 = r35 * r44;
    r41 = r18 + r41;
    r41 = fmaf(r8, r41, r10 * r17);
    r17 = r29 * r41;
    r17 = fmaf(r0, r17, r34 * r14);
    r39 = fmaf(r17, r39, r20 * r24);
    r24 = r4 * r2;
    r30 = r33 + r30;
    r30 = fmaf(r9, r30, r8 * r19);
    r25 = r26 + r25;
    r25 = r25 + r36;
    r25 = fmaf(r9, r25, r8 * r32);
    r32 = r7 * r25;
    r32 = fmaf(r0, r32, r30 * r46);
    r36 = r4 * r3;
    r26 = r30 * r44;
    r31 = r12 + r31;
    r37 = r42 + r37;
    r37 = fmaf(r8, r37, r9 * r31);
    r8 = r29 * r37;
    r8 = fmaf(r0, r8, r34 * r26);
    r36 = fmaf(r8, r36, r32 * r24);
    r24 = r7 * r4;
    r24 = r24 * r2;
    r24 = r24 * r0;
    write_sum_4<float, float>((float*)inout_shared, r15, r39, r36, r24);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r29 * r4;
    r24 = r24 * r3;
    r24 = r24 * r0;
    r36 = r3 * r43;
    r39 = r2 * r43;
    r39 = fmaf(r27, r39, r34 * r36);
    write_sum_2<float, float>((float*)inout_shared, r24, r39);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r7 * r7;
    r39 = r39 * r43;
    r24 = fmaf(r1, r1, r45 * r45);
    r36 = fmaf(r17, r17, r20 * r20);
    r15 = fmaf(r32, r32, r8 * r8);
    write_sum_4<float, float>((float*)inout_shared, r24, r36, r15, r39);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r29 * r29;
    r39 = r39 * r43;
    r16 = r28 * r16;
    r28 = r28 * r16;
    r28 = 1.0 / r28;
    r15 = r6 * r28;
    r36 = r29 * r34;
    r24 = r7 * r5;
    r24 = r24 * r28;
    r24 = fmaf(r27, r24, r36 * r15);
    write_sum_2<float, float>((float*)inout_shared, r39, r24);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r45, r17, r1 * r20);
    r39 = fmaf(r1, r32, r45 * r8);
    r15 = r7 * r1;
    r15 = r15 * r0;
    r26 = r29 * r45;
    r26 = r26 * r0;
    write_sum_4<float, float>((float*)inout_shared, r24, r39, r15, r26);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fmaf(r17, r8, r20 * r32);
    r15 = r7 * r20;
    r15 = r15 * r0;
    r39 = r29 * r17;
    r39 = r39 * r0;
    r24 = r45 * r44;
    r1 = fmaf(r1, r46, r34 * r24);
    write_sum_4<float, float>((float*)inout_shared, r1, r26, r15, r39);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = r7 * r32;
    r39 = r39 * r0;
    r15 = r29 * r8;
    r15 = r15 * r0;
    r0 = r17 * r44;
    r20 = fmaf(r20, r46, r34 * r0);
    r0 = r8 * r44;
    r0 = fmaf(r34, r0, r32 * r46);
    write_sum_4<float, float>((float*)inout_shared, r20, r39, r15, r0);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = 0.00000000000000000e+00;
    r27 = r7 * r27;
    r16 = 1.0 / r16;
    r16 = r4 * r16;
    r27 = r27 * r16;
    r36 = r16 * r36;
    write_sum_3<float, float>((float*)inout_shared, r0, r27, r36);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
}

void pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  pinhole_fixed_focal_and_extra_fixed_principal_point_fixed_point_res_jac_kernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              pixel,
              pixel_num_alloc,
              focal_and_extra,
              focal_and_extra_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar