#include "kernel_pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;

  if (global_thread_idx < problem_size) {
    read_idx_2<1024, float, float, float2>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1);
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
    read_idx_2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r7, r29);
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
    r24 = fmaf(r2, r2, r3 * r3);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r24);
  if (global_thread_idx < problem_size) {
    r24 = r4 * r2;
    r25 = r11 * r12;
    r25 = r25 * r21;
    r1 = r1 + r25;
    r22 = fmaf(r9, r22, r10 * r1);
    r1 = r7 * r22;
    r12 = r12 * r30;
    r16 = r16 + r12;
    r21 = r13 * r13;
    r35 = r4 * r26;
    r36 = r21 + r35;
    r14 = r14 * r14;
    r37 = r11 * r11;
    r37 = r37 * r4;
    r38 = r14 + r37;
    r39 = r36 + r38;
    r39 = fmaf(r9, r39, r10 * r16);
    r16 = r28 * r28;
    r40 = 1.0 / r16;
    r41 = r4 * r40;
    r42 = r41 * r27;
    r1 = fmaf(r39, r42, r0 * r1);
    r43 = r4 * r3;
    r44 = r11 * r11;
    r45 = r4 * r14;
    r46 = r44 + r45;
    r36 = r36 + r46;
    r36 = fmaf(r10, r36, r9 * r15);
    r15 = r29 * r36;
    r47 = r39 * r41;
    r47 = fmaf(r34, r47, r0 * r15);
    r43 = fmaf(r47, r43, r1 * r24);
    r24 = r4 * r2;
    r30 = r11 * r30;
    r20 = r20 + r30;
    r14 = r44 + r14;
    r13 = r13 * r13;
    r13 = r13 * r4;
    r14 = r14 + r35;
    r14 = r14 + r13;
    r14 = fmaf(r10, r14, r8 * r20);
    r20 = r7 * r14;
    r13 = r26 + r13;
    r46 = r46 + r13;
    r46 = fmaf(r8, r46, r10 * r23);
    r20 = fmaf(r46, r42, r0 * r20);
    r23 = r4 * r3;
    r12 = r18 + r12;
    r12 = fmaf(r8, r12, r10 * r17);
    r17 = r29 * r12;
    r10 = r46 * r41;
    r10 = fmaf(r34, r10, r0 * r17);
    r23 = fmaf(r10, r23, r20 * r24);
    r24 = r4 * r3;
    r30 = r33 + r30;
    r30 = fmaf(r9, r30, r8 * r19);
    r19 = r30 * r41;
    r31 = r25 + r31;
    r13 = r38 + r13;
    r13 = fmaf(r8, r13, r9 * r31);
    r31 = r29 * r13;
    r31 = fmaf(r0, r31, r34 * r19);
    r19 = r4 * r2;
    r21 = r26 + r21;
    r21 = r21 + r37;
    r21 = r21 + r45;
    r21 = fmaf(r9, r21, r8 * r32);
    r9 = r7 * r21;
    r9 = fmaf(r30, r42, r0 * r9);
    r19 = fmaf(r9, r19, r31 * r24);
    r24 = r7 * r4;
    r24 = r24 * r2;
    r24 = r24 * r0;
    write_sum_4<float, float>((float*)inout_shared, r43, r23, r19, r24);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r29 * r4;
    r24 = r24 * r3;
    r24 = r24 * r0;
    r19 = r2 * r40;
    r23 = r3 * r40;
    r23 = fmaf(r34, r23, r27 * r19);
    write_sum_2<float, float>((float*)inout_shared, r24, r23);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r7 * r7;
    r23 = r23 * r40;
    r24 = fmaf(r1, r1, r47 * r47);
    r19 = fmaf(r20, r20, r10 * r10);
    r43 = fmaf(r9, r9, r31 * r31);
    write_sum_4<float, float>((float*)inout_shared, r24, r19, r43, r23);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r29 * r29;
    r23 = r23 * r40;
    r43 = r7 * r5;
    r16 = r28 * r16;
    r28 = r28 * r16;
    r28 = 1.0 / r28;
    r43 = r43 * r28;
    r19 = r6 * r28;
    r24 = r29 * r34;
    r19 = fmaf(r24, r19, r27 * r43);
    write_sum_2<float, float>((float*)inout_shared, r23, r19);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r19 = fmaf(r47, r10, r1 * r20);
    r23 = fmaf(r47, r31, r1 * r9);
    r43 = r7 * r1;
    r43 = r43 * r0;
    r32 = r29 * r47;
    r32 = r32 * r0;
    write_sum_4<float, float>((float*)inout_shared, r19, r23, r43, r32);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r20, r9, r10 * r31);
    r43 = r7 * r20;
    r43 = r43 * r0;
    r23 = r29 * r10;
    r23 = r23 * r0;
    r19 = r47 * r41;
    r19 = fmaf(r34, r19, r1 * r42);
    write_sum_4<float, float>((float*)inout_shared, r19, r32, r43, r23);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r7 * r9;
    r23 = r23 * r0;
    r43 = r29 * r31;
    r43 = r43 * r0;
    r0 = r10 * r41;
    r0 = fmaf(r34, r0, r20 * r42);
    r20 = r31 * r41;
    r20 = fmaf(r34, r20, r9 * r42);
    write_sum_4<float, float>((float*)inout_shared, r0, r23, r43, r20);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = 0.00000000000000000e+00;
    r27 = r7 * r27;
    r16 = 1.0 / r16;
    r16 = r4 * r16;
    r27 = r27 * r16;
    r24 = r16 * r24;
    write_sum_3<float, float>((float*)inout_shared, r20, r27, r24);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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
  pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              pixel,
              pixel_num_alloc,
              focal,
              focal_num_alloc,
              extra_calib,
              extra_calib_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar