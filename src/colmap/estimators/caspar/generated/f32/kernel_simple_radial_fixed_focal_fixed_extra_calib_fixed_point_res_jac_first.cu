#include "kernel_simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel(
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
      r46, r47, r48, r49;

  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        extra_calib, 0 * extra_calib_num_alloc, global_thread_idx, r0, r1, r2);
    read_idx_2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r3, r4);
    r5 = -1.00000000000000000e+00;
    r3 = fmaf(r3, r5, r0);
    read_idx_1<1024, float, float, float>(
        focal, 0 * focal_num_alloc, global_thread_idx, r0);
  };
  load_shared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    read_idx_3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r9, r10, r11);
  };
  load_shared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_4<float>((float*)inout_shared,
                         pose_indices_loc[threadIdx.x].target,
                         r12,
                         r13,
                         r14,
                         r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r12 * r13;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = -2.00000000000000000e+00;
    r19 = r14 * r18;
    r20 = r15 * r19;
    r21 = r16 + r20;
    r6 = fmaf(r10, r21, r6);
    r22 = r12 * r14;
    r22 = r22 * r17;
    r23 = r13 * r15;
    r23 = r23 * r17;
    r24 = r22 + r23;
    r25 = r14 * r19;
    r26 = 1.00000000000000000e+00;
    r27 = r13 * r13;
    r28 = fmaf(r18, r27, r26);
    r29 = r25 + r28;
    r6 = fmaf(r11, r24, r6);
    r6 = fmaf(r9, r29, r6);
    r29 = r0 * r6;
    r30 = r14 * r15;
    r30 = r30 * r17;
    r16 = r16 + r30;
    r7 = fmaf(r9, r16, r7);
    r31 = r13 * r14;
    r31 = r31 * r17;
    r32 = r12 * r15;
    r32 = r32 * r18;
    r33 = r31 + r32;
    r25 = r26 + r25;
    r34 = r12 * r12;
    r35 = r18 * r34;
    r25 = r25 + r35;
    r7 = fmaf(r11, r33, r7);
    r7 = fmaf(r10, r25, r7);
    r25 = r7 * r7;
    r36 = 9.99999999999999955e-07;
    r37 = r12 * r15;
    r37 = r37 * r17;
    r31 = r31 + r37;
    r8 = fmaf(r10, r31, r8);
    r38 = r13 * r15;
    r38 = r38 * r18;
    r22 = r22 + r38;
    r28 = r35 + r28;
    r8 = fmaf(r9, r22, r8);
    r8 = fmaf(r11, r28, r8);
    r28 = copysign(1.0, r8);
    r28 = fmaf(r36, r28, r8);
    r36 = r28 * r28;
    r8 = 1.0 / r36;
    r35 = r6 * r6;
    r35 = fmaf(r8, r35, r8 * r25);
    r35 = fmaf(r2, r35, r26);
    r26 = 1.0 / r28;
    r25 = r35 * r26;
    r3 = fmaf(r29, r25, r3);
    r4 = fmaf(r4, r5, r1);
    r1 = r0 * r35;
    r1 = r1 * r26;
    r4 = fmaf(r7, r1, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r25 = fmaf(r3, r3, r4 * r4);
  };
  sum_store<float>(out_rTr_local,
                   (float*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r25);
  if (global_thread_idx < problem_size) {
    r25 = r5 * r4;
    r39 = r17 * r7;
    r40 = r14 * r14;
    r15 = r15 * r15;
    r41 = r5 * r15;
    r42 = r40 + r41;
    r43 = r5 * r27;
    r44 = r34 + r43;
    r45 = r42 + r44;
    r45 = fmaf(r11, r45, r10 * r33);
    r39 = r39 * r45;
    r33 = r18 * r7;
    r46 = r13 * r19;
    r32 = r32 + r46;
    r43 = r40 + r43;
    r40 = r5 * r34;
    r47 = r15 + r40;
    r43 = r43 + r47;
    r43 = fmaf(r10, r43, r11 * r32);
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r33 = r33 * r7;
    r33 = r33 * r43;
    r33 = fmaf(r28, r33, r8 * r39);
    r39 = r18 * r6;
    r39 = r39 * r6;
    r39 = r39 * r43;
    r33 = fmaf(r28, r39, r33);
    r32 = r17 * r6;
    r13 = r12 * r13;
    r13 = r13 * r18;
    r30 = r30 + r13;
    r24 = fmaf(r10, r24, r11 * r30);
    r32 = r32 * r24;
    r33 = fmaf(r8, r32, r33);
    r32 = r0 * r33;
    r39 = r2 * r7;
    r32 = r32 * r26;
    r45 = fmaf(r45, r1, r39 * r32);
    r32 = r0 * r7;
    r32 = r32 * r8;
    r32 = r32 * r35;
    r32 = r32 * r5;
    r45 = fmaf(r43, r32, r45);
    r30 = r5 * r3;
    r48 = r43 * r29;
    r35 = r35 * r5;
    r35 = r35 * r8;
    r49 = r2 * r33;
    r49 = r49 * r26;
    r49 = fmaf(r29, r49, r35 * r48);
    r49 = fmaf(r24, r1, r49);
    r30 = fmaf(r49, r30, r45 * r25);
    r25 = r5 * r3;
    r24 = r17 * r7;
    r46 = r37 + r46;
    r46 = fmaf(r9, r46, r11 * r16);
    r24 = r24 * r46;
    r16 = r17 * r6;
    r19 = r12 * r19;
    r38 = r38 + r19;
    r14 = r14 * r14;
    r14 = r14 * r5;
    r15 = r15 + r14;
    r15 = r15 + r44;
    r15 = fmaf(r11, r15, r9 * r38);
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r24);
    r24 = r18 * r6;
    r41 = r34 + r41;
    r14 = r27 + r14;
    r41 = r41 + r14;
    r41 = fmaf(r9, r41, r11 * r22);
    r24 = r24 * r6;
    r24 = r24 * r41;
    r16 = fmaf(r28, r24, r16);
    r22 = r18 * r7;
    r22 = r22 * r7;
    r22 = r22 * r41;
    r16 = fmaf(r28, r22, r16);
    r22 = r2 * r16;
    r22 = r22 * r26;
    r15 = fmaf(r15, r1, r29 * r22);
    r22 = r41 * r29;
    r15 = fmaf(r35, r22, r15);
    r22 = r5 * r4;
    r24 = r0 * r16;
    r24 = r24 * r26;
    r24 = fmaf(r41, r32, r39 * r24);
    r24 = fmaf(r46, r1, r24);
    r22 = fmaf(r24, r22, r15 * r25);
    r25 = r5 * r3;
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r31 = r19 * r29;
    r23 = r17 * r6;
    r40 = r27 + r40;
    r40 = r40 + r42;
    r40 = fmaf(r10, r40, r9 * r21);
    r23 = r23 * r40;
    r21 = r18 * r7;
    r21 = r21 * r7;
    r21 = r21 * r19;
    r21 = fmaf(r28, r21, r8 * r23);
    r23 = r18 * r6;
    r23 = r23 * r6;
    r23 = r23 * r19;
    r21 = fmaf(r28, r23, r21);
    r42 = r17 * r7;
    r20 = r13 + r20;
    r14 = r47 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r42 = r42 * r14;
    r21 = fmaf(r8, r42, r21);
    r42 = r2 * r21;
    r42 = r42 * r26;
    r42 = fmaf(r29, r42, r35 * r31);
    r42 = fmaf(r40, r1, r42);
    r40 = r5 * r4;
    r31 = r0 * r21;
    r31 = r31 * r26;
    r31 = fmaf(r39, r31, r19 * r32);
    r31 = fmaf(r14, r1, r31);
    r40 = fmaf(r31, r40, r42 * r25);
    r25 = r5 * r3;
    r14 = r28 * r29;
    r23 = r17 * r14;
    r8 = r2 * r6;
    r23 = fmaf(r8, r23, r1);
    r9 = r18 * r4;
    r20 = r39 * r14;
    r9 = fmaf(r20, r9, r23 * r25);
    write_sum_4<float, float>((float*)inout_shared, r30, r22, r40, r9);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r5 * r4;
    r40 = r18 * r7;
    r40 = r40 * r7;
    r22 = r18 * r6;
    r22 = r22 * r6;
    r22 = fmaf(r28, r22, r28 * r40);
    r40 = r0 * r22;
    r40 = r40 * r26;
    r40 = fmaf(r39, r40, r32);
    r32 = r5 * r3;
    r30 = r2 * r22;
    r30 = r30 * r26;
    r30 = fmaf(r29, r30, r29 * r35);
    r32 = fmaf(r30, r32, r40 * r9);
    r9 = r5 * r4;
    r35 = r0 * r17;
    r35 = r35 * r7;
    r35 = r35 * r28;
    r35 = fmaf(r39, r35, r1);
    r1 = r18 * r3;
    r1 = fmaf(r20, r1, r35 * r9);
    write_sum_2<float, float>((float*)inout_shared, r1, r32);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r45, r45, r49 * r49);
    r1 = fmaf(r15, r15, r24 * r24);
    r9 = fmaf(r42, r42, r31 * r31);
    r28 = r0 * r7;
    r26 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r28 = r28 * r26;
    r28 = r28 * r36;
    r28 = r28 * r29;
    r28 = r28 * r39;
    r28 = r28 * r8;
    r8 = fmaf(r23, r23, r28);
    write_sum_4<float, float>((float*)inout_shared, r32, r1, r9, r8);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r35, r35, r28);
    r8 = fmaf(r40, r40, r30 * r30);
    write_sum_2<float, float>((float*)inout_shared, r28, r8);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r49, r15, r45 * r24);
    r28 = fmaf(r45, r31, r49 * r42);
    r20 = r17 * r20;
    r9 = fmaf(r45, r20, r49 * r23);
    r1 = fmaf(r49, r20, r45 * r35);
    write_sum_4<float, float>((float*)inout_shared, r8, r28, r9, r1);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r49 = fmaf(r49, r30, r45 * r40);
    r45 = fmaf(r15, r42, r24 * r31);
    r1 = fmaf(r24, r20, r15 * r23);
    r9 = fmaf(r15, r20, r24 * r35);
    write_sum_4<float, float>((float*)inout_shared, r49, r45, r1, r9);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r24, r40, r15 * r30);
    r15 = fmaf(r31, r40, r42 * r30);
    r9 = fmaf(r31, r20, r42 * r23);
    r42 = fmaf(r42, r20, r31 * r35);
    write_sum_4<float, float>((float*)inout_shared, r24, r9, r42, r15);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r23, r20, r35 * r20);
    r23 = fmaf(r40, r20, r23 * r30);
    r20 = fmaf(r30, r20, r35 * r40);
    write_sum_3<float, float>((float*)inout_shared, r15, r23, r20);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  sum_flush_final<float>(out_rTr_local, out_rTr, 1);
}

void simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
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
  simple_radial_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first_kernel<<<
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