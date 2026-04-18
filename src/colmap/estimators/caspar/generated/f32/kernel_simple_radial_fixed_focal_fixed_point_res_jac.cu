#include "kernel_simple_radial_fixed_focal_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    simple_radial_fixed_focal_fixed_point_res_jac_kernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* extra_calib,
        unsigned int extra_calib_num_alloc,
        SharedIndex* extra_calib_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_extra_calib_jac,
        unsigned int out_extra_calib_jac_num_alloc,
        float* const out_extra_calib_njtr,
        unsigned int out_extra_calib_njtr_num_alloc,
        float* const out_extra_calib_precond_diag,
        unsigned int out_extra_calib_precond_diag_num_alloc,
        float* const out_extra_calib_precond_tril,
        unsigned int out_extra_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex extra_calib_indices_loc[1024];
  extra_calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? extra_calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51;
  load_shared<3, float, float>(extra_calib,
                               0 * extra_calib_num_alloc,
                               extra_calib_indices_loc,
                               (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    read_shared_3<float>((float*)inout_shared,
                         extra_calib_indices_loc[threadIdx.x].target,
                         r0,
                         r1,
                         r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r25 = fmaf(r2, r35, r26);
    r39 = 1.0 / r28;
    r40 = r25 * r39;
    r3 = fmaf(r29, r40, r3);
    r4 = fmaf(r4, r5, r1);
    r1 = r0 * r25;
    r1 = r1 * r39;
    r4 = fmaf(r7, r1, r4);
    write_idx_2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3, r4);
    r40 = r13 * r19;
    r32 = r32 + r40;
    r41 = r14 * r14;
    r42 = r5 * r27;
    r43 = r41 + r42;
    r15 = r15 * r15;
    r44 = r5 * r34;
    r45 = r15 + r44;
    r46 = r43 + r45;
    r46 = fmaf(r10, r46, r11 * r32);
    r32 = r46 * r29;
    r47 = r25 * r5;
    r47 = r47 * r8;
    r48 = r17 * r7;
    r49 = r5 * r15;
    r50 = r34 + r49;
    r43 = r43 + r50;
    r43 = fmaf(r11, r43, r10 * r33);
    r48 = r48 * r43;
    r33 = r18 * r7;
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r33 = r33 * r7;
    r33 = r33 * r46;
    r33 = fmaf(r28, r33, r8 * r48);
    r48 = r18 * r6;
    r48 = r48 * r6;
    r48 = r48 * r46;
    r33 = fmaf(r28, r48, r33);
    r51 = r17 * r6;
    r13 = r12 * r13;
    r13 = r13 * r18;
    r30 = r30 + r13;
    r24 = fmaf(r10, r24, r11 * r30);
    r51 = r51 * r24;
    r33 = fmaf(r8, r51, r33);
    r51 = r2 * r33;
    r51 = r51 * r39;
    r51 = fmaf(r29, r51, r47 * r32);
    r51 = fmaf(r24, r1, r51);
    r24 = r0 * r33;
    r32 = r2 * r7;
    r24 = r24 * r39;
    r43 = fmaf(r43, r1, r32 * r24);
    r24 = r0 * r7;
    r24 = r24 * r8;
    r24 = r24 * r25;
    r24 = r24 * r5;
    r43 = fmaf(r46, r24, r43);
    r25 = r17 * r7;
    r40 = r37 + r40;
    r40 = fmaf(r9, r40, r11 * r16);
    r25 = r25 * r40;
    r16 = r17 * r6;
    r19 = r12 * r19;
    r38 = r38 + r19;
    r15 = r34 + r15;
    r14 = r14 * r14;
    r14 = r14 * r5;
    r15 = r15 + r42;
    r15 = r15 + r14;
    r15 = fmaf(r11, r15, r9 * r38);
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r25);
    r25 = r18 * r6;
    r14 = r27 + r14;
    r50 = r50 + r14;
    r50 = fmaf(r9, r50, r11 * r22);
    r25 = r25 * r6;
    r25 = r25 * r50;
    r16 = fmaf(r28, r25, r16);
    r22 = r18 * r7;
    r22 = r22 * r7;
    r22 = r22 * r50;
    r16 = fmaf(r28, r22, r16);
    r22 = r2 * r16;
    r22 = r22 * r39;
    r15 = fmaf(r15, r1, r29 * r22);
    r22 = r50 * r29;
    r15 = fmaf(r47, r22, r15);
    r22 = r0 * r16;
    r22 = r22 * r39;
    r22 = fmaf(r50, r24, r32 * r22);
    r22 = fmaf(r40, r1, r22);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            0 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r51,
                                            r43,
                                            r15,
                                            r22);
    r40 = r2 * r0;
    r40 = r40 * r17;
    r40 = r40 * r6;
    r40 = r40 * r7;
    r40 = r40 * r28;
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r31 = r19 * r29;
    r23 = r17 * r6;
    r41 = r27 + r41;
    r41 = r41 + r44;
    r41 = r41 + r49;
    r41 = fmaf(r10, r41, r9 * r21);
    r23 = r23 * r41;
    r21 = r18 * r7;
    r21 = r21 * r7;
    r21 = r21 * r19;
    r21 = fmaf(r28, r21, r8 * r23);
    r23 = r18 * r6;
    r23 = r23 * r6;
    r23 = r23 * r19;
    r21 = fmaf(r28, r23, r21);
    r49 = r17 * r7;
    r20 = r13 + r20;
    r14 = r45 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r49 = r49 * r14;
    r21 = fmaf(r8, r49, r21);
    r49 = r2 * r21;
    r49 = r49 * r39;
    r49 = fmaf(r29, r49, r47 * r31);
    r49 = fmaf(r41, r1, r49);
    r41 = r0 * r21;
    r41 = r41 * r39;
    r41 = fmaf(r32, r41, r19 * r24);
    r41 = fmaf(r14, r1, r41);
    r14 = r28 * r29;
    r31 = r17 * r14;
    r23 = r2 * r6;
    r31 = fmaf(r23, r31, r1);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            4 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r49,
                                            r41,
                                            r31,
                                            r40);
    r9 = r0 * r17;
    r9 = r9 * r7;
    r9 = r9 * r28;
    r9 = fmaf(r32, r9, r1);
    r1 = r18 * r7;
    r1 = r1 * r7;
    r20 = r18 * r6;
    r20 = r20 * r6;
    r20 = fmaf(r28, r20, r28 * r1);
    r1 = r2 * r20;
    r1 = r1 * r39;
    r1 = fmaf(r29, r1, r29 * r47);
    r47 = r0 * r20;
    r47 = r47 * r39;
    r47 = fmaf(r32, r47, r24);
    write_idx_4<1024, float, float, float4>(out_pose_jac,
                                            8 * out_pose_jac_num_alloc,
                                            global_thread_idx,
                                            r40,
                                            r9,
                                            r1,
                                            r47);
    r24 = r5 * r4;
    r28 = r5 * r3;
    r28 = fmaf(r51, r28, r43 * r24);
    r24 = r5 * r3;
    r10 = r5 * r4;
    r10 = fmaf(r22, r10, r15 * r24);
    r24 = r5 * r3;
    r45 = r5 * r4;
    r45 = fmaf(r41, r45, r49 * r24);
    r24 = r5 * r3;
    r13 = r18 * r4;
    r44 = r32 * r14;
    r13 = fmaf(r44, r13, r31 * r24);
    write_sum_4<float, float>((float*)inout_shared, r28, r10, r45, r13);
  };
  flush_sum_shared<4, float>(out_pose_njtr,
                             0 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r5 * r4;
    r45 = r5 * r3;
    r45 = fmaf(r1, r45, r47 * r13);
    r13 = r5 * r4;
    r10 = r18 * r3;
    r10 = fmaf(r44, r10, r9 * r13);
    write_sum_2<float, float>((float*)inout_shared, r10, r45);
  };
  flush_sum_shared<2, float>(out_pose_njtr,
                             4 * out_pose_njtr_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fmaf(r43, r43, r51 * r51);
    r10 = fmaf(r15, r15, r22 * r22);
    r13 = fmaf(r49, r49, r41 * r41);
    r44 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r36 = r44 * r36;
    r44 = r0 * r7;
    r36 = r36 * r29;
    r36 = r36 * r32;
    r36 = r36 * r23;
    r36 = r36 * r44;
    r23 = fmaf(r31, r31, r36);
    write_sum_4<float, float>((float*)inout_shared, r45, r10, r13, r23);
  };
  flush_sum_shared<4, float>(out_pose_precond_diag,
                             0 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r9, r9, r36);
    r23 = fmaf(r47, r47, r1 * r1);
    write_sum_2<float, float>((float*)inout_shared, r36, r23);
  };
  flush_sum_shared<2, float>(out_pose_precond_diag,
                             4 * out_pose_precond_diag_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = fmaf(r51, r15, r43 * r22);
    r36 = fmaf(r43, r41, r51 * r49);
    r13 = fmaf(r43, r40, r51 * r31);
    r10 = fmaf(r51, r40, r43 * r9);
    write_sum_4<float, float>((float*)inout_shared, r23, r36, r13, r10);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             0 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fmaf(r51, r1, r43 * r47);
    r43 = fmaf(r15, r49, r22 * r41);
    r10 = fmaf(r22, r40, r15 * r31);
    r13 = fmaf(r15, r40, r22 * r9);
    write_sum_4<float, float>((float*)inout_shared, r51, r43, r10, r13);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             4 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fmaf(r22, r47, r15 * r1);
    r15 = fmaf(r41, r47, r49 * r1);
    r13 = fmaf(r41, r40, r49 * r31);
    r49 = fmaf(r49, r40, r41 * r9);
    write_sum_4<float, float>((float*)inout_shared, r22, r13, r49, r15);
  };
  flush_sum_shared<4, float>(out_pose_precond_tril,
                             8 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = fmaf(r31, r40, r9 * r40);
    r31 = fmaf(r47, r40, r31 * r1);
    r40 = fmaf(r1, r40, r9 * r47);
    write_sum_3<float, float>((float*)inout_shared, r15, r31, r40);
  };
  flush_sum_shared<3, float>(out_pose_precond_tril,
                             12 * out_pose_precond_tril_num_alloc,
                             pose_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r35 * r39;
    r40 = r40 * r29;
    r31 = r0 * r7;
    r31 = r31 * r35;
    r31 = r31 * r39;
    write_idx_2<1024, float, float, float2>(out_extra_calib_jac,
                                            0 * out_extra_calib_jac_num_alloc,
                                            global_thread_idx,
                                            r40,
                                            r31);
    r15 = r5 * r3;
    r1 = r5 * r4;
    r47 = r0 * r7;
    r47 = r47 * r35;
    r47 = r47 * r5;
    r47 = r47 * r4;
    r9 = r35 * r5;
    r9 = r9 * r3;
    r9 = r9 * r39;
    r9 = fmaf(r29, r9, r39 * r47);
    write_sum_3<float, float>((float*)inout_shared, r15, r1, r9);
  };
  flush_sum_shared<3, float>(out_extra_calib_njtr,
                             0 * out_extra_calib_njtr_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r0 * r8;
    r9 = r35 * r35;
    r8 = r8 * r9;
    r9 = r7 * r8;
    r1 = r6 * r29;
    r1 = fmaf(r8, r1, r44 * r9);
    write_sum_3<float, float>((float*)inout_shared, r26, r26, r1);
  };
  flush_sum_shared<3, float>(out_extra_calib_precond_diag,
                             0 * out_extra_calib_precond_diag_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = 0.00000000000000000e+00;
    write_sum_3<float, float>((float*)inout_shared, r1, r40, r31);
  };
  flush_sum_shared<3, float>(out_extra_calib_precond_tril,
                             0 * out_extra_calib_precond_tril_num_alloc,
                             extra_calib_indices_loc,
                             (float*)inout_shared);
}

void simple_radial_fixed_focal_fixed_point_res_jac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    float* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    float* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  simple_radial_fixed_focal_fixed_point_res_jac_kernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      extra_calib,
      extra_calib_num_alloc,
      extra_calib_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_extra_calib_jac,
      out_extra_calib_jac_num_alloc,
      out_extra_calib_njtr,
      out_extra_calib_njtr_num_alloc,
      out_extra_calib_precond_diag,
      out_extra_calib_precond_diag_num_alloc,
      out_extra_calib_precond_tril,
      out_extra_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar